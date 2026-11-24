import os
from dataclasses import dataclass

from . import features as F
from . import hud
from .classifier import PoseClassifier, VoteSmoother
from .fsm import AirPointFSM, DwellDetector
from .modes import CalibrateMode, LiveMode, RecordMode
from .pointer import FC_MIN, PointerPipeline
from .train import train_personal
from .tremor import TremorSim


SIM_NUDGES = {ord("["): (0.0, -0.05), ord("]"): (0.0, 0.05), ord("{"): (-0.5, 0.0), ord("}"): (0.5, 0.0)}


@dataclass
class Obs:
    """What one frame gave us."""
    t: float
    pts: object    # (21, 3) landmarks in normalized image coords, or None
    hand: object   # "Left" | "Right" | None
    raw: object    # pointer (x, y) in canonical view pixels (floats), or None
    w: int
    h: int
    fps: float


class Session:
    """Everything the frame loop shares, minus camera, tracker and window (those stay in app.py).

    tick() runs one frame through the current mode and returns the image to show.
    on_key() handles a key: the mode gets it first, then the global keys.
    """

    def __init__(self, runner, rules, mlp, *, root, user=None, fc_min=FC_MIN, dwell_on=False,
                 train_fn=train_personal):
        self.runner, self.rules, self.mlp = runner, rules, mlp
        self.clf = mlp or rules
        self.root, self.user, self.train_fn = root, user, train_fn
        self.data_dir, self.models_dir = os.path.join(root, "data"), os.path.join(root, "models")
        self.smoother, self.fsm, self.dwell = VoteSmoother(), AirPointFSM(), DwellDetector()
        self.pointer = PointerPipeline(fc_min)
        self.sim = TremorSim()
        self.dwell_on, self.mini, self.view_dirty = dwell_on, False, False
        self.zoom = 1.5
        self.recorder, self.rejected = None, 0
        self.mode = LiveMode()
        self.last_event, self.mini_label, self.prev_t = "none", "", None
        self.obs = None

    def tick(self, frame, pts, hand, t):
        fps = 0.0 if self.prev_t is None else 1.0 / max(t - self.prev_t, 1e-5)
        self.prev_t = t
        view = hud.canon_view(frame)
        h, w = view.shape[:2]
        raw = None
        if pts is not None:
            if self.sim.on:
                pts = self.sim.apply(pts, t, w, h)
            hud.draw_skeleton(view, pts)
            raw = (float(pts[F.MIDDLE_MCP, 0] * w), float(pts[F.MIDDLE_MCP, 1] * h))
        self.obs = Obs(t, pts, hand, raw, w, h, fps)
        self.mode.tick(self, self.obs, view)
        if self.mini and self.mode.name == "live":
            return hud.mini_frame(view, self.mini_label)
        return hud.zoomed(view, self.zoom)

    def set_mode(self, mode):
        self.mode.close(self)
        self.mode = mode
        if mode.name != "live":
            self.sim.on = False      # simulated shake never reaches recordings, calibration or a profile

    def use_model(self, model_path, labels_path):
        """Hot-swap to a personal model. Runs on the frame-loop thread."""
        self.mlp = self.clf = PoseClassifier.load(model_path, labels_path)
        self.mlp.name = f"mlp/{self.user}"
        self.smoother.reset()

    def on_key(self, key, t):
        if self.mode.on_key(self, key, t):
            return
        if key == ord("m"):
            self.runner.toggle()
        elif key == ord("d"):
            self.dwell_on = not self.dwell_on
            self.dwell.reset()
        elif key == ord("k") and self.mlp is not None:
            self.clf = self.rules if self.clf is self.mlp else self.mlp
            self.smoother.reset()
        elif key == ord("v"):
            self.mini, self.view_dirty = not self.mini, True
        elif key == ord("z"):
            self.zoom = hud.ZOOMS[(hud.ZOOMS.index(self.zoom) + 1) % len(hud.ZOOMS)]
            self.view_dirty = True
        elif key == ord("b"):
            self.pointer.cycle_mode()
        elif key == ord("t") and self.mode.name == "live":
            self.sim.on = not self.sim.on
        elif key in SIM_NUDGES:
            self.sim.nudge(*SIM_NUDGES[key])
        elif key == ord("r") and self.user and self.mode.name != "calibrate":
            self.runner.release()
            self.set_mode(RecordMode(self) if self.mode.name == "live" else LiveMode())
        elif key == ord("c") and self.mode.name != "record":
            if not self.user:
                print("[AirPoint] calibration needs --user <name>")
            elif self.mode.name == "live":
                self.runner.release()
                self.set_mode(CalibrateMode(self, t))
            elif not self.mode.busy:
                self.set_mode(LiveMode())

    def close(self):
        self.runner.release()
        self.mode.close(self)
        if self.recorder is not None:
            self.recorder.close()
