import json
import os
import time
from dataclasses import dataclass

import numpy as np

from . import features as F
from . import hud
from .actions import ACTIVE_REGION
from .classifier import PoseClassifier, VoteSmoother
from .effects import Effects
from .fsm import AirPointFSM, DwellDetector
from .hints import Hinter
from .modes import CalibrateMode, LiveMode, RecordMode
from .pointer import PointerPipeline
from .profile import Profile, fsm_parts
from .sounds import Sounds
from .train import train_personal
from .tremor import TremorSim


LIVE_EVERY_S = 1.0   # how often results/live.json is rewritten for the lab
UNLOCK_S, UNLOCK_STILL_PX = 2.0, 20.0   # an open palm held this still, this long, ends a pause
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
    luma: float = 255.0   # mean brightness of the view before anything is drawn on it


class Session:
    """Everything the frame loop shares, minus camera, tracker and window (those stay in app.py).

    tick() runs one frame through the current mode and returns the image to show.
    on_key() handles a key: the mode gets it first, then the global keys.
    """

    def __init__(self, runner, rules, mlp, *, root, user=None, profile=None, dwell_on=None,
                 train_fn=train_personal, sounds=None, effects=None):
        self.runner, self.rules, self.mlp = runner, rules, mlp
        self.clf = mlp or rules
        self.root, self.user, self.train_fn = root, user, train_fn
        self.data_dir, self.models_dir = os.path.join(root, "data"), os.path.join(root, "models")
        self.profile = profile or Profile(user=user or "guest")      # a guest profile is never saved
        self.smoother, self.dwell = VoteSmoother(), DwellDetector()
        self.unlock_dwell = DwellDetector(dwell_s=UNLOCK_S, still_px=UNLOCK_STILL_PX)
        self.apply_bindings()
        self.sounds = sounds or Sounds(enabled=self.profile.sounds)
        self.effects = effects or Effects(voice=self.profile.voice)
        self.hinter, self.hints_on, self.hint = Hinter(), self.profile.hints, None
        self.pointer = PointerPipeline(self.profile.fc_min)
        self.pointer.mode = self.profile.filter
        self.apply_region(self.profile.region)
        self.sim = TremorSim()
        self.dwell_on = self.profile.dwell if dwell_on is None else dwell_on
        self.mini, self.view_dirty = False, False
        self.zoom, self.xray, self.ms_tracker = 1.5, False, 0.0
        self.recorder, self.rejected = None, 0
        self.mode = LiveMode()
        self.last_event, self.mini_label, self.prev_t = "none", "", None
        self.obs, self.live_t = None, None

    def tick(self, frame, pts, hand, t, tracker_ms=0.0):
        self.ms_tracker = tracker_ms
        fps = 0.0 if self.prev_t is None else 1.0 / max(t - self.prev_t, 1e-5)
        self.prev_t = t
        view = hud.canon_view(frame)
        h, w = view.shape[:2]
        luma = float(view[::8, ::8].mean())
        raw = None
        if pts is not None:
            if self.sim.on:
                pts = self.sim.apply(pts, t, w, h)
            hud.draw_skeleton(view, pts)
            raw = (float(pts[F.MIDDLE_MCP, 0] * w), float(pts[F.MIDDLE_MCP, 1] * h))
        self.obs = Obs(t, pts, hand, raw, w, h, fps, luma)
        self.mode.tick(self, self.obs, view)
        self._write_live(t)
        if self.mode.name == "live":
            if self.mini:
                return hud.mini_frame(view, self.mini_label)
            if self.xray:
                view = np.hstack([view, hud.draw_xray(h, self.taps())])
        return hud.zoomed(view, self.zoom)

    def taps(self):
        """Everything the X-ray panel shows, read from where it already lives."""
        p, live = self.pointer, self.mode
        return dict(clf=self.clf.name, fps=self.obs.fps, ms_tracker=self.ms_tracker, ms_classifier=live.clf_ms,
                    labels=self.clf.labels, probs=live.probs, gate=self.smoother.min_conf,
                    votes=[label for label, _ in self.smoother.buf], label=live.label,
                    state=self.fsm.state, state_s=self.obs.t - self.fsm.t_enter, last_event=self.last_event,
                    trail=p.trail, reading=p.reading, left=p.left, mode=p.mode, sim=self.sim.on)

    def apply_bindings(self):
        """The profile's roles and taught signs decide which pose does what. Restarts the state machine."""
        grip, scroll, hold, hold_s = fsm_parts(self.profile.roles, self.profile.signs)
        self.fsm = AirPointFSM(hold_events=hold, grip=grip, scroll=scroll, hold_s=hold_s)

    def set_locked(self, locked):
        """The user's pause. Locking lets go of a held button; a still open palm unlocks (live mode)."""
        if locked:
            self.runner.lock()
        else:
            self.runner.unlock()
        self.unlock_dwell.reset()
        self.sounds.play("lock" if locked else "unlock")

    def apply_region(self, region):
        """Use this comfort box: it maps to the whole screen, and the pointer filters follow its gain."""
        self.runner.active_region = tuple(region)
        self.pointer.set_gain((ACTIVE_REGION[2] - ACTIVE_REGION[0]) / (region[2] - region[0]))

    def state(self):
        """What is driving the cursor right now, for the lab to label a test block with."""
        r = self.pointer.reading
        return dict(ts=time.time(), user=self.user, mode=self.mode.name, mouse=self.runner.enabled,
                    dwell=self.dwell_on, filter=self.pointer.mode,
                    sim=dict(on=self.sim.on, hz=self.sim.hz, amp=self.sim.amp),
                    locked=self.runner.locked, gain=round(self.pointer.gain, 3),
                    tremor_locked=r.locked, lock_hz=round(r.lock_hz, 2), classifier=self.clf.name,
                    fps=round(self.obs.fps, 1) if self.obs else 0.0)

    def _write_live(self, t):
        if self.live_t is not None and t - self.live_t < LIVE_EVERY_S:
            return
        self.live_t = t
        path = os.path.join(self.root, "results", "live.json")
        try:                                   # the lab hook must never break the frame loop
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".tmp", "w") as f:
                json.dump(self.state(), f)
            os.replace(path + ".tmp", path)
        except OSError:
            pass

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
        elif key == ord("x"):
            self.xray, self.view_dirty = not self.xray, True
        elif key == ord("p"):
            self.set_locked(not self.runner.locked)
        elif key == ord("s"):
            self.sounds.enabled = not self.sounds.enabled
        elif key == ord("h"):
            self.hints_on = not self.hints_on
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
        try:
            os.remove(os.path.join(self.root, "results", "live.json"))
        except OSError:
            pass
