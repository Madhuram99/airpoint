import os
from collections import Counter

from .. import hud
from .. import profile as profiles
from ..calibration import CalibSession, Trainer, comfort_box
from ..poses import POSES
from ..recorder import Recorder, capture


SKIP_MIN_FRAMES, SKIP_SHARE = 20, 0.30   # a pose is "could not be held" below either


class CalibrateMode:
    """`c`: hold still -> six short captures -> personal retrain off-thread -> hot-swap -> profile."""

    name = "calibrate"

    def __init__(self, s, t):
        self.rec = Recorder(os.path.join(s.data_dir, f"{s.user}_calib.csv"), s.user)
        self.calib = CalibSession(POSES, t)
        self.trainer, self.view = None, (640, 360)
        self.kept, self.lost = Counter(), Counter()     # per pose: frames saved / frames the guard refused

    @property
    def busy(self):
        return self.calib.phase == "training"

    def tick(self, s, o, frame):
        calib = self.calib
        phase, pose, left = calib.update(o.t)
        self.view = (o.w, o.h)
        if o.raw is not None and phase in ("still", "reach"):
            (calib.still_points if phase == "still" else calib.reach_points).append(o.raw)
        lost = capture(calib, self.rec, pose, o.pts, o.hand, o.t)
        if calib.capturing and o.pts is not None:
            (self.lost if lost else self.kept)[pose] += 1
        if phase == "training":
            if self.trainer is None:
                self.rec.close()
                path = self.rec.path
                self.trainer = Trainer(lambda: s.train_fn(s.data_dir, path, s.models_dir, s.user))
                self.trainer.start()
            elif self.trainer.done:
                self._finish(s)
        title, sub = {
            "countdown": (f"CALIBRATE   {left:.0f}", "open palm, hold still"),
            "still": (f"HOLD STILL   {left:.0f}s", "measuring how steady your hand is"),
            "reach": (f"REACH   {left:.0f}s", "move your hand as far as is comfortable, in every direction"),
            "poses_countdown": (f"make {pose}   {left:.0f}", "raise your hand"),
            "capturing": (f"HOLD {pose}   {left:.0f}s", "tilt a little, keep the sign"),
            "training": ("Learning your hand...", "a few seconds"),
            "done": ("done", calib.message),
            "failed": ("retrain failed", f"old model kept: {calib.message}"),
        }[phase]
        hud.draw_prompt(frame, title, f"{sub} | kept {self.rec.count} | c = close",
                        "SIGN LOST - this frame was not saved" if lost else None)

    def _finish(self, s):
        calib, trainer, self.trainer = self.calib, self.trainer, None
        if trainer.error:
            calib.phase, calib.message = "failed", trainer.error
            return
        r = trainer.result
        s.use_model(r["model"], r["labels"])
        s.pointer.fc_min = calib.fc_min
        prof = s.profile
        prof.model, prof.labels = os.path.relpath(r["model"], s.root), os.path.relpath(r["labels"], s.root)
        prof.acc_on_own_frames, prof.fc_min = r["acc_on_own_frames"], calib.fc_min
        prof.region = comfort_box(calib.reach_points, *self.view) or prof.region
        s.apply_region(prof.region)
        prof.skipped = [p for p in POSES if self.kept[p] < max(SKIP_MIN_FRAMES, SKIP_SHARE * (self.kept[p] + self.lost[p]))]
        prof.roles = profiles.drop_poses(prof.roles, prof.skipped)
        s.apply_roles(prof.roles)
        no_grip = not prof.roles["grip"]
        if no_grip:                                    # no sign left to click with: hold-still clicking takes over
            prof.dwell = s.dwell_on = True
        profiles.save(s.root, prof)
        calib.phase = "done"
        calib.message = (f"{r['acc_on_own_frames'] * 100:.0f}% on your frames"
                         + (f" | skipped: {', '.join(prof.skipped)}" if prof.skipped else "")
                         + (" | dwell click ON" if no_grip else ""))

    def on_key(self, s, key, t):
        return False

    def close(self, s):
        self.rec.close()
