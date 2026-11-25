import os
from collections import Counter

from .. import hud
from ..panel import name_of
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
                self.trainer = Trainer(lambda: s.train_fn(s.data_dir, s.models_dir, s.user))
                self.trainer.start()
            elif self.trainer.done:
                self._finish(s)
        sign = name_of(pose).lower() if pose else ""
        title, sub = {
            "countdown": (f"Calibration starts in {left:.0f}", "Open your palm and hold it still."),
            "still": (f"Hold still: {left:.0f} s", "Measuring how steady your hand is."),
            "reach": (f"Reach: {left:.0f} s", "Move your hand as far as is comfortable, in every direction."),
            "poses_countdown": (f"Make the {sign} sign in {left:.0f}", "Raise your hand."),
            "capturing": (f"Hold the {sign} sign: {left:.0f} s", "Tilt a little, keep the sign."),
            "training": ("Learning your hand...", "A few seconds."),
            "done": ("Done", calib.message + "."),
            "failed": ("Could not learn your hand", f"The old model is kept: {calib.message}."),
        }[phase]
        hud.draw_prompt(frame, title, f"{sub} Kept {self.rec.count}. C closes this.",
                        "Sign lost: this frame was not saved." if lost else None)

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
        s.apply_bindings()
        no_grip = not prof.roles["grip"]
        if no_grip:                                    # no sign left to click with: pointing alone takes over
            prof.dwell = s.dwell_on = True
            prof.wheel = s.wheel_on = True
        profiles.save(s.root, prof)
        calib.phase = "done"
        calib.message = (f"{r['acc_on_own_frames'] * 100:.0f}% on your frames"
                         + (f"; skipped: {', '.join(prof.skipped)}" if prof.skipped else "")
                         + ("; dwell and the wheel are on" if no_grip else ""))

    def on_key(self, s, key, t):
        return False

    def close(self, s):
        self.rec.close()
