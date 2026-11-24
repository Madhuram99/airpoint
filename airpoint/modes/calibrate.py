import os

from .. import hud
from ..calibration import CalibSession, Trainer, save_profile
from ..poses import POSES
from ..recorder import Recorder, capture


class CalibrateMode:
    """`c`: hold still -> six short captures -> personal retrain off-thread -> hot-swap -> profile."""

    name = "calibrate"

    def __init__(self, s, t):
        self.rec = Recorder(os.path.join(s.data_dir, f"{s.user}_calib.csv"), s.user)
        self.calib = CalibSession(POSES, t)
        self.trainer = None

    @property
    def busy(self):
        return self.calib.phase == "training"

    def tick(self, s, o, frame):
        calib = self.calib
        phase, pose, left = calib.update(o.t)
        if phase == "still" and o.raw is not None:
            calib.still_points.append(o.raw)
        lost = capture(calib, self.rec, pose, o.pts, o.hand, o.t)
        if phase == "training":
            if self.trainer is None:
                self.rec.close()
                path = self.rec.path
                self.trainer = Trainer(lambda: s.train_fn(s.data_dir, path, s.models_dir, s.user))
                self.trainer.start()
            elif self.trainer.done:
                self._finish(s)
        title = {
            "countdown": f"CALIBRATE: open palm, hold still   {left:.0f}",
            "still": f"HOLD STILL   {left:.0f}s",
            "poses_countdown": f"make {pose}   {left:.0f}",
            "capturing": f"HOLD {pose}   {left:.0f}s" + ("   << SIGN LOST" if lost else ""),
            "training": "Learning your hand...",
            "done": f"done - {calib.message}",
            "failed": f"retrain failed, old model kept - {calib.message}",
        }[phase]
        hud.draw_prompt(frame, title, f"kept {self.rec.count} | c = close | {hud.KEY_HINT}")

    def _finish(self, s):
        calib, trainer, self.trainer = self.calib, self.trainer, None
        if trainer.error:
            calib.phase, calib.message = "failed", trainer.error
            return
        r = trainer.result
        s.use_model(r["model"], r["labels"])
        s.pointer.fc_min = calib.fc_min
        save_profile(s.root, s.user, calib.fc_min, r["model"], r["labels"], r["acc_on_own_frames"])
        calib.phase = "done"
        calib.message = f"{r['acc_on_own_frames'] * 100:.0f}% on your frames, fc_min {calib.fc_min}"

    def on_key(self, s, key, t):
        return False

    def close(self, s):
        self.rec.close()
