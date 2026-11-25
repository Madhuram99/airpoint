from .. import hud
from .. import profile as profiles
from ..face import FaceCalib

PROMPTS = {
    "rest": ("Look at the screen", "Rest your head where it is comfortable."),
    "reach": ("Turn your head", "As far as is comfortable: left, right, up, down."),
    "mouth": ("Open your mouth", "Wide, and hold it."),
    "brows": ("Raise your eyebrows", "And hold them up."),
}


class FaceCalibrateMode:
    """`c` in face mode: resting pose, comfortable head range, and how big this face's expressions get.
    No training: the result is a handful of numbers in the profile."""

    name = "face_calibrate"
    busy = False

    def __init__(self, s, t):
        self.calib, self.message, self.finished, self.exit = FaceCalib(t), "", False, False

    def tick(self, s, o, frame):
        phase, left = self.calib.phase(o.t)
        if phase != "done":
            self.calib.add(o.t, o.face)
            title, sub = PROMPTS[phase]
            hud.draw_prompt(frame, f"{title}: {left:.0f} s", sub, None if o.face is not None else "No face in view.")
            return
        if not self.finished:
            self.finished = True
            result = self.calib.finish()
            if result is None:
                self.message = "your face was not in view long enough; nothing changed"
            else:
                s.face_cfg, s.face_centring = result, None
                s.profile.face = result
                if s.user:
                    profiles.save(s.root, s.profile)
                s.refresh_face()
                self.message = (f"head range {result['span_yaw']:.0f} x {result['span_pitch']:.0f} degrees"
                                + (f"; not seen: {', '.join(self.calib.dropped)}" if self.calib.dropped else ""))
        hud.draw_prompt(frame, "Done", f"{self.message[0].upper() + self.message[1:]}. Any key goes back.")

    def on_key(self, s, key, t):
        if self.finished or key == 27:
            self.exit = True
        return True

    def close(self, s):
        pass
