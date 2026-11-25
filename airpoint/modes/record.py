import os

from .. import hud
from ..panel import name_of
from ..recorder import Recorder, RecordSession, capture


class RecordMode:
    """Prompted captures into data/<user>.csv. SPACE = all poses, 1-6 = one pose."""

    name = "record"

    def __init__(self, s):
        s.recorder = s.recorder or Recorder(os.path.join(s.data_dir, f"{s.user}.csv"), s.user)
        self.session = RecordSession()

    def tick(self, s, o, frame):
        phase, pose, left = self.session.update(o.t)
        lost = capture(self.session, s.recorder, pose, o.pts, o.hand, o.t)
        s.rejected += lost
        sign = name_of(pose).lower() if pose else ""
        title, sub = {
            "idle": (f"Record signs for {s.user}", "Space records all six signs. 1 to 6 records one. R goes back."),
            "countdown": (f"Make the {sign} sign in {left:.0f}", "Raise your hand."),
            "capturing": (f"Hold the {sign} sign: {left:.0f} s", "Tilt it, move closer, move away."),
            "done": ("Done", "1 to 6 records one sign again. R goes back."),
        }[phase]
        hud.draw_prompt(frame, title, f"{sub} Kept {s.recorder.count}, rejected {s.rejected}.",
                        "Sign lost: this frame was not saved." if lost else None)

    def on_key(self, s, key, t):
        if key == ord(" "):
            self.session.redo(t)
            return True
        if ord("1") <= key <= ord("6"):
            self.session.select(key - ord("1"), t)
            return True
        return False

    def close(self, s):
        pass
