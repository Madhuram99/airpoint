import os

from .. import hud
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
        title, sub = {
            "idle": (f"RECORD {s.user}", "SPACE = all six poses | 1-6 = one pose | r = back"),
            "countdown": (f"make {pose}   {left:.0f}", "raise your hand"),
            "capturing": (f"HOLD {pose}   {left:.0f}s", "tilt, move closer, move away"),
            "done": ("done", "1-6 = redo one pose | r = back | q = quit"),
        }[phase]
        hud.draw_prompt(frame, title, f"{sub} | kept {s.recorder.count} | rejected {s.rejected}",
                        "SIGN LOST - this frame was not saved" if lost else None)

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
