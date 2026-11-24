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
        title = {
            "idle": f"RECORD {s.user}: SPACE = all poses, 1-6 = one pose",
            "countdown": f"raise hand, make {pose}   {left:.0f}",
            "capturing": f"HOLD {pose}   {left:.0f}s" + ("   << SIGN LOST" if lost else ""),
            "done": "done - q to quit, 1-6 to redo one pose",
        }[phase]
        hud.draw_prompt(frame, title, f"tilt, move closer, move away | kept {s.recorder.count} | "
                                      f"rejected {s.rejected} | {hud.KEY_HINT}")

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
