"""A short system sound when an action really reaches the operating system. macOS only; silent elsewhere."""
import os
import shutil
import subprocess

FOLDER = "/System/Library/Sounds"
FILES = {"click": "Tink", "double_click": "Tink", "right_click": "Pop", "drag_start": "Bottle", "drag_end": "Purr",
         "lock": "Submarine", "unlock": "Glass", "talk_start": "Morse", "talk_stop": "Frog"}


def _afplay():
    """-> a function that starts one sound without waiting, or None where `afplay` does not exist."""
    exe = shutil.which("afplay")
    if exe is None:
        return None
    running = []

    def play(path):
        running[:] = [p for p in running if p.poll() is None]      # reap finished players
        if len(running) < 4:                                        # never pile up during fast scrolling
            running.append(subprocess.Popen([exe, "-v", "0.5", path], stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL))
    return play


class Sounds:
    def __init__(self, enabled=True, player=None):
        self.enabled = enabled
        self.player = _afplay() if player is None else player

    def play(self, kind):
        """`kind` is an event kind (click, right_click, drag_start, drag_end) or lock / unlock."""
        name = FILES.get(kind)
        if self.enabled and name and self.player:
            path = os.path.join(FOLDER, name + ".aiff")
            if os.path.exists(path):
                self.player(path)
