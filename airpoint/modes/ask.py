from .. import hud
from ..helper import CONSENT

ENTER, ESC, BACKSPACE = (13, 10), 27, (8, 127)
MAX_REQUEST, SHOWN = 200, 54


class AskMode:
    """`a`: type a request for the AI helper. The first time in a session this is also where the helper
    is switched on, under the line that says what it sends; Enter with nothing typed only switches it on."""

    name, typing, busy = "ask", True, False

    def __init__(self, s):
        self.text, self.task, self.exit = "", None, False
        self.first = not s.helper.armed

    def tick(self, s, o, frame):
        hud.draw_prompt(frame, "SWITCH ON THE AI HELPER" if self.first else "ASK THE AI HELPER",
                        f"{self.text[-SHOWN:]}_   Enter = go | Esc = cancel",
                        f"this {CONSENT}" if self.first else None)

    def on_key(self, s, key, t):
        if key == ESC:
            self.exit = True
        elif key in ENTER:
            s.helper.armed = True
            self.task, self.exit = self.text.strip() or None, True
        elif key in BACKSPACE:
            self.text = self.text[:-1]
        elif 32 <= key <= 126 and len(self.text) < MAX_REQUEST:
            self.text += chr(key)
        return True                        # while asking, every key belongs to the prompt

    def close(self, s):
        pass
