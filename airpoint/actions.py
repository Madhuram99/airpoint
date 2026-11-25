import queue
import threading
import unicodedata

ACTIVE_REGION = (0.15, 0.15, 0.85, 0.80)  # inner frame box that maps to the whole screen
PG_KEYS = {"control": "ctrl"}             # our key names -> pyautogui's, where they differ
HOTKEY_GAP_S = 0.04                       # between the keys of a combo: macOS needs a moment to see a modifier go down
PLAIN = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-",
                       "\u2026": "...", "\u00a0": " "})


def typable(text):
    """Text as US-keyboard keystrokes can produce it: curly quotes and dashes straightened, accents
    dropped, anything else that has no key left out (pyautogui would skip it silently anyway)."""
    text = unicodedata.normalize("NFKD", text.translate(PLAIN))
    return "".join(ch for ch in text if ch == "\n" or 32 <= ord(ch) < 127)


class FakeMouse:
    def __init__(self):
        self.calls = []

    def size(self):
        return (1000, 500)

    def move(self, x, y):
        self.calls.append(("move", round(x), round(y)))

    def click(self):
        self.calls.append(("click",))

    def right_click(self):
        self.calls.append(("right_click",))

    def double_click(self):
        self.calls.append(("double_click",))

    def mouse_down(self):
        self.calls.append(("down",))

    def mouse_up(self):
        self.calls.append(("up",))

    def scroll(self, ticks):
        self.calls.append(("scroll", ticks))

    def hotkey(self, *keys):
        self.calls.append(("hotkey",) + keys)

    def write(self, text):
        self.calls.append(("write", text))

    def flush(self, timeout=2.0):
        return True

    typing = False


def in_front():
    """True when AirPoint's own window has the keyboard (macOS). Words typed then have nowhere to go."""
    try:
        import os
        from AppKit import NSWorkspace
        return NSWorkspace.sharedWorkspace().frontmostApplication().processIdentifier() == os.getpid()
    except Exception:
        return False


class PyAutoGuiMouse:
    """Mouse calls go straight to the OS. Keystrokes take 10 ms or more each, so they queue for one
    worker thread: in order, never two at once, and never holding up a frame."""

    def __init__(self, pg=None):
        if pg is None:
            import pyautogui as pg
            pg.FAILSAFE = False
            pg.PAUSE = 0
        self.pg = pg
        self.keys = queue.Queue()
        threading.Thread(target=self._press, daemon=True).start()

    def _press(self):
        while True:
            work = self.keys.get()
            try:
                work()
            except Exception as e:           # one bad keystroke must not end typing for the rest of the session
                print(f"[AirPoint] keystroke failed: {type(e).__name__}: {e}")
            finally:
                self.keys.task_done()

    @property
    def typing(self):
        """True while keystrokes are queued or going out."""
        return self.keys.unfinished_tasks > 0

    def size(self):
        return tuple(self.pg.size())

    def move(self, x, y):
        self.pg.moveTo(x, y)

    def click(self):
        self.pg.click()

    def right_click(self):
        self.pg.rightClick()

    def double_click(self):
        self.pg.doubleClick()

    def mouse_down(self):
        self.pg.mouseDown()

    def mouse_up(self):
        self.pg.mouseUp()

    def scroll(self, ticks):
        self.pg.scroll(ticks)

    def hotkey(self, *keys):
        keys = [PG_KEYS.get(k, k) for k in keys]
        self.keys.put(lambda: self.pg.hotkey(*keys, interval=HOTKEY_GAP_S))

    def write(self, text):
        self.keys.put(lambda: self.pg.write(typable(text), interval=0.0))

    def flush(self, timeout=2.0):
        """Wait for the keystrokes already queued. -> False if they did not finish in time."""
        done = threading.Event()
        self.keys.put(done.set)
        return done.wait(timeout)


class ActionRunner:
    """Starts disabled. toggle() flips it and always releases a held button first."""

    def __init__(self, mouse, screen_size, active_region=ACTIVE_REGION):
        self.mouse = mouse
        self.screen_w, self.screen_h = screen_size
        self.active_region = active_region
        self.enabled = False
        self.locked = False       # the user's pause: nothing reaches the OS until unlocked
        self.button_down = False
        self.last_xy = None       # where the cursor was last put, in screen points

    def toggle(self):
        self.release()
        self.enabled = not self.enabled

    def to_screen(self, nx, ny):
        x0, y0, x1, y1 = self.active_region
        sx = min(max((nx - x0) / (x1 - x0), 0.0), 1.0) * (self.screen_w - 1)
        sy = min(max((ny - y0) / (y1 - y0), 0.0), 1.0) * (self.screen_h - 1)
        return sx, sy

    def lock(self):
        self.release()
        self.locked = True

    def unlock(self):
        self.locked = False

    def move(self, nx, ny):
        if self.enabled and not self.locked:
            self.last_xy = self.to_screen(nx, ny)
            self.mouse.move(*self.last_xy)

    def apply(self, events):
        if not self.enabled or self.locked:
            return
        for e in events:
            if e.kind == "click":
                self.mouse.click()
            elif e.kind == "right_click":
                self.mouse.right_click()
            elif e.kind == "double_click":
                self.mouse.double_click()
            elif e.kind == "drag_start":
                self.mouse.mouse_down()
                self.button_down = True
            elif e.kind == "drag_end":
                self.release()
            elif e.kind == "scroll":
                self.mouse.scroll(e.arg)
            elif e.kind == "point":          # straight to a screen point: the AI helper's steps
                self.last_xy = tuple(e.arg)
                self.mouse.move(*self.last_xy)
            elif e.kind == "hotkey":
                self.mouse.hotkey(*e.arg)
            elif e.kind == "type":
                self.mouse.write(e.arg)

    def release(self):
        if self.button_down:
            self.mouse.mouse_up()
            self.button_down = False

    def flush(self, timeout=2.0):
        """Wait for keystrokes that are still on their way to the OS."""
        return self.mouse.flush(timeout)

    @property
    def typing(self):
        return self.mouse.typing
