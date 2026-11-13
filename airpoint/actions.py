ACTIVE_REGION = (0.15, 0.15, 0.85, 0.80)  # inner frame box that maps to the whole screen


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

    def mouse_down(self):
        self.calls.append(("down",))

    def mouse_up(self):
        self.calls.append(("up",))

    def scroll(self, ticks):
        self.calls.append(("scroll", ticks))

    def hotkey(self, *keys):
        self.calls.append(("hotkey",) + keys)


class PyAutoGuiMouse:
    def __init__(self):
        import pyautogui
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        self.pg = pyautogui

    def size(self):
        return tuple(self.pg.size())

    def move(self, x, y):
        self.pg.moveTo(x, y)

    def click(self):
        self.pg.click()

    def right_click(self):
        self.pg.rightClick()

    def mouse_down(self):
        self.pg.mouseDown()

    def mouse_up(self):
        self.pg.mouseUp()

    def scroll(self, ticks):
        self.pg.scroll(ticks)

    def hotkey(self, *keys):
        self.pg.hotkey(*keys)


class ActionRunner:
    """Starts disabled. toggle() flips it and always releases a held button first."""

    def __init__(self, mouse, screen_size, active_region=ACTIVE_REGION):
        self.mouse = mouse
        self.screen_w, self.screen_h = screen_size
        self.active_region = active_region
        self.enabled = False
        self.button_down = False

    def toggle(self):
        self.release()
        self.enabled = not self.enabled

    def to_screen(self, nx, ny):
        x0, y0, x1, y1 = self.active_region
        sx = min(max((nx - x0) / (x1 - x0), 0.0), 1.0) * (self.screen_w - 1)
        sy = min(max((ny - y0) / (y1 - y0), 0.0), 1.0) * (self.screen_h - 1)
        return sx, sy

    def move(self, nx, ny):
        if self.enabled:
            self.mouse.move(*self.to_screen(nx, ny))

    def apply(self, events):
        if not self.enabled:
            return
        for e in events:
            if e.kind == "click":
                self.mouse.click()
            elif e.kind == "right_click":
                self.mouse.right_click()
            elif e.kind == "drag_start":
                self.mouse.mouse_down()
                self.button_down = True
            elif e.kind == "drag_end":
                self.release()
            elif e.kind == "scroll":
                self.mouse.scroll(e.arg)
            elif e.kind == "hotkey":
                self.mouse.hotkey(*e.arg)

    def release(self):
        if self.button_down:
            self.mouse.mouse_up()
            self.button_down = False
