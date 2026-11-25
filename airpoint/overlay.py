"""A glow at the real cursor (macOS): dwell ring, action ripples, tags, hints and the point-only wheel.

Two halves. `Overlay` is the client AirPoint holds: it starts this module as a second process and
writes one JSON line per frame to its stdin, never blocking the frame loop. The process (`main`)
draws a transparent, click-through window centred on the cursor and exits when the pipe closes,
so an overlay cannot outlive AirPoint. AppKit is imported only inside the drawing process.

State, all optional:  show, x, y (screen points, origin top-left), ring 0..1, tag, hint,
pulse (click | right_click | double_click | drag_start | drag_end, one-shot),
wheel {items, pick, progress, dot}.
"""
import json
import math
import os
import subprocess
import sys
import time

from .proc import spawn

SIZE = 320                      # window side, points
PULSE_S = 0.35
R_GLOW, R_RING, R_HUB, R_WHEEL = 14.0, 21.0, 38.0, 120.0
LABELS = {"talk": "talk", "right": "right", "drag": "drag", "scroll": "scroll", "double": "double", "pause": "pause"}
CYAN, AMBER, GREEN, PURPLE, RED, WHITE, DARK = ((0.13, 0.83, 0.93), (0.98, 0.75, 0.14), (0.20, 0.83, 0.60),
                                               (0.65, 0.55, 0.98), (0.97, 0.44, 0.44), (0.95, 0.96, 0.98),
                                               (0.04, 0.06, 0.10))
PULSE_COLOR = {"click": GREEN, "double_click": GREEN, "right_click": AMBER, "drag_start": PURPLE, "drag_end": PURPLE}


class Overlay:
    """The client. Every failure just turns the overlay off; AirPoint carries on without it."""

    def __init__(self, cmd=None):
        self.proc = None
        try:
            self.proc = spawn(cmd or [sys.executable, "-m", "airpoint.overlay"], stdin=subprocess.PIPE)
            os.set_blocking(self.proc.stdin.fileno(), False)     # a stuck overlay must never stall a frame
        except (OSError, ValueError):
            self.proc = None

    @property
    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def send(self, state):
        """One state per frame. -> False if it could not be delivered (frame dropped or overlay gone)."""
        if not self.alive:
            return False
        try:
            os.write(self.proc.stdin.fileno(), (json.dumps(state, separators=(",", ":")) + "\n").encode())
            return True
        except BlockingIOError:
            return False                    # pipe full: skip this frame
        except (BrokenPipeError, OSError, ValueError):
            self.close()
            return False

    def close(self):
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()              # the process exits on end of input
            proc.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()


# ----------------------------------------------------------------- drawing (inside the overlay process)

def _color(rgb, alpha=1.0):
    from AppKit import NSColor
    return NSColor.colorWithCalibratedRed_green_blue_alpha_(rgb[0], rgb[1], rgb[2], alpha)


def _circle(c, r):
    from AppKit import NSBezierPath
    from Foundation import NSMakeRect
    return NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(c - r, c - r, 2 * r, 2 * r))


def _arc(c, r, start, end, clockwise=True):
    from AppKit import NSBezierPath
    path = NSBezierPath.bezierPath()
    path.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_clockwise_((c, c), r, start, end, clockwise)
    return path


def _text(text, x, y, size, rgb, bold=True):
    """Draw `text` centred on (x, y)."""
    from AppKit import NSFont, NSFontAttributeName, NSForegroundColorAttributeName
    from Foundation import NSString
    font = NSFont.boldSystemFontOfSize_(size) if bold else NSFont.systemFontOfSize_(size)
    attrs = {NSFontAttributeName: font, NSForegroundColorAttributeName: _color(rgb)}
    s = NSString.stringWithString_(text)
    w, h = s.sizeWithAttributes_(attrs)
    s.drawAtPoint_withAttributes_((x - w / 2, y - h / 2), attrs)
    return w, h


def _pill(text, y, rgb, c):
    """A dark rounded label centred under the cursor at height y."""
    from AppKit import NSBezierPath, NSFont, NSFontAttributeName
    from Foundation import NSMakeRect, NSString
    w, h = NSString.stringWithString_(text).sizeWithAttributes_({NSFontAttributeName: NSFont.boldSystemFontOfSize_(12)})
    w = min(w, SIZE - 28)
    _color(DARK, 0.82).set()
    NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(NSMakeRect(c - w / 2 - 9, y - h / 2 - 4, w + 18, h + 8), 9, 9).fill()
    _text(text, c, y, 12, rgb)


def paint(state, pulse=None, now=None, size=SIZE):
    """Draw one state into the current AppKit graphics context. `pulse` = (kind, started_at)."""
    from AppKit import NSBezierPath
    c = size / 2.0
    if not state.get("show", True):
        return
    wheel = state.get("wheel")
    if wheel:
        _color(DARK, 0.72).set()
        _circle(c, R_WHEEL + 6).fill()
        for i, item in enumerate(wheel["items"]):
            mid = 90.0 - 60.0 * i                               # AppKit angles: 0 = right, growing counter-clockwise
            lit = i == wheel["pick"]
            wedge = NSBezierPath.bezierPath()
            wedge.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_clockwise_((c, c), R_WHEEL, mid + 29, mid - 29, True)
            wedge.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_clockwise_((c, c), R_HUB + 4, mid - 29, mid + 29, False)
            wedge.closePath()
            _color(CYAN if lit else WHITE, 0.30 if lit else 0.07).set()
            wedge.fill()
            if lit:
                arc = _arc(c, R_WHEEL + 3, mid + 29, mid + 29 - 58 * wheel["progress"])
                arc.setLineWidth_(5)
                _color(CYAN).set()
                arc.stroke()
            a = math.radians(mid)
            _text(LABELS.get(item, item), c + 80 * math.cos(a), c + 80 * math.sin(a), 13, CYAN if lit else WHITE)
        hub = wheel["pick"] == -1
        _color(CYAN if hub else WHITE, 0.25 if hub else 0.08).set()
        _circle(c, R_HUB).fill()
        _text("click", c, c + 16, 11, CYAN if hub else WHITE)
        if hub:
            arc = _arc(c, R_HUB + 3, 90, 90 - 360 * wheel["progress"])
            arc.setLineWidth_(4)
            _color(CYAN).set()
            arc.stroke()
        dx, dy = wheel.get("dot", (0.0, 0.0))                   # where the hand is steering
        _color(AMBER).set()
        from Foundation import NSMakeRect
        NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(c + dx * R_WHEEL - 5, c - dy * R_WHEEL - 5, 10, 10)).fill()
    else:
        glow = _circle(c, R_GLOW)
        glow.setLineWidth_(2.5)
        _color(CYAN, 0.9).set()
        glow.stroke()
        ring = float(state.get("ring") or 0.0)
        if ring > 0.0:
            arc = _arc(c, R_RING, 90, 90 - 360 * min(ring, 1.0))
            arc.setLineWidth_(4)
            _color(AMBER).set()
            arc.stroke()
    if pulse is not None:
        k = ((now if now is not None else time.time()) - pulse[1]) / PULSE_S
        if 0.0 <= k < 1.0:
            ripple = _circle(c, R_GLOW + 34 * k)
            ripple.setLineWidth_(3)
            _color(PULSE_COLOR.get(pulse[0], GREEN), 1.0 - k).set()
            ripple.stroke()
    below = c - (R_WHEEL + 22 if wheel else 44)
    if state.get("tag"):
        tint = PURPLE if state["tag"].startswith("AI") else {"SIMULATED": RED, "LISTENING": GREEN, "THINKING": GREEN}.get(state["tag"], WHITE)
        _pill(state["tag"], below, tint, c)
        below -= 26
    if state.get("hint") and below > 12:
        _pill(state["hint"], below, AMBER, c)


def render_png(state, path, pulse=None, now=None, size=SIZE):
    """Paint one state off-screen and save it. Used to check the look without a person at the screen."""
    from AppKit import NSBitmapImageRep, NSDeviceRGBColorSpace, NSGraphicsContext, NSPNGFileType
    rep = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
        None, size, size, 8, 4, True, False, NSDeviceRGBColorSpace, 0, 0)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.setCurrentContext_(NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep))
    paint(state, pulse, now, size)
    NSGraphicsContext.restoreGraphicsState()
    rep.representationUsingType_properties_(NSPNGFileType, None).writeToFile_atomically_(path, True)
    return path


# ----------------------------------------------------------------- the overlay process

def main():
    import queue
    import threading

    from AppKit import (NSApplication, NSApplicationActivationPolicyAccessory, NSBackingStoreBuffered, NSColor,
                        NSScreen, NSScreenSaverWindowLevel, NSView, NSWindow, NSWindowAnimationBehaviorNone,
                        NSWindowCollectionBehaviorCanJoinAllSpaces, NSWindowCollectionBehaviorFullScreenAuxiliary,
                        NSWindowCollectionBehaviorIgnoresCycle, NSWindowCollectionBehaviorStationary)
    from Foundation import NSMakeRect, NSObject, NSTimer

    inbox, done = queue.Queue(), threading.Event()
    box = {"state": {"show": False}, "pulse": None}

    class OverlayView(NSView):
        def isOpaque(self):
            return False

        def drawRect_(self, rect):
            NSColor.clearColor().set()
            from AppKit import NSRectFill
            NSRectFill(rect)
            paint(box["state"], box["pulse"])

    class Ticker(NSObject):
        def tick_(self, timer):
            fresh = False
            while True:
                try:
                    state = inbox.get_nowait()
                except queue.Empty:
                    break
                if state.get("pulse"):
                    box["pulse"] = (state["pulse"], time.time())
                box["state"], fresh = state, True
            if fresh:
                s = box["state"]
                if s.get("show", True) and "x" in s:
                    top = NSScreen.screens()[0].frame().size.height        # AppKit's origin is bottom-left
                    window.setFrameOrigin_((s["x"] - SIZE / 2, top - s["y"] - SIZE / 2))
                    window.orderFrontRegardless()
                else:
                    window.orderOut_(None)
            rippling = box["pulse"] is not None and time.time() - box["pulse"][1] < PULSE_S
            if fresh or rippling:
                view.setNeedsDisplay_(True)
            if done.is_set() and inbox.empty():
                app.terminate_(None)

    def reader():
        for line in sys.stdin:
            try:
                inbox.put(json.loads(line))
            except ValueError:
                continue
        done.set()                          # end of input: AirPoint has gone

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)       # no Dock icon, never takes focus
    window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(NSMakeRect(0, 0, SIZE, SIZE), 0,
                                                                           NSBackingStoreBuffered, False)
    window.setOpaque_(False)
    window.setBackgroundColor_(NSColor.clearColor())
    window.setIgnoresMouseEvents_(True)                                    # clicks go straight through
    window.setHasShadow_(False)
    window.setAnimationBehavior_(NSWindowAnimationBehaviorNone)            # appear and vanish at once
    window.setLevel_(NSScreenSaverWindowLevel)
    window.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorStationary
                                  | NSWindowCollectionBehaviorFullScreenAuxiliary | NSWindowCollectionBehaviorIgnoresCycle)
    view = OverlayView.alloc().initWithFrame_(NSMakeRect(0, 0, SIZE, SIZE))
    window.setContentView_(view)
    ticker = Ticker.alloc().init()
    NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(1.0 / 60.0, ticker, "tick:", None, True)
    threading.Thread(target=reader, daemon=True).start()
    app.run()


if __name__ == "__main__":
    main()
