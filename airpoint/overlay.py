"""A glow at the real cursor (macOS): what AirPoint is doing, shown where the person is looking.

Two halves. `Overlay` is the client AirPoint holds: it starts this module as a second process and
writes one JSON line per frame to its stdin, never blocking the frame loop. The process (`main`)
draws a transparent, click-through window centred on the cursor and exits when the pipe closes,
so an overlay cannot outlive AirPoint. AppKit is imported only inside the drawing process.

Colour says the same two things as in the window. Blue: the cursor is following. Yellow: an action
is happening or a choice is lit. Red: a warning. A word under the ring always says it too.

State, all optional:  show, x, y (screen points, origin top-left), tone (track | act | warn | idle),
ring 0..1 (the dwell countdown), tag (a word for a state that lasts: Dragging, Paused, Listening),
hint, pulse (click | right_click | double_click | drag_start | drag_end, one-shot: a ripple and its word),
wheel {items, pick, progress, dot, preview}.
"""
import json
import math
import os
import subprocess
import sys
import time

from . import proc
from .proc import spawn

SIZE = 440                      # window side, points
PULSE_S, WORD_S = 0.35, 0.9     # how long a ripple runs; how long its word stays
R_GLOW, R_HALO, R_RING, R_HUB, R_WHEEL = 14.0, 34.0, 21.0, 42.0, 150.0
LABELS = {"talk": "talk", "right": "right click", "drag": "drag", "scroll": "scroll", "double": "double click",
          "pause": "pause", "yes": "yes", "no": "no"}
SYMBOLS = {"talk": "mic.fill", "right": "contextualmenu.and.cursorarrow", "drag": "hand.draw.fill",
           "scroll": "arrow.up.arrow.down", "double": "cursorarrow.click.2", "pause": "pause.fill",
           "yes": "checkmark", "no": "xmark", "click": "cursorarrow.click"}
PULSE_WORDS = {"click": "click", "double_click": "double click", "right_click": "right click", "drag_start": "drag",
               "drag_end": "drop"}
USE_LINE, PREVIEW_LINE = "Point at a choice. Stay in the middle to click.", "The wheel is on. Hold still for 1 s to open it."
BLUE, YELLOW, RED, GREY, WHITE, INK = ((0.24, 0.48, 1.00), (1.00, 0.82, 0.25), (1.00, 0.35, 0.32), (0.62, 0.65, 0.70),
                                       (0.96, 0.97, 0.98), (0.08, 0.09, 0.12))
TONES = {"track": BLUE, "act": YELLOW, "warn": RED, "idle": GREY}


class Overlay:
    """The client. Every failure just turns the overlay off; AirPoint carries on without it."""

    def __init__(self, cmd=None):
        self.proc = None
        try:
            own, env = proc.module("airpoint.overlay")
            self.proc = spawn(cmd or own, stdin=subprocess.PIPE, **({} if cmd else {"env": env}))
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


def _circle(c, r, cy=None):
    from AppKit import NSBezierPath
    from Foundation import NSMakeRect
    cy = c if cy is None else cy
    return NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(c - r, cy - r, 2 * r, 2 * r))


def _arc(c, r, start, end, clockwise=True):
    from AppKit import NSBezierPath
    path = NSBezierPath.bezierPath()
    path.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_clockwise_((c, c), r, start, end, clockwise)
    return path


def _stroke(path, rgb, width, alpha=1.0, edge=True):
    """A coloured line on a thin dark one: it reads on a white page and on a dark photo alike."""
    if edge:
        path.setLineWidth_(width + 2.0)
        _color(INK, 0.55 * alpha).set()
        path.stroke()
    path.setLineWidth_(width)
    _color(rgb, alpha).set()
    path.stroke()


def _text(text, x, y, size, rgb, bold=True):
    """Draw `text` centred on (x, y). -> (width, height)."""
    from AppKit import NSFont, NSFontAttributeName, NSForegroundColorAttributeName
    from Foundation import NSString
    font = NSFont.boldSystemFontOfSize_(size) if bold else NSFont.systemFontOfSize_(size)
    attrs = {NSFontAttributeName: font, NSForegroundColorAttributeName: _color(rgb)}
    s = NSString.stringWithString_(text)
    w, h = s.sizeWithAttributes_(attrs)
    s.drawAtPoint_withAttributes_((x - w / 2, y - h / 2), attrs)
    return w, h


def _pill(text, y, c, ink=WHITE, ground=INK, size=13, limit=SIZE - 28):
    """A rounded label centred under the cursor at height y."""
    from AppKit import NSBezierPath, NSFont, NSFontAttributeName
    from Foundation import NSMakeRect, NSString
    while len(text) > 4:
        w, h = NSString.stringWithString_(text).sizeWithAttributes_({NSFontAttributeName: NSFont.boldSystemFontOfSize_(size)})
        if w <= limit:
            break
        text = text[:-2].rstrip() + "\u2026"
    w, h = NSString.stringWithString_(text).sizeWithAttributes_({NSFontAttributeName: NSFont.boldSystemFontOfSize_(size)})
    _color(ground, 0.92 if ground != INK else 0.86).set()
    NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(NSMakeRect(c - w / 2 - 10, y - h / 2 - 4, w + 20, h + 8),
                                                            (h + 8) / 2, (h + 8) / 2).fill()
    _text(text, c, y, size, ink)


def _symbol(name, x, y, points, rgb):
    """Draw a system symbol centred on (x, y) in one colour. -> False where the system has none."""
    try:
        from AppKit import NSImage, NSImageSymbolConfiguration
        from Foundation import NSMakeRect
        image = NSImage.imageWithSystemSymbolName_accessibilityDescription_(name, None)
        if image is None:
            return False
        config = NSImageSymbolConfiguration.configurationWithPointSize_weight_(points, 0.3)
        config = config.configurationByApplyingConfiguration_(
            NSImageSymbolConfiguration.configurationWithPaletteColors_([_color(rgb)]))
        image = image.imageWithSymbolConfiguration_(config)
        w, h = image.size()
        image.drawInRect_(NSMakeRect(x - w / 2, y - h / 2, w, h))
        return True
    except Exception:
        return False


def _halo(c, rgb):
    """A soft light around the ring: the eye finds the cursor before it reads the colour."""
    from AppKit import NSGradient
    NSGradient.alloc().initWithStartingColor_endingColor_(_color(rgb, 0.34), _color(rgb, 0.0)) \
        .drawInBezierPath_relativeCenterPosition_(_circle(c, R_HALO), (0.0, 0.0))


def _wheel(wheel, c):
    """Six slices around the cursor, each a picture and a word; the middle clicks."""
    from AppKit import NSBezierPath
    from Foundation import NSMakeRect
    preview = bool(wheel.get("preview"))
    _color(INK, 0.86).set()
    _circle(c, R_WHEEL + 8).fill()
    for i, item in enumerate(wheel["items"]):
        if not item:
            continue
        mid = 90.0 - 60.0 * i                                   # AppKit angles: 0 = right, growing counter-clockwise
        lit = i == wheel["pick"] and not preview
        wedge = NSBezierPath.bezierPath()
        wedge.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_clockwise_((c, c), R_WHEEL, mid + 28.5, mid - 28.5, True)
        wedge.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_clockwise_((c, c), R_HUB + 6, mid - 28.5, mid + 28.5, False)
        wedge.closePath()
        _color(YELLOW if lit else WHITE, 0.97 if lit else 0.08).set()
        wedge.fill()
        if lit:
            _stroke(_arc(c, R_WHEEL + 4, mid + 28.5, mid + 28.5 - 57 * wheel["progress"]), WHITE, 5, edge=False)
        a = math.radians(mid)
        x, y, ink = c + 98 * math.cos(a), c + 98 * math.sin(a), INK if lit else WHITE
        has_picture = _symbol(SYMBOLS.get(item, ""), x, y + 9, 21, ink)
        _text(LABELS.get(item, item), x, y - 13 if has_picture else y, 12.5, ink)
    hub = wheel["pick"] == -1 and not preview
    _color(YELLOW if hub else WHITE, 0.22 if hub else 0.10).set()
    _circle(c, R_HUB).fill()
    if wheel.get("centre", "click"):
        has_picture = _symbol(SYMBOLS["click"], c, c + 8, 17, YELLOW if hub else WHITE)
        _text(wheel.get("centre", "click"), c, c - 12 if has_picture else c, 12, YELLOW if hub else WHITE)
    if hub:
        _stroke(_arc(c, R_HUB + 4, 90, 90 - 360 * wheel["progress"]), YELLOW, 4, edge=False)
    if not preview:
        dx, dy = wheel.get("dot", (0.0, 0.0))                   # where the hand is steering
        _color(INK, 0.6).set()
        NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(c + dx * R_WHEEL - 7, c - dy * R_WHEEL - 7, 14, 14)).fill()
        _color(WHITE).set()
        NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(c + dx * R_WHEEL - 5, c - dy * R_WHEEL - 5, 10, 10)).fill()


def paint(state, pulse=None, now=None, size=SIZE):
    """Draw one state into the current AppKit graphics context. `pulse` = (kind, started_at)."""
    c = size / 2.0
    if not state.get("show", True):
        return
    now = time.time() if now is None else now
    tone = state.get("tone") or "track"
    rgb = TONES.get(tone, BLUE)
    wheel = state.get("wheel")
    if wheel:
        _wheel(wheel, c)
    else:
        _halo(c, rgb)
        _stroke(_circle(c, R_GLOW), rgb, 3.0)
        if tone == "act":                                       # a button is down, or a sign is acting: the ring fills
            _color(rgb).set()
            _circle(c, 4.5).fill()
        ring = float(state.get("ring") or 0.0)
        if ring > 0.0:
            _stroke(_arc(c, R_RING, 90, 90 - 360 * min(ring, 1.0)), YELLOW, 4.0)
    age = None if pulse is None else now - pulse[1]
    if age is not None and 0.0 <= age < PULSE_S:
        k = age / PULSE_S
        _stroke(_circle(c, R_GLOW + 36 * k), YELLOW, 3.0, alpha=1.0 - k)
    below = c - (R_WHEEL + 30 if wheel else 46)
    word = state.get("tag")
    if not word and age is not None and 0.0 <= age < WORD_S:
        word, tone = PULSE_WORDS.get(pulse[0]), "act"
    if wheel:
        word, tone = (PREVIEW_LINE if wheel.get("preview") else USE_LINE), "track"
    if word:
        if tone == "act":
            _pill(word, below, c, ink=INK, ground=YELLOW)
        else:
            _pill(word, below, c, ink=RED if tone == "warn" else WHITE)
        below -= 30
    if state.get("hint") and below > 14:
        _pill(state["hint"], below, c, ink=WHITE, size=12)


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
