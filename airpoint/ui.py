"""Drawing kit for the window: one palette, one typeface, and a canvas that takes canonical units.

Everything is laid out in canonical units (the camera view is 640 of them wide) and drawn on an image
`k` times larger, k = zoom x the display's backing scale. Text and lines are sharp at any zoom and on a
Retina display, and no layout number depends on either.

Colour carries two meanings. Blue: AirPoint is tracking. Yellow: an action is happening or a choice is
lit. Red is for warnings. A colour never stands alone: a word or a shape says the same thing.
"""
import functools
import os
from dataclasses import dataclass

import cv2
import numpy as np

SHIFT = 4                      # sub-pixel bits for anti-aliased shapes
FONT_FILES = ("/System/Library/Fonts/SFNS.ttf",)       # the macOS system face; OpenCV's built-in sans elsewhere


def bgr(hex_color):
    """'#RRGGBB' -> the (b, g, r) tuple OpenCV draws with."""
    h = hex_color.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


@dataclass(frozen=True)
class Theme:
    name: str
    ground: tuple        # panel and message line
    raised: tuple        # key caps, the track of a row
    line: tuple          # hairlines
    text: tuple
    dim: tuple           # secondary text
    off: tuple           # things that are switched off
    blue: tuple          # tracking
    on_blue: tuple       # text on blue and on red: whichever of white and ink can be read on them in this theme
    yellow: tuple        # actions, the lit choice
    on_yellow: tuple
    red: tuple           # warnings


LIGHT = Theme("light", bgr("#F4F6F8"), bgr("#FFFFFF"), bgr("#DDE1E7"), bgr("#14181F"), bgr("#566070"), bgr("#98A1AE"),
              bgr("#1F5FE0"), bgr("#FFFFFF"), bgr("#FFD23F"), bgr("#14181F"), bgr("#D0342C"))
DARK = Theme("dark", bgr("#14181F"), bgr("#232934"), bgr("#2E3541"), bgr("#F4F6F8"), bgr("#A7B0BD"), bgr("#6C7584"),
             bgr("#4D8BFF"), bgr("#14181F"), bgr("#FFD23F"), bgr("#14181F"), bgr("#FF6B63"))
THEMES = {"light": LIGHT, "dark": DARK}

# On the camera picture the ground is unknown, so marks are the same in both themes: a light core on a dark edge.
CAM_BLUE, CAM_YELLOW, CAM_WHITE, CAM_EDGE, CAM_RED = bgr("#4D8BFF"), bgr("#FFD23F"), bgr("#FFFFFF"), bgr("#10141A"), bgr("#FF5A52")


def system_theme():
    """The theme that matches the system's appearance (macOS); dark where that cannot be told."""
    try:
        from Foundation import NSUserDefaults
        dark = NSUserDefaults.standardUserDefaults().stringForKey_("AppleInterfaceStyle") == "Dark"
        return DARK if dark else LIGHT
    except Exception:
        return DARK


def backing_scale():
    """Real pixels per point on the main display: 2 on a Retina screen, 1 otherwise."""
    try:
        from AppKit import NSScreen
        return max(float(NSScreen.mainScreen().backingScaleFactor()), 1.0)
    except Exception:
        return 1.0


@functools.lru_cache(maxsize=1)
def font():
    for path in FONT_FILES:
        if os.path.exists(path):
            try:
                return cv2.FontFace(path)
            except cv2.error:
                continue
    return cv2.FontFace("sans")


@functools.lru_cache(maxsize=256)
def emoji(char, px):
    """A colour emoji as a (px, px, 4) picture, blue-green-red-alpha with the colours already multiplied
    by alpha; None where the system cannot draw one (then the caller draws its own plain mark)."""
    try:
        from AppKit import NSBitmapImageRep, NSDeviceRGBColorSpace, NSFont, NSFontAttributeName, NSGraphicsContext
        from Foundation import NSString
        rep = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
            None, px, px, 8, 4, True, False, NSDeviceRGBColorSpace, 0, 0)
        attrs = {NSFontAttributeName: NSFont.fontWithName_size_("Apple Color Emoji", px * 0.8)}
        text = NSString.stringWithString_(char)
        w, h = text.sizeWithAttributes_(attrs)
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.setCurrentContext_(NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep))
        text.drawAtPoint_withAttributes_(((px - w) / 2.0, (px - h) / 2.0), attrs)
        NSGraphicsContext.restoreGraphicsState()
        rows = np.frombuffer(rep.bitmapData(), np.uint8, count=rep.bytesPerRow() * px).reshape(px, rep.bytesPerRow())
        rgba = rows[:, :px * 4].reshape(px, px, 4)
        if not rgba[..., 3].any():
            return None
        return np.ascontiguousarray(rgba[..., [2, 1, 0, 3]])
    except Exception:
        return None


@functools.lru_cache(maxsize=4096)
def _text_px(text, size_px, weight):
    return cv2.getTextSize((4096, 4096), text, (0, 0), font(), size_px, weight)[2]


class Canvas:
    """Shapes and text in canonical units on an image `k` times larger."""

    def __init__(self, img, k=1.0):
        self.img, self.k = img, float(k)

    # ---- units
    def px(self, v):
        return int(round(v * self.k))

    def _fx(self, v):
        return int(round(v * self.k * (1 << SHIFT)))

    def _pt(self, p):
        return self._fx(p[0]), self._fx(p[1])

    def _t(self, t):
        return max(int(round(t * self.k)), 1)

    @property
    def size(self):
        """(width, height) of the image in units."""
        return self.img.shape[1] / self.k, self.img.shape[0] / self.k

    # ---- text
    def width(self, text, size, weight=400):
        return _text_px(text, max(self.px(size), 1), weight) / self.k

    def text(self, text, x, y, size, color, weight=400, align="left"):
        """Draw `text` with its baseline at y; x is its left end, middle or right end. -> its width."""
        if not text:
            return 0.0
        w = self.width(text, size, weight)
        left = x - (w / 2.0 if align == "center" else w if align == "right" else 0.0)
        cv2.putText(self.img, text, (self.px(left), self.px(y)), color, font(), max(self.px(size), 1), weight)
        return w

    def fit(self, text, max_w, size, weight=400):
        """`text`, cut with an ellipsis if it is wider than max_w."""
        if self.width(text, size, weight) <= max_w:
            return text
        lo, hi = 0, len(text)
        while lo < hi:                               # the longest prefix that fits with the ellipsis
            mid = (lo + hi + 1) // 2
            if self.width(text[:mid].rstrip() + "…", size, weight) <= max_w:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo].rstrip() + "…" if lo else ""

    def wrap(self, text, max_w, size, weight=400, lines=2):
        """`text` as at most `lines` lines no wider than max_w; the last one is cut with an ellipsis."""
        out, words = [], text.split()
        while words and len(out) < lines:
            n = 1
            while n < len(words) and self.width(" ".join(words[:n + 1]), size, weight) <= max_w:
                n += 1
            if len(out) == lines - 1 and n < len(words):
                out.append(self.fit(" ".join(words), max_w, size, weight))
                return out
            out.append(self.fit(" ".join(words[:n]), max_w, size, weight))
            words = words[n:]
        return out

    def emoji(self, char, x, y, size):
        """Draw a colour emoji in a size x size box at (x, y). -> False when the system has none."""
        px = max(self.px(size), 4)
        pic = emoji(char, px)
        x0, y0 = self.px(x), self.px(y)
        if pic is None or x0 < 0 or y0 < 0 or y0 + px > self.img.shape[0] or x0 + px > self.img.shape[1]:
            return False
        part = self.img[y0:y0 + px, x0:x0 + px].astype(np.float32)
        alpha = pic[..., 3:4].astype(np.float32) / 255.0
        part[:] = pic[..., :3].astype(np.float32) + part * (1.0 - alpha)        # the picture is already times alpha
        self.img[y0:y0 + px, x0:x0 + px] = np.clip(part, 0, 255).astype(np.uint8)
        return True

    # ---- shapes
    def _round(self, x, y, w, h, r):
        r = max(min(r, w / 2.0, h / 2.0), 0.0)
        if r <= 0.0:
            return np.array([self._pt(p) for p in ((x, y), (x + w, y), (x + w, y + h), (x, y + h))], np.int32)
        pts = []
        for cx, cy, a0 in ((x + w - r, y + r, -90.0), (x + w - r, y + h - r, 0.0), (x + r, y + h - r, 90.0), (x + r, y + r, 180.0)):
            for a in np.radians(np.linspace(a0, a0 + 90.0, 9)):
                pts.append(self._pt((cx + r * np.cos(a), cy + r * np.sin(a))))
        return np.array(pts, np.int32)

    def rect(self, x, y, w, h, color, r=0.0):
        cv2.fillPoly(self.img, [self._round(x, y, w, h, r)], color, cv2.LINE_AA, SHIFT)

    def frame(self, x, y, w, h, color, r=0.0, t=1.0):
        cv2.polylines(self.img, [self._round(x, y, w, h, r)], True, color, self._t(t), cv2.LINE_AA, SHIFT)

    def glass(self, x, y, w, h, color, alpha, r=0.0):
        """A see-through sheet over what is already drawn."""
        x0, y0 = max(self.px(x) - 1, 0), max(self.px(y) - 1, 0)
        x1, y1 = min(self.px(x + w) + 2, self.img.shape[1]), min(self.px(y + h) + 2, self.img.shape[0])
        if x1 <= x0 or y1 <= y0:
            return
        part = self.img[y0:y1, x0:x1]
        sheet = part.copy()
        local = Canvas(sheet, self.k)
        cv2.fillPoly(sheet, [local._round(x - x0 / self.k, y - y0 / self.k, w, h, r)], color, cv2.LINE_AA, SHIFT)
        cv2.addWeighted(sheet, alpha, part, 1.0 - alpha, 0.0, dst=part)

    def line(self, a, b, color, t=1.0):
        cv2.line(self.img, self._pt(a), self._pt(b), color, self._t(t), cv2.LINE_AA, SHIFT)

    def poly(self, pts, color, t=1.0, closed=False):
        if len(pts) >= 2:
            cv2.polylines(self.img, [np.array([self._pt(p) for p in pts], np.int32)], closed, color, self._t(t),
                          cv2.LINE_AA, SHIFT)

    def dot(self, c, r, color):
        cv2.circle(self.img, self._pt(c), self._fx(r), color, -1, cv2.LINE_AA, SHIFT)

    def ring(self, c, r, color, t=1.0):
        cv2.circle(self.img, self._pt(c), self._fx(r), color, self._t(t), cv2.LINE_AA, SHIFT)

    def arc(self, c, r, start, end, color, t=1.0):
        """Part of a circle; angles in degrees, 0 = right, growing clockwise, as on the screen."""
        if end > start:
            cv2.ellipse(self.img, self._pt(c), (self._fx(r), self._fx(r)), 0.0, start, end, color, self._t(t),
                        cv2.LINE_AA, SHIFT)

    def wedge(self, c, r_in, r_out, start, end, color):
        """A filled slice of a ring between two angles."""
        angles = np.radians(np.linspace(start, end, max(int(abs(end - start) / 4.0), 2)))
        outer = [(c[0] + r_out * np.cos(a), c[1] + r_out * np.sin(a)) for a in angles]
        inner = [(c[0] + r_in * np.cos(a), c[1] + r_in * np.sin(a)) for a in angles[::-1]]
        cv2.fillPoly(self.img, [np.array([self._pt(p) for p in outer + inner], np.int32)], color, cv2.LINE_AA, SHIFT)

    # ---- marks on the camera picture: a light core on a dark edge reads on any ground
    def edged_line(self, a, b, color, t=1.6):
        self.line(a, b, CAM_EDGE, t + 1.8)
        self.line(a, b, color, t)

    def edged_ring(self, c, r, color, t=1.6):
        self.ring(c, r, CAM_EDGE, t + 1.8)
        self.ring(c, r, color, t)

    def edged_arc(self, c, r, start, end, color, t=2.2):
        self.arc(c, r, start, end, CAM_EDGE, t + 1.8)
        self.arc(c, r, start, end, color, t)

    def edged_text(self, text, x, y, size, color, weight=600, align="left"):
        w = self.width(text, size, weight)
        left = x - (w / 2.0 if align == "center" else w if align == "right" else 0.0)
        d = 1.0 / self.k if self.k >= 1.5 else 1.0
        for dx, dy in ((-d, 0), (d, 0), (0, -d), (0, d), (d, d)):
            self.text(text, left + dx, y + dy, size, CAM_EDGE, weight)
        return self.text(text, left, y, size, color, weight)
