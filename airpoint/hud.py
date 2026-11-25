"""What the window shows: the camera picture with its marks, the side panel, the message line.

All positions are canonical units (the camera view is CANON_W of them wide). Pictures are drawn `k`
times larger (zoom x the display's backing scale) through ui.Canvas, so nothing here knows the zoom.
Marks on the camera picture find k from the picture's own width.
"""
import cv2
import numpy as np

from . import ui
from .glyphs import HANDS

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]
CANON_W = 640           # every pixel threshold, the pointer and all drawing live in a view this wide
PANEL_W, BAR_H = 280, 36
MINI_W, MINI_STRIP = 320, 60
MINI_SIZE = (MINI_W, 240)
ZOOMS = (1.0, 1.25, 1.5)
XRAY_W, XRAY_ROWS = 400, 6
STATES = ("RESTING", "MOVING", "PRESSING", "DRAGGING", "HOLD", "SCROLLING")
PAD = 16
HELP = (("M", "mouse control on or off"), ("D", "dwell click: hold still to click"), ("W", "wheel: a menu at the cursor"),
        ("O", "glow at the real cursor"), ("B", "tremor filter: full, smooth, off"), ("F", "face or hand"),
        ("L", "talk: start, and stop"), ("A", "AI helper"), ("P", "pause"), ("C", "calibrate to your hand"),
        ("N", "teach a new sign"), ("R", "record signs"), ("X", "x-ray: what the system sees"),
        ("V", "small view, always on top"), ("Z", "zoom the window"), ("T", "simulated tremor, for the demo"),
        ("K", "trained network or written rules"), ("S", "sounds"), ("H", "hints"), ("Q", "quit"))
WHEEL_LABELS = {"talk": "talk", "right": "right", "drag": "drag", "scroll": "scroll", "double": "double", "pause": "pause"}
PICTURES = {"palm": "\u270b", "pinch": "\U0001f90f", "fist": "\u270a", "v_sign": "\u270c\ufe0f", "thumb_up": "\U0001f44d",
            "thumb_down": "\U0001f44e", "face:turn": "\U0001f642", "face:mouth": "\U0001f62e", "face:brows": "\U0001f928",
            "face:still": "\u23f3"}      # the system's own pictures of the signs; drawn by hand where it has none


# ----------------------------------------------------------------------------- the camera picture

def canon_size(frame):
    """(w, h) of the canonical view of this frame: CANON_W wide, aspect kept."""
    h, w = frame.shape[:2]
    return CANON_W, round(h * CANON_W / w)


def canon_view(frame):
    """The camera frame at CANON_W wide, aspect kept."""
    return stage(frame, 1.0)


def stage(frame, k=1.0):
    """The camera picture at the size it is shown: CANON_W * k wide. The tracker reads the native frame;
    the pointer and every threshold use canonical units, so they mean the same on any camera and zoom."""
    w, h = canon_size(frame)
    size = (int(round(w * k)), int(round(h * k)))
    if size == (frame.shape[1], frame.shape[0]):
        return frame
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA if size[0] < frame.shape[1] else cv2.INTER_LINEAR)


def brightness(frame):
    """Mean brightness (0-255) of the camera frame, from a thin sample of it."""
    step = max(frame.shape[1] // 80, 1)
    return float(frame[::step, ::step].mean())


def _canvas(img):
    return ui.Canvas(img, img.shape[1] / CANON_W)


def draw_skeleton(img, pts, lit=False):
    """The hand as the tracker sees it. Blue while it only moves the cursor, yellow while a sign acts."""
    c = _canvas(img)
    w, h = c.size
    p = [(float(x) * w, float(y) * h) for x, y, _ in pts]
    color = ui.CAM_YELLOW if lit else ui.CAM_BLUE
    for a, b in HAND_CONNECTIONS:
        c.line(p[a], p[b], ui.CAM_EDGE, 3.4)
    for a, b in HAND_CONNECTIONS:
        c.line(p[a], p[b], color, 1.6)
    for q in p:
        c.dot(q, 2.6, ui.CAM_EDGE)
        c.dot(q, 1.7, ui.CAM_WHITE)


def draw_face(img, face):
    """Face mode's skeleton: the face box, the nose, and the mouth opening."""
    c = _canvas(img)
    w, h = c.size
    m = {k: (x * w, y * h) for k, (x, y) in face.marks.items()}
    if len(m) < 7:
        return
    x0, x1 = min(m["side_a"][0], m["side_b"][0]), max(m["side_a"][0], m["side_b"][0])
    y0, y1 = m["forehead"][1], m["chin"][1]
    _brackets(c, x0, y0, x1, y1, ui.CAM_BLUE, 14.0)
    c.edged_line(m["lip_top"], m["lip_bottom"], ui.CAM_YELLOW, 2.4)
    c.dot(m["nose"], 4.2, ui.CAM_EDGE)
    c.dot(m["nose"], 3.0, ui.CAM_WHITE)


def draw_cursor(img, raw_xy, smooth_xy):
    """Where the pointer is: the small red dot is the raw hand, the ring is what the filters let through."""
    c = _canvas(img)
    c.dot(raw_xy, 2.8, ui.CAM_EDGE)
    c.dot(raw_xy, 1.9, ui.CAM_RED)
    c.edged_ring(smooth_xy, 7.0, ui.CAM_WHITE, 1.5)


def _brackets(c, x0, y0, x1, y1, color, arm):
    for (x, y), (dx, dy) in (((x0, y0), (1, 1)), ((x1, y0), (-1, 1)), ((x1, y1), (-1, -1)), ((x0, y1), (1, -1))):
        c.edged_line((x, y), (x + dx * arm, y), color, 1.6)
        c.edged_line((x, y), (x, y + dy * arm), color, 1.6)


def draw_region(img, region, active):
    """The comfort box: a hand moving inside it reaches the whole screen."""
    c = _canvas(img)
    w, h = c.size
    x0, y0, x1, y1 = region[0] * w, region[1] * h, region[2] * w, region[3] * h
    color = ui.CAM_BLUE if active else (200, 200, 200)
    _brackets(c, x0, y0, x1, y1, color, 16.0)
    c.edged_text("this box is your whole screen", x0 + 8, y0 + 15, 9.5, ui.CAM_WHITE if active else (215, 215, 215), 500)


def draw_dwell_ring(img, xy, progress):
    """Arc that fills clockwise from 12 o'clock as the dwell timer runs."""
    if progress > 0.0:
        _canvas(img).edged_arc(xy, 14.0, -90.0, -90.0 + 360.0 * min(progress, 1.0), ui.CAM_YELLOW, 2.4)


def draw_wheel(img, view, r_in=16.0, r_out=70.0):
    """The point-only wheel in the camera view: six directions around where the hand stopped."""
    if view["mode"] != "open":
        return
    c = _canvas(img)
    at = view["anchor"]
    c.glass(at[0] - r_out - 6, at[1] - r_out - 6, 2 * r_out + 12, 2 * r_out + 12, ui.CAM_EDGE, 0.62, r=r_out + 6)
    for i, item in enumerate(view["items"]):
        if not item:
            continue
        mid, lit = -90.0 + 60.0 * i, i == view["pick"]         # screen angles: 0 = right, growing clockwise
        if lit:
            c.wedge(at, r_in + 4, r_out, mid - 29, mid + 29, ui.CAM_YELLOW)
            c.arc(at, r_out + 3, mid - 29, mid - 29 + 58 * view["progress"], ui.CAM_WHITE, 2.4)
        a = np.radians(mid)
        c.text(WHEEL_LABELS.get(item, item), at[0] + 44 * np.cos(a), at[1] + 44 * np.sin(a) + 3.4, 9.5,
               ui.CAM_EDGE if lit else ui.CAM_WHITE, 700 if lit else 500, align="center")
    hub = view["pick"] == -1
    c.ring(at, r_in, ui.CAM_YELLOW if hub else (150, 150, 150), 1.4)
    centre = view.get("centre")
    if centre:                                                  # what staying put does, under the centre
        c.text(centre, at[0], at[1] + r_in + 11, 8.5, ui.CAM_YELLOW if hub else ui.CAM_WHITE, 600, align="center")
        if hub:
            c.arc(at, r_in + 3.5, -90.0, -90.0 + 360.0 * view["progress"], ui.CAM_YELLOW, 2.2)
    dot = view.get("dot")
    if dot:
        c.dot((at[0] + dot[0] * r_out, at[1] + dot[1] * r_out), 3.2, ui.CAM_WHITE)


def draw_prompt(img, title, subtitle, alert=None):
    """A card across the top of the camera picture for record, calibrate, teach and ask: a large title,
    what to do, and a warning when there is one. At the top so it never covers a hand held in view."""
    c = _canvas(img)
    w, _ = c.size
    lines = c.wrap(subtitle, w - 56, 12.5, 400, lines=2)
    height = 50 + 17 * len(lines) + (20 if alert else 0)
    c.glass(10, 10, w - 20, height, ui.CAM_EDGE, 0.86, r=12)
    c.text(c.fit(title, w - 56, 22, 700), 28, 42, 22, ui.CAM_WHITE, 700)
    for i, line in enumerate(lines):
        c.text(line, 28, 62 + 17 * i, 12.5, (222, 226, 232), 400)
    if alert:
        c.text(c.fit(alert, w - 56, 12.5, 600), 28, 62 + 17 * len(lines) + 3, 12.5, ui.CAM_RED, 600)


def draw_help(img):
    """Every key and what it does, over the camera picture."""
    c = _canvas(img)
    w, h = c.size
    rows = (len(HELP) + 1) // 2
    top, row = 46.0, min(22.0, (h - 76.0) / rows)
    c.glass(10, 10, w - 20, h - 20, ui.CAM_EDGE, 0.88, r=12)
    c.text("Keys", 28, 36, 16, ui.CAM_WHITE, 700)
    c.text("press ? again to close", w - 28, 36, 11.5, (190, 196, 204), 400, align="right")
    for i, (key, does) in enumerate(HELP):
        x, y = 28 + (i // rows) * ((w - 56) / 2.0), top + (i % rows) * row
        _keycap(c, x, y, key, (60, 52, 44), (110, 100, 90), ui.CAM_WHITE)
        c.text(does, x + 26, y + 13, 12, (230, 233, 238), 400)


# ----------------------------------------------------------------------------- the side panel

def _keycap(c, x, y, key, fill, edge, ink):
    c.rect(x, y, 17, 17, fill, r=4.5)
    c.frame(x, y, 17, 17, edge, r=4.5, t=1.0)
    c.text(key, x + 8.5, y + 12.6, 10.5, ink, 700, align="center")


def _tone(theme, tone):
    """-> (fill, ink) for a tone: track = blue, act = yellow, warn = red, anything else = quiet."""
    if tone == "track":
        return theme.blue, theme.on_blue
    if tone == "act":
        return theme.yellow, theme.on_yellow
    if tone == "warn":
        return theme.red, theme.on_blue
    return theme.raised, theme.dim


def draw_glyph(c, glyph, x, y, size, color):
    """A small picture of a sign in a size x size box: the system's emoji of it where there is one,
    else a drawing (a recorded hand, or a face for face mode)."""
    if glyph in PICTURES and c.emoji(PICTURES[glyph], x - size * 0.08, y - size * 0.08, size * 1.16):
        return
    hand = HANDS.get(glyph)
    if hand is not None:
        p = [(x + px * size, y + py * size) for px, py in hand]
        for a, b in HAND_CONNECTIONS:
            c.line(p[a], p[b], color, 1.15)
        return
    if not glyph or not glyph.startswith("face:"):
        return
    kind, centre, r = glyph[5:], (x + size / 2.0, y + size / 2.0), size * 0.42
    c.ring(centre, r, color, 1.15)
    eye_y = centre[1] - r * 0.22
    for dx in (-0.36, 0.36):
        c.dot((centre[0] + dx * r, eye_y), 0.9, color)
        if kind == "brows":
            c.line((centre[0] + (dx - 0.2) * r, eye_y - r * 0.42), (centre[0] + (dx + 0.2) * r, eye_y - r * 0.52), color, 1.15)
    if kind == "mouth":
        c.ring((centre[0], centre[1] + r * 0.42), r * 0.24, color, 1.15)
    else:
        c.line((centre[0] - r * 0.3, centre[1] + r * 0.42), (centre[0] + r * 0.3, centre[1] + r * 0.42), color, 1.15)
    if kind == "turn":
        for sx in (-1, 1):
            tip = centre[0] + sx * (r + 3.2)
            c.line((tip, centre[1]), (tip - sx * 2.2, centre[1] - 2.2), color, 1.15)
            c.line((tip, centre[1]), (tip - sx * 2.2, centre[1] + 2.2), color, 1.15)


def draw_panel(model, theme, k, height):
    """The side panel from its model (see panel.py): a word for what is happening, the signs, the switches."""
    img = np.full((int(round(height * k)), int(round(PANEL_W * k)), 3), theme.ground, np.uint8)
    c = ui.Canvas(img, k)
    inner = PANEL_W - 2 * PAD

    fill, ink = _tone(theme, model["tone"])                    # 1. what is happening, in one word
    word = c.fit(model["word"], inner - 24, 26, 700)
    chip_w = c.width(word, 26, 700) + 24
    c.rect(PAD, 14, chip_w, 38, fill, r=11)
    if model["tone"] not in ("track", "act", "warn"):
        c.frame(PAD, 14, chip_w, 38, theme.line, r=11)
    c.text(word, PAD + 12, 42.5, 26, ink, 700)
    for i, line in enumerate(c.wrap(model.get("detail", ""), inner, 12.5, 400, lines=2)):
        c.text(line, PAD, 70 + 16 * i, 12.5, theme.dim, 400)
    c.line((PAD, 97), (PANEL_W - PAD, 97), theme.line, 1.0)

    signs = model.get("signs", [])                             # 2. the signs and what each does
    row = 23.5 if len(signs) <= 6 else max(141.0 / len(signs), 16.0)
    for i, sign in enumerate(signs):
        y = 103 + i * row
        fill, ink = _tone(theme, sign.get("tone") if sign.get("lit") else None)
        name_ink, does_ink = (ink, ink) if sign.get("lit") else (theme.text, theme.dim)
        if sign.get("off"):
            name_ink = does_ink = theme.off
        g = min(row - 5, 18.5)
        if sign.get("lit"):
            c.rect(PAD - 7, y, inner + 14, row - 2, fill, r=7)
            c.dot((PAD + g / 2.0, y + (row - 2) / 2.0), g * 0.62, theme.raised)     # the picture keeps its own ground
        draw_glyph(c, sign.get("glyph"), PAD, y + (row - 2 - g) / 2.0, g, theme.text if sign.get("lit") else name_ink)
        base = y + (row - 2) / 2.0 + 4.4
        c.text(c.fit(sign["name"], 78, 12.5, 600), PAD + 27, base, 12.5, name_ink, 600)
        c.text(c.fit(sign["does"], inner - 108, 12, 400), PAD + 110, base, 12, does_ink, 400)
    c.line((PAD, 249), (PANEL_W - PAD, 249), theme.line, 1.0)

    switches = model.get("switches", [])                       # 3. the switches: key, name, state
    col_w, rows = (inner - 12) / 2.0, (len(switches) + 1) // 2
    step = min(18.4, (height - 268.0) / max(rows, 1))
    for i, sw in enumerate(switches):
        x, y = PAD + (i // rows) * (col_w + 12), 255 + (i % rows) * step
        _keycap(c, x, y, sw["key"], theme.raised, theme.line, theme.text)
        c.text(sw["name"], x + 23, y + 12.8, 12, theme.text, 500)
        value, tone = sw.get("value", ""), sw.get("tone")
        if value and tone == "act":
            vw = c.width(value, 10.5, 700) + 10
            c.rect(x + col_w - vw, y + 1.5, vw, 14, theme.yellow, r=7)
            c.text(value, x + col_w - vw / 2.0, y + 12.2, 10.5, theme.on_yellow, 700, align="center")
        elif value:
            c.text(value, x + col_w, y + 12.8, 11.5, theme.blue if tone == "track" else theme.off,
                   700 if tone == "track" else 500, align="right")
    return img


def draw_bar(model, theme, k, width):
    """The message line under the picture: one sentence, with a mark for what kind of sentence it is."""
    img = np.full((int(round(BAR_H * k)), int(round(width * k)), 3), theme.ground, np.uint8)
    c = ui.Canvas(img, k)
    c.line((0, 0.5), (width, 0.5), theme.line, 1.0)
    hint = "? all keys"
    c.text(hint, width - PAD, 22.5, 11.5, theme.dim, 500, align="right")
    text, tone = model.get("text", ""), model.get("tone")
    if text:
        mark = {"track": theme.blue, "act": theme.yellow, "warn": theme.red}.get(tone, theme.off)
        c.dot((PAD + 4, 18), 4.6, theme.text if tone == "act" else mark)      # yellow needs an edge on a light ground
        c.dot((PAD + 4, 18), 3.6, mark)
        c.text(c.fit(text, width - 3 * PAD - c.width(hint, 11.5, 500) - 18, 13, 500), PAD + 16, 22.6, 13, theme.text, 500)
    return img


def compose(picture, side, bar, theme):
    """Picture and side panel next to each other, the message line under both."""
    h = min(picture.shape[0], side.shape[0])
    top = np.hstack([picture[:h], side[:h]])
    cv2.line(top, (picture.shape[1], 0), (picture.shape[1], h), theme.line, max(top.shape[1] // 900, 1))
    return np.vstack([top, bar[:, :top.shape[1]]]) if bar is not None else top


def mini_frame(picture, model, theme, k):
    """The small always-on-top view: the picture, the one word, and one line under it."""
    w, h = canon_size(picture)
    ph = round(MINI_W * h / w)
    img = np.full((int(round((ph + MINI_STRIP) * k)), int(round(MINI_W * k)), 3), theme.ground, np.uint8)
    small = cv2.resize(picture, (img.shape[1], int(round(ph * k))), interpolation=cv2.INTER_AREA)
    img[:small.shape[0]] = small
    c = ui.Canvas(img, k)
    fill, ink = _tone(theme, model["tone"])
    word = c.fit(model["word"], MINI_W - 2 * PAD - 16, 15, 700)
    chip_w = c.width(word, 15, 700) + 16
    c.rect(PAD - 4, ph + 8, chip_w, 24, fill, r=7)
    if model["tone"] not in ("track", "act", "warn"):
        c.frame(PAD - 4, ph + 8, chip_w, 24, theme.line, r=7)
    c.text(word, PAD + 4, ph + 25.4, 15, ink, 700)
    c.text(c.fit(model.get("line", ""), MINI_W - 2 * PAD + 8, 11.5, 500), PAD - 4, ph + 49, 11.5, theme.dim, 500)
    return img


# ----------------------------------------------------------------------------- the X-ray panel

def _plot(c, box, curves, theme):
    """Draw each (x 0..1, y 0..1, color) curve inside box = (left, top, width, height)."""
    x0, y0, w, h = box
    c.frame(x0, y0, w, h, theme.line, t=1.0)
    for xs, ys, color in curves:
        if len(xs) >= 2:
            c.poly(list(zip(x0 + np.asarray(xs) * w, y0 + h - np.clip(ys, 0.0, 1.0) * h)), color, 1.2)


def draw_xray(height, x, k=1.0, theme=ui.DARK):
    """The X-ray panel: what the system saw and decided this frame. `x` is Session.taps()."""
    img = np.full((int(round(height * k)), int(round(XRAY_W * k)), 3), theme.ground, np.uint8)
    c = ui.Canvas(img, k)
    raw_c, out_c = theme.red, theme.blue
    c.text(f"X-ray   {x['clf']}   {x['fps']:.0f} fps   tracker {x['ms_tracker']:.0f} ms   "
           f"classifier {x['ms_classifier']:.1f} ms", 10, 16, 11, theme.text, 600)

    probs, labels = x["probs"], list(x["labels"])              # 1. class probabilities
    best = -1 if probs is None else int(np.argmax(probs))
    shown = list(range(len(labels)))
    if len(shown) > XRAY_ROWS:                                 # taught signs add classes: show the likeliest
        shown = shown[:XRAY_ROWS] if probs is None else sorted(np.argsort(-np.asarray(probs))[:XRAY_ROWS].tolist())
    for row, i in enumerate(shown):
        y, p = 23 + 16 * row, 0.0 if probs is None else float(probs[i])
        c.text(labels[i][:12], 10, y + 11.5, 11, theme.text if i == best else theme.dim, 600 if i == best else 400)
        c.rect(104, y + 3, 220, 9, theme.raised, r=2)
        if p > 0.004:
            c.rect(104, y + 3, 220 * p, 9, theme.yellow if i == best else theme.off, r=2)
        c.text(f"{p:.2f}", 334, y + 11.5, 11, theme.text, 400)
    gate_x = 104 + 220 * x["gate"]
    c.line((gate_x, 23), (gate_x, 23 + 16 * len(shown)), theme.text, 1.0)

    votes = " ".join(v.replace("thumb_", "")[:6] for v in x["votes"])   # 2. the vote buffer
    c.text(c.fit(f"votes: {votes}  =>  {x['label']}", XRAY_W - 20, 11, 400), 10, 133, 11, theme.text, 400)

    for i, name in enumerate(STATES):                          # 3. the state machine
        bx, by, on = 10 + (i % 3) * 128, 141 + (i // 3) * 21, name == x["state"]
        if on:
            c.rect(bx, by, 122, 17, theme.yellow, r=5)
        else:
            c.frame(bx, by, 122, 17, theme.line, r=5)
        c.text(name, bx + 8, by + 12.4, 10.5, theme.on_yellow if on else theme.dim, 700 if on else 500)
    c.text(c.fit(f"in state {x['state_s']:.1f} s   last: {x['last_event']}", XRAY_W - 20, 11, 400), 10, 195, 11,
           theme.text, 400)

    c.text("pointer x, last 4 s: raw red, output blue", 10, 211, 10.5, theme.dim, 400)      # 4. scope
    trail = list(x["trail"])
    curves = []
    if len(trail) >= 2:
        raw, out = np.array([r[0] for _, r, _ in trail]), np.array([o[0] for _, _, o in trail])
        mid = float(np.median(out))
        span = max(25.0, 1.2 * float(np.abs(raw - mid).max()))
        ts = np.linspace(0.0, 1.0, len(trail))
        curves = [(ts, (raw - mid) / (2 * span) + 0.5, raw_c), (ts, (out - mid) / (2 * span) + 0.5, out_c)]
    _plot(c, (10, 216, XRAY_W - 20, 40), curves, theme)

    r, left = x["reading"], x["left"]                          # 5. spectrum: raw against output
    c.text("shake by frequency: raw red, output blue", 10, 271, 10.5, theme.dim, 400)
    box, curves = (10, 276, XRAY_W - 20, 38), []
    if r.spectrum:
        top = float(np.max(r.spectrum[1])) + 1e-12
        for spec, color in ((r.spectrum, raw_c), (left.spectrum, out_c)):
            if spec:
                curves.append(((spec[0] - 3.0) / 9.0, np.sqrt(spec[1] / top), color))
    _plot(c, box, curves, theme)
    for hz in (3, 6, 9, 12):
        c.text(str(hz), box[0] + (hz - 3) / 9 * box[2], 325, 9.5, theme.dim, 400,
               align="right" if hz == 12 else "left" if hz == 3 else "center")
    if r.locked:
        lx = box[0] + (r.lock_hz - 3.0) / 9.0 * box[2]
        c.line((lx, box[1]), (lx, box[1] + box[3]), theme.yellow, 1.4)
        c.text(f"{r.lock_hz:.1f} Hz", min(lx + 5, XRAY_W - 56), box[1] + 13, 10.5, theme.text, 700)

    foot = (f"shake {r.amp:.1f} px -> {left.amp:.1f} px   filter {x['mode']}   "      # 6. numbers
            f"sees up to {r.max_hz:.0f} Hz" + ("   simulated" if x["sim"] else ""))
    c.text(c.fit(foot, XRAY_W - 20, 11, 500), 10, 343, 11, theme.red if x["sim"] else theme.text, 500)
    return img
