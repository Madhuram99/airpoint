import cv2
import numpy as np

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]
CANON_W = 640           # every pixel threshold, the pointer and all drawing live in a frame this wide
BANNER_LINE_H = 20
FONT = cv2.FONT_HERSHEY_SIMPLEX
ACCENT_ON, ACCENT_OFF, WHITE, MINT = (0, 200, 255), (90, 90, 90), (255, 255, 255), (167, 243, 208)
PANEL, ALERT = (25, 35, 30), (60, 60, 255)
MINI_SIZE = (320, 240)
ZOOMS = (1.0, 1.5, 2.0)
XRAY_W, XRAY_ROWS = 400, 6
STATES = ("RESTING", "MOVING", "PRESSING", "DRAGGING", "HOLD", "SCROLLING")
RAW_C, OUT_C, DIM, GRID_C = (80, 80, 255), (120, 230, 120), (150, 150, 150), (60, 70, 65)
KEY_HINT = "keys q p n f k c v z x t r"


def canon_view(frame):
    """The camera frame at CANON_W wide, aspect kept. The tracker reads the native frame; everything
    else (pointer pixels, thresholds, filters, drawing) uses this one, so it means the same on any camera."""
    h, w = frame.shape[:2]
    if w == CANON_W:
        return frame
    return cv2.resize(frame, (CANON_W, round(h * CANON_W / w)), interpolation=cv2.INTER_AREA)


def zoomed(img, zoom):
    if zoom == 1.0:
        return img
    return cv2.resize(img, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)


def fit_scale(text, scale, max_w, thickness=1):
    """The largest font scale <= `scale` at which `text` is no wider than `max_w` pixels."""
    w = cv2.getTextSize(text, FONT, scale, thickness)[0][0]
    return scale if w <= max_w else scale * max_w / w


def put_fit(img, text, org, scale, color, thickness=1):
    """putText that shrinks rather than runs off the right edge."""
    scale = fit_scale(text, scale, img.shape[1] - org[0] - 10, thickness)
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_skeleton(frame, pts):
    h, w = frame.shape[:2]
    px = [(int(x * w), int(y * h)) for x, y, _ in pts]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, px[a], px[b], (250, 44, 250), 2)
    for p in px:
        cv2.circle(frame, p, 3, (121, 22, 76), -1)


def draw_face(frame, face):
    """Face mode's skeleton: the face box, the nose, and the mouth opening."""
    h, w = frame.shape[:2]
    m = {k: (int(x * w), int(y * h)) for k, (x, y) in face.marks.items()}
    if len(m) < 7:
        return
    cv2.rectangle(frame, (min(m["side_a"][0], m["side_b"][0]), m["forehead"][1]),
                  (max(m["side_a"][0], m["side_b"][0]), m["chin"][1]), (250, 44, 250), 1)
    cv2.line(frame, m["lip_top"], m["lip_bottom"], (250, 44, 250), 3)
    cv2.circle(frame, m["nose"], 4, (121, 22, 76), -1)


def _px(xy):
    return int(round(xy[0])), int(round(xy[1]))


def draw_cursor(frame, raw_xy, smooth_xy):
    cv2.circle(frame, _px(raw_xy), 3, (0, 0, 255), -1)
    cv2.circle(frame, _px(smooth_xy), 7, (0, 255, 0), 1)


def draw_region(frame, region, active):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = region
    cv2.rectangle(frame, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)),
                  ACCENT_ON if active else ACCENT_OFF, 1)


def draw_banner(frame, line1, line2, active, line3=None):
    """Top strip: status line, switches line, optional third line (shake readout)."""
    w = frame.shape[1]
    n = 3 if line3 else 2
    cv2.rectangle(frame, (0, 0), (w, 8 + BANNER_LINE_H * n), PANEL, -1)
    put_fit(frame, line1, (10, 19), 0.50, MINT)
    put_fit(frame, line2, (10, 19 + BANNER_LINE_H), 0.45, ACCENT_ON if active else WHITE)
    if line3:
        put_fit(frame, line3[0], (10, 19 + 2 * BANNER_LINE_H), 0.45, line3[1])


def draw_prompt(frame, title, subtitle, alert=None):
    """Top strip for record / calibrate: big title, small instructions, optional red alert.
    At the top so it never covers the hand the user is trying to hold in view."""
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, 86), PANEL, -1)
    put_fit(frame, title, (16, 34), 0.95, MINT, 2)
    put_fit(frame, subtitle, (16, 58), 0.45, WHITE)
    if alert:
        put_fit(frame, alert, (16, 78), 0.45, ALERT)


WHEEL_LABELS = {"talk": "talk", "right": "right", "drag": "drag", "scroll": "scroll", "double": "2x", "pause": "pause"}


def draw_wheel(frame, view, r_in=16, r_out=70):
    """The point-only wheel in the camera view: six directions around where the hand stopped."""
    if view["mode"] != "open":
        return
    c = _px(view["anchor"])
    cv2.circle(frame, c, r_out, PANEL, 2)
    cv2.circle(frame, c, r_in, ACCENT_ON if view["pick"] == -1 else ACCENT_OFF, 1)
    for i, item in enumerate(view["items"]):
        mid = -90.0 + 60.0 * i                               # image angles: 0 = right, growing clockwise
        a = np.radians(mid)
        lit = i == view["pick"]
        if lit:
            cv2.ellipse(frame, c, (r_out, r_out), 0, mid - 30, mid - 30 + 60 * view["progress"], ACCENT_ON, 4)
        label = WHEEL_LABELS.get(item, item)
        size = cv2.getTextSize(label, FONT, 0.38, 1)[0]
        org = (int(c[0] + 44 * np.cos(a) - size[0] / 2), int(c[1] + 44 * np.sin(a) + size[1] / 2))
        cv2.putText(frame, label, org, FONT, 0.38, ACCENT_ON if lit else WHITE, 1, cv2.LINE_AA)
    size = cv2.getTextSize("click", FONT, 0.30, 1)[0]
    cv2.putText(frame, "click", (c[0] - size[0] // 2, c[1] + r_in + 6 + size[1]), FONT, 0.30,    # under the centre:
                ACCENT_ON if view["pick"] == -1 else WHITE, 1, cv2.LINE_AA)                       # the cursor dot sits in it
    if view["pick"] == -1:
        cv2.ellipse(frame, c, (r_in + 4, r_in + 4), -90, 0, 360 * view["progress"], ACCENT_ON, 2)


def draw_hint(frame, text):
    """Bottom strip: one plain instruction when something is in the way."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 26), (w, h), PANEL, -1)
    put_fit(frame, text, (10, h - 8), 0.50, ACCENT_ON)


def draw_dwell_ring(frame, xy, progress):
    """Arc that fills clockwise from 12 o'clock as the dwell timer runs."""
    if progress <= 0.0:
        return
    cv2.ellipse(frame, _px(xy), (14, 14), -90, 0, int(360 * min(progress, 1.0)), ACCENT_ON, 2)


def mini_frame(frame, label):
    """320x240 copy with a one-line strip at the bottom; for the always-on-top corner view."""
    small = cv2.resize(frame, MINI_SIZE, interpolation=cv2.INTER_AREA)
    w, h = MINI_SIZE
    cv2.rectangle(small, (0, h - 24), (w, h), PANEL, -1)
    put_fit(small, label, (8, h - 8), 0.45, MINT)
    return small


def _plot(panel, box, curves):
    """Draw each (x 0..1, y 0..1, color) curve inside box = (left, top, width, height)."""
    x0, y0, w, h = box
    cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), GRID_C, 1)
    for xs, ys, color in curves:
        if len(xs) < 2:
            continue
        pts = np.stack([x0 + np.asarray(xs) * w, y0 + h - np.clip(ys, 0.0, 1.0) * h], axis=1)
        cv2.polylines(panel, [pts.astype(np.int32)], False, color, 1, cv2.LINE_AA)


def draw_xray(height, x):
    """The X-ray panel: what the system saw and decided this frame. `x` is Session.taps()."""
    panel = np.full((height, XRAY_W, 3), PANEL, np.uint8)
    put_fit(panel, f"X-RAY | {x['clf']} | {x['fps']:.0f} fps | tracker {x['ms_tracker']:.0f} ms | "
                   f"classifier {x['ms_classifier']:.1f} ms", (8, 15), 0.40, WHITE)

    probs, labels = x["probs"], list(x["labels"])              # 1. class probabilities
    best = -1 if probs is None else int(np.argmax(probs))
    shown = list(range(len(labels)))
    if len(shown) > XRAY_ROWS:                                 # taught signs add classes: show the likeliest
        shown = shown[:XRAY_ROWS] if probs is None else sorted(np.argsort(-np.asarray(probs))[:XRAY_ROWS].tolist())
    for row, i in enumerate(shown):
        y, p = 24 + 16 * row, 0.0 if probs is None else float(probs[i])
        put_fit(panel, labels[i][:11], (8, y + 12), 0.40, MINT if i == best else DIM)
        cv2.rectangle(panel, (100, y + 3), (100 + int(220 * p), y + 13), ACCENT_ON if i == best else ACCENT_OFF, -1)
        put_fit(panel, f"{p:.2f}", (334, y + 12), 0.40, WHITE)
    gate_x, y = 100 + int(220 * x["gate"]), 24 + 16 * len(shown)
    cv2.line(panel, (gate_x, 24), (gate_x, y), WHITE, 1)

    votes = " ".join(v.replace("thumb_", "")[:6] for v in x["votes"])   # 2. the vote buffer
    put_fit(panel, f"votes: {votes}  =>  {x['label']}", (8, y + 14), 0.40, WHITE)

    y += 22                                                    # 3. the state machine
    for i, name in enumerate(STATES):
        bx, by, on = 8 + (i % 3) * 130, y + (i // 3) * 22, name == x["state"]
        cv2.rectangle(panel, (bx, by), (bx + 124, by + 18), ACCENT_ON if on else ACCENT_OFF, -1 if on else 1)
        put_fit(panel, name, (bx + 6, by + 13), 0.38, (0, 0, 0) if on else DIM)
    put_fit(panel, f"in state {x['state_s']:.1f} s | last: {x['last_event']}", (8, y + 54), 0.40, WHITE)

    y += 70                                                    # 4. scope: pointer x, raw vs output
    put_fit(panel, "pointer x, last 4 s: raw red, output green", (8, y), 0.38, DIM)
    trail = list(x["trail"])
    curves = []
    if len(trail) >= 2:
        raw, out = np.array([r[0] for _, r, _ in trail]), np.array([o[0] for _, _, o in trail])
        mid = float(np.median(out))
        span = max(25.0, 1.2 * float(np.abs(raw - mid).max()))
        ts = np.linspace(0.0, 1.0, len(trail))
        curves = [(ts, (raw - mid) / (2 * span) + 0.5, RAW_C), (ts, (out - mid) / (2 * span) + 0.5, OUT_C)]
    _plot(panel, (8, y + 4, XRAY_W - 16, 48), curves)

    y += 66                                                    # 5. spectrum: raw vs output
    r, left = x["reading"], x["left"]
    put_fit(panel, "shake by frequency: raw red, output green", (8, y), 0.38, DIM)
    box, curves = (8, y + 4, XRAY_W - 16, 44), []
    if r.spectrum:
        top = float(np.max(r.spectrum[1])) + 1e-12
        for spec, color in ((r.spectrum, RAW_C), (left.spectrum, OUT_C)):
            if spec:
                curves.append(((spec[0] - 3.0) / 9.0, np.sqrt(spec[1] / top), color))
    _plot(panel, box, curves)
    for hz in (3, 6, 9, 12):
        put_fit(panel, str(hz), (box[0] + int((hz - 3) / 9 * box[2]) - (12 if hz == 12 else 0), y + 60), 0.34, DIM)
    if r.locked:
        lx = box[0] + int((r.lock_hz - 3.0) / 9.0 * box[2])
        cv2.line(panel, (lx, box[1]), (lx, box[1] + box[3]), ACCENT_ON, 1)
        put_fit(panel, f"{r.lock_hz:.1f} Hz", (min(lx + 4, XRAY_W - 60), box[1] + 12), 0.38, ACCENT_ON)

    foot = (f"shake {r.amp:.1f} px -> {left.amp:.1f} px | filter {x['mode']} | "       # 6. numbers
            f"sees up to {r.max_hz:.0f} Hz" + (" | SIMULATED" if x["sim"] else ""))
    put_fit(panel, foot, (8, y + 77), 0.40, ALERT if x["sim"] else WHITE)
    return panel
