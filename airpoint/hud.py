import cv2

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
PANEL = (25, 35, 30)
MINI_SIZE = (320, 240)
ZOOMS = (1.0, 1.5, 2.0)
KEY_HINT = "keys q m d k c v z r"


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
        put_fit(frame, alert, (16, 78), 0.45, (60, 60, 255))


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
