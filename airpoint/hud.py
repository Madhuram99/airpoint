import cv2

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]
BANNER_H = 82
FONT = cv2.FONT_HERSHEY_SIMPLEX
ACCENT_ON, ACCENT_OFF, WHITE, MINT = (0, 200, 255), (90, 90, 90), (255, 255, 255), (167, 243, 208)


def draw_skeleton(frame, pts):
    h, w = frame.shape[:2]
    px = [(int(x * w), int(y * h)) for x, y, _ in pts]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, px[a], px[b], (250, 44, 250), 2)
    for p in px:
        cv2.circle(frame, p, 3, (121, 22, 76), -1)


def draw_cursor(frame, raw_xy, smooth_xy):
    cv2.circle(frame, raw_xy, 6, (0, 0, 255), -1)
    cv2.circle(frame, smooth_xy, 12, (0, 255, 0), 2)


def draw_region(frame, region, active):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = region
    cv2.rectangle(frame, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)),
                  ACCENT_ON if active else ACCENT_OFF, 1)


def draw_banner(frame, line1, line2, active):
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, BANNER_H), (25, 35, 30), -1)
    cv2.putText(frame, line1, (15, 32), FONT, 0.62, MINT, 2)
    cv2.putText(frame, line2, (15, 64), FONT, 0.55, ACCENT_ON if active else WHITE, 1)


def draw_prompt(frame, title, subtitle):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h // 2 - 70), (w, h // 2 + 50), (25, 35, 30), -1)
    cv2.putText(frame, title, (30, h // 2 - 10), FONT, 1.1, MINT, 2)
    cv2.putText(frame, subtitle, (30, h // 2 + 30), FONT, 0.6, WHITE, 1)
