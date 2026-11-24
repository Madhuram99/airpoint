import numpy as np

from airpoint import hud


def test_mini_frame_is_320x240_with_a_label_strip():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    small = hud.mini_frame(frame, "MOVING | palm 0.98 | mouse ON")
    assert small.shape == (240, 320, 3)
    assert small[230:, :, :].any()


def test_dwell_ring_draws_only_with_progress():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    hud.draw_dwell_ring(frame, (50, 50), 0.0)
    assert not frame.any()
    hud.draw_dwell_ring(frame, (50, 50), 0.5)
    assert frame.any()


def test_canon_view_is_640_wide_and_keeps_the_aspect():
    assert hud.canon_view(np.zeros((1080, 1920, 3), np.uint8)).shape == (360, 640, 3)
    assert hud.canon_view(np.zeros((720, 1280, 3), np.uint8)).shape == (360, 640, 3)
    assert hud.canon_view(np.zeros((480, 640, 3), np.uint8)).shape == (480, 640, 3)


def test_zoomed_scales_the_shown_image():
    img = np.zeros((360, 640, 3), np.uint8)
    assert hud.zoomed(img, 1.0) is img
    assert hud.zoomed(img, 1.5).shape == (540, 960, 3)


def _width(text, scale, thickness=1):
    import cv2
    return cv2.getTextSize(text, hud.FONT, scale, thickness)[0][0]


def test_fit_scale_shrinks_only_when_needed():
    assert hud.fit_scale("short", 0.5, 600) == 0.5
    long = "x" * 200
    scale = hud.fit_scale(long, 0.5, 600)
    assert scale < 0.5 and _width(long, scale) <= 601


def test_worst_case_hud_lines_stay_legible_on_the_640_canvas():
    line1 = "SCROLLING | pose thumb_down 0.99 | ratio 1.23 | mlp/madhuramrathi | 30 fps"
    line2 = "last: drag_end, click | mouse [m] OFF | dwell [d] OFF | " + hud.KEY_HINT
    title = "HOLD thumb_down   10s"
    sub = "tilt, move closer, move away | kept 1508 | rejected 1508"
    assert hud.fit_scale(line1, 0.50, 620) >= 0.42
    assert hud.fit_scale(line2, 0.45, 620) >= 0.40
    assert hud.fit_scale(title, 0.95, 610, 2) == 0.95
    assert hud.fit_scale(sub, 0.45, 610) >= 0.40


def test_banner_and_prompt_draw_inside_the_frame():
    frame = np.zeros((360, 640, 3), np.uint8)
    hud.draw_banner(frame, "MOVING | pose palm 0.98", "last: none", True, line3=("shake 12 px", hud.WHITE))
    assert frame[:68].any() and not frame[70:].any()
    frame = np.zeros((360, 640, 3), np.uint8)
    hud.draw_prompt(frame, "HOLD palm   7s", "tilt | kept 12", "SIGN LOST")
    assert frame[:86].any() and not frame[88:].any()      # top strip: never over the hand
