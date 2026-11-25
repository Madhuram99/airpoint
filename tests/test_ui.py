import numpy as np
import pytest

from airpoint import ui


def luminance(bgr):
    def lin(v):
        v /= 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    b, g, r = bgr
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def contrast(a, b):
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_hex_colours_become_opencv_tuples():
    assert ui.bgr("#FFD23F") == (63, 210, 255) and ui.bgr("14181F") == (31, 24, 20)


@pytest.mark.parametrize("theme", [ui.LIGHT, ui.DARK])
def test_every_text_colour_can_be_read_on_its_ground(theme):
    assert contrast(theme.text, theme.ground) >= 7.0            # body text: the stricter accessibility level
    assert contrast(theme.dim, theme.ground) >= 4.5             # secondary text
    assert contrast(theme.on_yellow, theme.yellow) >= 7.0       # the lit row and the action word
    assert contrast(theme.on_blue, theme.blue) >= 4.5           # the lit row while tracking
    assert contrast(theme.on_blue, theme.red) >= 4.5            # the word while paused
    assert contrast(theme.blue, theme.ground) >= 4.5            # "on"
    assert contrast(theme.off, theme.ground) >= 2.4             # "off" is meant to recede; the word still says it


def test_text_takes_the_same_room_in_units_at_any_scale():
    widths = [ui.Canvas(np.zeros((10, 10, 3), np.uint8), k).width("Mouse control is on", 13, 500) for k in (1.0, 2.0, 2.5)]
    assert max(widths) - min(widths) < 0.06 * widths[0] and 90 < widths[0] < 140
    c = ui.Canvas(np.zeros((40, 300, 3), np.uint8), 1.0)
    assert c.width("Moving", 26, 700) > c.width("Moving", 26, 400) > c.width("Moving", 13, 400)


def test_text_is_drawn_where_it_is_asked_for_and_the_right_way_up():
    for k in (1.0, 2.0):
        img = np.zeros((int(60 * k), int(300 * k), 3), np.uint8)
        c = ui.Canvas(img, k)
        w = c.text("Lj", 20, 40, 26, (255, 255, 255), 700)
        ys, xs = np.nonzero(img.any(axis=2))
        assert 19 <= xs.min() / k <= 24 and abs(xs.max() / k - (20 + w)) < 4
        assert 14 < ys.min() / k < 26 and 40 < ys.max() / k < 50          # L stands on the baseline, j hangs below it
        top = img[:int(40 * k)].any(axis=2).sum()
        assert top > 4 * img[int(41 * k):].any(axis=2).sum()              # most of the ink is above the baseline
    right = np.zeros((40, 200, 3), np.uint8)
    ui.Canvas(right).text("off", 190, 30, 12, (255, 255, 255), align="right")
    assert np.nonzero(right.any(axis=2))[1].max() <= 191


def test_text_that_does_not_fit_is_cut_or_wrapped_never_drawn_over_the_edge():
    c = ui.Canvas(np.zeros((10, 10, 3), np.uint8), 2.0)
    assert c.fit("short", 200, 12) == "short"
    cut = c.fit("a very long sentence that will not fit in this space at all", 120, 12)
    assert cut.endswith("…") and c.width(cut, 12) <= 120 and len(cut) > 10
    lines = c.wrap("Pinch: let go to click, keep holding to drag something far away from here", 150, 12.5, lines=2)
    assert len(lines) == 2 and all(c.width(line, 12.5) <= 150 for line in lines) and lines[1].endswith("…")
    assert c.wrap("Short.", 150, 12.5) == ["Short."] and c.wrap("", 150, 12.5) == []


def test_shapes_keep_their_place_in_units():
    for k in (1.0, 2.5):
        img = np.zeros((int(60 * k), int(100 * k), 3), np.uint8)
        c = ui.Canvas(img, k)
        c.rect(10, 20, 40, 20, (0, 0, 255), r=6)
        ys, xs = np.nonzero(img[..., 2] > 200)
        assert abs(xs.min() / k - 10) < 1 and abs(xs.max() / k - 50) < 1 and abs(ys.min() / k - 20) < 1 and abs(ys.max() / k - 40) < 1
        assert img[int(21 * k), int(10.5 * k), 2] < 200                    # the corner is rounded off
        assert c.size == (pytest.approx(100, abs=1), pytest.approx(60, abs=1))


def test_glass_lets_the_picture_show_through():
    img = np.full((60, 100, 3), 200, np.uint8)
    ui.Canvas(img).glass(10, 10, 60, 30, (0, 0, 0), 0.5, r=6)
    assert 90 <= img[25, 40, 0] <= 110 and img[5, 5, 0] == 200 and img[50, 90, 0] == 200


def test_an_emoji_is_a_small_picture_with_see_through_corners_or_nothing():
    pic = ui.emoji("✋", 40)
    if pic is None:
        pytest.skip("this system draws no colour emoji")
    assert pic.shape == (40, 40, 4) and pic[..., 3].max() == 255 and pic[0, 0, 3] == 0
    img = np.full((60, 60, 3), 255, np.uint8)
    assert ui.Canvas(img).emoji("✋", 10, 10, 40) and (img[30, 30] != 255).any() and (img[2, 2] == 255).all()
    assert ui.Canvas(img).emoji("✋", 40, 40, 40) is False               # would hang over the edge: not drawn


def test_what_the_system_says_about_the_display_never_raises():
    assert ui.backing_scale() >= 1.0 and ui.system_theme() in (ui.LIGHT, ui.DARK)
    assert ui.font() is ui.font()
