from airpoint.hints import Hinter

OK = dict(luma=120.0, palm_px=70.0, centre=(0.5, 0.5), label="palm", fps=30.0)


def hint(h, t, **over):
    return h.update(t, **dict(OK, **over))


def test_nothing_to_say_when_all_is_well():
    assert hint(Hinter(), 0.0) is None


def test_missing_hand_is_mentioned_only_after_three_seconds():
    h = Hinter()
    assert hint(h, 0.0, palm_px=None) is None
    assert hint(h, 2.9, palm_px=None) is None
    assert hint(h, 3.0, palm_px=None) == "show your hand to the camera"
    assert hint(h, 3.1) is None                                   # hand back: timer cleared
    assert hint(h, 3.2, palm_px=None) is None


def test_darkness_explains_a_missing_hand_and_outranks_everything_else():
    h = Hinter()
    hint(h, 0.0, palm_px=None, luma=20.0)
    assert hint(h, 3.0, palm_px=None, luma=20.0) == "too dark: put more light on your hand"
    assert hint(Hinter(), 0.0, luma=20.0, palm_px=20.0) == "too dark: put more light on your hand"


def test_distance_and_position():
    h = Hinter()
    assert hint(h, 0.0, palm_px=30.0) == "move closer to the camera"
    assert hint(h, 0.0, palm_px=200.0) == "move back a little"
    assert hint(h, 0.0, centre=(0.03, 0.5)) == "move your hand toward the middle"
    assert hint(h, 0.0, centre=(0.5, 0.97)) == "move your hand toward the middle"


def test_unclear_sign_and_slow_camera_need_to_last():
    h = Hinter()
    assert hint(h, 0.0, label="unknown") is None
    assert hint(h, 1.5, label="unknown") == "sign not clear: open your palm, then try again"
    assert hint(h, 1.6) is None
    assert hint(h, 2.0, fps=12.0) is None
    assert hint(h, 4.0, fps=12.0) == "camera is slow: add light or close other apps"
    assert hint(h, 4.1, fps=0.0) is None                          # first frame has no rate yet
