import json
import sys
import time

import pytest

from airpoint.overlay import Overlay

ECHO = "import sys\nwith open(sys.argv[1], 'w') as f:\n    for line in sys.stdin:\n        f.write(line); f.flush()\n"


def test_client_writes_one_json_line_per_state_and_closing_ends_the_process(tmp_path):
    out = tmp_path / "seen.txt"
    ov = Overlay(cmd=[sys.executable, "-c", ECHO, str(out)])
    assert ov.alive
    for i in range(3):
        assert ov.send({"show": True, "x": 100 + i, "y": 50, "ring": i / 3})
    proc = ov.proc
    ov.close()
    assert proc.poll() is not None and not ov.alive
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert [s["x"] for s in lines] == [100, 101, 102]
    assert ov.send({"show": True}) is False                     # after close nothing is sent, nothing raises


def test_a_dead_or_missing_overlay_just_turns_itself_off():
    ov = Overlay(cmd=[sys.executable, "-c", "pass"])           # exits at once
    time.sleep(0.5)
    assert ov.send({"show": True}) is False and not ov.alive
    assert Overlay(cmd=["/no/such/program"]).send({"show": True}) is False


def test_every_state_paints_off_screen(tmp_path):
    pytest.importorskip("AppKit")
    import cv2

    from airpoint import overlay
    items = ["talk", "right", "drag", "scroll", "double", "pause"]

    def alpha(state, pulse=None):
        path = overlay.render_png(state, str(tmp_path / "s.png"), pulse=pulse, now=10.1)
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, 3]

    c = overlay.SIZE // 2
    assert not alpha({"show": False}).any()                                          # hidden: nothing at all
    glow = alpha({"show": True})
    assert glow[c, c + 14] > 100 and glow[c, c] == 0 and glow[c, c + 60] == 0        # a thin ring, hollow inside
    ring = alpha({"show": True, "ring": 0.5})
    assert ring[c, c + 21] > 100 and ring[c, c - 21] == 0                            # half an arc: right side only
    ripple = alpha({"show": True}, pulse=("click", 10.0))
    assert ripple.sum() > glow.sum()
    tagged = alpha({"show": True, "tag": "DRAG", "hint": "move closer to the camera"})
    assert tagged[c + 30:c + 90].any() and not glow[c + 30:c + 90].any()             # pills under the cursor
    wheel = alpha({"show": True, "wheel": {"items": items, "pick": 1, "progress": 0.5, "dot": (0.3, -0.2)}})
    assert wheel[c, c + 100] > 100 and wheel[c - 100, c] > 100 and wheel[5, 5] == 0  # a disc, not the corners
