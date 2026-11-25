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
    assert glow[c, c + 14] > 200 and glow[c, c] < 120 and glow[c, c + 60] == 0       # a ring in a soft halo, see-through inside
    acting = alpha({"show": True, "tone": "act"})
    assert acting[c, c] > 200                                                        # an action: the ring fills
    ring = alpha({"show": True, "ring": 0.5})
    assert ring[c, c + 21] > 200 and ring[c, c - 21] < 90                            # half an arc: right side only
    ripple = alpha({"show": True}, pulse=("click", 10.0))
    assert ripple.sum() > glow.sum()
    assert ripple[c + 36:c + 70].any() and not glow[c + 36:c + 70].any()             # the ripple brings its word: click
    assert not alpha({"show": True}, pulse=("click", 8.0))[c + 36:c + 70].any()      # which goes again
    tagged = alpha({"show": True, "tone": "act", "tag": "Dragging", "hint": "move closer to the camera"})
    assert tagged[c + 36:c + 60].any() and tagged[c + 66:c + 90].any()               # the word, and the hint under it
    wheel = alpha({"show": True, "wheel": {"items": items, "pick": 1, "progress": 0.5, "dot": (0.3, -0.2)}})
    assert wheel[c, c + 100] > 100 and wheel[c - 100, c] > 100 and wheel[5, 5] == 0  # a disc, not the corners
    assert wheel[c + 170:c + 195].any()                                              # the line that says how to use it
    preview = alpha({"show": True, "wheel": {"items": items, "pick": -1, "progress": 0.0, "preview": True}})
    assert preview[c, c + 100] > 100 and preview[c + 170:c + 195].any()
