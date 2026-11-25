import os
import subprocess

import cv2
import numpy as np
import pytest

from airpoint import screen


def test_a_shot_is_the_screen_in_points_with_our_own_windows_blacked_out():
    retina = np.full((200, 400, 3), 200, np.uint8)                   # a 200x100-point screen at 2x
    s = screen.shot((200, 100), hide=[(120, 10, 60, 40), (190, 90, 500, 500), (-20, -20, 30, 30)],
                    capture=lambda: retina, front=lambda: "Notes")
    assert s.app == "Notes" and s.size == (200, 100)
    img = cv2.imdecode(np.frombuffer(s.jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (100, 200, 3)
    assert img[30, 150].max() < 12 and img[5, 5].max() < 12 and img[95, 195].max() < 12      # hidden, clipped to the screen
    assert img[30, 60].min() > 190 and img[80, 100].min() > 190                              # everything else is there
    same = screen.shot((400, 200), capture=lambda: retina, front=lambda: "")
    assert cv2.imdecode(np.frombuffer(same.jpeg, np.uint8), cv2.IMREAD_COLOR).shape == (200, 400, 3)


def test_capture_reads_what_screencapture_wrote_and_deletes_the_file():
    seen = {}

    def fake(cmd, **kw):
        seen["cmd"], seen["path"] = cmd, cmd[-1]
        cv2.imwrite(cmd[-1], np.full((20, 30, 3), 77, np.uint8))

    img = screen.capture(run=fake)
    assert img.shape == (20, 30, 3) and int(img[0, 0, 0]) == 77
    assert seen["cmd"][:5] == ["screencapture", "-x", "-m", "-t", "png"] and not os.path.exists(seen["path"])


def test_no_picture_is_one_sentence_and_still_leaves_no_file():
    paths = []
    with pytest.raises(OSError, match="allow Screen Recording"):
        screen.capture(run=lambda cmd, **kw: paths.append(cmd[-1]) or open(cmd[-1], "w").close())
    assert not os.path.exists(paths[0])

    def missing(cmd, **kw):
        paths.append(cmd[-1])
        raise FileNotFoundError("screencapture")

    with pytest.raises(OSError, match="no picture of the screen \\(FileNotFoundError\\)"):
        screen.capture(run=missing)

    def slow(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 10)

    with pytest.raises(OSError, match="TimeoutExpired"):
        screen.capture(run=slow)
    assert not os.path.exists(paths[1])


def test_the_system_queries_answer_without_raising():
    assert screen.allowed() in (True, False, None)
    assert isinstance(screen.front_app(), str)
    assert screen.own_windows({os.getpid()}) == []                   # a test process has no window
    assert screen.own_windows(set()) == []
