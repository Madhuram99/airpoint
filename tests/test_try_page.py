"""The browser demo's static files: what must be there, and what must agree with the Python side."""
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def text(*parts):
    with open(os.path.join(ROOT, *parts)) as f:
        return f.read()


def test_the_page_and_everything_it_loads_are_in_the_repo():
    page = text("try", "index.html")
    for name in re.findall(r'(?:src|href)="([^"?#:]+)"', page):
        assert os.path.exists(os.path.normpath(os.path.join(ROOT, "try", name))), name
    app = text("try", "app.js")
    for name in re.findall(r'from "\./([^"]+)"|import\("\./([^"]+)"\)|fetch\("([^"]+)"\)', app):
        assert os.path.exists(os.path.join(ROOT, "try", next(n for n in name if n))), name
    assert os.path.exists(os.path.join(ROOT, "hand_landmarker.task")) and 'HAND_MODEL = "../hand_landmarker.task"' in app
    assert os.path.exists(os.path.join(ROOT, ".nojekyll"))               # Pages must serve the files as they are
    assert page.index("window.AirPointStatic") < page.index("<iframe")   # the frame's pages look for it as they load


def test_the_browser_tracker_is_the_same_version_as_the_python_one():
    mediapipe = pytest.importorskip("mediapipe")
    pinned = re.search(r"@mediapipe/tasks-vision@([0-9.]+)", text("try", "app.js")).group(1)
    assert pinned == mediapipe.__version__
    assert text("try", "app.js").count("0.7") >= 3                       # the three confidence floors, as in tracker.py
    assert "[1 - p.x, p.y, p.z]" in text("try", "app.js")                # mirrored, as camera.py flips its frames


def test_the_lab_pages_ask_their_parent_when_embedded_and_say_nothing_is_saved():
    lab = text("lab", "lab.js")
    assert "window.parent.AirPointStatic" in lab and "if (this.embed) return this.embed.api(path, body);" in lab
    assert 'classList.add("embedded")' in lab and '.embedded a[href="index.html"]' in text("lab", "lab.css")
    assert "out.note ||" in text("lab", "tutorial.js") and "Not saved anywhere" in text("try", "app.js")
    assert "never leaves this device" in text("try", "index.html")


@pytest.mark.skipif(sys.platform != "darwin", reason="Core Image draws the code")
def test_a_qr_code_reads_back_as_what_went_in(tmp_path):
    pytest.importorskip("Quartz")
    from airpoint import qr
    side, back = qr.make("https://example.org/try/?a=1", str(tmp_path / "q.png"))
    assert back == "https://example.org/try/?a=1" and side > 300
    assert qr.read(os.path.join(ROOT, "try", "qr.png")) == "https://madhuram99.github.io/airpoint/try/"
    with pytest.raises(SystemExit, match="usage"):
        qr.main([])
