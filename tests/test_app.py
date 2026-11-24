"""app.main wiring with the camera, tracker, OS mouse and window all faked."""
import numpy as np

from airpoint import app
from airpoint.actions import FakeMouse
from tests.hands import BY_POSE


class FakeCamera:
    index, size, fps = 0, (1920, 1080), 30.0

    def frames(self):
        for _ in range(8):
            yield np.zeros((1080, 1920, 3), np.uint8)

    def close(self):
        self.closed = True


class FakeTracker:
    def detect(self, frame):
        return BY_POSE["palm"], "Right"

    def close(self):
        pass


def test_main_runs_frames_through_the_session_and_quits_on_q(monkeypatch, capsys):
    shown, keys = [], iter([255, ord("x"), 255, ord("m"), 255, ord("q")])
    monkeypatch.setattr(app, "Camera", FakeCamera)
    monkeypatch.setattr(app, "HandTracker", FakeTracker)
    monkeypatch.setattr(app, "PyAutoGuiMouse", FakeMouse)
    monkeypatch.setattr(app.cv2, "imshow", lambda name, img: shown.append(img.shape))
    monkeypatch.setattr(app.cv2, "waitKey", lambda ms: next(keys))
    monkeypatch.setattr(app.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(app, "show_window", lambda mini, screen_w: None)

    app.main(["--rules"])

    assert len(shown) == 6                                  # q on the sixth frame ends the loop
    assert shown[0] == (540, 960, 3)                        # 1080p camera -> 640-wide view at zoom 1.5
    assert shown[2] == (540, 1560, 3)                       # x added the X-ray panel
    out = capsys.readouterr().out
    assert "1920x1080 @ 30 fps -> view 640x360" in out and "classifier rules" in out
