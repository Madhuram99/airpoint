"""No test may touch the real desktop: no window is opened, no app is looked at or brought forward."""
import cv2
import pytest

from airpoint import app, session, ui


class NoFocus:
    def note(self):
        return False

    def give_back(self):
        return None


@pytest.fixture(autouse=True)
def no_real_desktop(monkeypatch):
    monkeypatch.setattr(session, "Focus", NoFocus)
    for name in ("namedWindow", "imshow", "resizeWindow", "moveWindow", "setWindowProperty", "destroyAllWindows",
                 "destroyWindow"):
        monkeypatch.setattr(cv2, name, lambda *a, **k: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *a, **k: 255)
    monkeypatch.setattr(app.ui, "backing_scale", lambda: 1.0)       # sizes in tests do not depend on this display
    monkeypatch.setattr(app.ui, "system_theme", lambda: ui.DARK)
