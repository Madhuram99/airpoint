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
    overlays = []

    class NoOverlay:
        alive = True

        def __init__(self):
            overlays.append(self)
            self.sent, self.closed = 0, False

        def send(self, state):
            self.sent += 1
            return True

        def close(self):
            self.closed = True

    monkeypatch.setattr(app, "Overlay", NoOverlay)          # the real one would draw on the screen

    app.main(["--rules"])

    assert len(shown) == 6                                  # q on the sixth frame ends the loop
    assert shown[0] == (540, 960, 3)                        # 1080p camera -> 640-wide view at zoom 1.5
    assert shown[2] == (540, 1560, 3)                       # x added the X-ray panel
    assert len(overlays) == 1 and overlays[0].sent >= 6 and overlays[0].closed
    out = capsys.readouterr().out
    assert "1920x1080 @ 30 fps -> view 640x360" in out and "classifier rules" in out


def test_no_overlay_flag_starts_without_one(monkeypatch):
    made = []
    monkeypatch.setattr(app, "Camera", FakeCamera)
    monkeypatch.setattr(app, "HandTracker", FakeTracker)
    monkeypatch.setattr(app, "PyAutoGuiMouse", FakeMouse)
    monkeypatch.setattr(app.cv2, "imshow", lambda name, img: None)
    monkeypatch.setattr(app.cv2, "waitKey", lambda ms: ord("q"))
    monkeypatch.setattr(app.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(app, "Overlay", lambda: made.append(1))
    app.main(["--rules", "--no-overlay"])
    assert made == []


def _patch_loop(monkeypatch, keys):
    it = iter(keys)
    monkeypatch.setattr(app, "Camera", FakeCamera)
    monkeypatch.setattr(app, "HandTracker", FakeTracker)
    monkeypatch.setattr(app, "PyAutoGuiMouse", FakeMouse)
    monkeypatch.setattr(app.cv2, "imshow", lambda name, img: None)
    monkeypatch.setattr(app.cv2, "waitKey", lambda ms: next(it, ord("q")))
    monkeypatch.setattr(app.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(app, "show_window", lambda mini, screen_w: None)


def test_f_starts_the_face_tracker_and_feeds_its_observations(monkeypatch):
    from airpoint.face import FaceObs
    events = []

    class FakeFace:
        def __init__(self):
            events.append("made")

        def detect(self, frame):
            events.append("detect")
            return FaceObs(0.0, 0.0, {"jawOpen": 0.0})

        def close(self):
            events.append("closed")

    _patch_loop(monkeypatch, [ord("f"), 255, 255, ord("q")])
    monkeypatch.setattr(app, "FaceTracker", FakeFace)
    app.main(["--rules", "--no-overlay"])
    assert events[0] == "made" and events.count("detect") == 3 and events[-1] == "closed"


def test_a_missing_face_model_falls_back_to_the_hand_with_the_download_line(monkeypatch, capsys):
    def no_model():
        raise FileNotFoundError("face_landmarker.task missing. Download with:\n  curl -L -o face_landmarker.task ...")

    _patch_loop(monkeypatch, [ord("f"), 255, ord("q")])
    monkeypatch.setattr(app, "FaceTracker", no_model)
    app.main(["--rules", "--no-overlay"])
    out = capsys.readouterr().out
    assert "face mode needs its model" in out and "curl -L -o face_landmarker.task" in out


def test_keys_airpoint_types_itself_never_come_back_to_it_as_commands(monkeypatch):
    """With AirPoint's own window in front, text it types arrives at its own key reader. A dictated
    "q" must not quit and an "m" must not flip the mouse switch."""
    sessions, real = [], app.Session

    class TypingMouse(FakeMouse):
        frames = 0

        @property
        def typing(self):                       # AirPoint is typing during the second and third frames
            return TypingMouse.frames in (2, 3)

    class Camera(FakeCamera):
        def frames(self):
            for _ in range(20):
                TypingMouse.frames += 1
                yield np.zeros((1080, 1920, 3), np.uint8)

    #         frame: 1    2         3         4         5    6    7    8    9    10        11
    _patch_loop(monkeypatch, [255, ord("q"), ord("m"), ord("m"), 255, 255, 255, 255, 255, ord("d"), ord("q")])
    monkeypatch.setattr(app, "Camera", Camera)
    monkeypatch.setattr(app, "PyAutoGuiMouse", TypingMouse)
    monkeypatch.setattr(app, "Session", lambda *a, **k: sessions.append(real(*a, **k)) or sessions[-1])
    app.main(["--rules", "--no-overlay"])
    s = sessions[0]
    assert TypingMouse.frames == 11             # the q on frame 2 was AirPoint's own; the one on frame 11 quit
    assert not s.runner.enabled                 # neither m got through: the second was still the tail of the typing
    assert s.dwell_on                           # five quiet frames later, keys count again


def test_helper_flag_switches_it_on_only_when_it_can_work(monkeypatch, capsys):
    sessions, real = [], app.Session
    _patch_loop(monkeypatch, [255])
    monkeypatch.setattr(app, "Session", lambda *a, **k: sessions.append(real(*a, **k)) or sessions[-1])
    monkeypatch.setenv("OPENAI_API_KEY", "")
    app.main(["--rules", "--no-overlay", "--helper"])
    assert not sessions[0].helper.armed and "AI helper: no API key" in capsys.readouterr().out
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test0001")
    monkeypatch.setattr("airpoint.helper.screen.allowed", lambda: True)
    app.main(["--rules", "--no-overlay", "--helper"])
    out = capsys.readouterr().out
    assert sessions[1].helper.armed and "AI helper is ON: it sends your request and a picture of your screen" in out
    assert "sk-test" not in out
    app.main(["--rules", "--no-overlay"])
    assert not sessions[2].helper.armed
