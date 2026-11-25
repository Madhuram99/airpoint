"""One object per screen the app can be in. Each has tick(session, obs, frame), on_key, close."""
from .ask import AskMode
from .calibrate import CalibrateMode
from .face_calibrate import FaceCalibrateMode
from .live import LiveMode
from .record import RecordMode
from .teach import TeachMode

__all__ = ["AskMode", "CalibrateMode", "FaceCalibrateMode", "LiveMode", "RecordMode", "TeachMode"]
