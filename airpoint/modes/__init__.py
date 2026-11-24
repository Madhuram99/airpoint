"""One object per screen the app can be in. Each has tick(session, obs, frame), on_key, close."""
from .calibrate import CalibrateMode
from .face_calibrate import FaceCalibrateMode
from .live import LiveMode
from .record import RecordMode
from .teach import TeachMode

__all__ = ["CalibrateMode", "FaceCalibrateMode", "LiveMode", "RecordMode", "TeachMode"]
