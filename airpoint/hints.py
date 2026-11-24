"""One short instruction when something the camera sees is getting in the way. Most urgent first."""
from .poses import UNKNOWN

NO_HAND_S, UNCLEAR_S, SLOW_S = 3.0, 1.5, 2.0
DARK, SMALL_PX, BIG_PX, EDGE, SLOW_FPS = 45.0, 38.0, 170.0, 0.06, 18.0   # palm sizes in the 640-wide view


class Hinter:
    """update(...) -> a hint or None. Keeps only the timers it needs: how long the hand has been
    missing, the sign unclear, the camera slow."""

    def __init__(self):
        self.since = {}

    def _lasting(self, key, on, t, need):
        if not on:
            self.since.pop(key, None)
            return False
        return t - self.since.setdefault(key, t) >= need

    def update(self, t, luma, palm_px, centre, label, fps, what="hand"):
        """`luma` = mean brightness of the view (0-255); `palm_px` = wrist to middle knuckle, None when
        no hand; `centre` = the pointer as frame fractions; `label` = the voted pose; `what` = the body
        part being tracked, for the wording."""
        hand = palm_px is not None
        missing = self._lasting("hand", not hand, t, NO_HAND_S)
        unclear = self._lasting("sign", hand and label == UNKNOWN, t, UNCLEAR_S)
        slow = self._lasting("fps", 0.0 < fps < SLOW_FPS, t, SLOW_S)
        if not hand:
            if not missing:
                return None
            return f"too dark: put more light on your {what}" if luma < DARK else f"show your {what} to the camera"
        if luma < DARK:
            return f"too dark: put more light on your {what}"
        if palm_px < SMALL_PX:
            return "move closer to the camera"
        if palm_px > BIG_PX:
            return "move back a little"
        if what == "hand" and min(centre[0], centre[1], 1.0 - centre[0], 1.0 - centre[1]) < EDGE:
            return "move your hand toward the middle"
        if unclear:
            return "sign not clear: open your palm, then try again"
        if slow:
            return "camera is slow: add light or close other apps"
        return None
