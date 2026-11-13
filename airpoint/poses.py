PALM, PINCH, FIST, V_SIGN, THUMB_UP, THUMB_DOWN = (
    "palm", "pinch", "fist", "v_sign", "thumb_up", "thumb_down",
)
POSES = (PALM, PINCH, FIST, V_SIGN, THUMB_UP, THUMB_DOWN)

NONE = "none"        # no hand in frame
UNKNOWN = "unknown"  # hand present, confidence below gate

GRIP = frozenset({PINCH, FIST})
SCROLL = {THUMB_UP: 1, THUMB_DOWN: -1}
