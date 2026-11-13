"""Synthetic 21-point hands in MediaPipe image coordinates (x right, y down, 0..1).

Wrist sits at (0.50, 0.80); the middle-finger knuckle at (0.50, 0.60), so the palm
length is exactly 0.2. Each finger's joints are interpolated from its base to the
given fingertip. z is 0 everywhere.
"""
import numpy as np

_BASES = ((1, (0.44, 0.74)), (5, (0.42, 0.62)), (9, (0.50, 0.60)),
          (13, (0.57, 0.62)), (17, (0.63, 0.66)))


def hand(thumb, index, middle, ring, pinky):
    pts = np.zeros((21, 3))
    pts[0, :2] = (0.50, 0.80)
    for (base_idx, base), tip in zip(_BASES, (thumb, index, middle, ring, pinky)):
        for k in range(4):
            f = k / 3
            pts[base_idx + k, 0] = base[0] + (tip[0] - base[0]) * f
            pts[base_idx + k, 1] = base[1] + (tip[1] - base[1]) * f
    return pts


CURLED = dict(index=(0.44, 0.68), middle=(0.50, 0.67), ring=(0.56, 0.68), pinky=(0.61, 0.70))

OPEN_PALM = hand(thumb=(0.33, 0.62), index=(0.40, 0.38), middle=(0.50, 0.34),
                 ring=(0.59, 0.38), pinky=(0.68, 0.48))
PINCH = hand(thumb=(0.42, 0.44), index=(0.43, 0.43), middle=(0.50, 0.34),
             ring=(0.59, 0.38), pinky=(0.68, 0.48))
FIST = hand(thumb=(0.45, 0.66), **CURLED)
V_SIGN = hand(thumb=(0.45, 0.66), index=(0.40, 0.38), middle=(0.50, 0.34),
              ring=(0.56, 0.68), pinky=(0.61, 0.70))
THUMB_UP = hand(thumb=(0.50, 0.42), **CURLED)
THUMB_DOWN = hand(thumb=(0.50, 1.18), **CURLED)

BY_POSE = {"palm": OPEN_PALM, "pinch": PINCH, "fist": FIST, "v_sign": V_SIGN,
           "thumb_up": THUMB_UP, "thumb_down": THUMB_DOWN}
