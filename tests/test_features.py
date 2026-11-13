import numpy as np
from airpoint import features as F
from tests.hands import OPEN_PALM, PINCH


def test_normalize_puts_wrist_at_origin_and_palm_at_unit_length():
    pts = F.normalize_63(OPEN_PALM).reshape(21, 3)
    assert np.allclose(pts[F.WRIST], 0.0)
    assert abs(np.linalg.norm(pts[F.MIDDLE_MCP]) - 1.0) < 1e-5


def test_normalize_is_invariant_to_scale_and_translation():
    moved = OPEN_PALM * 3.0 + np.array([7.0, -2.0, 0.5])
    assert np.allclose(F.normalize_63(moved), F.normalize_63(OPEN_PALM), atol=1e-6)


def test_normalize_accepts_landmark_objects_with_xyz_attrs():
    class LM:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z
    objs = [LM(*row) for row in OPEN_PALM]
    assert np.allclose(F.normalize_63(objs), F.normalize_63(OPEN_PALM))


def test_pinch_ratio_is_thumb_index_gap_over_palm():
    assert F.pinch_ratio(OPEN_PALM) > 1.0
    assert F.pinch_ratio(PINCH) < 0.10
