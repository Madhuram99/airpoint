import numpy as np

WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 0, 4, 8, 9, 12, 16, 20


def to_array(landmarks) -> np.ndarray:
    """21 landmark objects with .x/.y/.z, or any 21x3 iterable -> (21, 3) float64."""
    if hasattr(landmarks[0], "x"):
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
    return np.asarray(landmarks, dtype=np.float64).reshape(21, 3)


def normalize_63(landmarks) -> np.ndarray:
    """Wrist to origin, divide by 3-D palm length ||P9 - P0||. Scale- and position-free."""
    pts = to_array(landmarks)
    pts = pts - pts[WRIST]
    palm = np.linalg.norm(pts[MIDDLE_MCP]) + 1e-6
    return (pts / palm).flatten()


def pinch_ratio(landmarks) -> float:
    """2-D thumb-tip to index-tip distance over 2-D palm length (the live thresholds are 2-D)."""
    pts = to_array(landmarks)
    palm = np.linalg.norm(pts[MIDDLE_MCP, :2] - pts[WRIST, :2]) + 1e-6
    return float(np.linalg.norm(pts[THUMB_TIP, :2] - pts[INDEX_TIP, :2]) / palm)
