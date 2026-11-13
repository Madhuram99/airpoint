import json
from collections import Counter, deque
from pathlib import Path

import numpy as np

from . import features as F
from .poses import FIST, NONE, PALM, PINCH, THUMB_DOWN, THUMB_UP, UNKNOWN, V_SIGN


class RuleClassifier:
    """Scale-free 2-D ratio rules. Zero dependencies; the fallback and the on-stage baseline."""

    name = "rules"

    def predict(self, landmarks):
        p = F.to_array(landmarks)
        palm = np.linalg.norm(p[F.MIDDLE_MCP, :2] - p[F.WRIST, :2]) + 1e-6

        def ext(i):
            return np.linalg.norm(p[i, :2] - p[F.WRIST, :2]) / palm

        gap = np.linalg.norm(p[F.THUMB_TIP, :2] - p[F.INDEX_TIP, :2]) / palm
        idx, mid, ring, pinky, thumb = (ext(i) for i in
                                        (F.INDEX_TIP, F.MIDDLE_TIP, F.RING_TIP, F.PINKY_TIP, F.THUMB_TIP))
        if gap < 0.30 and mid > 1.35:
            return PINCH, 1.0
        if idx > 1.55 and mid > 1.55 and ring < 1.05 and pinky < 1.00:
            return V_SIGN, 1.0
        if thumb > 1.35 and idx < 0.95 and mid < 0.95 and ring < 0.95:
            return (THUMB_UP if p[F.THUMB_TIP, 1] < p[F.WRIST, 1] else THUMB_DOWN), 1.0
        if idx < 0.92 and mid < 0.92 and ring < 0.92 and pinky < 0.92:
            return FIST, 1.0
        return PALM, 1.0


class PoseClassifier:
    """sklearn model over normalize_63 features. `labels` must match model.classes_ order."""

    name = "mlp"

    def __init__(self, model, labels):
        self.model, self.labels = model, list(labels)

    @classmethod
    def load(cls, model_path, labels_path):
        import joblib
        return cls(joblib.load(model_path), json.loads(Path(labels_path).read_text()))

    def predict(self, landmarks):
        proba = self.model.predict_proba(F.normalize_63(landmarks).reshape(1, -1))[0]
        i = int(np.argmax(proba))
        return self.labels[i], float(proba[i])


class VoteSmoother:
    """Majority label over the last `window` frames; winner must hold a strict majority
    with mean confidence >= min_conf, else UNKNOWN. No hand -> NONE and history cleared."""

    def __init__(self, window=5, min_conf=0.75):
        self.buf = deque(maxlen=window)
        self.min_conf = min_conf

    def reset(self):
        self.buf.clear()

    def update(self, label, conf):
        if label is None:
            self.buf.clear()
            return NONE
        self.buf.append((label, conf))
        winner, n = Counter(l for l, _ in self.buf).most_common(1)[0]
        if n * 2 <= len(self.buf):
            return UNKNOWN
        mean_conf = sum(c for l, c in self.buf if l == winner) / n
        return winner if mean_conf >= self.min_conf else UNKNOWN
