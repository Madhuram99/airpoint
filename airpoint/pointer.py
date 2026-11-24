from .filters import OneEuroFilter

FC_MIN, FC_MIN_PRECLICK, PRECLICK_RATIO = 0.80, 0.30, 0.42
BETA = 0.030   # 1-Euro speed gain per canonical px/s: Week 1's 0.010 at a 1920-wide frame, x 1920 / 640


class PointerPipeline:
    """Raw pointer -> the pointer the cursor follows. Owns the filter chain."""

    def __init__(self, fc_min=FC_MIN):
        self.fc_min = fc_min
        self.fx, self.fy = OneEuroFilter(FC_MIN, BETA), OneEuroFilter(FC_MIN, BETA)
        self.out = None

    def update(self, raw, t, preclick=False):
        """`preclick` = fingers closing for a pinch: smooth harder so the click lands where aimed.
        Floats in, floats out: rounding here would step the cursor 3 screen points at a time."""
        fc = min(FC_MIN_PRECLICK, self.fc_min) if preclick else self.fc_min
        self.out = (self.fx.filter(raw[0], t, fc), self.fy.filter(raw[1], t, fc))
        return self.out
