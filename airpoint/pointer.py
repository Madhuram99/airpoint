from .filters import OneEuroFilter

FC_MIN, FC_MIN_PRECLICK, PRECLICK_RATIO = 0.80, 0.30, 0.42


class PointerPipeline:
    """Raw pointer -> the pointer the cursor follows. Owns the filter chain."""

    def __init__(self, fc_min=FC_MIN):
        self.fc_min = fc_min
        self.fx, self.fy = OneEuroFilter(FC_MIN), OneEuroFilter(FC_MIN)

    def update(self, raw, t, preclick=False):
        """`preclick` = fingers closing for a pinch: smooth harder so the click lands where aimed."""
        fc = min(FC_MIN_PRECLICK, self.fc_min) if preclick else self.fc_min
        return int(self.fx.filter(raw[0], t, fc)), int(self.fy.filter(raw[1], t, fc))
