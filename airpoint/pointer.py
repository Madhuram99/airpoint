from collections import deque

from .filters import OneEuroFilter
from .tremor import Reading, TremorCanceller, TremorMeter

FC_MIN, FC_MIN_PRECLICK, PRECLICK_RATIO = 0.80, 0.30, 0.42
BETA = 0.030   # 1-Euro speed gain per canonical px/s: Week 1's 0.010 at a 1920-wide frame, x 1920 / 640
FC_FLOOR, FC_CEIL, GAIN_RANGE = 0.30, 1.0, (0.5, 3.0)
MODES = ("full", "smooth", "off")   # tremor canceller + smoothing | smoothing only | raw pointer
GAP_S, STALE_S, TRAIL_S = 0.25, 2.0, 4.0


class PointerPipeline:
    """Raw pointer -> the pointer the cursor follows:  canceller -> 1-Euro smoothing.

    With no tremor lock the canceller passes its input through untouched, so `full` is then
    exactly `smooth`. All filters are fed in every mode, so switching modes never jumps."""

    def __init__(self, fc_min=FC_MIN):
        self.fc_min, self.mode, self.gain = fc_min, "full", 1.0
        self.fx, self.fy = OneEuroFilter(FC_MIN, BETA), OneEuroFilter(FC_MIN, BETA)
        self.meter, self.out_meter, self.canceller = TremorMeter(), TremorMeter(), TremorCanceller()
        self.reading, self.left = Reading(), Reading()   # shake going in, shake left in the output
        self.trail = deque()                             # (t, raw, out) for the last TRAIL_S seconds
        self.t, self.out = None, None

    def set_gain(self, gain):
        """`gain` = default box width / this user's comfort-box width. In a smaller box the same hand
        shake travels further on screen and the same cursor speed is a slower hand, so the smoothing
        cutoff comes down by the gain and the 1-Euro speed term goes up by it. 1.0 changes nothing."""
        self.gain = min(max(gain, GAIN_RANGE[0]), GAIN_RANGE[1])
        self.fx.beta = self.fy.beta = BETA * self.gain

    def cutoff(self, preclick):
        fc = min(max(self.fc_min / self.gain, FC_FLOOR), FC_CEIL)
        return min(FC_MIN_PRECLICK, fc) if preclick else fc

    def cycle_mode(self):
        self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]

    def update(self, raw, t, preclick=False):
        """`preclick` = fingers closing for a pinch: smooth harder so the click lands where aimed.
        Floats in, floats out: rounding here would step the cursor 3 screen points at a time."""
        gap = None if self.t is None else t - self.t
        self.t = t
        if gap is None or gap > GAP_S:                   # hand was away: old samples no longer apply
            self.meter.reset() if gap is None or gap > STALE_S else self.meter.clear()
            self.out_meter.reset()
            self.canceller.reset(raw)
            self.trail.clear()
        self.meter.add(t, raw)
        self.reading = self.meter.read(t)
        cancelled = self.canceller.step(raw, t, self.reading)
        x = cancelled if self.mode == "full" else raw
        fc = self.cutoff(preclick)
        smooth = (self.fx.filter(x[0], t, fc), self.fy.filter(x[1], t, fc))
        self.out = raw if self.mode == "off" else smooth
        self.out_meter.add(t, self.out)
        self.left = self.out_meter.read(t)
        self.trail.append((t, raw, self.out))
        while t - self.trail[0][0] > TRAIL_S:
            self.trail.popleft()
        return self.out
