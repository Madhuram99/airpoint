"""Calibration (`c` key): tremor -> fc_min, six short pose captures, personal retrain.
The result is saved through airpoint.profile."""
import threading

import numpy as np

from .recorder import COUNTDOWN_S, RecordSession

STILL_S = 5.0
CALIB_CAPTURE_S = 5.0
JITTER_STEADY_PX, JITTER_SHAKY_PX = 3.0, 12.0
FC_STEADY, FC_SHAKY = 1.0, 0.3


def jitter_px(points):
    """Spread of the raw cursor while the user holds still: hypot of the per-axis std."""
    p = np.asarray(points, dtype=float)
    if len(p) < 2:
        return 0.0
    return float(np.hypot(p[:, 0].std(), p[:, 1].std()))


def fc_min_from_jitter(std_px):
    """<= 3 px -> 1.0 Hz (light smoothing); >= 12 px -> 0.3 Hz (heavy); linear between."""
    f = (std_px - JITTER_STEADY_PX) / (JITTER_SHAKY_PX - JITTER_STEADY_PX)
    f = min(max(f, 0.0), 1.0)
    return round(FC_STEADY + (FC_SHAKY - FC_STEADY) * f, 3)


class CalibSession:
    """Timing state machine: countdown (3 s) -> still (5 s) -> poses (RecordSession, 5 s each)
    -> training -> done | failed.

    The caller appends raw cursor points during `still`, writes landmark rows while
    `capturing` (same `fresh` / `capturing` interface as RecordSession), starts the trainer
    when the phase becomes `training`, and sets `phase` + `message` when it finishes.
    """

    def __init__(self, poses, t):
        self.phase, self.t0 = "countdown", t
        self.still_points = []
        self.fc_min = None
        self.rec = RecordSession(poses, capture_s=CALIB_CAPTURE_S)
        self.message = ""

    @property
    def pose(self):
        return self.rec.pose

    @property
    def capturing(self):
        return self.phase == "poses" and self.rec.capturing

    @property
    def fresh(self):
        return self.rec.fresh

    @fresh.setter
    def fresh(self, value):
        self.rec.fresh = value

    def update(self, t):
        """-> (phase, pose, seconds_left); phase in
        countdown | still | poses_countdown | capturing | training | done | failed."""
        if self.phase == "countdown" and t - self.t0 >= COUNTDOWN_S:
            self.phase, self.t0 = "still", t
        elif self.phase == "still" and t - self.t0 >= STILL_S:
            self.fc_min = fc_min_from_jitter(jitter_px(self.still_points))
            self.phase = "poses"
            self.rec.start(t)
        if self.phase == "poses":
            sub, pose, left = self.rec.update(t)
            if sub == "done":
                self.phase = "training"
                return self.phase, None, 0.0
            return ("capturing" if sub == "capturing" else "poses_countdown"), pose, left
        total = {"countdown": COUNTDOWN_S, "still": STILL_S}.get(self.phase)
        left = max(total - (t - self.t0), 0.0) if total else 0.0
        return self.phase, None, left


class Trainer(threading.Thread):
    """Runs `fn` off the camera loop. When `done`, exactly one of `result` / `error` is set."""

    def __init__(self, fn):
        super().__init__(daemon=True)
        self.fn, self.result, self.error = fn, None, None

    @property
    def done(self):
        return not self.is_alive() and (self.result is not None or self.error is not None)

    def run(self):
        try:
            self.result = self.fn()
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
