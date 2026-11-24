"""Tremor: a simulator for demos, a meter that measures the shake, a canceller that removes it."""
import math
from collections import deque
from dataclasses import dataclass, replace

import numpy as np
from scipy.signal import butter, sosfiltfilt

from . import features as F

SIM_HZ, SIM_AMP = 6.0, 0.20                          # amplitude in palm lengths
SIM_HZ_RANGE, SIM_AMP_RANGE = (3.5, 10.0), (0.05, 0.60)


class TremorSim:
    """Adds one whole-hand oscillation to the landmarks so a steady presenter can show what a
    tremor does. Fundamental + 20% second harmonic, slow frequency wander, amplitude modulation.
    Every landmark moves together, so wrist-centred features (and the sign read) do not change."""

    def __init__(self, hz=SIM_HZ, amp=SIM_AMP):
        self.on, self.hz, self.amp = False, hz, amp
        self.phi, self.t = 0.0, None

    def nudge(self, d_hz=0.0, d_amp=0.0):
        self.hz = round(min(max(self.hz + d_hz, SIM_HZ_RANGE[0]), SIM_HZ_RANGE[1]), 2)
        self.amp = round(min(max(self.amp + d_amp, SIM_AMP_RANGE[0]), SIM_AMP_RANGE[1]), 2)

    def offset(self, t, palm_px):
        """-> (dx, dy) pixels at time t. Phase is integrated, so changing hz mid-run stays smooth."""
        dt = 0.0 if self.t is None else min(max(t - self.t, 0.0), 0.2)
        self.t = t
        f = self.hz + 0.3 * math.sin(2 * math.pi * 0.11 * t + 0.4) + 0.15 * math.sin(2 * math.pi * 0.047 * t)
        self.phi += 2 * math.pi * f * dt
        a = self.amp * palm_px * (1.0 + 0.25 * math.sin(2 * math.pi * 0.35 * t + 1.0))
        return (a * math.sin(self.phi) + 0.2 * a * math.sin(2 * self.phi + 0.7),
                0.6 * a * math.sin(self.phi + 0.9) + 0.1 * a * math.sin(2 * self.phi))

    def apply(self, pts, t, w, h):
        """Landmarks (normalized image coords) with the shake added. `pts` itself is not modified."""
        p = F.to_array(pts)
        palm_px = math.hypot((p[F.MIDDLE_MCP, 0] - p[F.WRIST, 0]) * w, (p[F.MIDDLE_MCP, 1] - p[F.WRIST, 1]) * h)
        dx, dy = self.offset(t, palm_px)
        out = p.copy()
        out[:, 0] += dx / w
        out[:, 1] += dy / h
        return out


BAND = (3.0, 12.0)              # Hz: above intended movement, within what a 30 fps camera can show
FLOOR_PX, PROMINENCE = 1.5, 8.0  # a tremor is at least this big and this far above the band's median
WINDOW_S, EVERY_S, MIN_SPAN_S, MIN_SAMPLES, MIN_FPS = 3.0, 0.33, 1.5, 28, 14.0
AGREE = 3                        # estimates in a row needed to lock, and to let go


@dataclass(frozen=True)
class Reading:
    status: str = "wait"   # wait (too little data) | slow (camera) | steady | noisy | tremor
    hz: float = 0.0        # strongest frequency in the band for this estimate
    amp: float = 0.0       # shake inside the band: px rms over both axes
    fs: float = 0.0        # measured frame rate
    max_hz: float = 0.0    # the highest frequency this frame rate can show
    locked: bool = False   # AGREE tremor estimates in a row agreed on the frequency
    lock_hz: float = 0.0   # that frequency, smoothed; 0 when not locked
    spectrum: tuple = ()   # (freqs, power) of this estimate, for the X-ray panel


def estimate(t, xy):
    """One look at a window of pointer samples (t in seconds, xy in px). Lock fields are left unset."""
    t, xy = np.asarray(t, dtype=float), np.asarray(xy, dtype=float)
    if len(t) < MIN_SAMPLES or t[-1] - t[0] < MIN_SPAN_S:
        return Reading()
    fs = (len(t) - 1) / (t[-1] - t[0])
    hi = min(BAND[1], 0.42 * fs)
    if fs < MIN_FPS:
        return Reading(status="slow", fs=fs, max_hz=hi)
    tu = np.arange(t[0], t[-1], 1.0 / fs)                 # uniform grid: real frames arrive unevenly
    sos = butter(3, [BAND[0], hi], btype="band", fs=fs, output="sos")
    bp = np.stack([sosfiltfilt(sos, np.interp(tu, t, xy[:, k])) for k in (0, 1)], axis=1)
    edge = int(0.25 * fs)                                 # drop the filter's start-up at both ends
    bp, tu = bp[edge:-edge], tu[edge:-edge] - tu[0]
    amp = float(np.hypot(bp[:, 0].std(), bp[:, 1].std()))
    freqs = np.arange(BAND[0], hi + 1e-9, 0.1)
    e = np.exp(-2j * np.pi * freqs[:, None] * tu[None, :]) * np.hanning(len(tu))
    power = np.abs(e @ bp[:, 0]) ** 2 + np.abs(e @ bp[:, 1]) ** 2
    i = int(np.argmax(power))
    prominent = power[i] >= PROMINENCE * (float(np.median(power)) + 1e-12)
    status = "steady" if amp < FLOOR_PX else ("tremor" if prominent else "noisy")
    return Reading(status=status, hz=float(freqs[i]), amp=amp, fs=fs, max_hz=hi, spectrum=(freqs, power))


class TremorMeter:
    """Sliding window of pointer samples -> Reading. Locks after AGREE tremor estimates that agree
    within 1 Hz; lets go after AGREE estimates with none."""

    def __init__(self):
        self.t, self.xy = deque(), deque()
        self.hist = deque(maxlen=AGREE)
        self.locked, self.lock_hz = False, 0.0
        self.last, self.reading = None, Reading()

    def clear(self):
        """Forget the samples (hand briefly lost). A lock survives."""
        self.t.clear()
        self.xy.clear()

    def reset(self):
        self.clear()
        self.hist.clear()
        self.locked, self.lock_hz, self.last, self.reading = False, 0.0, None, Reading()

    def add(self, t, xy):
        self.t.append(t)
        self.xy.append(xy)
        while t - self.t[0] > WINDOW_S:
            self.t.popleft()
            self.xy.popleft()

    def read(self, t):
        """The current Reading; re-estimated at most every EVERY_S seconds."""
        if self.last is not None and t - self.last < EVERY_S:
            return self.reading
        self.last = t
        r = estimate(self.t, self.xy)
        if r.status == "slow":
            self.hist.clear()
            self.locked = False
        elif r.status != "wait":
            self.hist.append(r)
            tremors = [h.hz for h in self.hist if h.status == "tremor"]
            if len(tremors) == AGREE and max(tremors) - min(tremors) <= 1.0:
                self.locked, self.lock_hz = True, float(np.median(tremors))
            elif len(self.hist) == AGREE and not tremors:
                self.locked = False
            elif self.locked and r.status == "tremor":
                self.lock_hz = 0.6 * self.lock_hz + 0.4 * r.hz
        self.reading = replace(r, locked=self.locked, lock_hz=self.lock_hz if self.locked else 0.0)
        return self.reading


class TremorCanceller:
    """A notch at the locked tremor frequency, blended in while the meter is locked.

    A low-pass must choose between shake and lag; a notch removes only the band the tremor
    lives in. It steps on every sample so its state is warm the moment a lock arrives."""

    Q, RAMP_S, IDLE_HZ = 1.5, 0.3, 6.0

    def __init__(self):
        self.z = np.zeros((2, 2))          # filter state: [stage, axis]
        self.tuned, self.b, self.a = None, None, None
        self.g, self.t = 0.0, None         # g: how much of the notch output is used, 0..1

    def _tune(self, hz, fs):
        tuned = (round(hz, 1), round(fs))
        if tuned == self.tuned:
            return
        self.tuned = tuned
        w0 = 2 * math.pi * tuned[0] / tuned[1]
        alpha = math.sin(w0) / (2 * self.Q)
        self.b = np.array([1.0, -2 * math.cos(w0), 1.0]) / (1 + alpha)
        self.a = np.array([1.0, -2 * math.cos(w0) / (1 + alpha), (1 - alpha) / (1 + alpha)])

    def reset(self, xy):
        """Settle the filter at `xy` (after a gap), so the next output equals its input."""
        if self.b is not None:
            x = np.asarray(xy, dtype=float)
            self.z[0], self.z[1] = x * (1.0 - self.b[0]), x * (self.b[2] - self.a[2])

    def step(self, xy, t, reading):
        """-> the pointer with the tremor band removed in proportion to `g`."""
        x = np.asarray(xy, dtype=float)
        dt, self.t = (0.0 if self.t is None else t - self.t), t
        if reading.fs <= 0.0:              # the meter does not know the frame rate yet
            return float(x[0]), float(x[1])
        fresh = self.b is None
        hz = reading.lock_hz if reading.locked else (self.tuned[0] if self.tuned else self.IDLE_HZ)
        self._tune(min(hz, 0.45 * reading.fs), reading.fs)
        if fresh:
            self.reset(x)
        y = self.b[0] * x + self.z[0]
        self.z[0] = self.b[1] * x - self.a[1] * y + self.z[1]
        self.z[1] = self.b[2] * x - self.a[2] * y
        self.g += ((1.0 if reading.locked else 0.0) - self.g) * min(dt / self.RAMP_S, 1.0)
        out = x + self.g * (y - x)
        return float(out[0]), float(out[1])
