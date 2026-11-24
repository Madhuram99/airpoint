"""Synthetic pointer sessions for the tremor tests: reaches and holds, landmark noise, optional tremor.

Space is the 640-wide canonical frame. The tremor model matches airpoint.tremor.TremorSim:
fundamental + 20% second harmonic, frequency wander, 25% amplitude modulation.
"""
import math
from dataclasses import dataclass

import numpy as np

from airpoint.fsm import DwellDetector


@dataclass
class PointerSession:
    t: np.ndarray        # (n,) timestamps, jittered like a real camera
    vol: np.ndarray      # (n, 2) the path the user means to follow
    trem: np.ndarray     # (n, 2) the tremor added to it
    raw: np.ndarray      # (n, 2) vol + trem + landmark noise: what the tracker reports
    moving: np.ndarray   # (n,) True during a reach, False during a hold
    freq: np.ndarray     # (n,) instantaneous tremor frequency, Hz


def pointer_session(seconds, seed, fs=30.0, hz=None, amp=0.0, noise=1.0, start=0.0, stop=None, w=640, h=360):
    """`hz`/`amp` (peak px of the fundamental on x) add a tremor between `start` and `stop` seconds."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t = np.sort(np.arange(n) / fs + rng.normal(0, 0.003, n))
    segs, tt, p = [], 0.0, np.array([w / 2, h / 2])
    while tt < seconds + 1:
        hold = rng.uniform(0.9, 2.6)
        segs.append((tt, tt + hold, p, p, False))
        tt += hold
        ang, dist = rng.uniform(0, 2 * math.pi), rng.uniform(60, 380)
        q = np.clip(p + dist * np.array([math.cos(ang), math.sin(ang)]), [60, 50], [w - 60, h - 50])
        dur = 0.25 + 0.12 * math.log2(np.linalg.norm(q - p) / 30 + 1) + rng.uniform(0, 0.15)
        segs.append((tt, tt + dur, p, q, True))
        tt += dur
        p = q
    vol, moving = np.zeros((n, 2)), np.zeros(n, bool)
    for a, b, p0, p1, mv in segs:
        m = (t >= a) & (t < b)
        s = (t[m] - a) / (b - a)
        vol[m] = p0 + (p1 - p0) * (10 * s**3 - 15 * s**4 + 6 * s**5)[:, None]   # minimum-jerk reach
        moving[m] = mv
    trem, freq = np.zeros((n, 2)), np.zeros(n)
    if hz:
        freq = hz + 0.3 * np.sin(2 * math.pi * 0.11 * t + 0.4) + 0.15 * np.sin(2 * math.pi * 0.047 * t)
        phi = 2 * math.pi * np.concatenate([[0.0], np.cumsum(0.5 * (freq[1:] + freq[:-1]) * np.diff(t))])
        a = amp * (1 + 0.25 * np.sin(2 * math.pi * 0.35 * t + 1.0))
        a = a * ((t >= start) & (t < (seconds if stop is None else stop)))
        trem[:, 0] = a * np.sin(phi) + 0.2 * a * np.sin(2 * phi + 0.7)
        trem[:, 1] = 0.6 * a * np.sin(phi + 0.9) + 0.1 * a * np.sin(2 * phi)
    return PointerSession(t, vol, trem, vol + trem + rng.normal(0, noise, (n, 2)), moving, freq)


def _holds(s, skip=6.0):
    """(first, last+1) sample index of each hold that starts after `skip` seconds."""
    ends = np.where(s.moving[:-1] & ~s.moving[1:])[0] + 1
    for e in ends:
        nxt = np.where(s.moving[e:])[0]
        stop = e + (nxt[0] if len(nxt) else len(s.t) - e)
        if s.t[e] >= skip:
            yield e, stop


def shake_left(s, out):
    """RMS distance (px) between `out` and the intended path during holds, 0.4 s after arriving."""
    err = np.linalg.norm(np.asarray(out) - s.vol, axis=1)
    keep = np.zeros(len(s.t), bool)
    for e, stop in _holds(s):
        keep[e:stop] = s.t[e:stop] - s.t[e] > 0.4
    return float(np.sqrt(np.mean(err[keep] ** 2)))


def dwell_rate(s, out):
    """Share of holds (>= 1.6 s long) in which the real DwellDetector clicks within 1.6 s."""
    ok = []
    for e, stop in _holds(s):
        if s.t[stop - 1] - s.t[e] < 1.6:
            continue
        d, fired = DwellDetector(), False
        for k in range(e, stop):
            ev, _ = d.update((out[k][0], out[k][1]), s.t[k], True)
            if ev is not None:
                fired = s.t[k] - s.t[e] <= 1.6
                break
        ok.append(fired)
    return float(np.mean(ok))
