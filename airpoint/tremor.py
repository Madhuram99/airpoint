"""Tremor: a simulator for demos, a meter that measures the shake, a canceller that removes it."""
import math

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
