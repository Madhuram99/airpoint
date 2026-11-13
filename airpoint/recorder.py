import csv
import os

import numpy as np

from .poses import POSES

COUNTDOWN_S, CAPTURE_S = 3.0, 10.0
HEADER = ["user", "pose", "ts", "handedness"] + [f"{a}{i}" for i in range(21) for a in "xyz"]


class Recorder:
    """Appends raw MediaPipe landmark rows to one CSV per user."""

    def __init__(self, path, user):
        self.user = user
        is_new = not os.path.exists(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.f = open(path, "a", newline="")
        self.w = csv.writer(self.f)
        if is_new:
            self.w.writerow(HEADER)
        self.count = 0

    def write(self, pose, ts, handedness, pts):
        coords = [f"{v:.5f}" for v in np.asarray(pts, dtype=float).reshape(-1)]
        self.w.writerow([self.user, pose, f"{ts:.3f}", handedness] + coords)
        self.count += 1

    def close(self):
        self.f.close()


class RecordSession:
    """Timing state machine: idle -> countdown (3 s) -> capturing (10 s) -> next pose ... -> done."""

    def __init__(self, poses=POSES):
        self.poses = list(poses)
        self.i = 0
        self.phase = "idle"
        self.t0 = 0.0

    @property
    def pose(self):
        return self.poses[self.i] if self.i < len(self.poses) else None

    @property
    def capturing(self):
        return self.phase == "capturing"

    def start(self, t):
        self.phase, self.t0 = "countdown", t

    def select(self, idx, t):
        if 0 <= idx < len(self.poses):
            self.i = idx
            self.start(t)

    def redo(self, t):
        self.select(self.i if self.i < len(self.poses) else 0, t)

    def update(self, t):
        """-> (phase, pose, seconds_left)."""
        if self.phase == "countdown" and t - self.t0 >= COUNTDOWN_S:
            self.phase, self.t0 = "capturing", t
        elif self.phase == "capturing" and t - self.t0 >= CAPTURE_S:
            self.i += 1
            if self.i < len(self.poses):
                self.phase, self.t0 = "countdown", t
            else:
                self.phase = "done"
        total = {"countdown": COUNTDOWN_S, "capturing": CAPTURE_S}.get(self.phase)
        left = max(total - (t - self.t0), 0.0) if total else 0.0
        return self.phase, self.pose, left
