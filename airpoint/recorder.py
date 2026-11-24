import csv
import os

import numpy as np

from .features import INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, RING_TIP, THUMB_TIP, WRIST, to_array
from .poses import FIST, PALM, PINCH, POSES, THUMB_DOWN, THUMB_UP, V_SIGN

COUNTDOWN_S, CAPTURE_S = 3.0, 10.0
HEADER = ["user", "pose", "ts", "handedness"] + [f"{a}{i}" for i in range(21) for a in "xyz"]


def looks_like(pose, pts):
    """Loose sanity check: could this frame be `pose` at all? Catches gross drift during a
    capture (hand closed mid-palm, tips apart mid-pinch). Deliberately not a classifier."""
    p = to_array(pts)
    palm = np.linalg.norm(p[MIDDLE_MCP, :2] - p[WRIST, :2]) + 1e-6

    def ext(i):
        return np.linalg.norm(p[i, :2] - p[WRIST, :2]) / palm

    idx, mid, ring = ext(INDEX_TIP), ext(MIDDLE_TIP), ext(RING_TIP)
    gap = np.linalg.norm(p[THUMB_TIP, :2] - p[INDEX_TIP, :2]) / palm
    thumb_above_wrist = p[THUMB_TIP, 1] < p[WRIST, 1]
    return {
        PALM: idx > 1.2 and mid > 1.2,
        PINCH: gap < 0.5,
        FIST: idx < 1.1 and mid < 1.1,
        V_SIGN: idx > 1.2 and mid > 1.2 and ring < 1.2,
        THUMB_UP: thumb_above_wrist and idx < 1.7,
        THUMB_DOWN: not thumb_above_wrist,
    }[pose]


class Recorder:
    """Appends raw MediaPipe landmark rows to one CSV per user."""

    def __init__(self, path, user):
        self.path, self.user = path, user
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

    def drop(self, pose):
        """Remove this user's rows for `pose`, so a re-record replaces instead of appends."""
        self.f.close()
        with open(self.path, newline="") as f:
            rows = list(csv.reader(f))[1:]
        kept = [r for r in rows if not (r[0] == self.user and r[1] == pose)]
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            w.writerows(kept)
        self.f = open(self.path, "a", newline="")
        self.w = csv.writer(self.f)
        return len(rows) - len(kept)

    def close(self):
        self.f.close()


class RecordSession:
    """Timing state machine: idle -> countdown (3 s) -> capturing (10 s) -> next pose ... -> done.

    `select` records one pose only. `fresh` is set when a capture starts so the caller can
    drop that pose's old rows once; the caller clears it.
    """

    def __init__(self, poses=POSES, capture_s=CAPTURE_S):
        self.poses = list(poses)
        self.capture_s = capture_s
        self.i = 0
        self.phase = "idle"
        self.t0 = 0.0
        self.single = False
        self.fresh = False

    @property
    def pose(self):
        return self.poses[self.i] if self.i < len(self.poses) else None

    @property
    def capturing(self):
        return self.phase == "capturing"

    def start(self, t, single=False):
        self.single = single
        self.phase, self.t0 = "countdown", t

    def select(self, idx, t):
        if 0 <= idx < len(self.poses):
            self.i = idx
            self.start(t, single=True)

    def redo(self, t):
        if self.phase in ("idle", "done"):
            self.i = 0
            self.start(t)
        else:
            self.start(t, single=self.single)

    def update(self, t):
        """-> (phase, pose, seconds_left)."""
        if self.phase == "countdown" and t - self.t0 >= COUNTDOWN_S:
            self.phase, self.t0, self.fresh = "capturing", t, True
        elif self.phase == "capturing" and t - self.t0 >= self.capture_s:
            if not self.single and self.i + 1 < len(self.poses):
                self.i += 1
                self.start(t)
            else:
                self.i = min(self.i + 1, len(self.poses)) if not self.single else self.i
                self.phase = "done"
        total = {"countdown": COUNTDOWN_S, "capturing": self.capture_s}.get(self.phase)
        left = max(total - (t - self.t0), 0.0) if total else 0.0
        return self.phase, self.pose, left
