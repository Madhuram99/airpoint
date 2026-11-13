# AirPoint Week 1 — Package Split, Recorder, Trained Pose Classifier Live

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** All six hand poses (`palm pinch fist v_sign thumb_up thumb_down`) drive the real mouse through a trained MLP running in the live loop, with a rules fallback toggled by `k`, and a record mode that builds the real dataset.

**Architecture:** The single live script becomes the `airpoint/` package: camera and tracker produce landmarks; `features` normalizes; a classifier (`PoseClassifier` MLP or `RuleClassifier`) plus `VoteSmoother` yields a stable label; a pure `AirPointFSM` turns label + time into events; `ActionRunner` maps events to pyautogui. `recorder` writes raw landmarks to CSV; `train` fits the MLP with leave-one-user-out reporting.

**Tech Stack:** Python 3.12, OpenCV 5, MediaPipe 1.2 Tasks API, scikit-learn MLPClassifier, joblib, pyautogui, pytest.

**Spec:** `docs/superpowers/specs/2026-11-13-airpoint-pose-classifier-design.md`. This plan covers the spec's Week 1 only. Weeks 2 (dwell, calibration, mini HUD) and 3 (hotkey poses, dashboard card) get their own plans once real data exists.

**Conventions:** project root is `~/Projects/airpoint` after Task 1. All paths below are relative to it. Run everything with `venv/bin/python` / `venv/bin/pytest`. Commit messages carry no co-author trailer.

---

## File map

| File | Responsibility |
|---|---|
| `airpoint/__init__.py` | empty |
| `airpoint/poses.py` | pose label constants, grip/scroll groupings |
| `airpoint/features.py` | landmark array conversion, `normalize_63`, `pinch_ratio` |
| `airpoint/filters.py` | `OneEuroFilter` |
| `airpoint/classifier.py` | `RuleClassifier`, `PoseClassifier`, `VoteSmoother` |
| `airpoint/fsm.py` | `Event`, `AirPointFSM` (pure) |
| `airpoint/actions.py` | `FakeMouse`, `PyAutoGuiMouse`, `ActionRunner` |
| `airpoint/camera.py` | `Camera` with index fallback |
| `airpoint/tracker.py` | `HandTracker` MediaPipe wrapper |
| `airpoint/hud.py` | drawing helpers |
| `airpoint/recorder.py` | `Recorder` (CSV) + `RecordSession` (timing state machine) |
| `airpoint/train.py` | load, augment, LOUO report, fit, save |
| `airpoint/app.py` | wiring + keys |
| `run_demo.py` | entry point |
| `tests/hands.py` | synthetic hand fixtures |
| `tests/test_*.py` | one per module |
| `conftest.py` | empty; makes root importable for pytest |

---

### Task 1: Housekeeping — stop the other session, move, fresh venv, pytest

**Files:**
- Move: `~/Downloads/Major Project` → `~/Projects/airpoint`
- Create: `conftest.py`, `pytest.ini`
- Modify: `.gitignore`

- [ ] **Step 1: Stop the other Claude session that has this folder open**

A Claude desktop-app session (started 10:12 today) created the venv and edited both scripts this morning. Close it in its own window. Do not proceed while two sessions edit the folder.

- [ ] **Step 2: Move the folder**

```bash
mkdir -p ~/Projects && mv ~/Downloads/"Major Project" ~/Projects/airpoint && cd ~/Projects/airpoint && git status --short
```
Expected: no output from `git status` (clean tree, the move keeps `.git`).

- [ ] **Step 3: Recreate the venv (absolute paths inside it broke on the move)**

```bash
cd ~/Projects/airpoint && rm -rf venv && /opt/homebrew/opt/python@3.12/bin/python3.12 -m venv venv && venv/bin/pip install --quiet numpy scikit-learn opencv-python mediapipe pyautogui pytest && venv/bin/python -c "import cv2, mediapipe, sklearn, pyautogui, pytest; print('ok')"
```
Expected: `ok`. The pip cache holds every wheel from this morning, so this is fast even on a flaky network.

- [ ] **Step 4: Git identity and pytest config**

```bash
cd ~/Projects/airpoint && git config user.name "Madhuram Rathi" && git config user.email "madhuram@solfoundry.co"
touch conftest.py
printf '[pytest]\ntestpaths = tests\n' > pytest.ini
printf '\nmodels/train_report.json\n' >> .gitignore
mkdir -p airpoint tests data models && touch airpoint/__init__.py tests/__init__.py
```

- [ ] **Step 5: Verify pytest runs (no tests yet)**

Run: `venv/bin/pytest -q`
Expected: `no tests ran`

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "chore: package skeleton, pytest config, fresh venv notes"
```

---

### Task 2: `poses.py` and `features.py`

**Files:**
- Create: `airpoint/poses.py`, `airpoint/features.py`, `tests/hands.py`, `tests/test_features.py`

- [ ] **Step 1: Write the pose constants**

`airpoint/poses.py`:
```python
PALM, PINCH, FIST, V_SIGN, THUMB_UP, THUMB_DOWN = (
    "palm", "pinch", "fist", "v_sign", "thumb_up", "thumb_down",
)
POSES = (PALM, PINCH, FIST, V_SIGN, THUMB_UP, THUMB_DOWN)

NONE = "none"        # no hand in frame
UNKNOWN = "unknown"  # hand present, confidence below gate

GRIP = frozenset({PINCH, FIST})
SCROLL = {THUMB_UP: 1, THUMB_DOWN: -1}
```

- [ ] **Step 2: Write the synthetic hand fixtures**

`tests/hands.py`:
```python
"""Synthetic 21-point hands in MediaPipe image coordinates (x right, y down, 0..1).

Wrist sits at (0.50, 0.80); the middle-finger knuckle at (0.50, 0.60), so the palm
length is exactly 0.2. Each finger's joints are interpolated from its base to the
given fingertip. z is 0 everywhere.
"""
import numpy as np

_BASES = ((1, (0.44, 0.74)), (5, (0.42, 0.62)), (9, (0.50, 0.60)),
          (13, (0.57, 0.62)), (17, (0.63, 0.66)))


def hand(thumb, index, middle, ring, pinky):
    pts = np.zeros((21, 3))
    pts[0, :2] = (0.50, 0.80)
    for (base_idx, base), tip in zip(_BASES, (thumb, index, middle, ring, pinky)):
        for k in range(4):
            f = k / 3
            pts[base_idx + k, 0] = base[0] + (tip[0] - base[0]) * f
            pts[base_idx + k, 1] = base[1] + (tip[1] - base[1]) * f
    return pts


CURLED = dict(index=(0.44, 0.68), middle=(0.50, 0.67), ring=(0.56, 0.68), pinky=(0.61, 0.70))

OPEN_PALM = hand(thumb=(0.33, 0.62), index=(0.40, 0.38), middle=(0.50, 0.34),
                 ring=(0.59, 0.38), pinky=(0.68, 0.48))
PINCH = hand(thumb=(0.42, 0.44), index=(0.43, 0.43), middle=(0.50, 0.34),
             ring=(0.59, 0.38), pinky=(0.68, 0.48))
FIST = hand(thumb=(0.45, 0.66), **CURLED)
V_SIGN = hand(thumb=(0.45, 0.66), index=(0.40, 0.38), middle=(0.50, 0.34),
              ring=(0.56, 0.68), pinky=(0.61, 0.70))
THUMB_UP = hand(thumb=(0.50, 0.42), **CURLED)
THUMB_DOWN = hand(thumb=(0.50, 1.18), **CURLED)

BY_POSE = {"palm": OPEN_PALM, "pinch": PINCH, "fist": FIST, "v_sign": V_SIGN,
           "thumb_up": THUMB_UP, "thumb_down": THUMB_DOWN}
```

- [ ] **Step 3: Write the failing feature tests**

`tests/test_features.py`:
```python
import numpy as np
from airpoint import features as F
from tests.hands import OPEN_PALM, PINCH


def test_normalize_puts_wrist_at_origin_and_palm_at_unit_length():
    pts = F.normalize_63(OPEN_PALM).reshape(21, 3)
    assert np.allclose(pts[F.WRIST], 0.0)
    assert abs(np.linalg.norm(pts[F.MIDDLE_MCP]) - 1.0) < 1e-5


def test_normalize_is_invariant_to_scale_and_translation():
    moved = OPEN_PALM * 3.0 + np.array([7.0, -2.0, 0.5])
    assert np.allclose(F.normalize_63(moved), F.normalize_63(OPEN_PALM), atol=1e-6)


def test_normalize_accepts_landmark_objects_with_xyz_attrs():
    class LM:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z
    objs = [LM(*row) for row in OPEN_PALM]
    assert np.allclose(F.normalize_63(objs), F.normalize_63(OPEN_PALM))


def test_pinch_ratio_is_thumb_index_gap_over_palm():
    assert F.pinch_ratio(OPEN_PALM) > 1.0
    assert F.pinch_ratio(PINCH) < 0.10
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_features.py -q`
Expected: `ImportError` / `ModuleNotFoundError: airpoint.features`

- [ ] **Step 5: Implement features**

`airpoint/features.py`:
```python
import numpy as np

WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 0, 4, 8, 9, 12, 16, 20


def to_array(landmarks) -> np.ndarray:
    """21 landmark objects with .x/.y/.z, or any 21x3 iterable -> (21, 3) float64."""
    if hasattr(landmarks[0], "x"):
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
    return np.asarray(landmarks, dtype=np.float64).reshape(21, 3)


def normalize_63(landmarks) -> np.ndarray:
    """Wrist to origin, divide by 3-D palm length ||P9 - P0||. Scale- and position-free."""
    pts = to_array(landmarks)
    pts = pts - pts[WRIST]
    palm = np.linalg.norm(pts[MIDDLE_MCP]) + 1e-6
    return (pts / palm).flatten()


def pinch_ratio(landmarks) -> float:
    """2-D thumb-tip to index-tip distance over 2-D palm length (the live thresholds are 2-D)."""
    pts = to_array(landmarks)
    palm = np.linalg.norm(pts[MIDDLE_MCP, :2] - pts[WRIST, :2]) + 1e-6
    return float(np.linalg.norm(pts[THUMB_TIP, :2] - pts[INDEX_TIP, :2]) / palm)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_features.py -q`
Expected: `4 passed`

- [ ] **Step 7: Commit**

```bash
git add airpoint/poses.py airpoint/features.py tests/hands.py tests/test_features.py
git commit -m "feat: pose constants, landmark normalization, synthetic hand fixtures"
```

---

### Task 3: `filters.py` — OneEuroFilter

**Files:**
- Create: `airpoint/filters.py`, `tests/test_filters.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_filters.py`:
```python
import math
from airpoint.filters import OneEuroFilter


def test_first_sample_passes_through():
    f = OneEuroFilter(fc_min=0.8, beta=0.01)
    assert f.filter(100.0, t=0.0) == 100.0


def test_steady_jitter_is_damped():
    f = OneEuroFilter(fc_min=0.8, beta=0.01)
    outs = [f.filter(200.0 + 14.0 * math.sin(2 * math.pi * 6 * (i / 30)), t=i / 30) for i in range(60)]
    settled = outs[30:]
    assert max(abs(v - 200.0) for v in settled) < 7.0  # raw peak is 14 px


def test_fast_move_is_followed_closely():
    f = OneEuroFilter(fc_min=0.8, beta=0.01)
    for i in range(30):
        f.filter(200.0, t=i / 30)
    out = None
    for i in range(30, 60):
        out = f.filter(800.0, t=i / 30)
    assert abs(out - 800.0) < 20.0


def test_custom_fc_min_lowers_cutoff_for_more_smoothing():
    slow, fast = OneEuroFilter(fc_min=0.8), OneEuroFilter(fc_min=0.8)
    for i in range(10):
        slow.filter(0.0, t=i / 30, custom_fc_min=0.3)
        fast.filter(0.0, t=i / 30)
    assert slow.filter(100.0, t=11 / 30, custom_fc_min=0.3) < fast.filter(100.0, t=11 / 30)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_filters.py -q`
Expected: `ModuleNotFoundError: airpoint.filters`

- [ ] **Step 3: Implement (lifted from the live script, unchanged behaviour)**

`airpoint/filters.py`:
```python
import math


class OneEuroFilter:
    """Casiez, Roussel, Vogel (CHI 2012). fc = fc_min + beta * |speed|."""

    def __init__(self, fc_min=0.80, beta=0.010, d_cutoff=1.0):
        self.fc_min, self.beta, self.d_cutoff = fc_min, beta, d_cutoff
        self.x_prev, self.dx_prev, self.t_prev = None, 0.0, None

    @staticmethod
    def _alpha(fc, dt):
        return 1.0 / (1.0 + 1.0 / (2.0 * math.pi * fc * dt))

    def filter(self, x, t, custom_fc_min=None):
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-5)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        edx = a_d * dx + (1.0 - a_d) * self.dx_prev
        fc = (self.fc_min if custom_fc_min is None else custom_fc_min) + self.beta * abs(edx)
        alpha = self._alpha(fc, dt)
        x_out = alpha * x + (1.0 - alpha) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_out, edx, t
        return x_out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_filters.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add airpoint/filters.py tests/test_filters.py && git commit -m "feat: OneEuroFilter module"
```

---

### Task 4: `classifier.py` — rules, MLP wrapper, vote smoother

**Files:**
- Create: `airpoint/classifier.py`, `tests/test_classifier.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_classifier.py`:
```python
import numpy as np
from airpoint.classifier import PoseClassifier, RuleClassifier, VoteSmoother
from airpoint.poses import NONE, UNKNOWN
from tests.hands import BY_POSE


def test_rules_classify_every_synthetic_pose():
    clf = RuleClassifier()
    for pose, pts in BY_POSE.items():
        label, conf = clf.predict(pts)
        assert label == pose, f"{pose} read as {label}"
        assert conf == 1.0


def test_rules_report_name():
    assert RuleClassifier().name == "rules"


class _Model:
    classes_ = np.array(["fist", "palm", "pinch"])

    def predict_proba(self, x):
        assert x.shape == (1, 63)
        return np.array([[0.1, 0.2, 0.7]])


def test_pose_classifier_returns_label_from_labels_list_and_confidence():
    clf = PoseClassifier(_Model(), ["fist", "palm", "pinch"])
    assert clf.predict(BY_POSE["pinch"]) == ("pinch", 0.7)
    assert clf.name == "mlp"


def test_smoother_ignores_one_or_two_frame_flicker():
    s = VoteSmoother(window=5, min_conf=0.75)
    for _ in range(5):
        s.update("palm", 0.9)
    assert s.update("pinch", 0.9) == "palm"
    assert s.update("pinch", 0.9) == "palm"


def test_smoother_switches_on_majority():
    s = VoteSmoother(window=5, min_conf=0.75)
    for _ in range(5):
        s.update("palm", 0.9)
    s.update("pinch", 0.9)
    s.update("pinch", 0.9)
    assert s.update("pinch", 0.9) == "pinch"


def test_smoother_low_confidence_is_unknown():
    s = VoteSmoother(window=5, min_conf=0.75)
    for _ in range(4):
        s.update("pinch", 0.5)
    assert s.update("pinch", 0.5) == UNKNOWN


def test_smoother_no_hand_is_none_and_clears_history():
    s = VoteSmoother(window=5, min_conf=0.75)
    for _ in range(5):
        s.update("palm", 0.9)
    assert s.update(None, 0.0) == NONE
    assert s.update("pinch", 0.9) == "pinch"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_classifier.py -q`
Expected: `ModuleNotFoundError: airpoint.classifier`

- [ ] **Step 3: Implement**

`airpoint/classifier.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_classifier.py -q`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add airpoint/classifier.py tests/test_classifier.py && git commit -m "feat: rule and MLP pose classifiers with vote smoother"
```

---

### Task 5: `fsm.py` — pose + time → events

**Files:**
- Create: `airpoint/fsm.py`, `tests/test_fsm.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_fsm.py`:
```python
from airpoint.fsm import AirPointFSM
from airpoint.poses import NONE, UNKNOWN


def run(seq, fsm=None):
    fsm = fsm or AirPointFSM()
    out = []
    for pose, t in seq:
        _, events = fsm.update(pose, t)
        out += [(e.kind, e.arg) for e in events]
    return out, fsm


def test_short_pinch_is_a_click():
    events, fsm = run([("palm", 0.0), ("pinch", 0.1), ("pinch", 0.2), ("palm", 0.3)])
    assert events == [("click", None)]
    assert fsm.state == "MOVING"


def test_held_pinch_starts_and_ends_a_drag():
    events, _ = run([("palm", 0.0), ("pinch", 0.1), ("pinch", 0.65), ("pinch", 0.7), ("palm", 0.8)])
    assert events == [("drag_start", None), ("drag_end", None)]


def test_pinch_drifting_into_fist_keeps_the_drag():
    events, fsm = run([("palm", 0.0), ("pinch", 0.1), ("pinch", 0.7), ("fist", 0.8)])
    assert events == [("drag_start", None)]
    assert fsm.state == "DRAGGING"


def test_hand_lost_mid_drag_ends_it_and_rests():
    events, fsm = run([("palm", 0.0), ("pinch", 0.1), ("pinch", 0.7), (NONE, 0.8)])
    assert events == [("drag_start", None), ("drag_end", None)]
    assert fsm.state == "RESTING"


def test_unknown_holds_state():
    events, fsm = run([("palm", 0.0), ("pinch", 0.1), ("pinch", 0.7), (UNKNOWN, 0.8), (UNKNOWN, 0.9)])
    assert events == [("drag_start", None)]
    assert fsm.state == "DRAGGING"


def test_v_sign_fires_once_and_rearms_only_after_palm():
    events, _ = run([("palm", 0.0), ("v_sign", 0.1), ("v_sign", 0.45), ("v_sign", 0.6), ("v_sign", 0.9),
                     ("palm", 1.0), ("v_sign", 1.1), ("v_sign", 1.5)])
    assert events == [("right_click", None), ("right_click", None)]


def test_v_sign_released_early_does_nothing():
    events, fsm = run([("palm", 0.0), ("v_sign", 0.1), ("palm", 0.2)])
    assert events == []
    assert fsm.state == "MOVING"


def test_thumb_scrolls_at_fixed_rate_and_direction_follows_pose():
    events, fsm = run([("palm", 0.0), ("thumb_up", 0.1), ("thumb_up", 0.15), ("thumb_up", 0.2),
                       ("thumb_up", 0.26), ("thumb_down", 0.3), ("thumb_down", 0.4), ("palm", 0.5)])
    assert events == [("scroll", 2), ("scroll", 2), ("scroll", -2)]
    assert fsm.state == "MOVING"


def test_cursor_frozen_only_while_resting_or_scrolling():
    fsm = AirPointFSM()
    assert fsm.cursor_frozen
    fsm.update("palm", 0.0)
    assert not fsm.cursor_frozen
    fsm.update("thumb_up", 0.1)
    assert fsm.cursor_frozen
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_fsm.py -q`
Expected: `ModuleNotFoundError: airpoint.fsm`

- [ ] **Step 3: Implement**

`airpoint/fsm.py`:
```python
from dataclasses import dataclass

from .poses import GRIP, NONE, PALM, SCROLL, UNKNOWN, V_SIGN

PRESS_HOLD_S = 0.50     # grip held this long -> drag instead of click
HOLD_FIRE_S = 0.30      # one-shot poses (v_sign, hotkeys) fire after this
SCROLL_PERIOD_S = 0.10
SCROLL_TICKS = 2


@dataclass(frozen=True)
class Event:
    kind: str          # click | right_click | drag_start | drag_end | scroll | hotkey
    arg: object = None  # scroll: signed ticks; hotkey: tuple of keys


DEFAULT_HOLD_EVENTS = {V_SIGN: Event("right_click")}


class AirPointFSM:
    """Pure: update(pose, t) -> (state, [Event]). Clock is a parameter."""

    def __init__(self, hold_events=None):
        self.hold_events = dict(DEFAULT_HOLD_EVENTS if hold_events is None else hold_events)
        self.state = "RESTING"
        self.t_enter = 0.0
        self.hold_pose = None
        self.hold_fired = False
        self.hold_armed = True
        self.scroll_dir = 0
        self.next_scroll_t = 0.0

    @property
    def cursor_frozen(self):
        return self.state in ("RESTING", "SCROLLING")

    def _to(self, state, t):
        self.state, self.t_enter = state, t

    def update(self, pose, t):
        events = []
        if pose == UNKNOWN:
            return self.state, events
        if pose == NONE:
            if self.state == "DRAGGING":
                events.append(Event("drag_end"))
            self._to("RESTING", t)
            return self.state, events

        s = self.state
        if s == "RESTING":
            self._to("MOVING", t)
        elif s == "MOVING":
            if pose == PALM:
                self.hold_armed = True
            elif pose in GRIP:
                self._to("PRESSING", t)
            elif pose in SCROLL:
                self.scroll_dir, self.next_scroll_t = SCROLL[pose], t
                self._to("SCROLLING", t)
            elif pose in self.hold_events and self.hold_armed:
                self.hold_pose, self.hold_fired = pose, False
                self._to("HOLD", t)
        elif s == "PRESSING":
            held = t - self.t_enter
            if pose not in GRIP:
                if held < PRESS_HOLD_S:
                    events.append(Event("click"))
                self._to("MOVING", t)
            elif held >= PRESS_HOLD_S:
                events.append(Event("drag_start"))
                self._to("DRAGGING", t)
        elif s == "DRAGGING":
            if pose not in GRIP:
                events.append(Event("drag_end"))
                self._to("MOVING", t)
        elif s == "HOLD":
            if pose != self.hold_pose:
                if pose == PALM:
                    self.hold_armed = True
                self._to("MOVING", t)
            elif not self.hold_fired and t - self.t_enter >= HOLD_FIRE_S:
                events.append(self.hold_events[pose])
                self.hold_fired, self.hold_armed = True, False
        elif s == "SCROLLING":
            if pose not in SCROLL:
                self._to("MOVING", t)
            else:
                self.scroll_dir = SCROLL[pose]
                if t >= self.next_scroll_t:
                    events.append(Event("scroll", self.scroll_dir * SCROLL_TICKS))
                    self.next_scroll_t = t + SCROLL_PERIOD_S
        return self.state, events
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_fsm.py -q`
Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add airpoint/fsm.py tests/test_fsm.py && git commit -m "feat: pose-driven FSM with grip, hold and scroll states"
```

---

### Task 6: `actions.py` — events → mouse

**Files:**
- Create: `airpoint/actions.py`, `tests/test_actions.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_actions.py`:
```python
from airpoint.actions import ActionRunner, FakeMouse
from airpoint.fsm import Event


def make():
    m = FakeMouse()
    return m, ActionRunner(m, m.size())


def test_disabled_runner_moves_nothing_and_applies_nothing():
    m, r = make()
    r.move(0.5, 0.5)
    r.apply([Event("click"), Event("scroll", 2)])
    assert m.calls == []


def test_active_region_maps_to_full_screen_and_clamps():
    _, r = make()
    assert r.to_screen(0.0, 0.0) == (0.0, 0.0)
    assert r.to_screen(1.0, 1.0) == (999.0, 499.0)
    sx, sy = r.to_screen(0.5, 0.475)
    assert abs(sx - 499.5) < 1e-6 and abs(sy - 249.5) < 1e-6


def test_events_map_to_mouse_calls_when_enabled():
    m, r = make()
    r.toggle()
    r.move(0.15, 0.15)
    r.apply([Event("click"), Event("right_click"), Event("scroll", -2),
             Event("hotkey", ("command", "tab"))])
    assert m.calls == [("move", 0, 0), ("click",), ("right_click",), ("scroll", -2),
                       ("hotkey", "command", "tab")]


def test_drag_end_without_drag_start_is_a_noop():
    m, r = make()
    r.toggle()
    r.apply([Event("drag_end")])
    assert m.calls == []


def test_toggle_off_releases_a_held_button():
    m, r = make()
    r.toggle()
    r.apply([Event("drag_start")])
    r.toggle()
    assert m.calls == [("down",), ("up",)]
    assert not r.enabled and not r.button_down
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_actions.py -q`
Expected: `ModuleNotFoundError: airpoint.actions`

- [ ] **Step 3: Implement**

`airpoint/actions.py`:
```python
ACTIVE_REGION = (0.15, 0.15, 0.85, 0.80)  # inner frame box that maps to the whole screen


class FakeMouse:
    def __init__(self):
        self.calls = []

    def size(self):
        return (1000, 500)

    def move(self, x, y):
        self.calls.append(("move", round(x), round(y)))

    def click(self):
        self.calls.append(("click",))

    def right_click(self):
        self.calls.append(("right_click",))

    def mouse_down(self):
        self.calls.append(("down",))

    def mouse_up(self):
        self.calls.append(("up",))

    def scroll(self, ticks):
        self.calls.append(("scroll", ticks))

    def hotkey(self, *keys):
        self.calls.append(("hotkey",) + keys)


class PyAutoGuiMouse:
    def __init__(self):
        import pyautogui
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        self.pg = pyautogui

    def size(self):
        return tuple(self.pg.size())

    def move(self, x, y):
        self.pg.moveTo(x, y)

    def click(self):
        self.pg.click()

    def right_click(self):
        self.pg.rightClick()

    def mouse_down(self):
        self.pg.mouseDown()

    def mouse_up(self):
        self.pg.mouseUp()

    def scroll(self, ticks):
        self.pg.scroll(ticks)

    def hotkey(self, *keys):
        self.pg.hotkey(*keys)


class ActionRunner:
    """Starts disabled. toggle() flips it and always releases a held button first."""

    def __init__(self, mouse, screen_size, active_region=ACTIVE_REGION):
        self.mouse = mouse
        self.screen_w, self.screen_h = screen_size
        self.active_region = active_region
        self.enabled = False
        self.button_down = False

    def toggle(self):
        self.release()
        self.enabled = not self.enabled

    def to_screen(self, nx, ny):
        x0, y0, x1, y1 = self.active_region
        sx = min(max((nx - x0) / (x1 - x0), 0.0), 1.0) * (self.screen_w - 1)
        sy = min(max((ny - y0) / (y1 - y0), 0.0), 1.0) * (self.screen_h - 1)
        return sx, sy

    def move(self, nx, ny):
        if self.enabled:
            self.mouse.move(*self.to_screen(nx, ny))

    def apply(self, events):
        if not self.enabled:
            return
        for e in events:
            if e.kind == "click":
                self.mouse.click()
            elif e.kind == "right_click":
                self.mouse.right_click()
            elif e.kind == "drag_start":
                self.mouse.mouse_down()
                self.button_down = True
            elif e.kind == "drag_end":
                self.release()
            elif e.kind == "scroll":
                self.mouse.scroll(e.arg)
            elif e.kind == "hotkey":
                self.mouse.hotkey(*e.arg)

    def release(self):
        if self.button_down:
            self.mouse.mouse_up()
            self.button_down = False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_actions.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add airpoint/actions.py tests/test_actions.py && git commit -m "feat: ActionRunner maps FSM events to the OS mouse"
```

---

### Task 7: `camera.py`, `tracker.py`, `hud.py` (manual-verified)

**Files:**
- Create: `airpoint/camera.py`, `airpoint/tracker.py`, `airpoint/hud.py`

- [ ] **Step 1: Camera with index fallback**

`airpoint/camera.py`:
```python
import cv2


class Camera:
    """Index 1 is the MacBook camera when an iPhone takes index 0; otherwise 0."""

    def __init__(self, indices=(1, 0)):
        for i in indices:
            cap = cv2.VideoCapture(i)
            if cap.isOpened() and cap.read()[0]:
                self.cap, self.index = cap, i
                return
            cap.release()
        raise RuntimeError("Cannot open webcam. System Settings > Privacy & Security > Camera.")

    def frames(self):
        while self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                return
            yield cv2.flip(frame, 1)

    def close(self):
        self.cap.release()
```

- [ ] **Step 2: Tracker wrapper**

`airpoint/tracker.py`:
```python
import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "hand_landmarker.task")
DOWNLOAD = ("curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")


class HandTracker:
    def __init__(self, model_path=MODEL_PATH, num_hands=1):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} missing. Download with:\n  {DOWNLOAD}")
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr):
        """-> ((21, 3) array in normalized image coords, handedness) or (None, None)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.hand_landmarks:
            return None, None
        pts = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]])
        hand = result.handedness[0][0].category_name if result.handedness else "Unknown"
        return pts, hand

    def close(self):
        self.detector.close()
```

- [ ] **Step 3: HUD drawing helpers**

`airpoint/hud.py`:
```python
import cv2

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]
BANNER_H = 82
FONT = cv2.FONT_HERSHEY_SIMPLEX
ACCENT_ON, ACCENT_OFF, WHITE, MINT = (0, 200, 255), (90, 90, 90), (255, 255, 255), (167, 243, 208)


def draw_skeleton(frame, pts):
    h, w = frame.shape[:2]
    px = [(int(x * w), int(y * h)) for x, y, _ in pts]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, px[a], px[b], (250, 44, 250), 2)
    for p in px:
        cv2.circle(frame, p, 3, (121, 22, 76), -1)


def draw_cursor(frame, raw_xy, smooth_xy):
    cv2.circle(frame, raw_xy, 6, (0, 0, 255), -1)
    cv2.circle(frame, smooth_xy, 12, (0, 255, 0), 2)


def draw_region(frame, region, active):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = region
    cv2.rectangle(frame, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)),
                  ACCENT_ON if active else ACCENT_OFF, 1)


def draw_banner(frame, line1, line2, active):
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, BANNER_H), (25, 35, 30), -1)
    cv2.putText(frame, line1, (15, 32), FONT, 0.62, MINT, 2)
    cv2.putText(frame, line2, (15, 64), FONT, 0.55, ACCENT_ON if active else WHITE, 1)


def draw_prompt(frame, title, subtitle):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h // 2 - 70), (w, h // 2 + 50), (25, 35, 30), -1)
    cv2.putText(frame, title, (30, h // 2 - 10), FONT, 1.1, MINT, 2)
    cv2.putText(frame, subtitle, (30, h // 2 + 30), FONT, 0.6, WHITE, 1)
```

- [ ] **Step 4: Smoke-test tracker + camera from the terminal (window opens, 60 frames, closes)**

Run:
```bash
venv/bin/python - <<'EOF'
from airpoint.camera import Camera
from airpoint.tracker import HandTracker
from airpoint import hud
import cv2
cam, tr = Camera(), HandTracker()
seen = 0
for i, frame in enumerate(cam.frames()):
    pts, hand = tr.detect(frame)
    if pts is not None:
        seen += 1
        hud.draw_skeleton(frame, pts)
    hud.draw_banner(frame, f"frame {i} hand={hand}", "smoke test", False)
    cv2.imshow("smoke", frame); cv2.waitKey(1)
    if i >= 60: break
cam.close(); tr.close(); cv2.destroyAllWindows()
print("frames with a hand:", seen)
EOF
```
Expected: a window for ~2 s; hold your hand up; prints `frames with a hand: <n>` with n > 0.

- [ ] **Step 5: Commit**

```bash
git add airpoint/camera.py airpoint/tracker.py airpoint/hud.py && git commit -m "feat: camera, MediaPipe tracker and HUD helpers"
```

---

### Task 8: `recorder.py` — CSV writer and record-session timing

**Files:**
- Create: `airpoint/recorder.py`, `tests/test_recorder.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_recorder.py`:
```python
import csv

from airpoint.recorder import HEADER, Recorder, RecordSession
from tests.hands import OPEN_PALM


def test_recorder_writes_header_once_and_rows_with_63_coords(tmp_path):
    path = tmp_path / "a.csv"
    r = Recorder(str(path), "a")
    r.write("palm", 1.5, "Right", OPEN_PALM)
    r.close()
    r2 = Recorder(str(path), "a")
    r2.write("pinch", 2.0, "Right", OPEN_PALM)
    r2.close()
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == 2 and len(HEADER) == 4 + 63
    assert rows[0]["pose"] == "palm" and rows[1]["pose"] == "pinch"
    assert float(rows[0]["x0"]) == 0.5 and float(rows[0]["y0"]) == 0.8
    assert r2.count == 1


def test_session_counts_down_then_captures_then_advances_then_finishes():
    s = RecordSession(("palm", "pinch"))
    assert s.update(0.0) == ("idle", "palm", 0.0)
    s.start(0.0)
    assert s.update(0.0) == ("countdown", "palm", 3.0)
    assert s.update(2.9)[0] == "countdown"
    assert s.update(3.0) == ("capturing", "palm", 10.0)
    assert s.capturing
    assert s.update(12.9)[0] == "capturing"
    assert s.update(13.0) == ("countdown", "pinch", 3.0)
    assert s.update(16.0)[0] == "capturing"
    assert s.update(26.0) == ("done", None, 0.0)
    assert not s.capturing


def test_session_select_jumps_to_a_pose_and_redo_restarts_current():
    s = RecordSession(("palm", "pinch", "fist"))
    s.select(2, 5.0)
    assert s.update(5.0) == ("countdown", "fist", 3.0)
    s.update(8.0)
    s.redo(9.0)
    assert s.update(9.0) == ("countdown", "fist", 3.0)
    s.update(12.0)
    s.update(22.0)
    assert s.update(22.0)[0] == "done"
    s.redo(23.0)
    assert s.update(23.0) == ("countdown", "palm", 3.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_recorder.py -q`
Expected: `ModuleNotFoundError: airpoint.recorder`

- [ ] **Step 3: Implement**

`airpoint/recorder.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_recorder.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add airpoint/recorder.py tests/test_recorder.py && git commit -m "feat: landmark CSV recorder and record-session timing"
```

---

### Task 9: `train.py` — load, augment, leave-one-user-out, fit, save

**Files:**
- Create: `airpoint/train.py`, `tests/test_train.py`

- [ ] **Step 1: Write the failing test**

`tests/test_train.py`:
```python
import json

import numpy as np

from airpoint.recorder import Recorder
from airpoint.train import train_and_save
from tests.hands import BY_POSE


def _write_user(path, user, rng, scale, shift):
    r = Recorder(str(path), user)
    for pose, pts in BY_POSE.items():
        for _ in range(30):
            jitter = rng.normal(0.0, 0.006, size=pts.shape)
            p = (pts - pts[0]) * scale + pts[0] + shift + jitter
            r.write(pose, 0.0, "Right", p)
    r.close()


def test_train_reports_louo_accuracy_and_saves_model(tmp_path):
    rng = np.random.default_rng(0)
    data, models = tmp_path / "data", tmp_path / "models"
    data.mkdir()
    _write_user(data / "a.csv", "a", rng, 1.0, np.array([0.0, 0.0, 0.0]))
    _write_user(data / "b.csv", "b", rng, 1.25, np.array([0.1, -0.05, 0.0]))

    report = train_and_save(str(data), str(models), seed=0)

    assert report["n_rows"] == 360 and sorted(report["users"]) == ["a", "b"]
    assert report["louo_acc"] >= 0.9
    assert set(report["per_pose"]) == set(BY_POSE)
    labels = json.loads((models / "labels.json").read_text())
    assert labels == sorted(BY_POSE)
    assert (models / "pose_mlp.joblib").exists()
    assert (models / "train_report.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_train.py -q`
Expected: `ModuleNotFoundError: airpoint.train`

- [ ] **Step 3: Implement**

`airpoint/train.py`:
```python
"""Train the 63-input pose MLP on data/*.csv. Reports leave-one-user-out accuracy.

    venv/bin/python -m airpoint.train            # data/ -> models/
"""
import csv
import glob
import json
import os
import sys
import time

import joblib
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier

from .features import normalize_63

TILT_MAX_RAD = 0.60   # +/- 34 degrees in-plane
NOISE_SIGMA = 0.12    # palm units
AUG_COPIES = 3


def load_rows(data_dir):
    X, y, users = [], [], []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                X.append([float(row[f"{a}{i}"]) for i in range(21) for a in "xyz"])
                y.append(row["pose"])
                users.append(row["user"])
    if not X:
        raise SystemExit(f"no rows in {data_dir}/*.csv — record first: run_demo.py --record --user <name>")
    return np.array(X), np.array(y), np.array(users)


def normalize_rows(X_raw):
    return np.array([normalize_63(r) for r in X_raw])


def augment(Xn, y, copies, rng):
    """Rotate each normalized hand in-plane, add landmark noise, re-normalize."""
    out_X, out_y = [Xn], [y]
    for _ in range(copies):
        pts = Xn.reshape(-1, 21, 3).copy()
        theta = rng.uniform(-TILT_MAX_RAD, TILT_MAX_RAD, size=len(pts))
        c, s = np.cos(theta)[:, None], np.sin(theta)[:, None]
        x, yy = pts[:, :, 0].copy(), pts[:, :, 1].copy()
        pts[:, :, 0], pts[:, :, 1] = c * x - s * yy, s * x + c * yy
        pts += rng.normal(0.0, NOISE_SIGMA, size=pts.shape)
        out_X.append(np.array([normalize_63(p) for p in pts]))
        out_y.append(y)
    return np.concatenate(out_X), np.concatenate(out_y)


def make_model(seed):
    return MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
                         max_iter=400, random_state=seed)


def train_and_save(data_dir, models_dir, seed=42):
    rng = np.random.default_rng(seed)
    X_raw, y, users = load_rows(data_dir)
    Xn = normalize_rows(X_raw)
    labels = sorted(set(y))

    louo, y_true, y_pred = {}, [], []
    for u in sorted(set(users)):
        tr, te = users != u, users == u
        X_tr, y_tr = augment(Xn[tr], y[tr], AUG_COPIES, rng)
        m = make_model(seed).fit(X_tr, y_tr)
        p = m.predict(Xn[te])
        louo[u] = float(np.mean(p == y[te]))
        y_true.extend(y[te]); y_pred.extend(p)
    prec, rec, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

    t0 = time.perf_counter()
    X_all, y_all = augment(Xn, y, AUG_COPIES, rng)
    model = make_model(seed).fit(X_all, y_all)
    fit_s = time.perf_counter() - t0

    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, "pose_mlp.joblib"))
    with open(os.path.join(models_dir, "labels.json"), "w") as f:
        json.dump(list(model.classes_), f)
    report = {
        "n_rows": int(len(y)), "users": sorted(set(users)), "labels": labels,
        "louo": louo, "louo_acc": float(np.mean(list(louo.values()))),
        "per_pose": {l: {"precision": float(p), "recall": float(r)} for l, p, r in zip(labels, prec, rec)},
        "fit_seconds": fit_s, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(models_dir, "train_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def main(argv=None):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = train_and_save(os.path.join(root, "data"), os.path.join(root, "models"))
    print(f"rows {r['n_rows']} | users {', '.join(r['users'])} | fit {r['fit_seconds']:.1f}s")
    print(f"leave-one-user-out accuracy: {r['louo_acc'] * 100:.1f}%  "
          + "  ".join(f"{u} {a * 100:.0f}%" for u, a in r["louo"].items()))
    print(f"{'pose':<12}{'precision':>10}{'recall':>8}")
    for l, m in r["per_pose"].items():
        print(f"{l:<12}{m['precision']:>10.3f}{m['recall']:>8.3f}")
    print("saved models/pose_mlp.joblib, models/labels.json, models/train_report.json")


if __name__ == "__main__":
    main(sys.argv[1:])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pytest tests/test_train.py -q`
Expected: `1 passed` (takes a few seconds — three MLP fits).

- [ ] **Step 5: Commit**

```bash
git add airpoint/train.py tests/test_train.py && git commit -m "feat: pose MLP training with leave-one-user-out report"
```

---

### Task 10: `app.py` + `run_demo.py` — wire the loop, retire the old script

**Files:**
- Create: `airpoint/app.py`, `run_demo.py`
- Delete: `airpoint_live_webcam_prototype_1_euro_filter_fsm_pinch.py`
- Modify: `run.txt`

- [ ] **Step 1: Write the app**

`airpoint/app.py`:
```python
import argparse
import os
import time

import cv2

from . import features as F
from . import hud
from .actions import ActionRunner, PyAutoGuiMouse
from .camera import Camera
from .classifier import PoseClassifier, RuleClassifier, VoteSmoother
from .filters import OneEuroFilter
from .fsm import AirPointFSM
from .recorder import Recorder, RecordSession
from .tracker import HandTracker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "pose_mlp.joblib")
LABELS_PATH = os.path.join(ROOT, "models", "labels.json")
FC_MIN, FC_MIN_PRECLICK, PRECLICK_RATIO = 0.80, 0.30, 0.42
WINDOW = "AirPoint"
KEYS = "q quit | m mouse | k rules/mlp | r record mode | 1-6 pick pose | space start/redo"


def load_classifiers(force_rules):
    rules = RuleClassifier()
    if force_rules or not os.path.exists(MODEL_PATH):
        return rules, None
    return rules, PoseClassifier.load(MODEL_PATH, LABELS_PATH)


def main(argv=None):
    ap = argparse.ArgumentParser(description="AirPoint live demo")
    ap.add_argument("--user", help="name for data/<user>.csv; enables record mode")
    ap.add_argument("--record", action="store_true", help="start in record mode")
    ap.add_argument("--rules", action="store_true", help="force the rule classifier")
    args = ap.parse_args(argv)
    if args.record and not args.user:
        ap.error("--record needs --user")

    rules, mlp = load_classifiers(args.rules)
    clf = mlp or rules
    smoother, fsm = VoteSmoother(), AirPointFSM()
    fx, fy = OneEuroFilter(FC_MIN), OneEuroFilter(FC_MIN)
    mouse = PyAutoGuiMouse()
    runner = ActionRunner(mouse, mouse.size())
    recorder = Recorder(os.path.join(ROOT, "data", f"{args.user}.csv"), args.user) if args.user else None
    session = RecordSession() if args.record else None
    camera, tracker = Camera(), HandTracker()
    last_event, prev_t = "none", time.time()
    print(f"[AirPoint] camera {camera.index} | classifier {clf.name} | {KEYS}")

    try:
        for frame in camera.frames():
            t = time.time()
            fps, prev_t = 1.0 / max(t - prev_t, 1e-5), t
            h, w = frame.shape[:2]
            pts, hand = tracker.detect(frame)
            if pts is not None:
                hud.draw_skeleton(frame, pts)

            if session is not None:
                phase, pose, left = session.update(t)
                if session.capturing and pts is not None:
                    recorder.write(pose, t, hand, pts)
                title = {
                    "idle": f"RECORD {args.user}: press SPACE to start",
                    "countdown": f"get ready: {pose}   {left:.0f}",
                    "capturing": f"HOLD {pose}   {left:.0f}s",
                    "done": "done - press q, or 1-6 to redo a pose",
                }[phase]
                hud.draw_prompt(frame, title, f"tilt, move closer, move away | rows {recorder.count} | {KEYS}")
            else:
                if pts is None:
                    label, conf, ratio = smoother.update(None, 0.0), 0.0, 1.0
                else:
                    raw_label, conf = clf.predict(pts)
                    label, ratio = smoother.update(raw_label, conf), F.pinch_ratio(pts)
                state, events = fsm.update(label, t)
                if pts is not None:
                    rx, ry = int(pts[F.MIDDLE_MCP, 0] * w), int(pts[F.MIDDLE_MCP, 1] * h)
                    fc = FC_MIN_PRECLICK if ratio < PRECLICK_RATIO else FC_MIN
                    sx, sy = int(fx.filter(rx, t, fc)), int(fy.filter(ry, t, fc))
                    hud.draw_cursor(frame, (rx, ry), (sx, sy))
                    if not fsm.cursor_frozen:
                        runner.move(sx / w, sy / h)
                runner.apply(events)
                if events:
                    last_event = ", ".join(e.kind + ("" if e.arg is None else f" {e.arg}") for e in events)
                hud.draw_region(frame, runner.active_region, runner.enabled)
                hud.draw_banner(
                    frame,
                    f"{state} | pose {label} {conf:.2f} | ratio {ratio:.2f} | {clf.name} | {fps:.0f} fps",
                    f"last: {last_event} | mouse [m] {'ON' if runner.enabled else 'OFF'} | {KEYS}",
                    runner.enabled,
                )

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                runner.toggle()
            elif key == ord("k") and mlp is not None:
                clf = rules if clf is mlp else mlp
                smoother.reset()
            elif key == ord("r") and recorder is not None:
                runner.release()
                session = None if session is not None else RecordSession()
            elif session is not None and key == ord(" "):
                session.redo(t)
            elif session is not None and ord("1") <= key <= ord("6"):
                session.select(key - ord("1"), t)
    finally:
        runner.release()
        camera.close()
        tracker.close()
        if recorder is not None:
            recorder.close()
        cv2.destroyAllWindows()
```

- [ ] **Step 2: Entry point**

`run_demo.py`:
```python
from airpoint.app import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Retire the old script and fix `run.txt`**

```bash
git rm -q airpoint_live_webcam_prototype_1_euro_filter_fsm_pinch.py
cat > run.txt <<'EOF'
# 1. Web dashboard
open index.html

# 2. Benchmarks (writes results.js for the dashboard)
venv/bin/python airpoint_offline_training_filter_fitts_law_benchmark_suite.py

# 3. Live demo (q quit | m mouse | k rules/mlp | r record | 1-6 pose | space start/redo)
venv/bin/python run_demo.py                    # MLP if models/pose_mlp.joblib exists, else rules
venv/bin/python run_demo.py --rules            # force rules
venv/bin/python run_demo.py --record --user <name>   # record poses to data/<name>.csv

# 4. Train on data/*.csv
venv/bin/python -m airpoint.train
EOF
```

- [ ] **Step 4: Full test suite**

Run: `venv/bin/pytest -q`
Expected: `33 passed`

- [ ] **Step 5: Manual check — rules mode reproduces today's demo**

Run: `venv/bin/python run_demo.py --rules`

Verify, with the video window focused:
1. Banner shows `MOVING | pose palm 1.00 | ... | rules | ~30 fps` with an open hand.
2. Quick pinch → `last: click`. Hold pinch → `PRESSING` then `DRAGGING`, open → `drag_end`.
3. Fist held → `DRAGGING` too. V-sign held 0.3 s → `last: right_click`, stays silent until palm.
4. Thumb up / thumb down → `SCROLLING`, `last: scroll 2` / `scroll -2`.
5. `m` → mouse ON: cursor follows, pinch clicks in a real window, thumb scrolls a web page.
6. `q` → exits, no stuck button.

- [ ] **Step 6: Commit**

```bash
git add airpoint/app.py run_demo.py run.txt && git commit -m "feat: run_demo wires classifier, FSM and mouse; retire the single-file prototype"
```

---

### Task 11: Record the team, train, go live on the MLP

**Files:**
- Create: `data/<name>.csv` × 4, `models/pose_mlp.joblib`, `models/labels.json`

- [ ] **Step 1: Record each teammate (~1.5 min per person, all in one room)**

For each of the four:
```bash
venv/bin/python run_demo.py --record --user madhuram   # then ishan, dakshmeet, bhuvi
```
Press `space`. Follow the prompts: 3 s get-ready, 10 s hold per pose, six poses in order (palm, pinch, fist, v_sign, thumb_up, thumb_down). During the 10 s: tilt the hand, move closer, move away, rotate slightly. Prompt says `done` → `q`. Redo one pose with `1`–`6` if the hand left the frame.

Check row counts:
```bash
for f in data/*.csv; do echo "$f $(($(wc -l < "$f") - 1))"; done
```
Expected: each file ~1500–1900 rows (6 poses × ~300 frames at ~30 fps).

- [ ] **Step 2: Train**

Run: `venv/bin/python -m airpoint.train`
Expected output shape:
```
rows 7000 | users bhuvi, dakshmeet, ishan, madhuram | fit 12.3s
leave-one-user-out accuracy: 9x.x%  bhuvi 9x%  dakshmeet 9x%  ishan 9x%  madhuram 9x%
pose         precision  recall
fist             0.9xx   0.9xx
...
saved models/pose_mlp.joblib, models/labels.json, models/train_report.json
```
If any user is below 85%: that user's recording likely has a mislabeled pose (hand left the frame, or thumb_down done with the hand upright). Re-record that user's weak pose with `1`–`6` and train again.

- [ ] **Step 3: Go live and compare**

Run: `venv/bin/python run_demo.py`

Banner shows `mlp`. Repeat the Task 10 Step 5 checklist. Then press `k` to flip to rules and back while holding a tilted V-sign or a loose fist: rules flicker between labels, MLP holds. That is the stage beat.

- [ ] **Step 4: Commit data and model**

```bash
git add data/*.csv models/pose_mlp.joblib models/labels.json && git commit -m "data: team pose recordings and first trained pose MLP"
```

---

## Done when

- `venv/bin/pytest -q` → `33 passed`.
- `run_demo.py` with no flags boots on the MLP, all six poses produce the right events, `m` drives the real mouse, `k` swaps classifiers live.
- `models/train_report.json` shows leave-one-user-out accuracy across four users.

Next plans: Week 2 (dwell mode, calibration with tremor → `fc_min`, hot-swap retrain, mini always-on-top HUD), Week 3 (hotkey poses, two-hand zoom, dashboard card).
