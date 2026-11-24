# AirPoint Week 2 — Dwell Click, Calibration, Mini HUD

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A user with no usable pinch can click by holding still; `c` teaches AirPoint a new hand in under a minute (tremor → filter cutoff, six 5-second captures, personal retrain hot-swapped live, saved as a profile); `v` shrinks the window to a 320×240 always-on-top view so the audience sees what the system reads while a real app is driven.

**Architecture:** Three additions on top of the Week 1 package. `DwellDetector` in `fsm.py` is a second pure state machine fed by the smoothed cursor and gated on the FSM being in `MOVING`. `calibration.py` holds the calibration timing machine (`CalibSession`, delegating pose capture to `RecordSession`), the tremor → `fc_min` mapping, a `Trainer` thread wrapper, and profile read/write. `train.py` gains `train_personal` (base rows + one user's calibration rows ×3, scored on a hold-out of those rows). `app.py` wires keys `d`, `c`, `v` and loads a profile on `--user`.

**Tech Stack:** Python 3.12, OpenCV 5 (`WND_PROP_TOPMOST` present), scikit-learn MLPClassifier, joblib, threading, pytest.

**Spec:** `docs/superpowers/specs/2026-11-13-airpoint-pose-classifier-design.md` §2 (dwell row), §5 (dwell paragraph), §6, §7, §8, §9. Week 3 (hotkey poses, dashboard card) is not here.

**Conventions:** project root `~/Projects/airpoint`; all paths relative to it; run with `venv/bin/python` / `venv/bin/pytest`. Commit messages carry no co-author trailer. Baseline before Task 1: `venv/bin/pytest -q` → `40 passed`.

**Spec deviations (decided here):**
- Dwell "cursor speed < 15 px/s for 1.0 s" is implemented as "smoothed cursor stays within 15 px of where it first stopped, for 1.0 s". Same intent, no derivative of a noisy signal. Pixels are frame pixels (640-wide frame), not screen pixels.
- Calibration needs `--user <name>`: the rows and the profile are per person. Pressing `c` without it prints a hint.
- Personal models are saved as `models/personal_<user>.joblib` + `models/personal_<user>_labels.json` (gitignored); the shared `models/pose_mlp.joblib` is never overwritten by calibration.

---

## File map

| File | Change | Responsibility |
|---|---|---|
| `airpoint/fsm.py` | add `DwellDetector` | still-cursor → `click`, re-arm after 30 px |
| `airpoint/hud.py` | add `draw_dwell_ring`, `mini_frame`, `MINI_SIZE` | ring that fills over the dwell second; 320×240 view |
| `airpoint/recorder.py` | `RecordSession(poses, capture_s=CAPTURE_S)` | calibration captures 5 s per pose instead of 10 |
| `airpoint/train.py` | `read_csv_rows`, skip `*_calib.csv` in `load_rows`, `FileNotFoundError`, `train_personal` | personal retrain |
| `airpoint/calibration.py` | new | `jitter_px`, `fc_min_from_jitter`, `CalibSession`, `Trainer`, `save_profile`, `load_profile` |
| `airpoint/app.py` | rewrite | keys `d c v`, `--dwell`, profile on `--user`, hot-swap |
| `tests/test_fsm.py` | +4 tests | dwell |
| `tests/test_hud.py` | new, 2 tests | ring, mini frame |
| `tests/test_recorder.py` | +1 test | `capture_s` |
| `tests/test_train.py` | +2 tests | personal model, calib files excluded from base |
| `tests/test_calibration.py` | new, 6 tests | mapping, jitter, session phases, trainer, profile |
| `.gitignore`, `README.md`, `run.txt` | docs | keys, profiles |

Expected after Task 7: `55 passed`.

---

### Task 1: DwellDetector

**Files:**
- Modify: `airpoint/fsm.py`
- Test: `tests/test_fsm.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_fsm.py`:

```python
from airpoint.fsm import DwellDetector


def dwell(seq, d=None):
    d = d or DwellDetector()
    out = []
    for xy, t, active in seq:
        e, p = d.update(xy, t, active)
        out.append((e.kind if e else None, round(p, 2)))
    return out, d


def test_dwell_clicks_after_one_second_still():
    out, _ = dwell([((100, 100), 0.0, True), ((104, 98), 0.5, True), ((102, 101), 1.0, True)])
    assert out == [(None, 0.0), (None, 0.5), ("click", 1.0)]


def test_dwell_moving_restarts_the_timer():
    out, _ = dwell([((100, 100), 0.0, True), ((140, 100), 0.5, True),
                    ((141, 100), 1.2, True), ((141, 100), 1.5, True)])
    assert out[-2:] == [(None, 0.7), ("click", 1.0)]


def test_dwell_rearms_only_after_moving_30px():
    out, _ = dwell([((100, 100), 0.0, True), ((100, 100), 1.0, True),
                    ((100, 100), 3.0, True), ((120, 100), 4.5, True),
                    ((135, 100), 5.0, True), ((135, 100), 6.0, True)])
    assert [e for e, _ in out] == [None, "click", None, None, None, "click"]


def test_dwell_inactive_resets_progress():
    out, _ = dwell([((100, 100), 0.0, True), ((100, 100), 0.8, False),
                    ((100, 100), 0.9, True), ((100, 100), 1.5, True)])
    assert out == [(None, 0.0), (None, 0.0), (None, 0.0), (None, 0.6)]
```

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/pytest tests/test_fsm.py -q`
Expected: `ImportError: cannot import name 'DwellDetector'`

- [ ] **Step 3: Implement** — in `airpoint/fsm.py`, add `import math` at the top, these constants after `SCROLL_TICKS = 2`, and the class at the end of the file:

```python
DWELL_S = 1.0           # still this long -> click
DWELL_STILL_PX = 15.0   # "still" = within this radius of where the cursor stopped
DWELL_REARM_PX = 30.0   # after a dwell click, move this far before the next one can arm
```

```python
def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class DwellDetector:
    """Pure: update(xy, t, active) -> (Event | None, progress 0..1).

    Click when the cursor stays within `still_px` of its anchor for `dwell_s` while `active`.
    Disarmed after a click until the cursor moves `rearm_px` from the click spot.
    """

    def __init__(self, dwell_s=DWELL_S, still_px=DWELL_STILL_PX, rearm_px=DWELL_REARM_PX):
        self.dwell_s, self.still_px, self.rearm_px = dwell_s, still_px, rearm_px
        self.anchor, self.t_anchor, self.click_spot = None, 0.0, None

    def reset(self):
        self.anchor, self.click_spot = None, None

    def update(self, xy, t, active):
        if not active or xy is None:
            self.anchor = None
            return None, 0.0
        if self.click_spot is not None:
            if _dist(xy, self.click_spot) < self.rearm_px:
                self.anchor = None
                return None, 0.0
            self.click_spot = None
        if self.anchor is None or _dist(xy, self.anchor) > self.still_px:
            self.anchor, self.t_anchor = xy, t
            return None, 0.0
        progress = (t - self.t_anchor) / self.dwell_s
        if progress >= 1.0:
            self.click_spot, self.anchor = xy, None
            return Event("click"), 1.0
        return None, progress
```

- [ ] **Step 4: Run to verify they pass**

Run: `venv/bin/pytest tests/test_fsm.py -q`
Expected: `13 passed`

- [ ] **Step 5: Commit**

```bash
git add airpoint/fsm.py tests/test_fsm.py
git commit -m "feat(fsm): DwellDetector — still cursor 1 s -> click, re-arms after 30 px"
```

---

### Task 2: HUD — dwell ring and mini frame

**Files:**
- Modify: `airpoint/hud.py`
- Create: `tests/test_hud.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_hud.py`:

```python
import numpy as np

from airpoint import hud


def test_mini_frame_is_320x240_with_a_label_strip():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    small = hud.mini_frame(frame, "MOVING | palm 0.98 | mouse ON")
    assert small.shape == (240, 320, 3)
    assert small[230:, :, :].any()


def test_dwell_ring_draws_only_with_progress():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    hud.draw_dwell_ring(frame, (50, 50), 0.0)
    assert not frame.any()
    hud.draw_dwell_ring(frame, (50, 50), 0.5)
    assert frame.any()
```

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/pytest tests/test_hud.py -q`
Expected: `AttributeError: module 'airpoint.hud' has no attribute 'mini_frame'`

- [ ] **Step 3: Implement** — in `airpoint/hud.py`, add after the colour constants:

```python
MINI_SIZE = (320, 240)
```

and append at the end:

```python
def draw_dwell_ring(frame, xy, progress):
    """Arc that fills clockwise from 12 o'clock as the dwell timer runs."""
    if progress <= 0.0:
        return
    cv2.ellipse(frame, xy, (22, 22), -90, 0, int(360 * min(progress, 1.0)), ACCENT_ON, 3)


def mini_frame(frame, label):
    """320x240 copy with a one-line strip at the bottom; for the always-on-top corner view."""
    small = cv2.resize(frame, MINI_SIZE, interpolation=cv2.INTER_AREA)
    w, h = MINI_SIZE
    cv2.rectangle(small, (0, h - 24), (w, h), (25, 35, 30), -1)
    cv2.putText(small, label, (8, h - 8), FONT, 0.45, MINT, 1)
    return small
```

- [ ] **Step 4: Run to verify they pass**

Run: `venv/bin/pytest tests/test_hud.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add airpoint/hud.py tests/test_hud.py
git commit -m "feat(hud): dwell ring and 320x240 mini frame"
```

---

### Task 3: RecordSession capture length

**Files:**
- Modify: `airpoint/recorder.py:71-123`
- Test: `tests/test_recorder.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_recorder.py`:

```python
def test_session_capture_length_is_configurable():
    s = RecordSession(("palm",), capture_s=5.0)
    s.start(0.0)
    assert s.update(3.0) == ("capturing", "palm", 5.0)
    assert s.update(7.9)[0] == "capturing"
    assert s.update(8.0)[0] == "done"
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/pytest tests/test_recorder.py -q`
Expected: `TypeError: RecordSession.__init__() got an unexpected keyword argument 'capture_s'`

- [ ] **Step 3: Implement** — in `RecordSession`:

```python
    def __init__(self, poses=POSES, capture_s=CAPTURE_S):
        self.poses = list(poses)
        self.capture_s = capture_s
        self.i = 0
        self.phase = "idle"
        self.t0 = 0.0
        self.single = False
        self.fresh = False
```

and in `update`, replace the two uses of `CAPTURE_S`:

```python
        elif self.phase == "capturing" and t - self.t0 >= self.capture_s:
```

```python
        total = {"countdown": COUNTDOWN_S, "capturing": self.capture_s}.get(self.phase)
```

- [ ] **Step 4: Run to verify it passes**

Run: `venv/bin/pytest tests/test_recorder.py -q`
Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add airpoint/recorder.py tests/test_recorder.py
git commit -m "feat(recorder): RecordSession takes capture_s"
```

---

### Task 4: train.py — personal retrain

**Files:**
- Modify: `airpoint/train.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Write the failing tests** — in `tests/test_train.py`, change the import line to `from airpoint.train import train_and_save, train_personal` and append:

```python
def test_personal_model_scores_calibration_holdout(tmp_path):
    rng = np.random.default_rng(2)
    data, models = tmp_path / "data", tmp_path / "models"
    data.mkdir()
    _write_user(data / "a.csv", "a", rng, 1.0, np.array([0.0, 0.0, 0.0]))
    _write_user(data / "b_calib.csv", "b", rng, 1.3, np.array([0.1, -0.05, 0.0]))

    report = train_personal(str(data), str(data / "b_calib.csv"), str(models), "b", seed=0)

    assert report["n_calib"] == 180 and report["acc_on_own_frames"] >= 0.9
    assert report["model"] == str(models / "personal_b.joblib")
    assert (models / "personal_b.joblib").exists() and (models / "personal_b_labels.json").exists()


def test_base_training_ignores_calibration_files(tmp_path):
    rng = np.random.default_rng(3)
    data, models = tmp_path / "data", tmp_path / "models"
    data.mkdir()
    _write_user(data / "a.csv", "a", rng, 1.0, np.array([0.0, 0.0, 0.0]))
    _write_user(data / "b_calib.csv", "b", rng, 1.3, np.array([0.1, -0.05, 0.0]))

    report = train_and_save(str(data), str(models), seed=0)

    assert report["users"] == ["a"] and report["n_rows"] == 180
```

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/pytest tests/test_train.py -q`
Expected: `ImportError: cannot import name 'train_personal'`

- [ ] **Step 3: Implement** — in `airpoint/train.py`, add after `AUG_COPIES = 3`:

```python
PERSONAL_WEIGHT = 3        # calibration rows duplicated this many times
CALIB_SUFFIX = "_calib.csv"
```

Replace `load_rows` with:

```python
def read_csv_rows(path):
    """One CSV -> (X, y, users, dropped). Rows whose geometry contradicts their label are skipped."""
    X, y, users, dropped = [], [], [], 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            coords = [float(row[f"{a}{i}"]) for i in range(21) for a in "xyz"]
            if not looks_like(row["pose"], coords):
                dropped += 1
                continue
            X.append(coords)
            y.append(row["pose"])
            users.append(row["user"])
    return X, y, users, dropped


def load_rows(data_dir):
    """Base dataset: data/*.csv minus *_calib.csv (one person's calibration, used by train_personal)."""
    X, y, users, dropped = [], [], [], 0
    for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        if path.endswith(CALIB_SUFFIX):
            continue
        x_, y_, u_, d_ = read_csv_rows(path)
        X += x_; y += y_; users += u_; dropped += d_
    if not X:
        raise FileNotFoundError(f"no rows in {data_dir}/*.csv — record first: run_demo.py --record --user <name>")
    if dropped:
        print(f"[train] skipped {dropped} rows whose geometry contradicts their label")
    return np.array(X), np.array(y), np.array(users)
```

Add after `train_and_save`:

```python
def train_personal(data_dir, calib_path, models_dir, user, seed=42, weight=PERSONAL_WEIGHT):
    """Base rows + this user's calibration rows x`weight` -> models/personal_<user>.joblib.
    Accuracy is on a 25% hold-out of the calibration rows, which the fit never sees."""
    rng = np.random.default_rng(seed)
    X_base, y_base, _ = load_rows(data_dir)
    Xc, yc, _, _ = read_csv_rows(calib_path)
    if len(Xc) < 60:
        raise ValueError(f"only {len(Xc)} usable calibration frames; redo calibration")
    Xc, yc = np.array(Xc), np.array(yc)
    te = rng.random(len(yc)) < 0.25
    Xn_base, Xn_c = normalize_rows(X_base), normalize_rows(Xc)
    X_tr = np.concatenate([Xn_base] + [Xn_c[~te]] * weight)
    y_tr = np.concatenate([y_base] + [yc[~te]] * weight)
    X_aug, y_aug = augment(X_tr, y_tr, AUG_COPIES, rng)

    t0 = time.perf_counter()
    model = make_model(seed).fit(X_aug, y_aug)
    fit_s = time.perf_counter() - t0
    acc = float(np.mean(model.predict(Xn_c[te]) == yc[te]))

    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"personal_{user}.joblib")
    labels_path = os.path.join(models_dir, f"personal_{user}_labels.json")
    joblib.dump(model, model_path)
    with open(labels_path, "w") as f:
        json.dump(list(model.classes_), f)
    return {"user": user, "n_calib": int(len(yc)), "acc_on_own_frames": acc,
            "fit_seconds": fit_s, "model": model_path, "labels": labels_path}
```

And in `main`, wrap the call so the CLI message stays clean:

```python
def main(argv=None):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        r = train_and_save(os.path.join(root, "data"), os.path.join(root, "models"))
    except FileNotFoundError as e:
        raise SystemExit(e)
```

(rest of `main` unchanged.)

- [ ] **Step 4: Run to verify they pass**

Run: `venv/bin/pytest tests/test_train.py -q`
Expected: `4 passed`

- [ ] **Step 5: Confirm the real retrain still works**

Run: `venv/bin/python -m airpoint.train | head -2`
Expected: `rows 1508 | users madhuram | fit ...s` (same numbers as before: calib files do not exist yet).

- [ ] **Step 6: Commit**

```bash
git add airpoint/train.py tests/test_train.py
git commit -m "feat(train): train_personal — base + one user's calibration rows x3, scored on their hold-out"
```

---

### Task 5: calibration.py — tremor mapping and session

**Files:**
- Create: `airpoint/calibration.py`
- Create: `tests/test_calibration.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_calibration.py`:

```python
from airpoint.calibration import CalibSession, fc_min_from_jitter, jitter_px


def test_fc_min_maps_jitter_linearly_and_clamps():
    assert fc_min_from_jitter(0.0) == 1.0
    assert fc_min_from_jitter(3.0) == 1.0
    assert fc_min_from_jitter(7.5) == 0.65
    assert fc_min_from_jitter(12.0) == 0.3
    assert fc_min_from_jitter(40.0) == 0.3


def test_jitter_is_spread_of_the_raw_cursor():
    assert jitter_px([(1, 1)]) == 0.0
    assert jitter_px([(0, 0), (0, 0)]) == 0.0
    assert abs(jitter_px([(0, 0), (6, 0), (0, 0), (6, 0)]) - 3.0) < 1e-9


def test_calib_session_phases():
    s = CalibSession(("palm", "pinch"), 0.0)
    assert s.update(0.0) == ("countdown", None, 3.0)
    assert s.update(3.0) == ("still", None, 5.0)
    s.still_points += [(100, 100), (104, 100), (100, 104), (104, 104)]
    assert s.update(7.9)[0] == "still"
    assert s.update(8.0) == ("poses_countdown", "palm", 3.0)
    assert s.fc_min == fc_min_from_jitter(jitter_px(s.still_points))
    assert s.update(11.0) == ("capturing", "palm", 5.0)
    assert s.capturing and s.fresh
    s.fresh = False
    assert s.update(16.0) == ("poses_countdown", "pinch", 3.0)
    assert s.update(19.0)[0] == "capturing"
    assert s.update(24.0) == ("training", None, 0.0)
    assert not s.capturing
```

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/pytest tests/test_calibration.py -q`
Expected: `ModuleNotFoundError: No module named 'airpoint.calibration'`

- [ ] **Step 3: Implement** — create `airpoint/calibration.py`:

```python
"""Calibration (`c` key): tremor -> fc_min, six short pose captures, personal retrain, profile."""
import json
import os
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
```

- [ ] **Step 4: Run to verify they pass**

Run: `venv/bin/pytest tests/test_calibration.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add airpoint/calibration.py tests/test_calibration.py
git commit -m "feat(calibration): tremor -> fc_min mapping and CalibSession timing machine"
```

---

### Task 6: calibration.py — Trainer thread and profile

**Files:**
- Modify: `airpoint/calibration.py`
- Test: `tests/test_calibration.py`

- [ ] **Step 1: Write the failing tests** — change the import in `tests/test_calibration.py` to:

```python
from airpoint.calibration import (CalibSession, Trainer, fc_min_from_jitter, jitter_px,
                                  load_profile, save_profile)
```

and append:

```python
def test_trainer_reports_result_or_error_off_thread():
    ok = Trainer(lambda: {"acc": 1.0})
    ok.start()
    ok.join(2.0)
    assert ok.done and ok.result == {"acc": 1.0} and ok.error is None

    bad = Trainer(lambda: 1 / 0)
    bad.start()
    bad.join(2.0)
    assert bad.done and bad.result is None
    assert bad.error == "ZeroDivisionError: division by zero"


def test_trainer_is_not_done_before_it_runs():
    assert not Trainer(lambda: {}).done


def test_profile_round_trip(tmp_path):
    root = str(tmp_path)
    model = str(tmp_path / "models" / "personal_ana.joblib")
    labels = str(tmp_path / "models" / "personal_ana_labels.json")
    path = save_profile(root, "ana", 0.55, model, labels, 0.97)
    assert path == str(tmp_path / "profiles" / "ana.json")
    p = load_profile(root, "ana")
    assert p["user"] == "ana" and p["fc_min"] == 0.55 and p["acc_on_own_frames"] == 0.97
    assert p["model"] == "models/personal_ana.joblib"
    assert p["labels"] == "models/personal_ana_labels.json"
    assert load_profile(root, "nobody") is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/pytest tests/test_calibration.py -q`
Expected: `ImportError: cannot import name 'Trainer'`

- [ ] **Step 3: Implement** — append to `airpoint/calibration.py`:

```python
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


def profile_path(root, user):
    return os.path.join(root, "profiles", f"{user}.json")


def save_profile(root, user, fc_min, model_path, labels_path, acc):
    """profiles/<user>.json: filter cutoff and personal model, paths relative to the project root."""
    path = profile_path(root, user)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"user": user, "fc_min": fc_min,
            "model": os.path.relpath(model_path, root), "labels": os.path.relpath(labels_path, root),
            "acc_on_own_frames": acc}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_profile(root, user):
    path = profile_path(root, user)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
```

- [ ] **Step 4: Run to verify they pass**

Run: `venv/bin/pytest tests/test_calibration.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add airpoint/calibration.py tests/test_calibration.py
git commit -m "feat(calibration): Trainer thread wrapper and per-user profile files"
```

---

### Task 7: app.py — keys d, c, v; profile on --user; hot-swap

**Files:**
- Rewrite: `airpoint/app.py`
- Modify: `.gitignore`

- [ ] **Step 1: Replace `airpoint/app.py` with:**

```python
import argparse
import os
import time

import cv2

from . import features as F
from . import hud
from .actions import ActionRunner, PyAutoGuiMouse
from .calibration import CalibSession, Trainer, load_profile, save_profile
from .camera import Camera
from .classifier import PoseClassifier, RuleClassifier, VoteSmoother
from .filters import OneEuroFilter
from .fsm import AirPointFSM, DwellDetector
from .poses import POSES
from .recorder import Recorder, RecordSession, looks_like
from .tracker import HandTracker
from .train import train_personal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR, MODELS_DIR = os.path.join(ROOT, "data"), os.path.join(ROOT, "models")
MODEL_PATH, LABELS_PATH = os.path.join(MODELS_DIR, "pose_mlp.joblib"), os.path.join(MODELS_DIR, "labels.json")
FC_MIN, FC_MIN_PRECLICK, PRECLICK_RATIO = 0.80, 0.30, 0.42
WINDOW, MINI_MARGIN = "AirPoint", 20
KEYS = "q quit | m mouse | d dwell | k rules/mlp | c calibrate | v mini | r record | 1-6 pose | space start/redo"
KEY_HINT = "keys q m d k c v r"


def load_classifiers(force_rules, profile=None):
    """-> (rules, mlp | None). A profile with an existing personal model wins over models/pose_mlp."""
    rules = RuleClassifier()
    model, labels = MODEL_PATH, LABELS_PATH
    if profile is not None and os.path.exists(os.path.join(ROOT, profile["model"])):
        model, labels = os.path.join(ROOT, profile["model"]), os.path.join(ROOT, profile["labels"])
    if force_rules or not os.path.exists(model):
        return rules, None
    mlp = PoseClassifier.load(model, labels)
    if model != MODEL_PATH:
        mlp.name = f"mlp/{profile['user']}"
    return rules, mlp


def capture(session, recorder, pose, pts, hand, t):
    """Write this frame if the session is capturing and it passes the guard. -> True if rejected."""
    if session.fresh:
        recorder.drop(pose)
        session.fresh = False
    if not session.capturing or pts is None:
        return False
    if looks_like(pose, pts):
        recorder.write(pose, t, hand, pts)
        return False
    return True


def show_window(mini, screen_w):
    """Recreate the window; mini = always on top in the top-right corner."""
    try:
        cv2.destroyWindow(WINDOW)
    except cv2.error:
        pass
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    if mini:
        try:
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            print("[AirPoint] always-on-top not supported by this OpenCV build")
        cv2.moveWindow(WINDOW, screen_w - hud.MINI_SIZE[0] - MINI_MARGIN, MINI_MARGIN)


def main(argv=None):
    ap = argparse.ArgumentParser(description="AirPoint live demo")
    ap.add_argument("--user", help="your name: loads profiles/<user>.json; enables record (r) and calibrate (c)")
    ap.add_argument("--record", action="store_true", help="start in record mode")
    ap.add_argument("--rules", action="store_true", help="force the rule classifier")
    ap.add_argument("--mouse", action="store_true", help="start with OS mouse control ON")
    ap.add_argument("--dwell", action="store_true", help="start with dwell click ON")
    args = ap.parse_args(argv)
    if args.record and not args.user:
        ap.error("--record needs --user")

    profile = load_profile(ROOT, args.user) if args.user else None
    rules, mlp = load_classifiers(args.rules, profile)
    clf = mlp or rules
    fc_min = profile["fc_min"] if profile else FC_MIN
    smoother, fsm, dwell = VoteSmoother(), AirPointFSM(), DwellDetector()
    fx, fy = OneEuroFilter(FC_MIN), OneEuroFilter(FC_MIN)
    mouse = PyAutoGuiMouse()
    runner = ActionRunner(mouse, mouse.size())
    if args.mouse:
        runner.toggle()
    dwell_on, mini = args.dwell, False
    recorder = Recorder(os.path.join(DATA_DIR, f"{args.user}.csv"), args.user) if args.record else None
    session = RecordSession() if args.record else None
    calib, calib_rec, trainer = None, None, None
    camera, tracker = Camera(), HandTracker()
    last_event, prev_t, rejected = "none", time.time(), 0
    print(f"[AirPoint] camera {camera.index} | classifier {clf.name} | fc_min {fc_min} | {KEYS}")

    try:
        for frame in camera.frames():
            t = time.time()
            fps, prev_t = 1.0 / max(t - prev_t, 1e-5), t
            h, w = frame.shape[:2]
            pts, hand = tracker.detect(frame)
            raw, mini_label = None, ""
            if pts is not None:
                hud.draw_skeleton(frame, pts)
                raw = (int(pts[F.MIDDLE_MCP, 0] * w), int(pts[F.MIDDLE_MCP, 1] * h))

            if session is not None:
                phase, pose, left = session.update(t)
                lost = capture(session, recorder, pose, pts, hand, t)
                rejected += lost
                title = {
                    "idle": f"RECORD {args.user}: SPACE = all poses, 1-6 = one pose",
                    "countdown": f"raise hand, make {pose}   {left:.0f}",
                    "capturing": f"HOLD {pose}   {left:.0f}s" + ("   << SIGN LOST" if lost else ""),
                    "done": "done - q to quit, 1-6 to redo one pose",
                }[phase]
                hud.draw_prompt(frame, title,
                                f"tilt, move closer, move away | kept {recorder.count} | rejected {rejected} | {KEY_HINT}")
            elif calib is not None:
                phase, pose, left = calib.update(t)
                if phase == "still" and raw is not None:
                    calib.still_points.append(raw)
                lost = capture(calib, calib_rec, pose, pts, hand, t)
                if phase == "training":
                    if trainer is None:
                        calib_rec.close()
                        trainer = Trainer(lambda: train_personal(DATA_DIR, calib_rec.path, MODELS_DIR, args.user))
                        trainer.start()
                    elif trainer.done:
                        if trainer.error:
                            calib.phase, calib.message = "failed", trainer.error
                        else:
                            r = trainer.result
                            mlp = clf = PoseClassifier.load(r["model"], r["labels"])
                            mlp.name = f"mlp/{args.user}"
                            smoother.reset()
                            fc_min = calib.fc_min
                            save_profile(ROOT, args.user, fc_min, r["model"], r["labels"], r["acc_on_own_frames"])
                            calib.phase = "done"
                            calib.message = f"{r['acc_on_own_frames'] * 100:.0f}% on your frames, fc_min {fc_min}"
                        trainer = None
                title = {
                    "countdown": f"CALIBRATE: open palm, hold still   {left:.0f}",
                    "still": f"HOLD STILL   {left:.0f}s",
                    "poses_countdown": f"make {pose}   {left:.0f}",
                    "capturing": f"HOLD {pose}   {left:.0f}s" + ("   << SIGN LOST" if lost else ""),
                    "training": "Learning your hand...",
                    "done": f"done - {calib.message}",
                    "failed": f"retrain failed, old model kept - {calib.message}",
                }[phase]
                hud.draw_prompt(frame, title, f"kept {calib_rec.count} | c = close | {KEY_HINT}")
            else:
                if pts is None:
                    label, conf, ratio = smoother.update(None, 0.0), 0.0, 1.0
                else:
                    raw_label, conf = clf.predict(pts)
                    label, ratio = smoother.update(raw_label, conf), F.pinch_ratio(pts)
                state, events = fsm.update(label, t)
                smooth = None
                if raw is not None:
                    fc = min(FC_MIN_PRECLICK, fc_min) if ratio < PRECLICK_RATIO else fc_min
                    smooth = (int(fx.filter(raw[0], t, fc)), int(fy.filter(raw[1], t, fc)))
                    hud.draw_cursor(frame, raw, smooth)
                    if not fsm.cursor_frozen:
                        runner.move(smooth[0] / w, smooth[1] / h)
                dwell_event, progress = dwell.update(smooth, t, dwell_on and state == "MOVING")
                if dwell_event is not None:
                    events = events + [dwell_event]
                if smooth is not None:
                    hud.draw_dwell_ring(frame, smooth, progress)
                runner.apply(events)
                if events:
                    last_event = ", ".join(e.kind + ("" if e.arg is None else f" {e.arg}") for e in events)
                hud.draw_region(frame, runner.active_region, runner.enabled)
                mini_label = f"{state} | {label} {conf:.2f} | mouse {'ON' if runner.enabled else 'OFF'}"
                if not mini:
                    hud.draw_banner(
                        frame,
                        f"{state} | pose {label} {conf:.2f} | ratio {ratio:.2f} | {clf.name} | {fps:.0f} fps",
                        f"last: {last_event} | mouse [m] {'ON' if runner.enabled else 'OFF'} | "
                        f"dwell [d] {'ON' if dwell_on else 'OFF'} | {KEY_HINT}",
                        runner.enabled,
                    )

            shown = hud.mini_frame(frame, mini_label) if mini and session is None and calib is None else frame
            cv2.imshow(WINDOW, shown)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                runner.toggle()
            elif key == ord("d"):
                dwell_on = not dwell_on
                dwell.reset()
            elif key == ord("k") and mlp is not None:
                clf = rules if clf is mlp else mlp
                smoother.reset()
            elif key == ord("v"):
                mini = not mini
                show_window(mini, runner.screen_w)
            elif key == ord("r") and args.user and calib is None:
                runner.release()
                if session is None:
                    recorder = recorder or Recorder(os.path.join(DATA_DIR, f"{args.user}.csv"), args.user)
                    session = RecordSession()
                else:
                    session = None
            elif key == ord("c") and session is None:
                if not args.user:
                    print("[AirPoint] calibration needs --user <name>")
                elif calib is None:
                    runner.release()
                    calib_rec = Recorder(os.path.join(DATA_DIR, f"{args.user}_calib.csv"), args.user)
                    calib = CalibSession(POSES, t)
                elif calib.phase != "training":
                    calib_rec.close()
                    calib, calib_rec = None, None
            elif session is not None and key == ord(" "):
                session.redo(t)
            elif session is not None and ord("1") <= key <= ord("6"):
                session.select(key - ord("1"), t)
    finally:
        runner.release()
        camera.close()
        tracker.close()
        for r in (recorder, calib_rec):
            if r is not None:
                r.close()
        cv2.destroyAllWindows()
```

- [ ] **Step 2: Ignore personal models** — append to `.gitignore`:

```
models/personal_*
```

- [ ] **Step 3: Import check and full suite**

Run: `venv/bin/python -c "import airpoint.app" && venv/bin/pytest -q`
Expected: `55 passed`

- [ ] **Step 4: Manual check — calibration end to end** (camera + Accessibility already granted)

Run: `venv/bin/python run_demo.py --user madhuram`

1. Launch line shows `classifier mlp | fc_min 0.8`.
2. Press `c`. Prompt: "CALIBRATE: open palm, hold still 3…", then "HOLD STILL 5s…", then six "make <pose> 3" + "HOLD <pose> 5s" rounds with the kept counter rising and "SIGN LOST" on drift.
3. "Learning your hand..." for a few seconds, then "done - NN% on your frames, fc_min X.XX". Banner's classifier badge reads `mlp/madhuram`.
4. Press `c` to close the prompt. Drive the cursor; all six poses still work.
5. Confirm files: `ls profiles/ models/ data/` → `profiles/madhuram.json`, `models/personal_madhuram.joblib`, `models/personal_madhuram_labels.json`, `data/madhuram_calib.csv`. `git status --short` shows none of them (ignored).
6. Quit, relaunch the same command: launch line shows `classifier mlp/madhuram | fc_min X.XX` (profile loaded).

- [ ] **Step 5: Manual check — dwell**

Still running: press `m` (mouse on), `d` (dwell on; banner shows `dwell [d] ON`). Hold the palm still over a desktop icon: ring fills over ~1 s, click fires, `last: click`. Stay still: no second click. Move away ~30 px and stop: ring fills again.

- [ ] **Step 6: Manual check — mini view**

Press `v`: window becomes 320×240 in the top-right corner with the strip `MOVING | palm 0.98 | mouse ON`. Open Chrome full-screen behind it: the mini window stays on top (if it does not, the launch log says `always-on-top not supported`; note it in the commit and keep going). Press `v` again: full window returns.

- [ ] **Step 7: Commit**

```bash
git add airpoint/app.py .gitignore
git commit -m "feat(app): d dwell click, c calibration with hot-swapped personal model + profile, v mini always-on-top view"
```

---

### Task 8: Docs and plan ticks

**Files:**
- Modify: `README.md`, `run.txt`, this plan

- [ ] **Step 1: README** — in the pose table add a row after the `thumb up / down` row:

```
| open palm, still 1 s (dwell mode, `d`) | click with no finger motion at all |
```

In the Run block, after the `--mouse` line add:

```bash
venv/bin/python run_demo.py --user you           # loads profiles/you.json once calibrated; c = calibrate (~1 min)
#   d = dwell click on/off | v = mini always-on-top view | c = hold still 5 s, 6 poses x 5 s, retrain, hot-swap
```

In Layout add:

```
profiles/        per-user calibration: filter cutoff + personal model (gitignored)
```

- [ ] **Step 2: run.txt** — replace the live-demo key line and add the profile line:

```
# 3. Live demo (q quit | m mouse | d dwell | k rules/mlp | c calibrate | v mini | r record | 1-6 pose | space start/redo)
venv/bin/python run_demo.py --user <name>      # loads profiles/<name>.json (fc_min + personal model) if calibrated
```

- [ ] **Step 3: Tick the boxes in this plan, run the suite one last time, commit, push**

```bash
venv/bin/pytest -q
git add README.md run.txt docs/superpowers/plans/2026-11-13-airpoint-week2-dwell-calibration-mini.md
git commit -m "docs: week 2 keys, calibration flow, profiles"
git push
```

---

## Self-review

- **Spec coverage.** §2 dwell row → Task 1 + 7. §5 dwell paragraph (1.0 s, 30 px re-arm, ring) → Tasks 1, 2, 7. §6 steps 1–4 (still 5 s → `fc_min` mapping 3→1.0 / 12→0.3; 6 × 5 s via recorder; base + user ×3 retrain, atomic swap, HUD "Learning your hand… done, NN%"; profile loaded by `--user`) → Tasks 3–7. §6 failure (old model kept, error shown) → Trainer.error → `failed` phase. §7 mini view 320×240 topmost corner, keys `m d k c v r q` printed on launch → Task 7. §8 mouse released on calibration start, `finally` closes everything → Task 7. §9 dwell arms/disarms tests → Task 1. Not here by design: dashboard card (Week 3).
- **Placeholders.** None; every step has its code or command.
- **Type consistency.** `DwellDetector.update(xy, t, active) -> (Event|None, float)` used identically in Task 7. `CalibSession` exposes `fresh` (get/set), `capturing`, `pose`, `phase`, `message`, `fc_min`, `still_points` — all used in Task 7's `capture()` and calib branch. `train_personal` returns keys `model`, `labels`, `acc_on_own_frames` consumed in Task 7. `save_profile(root, user, fc_min, model_path, labels_path, acc)` / `load_profile(root, user)` → `profile["model"]`, `["labels"]`, `["fc_min"]`, `["user"]` used in `load_classifiers`.
