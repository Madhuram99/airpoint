# AirPoint Stage 1 — Tremor Switch, Tremor Canceller, X-ray View: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A steady-handed presenter presses `t` and the real cursor shakes like a tremor hand; AirPoint measures the shake, locks onto its frequency in about two seconds and removes it; `b` compares the filters on the real cursor and `x` opens a panel that shows the system thinking. Underneath, the camera runs at a pinned 30 fps and every pixel threshold lives in one 640-wide space.

**Architecture:** `app.py` shrinks to camera, tracker, window and the loop. A `Session` holds the shared state and runs each frame through one mode object (`live`, `record`, `calibrate`). A `PointerPipeline` owns the pointer chain: tremor meter → notch canceller at the locked frequency → the existing 1-Euro filter. `tremor.py` holds the simulator, the meter and the canceller as pure, clock-as-parameter classes. The HUD draws on a 640-wide copy of the frame; the tracker still reads the native frame.

**Tech Stack:** Python 3.12, OpenCV 5, MediaPipe, NumPy, SciPy (`scipy.signal`, already installed with scikit-learn), scikit-learn, pytest.

**Spec:** `docs/superpowers/specs/2026-11-25-airpoint-stage1-tremor-xray-design.md`. Roadmap: `docs/superpowers/specs/2026-11-25-airpoint-v2-roadmap.md`, Stage 1.

**Conventions:**
- Project root `~/Projects/airpoint`; every path is relative to it; run with `venv/bin/python` and `venv/bin/pytest`.
- Commit messages carry no co-author trailer.
- Baseline before Task 1: `venv/bin/pytest -q` → `55 passed`.
- Three kinds of code block. **Create** and **Replace** blocks are whole files. **Apply to** blocks are unified diffs made by `git diff`: save the block to a file and run `git apply <file>` from the project root (or make the same edit by hand; the context lines show exactly where).
- Every block in this plan was built and run on a scratch copy of the repo at commit `9f4b4ea`. The test counts and failure messages quoted are the ones observed there.
- From Task 4 on, "px" means pixels of the 640-wide canonical view.

**Deviations from the spec (decided here; Task 10 edits the spec to match):**
- "The full-view window becomes resizable" is built as a zoom key: `z` cycles 1×, 1.5×, 2×, default 1.5×. Dragging an OpenCV window on macOS scales unpredictably; a zoom step is deterministic and testable.
- Simulator amplitude defaults to 0.20 palm lengths (spec: 0.15) with range 0.05–0.60 (spec: 0.05–0.45). 0.20 is the "moderate" evidence row at a typical palm size, and 0.60 reaches "very severe".
- The meter's peak rule is 8× the band median (spec: 6×). On the prototype, 6× falsely locked about 2% of the time on noisier landmarks; 8× gave none, with the same lock-on time.
- Record and calibrate prompts move from a centre strip to a top strip. On the 640 canvas the centre strip covered the hand the user is trying to hold in view.
- Added `tests/test_app.py`: `app.main` run end to end with the camera, tracker, OS mouse and window faked.

## File map

| File | Change | Responsibility |
|---|---|---|
| `airpoint/camera.py` | modify | asks for 30 fps; exposes `size`, `fps` |
| `airpoint/classifier.py` | modify | `proba()` on both classifiers |
| `airpoint/app.py` | rewrite | args, camera, tracker, window, the loop |
| `airpoint/session.py` | new | `Session`: shared state, `tick`, global keys, mode switching, hot-swap, X-ray taps |
| `airpoint/pointer.py` | new | `PointerPipeline`: canceller → 1-Euro, filter mode, meters, trail |
| `airpoint/tremor.py` | new | `TremorSim`, `estimate`, `TremorMeter`, `TremorCanceller` |
| `airpoint/modes/` | new | `live.py`, `record.py`, `calibrate.py`: one object per screen |
| `airpoint/recorder.py` | modify | gains `capture()` (shared by record and calibrate) |
| `airpoint/hud.py` | modify | canonical view, fitted text, 640 layout, zoom, X-ray panel |
| `tests/test_session.py` | new | whole flows through `Session`, no camera |
| `tests/pointer_sim.py` | new | synthetic pointer sessions and the two scores the evidence uses |
| `tests/test_tremor.py`, `tests/test_pointer.py`, `tests/test_app.py` | new | simulator, meter, canceller, pipeline, loop wiring |
| `tests/test_classifier.py`, `tests/test_hud.py` | modify | `proba`, canonical view, fitted text, X-ray |
| `README.md`, `run.txt`, the Stage 1 spec | docs | keys, the tremor section, spec matched to the build |

Expected after Task 9: `106 passed`.

---

### Task 1: Camera asks for 30 fps and reports what it got

The MacBook camera idles at 15 fps in dim light unless a rate is requested; asked for 30 it delivers 29.8 in the same light (measured 2026-11-25). `size` and `fps` feed the launch line in Task 3.

**Files:**
- Modify: `airpoint/camera.py`

- [x] **Step 1: Implement**

**Apply to `airpoint/camera.py`:**

````diff
diff --git a/airpoint/camera.py b/airpoint/camera.py
index db8dd70..88fd839 100644
--- a/airpoint/camera.py
+++ b/airpoint/camera.py
@@ -4,12 +4,18 @@ import cv2
 class Camera:
     """Index 1 is the MacBook camera when an iPhone takes index 0; otherwise 0."""
 
+    FPS = 30  # macOS lets the camera idle at 15 fps in dim light unless a rate is requested
+
     def __init__(self, indices=(1, 0)):
         for i in indices:
             cap = cv2.VideoCapture(i)
-            if cap.isOpened() and cap.read()[0]:
-                self.cap, self.index = cap, i
-                return
+            if cap.isOpened():
+                cap.set(cv2.CAP_PROP_FPS, self.FPS)
+                if cap.read()[0]:
+                    self.cap, self.index = cap, i
+                    self.size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
+                    self.fps = cap.get(cv2.CAP_PROP_FPS)
+                    return
             cap.release()
         raise RuntimeError("Cannot open webcam. System Settings > Privacy & Security > Camera.")
 
````

- [x] **Step 2: Check on the real camera** (turns the camera on for about 2 seconds, stores nothing)

Run:
```bash
venv/bin/python -c "
from airpoint.camera import Camera
c = Camera(); print(c.index, c.size, c.fps); c.close()"
```
Expected on this MacBook: `0 (1920, 1080) 30.0` (OpenCV may first print `out device of bound (0-0): 1`; that is
the probe for a second camera and is harmless).

- [x] **Step 3: Suite still green**

Run: `venv/bin/pytest -q`
Expected: `55 passed`

- [x] **Step 4: Commit**

```bash
git add airpoint/camera.py
git commit -m "fix(camera): ask the driver for 30 fps; expose native size and rate"
```

---

### Task 2: Class probabilities from both classifiers

The X-ray panel (Task 9) draws one bar per class. `predict` keeps its contract and is now derived from `proba`.

**Files:**
- Modify: `airpoint/classifier.py`
- Modify: `tests/test_classifier.py`

- [x] **Step 1: Write the failing tests**

**Apply to `tests/test_classifier.py`:**

````diff
diff --git a/tests/test_classifier.py b/tests/test_classifier.py
index fc9f42c..99dc8e7 100644
--- a/tests/test_classifier.py
+++ b/tests/test_classifier.py
@@ -60,3 +60,18 @@ def test_smoother_no_hand_is_none_and_clears_history():
         s.update("palm", 0.9)
     assert s.update(None, 0.0) == NONE
     assert s.update("pinch", 0.9) == "pinch"
+
+
+def test_proba_is_in_label_order_and_agrees_with_predict():
+    clf = PoseClassifier(_Model(), ["fist", "palm", "pinch"])
+    p = clf.proba(BY_POSE["pinch"])
+    assert p.shape == (3,) and abs(p.sum() - 1.0) < 1e-9
+    assert clf.labels[int(np.argmax(p))] == clf.predict(BY_POSE["pinch"])[0]
+
+
+def test_rules_proba_is_one_hot_over_the_six_poses():
+    clf = RuleClassifier()
+    assert clf.labels == ["palm", "pinch", "fist", "v_sign", "thumb_up", "thumb_down"]
+    for pose, pts in BY_POSE.items():
+        p = clf.proba(pts)
+        assert p.sum() == 1.0 and clf.labels[int(np.argmax(p))] == pose
````

- [x] **Step 2: Run to verify they fail**

Run: `venv/bin/pytest tests/test_classifier.py -q`
Expected: `2 failed, 7 passed` with `AttributeError: 'PoseClassifier' object has no attribute 'proba'` and `AttributeError: 'RuleClassifier' object has no attribute 'labels'`

- [x] **Step 3: Implement**

**Apply to `airpoint/classifier.py`:**

````diff
diff --git a/airpoint/classifier.py b/airpoint/classifier.py
index 28c96ca..0a823ca 100644
--- a/airpoint/classifier.py
+++ b/airpoint/classifier.py
@@ -5,13 +5,20 @@ from pathlib import Path
 import numpy as np
 
 from . import features as F
-from .poses import FIST, NONE, PALM, PINCH, THUMB_DOWN, THUMB_UP, UNKNOWN, V_SIGN
+from .poses import FIST, NONE, PALM, PINCH, POSES, THUMB_DOWN, THUMB_UP, UNKNOWN, V_SIGN
 
 
 class RuleClassifier:
     """Scale-free 2-D ratio rules. Zero dependencies; the fallback and the on-stage baseline."""
 
     name = "rules"
+    labels = list(POSES)
+
+    def proba(self, landmarks):
+        """One-hot over `labels`: rules have no confidence."""
+        p = np.zeros(len(self.labels))
+        p[self.labels.index(self.predict(landmarks)[0])] = 1.0
+        return p
 
     def predict(self, landmarks):
         p = F.to_array(landmarks)
@@ -47,8 +54,12 @@ class PoseClassifier:
         import joblib
         return cls(joblib.load(model_path), json.loads(Path(labels_path).read_text()))
 
+    def proba(self, landmarks):
+        """Class probabilities in `labels` order."""
+        return self.model.predict_proba(F.normalize_63(landmarks).reshape(1, -1))[0]
+
     def predict(self, landmarks):
-        proba = self.model.predict_proba(F.normalize_63(landmarks).reshape(1, -1))[0]
+        proba = self.proba(landmarks)
         i = int(np.argmax(proba))
         return self.labels[i], float(proba[i])
 
````

- [x] **Step 4: Run to verify they pass, then the full suite**

Run: `venv/bin/pytest tests/test_classifier.py -q && venv/bin/pytest -q`
Expected: the full suite ends with `57 passed`

- [x] **Step 5: Commit**

```bash
git add airpoint/classifier.py tests/test_classifier.py
git commit -m "feat(classifier): proba() on both classifiers; predict derives from it"
```

---

### Task 3: Frame loop in parts: Session, PointerPipeline, one object per mode

A pure move: behaviour is unchanged, including integer pointer pixels and drawing on the native frame (Task 4 changes those). `app.py` keeps only the camera, the tracker, the window and the loop. `Session` receives its parts, so the new tests drive whole flows with a `FakeMouse`, a blank image and the synthetic hands. `capture()` moves into `recorder.py` because both record and calibrate use it.

**Files:**
- Create: `airpoint/pointer.py`
- Modify: `airpoint/recorder.py`
- Modify: `airpoint/hud.py`
- Create: `airpoint/modes/live.py`
- Create: `airpoint/modes/record.py`
- Create: `airpoint/modes/calibrate.py`
- Create: `airpoint/modes/__init__.py`
- Create: `airpoint/session.py`
- Modify: `airpoint/app.py`
- Create: `tests/test_session.py`

- [x] **Step 1: Write the failing tests**

**Create `tests/test_session.py`:**

````python
"""Whole flows through Session with synthetic hands, a FakeMouse and a blank image. No camera."""
import csv
import json
import time

import numpy as np

from airpoint.actions import ActionRunner, FakeMouse
from airpoint.classifier import RuleClassifier
from airpoint.session import Session
from tests.hands import BY_POSE
from tests.test_train import _write_user

DT = 1 / 30


def make(tmp_path, user="ana", mouse_on=True, **kw):
    mouse = FakeMouse()
    runner = ActionRunner(mouse, mouse.size())
    if mouse_on:
        runner.toggle()
    return Session(runner, RuleClassifier(), None, root=str(tmp_path), user=user, **kw), mouse


def blank(w=640, h=360):
    return np.zeros((h, w, 3), dtype=np.uint8)


def feed(s, pose, seconds, t, dt=DT):
    """Tick with one synthetic pose (None = no hand) for `seconds`. -> the time after the last tick."""
    for _ in range(round(seconds / dt)):
        s.tick(blank(), None if pose is None else BY_POSE[pose], "Right" if pose else None, t)
        t += dt
    return t


def kinds(mouse):
    return [c[0] for c in mouse.calls if c[0] != "move"]


def test_short_pinch_is_one_click_through_the_whole_pipeline(tmp_path):
    s, mouse = make(tmp_path)
    t = feed(s, "palm", 0.5, 0.0)
    t = feed(s, "pinch", 0.2, t)
    feed(s, "palm", 0.3, t)
    assert kinds(mouse) == ["click"]
    assert any(c[0] == "move" for c in mouse.calls)


def test_mouse_off_means_no_os_calls(tmp_path):
    s, mouse = make(tmp_path, mouse_on=False)
    t = feed(s, "palm", 0.5, 0.0)
    t = feed(s, "pinch", 0.2, t)
    feed(s, "palm", 0.3, t)
    assert mouse.calls == []


def test_held_pinch_drags_and_leaving_live_mode_releases_the_button(tmp_path):
    s, mouse = make(tmp_path)
    t = feed(s, "palm", 0.5, 0.0)
    t = feed(s, "pinch", 0.8, t)
    assert kinds(mouse) == ["down"]
    s.on_key(ord("r"), t)
    assert s.mode.name == "record" and kinds(mouse) == ["down", "up"]


def test_dwell_clicks_when_the_palm_stays_still(tmp_path):
    s, mouse = make(tmp_path, dwell_on=True)
    feed(s, "palm", 1.5, 0.0)
    assert kinds(mouse) == ["click"]
    assert s.last_event == "click"


def test_dwell_key_toggles(tmp_path):
    s, mouse = make(tmp_path)
    s.on_key(ord("d"), 0.0)
    feed(s, "palm", 1.5, 0.0)
    assert kinds(mouse) == ["click"]


def test_hand_lost_mid_drag_releases(tmp_path):
    s, mouse = make(tmp_path)
    t = feed(s, "palm", 0.5, 0.0)
    t = feed(s, "pinch", 0.8, t)
    feed(s, None, 0.3, t)
    assert kinds(mouse) == ["down", "up"]


def test_mini_view_is_320x240_in_live_mode_only(tmp_path):
    s, _ = make(tmp_path)
    s.on_key(ord("v"), 0.0)
    assert s.view_dirty
    assert s.tick(blank(), BY_POSE["palm"], "Right", 0.0).shape == (240, 320, 3)
    s.on_key(ord("r"), 0.1)
    assert s.tick(blank(), BY_POSE["palm"], "Right", 0.2).shape == (360, 640, 3)


class _Mlp:
    name, labels = "mlp", ["palm"]

    def predict(self, pts):
        return "palm", 1.0


def test_k_switches_classifier_and_clears_the_vote_buffer(tmp_path):
    mouse = FakeMouse()
    mlp = _Mlp()
    s = Session(ActionRunner(mouse, mouse.size()), RuleClassifier(), mlp, root=str(tmp_path))
    assert s.clf is mlp
    feed(s, "palm", 0.2, 0.0)
    s.on_key(ord("k"), 0.3)
    assert s.clf is s.rules and len(s.smoother.buf) == 0
    s.on_key(ord("k"), 0.4)
    assert s.clf is mlp


def test_record_and_calibrate_need_a_user(tmp_path, capsys):
    s, _ = make(tmp_path, user=None)
    s.on_key(ord("r"), 0.0)
    s.on_key(ord("c"), 0.0)
    assert s.mode.name == "live"
    assert "needs --user" in capsys.readouterr().out


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_record_mode_writes_rows_and_the_guard_rejects_the_wrong_sign(tmp_path):
    s, _ = make(tmp_path)
    s.on_key(ord("r"), 0.0)
    s.on_key(ord(" "), 0.0)
    t = 0.0
    while s.mode.session.phase != "done":
        s.tick(blank(), BY_POSE[s.mode.session.pose], "Right", t)
        t += DT
    path = tmp_path / "data" / "ana.csv"
    s.recorder.f.flush()                      # rows are buffered until the file is closed
    by_pose = {p: sum(r["pose"] == p for r in _rows(path)) for p in BY_POSE}
    assert all(295 <= n <= 302 for n in by_pose.values()), by_pose
    assert s.rejected == 0 and s.recorder.count == sum(by_pose.values())

    s.on_key(ord("1"), t)                     # redo palm, but show a fist the whole time
    while s.mode.session.phase != "done":
        s.tick(blank(), BY_POSE["fist"], "Right", t)
        t += DT
    assert s.rejected >= 290
    assert sum(r["pose"] == "palm" for r in _rows(path)) == 0   # old palm rows dropped, none accepted

    s.on_key(ord("r"), t)
    assert s.mode.name == "live"
    s.close()


def test_calibration_measures_trains_hot_swaps_and_saves_a_profile(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write_user(data / "base.csv", "base", np.random.default_rng(0), 1.0, np.array([0.0, 0.0, 0.0]))
    s, _ = make(tmp_path, user="ana")
    s.on_key(ord("c"), 0.0)
    assert s.mode.name == "calibrate"
    t, dt = 0.0, 0.1
    while s.mode.calib.phase != "training":
        pose = s.mode.calib.pose if s.mode.calib.phase == "poses" else "palm"
        s.tick(blank(), BY_POSE[pose], "Right", t)
        t += dt
    s.on_key(ord("c"), t)                     # ignored while training
    assert s.mode.name == "calibrate"
    deadline = time.time() + 60
    while s.mode.calib.phase in ("training",) and time.time() < deadline:
        s.tick(blank(), BY_POSE["palm"], "Right", t)
        t += dt
        time.sleep(0.02)
    assert s.mode.calib.phase == "done", s.mode.calib.message
    assert s.clf.name == "mlp/ana" and s.clf is s.mlp
    assert s.pointer.fc_min == 1.0            # a perfectly still synthetic hand
    profile = json.loads((tmp_path / "profiles" / "ana.json").read_text())
    assert profile["fc_min"] == 1.0 and profile["model"] == "models/personal_ana.joblib"
    assert (tmp_path / "models" / "personal_ana.joblib").exists()
    s.on_key(ord("c"), t)
    assert s.mode.name == "live"
    s.close()


def test_failed_retrain_keeps_the_old_model(tmp_path):
    def boom(*a):
        raise ValueError("only 3 usable calibration frames; redo calibration")

    s, _ = make(tmp_path, user="ana", train_fn=boom)
    old = s.clf
    s.on_key(ord("c"), 0.0)
    t = 0.0
    while s.mode.calib.phase not in ("done", "failed") and t < 400:
        pose = s.mode.calib.pose if s.mode.calib.phase == "poses" else "palm"
        s.tick(blank(), BY_POSE[pose], "Right", t)
        t += 0.1
        if s.mode.calib.phase == "training":
            time.sleep(0.01)
    assert s.mode.calib.phase == "failed" and "only 3 usable" in s.mode.calib.message
    assert s.clf is old
    s.close()
````

- [x] **Step 2: Run to verify they fail**

Run: `venv/bin/pytest tests/test_session.py -q`
Expected: `1 error` with `ModuleNotFoundError: No module named 'airpoint.session'`

- [x] **Step 3: Implement**

**Create `airpoint/pointer.py`:**

````python
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
````

**Apply to `airpoint/recorder.py`:**

````diff
diff --git a/airpoint/recorder.py b/airpoint/recorder.py
index c8416a0..2e49a96 100644
--- a/airpoint/recorder.py
+++ b/airpoint/recorder.py
@@ -122,3 +122,17 @@ class RecordSession:
         total = {"countdown": COUNTDOWN_S, "capturing": self.capture_s}.get(self.phase)
         left = max(total - (t - self.t0), 0.0) if total else 0.0
         return self.phase, self.pose, left
+
+
+def capture(session, recorder, pose, pts, hand, t):
+    """Write this frame if the session is capturing and it passes the guard. -> True if rejected.
+    `session` is anything with `fresh` and `capturing` (RecordSession, CalibSession)."""
+    if session.fresh:
+        recorder.drop(pose)
+        session.fresh = False
+    if not session.capturing or pts is None:
+        return False
+    if looks_like(pose, pts):
+        recorder.write(pose, t, hand, pts)
+        return False
+    return True
````

**Apply to `airpoint/hud.py`:**

````diff
diff --git a/airpoint/hud.py b/airpoint/hud.py
index a8b03a2..2cbeadb 100644
--- a/airpoint/hud.py
+++ b/airpoint/hud.py
@@ -12,6 +12,7 @@ BANNER_H = 82
 FONT = cv2.FONT_HERSHEY_SIMPLEX
 ACCENT_ON, ACCENT_OFF, WHITE, MINT = (0, 200, 255), (90, 90, 90), (255, 255, 255), (167, 243, 208)
 MINI_SIZE = (320, 240)
+KEY_HINT = "keys q m d k c v r"
 
 
 def draw_skeleton(frame, pts):
````

**Create `airpoint/modes/live.py`:**

````python
from .. import features as F
from .. import hud
from ..pointer import PRECLICK_RATIO


class LiveMode:
    """Classify -> vote -> state machine -> pointer -> dwell -> OS."""

    name = "live"

    def tick(self, s, o, frame):
        if o.pts is None:
            label, conf, ratio = s.smoother.update(None, 0.0), 0.0, 1.0
        else:
            raw_label, conf = s.clf.predict(o.pts)
            label, ratio = s.smoother.update(raw_label, conf), F.pinch_ratio(o.pts)
        state, events = s.fsm.update(label, o.t)
        smooth = None
        if o.raw is not None:
            smooth = s.pointer.update(o.raw, o.t, preclick=ratio < PRECLICK_RATIO)
            hud.draw_cursor(frame, o.raw, smooth)
            if not s.fsm.cursor_frozen:
                s.runner.move(smooth[0] / o.w, smooth[1] / o.h)
        dwell_event, progress = s.dwell.update(smooth, o.t, s.dwell_on and state == "MOVING")
        if dwell_event is not None:
            events = events + [dwell_event]
        if smooth is not None:
            hud.draw_dwell_ring(frame, smooth, progress)
        s.runner.apply(events)
        if events:
            s.last_event = ", ".join(e.kind + ("" if e.arg is None else f" {e.arg}") for e in events)
        hud.draw_region(frame, s.runner.active_region, s.runner.enabled)
        s.mini_label = f"{state} | {label} {conf:.2f} | mouse {'ON' if s.runner.enabled else 'OFF'}"
        if not s.mini:
            hud.draw_banner(
                frame,
                f"{state} | pose {label} {conf:.2f} | ratio {ratio:.2f} | {s.clf.name} | {o.fps:.0f} fps",
                f"last: {s.last_event} | mouse [m] {'ON' if s.runner.enabled else 'OFF'} | "
                f"dwell [d] {'ON' if s.dwell_on else 'OFF'} | {hud.KEY_HINT}",
                s.runner.enabled,
            )

    def on_key(self, s, key, t):
        return False

    def close(self, s):
        pass
````

**Create `airpoint/modes/record.py`:**

````python
import os

from .. import hud
from ..recorder import Recorder, RecordSession, capture


class RecordMode:
    """Prompted captures into data/<user>.csv. SPACE = all poses, 1-6 = one pose."""

    name = "record"

    def __init__(self, s):
        s.recorder = s.recorder or Recorder(os.path.join(s.data_dir, f"{s.user}.csv"), s.user)
        self.session = RecordSession()

    def tick(self, s, o, frame):
        phase, pose, left = self.session.update(o.t)
        lost = capture(self.session, s.recorder, pose, o.pts, o.hand, o.t)
        s.rejected += lost
        title = {
            "idle": f"RECORD {s.user}: SPACE = all poses, 1-6 = one pose",
            "countdown": f"raise hand, make {pose}   {left:.0f}",
            "capturing": f"HOLD {pose}   {left:.0f}s" + ("   << SIGN LOST" if lost else ""),
            "done": "done - q to quit, 1-6 to redo one pose",
        }[phase]
        hud.draw_prompt(frame, title, f"tilt, move closer, move away | kept {s.recorder.count} | "
                                      f"rejected {s.rejected} | {hud.KEY_HINT}")

    def on_key(self, s, key, t):
        if key == ord(" "):
            self.session.redo(t)
            return True
        if ord("1") <= key <= ord("6"):
            self.session.select(key - ord("1"), t)
            return True
        return False

    def close(self, s):
        pass
````

**Create `airpoint/modes/calibrate.py`:**

````python
import os

from .. import hud
from ..calibration import CalibSession, Trainer, save_profile
from ..poses import POSES
from ..recorder import Recorder, capture


class CalibrateMode:
    """`c`: hold still -> six short captures -> personal retrain off-thread -> hot-swap -> profile."""

    name = "calibrate"

    def __init__(self, s, t):
        self.rec = Recorder(os.path.join(s.data_dir, f"{s.user}_calib.csv"), s.user)
        self.calib = CalibSession(POSES, t)
        self.trainer = None

    @property
    def busy(self):
        return self.calib.phase == "training"

    def tick(self, s, o, frame):
        calib = self.calib
        phase, pose, left = calib.update(o.t)
        if phase == "still" and o.raw is not None:
            calib.still_points.append(o.raw)
        lost = capture(calib, self.rec, pose, o.pts, o.hand, o.t)
        if phase == "training":
            if self.trainer is None:
                self.rec.close()
                path = self.rec.path
                self.trainer = Trainer(lambda: s.train_fn(s.data_dir, path, s.models_dir, s.user))
                self.trainer.start()
            elif self.trainer.done:
                self._finish(s)
        title = {
            "countdown": f"CALIBRATE: open palm, hold still   {left:.0f}",
            "still": f"HOLD STILL   {left:.0f}s",
            "poses_countdown": f"make {pose}   {left:.0f}",
            "capturing": f"HOLD {pose}   {left:.0f}s" + ("   << SIGN LOST" if lost else ""),
            "training": "Learning your hand...",
            "done": f"done - {calib.message}",
            "failed": f"retrain failed, old model kept - {calib.message}",
        }[phase]
        hud.draw_prompt(frame, title, f"kept {self.rec.count} | c = close | {hud.KEY_HINT}")

    def _finish(self, s):
        calib, trainer, self.trainer = self.calib, self.trainer, None
        if trainer.error:
            calib.phase, calib.message = "failed", trainer.error
            return
        r = trainer.result
        s.use_model(r["model"], r["labels"])
        s.pointer.fc_min = calib.fc_min
        save_profile(s.root, s.user, calib.fc_min, r["model"], r["labels"], r["acc_on_own_frames"])
        calib.phase = "done"
        calib.message = f"{r['acc_on_own_frames'] * 100:.0f}% on your frames, fc_min {calib.fc_min}"

    def on_key(self, s, key, t):
        return False

    def close(self, s):
        self.rec.close()
````

**Create `airpoint/modes/__init__.py`:**

````python
"""One object per screen the app can be in. Each has tick(session, obs, frame), on_key, close."""
from .calibrate import CalibrateMode
from .live import LiveMode
from .record import RecordMode

__all__ = ["CalibrateMode", "LiveMode", "RecordMode"]
````

**Create `airpoint/session.py`:**

````python
import os
from dataclasses import dataclass

from . import features as F
from . import hud
from .classifier import PoseClassifier, VoteSmoother
from .fsm import AirPointFSM, DwellDetector
from .modes import CalibrateMode, LiveMode, RecordMode
from .pointer import FC_MIN, PointerPipeline
from .train import train_personal


@dataclass
class Obs:
    """What one frame gave us."""
    t: float
    pts: object    # (21, 3) landmarks in normalized image coords, or None
    hand: object   # "Left" | "Right" | None
    raw: object    # pointer (x, y) in frame pixels, or None
    w: int
    h: int
    fps: float


class Session:
    """Everything the frame loop shares, minus camera, tracker and window (those stay in app.py).

    tick() runs one frame through the current mode and returns the image to show.
    on_key() handles a key: the mode gets it first, then the global keys.
    """

    def __init__(self, runner, rules, mlp, *, root, user=None, fc_min=FC_MIN, dwell_on=False,
                 train_fn=train_personal):
        self.runner, self.rules, self.mlp = runner, rules, mlp
        self.clf = mlp or rules
        self.root, self.user, self.train_fn = root, user, train_fn
        self.data_dir, self.models_dir = os.path.join(root, "data"), os.path.join(root, "models")
        self.smoother, self.fsm, self.dwell = VoteSmoother(), AirPointFSM(), DwellDetector()
        self.pointer = PointerPipeline(fc_min)
        self.dwell_on, self.mini, self.view_dirty = dwell_on, False, False
        self.recorder, self.rejected = None, 0
        self.mode = LiveMode()
        self.last_event, self.mini_label, self.prev_t = "none", "", None

    def tick(self, frame, pts, hand, t):
        fps = 0.0 if self.prev_t is None else 1.0 / max(t - self.prev_t, 1e-5)
        self.prev_t = t
        h, w = frame.shape[:2]
        raw = None
        if pts is not None:
            hud.draw_skeleton(frame, pts)
            raw = (int(pts[F.MIDDLE_MCP, 0] * w), int(pts[F.MIDDLE_MCP, 1] * h))
        self.mode.tick(self, Obs(t, pts, hand, raw, w, h, fps), frame)
        return hud.mini_frame(frame, self.mini_label) if self.mini and self.mode.name == "live" else frame

    def set_mode(self, mode):
        self.mode.close(self)
        self.mode = mode

    def use_model(self, model_path, labels_path):
        """Hot-swap to a personal model. Runs on the frame-loop thread."""
        self.mlp = self.clf = PoseClassifier.load(model_path, labels_path)
        self.mlp.name = f"mlp/{self.user}"
        self.smoother.reset()

    def on_key(self, key, t):
        if self.mode.on_key(self, key, t):
            return
        if key == ord("m"):
            self.runner.toggle()
        elif key == ord("d"):
            self.dwell_on = not self.dwell_on
            self.dwell.reset()
        elif key == ord("k") and self.mlp is not None:
            self.clf = self.rules if self.clf is self.mlp else self.mlp
            self.smoother.reset()
        elif key == ord("v"):
            self.mini, self.view_dirty = not self.mini, True
        elif key == ord("r") and self.user and self.mode.name != "calibrate":
            self.runner.release()
            self.set_mode(RecordMode(self) if self.mode.name == "live" else LiveMode())
        elif key == ord("c") and self.mode.name != "record":
            if not self.user:
                print("[AirPoint] calibration needs --user <name>")
            elif self.mode.name == "live":
                self.runner.release()
                self.set_mode(CalibrateMode(self, t))
            elif not self.mode.busy:
                self.set_mode(LiveMode())

    def close(self):
        self.runner.release()
        self.mode.close(self)
        if self.recorder is not None:
            self.recorder.close()
````

**Replace `airpoint/app.py` with:**

````python
import argparse
import os
import time

import cv2

from . import hud
from .actions import ActionRunner, PyAutoGuiMouse
from .calibration import load_profile
from .camera import Camera
from .classifier import PoseClassifier, RuleClassifier
from .modes import RecordMode
from .pointer import FC_MIN
from .session import Session
from .tracker import HandTracker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
MODEL_PATH, LABELS_PATH = os.path.join(MODELS_DIR, "pose_mlp.joblib"), os.path.join(MODELS_DIR, "labels.json")
WINDOW, MINI_MARGIN = "AirPoint", 20
KEYS = "q quit | m mouse | d dwell | k rules/mlp | c calibrate | v mini | r record | 1-6 pose | space start/redo"


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
    mouse = PyAutoGuiMouse()
    runner = ActionRunner(mouse, mouse.size())
    if args.mouse:
        runner.toggle()
    s = Session(runner, rules, mlp, root=ROOT, user=args.user,
                fc_min=profile["fc_min"] if profile else FC_MIN, dwell_on=args.dwell)
    if args.record:
        s.set_mode(RecordMode(s))
    camera, tracker = Camera(), HandTracker()
    print(f"[AirPoint] camera {camera.index} {camera.size[0]}x{camera.size[1]} @ {camera.fps:.0f} fps | "
          f"classifier {s.clf.name} | fc_min {s.pointer.fc_min} | {KEYS}")

    try:
        for frame in camera.frames():
            t = time.time()
            pts, hand = tracker.detect(frame)
            shown = s.tick(frame, pts, hand, t)
            if s.view_dirty:
                show_window(s.mini, runner.screen_w)
                s.view_dirty = False
            cv2.imshow(WINDOW, shown)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key != 255:
                s.on_key(key, t)
    finally:
        s.close()
        camera.close()
        tracker.close()
        cv2.destroyAllWindows()
````

- [x] **Step 4: Run to verify they pass**

Run: `venv/bin/pytest tests/test_session.py -q`
Expected: `12 passed`

- [x] **Step 5: Import check and full suite**

Run: `venv/bin/python -c "import airpoint.app" && venv/bin/pytest -q`
Expected: `69 passed`

- [x] **Step 6: Commit**

```bash
git add airpoint/app.py airpoint/hud.py airpoint/modes airpoint/pointer.py airpoint/recorder.py airpoint/session.py tests/test_session.py
git commit -m "refactor(app): frame loop split into Session, PointerPipeline and one object per mode; headless flow tests"
```

---

### Task 4: One 640-wide space: pointer maths, thresholds and drawing

The camera delivers 1920 x 1080, so Week 2's pixel thresholds (written for a 640-wide frame) are three times too strict as shipped. The tracker keeps reading the native frame; everything after it now works on `hud.canon_view(frame)`. The pointer stays in floats (rounding to canonical pixels would step the cursor about 3 screen points at a time) and `BETA` becomes 0.030, which reproduces the Week 1 feel exactly (0.010 x 1920 / 640). HUD text shrinks to fit instead of running off the edge, prompts move to a top strip so they never cover the hand, and `z` cycles the zoom of the shown image.

**Files:**
- Modify: `airpoint/hud.py`
- Modify: `airpoint/pointer.py`
- Modify: `airpoint/session.py`
- Modify: `airpoint/modes/record.py`
- Modify: `airpoint/modes/calibrate.py`
- Modify: `airpoint/app.py`
- Modify: `tests/test_hud.py`
- Modify: `tests/test_session.py`

- [x] **Step 1: Write the failing tests**

**Replace `tests/test_hud.py` with:**

````python
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


def test_canon_view_is_640_wide_and_keeps_the_aspect():
    assert hud.canon_view(np.zeros((1080, 1920, 3), np.uint8)).shape == (360, 640, 3)
    assert hud.canon_view(np.zeros((720, 1280, 3), np.uint8)).shape == (360, 640, 3)
    assert hud.canon_view(np.zeros((480, 640, 3), np.uint8)).shape == (480, 640, 3)


def test_zoomed_scales_the_shown_image():
    img = np.zeros((360, 640, 3), np.uint8)
    assert hud.zoomed(img, 1.0) is img
    assert hud.zoomed(img, 1.5).shape == (540, 960, 3)


def _width(text, scale, thickness=1):
    import cv2
    return cv2.getTextSize(text, hud.FONT, scale, thickness)[0][0]


def test_fit_scale_shrinks_only_when_needed():
    assert hud.fit_scale("short", 0.5, 600) == 0.5
    long = "x" * 200
    scale = hud.fit_scale(long, 0.5, 600)
    assert scale < 0.5 and _width(long, scale) <= 601


def test_worst_case_hud_lines_stay_legible_on_the_640_canvas():
    line1 = "SCROLLING | pose thumb_down 0.99 | ratio 1.23 | mlp/madhuramrathi | 30 fps"
    line2 = "last: drag_end, click | mouse [m] OFF | dwell [d] OFF | " + hud.KEY_HINT
    title = "HOLD thumb_down   10s"
    sub = "tilt, move closer, move away | kept 1508 | rejected 1508"
    assert hud.fit_scale(line1, 0.50, 620) >= 0.42
    assert hud.fit_scale(line2, 0.45, 620) >= 0.40
    assert hud.fit_scale(title, 0.95, 610, 2) == 0.95
    assert hud.fit_scale(sub, 0.45, 610) >= 0.40


def test_banner_and_prompt_draw_inside_the_frame():
    frame = np.zeros((360, 640, 3), np.uint8)
    hud.draw_banner(frame, "MOVING | pose palm 0.98", "last: none", True, line3=("shake 12 px", hud.WHITE))
    assert frame[:68].any() and not frame[70:].any()
    frame = np.zeros((360, 640, 3), np.uint8)
    hud.draw_prompt(frame, "HOLD palm   7s", "tilt | kept 12", "SIGN LOST")
    assert frame[:86].any() and not frame[88:].any()      # top strip: never over the hand
````

**Apply to `tests/test_session.py`:**

````diff
diff --git a/tests/test_session.py b/tests/test_session.py
index 3f1f369..eff436c 100644
--- a/tests/test_session.py
+++ b/tests/test_session.py
@@ -92,7 +92,39 @@ def test_mini_view_is_320x240_in_live_mode_only(tmp_path):
     assert s.view_dirty
     assert s.tick(blank(), BY_POSE["palm"], "Right", 0.0).shape == (240, 320, 3)
     s.on_key(ord("r"), 0.1)
-    assert s.tick(blank(), BY_POSE["palm"], "Right", 0.2).shape == (360, 640, 3)
+    assert s.tick(blank(), BY_POSE["palm"], "Right", 0.2).shape == (540, 960, 3)   # full view at zoom 1.5
+
+
+def test_z_cycles_the_zoom_of_the_full_view(tmp_path):
+    s, _ = make(tmp_path)
+    shapes = []
+    for i in range(3):
+        shapes.append(s.tick(blank(), None, None, i * DT).shape[:2])
+        s.view_dirty = False
+        s.on_key(ord("z"), i * DT)
+        assert s.view_dirty
+    assert shapes == [(540, 960), (720, 1280), (360, 640)]
+
+
+def test_the_pointer_does_not_depend_on_camera_resolution(tmp_path):
+    def run(w, h):
+        s, mouse = make(tmp_path)
+        t = 0.0
+        for i in range(60):
+            pts = BY_POSE["palm"] + np.array([0.002 * i, -0.001 * i, 0.0])   # hand drifting right and up
+            s.tick(blank(w, h), pts, "Right", t)
+            t += DT
+        return mouse.calls
+
+    assert run(640, 360) == run(1920, 1080) == run(1280, 720)
+
+
+def test_the_pointer_keeps_sub_pixel_precision(tmp_path):
+    s, _ = make(tmp_path)
+    pts = BY_POSE["palm"] + np.array([0.00037, 0.0, 0.0])
+    for i in range(45):
+        s.tick(blank(), pts, "Right", i * DT)
+    assert abs(s.pointer.out[0] - (0.50037 * 640)) < 0.01      # 320.2368, not 320
 
 
 class _Mlp:
````

- [x] **Step 2: Run to verify they fail**

Run: `venv/bin/pytest tests/test_hud.py tests/test_session.py -q`
Expected: failures with `AttributeError: module 'airpoint.hud' has no attribute 'canon_view'` (also `zoomed`, `fit_scale`) and `TypeError: draw_banner() got an unexpected keyword argument 'line3'`

- [x] **Step 3: Implement**

**Replace `airpoint/hud.py` with:**

````python
import cv2

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]
CANON_W = 640           # every pixel threshold, the pointer and all drawing live in a frame this wide
BANNER_LINE_H = 20
FONT = cv2.FONT_HERSHEY_SIMPLEX
ACCENT_ON, ACCENT_OFF, WHITE, MINT = (0, 200, 255), (90, 90, 90), (255, 255, 255), (167, 243, 208)
PANEL = (25, 35, 30)
MINI_SIZE = (320, 240)
ZOOMS = (1.0, 1.5, 2.0)
KEY_HINT = "keys q m d k c v z r"


def canon_view(frame):
    """The camera frame at CANON_W wide, aspect kept. The tracker reads the native frame; everything
    else (pointer pixels, thresholds, filters, drawing) uses this one, so it means the same on any camera."""
    h, w = frame.shape[:2]
    if w == CANON_W:
        return frame
    return cv2.resize(frame, (CANON_W, round(h * CANON_W / w)), interpolation=cv2.INTER_AREA)


def zoomed(img, zoom):
    if zoom == 1.0:
        return img
    return cv2.resize(img, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)


def fit_scale(text, scale, max_w, thickness=1):
    """The largest font scale <= `scale` at which `text` is no wider than `max_w` pixels."""
    w = cv2.getTextSize(text, FONT, scale, thickness)[0][0]
    return scale if w <= max_w else scale * max_w / w


def put_fit(img, text, org, scale, color, thickness=1):
    """putText that shrinks rather than runs off the right edge."""
    scale = fit_scale(text, scale, img.shape[1] - org[0] - 10, thickness)
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_skeleton(frame, pts):
    h, w = frame.shape[:2]
    px = [(int(x * w), int(y * h)) for x, y, _ in pts]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, px[a], px[b], (250, 44, 250), 2)
    for p in px:
        cv2.circle(frame, p, 3, (121, 22, 76), -1)


def _px(xy):
    return int(round(xy[0])), int(round(xy[1]))


def draw_cursor(frame, raw_xy, smooth_xy):
    cv2.circle(frame, _px(raw_xy), 3, (0, 0, 255), -1)
    cv2.circle(frame, _px(smooth_xy), 7, (0, 255, 0), 1)


def draw_region(frame, region, active):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = region
    cv2.rectangle(frame, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)),
                  ACCENT_ON if active else ACCENT_OFF, 1)


def draw_banner(frame, line1, line2, active, line3=None):
    """Top strip: status line, switches line, optional third line (shake readout)."""
    w = frame.shape[1]
    n = 3 if line3 else 2
    cv2.rectangle(frame, (0, 0), (w, 8 + BANNER_LINE_H * n), PANEL, -1)
    put_fit(frame, line1, (10, 19), 0.50, MINT)
    put_fit(frame, line2, (10, 19 + BANNER_LINE_H), 0.45, ACCENT_ON if active else WHITE)
    if line3:
        put_fit(frame, line3[0], (10, 19 + 2 * BANNER_LINE_H), 0.45, line3[1])


def draw_prompt(frame, title, subtitle, alert=None):
    """Top strip for record / calibrate: big title, small instructions, optional red alert.
    At the top so it never covers the hand the user is trying to hold in view."""
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, 86), PANEL, -1)
    put_fit(frame, title, (16, 34), 0.95, MINT, 2)
    put_fit(frame, subtitle, (16, 58), 0.45, WHITE)
    if alert:
        put_fit(frame, alert, (16, 78), 0.45, (60, 60, 255))


def draw_dwell_ring(frame, xy, progress):
    """Arc that fills clockwise from 12 o'clock as the dwell timer runs."""
    if progress <= 0.0:
        return
    cv2.ellipse(frame, _px(xy), (14, 14), -90, 0, int(360 * min(progress, 1.0)), ACCENT_ON, 2)


def mini_frame(frame, label):
    """320x240 copy with a one-line strip at the bottom; for the always-on-top corner view."""
    small = cv2.resize(frame, MINI_SIZE, interpolation=cv2.INTER_AREA)
    w, h = MINI_SIZE
    cv2.rectangle(small, (0, h - 24), (w, h), PANEL, -1)
    put_fit(small, label, (8, h - 8), 0.45, MINT)
    return small
````

**Apply to `airpoint/pointer.py`:**

````diff
diff --git a/airpoint/pointer.py b/airpoint/pointer.py
index 7687e98..e1aa32e 100644
--- a/airpoint/pointer.py
+++ b/airpoint/pointer.py
@@ -1,6 +1,7 @@
 from .filters import OneEuroFilter
 
 FC_MIN, FC_MIN_PRECLICK, PRECLICK_RATIO = 0.80, 0.30, 0.42
+BETA = 0.030   # 1-Euro speed gain per canonical px/s: Week 1's 0.010 at a 1920-wide frame, x 1920 / 640
 
 
 class PointerPipeline:
@@ -8,9 +9,12 @@ class PointerPipeline:
 
     def __init__(self, fc_min=FC_MIN):
         self.fc_min = fc_min
-        self.fx, self.fy = OneEuroFilter(FC_MIN), OneEuroFilter(FC_MIN)
+        self.fx, self.fy = OneEuroFilter(FC_MIN, BETA), OneEuroFilter(FC_MIN, BETA)
+        self.out = None
 
     def update(self, raw, t, preclick=False):
-        """`preclick` = fingers closing for a pinch: smooth harder so the click lands where aimed."""
+        """`preclick` = fingers closing for a pinch: smooth harder so the click lands where aimed.
+        Floats in, floats out: rounding here would step the cursor 3 screen points at a time."""
         fc = min(FC_MIN_PRECLICK, self.fc_min) if preclick else self.fc_min
-        return int(self.fx.filter(raw[0], t, fc)), int(self.fy.filter(raw[1], t, fc))
+        self.out = (self.fx.filter(raw[0], t, fc), self.fy.filter(raw[1], t, fc))
+        return self.out
````

**Apply to `airpoint/session.py`:**

````diff
diff --git a/airpoint/session.py b/airpoint/session.py
index b8bfcf6..5d732ef 100644
--- a/airpoint/session.py
+++ b/airpoint/session.py
@@ -16,7 +16,7 @@ class Obs:
     t: float
     pts: object    # (21, 3) landmarks in normalized image coords, or None
     hand: object   # "Left" | "Right" | None
-    raw: object    # pointer (x, y) in frame pixels, or None
+    raw: object    # pointer (x, y) in canonical view pixels (floats), or None
     w: int
     h: int
     fps: float
@@ -38,6 +38,7 @@ class Session:
         self.smoother, self.fsm, self.dwell = VoteSmoother(), AirPointFSM(), DwellDetector()
         self.pointer = PointerPipeline(fc_min)
         self.dwell_on, self.mini, self.view_dirty = dwell_on, False, False
+        self.zoom = 1.5
         self.recorder, self.rejected = None, 0
         self.mode = LiveMode()
         self.last_event, self.mini_label, self.prev_t = "none", "", None
@@ -45,13 +46,16 @@ class Session:
     def tick(self, frame, pts, hand, t):
         fps = 0.0 if self.prev_t is None else 1.0 / max(t - self.prev_t, 1e-5)
         self.prev_t = t
-        h, w = frame.shape[:2]
+        view = hud.canon_view(frame)
+        h, w = view.shape[:2]
         raw = None
         if pts is not None:
-            hud.draw_skeleton(frame, pts)
-            raw = (int(pts[F.MIDDLE_MCP, 0] * w), int(pts[F.MIDDLE_MCP, 1] * h))
-        self.mode.tick(self, Obs(t, pts, hand, raw, w, h, fps), frame)
-        return hud.mini_frame(frame, self.mini_label) if self.mini and self.mode.name == "live" else frame
+            hud.draw_skeleton(view, pts)
+            raw = (float(pts[F.MIDDLE_MCP, 0] * w), float(pts[F.MIDDLE_MCP, 1] * h))
+        self.mode.tick(self, Obs(t, pts, hand, raw, w, h, fps), view)
+        if self.mini and self.mode.name == "live":
+            return hud.mini_frame(view, self.mini_label)
+        return hud.zoomed(view, self.zoom)
 
     def set_mode(self, mode):
         self.mode.close(self)
@@ -76,6 +80,9 @@ class Session:
             self.smoother.reset()
         elif key == ord("v"):
             self.mini, self.view_dirty = not self.mini, True
+        elif key == ord("z"):
+            self.zoom = hud.ZOOMS[(hud.ZOOMS.index(self.zoom) + 1) % len(hud.ZOOMS)]
+            self.view_dirty = True
         elif key == ord("r") and self.user and self.mode.name != "calibrate":
             self.runner.release()
             self.set_mode(RecordMode(self) if self.mode.name == "live" else LiveMode())
````

**Apply to `airpoint/modes/record.py`:**

````diff
diff --git a/airpoint/modes/record.py b/airpoint/modes/record.py
index 4cbb384..5eb0bee 100644
--- a/airpoint/modes/record.py
+++ b/airpoint/modes/record.py
@@ -17,14 +17,14 @@ class RecordMode:
         phase, pose, left = self.session.update(o.t)
         lost = capture(self.session, s.recorder, pose, o.pts, o.hand, o.t)
         s.rejected += lost
-        title = {
-            "idle": f"RECORD {s.user}: SPACE = all poses, 1-6 = one pose",
-            "countdown": f"raise hand, make {pose}   {left:.0f}",
-            "capturing": f"HOLD {pose}   {left:.0f}s" + ("   << SIGN LOST" if lost else ""),
-            "done": "done - q to quit, 1-6 to redo one pose",
+        title, sub = {
+            "idle": (f"RECORD {s.user}", "SPACE = all six poses | 1-6 = one pose | r = back"),
+            "countdown": (f"make {pose}   {left:.0f}", "raise your hand"),
+            "capturing": (f"HOLD {pose}   {left:.0f}s", "tilt, move closer, move away"),
+            "done": ("done", "1-6 = redo one pose | r = back | q = quit"),
         }[phase]
-        hud.draw_prompt(frame, title, f"tilt, move closer, move away | kept {s.recorder.count} | "
-                                      f"rejected {s.rejected} | {hud.KEY_HINT}")
+        hud.draw_prompt(frame, title, f"{sub} | kept {s.recorder.count} | rejected {s.rejected}",
+                        "SIGN LOST - this frame was not saved" if lost else None)
 
     def on_key(self, s, key, t):
         if key == ord(" "):
````

**Apply to `airpoint/modes/calibrate.py`:**

````diff
diff --git a/airpoint/modes/calibrate.py b/airpoint/modes/calibrate.py
index 7dfacf7..024c20e 100644
--- a/airpoint/modes/calibrate.py
+++ b/airpoint/modes/calibrate.py
@@ -34,16 +34,17 @@ class CalibrateMode:
                 self.trainer.start()
             elif self.trainer.done:
                 self._finish(s)
-        title = {
-            "countdown": f"CALIBRATE: open palm, hold still   {left:.0f}",
-            "still": f"HOLD STILL   {left:.0f}s",
-            "poses_countdown": f"make {pose}   {left:.0f}",
-            "capturing": f"HOLD {pose}   {left:.0f}s" + ("   << SIGN LOST" if lost else ""),
-            "training": "Learning your hand...",
-            "done": f"done - {calib.message}",
-            "failed": f"retrain failed, old model kept - {calib.message}",
+        title, sub = {
+            "countdown": (f"CALIBRATE   {left:.0f}", "open palm, hold still"),
+            "still": (f"HOLD STILL   {left:.0f}s", "measuring how steady your hand is"),
+            "poses_countdown": (f"make {pose}   {left:.0f}", "raise your hand"),
+            "capturing": (f"HOLD {pose}   {left:.0f}s", "tilt a little, keep the sign"),
+            "training": ("Learning your hand...", "a few seconds"),
+            "done": ("done", calib.message),
+            "failed": ("retrain failed", f"old model kept: {calib.message}"),
         }[phase]
-        hud.draw_prompt(frame, title, f"kept {self.rec.count} | c = close | {hud.KEY_HINT}")
+        hud.draw_prompt(frame, title, f"{sub} | kept {self.rec.count} | c = close",
+                        "SIGN LOST - this frame was not saved" if lost else None)
 
     def _finish(self, s):
         calib, trainer, self.trainer = self.calib, self.trainer, None
````

**Apply to `airpoint/app.py`:**

````diff
diff --git a/airpoint/app.py b/airpoint/app.py
index ddbcaae..f21b1ed 100644
--- a/airpoint/app.py
+++ b/airpoint/app.py
@@ -18,7 +18,7 @@ ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
 MODELS_DIR = os.path.join(ROOT, "models")
 MODEL_PATH, LABELS_PATH = os.path.join(MODELS_DIR, "pose_mlp.joblib"), os.path.join(MODELS_DIR, "labels.json")
 WINDOW, MINI_MARGIN = "AirPoint", 20
-KEYS = "q quit | m mouse | d dwell | k rules/mlp | c calibrate | v mini | r record | 1-6 pose | space start/redo"
+KEYS = "q quit | m mouse | d dwell | k rules/mlp | c calibrate | v mini | z zoom | r record | 1-6 pose | space start/redo"
 
 
 def load_classifiers(force_rules, profile=None):
@@ -72,8 +72,9 @@ def main(argv=None):
     if args.record:
         s.set_mode(RecordMode(s))
     camera, tracker = Camera(), HandTracker()
-    print(f"[AirPoint] camera {camera.index} {camera.size[0]}x{camera.size[1]} @ {camera.fps:.0f} fps | "
-          f"classifier {s.clf.name} | fc_min {s.pointer.fc_min} | {KEYS}")
+    view_h = round(camera.size[1] * hud.CANON_W / camera.size[0])
+    print(f"[AirPoint] camera {camera.index} {camera.size[0]}x{camera.size[1]} @ {camera.fps:.0f} fps -> "
+          f"view {hud.CANON_W}x{view_h} | classifier {s.clf.name} | fc_min {s.pointer.fc_min} | {KEYS}")
 
     try:
         for frame in camera.frames():
````

- [x] **Step 4: Run to verify they pass, then the full suite**

Run: `venv/bin/pytest tests/test_hud.py tests/test_session.py -q && venv/bin/pytest -q`
Expected: the full suite ends with `77 passed`

- [x] **Step 5: Commit**

```bash
git add airpoint/app.py airpoint/hud.py airpoint/modes airpoint/pointer.py airpoint/session.py tests/test_hud.py tests/test_session.py
git commit -m "feat: one 640-wide space for pointer maths and drawing; float pointer, beta 0.030; HUD laid out for it; z zoom"
```

---

### Task 5: Tremor simulator

One offset added to all 21 landmarks before anything reads them, so the skeleton and the pointer both shake while wrist-centred features (and the sign read) do not change. Amplitude is in palm lengths, so it looks the same at any distance. Live mode only: `Session.set_mode` switches it off on the way into record or calibrate, so simulated shake can never reach recorded data or a profile.

**Files:**
- Create: `airpoint/tremor.py`
- Modify: `airpoint/session.py`
- Modify: `airpoint/modes/live.py`
- Modify: `airpoint/hud.py`
- Modify: `tests/test_session.py`
- Create: `tests/test_tremor.py`

- [x] **Step 1: Write the failing tests**

**Apply to `tests/test_session.py`:**

````diff
diff --git a/tests/test_session.py b/tests/test_session.py
index eff436c..41165e7 100644
--- a/tests/test_session.py
+++ b/tests/test_session.py
@@ -232,3 +232,32 @@ def test_failed_retrain_keeps_the_old_model(tmp_path):
     assert s.mode.calib.phase == "failed" and "only 3 usable" in s.mode.calib.message
     assert s.clf is old
     s.close()
+
+
+def test_t_shakes_the_pointer_but_not_the_sign(tmp_path):
+    s, mouse = make(tmp_path)
+    s.on_key(ord("t"), 0.0)
+    xs, t = [], 0.0
+    for _ in range(90):
+        s.tick(blank(), BY_POSE["palm"], "Right", t)
+        xs.append(s.obs.raw[0])
+        t += DT
+    assert s.sim.on and np.std(xs) > 5.0
+    assert s.fsm.state == "MOVING" and kinds(mouse) == []
+    s.on_key(ord("]"), t)
+    s.on_key(ord("}"), t)
+    assert (s.sim.amp, s.sim.hz) == (0.25, 6.5)
+
+
+def test_simulated_tremor_cannot_follow_you_into_record_or_calibrate(tmp_path):
+    s, _ = make(tmp_path)
+    s.on_key(ord("t"), 0.0)
+    s.on_key(ord("r"), 0.0)
+    assert s.mode.name == "record" and not s.sim.on
+    s.on_key(ord("t"), 0.0)
+    assert not s.sim.on                       # t only works in live mode
+    s.on_key(ord("r"), 0.0)
+    s.on_key(ord("t"), 0.0)
+    s.on_key(ord("c"), 0.0)
+    assert s.mode.name == "calibrate" and not s.sim.on
+    s.close()
````

**Create `tests/test_tremor.py`:**

````python
import numpy as np

from airpoint.features import normalize_63
from airpoint.tremor import TremorSim
from tests.hands import BY_POSE


def test_sim_moves_every_landmark_together_and_leaves_features_unchanged():
    sim, base, xs = TremorSim(hz=6.0, amp=0.2), BY_POSE["palm"], []
    for i in range(90):
        out = sim.apply(base, i / 30, 640, 360)
        assert np.allclose(normalize_63(out), normalize_63(base), atol=1e-9)
        xs.append(out[9, 0] * 640)
    assert base[9, 0] == 0.5                              # the input is not modified
    swing, palm_px = (max(xs) - min(xs)) / 2, 0.2 * 360
    assert 0.7 * 0.2 * palm_px < swing < 1.6 * 0.2 * palm_px


def test_sim_amplitude_scales_with_palm_length():
    near, far = TremorSim(), TremorSim()
    a = max(abs(near.offset(i / 30, 100.0)[0]) for i in range(60))
    b = max(abs(far.offset(i / 30, 50.0)[0]) for i in range(60))
    assert abs(a / b - 2.0) < 1e-9


def test_sim_nudges_clamp_to_their_ranges():
    sim = TremorSim()
    for _ in range(40):
        sim.nudge(d_hz=0.5, d_amp=0.05)
    assert (sim.hz, sim.amp) == (10.0, 0.60)
    for _ in range(40):
        sim.nudge(d_hz=-0.5, d_amp=-0.05)
    assert (sim.hz, sim.amp) == (3.5, 0.05)
````

- [x] **Step 2: Run to verify they fail**

Run: `venv/bin/pytest tests/test_tremor.py tests/test_session.py -q`
Expected: `1 error` with `ModuleNotFoundError: No module named 'airpoint.tremor'`

- [x] **Step 3: Implement**

**Create `airpoint/tremor.py`:**

````python
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
````

**Apply to `airpoint/session.py`:**

````diff
diff --git a/airpoint/session.py b/airpoint/session.py
index 5d732ef..2e7c920 100644
--- a/airpoint/session.py
+++ b/airpoint/session.py
@@ -8,6 +8,10 @@ from .fsm import AirPointFSM, DwellDetector
 from .modes import CalibrateMode, LiveMode, RecordMode
 from .pointer import FC_MIN, PointerPipeline
 from .train import train_personal
+from .tremor import TremorSim
+
+
+SIM_NUDGES = {ord("["): (0.0, -0.05), ord("]"): (0.0, 0.05), ord("{"): (-0.5, 0.0), ord("}"): (0.5, 0.0)}
 
 
 @dataclass
@@ -37,11 +41,13 @@ class Session:
         self.data_dir, self.models_dir = os.path.join(root, "data"), os.path.join(root, "models")
         self.smoother, self.fsm, self.dwell = VoteSmoother(), AirPointFSM(), DwellDetector()
         self.pointer = PointerPipeline(fc_min)
+        self.sim = TremorSim()
         self.dwell_on, self.mini, self.view_dirty = dwell_on, False, False
         self.zoom = 1.5
         self.recorder, self.rejected = None, 0
         self.mode = LiveMode()
         self.last_event, self.mini_label, self.prev_t = "none", "", None
+        self.obs = None
 
     def tick(self, frame, pts, hand, t):
         fps = 0.0 if self.prev_t is None else 1.0 / max(t - self.prev_t, 1e-5)
@@ -50,9 +56,12 @@ class Session:
         h, w = view.shape[:2]
         raw = None
         if pts is not None:
+            if self.sim.on:
+                pts = self.sim.apply(pts, t, w, h)
             hud.draw_skeleton(view, pts)
             raw = (float(pts[F.MIDDLE_MCP, 0] * w), float(pts[F.MIDDLE_MCP, 1] * h))
-        self.mode.tick(self, Obs(t, pts, hand, raw, w, h, fps), view)
+        self.obs = Obs(t, pts, hand, raw, w, h, fps)
+        self.mode.tick(self, self.obs, view)
         if self.mini and self.mode.name == "live":
             return hud.mini_frame(view, self.mini_label)
         return hud.zoomed(view, self.zoom)
@@ -60,6 +69,8 @@ class Session:
     def set_mode(self, mode):
         self.mode.close(self)
         self.mode = mode
+        if mode.name != "live":
+            self.sim.on = False      # simulated shake never reaches recordings, calibration or a profile
 
     def use_model(self, model_path, labels_path):
         """Hot-swap to a personal model. Runs on the frame-loop thread."""
@@ -83,6 +94,10 @@ class Session:
         elif key == ord("z"):
             self.zoom = hud.ZOOMS[(hud.ZOOMS.index(self.zoom) + 1) % len(hud.ZOOMS)]
             self.view_dirty = True
+        elif key == ord("t") and self.mode.name == "live":
+            self.sim.on = not self.sim.on
+        elif key in SIM_NUDGES:
+            self.sim.nudge(*SIM_NUDGES[key])
         elif key == ord("r") and self.user and self.mode.name != "calibrate":
             self.runner.release()
             self.set_mode(RecordMode(self) if self.mode.name == "live" else LiveMode())
````

**Apply to `airpoint/modes/live.py`:**

````diff
diff --git a/airpoint/modes/live.py b/airpoint/modes/live.py
index 5950eed..a7d88c7 100644
--- a/airpoint/modes/live.py
+++ b/airpoint/modes/live.py
@@ -30,7 +30,9 @@ class LiveMode:
         if events:
             s.last_event = ", ".join(e.kind + ("" if e.arg is None else f" {e.arg}") for e in events)
         hud.draw_region(frame, s.runner.active_region, s.runner.enabled)
-        s.mini_label = f"{state} | {label} {conf:.2f} | mouse {'ON' if s.runner.enabled else 'OFF'}"
+        sim = f"SIMULATED TREMOR {s.sim.hz:.1f} Hz, {s.sim.amp:.2f} palm" if s.sim.on else None
+        s.mini_label = (f"{state} | {label} {conf:.2f} | mouse {'ON' if s.runner.enabled else 'OFF'}"
+                        + (" | SIM" if sim else ""))
         if not s.mini:
             hud.draw_banner(
                 frame,
@@ -38,6 +40,7 @@ class LiveMode:
                 f"last: {s.last_event} | mouse [m] {'ON' if s.runner.enabled else 'OFF'} | "
                 f"dwell [d] {'ON' if s.dwell_on else 'OFF'} | {hud.KEY_HINT}",
                 s.runner.enabled,
+                line3=(sim, hud.ALERT) if sim else None,
             )
 
     def on_key(self, s, key, t):
````

**Apply to `airpoint/hud.py`:**

````diff
diff --git a/airpoint/hud.py b/airpoint/hud.py
index 0995f8e..09388dd 100644
--- a/airpoint/hud.py
+++ b/airpoint/hud.py
@@ -12,10 +12,10 @@ CANON_W = 640           # every pixel threshold, the pointer and all drawing liv
 BANNER_LINE_H = 20
 FONT = cv2.FONT_HERSHEY_SIMPLEX
 ACCENT_ON, ACCENT_OFF, WHITE, MINT = (0, 200, 255), (90, 90, 90), (255, 255, 255), (167, 243, 208)
-PANEL = (25, 35, 30)
+PANEL, ALERT = (25, 35, 30), (60, 60, 255)
 MINI_SIZE = (320, 240)
 ZOOMS = (1.0, 1.5, 2.0)
-KEY_HINT = "keys q m d k c v z r"
+KEY_HINT = "keys q m d k c v z t r"
 
 
 def canon_view(frame):
@@ -89,7 +89,7 @@ def draw_prompt(frame, title, subtitle, alert=None):
     put_fit(frame, title, (16, 34), 0.95, MINT, 2)
     put_fit(frame, subtitle, (16, 58), 0.45, WHITE)
     if alert:
-        put_fit(frame, alert, (16, 78), 0.45, (60, 60, 255))
+        put_fit(frame, alert, (16, 78), 0.45, ALERT)
 
 
 def draw_dwell_ring(frame, xy, progress):
````

- [x] **Step 4: Run to verify they pass, then the full suite**

Run: `venv/bin/pytest tests/test_tremor.py tests/test_session.py -q && venv/bin/pytest -q`
Expected: the full suite ends with `82 passed`

- [x] **Step 5: Commit**

```bash
git add airpoint/hud.py airpoint/modes airpoint/session.py airpoint/tremor.py tests/test_session.py tests/test_tremor.py
git commit -m "feat(tremor): simulator — t on/off, [ ] size, { } speed; live mode only, always labelled"
```

---

### Task 6: Tremor meter

`estimate` looks at one window of pointer samples: resample to a uniform grid, zero-phase Butterworth band-pass 3–12 Hz (upper edge capped at 0.42 x frame rate), amplitude = RMS over both axes, frequency = strongest peak, tremor if amplitude >= 1.5 px and the peak is >= 8 x the band median. `TremorMeter` keeps the last 3 s, re-estimates every 0.33 s, locks after three agreeing tremor estimates and lets go after three without. `tests/pointer_sim.py` is the design spike's generator: reaches, holds, landmark noise and a tremor with the same model as the simulator.

**Files:**
- Modify: `airpoint/tremor.py`
- Create: `tests/pointer_sim.py`
- Modify: `tests/test_tremor.py`

- [x] **Step 1: Write the failing tests**

**Create `tests/pointer_sim.py`:**

````python
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
````

**Replace `tests/test_tremor.py` with:**

````python
import numpy as np

from airpoint.features import normalize_63
from airpoint.tremor import TremorMeter, TremorSim, estimate
from tests.hands import BY_POSE
from tests.pointer_sim import pointer_session


def test_sim_moves_every_landmark_together_and_leaves_features_unchanged():
    sim, base, xs = TremorSim(hz=6.0, amp=0.2), BY_POSE["palm"], []
    for i in range(90):
        out = sim.apply(base, i / 30, 640, 360)
        assert np.allclose(normalize_63(out), normalize_63(base), atol=1e-9)
        xs.append(out[9, 0] * 640)
    assert base[9, 0] == 0.5                              # the input is not modified
    swing, palm_px = (max(xs) - min(xs)) / 2, 0.2 * 360
    assert 0.7 * 0.2 * palm_px < swing < 1.6 * 0.2 * palm_px


def test_sim_amplitude_scales_with_palm_length():
    near, far = TremorSim(), TremorSim()
    a = max(abs(near.offset(i / 30, 100.0)[0]) for i in range(60))
    b = max(abs(far.offset(i / 30, 50.0)[0]) for i in range(60))
    assert abs(a / b - 2.0) < 1e-9


def test_sim_nudges_clamp_to_their_ranges():
    sim = TremorSim()
    for _ in range(40):
        sim.nudge(d_hz=0.5, d_amp=0.05)
    assert (sim.hz, sim.amp) == (10.0, 0.60)
    for _ in range(40):
        sim.nudge(d_hz=-0.5, d_amp=-0.05)
    assert (sim.hz, sim.amp) == (3.5, 0.05)


def run_meter(s):
    """Feed a whole session through a TremorMeter. -> list of (t, Reading), one per sample."""
    m, out = TremorMeter(), []
    for t, xy in zip(s.t, s.raw):
        m.add(t, xy)
        out.append((t, m.read(t)))
    return out


def test_meter_waits_for_enough_samples():
    m = TremorMeter()
    for i in range(20):
        m.add(i / 30, (320.0, 180.0))
    assert m.read(0.7).status == "wait"


def test_meter_locks_fast_and_reads_frequency_and_size():
    s = pointer_session(20, seed=1, hz=6.0, amp=14.0)
    readings = run_meter(s)
    first_lock = next(t for t, r in readings if r.locked)
    assert first_lock <= 2.5
    for end in (8.0, 12.0, 16.0):
        r = next(r for t, r in readings if t >= end)
        win = (s.t > end - 3.0) & (s.t <= end)
        true_amp = float(np.hypot(s.trem[win, 0].std(), s.trem[win, 1].std()))
        assert r.locked and abs(r.lock_hz - s.freq[win].mean()) <= 0.4
        assert abs(r.amp / true_amp - 1.0) <= 0.25


def test_meter_never_locks_without_a_tremor():
    for seed in (2, 3):
        assert not any(r.locked for _, r in run_meter(pointer_session(40, seed=seed)))


def test_meter_lets_go_within_four_seconds_of_the_tremor_stopping():
    s = pointer_session(30, seed=4, hz=6.0, amp=14.0, stop=15.0)
    readings = run_meter(s)
    assert next(r for t, r in readings if t >= 14.0).locked
    released = next(t for t, r in readings if t >= 15.0 and not r.locked)
    assert released - 15.0 <= 4.0


def test_meter_at_a_dim_room_frame_rate_sees_only_slower_tremor():
    s = pointer_session(20, seed=5, fs=16.5, hz=5.0, amp=14.0)
    r = run_meter(s)[-1][1]
    assert r.locked and abs(r.lock_hz - 5.0) <= 0.6
    assert 6.5 <= r.max_hz <= 7.2                         # 0.42 x 16.5 fps


def test_meter_refuses_a_camera_that_is_too_slow():
    s = pointer_session(12, seed=6, fs=10.0, hz=4.0, amp=14.0)
    r = run_meter(s)[-1][1]
    assert r.status == "slow" and not r.locked


def test_estimate_reports_the_spectrum_it_judged():
    s = pointer_session(4, seed=7, hz=6.0, amp=14.0)
    r = estimate(s.t[-90:], s.raw[-90:])
    freqs, power = r.spectrum
    assert r.status == "tremor" and len(freqs) == len(power) == 91 and abs(freqs[int(np.argmax(power))] - r.hz) < 1e-9
````

- [x] **Step 2: Run to verify they fail**

Run: `venv/bin/pytest tests/test_tremor.py -q`
Expected: `1 error` with `ImportError: cannot import name 'TremorMeter' from 'airpoint.tremor'`

- [x] **Step 3: Implement**

**Replace `airpoint/tremor.py` with:**

````python
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
````

- [x] **Step 4: Run to verify they pass, then the full suite**

Run: `venv/bin/pytest tests/test_tremor.py -q && venv/bin/pytest -q`
Expected: the full suite ends with `89 passed`

- [x] **Step 5: Commit**

```bash
git add airpoint/tremor.py tests/pointer_sim.py tests/test_tremor.py
git commit -m "feat(tremor): meter — shake frequency and size from the pointer, with lock-on"
```

---

### Task 7: Tremor canceller and filter modes

A second-order notch per axis at the locked frequency (Q 1.5), stepped on every sample so its state is warm, blended in over 0.3 s when the meter locks. Chain: raw → canceller → 1-Euro. With no lock the canceller returns its input untouched, so `full` is then exactly `smooth` (a test asserts equality). `b` cycles full / smoothing only / off. A gap in tracking longer than 0.25 s clears the samples and settles the notch on the new position; longer than 2 s also drops the lock.

**Files:**
- Modify: `airpoint/tremor.py`
- Modify: `airpoint/pointer.py`
- Modify: `airpoint/session.py`
- Modify: `airpoint/modes/live.py`
- Modify: `airpoint/hud.py`
- Modify: `airpoint/app.py`
- Modify: `tests/test_hud.py`
- Create: `tests/test_pointer.py`
- Modify: `tests/test_session.py`

- [x] **Step 1: Write the failing tests**

**Apply to `tests/test_hud.py`:**

````diff
diff --git a/tests/test_hud.py b/tests/test_hud.py
index 43f029e..22dfc1c 100644
--- a/tests/test_hud.py
+++ b/tests/test_hud.py
@@ -44,7 +44,7 @@ def test_fit_scale_shrinks_only_when_needed():
 
 def test_worst_case_hud_lines_stay_legible_on_the_640_canvas():
     line1 = "SCROLLING | pose thumb_down 0.99 | ratio 1.23 | mlp/madhuramrathi | 30 fps"
-    line2 = "last: drag_end, click | mouse [m] OFF | dwell [d] OFF | " + hud.KEY_HINT
+    line2 = "last: drag_end, click | mouse [m] OFF | dwell [d] OFF | filter [b] smooth only | " + hud.KEY_HINT
     title = "HOLD thumb_down   10s"
     sub = "tilt, move closer, move away | kept 1508 | rejected 1508"
     assert hud.fit_scale(line1, 0.50, 620) >= 0.42
````

**Create `tests/test_pointer.py`:**

````python
import math

import numpy as np

from airpoint.filters import OneEuroFilter
from airpoint.pointer import BETA, FC_MIN, PointerPipeline
from airpoint.tremor import Reading, TremorCanceller
from tests.pointer_sim import dwell_rate, pointer_session, shake_left


def run(s, mode):
    p = PointerPipeline()
    p.mode = mode
    return np.array([p.update((float(x), float(y)), float(t)) for t, (x, y) in zip(s.t, s.raw)]), p


def test_without_a_tremor_full_is_exactly_smoothing_only():
    s = pointer_session(30, seed=11)
    full, p = run(s, "full")
    smooth, _ = run(s, "smooth")
    assert not p.reading.locked and np.array_equal(full, smooth)


def test_smoothing_only_is_the_plain_one_euro_filter():
    s = pointer_session(10, seed=12)
    smooth, _ = run(s, "smooth")
    fx, fy = OneEuroFilter(FC_MIN, BETA), OneEuroFilter(FC_MIN, BETA)
    plain = np.array([(fx.filter(x, t), fy.filter(y, t)) for t, (x, y) in zip(s.t, s.raw)])
    assert np.array_equal(smooth, plain)


def test_off_returns_the_raw_pointer_and_modes_cycle():
    s = pointer_session(5, seed=13, hz=6.0, amp=14.0)
    off, p = run(s, "off")
    assert np.array_equal(off, s.raw)
    seen = []
    for _ in range(3):
        p.cycle_mode()
        seen.append(p.mode)
    assert seen == ["full", "smooth", "off"]


def test_canceller_leaves_less_than_half_the_shake_of_smoothing_alone():
    for seed in (14, 15):
        s = pointer_session(40, seed=seed, hz=6.0, amp=14.0)
        full, p = run(s, "full")
        smooth, _ = run(s, "smooth")
        assert p.reading.locked and abs(p.reading.lock_hz - 6.0) < 0.8
        assert shake_left(s, full) < 0.5 * shake_left(s, smooth)
        assert p.left.amp < 0.5 * p.reading.amp        # the live readout agrees


def test_dwell_clicking_survives_a_severe_tremor_only_with_the_canceller():
    full_ok, smooth_ok = [], []
    for seed in (16, 17, 18):
        s = pointer_session(60, seed=seed, hz=5.0, amp=25.0)
        full_ok.append(dwell_rate(s, run(s, "full")[0]))
        smooth_ok.append(dwell_rate(s, run(s, "smooth")[0]))
    assert np.mean(full_ok) >= 0.95 and np.mean(smooth_ok) < 0.5


def test_canceller_still_helps_at_a_dim_room_frame_rate():
    s = pointer_session(40, seed=19, fs=16.5, hz=6.0, amp=14.0)
    assert shake_left(s, run(s, "full")[0]) < 0.6 * shake_left(s, run(s, "smooth")[0])


def test_a_gap_in_tracking_resets_the_canceller_so_the_pointer_glides_in_without_ringing():
    s = pointer_session(12, seed=20, hz=6.0, amp=14.0)
    p = PointerPipeline()
    for t, (x, y) in zip(s.t, s.raw):
        p.update((float(x), float(y)), float(t))
    assert p.reading.locked
    t0 = float(s.t[-1]) + 0.8                              # hand away for 0.8 s, back somewhere else
    outs = [p.update((100.0, 80.0), t0 + i / 30) for i in range(20)]
    assert p.reading.locked                                # a short gap keeps the lock
    d = [math.hypot(x - 100.0, y - 80.0) for x, y in outs]
    assert all(b <= a + 1e-9 for a, b in zip(d, d[1:]))    # closes in steadily, never swings back out
    assert d[-1] < 1.0


def test_canceller_reset_settles_it_on_the_new_position():
    c, fs = TremorCanceller(), 30.0
    locked = Reading(status="tremor", hz=6.0, amp=10.0, fs=fs, locked=True, lock_hz=6.0)
    for i in range(90):
        c.step((320.0 + 14.0 * math.sin(2 * math.pi * 6.0 * i / fs), 180.0), i / fs, locked)
    c.reset((100.0, 80.0))
    for i in range(10):
        x, y = c.step((100.0, 80.0), (90 + i) / fs, locked)
        assert abs(x - 100.0) < 1e-9 and abs(y - 80.0) < 1e-9


def test_canceller_passes_through_until_locked_then_removes_the_tone():
    c, fs = TremorCanceller(), 30.0
    tone = [(320.0 + 14.0 * math.sin(2 * math.pi * 6.0 * i / fs), 180.0) for i in range(150)]
    idle = Reading(status="steady", fs=fs)
    assert [c.step(xy, i / fs, idle) for i, xy in enumerate(tone[:30])] == tone[:30]
    locked = Reading(status="tremor", hz=6.0, amp=10.0, fs=fs, locked=True, lock_hz=6.0)
    out = [c.step(xy, (30 + i) / fs, locked) for i, xy in enumerate(tone[30:])]
    assert max(abs(x - 320.0) for x, _ in out[60:]) < 1.5   # 14 px of shake, under 1.5 px left
````

**Apply to `tests/test_session.py`:**

````diff
diff --git a/tests/test_session.py b/tests/test_session.py
index 41165e7..d5f5270 100644
--- a/tests/test_session.py
+++ b/tests/test_session.py
@@ -261,3 +261,18 @@ def test_simulated_tremor_cannot_follow_you_into_record_or_calibrate(tmp_path):
     s.on_key(ord("c"), 0.0)
     assert s.mode.name == "calibrate" and not s.sim.on
     s.close()
+
+
+def test_simulated_tremor_is_locked_and_cancelled_in_the_live_loop(tmp_path):
+    s, _ = make(tmp_path)
+    s.on_key(ord("t"), 0.0)
+    t = feed(s, "palm", 6.0, 0.0)
+    r = s.pointer.reading
+    assert r.locked and abs(r.lock_hz - 6.0) < 0.8 and r.amp > 5.0
+    assert s.pointer.left.amp < 0.5 * r.amp
+    s.on_key(ord("b"), t)
+    assert s.pointer.mode == "smooth"
+    s.on_key(ord("b"), t)
+    assert s.pointer.mode == "off"
+    feed(s, "palm", 0.1, t)
+    assert s.pointer.out == s.obs.raw                        # off = the raw pointer
````

- [x] **Step 2: Run to verify they fail**

Run: `venv/bin/pytest tests/test_pointer.py tests/test_session.py tests/test_hud.py -q`
Expected: `1 error` with `ImportError: cannot import name 'TremorCanceller' from 'airpoint.tremor'`

- [x] **Step 3: Implement**

**Apply to `airpoint/tremor.py`:**

````diff
diff --git a/airpoint/tremor.py b/airpoint/tremor.py
index 1291708..74b6d55 100644
--- a/airpoint/tremor.py
+++ b/airpoint/tremor.py
@@ -135,3 +135,51 @@ class TremorMeter:
                 self.lock_hz = 0.6 * self.lock_hz + 0.4 * r.hz
         self.reading = replace(r, locked=self.locked, lock_hz=self.lock_hz if self.locked else 0.0)
         return self.reading
+
+
+class TremorCanceller:
+    """A notch at the locked tremor frequency, blended in while the meter is locked.
+
+    A low-pass must choose between shake and lag; a notch removes only the band the tremor
+    lives in. It steps on every sample so its state is warm the moment a lock arrives."""
+
+    Q, RAMP_S, IDLE_HZ = 1.5, 0.3, 6.0
+
+    def __init__(self):
+        self.z = np.zeros((2, 2))          # filter state: [stage, axis]
+        self.tuned, self.b, self.a = None, None, None
+        self.g, self.t = 0.0, None         # g: how much of the notch output is used, 0..1
+
+    def _tune(self, hz, fs):
+        tuned = (round(hz, 1), round(fs))
+        if tuned == self.tuned:
+            return
+        self.tuned = tuned
+        w0 = 2 * math.pi * tuned[0] / tuned[1]
+        alpha = math.sin(w0) / (2 * self.Q)
+        self.b = np.array([1.0, -2 * math.cos(w0), 1.0]) / (1 + alpha)
+        self.a = np.array([1.0, -2 * math.cos(w0) / (1 + alpha), (1 - alpha) / (1 + alpha)])
+
+    def reset(self, xy):
+        """Settle the filter at `xy` (after a gap), so the next output equals its input."""
+        if self.b is not None:
+            x = np.asarray(xy, dtype=float)
+            self.z[0], self.z[1] = x * (1.0 - self.b[0]), x * (self.b[2] - self.a[2])
+
+    def step(self, xy, t, reading):
+        """-> the pointer with the tremor band removed in proportion to `g`."""
+        x = np.asarray(xy, dtype=float)
+        dt, self.t = (0.0 if self.t is None else t - self.t), t
+        if reading.fs <= 0.0:              # the meter does not know the frame rate yet
+            return float(x[0]), float(x[1])
+        fresh = self.b is None
+        hz = reading.lock_hz if reading.locked else (self.tuned[0] if self.tuned else self.IDLE_HZ)
+        self._tune(min(hz, 0.45 * reading.fs), reading.fs)
+        if fresh:
+            self.reset(x)
+        y = self.b[0] * x + self.z[0]
+        self.z[0] = self.b[1] * x - self.a[1] * y + self.z[1]
+        self.z[1] = self.b[2] * x - self.a[2] * y
+        self.g += ((1.0 if reading.locked else 0.0) - self.g) * min(dt / self.RAMP_S, 1.0)
+        out = x + self.g * (y - x)
+        return float(out[0]), float(out[1])
````

**Replace `airpoint/pointer.py` with:**

````python
from collections import deque

from .filters import OneEuroFilter
from .tremor import Reading, TremorCanceller, TremorMeter

FC_MIN, FC_MIN_PRECLICK, PRECLICK_RATIO = 0.80, 0.30, 0.42
BETA = 0.030   # 1-Euro speed gain per canonical px/s: Week 1's 0.010 at a 1920-wide frame, x 1920 / 640
MODES = ("full", "smooth", "off")   # tremor canceller + smoothing | smoothing only | raw pointer
GAP_S, STALE_S, TRAIL_S = 0.25, 2.0, 4.0


class PointerPipeline:
    """Raw pointer -> the pointer the cursor follows:  canceller -> 1-Euro smoothing.

    With no tremor lock the canceller passes its input through untouched, so `full` is then
    exactly `smooth`. All filters are fed in every mode, so switching modes never jumps."""

    def __init__(self, fc_min=FC_MIN):
        self.fc_min, self.mode = fc_min, "full"
        self.fx, self.fy = OneEuroFilter(FC_MIN, BETA), OneEuroFilter(FC_MIN, BETA)
        self.meter, self.out_meter, self.canceller = TremorMeter(), TremorMeter(), TremorCanceller()
        self.reading, self.left = Reading(), Reading()   # shake going in, shake left in the output
        self.trail = deque()                             # (t, raw, out) for the last TRAIL_S seconds
        self.t, self.out = None, None

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
        fc = min(FC_MIN_PRECLICK, self.fc_min) if preclick else self.fc_min
        smooth = (self.fx.filter(x[0], t, fc), self.fy.filter(x[1], t, fc))
        self.out = raw if self.mode == "off" else smooth
        self.out_meter.add(t, self.out)
        self.left = self.out_meter.read(t)
        self.trail.append((t, raw, self.out))
        while t - self.trail[0][0] > TRAIL_S:
            self.trail.popleft()
        return self.out
````

**Apply to `airpoint/session.py`:**

````diff
diff --git a/airpoint/session.py b/airpoint/session.py
index 2e7c920..02a2220 100644
--- a/airpoint/session.py
+++ b/airpoint/session.py
@@ -94,6 +94,8 @@ class Session:
         elif key == ord("z"):
             self.zoom = hud.ZOOMS[(hud.ZOOMS.index(self.zoom) + 1) % len(hud.ZOOMS)]
             self.view_dirty = True
+        elif key == ord("b"):
+            self.pointer.cycle_mode()
         elif key == ord("t") and self.mode.name == "live":
             self.sim.on = not self.sim.on
         elif key in SIM_NUDGES:
````

**Apply to `airpoint/modes/live.py`:**

````diff
diff --git a/airpoint/modes/live.py b/airpoint/modes/live.py
index a7d88c7..d209b85 100644
--- a/airpoint/modes/live.py
+++ b/airpoint/modes/live.py
@@ -3,6 +3,9 @@ from .. import hud
 from ..pointer import PRECLICK_RATIO
 
 
+FILTER_NAMES = {"full": "full", "smooth": "smooth only", "off": "OFF"}
+
+
 class LiveMode:
     """Classify -> vote -> state machine -> pointer -> dwell -> OS."""
 
@@ -38,7 +41,8 @@ class LiveMode:
                 frame,
                 f"{state} | pose {label} {conf:.2f} | ratio {ratio:.2f} | {s.clf.name} | {o.fps:.0f} fps",
                 f"last: {s.last_event} | mouse [m] {'ON' if s.runner.enabled else 'OFF'} | "
-                f"dwell [d] {'ON' if s.dwell_on else 'OFF'} | {hud.KEY_HINT}",
+                f"dwell [d] {'ON' if s.dwell_on else 'OFF'} | filter [b] {FILTER_NAMES[s.pointer.mode]} | "
+                f"{hud.KEY_HINT}",
                 s.runner.enabled,
                 line3=(sim, hud.ALERT) if sim else None,
             )
````

**Apply to `airpoint/hud.py`:**

````diff
diff --git a/airpoint/hud.py b/airpoint/hud.py
index 09388dd..909263a 100644
--- a/airpoint/hud.py
+++ b/airpoint/hud.py
@@ -15,7 +15,7 @@ ACCENT_ON, ACCENT_OFF, WHITE, MINT = (0, 200, 255), (90, 90, 90), (255, 255, 255
 PANEL, ALERT = (25, 35, 30), (60, 60, 255)
 MINI_SIZE = (320, 240)
 ZOOMS = (1.0, 1.5, 2.0)
-KEY_HINT = "keys q m d k c v z t r"
+KEY_HINT = "keys q k c v z t r"
 
 
 def canon_view(frame):
````

**Apply to `airpoint/app.py`:**

````diff
diff --git a/airpoint/app.py b/airpoint/app.py
index f21b1ed..d67dc7b 100644
--- a/airpoint/app.py
+++ b/airpoint/app.py
@@ -18,7 +18,8 @@ ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
 MODELS_DIR = os.path.join(ROOT, "models")
 MODEL_PATH, LABELS_PATH = os.path.join(MODELS_DIR, "pose_mlp.joblib"), os.path.join(MODELS_DIR, "labels.json")
 WINDOW, MINI_MARGIN = "AirPoint", 20
-KEYS = "q quit | m mouse | d dwell | k rules/mlp | c calibrate | v mini | z zoom | r record | 1-6 pose | space start/redo"
+KEYS = ("q quit | m mouse | d dwell | b filter full/smooth/off | k rules/mlp | c calibrate | v mini | z zoom | "
+        "t simulated tremor ([ ] size, { } speed) | r record | 1-6 pose | space start/redo")
 
 
 def load_classifiers(force_rules, profile=None):
````

- [x] **Step 4: Run to verify they pass, then the full suite**

Run: `venv/bin/pytest tests/test_pointer.py tests/test_session.py tests/test_hud.py -q && venv/bin/pytest -q`
Expected: the full suite ends with `99 passed`

- [x] **Step 5: Commit**

```bash
git add airpoint/app.py airpoint/hud.py airpoint/modes airpoint/pointer.py airpoint/session.py airpoint/tremor.py tests/test_hud.py tests/test_pointer.py tests/test_session.py
git commit -m "feat(tremor): notch canceller before the 1-Euro filter; b cycles full / smoothing only / off"
```

---

### Task 8: Shake readout in the banner and the mini label

A third banner line appears only when there is something to say: the simulator is on, or the meter is locked. It shows the measured shake and what is left after the filters, both measured live.

**Files:**
- Modify: `airpoint/modes/live.py`
- Modify: `tests/test_session.py`

- [x] **Step 1: Write the failing tests**

**Apply to `tests/test_session.py`:**

````diff
diff --git a/tests/test_session.py b/tests/test_session.py
index d5f5270..7c9c322 100644
--- a/tests/test_session.py
+++ b/tests/test_session.py
@@ -7,6 +7,8 @@ import numpy as np
 
 from airpoint.actions import ActionRunner, FakeMouse
 from airpoint.classifier import RuleClassifier
+from airpoint.modes.live import shake_line
+from airpoint.tremor import Reading, TremorSim
 from airpoint.session import Session
 from tests.hands import BY_POSE
 from tests.test_train import _write_user
@@ -276,3 +278,22 @@ def test_simulated_tremor_is_locked_and_cancelled_in_the_live_loop(tmp_path):
     assert s.pointer.mode == "off"
     feed(s, "palm", 0.1, t)
     assert s.pointer.out == s.obs.raw                        # off = the raw pointer
+
+
+def test_shake_line_says_what_is_measured_and_what_is_left():
+    sim, idle = TremorSim(), Reading(status="steady")
+    assert shake_line(sim, idle, idle) is None
+    sim.on = True
+    assert shake_line(sim, idle, idle) == "SIMULATED TREMOR 6.0 Hz | measuring..."
+    locked = Reading(status="tremor", amp=12.4, locked=True, lock_hz=6.1)
+    left = Reading(status="steady", amp=0.83)
+    assert shake_line(sim, locked, left) == "SIMULATED TREMOR 6.0 Hz | LOCKED 6.1 Hz: shake 12 px -> 0.8 px"
+    sim.on = False
+    assert shake_line(sim, locked, left) == "LOCKED 6.1 Hz: shake 12 px -> 0.8 px"
+
+
+def test_mini_label_carries_the_simulator_and_lock_tags(tmp_path):
+    s, _ = make(tmp_path)
+    s.on_key(ord("t"), 0.0)
+    feed(s, "palm", 6.0, 0.0)
+    assert " | SIM" in s.mini_label and " | LOCK " in s.mini_label
````

- [x] **Step 2: Run to verify they fail**

Run: `venv/bin/pytest tests/test_session.py -q`
Expected: `1 error` with `ImportError: cannot import name 'shake_line' from 'airpoint.modes.live'`

- [x] **Step 3: Implement**

**Apply to `airpoint/modes/live.py`:**

````diff
diff --git a/airpoint/modes/live.py b/airpoint/modes/live.py
index d209b85..51953b7 100644
--- a/airpoint/modes/live.py
+++ b/airpoint/modes/live.py
@@ -6,6 +6,19 @@ from ..pointer import PRECLICK_RATIO
 FILTER_NAMES = {"full": "full", "smooth": "smooth only", "off": "OFF"}
 
 
+def shake_line(sim, reading, left):
+    """Third banner line: what the tremor meter sees, and what is left after the filters.
+    None when there is nothing to say (no simulator, no lock)."""
+    parts = []
+    if sim.on:
+        parts.append(f"SIMULATED TREMOR {sim.hz:.1f} Hz")
+    if reading.locked:
+        parts.append(f"LOCKED {reading.lock_hz:.1f} Hz: shake {reading.amp:.0f} px -> {left.amp:.1f} px")
+    elif sim.on:
+        parts.append("measuring...")
+    return " | ".join(parts) or None
+
+
 class LiveMode:
     """Classify -> vote -> state machine -> pointer -> dwell -> OS."""
 
@@ -33,9 +46,10 @@ class LiveMode:
         if events:
             s.last_event = ", ".join(e.kind + ("" if e.arg is None else f" {e.arg}") for e in events)
         hud.draw_region(frame, s.runner.active_region, s.runner.enabled)
-        sim = f"SIMULATED TREMOR {s.sim.hz:.1f} Hz, {s.sim.amp:.2f} palm" if s.sim.on else None
+        shake = shake_line(s.sim, s.pointer.reading, s.pointer.left)
         s.mini_label = (f"{state} | {label} {conf:.2f} | mouse {'ON' if s.runner.enabled else 'OFF'}"
-                        + (" | SIM" if sim else ""))
+                        + (" | SIM" if s.sim.on else "")
+                        + (f" | LOCK {s.pointer.reading.lock_hz:.1f}Hz" if s.pointer.reading.locked else ""))
         if not s.mini:
             hud.draw_banner(
                 frame,
@@ -44,7 +58,7 @@ class LiveMode:
                 f"dwell [d] {'ON' if s.dwell_on else 'OFF'} | filter [b] {FILTER_NAMES[s.pointer.mode]} | "
                 f"{hud.KEY_HINT}",
                 s.runner.enabled,
-                line3=(sim, hud.ALERT) if sim else None,
+                line3=(shake, hud.ALERT if s.sim.on else hud.MINT) if shake else None,
             )
 
     def on_key(self, s, key, t):
````

- [x] **Step 4: Run to verify they pass, then the full suite**

Run: `venv/bin/pytest tests/test_session.py -q && venv/bin/pytest -q`
Expected: the full suite ends with `101 passed`

- [x] **Step 5: Commit**

```bash
git add airpoint/modes tests/test_session.py
git commit -m "feat(hud): shake readout — measured tremor and what is left, in the banner and mini label"
```

---

### Task 9: X-ray panel

`hud.draw_xray(height, taps)` draws a 400 px panel from a plain dict: probability bars with the vote gate, the vote buffer, the six states, a scope of pointer x (raw red, output green), the shake spectrum with the lock marker, and timings. `Session.taps()` reads those values from where they already live. Live mode now calls `proba` once per frame and derives the label from it. `tests/test_app.py` fakes the camera, tracker, OS mouse and window, so `app.main` itself is finally covered.

**Files:**
- Modify: `airpoint/hud.py`
- Modify: `airpoint/modes/live.py`
- Modify: `airpoint/session.py`
- Modify: `airpoint/app.py`
- Create: `tests/test_app.py`
- Modify: `tests/test_hud.py`
- Modify: `tests/test_session.py`

- [x] **Step 1: Write the failing tests**

**Create `tests/test_app.py`:**

````python
"""app.main wiring with the camera, tracker, OS mouse and window all faked."""
import numpy as np

from airpoint import app
from airpoint.actions import FakeMouse
from tests.hands import BY_POSE


class FakeCamera:
    index, size, fps = 0, (1920, 1080), 30.0

    def frames(self):
        for _ in range(8):
            yield np.zeros((1080, 1920, 3), np.uint8)

    def close(self):
        self.closed = True


class FakeTracker:
    def detect(self, frame):
        return BY_POSE["palm"], "Right"

    def close(self):
        pass


def test_main_runs_frames_through_the_session_and_quits_on_q(monkeypatch, capsys):
    shown, keys = [], iter([255, ord("x"), 255, ord("m"), 255, ord("q")])
    monkeypatch.setattr(app, "Camera", FakeCamera)
    monkeypatch.setattr(app, "HandTracker", FakeTracker)
    monkeypatch.setattr(app, "PyAutoGuiMouse", FakeMouse)
    monkeypatch.setattr(app.cv2, "imshow", lambda name, img: shown.append(img.shape))
    monkeypatch.setattr(app.cv2, "waitKey", lambda ms: next(keys))
    monkeypatch.setattr(app.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(app, "show_window", lambda mini, screen_w: None)

    app.main(["--rules"])

    assert len(shown) == 6                                  # q on the sixth frame ends the loop
    assert shown[0] == (540, 960, 3)                        # 1080p camera -> 640-wide view at zoom 1.5
    assert shown[2] == (540, 1560, 3)                       # x added the X-ray panel
    out = capsys.readouterr().out
    assert "1920x1080 @ 30 fps -> view 640x360" in out and "classifier rules" in out
````

**Apply to `tests/test_hud.py`:**

````diff
diff --git a/tests/test_hud.py b/tests/test_hud.py
index 22dfc1c..878c1cf 100644
--- a/tests/test_hud.py
+++ b/tests/test_hud.py
@@ -60,3 +60,39 @@ def test_banner_and_prompt_draw_inside_the_frame():
     frame = np.zeros((360, 640, 3), np.uint8)
     hud.draw_prompt(frame, "HOLD palm   7s", "tilt | kept 12", "SIGN LOST")
     assert frame[:86].any() and not frame[88:].any()      # top strip: never over the hand
+
+
+def _taps(**over):
+    from airpoint.tremor import Reading
+    x = dict(clf="mlp/ana", fps=30.0, ms_tracker=5.0, ms_classifier=0.2,
+             labels=["palm", "pinch", "fist", "v_sign", "thumb_up", "thumb_down"],
+             probs=np.array([0.02, 0.9, 0.03, 0.02, 0.02, 0.01]), gate=0.75,
+             votes=["palm", "pinch", "pinch", "pinch", "pinch"], label="pinch",
+             state="PRESSING", state_s=0.21, last_event="click", trail=[],
+             reading=Reading(), left=Reading(), mode="full", sim=False)
+    x.update(over)
+    return x
+
+
+def test_xray_panel_draws_at_both_common_heights_and_with_nothing_to_show():
+    for h in (360, 480):
+        assert hud.draw_xray(h, _taps()).shape == (h, hud.XRAY_W, 3)
+    empty = hud.draw_xray(360, _taps(probs=None, votes=[], label="none", state="RESTING"))
+    assert empty.shape == (360, 400, 3)
+
+
+def test_xray_panel_shows_the_winning_bar_the_active_state_and_the_curves():
+    from airpoint.tremor import Reading
+    freqs = np.arange(3.0, 12.01, 0.1)
+    power = np.exp(-((freqs - 6.0) ** 2) / 0.1)
+    trail = [(i / 30, (320 + 14 * np.sin(2 * np.pi * 6 * i / 30), 180.0), (320.0, 180.0)) for i in range(120)]
+    locked = Reading(status="tremor", hz=6.0, amp=12.0, fs=30.0, max_hz=12.0, locked=True, lock_hz=6.0,
+                     spectrum=(freqs, power))
+    left = Reading(status="steady", amp=0.8, fs=30.0, max_hz=12.0, spectrum=(freqs, power * 0.01))
+    panel = hud.draw_xray(360, _taps(trail=trail, reading=locked, left=left, sim=True))
+    plain = hud.draw_xray(360, _taps())
+    accent = np.array(hud.ACCENT_ON, np.uint8)
+    assert (panel[44:52, 100:260] == accent).all(axis=2).all()            # the pinch bar (gate tick is at x 265)
+    assert (panel[143:159, 270:390] == accent).all(axis=2).mean() > 0.5   # the PRESSING box is filled
+    assert (panel[217:263] != plain[217:263]).any()                        # the scope has curves
+    assert (panel[283:325] != plain[283:325]).any()                        # the spectrum has curves
````

**Apply to `tests/test_session.py`:**

````diff
diff --git a/tests/test_session.py b/tests/test_session.py
index 7c9c322..c467131 100644
--- a/tests/test_session.py
+++ b/tests/test_session.py
@@ -132,8 +132,8 @@ def test_the_pointer_keeps_sub_pixel_precision(tmp_path):
 class _Mlp:
     name, labels = "mlp", ["palm"]
 
-    def predict(self, pts):
-        return "palm", 1.0
+    def proba(self, pts):
+        return np.array([1.0])
 
 
 def test_k_switches_classifier_and_clears_the_vote_buffer(tmp_path):
@@ -297,3 +297,30 @@ def test_mini_label_carries_the_simulator_and_lock_tags(tmp_path):
     s.on_key(ord("t"), 0.0)
     feed(s, "palm", 6.0, 0.0)
     assert " | SIM" in s.mini_label and " | LOCK " in s.mini_label
+
+
+def test_x_adds_the_xray_panel_in_live_mode_only(tmp_path):
+    s, _ = make(tmp_path)
+    s.zoom = 1.0
+    s.on_key(ord("x"), 0.0)
+    assert s.view_dirty
+    t = feed(s, "palm", 0.5, 0.0)
+    assert s.tick(blank(), BY_POSE["palm"], "Right", t).shape == (360, 640 + 400, 3)
+    assert s.tick(blank(), None, None, t + DT).shape == (360, 1040, 3)          # no hand: still draws
+    s.on_key(ord("v"), t)
+    assert s.tick(blank(), BY_POSE["palm"], "Right", t + 2 * DT).shape == (240, 320, 3)   # mini wins
+    s.on_key(ord("v"), t)
+    s.on_key(ord("r"), t)
+    assert s.tick(blank(), BY_POSE["palm"], "Right", t + 3 * DT).shape == (360, 640, 3)   # not while recording
+    s.close()
+
+
+def test_xray_taps_report_the_live_pipeline(tmp_path):
+    s, _ = make(tmp_path)
+    t = feed(s, "palm", 0.5, 0.0)
+    feed(s, "pinch", 0.2, t)
+    x = s.taps()
+    assert x["labels"][int(np.argmax(x["probs"]))] == "pinch" and x["label"] == "pinch"
+    assert x["state"] == "PRESSING" and 0.0 < x["state_s"] < 0.2
+    assert x["votes"][-3:] == ["pinch"] * 3 and x["gate"] == 0.75
+    assert len(x["trail"]) == 21 and x["mode"] == "full" and not x["sim"]
````

- [x] **Step 2: Run to verify they fail**

Run: `venv/bin/pytest tests/test_hud.py tests/test_session.py tests/test_app.py -q`
Expected: `6 failed` with `AttributeError: module 'airpoint.hud' has no attribute 'draw_xray'`, `AttributeError: 'Session' object has no attribute 'taps'` and `AttributeError: '_Mlp' object has no attribute 'predict'`

- [x] **Step 3: Implement**

**Apply to `airpoint/hud.py`:**

````diff
diff --git a/airpoint/hud.py b/airpoint/hud.py
index 909263a..4d3a77c 100644
--- a/airpoint/hud.py
+++ b/airpoint/hud.py
@@ -1,4 +1,5 @@
 import cv2
+import numpy as np
 
 HAND_CONNECTIONS = [
     (0, 1), (1, 2), (2, 3), (3, 4),
@@ -15,7 +16,10 @@ ACCENT_ON, ACCENT_OFF, WHITE, MINT = (0, 200, 255), (90, 90, 90), (255, 255, 255
 PANEL, ALERT = (25, 35, 30), (60, 60, 255)
 MINI_SIZE = (320, 240)
 ZOOMS = (1.0, 1.5, 2.0)
-KEY_HINT = "keys q k c v z t r"
+XRAY_W = 400
+STATES = ("RESTING", "MOVING", "PRESSING", "DRAGGING", "HOLD", "SCROLLING")
+RAW_C, OUT_C, DIM, GRID_C = (80, 80, 255), (120, 230, 120), (150, 150, 150), (60, 70, 65)
+KEY_HINT = "keys q k c v z x t r"
 
 
 def canon_view(frame):
@@ -106,3 +110,75 @@ def mini_frame(frame, label):
     cv2.rectangle(small, (0, h - 24), (w, h), PANEL, -1)
     put_fit(small, label, (8, h - 8), 0.45, MINT)
     return small
+
+
+def _plot(panel, box, curves):
+    """Draw each (x 0..1, y 0..1, color) curve inside box = (left, top, width, height)."""
+    x0, y0, w, h = box
+    cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), GRID_C, 1)
+    for xs, ys, color in curves:
+        if len(xs) < 2:
+            continue
+        pts = np.stack([x0 + np.asarray(xs) * w, y0 + h - np.clip(ys, 0.0, 1.0) * h], axis=1)
+        cv2.polylines(panel, [pts.astype(np.int32)], False, color, 1, cv2.LINE_AA)
+
+
+def draw_xray(height, x):
+    """The X-ray panel: what the system saw and decided this frame. `x` is Session.taps()."""
+    panel = np.full((height, XRAY_W, 3), PANEL, np.uint8)
+    put_fit(panel, f"X-RAY | {x['clf']} | {x['fps']:.0f} fps | tracker {x['ms_tracker']:.0f} ms | "
+                   f"classifier {x['ms_classifier']:.1f} ms", (8, 15), 0.40, WHITE)
+
+    probs = x["probs"]                                         # 1. class probabilities
+    best = -1 if probs is None else int(np.argmax(probs))
+    for i, name in enumerate(x["labels"]):
+        y, p = 24 + 16 * i, 0.0 if probs is None else float(probs[i])
+        put_fit(panel, name, (8, y + 12), 0.40, MINT if i == best else DIM)
+        cv2.rectangle(panel, (100, y + 3), (100 + int(220 * p), y + 13), ACCENT_ON if i == best else ACCENT_OFF, -1)
+        put_fit(panel, f"{p:.2f}", (334, y + 12), 0.40, WHITE)
+    gate_x, y = 100 + int(220 * x["gate"]), 24 + 16 * len(x["labels"])
+    cv2.line(panel, (gate_x, 24), (gate_x, y), WHITE, 1)
+
+    votes = " ".join(v.replace("thumb_", "")[:6] for v in x["votes"])   # 2. the vote buffer
+    put_fit(panel, f"votes: {votes}  =>  {x['label']}", (8, y + 14), 0.40, WHITE)
+
+    y += 22                                                    # 3. the state machine
+    for i, name in enumerate(STATES):
+        bx, by, on = 8 + (i % 3) * 130, y + (i // 3) * 22, name == x["state"]
+        cv2.rectangle(panel, (bx, by), (bx + 124, by + 18), ACCENT_ON if on else ACCENT_OFF, -1 if on else 1)
+        put_fit(panel, name, (bx + 6, by + 13), 0.38, (0, 0, 0) if on else DIM)
+    put_fit(panel, f"in state {x['state_s']:.1f} s | last: {x['last_event']}", (8, y + 54), 0.40, WHITE)
+
+    y += 70                                                    # 4. scope: pointer x, raw vs output
+    put_fit(panel, "pointer x, last 4 s: raw red, output green", (8, y), 0.38, DIM)
+    trail = list(x["trail"])
+    curves = []
+    if len(trail) >= 2:
+        raw, out = np.array([r[0] for _, r, _ in trail]), np.array([o[0] for _, _, o in trail])
+        mid = float(np.median(out))
+        span = max(25.0, 1.2 * float(np.abs(raw - mid).max()))
+        ts = np.linspace(0.0, 1.0, len(trail))
+        curves = [(ts, (raw - mid) / (2 * span) + 0.5, RAW_C), (ts, (out - mid) / (2 * span) + 0.5, OUT_C)]
+    _plot(panel, (8, y + 4, XRAY_W - 16, 48), curves)
+
+    y += 66                                                    # 5. spectrum: raw vs output
+    r, left = x["reading"], x["left"]
+    put_fit(panel, "shake by frequency: raw red, output green", (8, y), 0.38, DIM)
+    box, curves = (8, y + 4, XRAY_W - 16, 44), []
+    if r.spectrum:
+        top = float(np.max(r.spectrum[1])) + 1e-12
+        for spec, color in ((r.spectrum, RAW_C), (left.spectrum, OUT_C)):
+            if spec:
+                curves.append(((spec[0] - 3.0) / 9.0, np.sqrt(spec[1] / top), color))
+    _plot(panel, box, curves)
+    for hz in (3, 6, 9, 12):
+        put_fit(panel, str(hz), (box[0] + int((hz - 3) / 9 * box[2]) - (12 if hz == 12 else 0), y + 60), 0.34, DIM)
+    if r.locked:
+        lx = box[0] + int((r.lock_hz - 3.0) / 9.0 * box[2])
+        cv2.line(panel, (lx, box[1]), (lx, box[1] + box[3]), ACCENT_ON, 1)
+        put_fit(panel, f"{r.lock_hz:.1f} Hz", (min(lx + 4, XRAY_W - 60), box[1] + 12), 0.38, ACCENT_ON)
+
+    foot = (f"shake {r.amp:.1f} px -> {left.amp:.1f} px | filter {x['mode']} | "       # 6. numbers
+            f"sees up to {r.max_hz:.0f} Hz" + (" | SIMULATED" if x["sim"] else ""))
+    put_fit(panel, foot, (8, y + 77), 0.40, ALERT if x["sim"] else WHITE)
+    return panel
````

**Apply to `airpoint/modes/live.py`:**

````diff
diff --git a/airpoint/modes/live.py b/airpoint/modes/live.py
index 51953b7..955c7ba 100644
--- a/airpoint/modes/live.py
+++ b/airpoint/modes/live.py
@@ -1,3 +1,7 @@
+import time
+
+import numpy as np
+
 from .. import features as F
 from .. import hud
 from ..pointer import PRECLICK_RATIO
@@ -24,12 +28,20 @@ class LiveMode:
 
     name = "live"
 
+    def __init__(self):
+        self.probs, self.label, self.clf_ms = None, "none", 0.0     # kept for the X-ray panel
+
     def tick(self, s, o, frame):
         if o.pts is None:
-            label, conf, ratio = s.smoother.update(None, 0.0), 0.0, 1.0
+            self.probs, label, conf, ratio = None, s.smoother.update(None, 0.0), 0.0, 1.0
         else:
-            raw_label, conf = s.clf.predict(o.pts)
-            label, ratio = s.smoother.update(raw_label, conf), F.pinch_ratio(o.pts)
+            t0 = time.perf_counter()
+            self.probs = s.clf.proba(o.pts)
+            self.clf_ms = (time.perf_counter() - t0) * 1000.0
+            i = int(np.argmax(self.probs))
+            conf = float(self.probs[i])
+            label, ratio = s.smoother.update(s.clf.labels[i], conf), F.pinch_ratio(o.pts)
+        self.label = label
         state, events = s.fsm.update(label, o.t)
         smooth = None
         if o.raw is not None:
````

**Apply to `airpoint/session.py`:**

````diff
diff --git a/airpoint/session.py b/airpoint/session.py
index 02a2220..4436c7a 100644
--- a/airpoint/session.py
+++ b/airpoint/session.py
@@ -1,6 +1,8 @@
 import os
 from dataclasses import dataclass
 
+import numpy as np
+
 from . import features as F
 from . import hud
 from .classifier import PoseClassifier, VoteSmoother
@@ -43,13 +45,14 @@ class Session:
         self.pointer = PointerPipeline(fc_min)
         self.sim = TremorSim()
         self.dwell_on, self.mini, self.view_dirty = dwell_on, False, False
-        self.zoom = 1.5
+        self.zoom, self.xray, self.ms_tracker = 1.5, False, 0.0
         self.recorder, self.rejected = None, 0
         self.mode = LiveMode()
         self.last_event, self.mini_label, self.prev_t = "none", "", None
         self.obs = None
 
-    def tick(self, frame, pts, hand, t):
+    def tick(self, frame, pts, hand, t, tracker_ms=0.0):
+        self.ms_tracker = tracker_ms
         fps = 0.0 if self.prev_t is None else 1.0 / max(t - self.prev_t, 1e-5)
         self.prev_t = t
         view = hud.canon_view(frame)
@@ -62,10 +65,22 @@ class Session:
             raw = (float(pts[F.MIDDLE_MCP, 0] * w), float(pts[F.MIDDLE_MCP, 1] * h))
         self.obs = Obs(t, pts, hand, raw, w, h, fps)
         self.mode.tick(self, self.obs, view)
-        if self.mini and self.mode.name == "live":
-            return hud.mini_frame(view, self.mini_label)
+        if self.mode.name == "live":
+            if self.mini:
+                return hud.mini_frame(view, self.mini_label)
+            if self.xray:
+                view = np.hstack([view, hud.draw_xray(h, self.taps())])
         return hud.zoomed(view, self.zoom)
 
+    def taps(self):
+        """Everything the X-ray panel shows, read from where it already lives."""
+        p, live = self.pointer, self.mode
+        return dict(clf=self.clf.name, fps=self.obs.fps, ms_tracker=self.ms_tracker, ms_classifier=live.clf_ms,
+                    labels=self.clf.labels, probs=live.probs, gate=self.smoother.min_conf,
+                    votes=[label for label, _ in self.smoother.buf], label=live.label,
+                    state=self.fsm.state, state_s=self.obs.t - self.fsm.t_enter, last_event=self.last_event,
+                    trail=p.trail, reading=p.reading, left=p.left, mode=p.mode, sim=self.sim.on)
+
     def set_mode(self, mode):
         self.mode.close(self)
         self.mode = mode
@@ -94,6 +109,8 @@ class Session:
         elif key == ord("z"):
             self.zoom = hud.ZOOMS[(hud.ZOOMS.index(self.zoom) + 1) % len(hud.ZOOMS)]
             self.view_dirty = True
+        elif key == ord("x"):
+            self.xray, self.view_dirty = not self.xray, True
         elif key == ord("b"):
             self.pointer.cycle_mode()
         elif key == ord("t") and self.mode.name == "live":
````

**Apply to `airpoint/app.py`:**

````diff
diff --git a/airpoint/app.py b/airpoint/app.py
index d67dc7b..aaffa0f 100644
--- a/airpoint/app.py
+++ b/airpoint/app.py
@@ -18,7 +18,7 @@ ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
 MODELS_DIR = os.path.join(ROOT, "models")
 MODEL_PATH, LABELS_PATH = os.path.join(MODELS_DIR, "pose_mlp.joblib"), os.path.join(MODELS_DIR, "labels.json")
 WINDOW, MINI_MARGIN = "AirPoint", 20
-KEYS = ("q quit | m mouse | d dwell | b filter full/smooth/off | k rules/mlp | c calibrate | v mini | z zoom | "
+KEYS = ("q quit | m mouse | d dwell | b filter full/smooth/off | k rules/mlp | c calibrate | v mini | z zoom | x x-ray | "
         "t simulated tremor ([ ] size, { } speed) | r record | 1-6 pose | space start/redo")
 
 
@@ -80,8 +80,9 @@ def main(argv=None):
     try:
         for frame in camera.frames():
             t = time.time()
+            t0 = time.perf_counter()
             pts, hand = tracker.detect(frame)
-            shown = s.tick(frame, pts, hand, t)
+            shown = s.tick(frame, pts, hand, t, tracker_ms=(time.perf_counter() - t0) * 1000.0)
             if s.view_dirty:
                 show_window(s.mini, runner.screen_w)
                 s.view_dirty = False
````

- [x] **Step 4: Run to verify they pass, then the full suite**

Run: `venv/bin/pytest tests/test_hud.py tests/test_session.py tests/test_app.py -q && venv/bin/pytest -q`
Expected: the full suite ends with `106 passed`

- [x] **Step 5: Commit**

```bash
git add airpoint/app.py airpoint/hud.py airpoint/modes airpoint/session.py tests/test_app.py tests/test_hud.py tests/test_session.py
git commit -m "feat(hud): X-ray panel (x) — probabilities, votes, state machine, pointer scope, shake spectrum"
```

---

### Task 10: Docs, full suite, push

The README gains the tremor section and the new keys, `run.txt` the key line and a demo recipe, and the spec is brought in line with the four choices made while building (see "Deviations" at the top).

**Files:**
- Modify: `README.md`
- Modify: `run.txt`
- Modify: `docs/superpowers/specs/2026-11-25-airpoint-stage1-tremor-xray-design.md`
- Modify: this plan (tick the boxes)

- [x] **Step 1: Apply the docs changes**

**Apply to `README.md`:**

````diff
diff --git a/README.md b/README.md
index c369f44..b81dda8 100644
--- a/README.md
+++ b/README.md
@@ -10,6 +10,7 @@ Team: Ishan Gupta, Dakshmeet Singh, Madhuram Rathi, Bhuvi Kansal.
 ```
 webcam → MediaPipe Hands (21 points) → wrist-centred, palm-scaled 63-d vector
        → pose classifier (MLP, rules fallback) → 5-frame vote → state machine → OS mouse
+cursor : middle knuckle → tremor canceller → One-Euro filter → screen
 ```
 
 | Pose | Action |
@@ -22,6 +23,23 @@ webcam → MediaPipe Hands (21 points) → wrist-centred, palm-scaled 63-d vecto
 | open palm, still 1 s (dwell mode, `d`) | click with no finger motion at all |
 | hand out of frame | rest, anything held is dropped |
 
+### Tremor
+
+A smoothing filter has to choose between shake and lag, and a rhythmic tremor fools it: the
+filter reads the shake as fast movement and opens up. AirPoint measures the shake instead. A
+meter watches the last three seconds of pointer movement, and when it finds one strong frequency
+between 3 and 12 Hz three times in a row it locks on. A notch filter then removes only that band,
+before the One-Euro filter. In simulation at this project's settings that leaves 4 to 6 times
+less shake than smoothing alone, and dwell clicking keeps working under a tremor that defeats
+smoothing.
+
+`t` adds a labelled, simulated tremor so a steady hand can show this live; `b` switches the
+filters (full, smoothing only, off) for a comparison on the real cursor; `x` opens a panel with
+the class probabilities, the state machine, the pointer before and after, and the shake spectrum.
+
+Limits: the numbers come from simulation, there is no patient data, a 30 fps webcam cannot see
+tremor above about 12 Hz, and none of this is a medical measurement.
+
 ## Run
 
 ```bash
@@ -32,6 +50,7 @@ venv/bin/python run_demo.py                      # live demo; m = mouse on/off,
 venv/bin/python run_demo.py --mouse              # start with the mouse already on
 venv/bin/python run_demo.py --user you           # loads profiles/you.json once calibrated; c = calibrate (~1 min)
 #   d = dwell click on/off | v = mini always-on-top view | c = hold still 5 s, 6 poses x 5 s, retrain, hot-swap
+#   t = simulated tremor ([ ] size, { } speed) | b = filter full / smoothing only / off | x = X-ray panel | z = zoom
 venv/bin/python run_demo.py --record --user you  # record your hand: space = all poses, 1-6 = one pose
 venv/bin/python -m airpoint.train                # retrain on data/*.csv, leave-one-user-out report
 venv/bin/pytest -q                               # unit tests (no camera needed)
@@ -45,7 +64,8 @@ macOS: allow your terminal under System Settings → Privacy & Security → Came
 ## Layout
 
 ```
-airpoint/        package — camera, tracker, features, filters, classifier, fsm, actions, hud, recorder, calibration, train, app
+airpoint/        package — camera, tracker, features, filters, tremor, pointer, classifier, fsm, actions, hud,
+                 recorder, calibration, train, session, modes/ (live, record, calibrate), app
 data/            recorded landmark CSVs, one per person
 models/          trained pose classifier + labels
 profiles/        per-user calibration: filter cutoff + personal model (gitignored)
````

**Apply to `run.txt`:**

````diff
diff --git a/run.txt b/run.txt
index 5f09f44..884b473 100644
--- a/run.txt
+++ b/run.txt
@@ -6,7 +6,8 @@ open index.html
 # 2. Benchmarks (writes results.js for the dashboard, ~5 s)
 venv/bin/python airpoint_offline_training_filter_fitts_law_benchmark_suite.py
 
-# 3. Live demo (q quit | m mouse | d dwell | k rules/mlp | c calibrate | v mini | r record | 1-6 pose | space start/redo)
+# 3. Live demo (q quit | m mouse | d dwell | b filter | k rules/mlp | c calibrate | v mini | z zoom | x x-ray |
+#               t simulated tremor, [ ] size, { } speed | r record | 1-6 pose | space start/redo)
 venv/bin/python run_demo.py                    # MLP if models/pose_mlp.joblib exists, else rules
 venv/bin/python run_demo.py --mouse            # same, OS mouse control already ON
 venv/bin/python run_demo.py --rules            # force rules
@@ -14,6 +15,8 @@ venv/bin/python run_demo.py --user <name>      # loads profiles/<name>.json (fc_
 venv/bin/python run_demo.py --record --user <name>   # record poses to data/<name>.csv
 #    palm = move | pinch/fist < 0.5 s = click | hold = drag | v-sign 0.3 s = right click
 #    thumb up / down = scroll | hand out of frame = rest
+#    tremor demo: m (mouse on) -> t (cursor shakes, tag reads SIMULATED) -> wait 2 s for LOCKED ->
+#                 b twice (smoothing only, then off) to compare on the real cursor -> x to show why
 
 # 4. Train on data/*.csv -> models/
 venv/bin/python -m airpoint.train
````

**Apply to `docs/superpowers/specs/2026-11-25-airpoint-stage1-tremor-xray-design.md`:**

````diff
diff --git a/docs/superpowers/specs/2026-11-25-airpoint-stage1-tremor-xray-design.md b/docs/superpowers/specs/2026-11-25-airpoint-stage1-tremor-xray-design.md
index 6d9938a..5191b4d 100644
--- a/docs/superpowers/specs/2026-11-25-airpoint-stage1-tremor-xray-design.md
+++ b/docs/superpowers/specs/2026-11-25-airpoint-stage1-tremor-xray-design.md
@@ -1,7 +1,7 @@
 # AirPoint Stage 1 — Tremor switch, tremor canceller, X-ray view
 
 Date: 2026-11-25
-Status: design, awaiting user review
+Status: approved 2026-11-25; built per `docs/superpowers/plans/2026-11-25-airpoint-stage1-tremor-xray.md`
 Roadmap: `2026-11-25-airpoint-v2-roadmap.md`, Stage 1
 Builder: Madhuram + Claude
 
@@ -26,6 +26,7 @@ and a frame loop split into parts that eight more stages can extend.
 | `{` `}` | simulated tremor slower / faster |
 | `b` | pointer filter: full → smoothing only → off → full |
 | `x` | X-ray panel beside the camera view |
+| `z` | zoom of the full view: 1×, 1.5×, 2× |
 
 - **full** = tremor canceller + 1-Euro smoothing (new default).
 - **smoothing only** = today's behaviour.
@@ -120,8 +121,11 @@ Design:
   Stage 9 browser port can share them.
 - `beta` becomes 0.030 in that space (0.010 × 1920 / 640). This reproduces the
   Week 1 cursor feel exactly.
-- The full-view window becomes resizable; the HUD is laid out for the 640 canvas.
-  A test measures every HUD string with `cv2.getTextSize` so none overflows.
+- The view is shown at 1.5× by default and `z` cycles 1×, 1.5×, 2× (decided in the
+  plan: dragging an OpenCV window on macOS scales unpredictably). The HUD is laid out
+  for the 640 canvas; text shrinks to fit rather than run off the edge, and a test
+  keeps the worst-case strings legible. Record and calibrate prompts sit in a top
+  strip so they never cover the hand.
 
 ### 4.2 Frame loop in parts (no behaviour change)
 
@@ -153,8 +157,9 @@ modes/calibrate.py CalibSession + Recorder + Trainer + prompt
   slow frequency wander of about ±0.45 Hz, 25% amplitude modulation (the model the
   evidence in §3 used). Phase is integrated
   step by step, so changing the frequency live is smooth.
-- Amplitude is in palm lengths (default 0.15, range 0.05–0.45), so it looks the
-  same at any distance. Frequency default 6.0 Hz, range 3.5–10.
+- Amplitude is in palm lengths (default 0.20, range 0.05–0.60), so it looks the
+  same at any distance. 0.20 is the "moderate" row of §3 at a typical palm size.
+  Frequency default 6.0 Hz, range 3.5–10.
 - Live mode only. Entering record or calibrate turns it off: simulated shake never
   reaches recorded data or a profile.
 
@@ -165,7 +170,9 @@ modes/calibrate.py CalibSession + Recorder + Trainer + prompt
   (upper edge capped at 0.42 × frame rate), amplitude = RMS across both axes,
   frequency = strongest peak of the windowed spectrum, prominence = peak over the
   band median.
-- An estimate counts as tremor when amplitude ≥ 1.5 px and prominence ≥ 6.
+- An estimate counts as tremor when amplitude ≥ 1.5 px and prominence ≥ 8. (The
+  prototype showed 6 falsely locks about 2% of the time on noisier landmarks; 8
+  gave none, with the same lock-on time.)
 - Locked after three consecutive tremor estimates agreeing within 1 Hz; released
   after three consecutive without. Frequency is smoothed while locked.
 - The band's upper edge follows the frame rate: at 16.5 fps it sees tremor only
````

- [x] **Step 2: Tick the boxes of Tasks 1–10 in this plan**

- [x] **Step 3: Import check and full suite**

Run: `venv/bin/python -c "import airpoint.app" && venv/bin/pytest -q`
Expected: `106 passed`

- [x] **Step 4: Commit and push**

```bash
git add README.md run.txt docs/superpowers/specs/2026-11-25-airpoint-stage1-tremor-xray-design.md docs/superpowers/plans/2026-11-25-airpoint-stage1-tremor-xray.md
git commit -m "docs: stage 1 keys, tremor canceller, spec matched to what was built"
git push
```

---

### Task 11: Camera session (user)

Only a person at the camera can do this. It also covers the three Week 2 checks that were never run; those
should not be judged before Task 4 is in, because their pixel thresholds were three times too strict.

Run: `venv/bin/python run_demo.py --user madhuram`

- [ ] **0. Launch line** reads `camera 0 1920x1080 @ 30 fps -> view 640x360`. The banner shows about 30 fps, also in a dim room.
- [ ] **1. Feel.** `m` turns the mouse on. Cursor speed and smoothness match Week 1. `z` steps the window size.
- [ ] **2. Calibration (Week 2).** `c`: hold still, six poses, "Learning your hand...", then "done" with a percentage. The prompt strip is at the top and never covers the hand. `profiles/madhuram.json` exists; relaunching shows `mlp/madhuram`.
- [ ] **3. Dwell (Week 2).** `d`, hold the palm still over an icon: the ring fills in about a second, one click, no repeat until the hand moves away and stops again.
- [ ] **4. Mini view (Week 2).** `v`: 320×240 in the top-right corner, staying above a full-screen browser.
- [ ] **5. Simulated tremor.** `t`: the cursor shakes and a red line reads SIMULATED TREMOR. Within about 2 seconds it reads LOCKED near 6 Hz and the cursor steadies. `]` makes the shake bigger, `}` faster; the lock follows.
- [ ] **6. Filter comparison.** `b` once (smoothing only): a visible wobble returns. `b` again (OFF): the raw shake. `b` again: full.
- [ ] **7. A real shake.** Simulator off (`t`). Shake your own hand on purpose for 5 seconds. Does LOCKED appear, and at what frequency?
- [ ] **8. False locks.** Two minutes of normal use: move, click, drag, scroll. Note any LOCKED flash.
- [ ] **9. X-ray.** `x`: the bars follow your signs, the state boxes light up, the scope and spectrum move, and the frame rate stays near 30.
- [ ] **10. Guard rails.** In record (`r`) and calibrate (`c`) the `t` key does nothing.

Report back: anything that fails, the frequency from check 7, and any false locks from check 8. Those three
decide whether `full` stays the default filter mode and whether the meter's thresholds need tuning.

---

## Self-review

- **Spec coverage.** §2 keys `t [ ] { } b x` → Tasks 5, 7, 9 (`z` added in Task 4). §3 evidence → the same model lives in `tests/pointer_sim.py` and the claims are regression tests in Task 7 (under half the shake; dwell ≥ 95% with the canceller and under 50% without; 16.5 fps still helps). §4.1 camera pin → Task 1; canonical space, `beta` 0.030, HUD for 640 → Task 4. §4.2 loop in parts with headless flow tests → Task 3. §4.3 simulator, live mode only → Task 5. §4.4 meter, including the frame-rate cap and "too slow" → Task 6. §4.5 canceller, filter modes, identical output with no lock → Task 7. §4.6 X-ray and HUD → Tasks 8, 9. §4.7 `proba` → Task 2. §5 labels and guard rails → Tasks 5, 8, 9. §6 tests → Tasks 2–9. §7 camera session → Task 11. §8 docs → Task 10.
- **Placeholders.** None: every step carries its code, its command and the output observed on the prototype.
- **Type consistency.** `Session.tick(frame, pts, hand, t, tracker_ms=0.0)` gains its last parameter in Task 9 and `app.py` passes it from then on. `Obs` fields `t pts hand raw w h fps` are fixed in Task 3 and only `raw` changes meaning (frame pixels → canonical floats) in Task 4. `PointerPipeline.update(raw, t, preclick=False)` keeps its signature from Task 3 to Task 7; `.out` appears in Task 4, `.mode .reading .left .trail .cycle_mode()` in Task 7. `Reading` fields `status hz amp fs max_hz locked lock_hz spectrum` are defined in Task 6 and read by the canceller (Task 7), `shake_line` (Task 8) and `draw_xray` (Task 9). `hud.draw_banner(frame, line1, line2, active, line3=None)` and `hud.draw_prompt(frame, title, subtitle, alert=None)` take their final shape in Task 4.
