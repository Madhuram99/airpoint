# AirPoint — Learned Pose Classifier, Full Gesture Set, Calibration

Date: 2026-11-13
Status: approved design, awaiting implementation plan
Builder: Madhuram + Claude; teammates record hand data and test
Deadline: demo in 2–3 weeks (early December 2026)

## 1. Goal

Mentor asked for more gestures. Today the live demo reads two signs by hand-coded
ratios (pinch, thumb-scroll). Adding fist, V-sign, thumb direction, dwell and
hotkeys to a rule set does not scale: the project's own benchmark scores the rules
at 85.8%, and every new sign forces a retune of every old threshold.

This design replaces the ratio rules with the 63-input MLP that already exists in
the benchmark script, trained on real recorded hands, running in the live loop,
with a state machine that maps pose + time to mouse actions. A sub-minute
calibration retrains the classifier on the current user's hand and tunes the
1-Euro filter to their tremor.

Side effect: it fulfils report Milestone 1 (≥5k real frames) and Milestone 2
(per-user calibration) and the slide-4 gesture table.

## 2. Scope

### Pose vocabulary (classifier classes)

`palm` · `pinch` · `fist` · `v_sign` · `thumb_up` · `thumb_down`

Two non-class outputs from the smoother: `none` = no hand detected; `unknown` =
hand present but confidence below gate. They differ in the FSM: `none` means the
hand is gone (rest, drop holds); `unknown` holds the current state and keeps the
cursor following, so a confidence dip mid-drag never drops the drag.

Week 3 may add 2–3 hotkey poses (candidates: `point`, `three`); chosen then.

### Pose + time → action

| Pose, timing | Action |
|---|---|
| `palm` | move cursor (middle knuckle, 1-Euro filtered) |
| `pinch` or `fist` released < 0.5 s | left click |
| two clicks inside the OS double-click window | double click (OS detects; no separate event) |
| `pinch` or `fist` held ≥ 0.5 s | drag until release |
| `v_sign` held 0.3 s | right click, fires once; re-arms after `palm` |
| `thumb_up` / `thumb_down` held | scroll up / down continuously; cursor frozen; no hand motion needed |
| `palm`, cursor still 1.0 s, dwell mode on | dwell click; re-arms after cursor moves 30 px |
| `none` | rest; drop anything held |
| week 3: hotkey pose held 0.3 s | `Hotkey(name)`, same one-shot pattern as right click |

Pinch ratio stays a continuous signal beside the class label: ratio < 0.42 lowers
the filter cutoff (pre-click slowdown), as today.

### Out of scope

Fitts target harness, user study, NASA-TLX, ISL fingerspelling, head/voice modes,
dynamic (sequence) gestures, Windows/Linux testing. The mismatch between report
numbers and benchmark output is a report fix, not code.

Stretch (week 3, only if ahead): two-hand pinch-zoom — distance between the two
hands' pinch points, static per-frame math, no sequence model.

## 3. Structure

The single 370-line script becomes a package. Each module has one job and is
testable without a camera.

```
airpoint/
  camera.py      open camera with index fallback (1 then 0), yield flipped BGR frames
  tracker.py     MediaPipe HandLandmarker wrapper → 21 landmarks (+ handedness) or None
  features.py    normalize_63 (wrist origin, palm length = 1), pinch_ratio
  filters.py     OneEuroFilter
  classifier.py  PoseClassifier (MLP via joblib) and RuleClassifier, same interface
                 → (label, confidence); VoteSmoother
  fsm.py         (pose, ratio, cursor, t) → (state, events); pure, injected clock
  actions.py     events → pyautogui; FakeMouse for tests
  hud.py         skeleton, banner, pose bar, dwell ring, prompts; full and mini windows
  recorder.py    record mode → data/<user>.csv
  train.py       data/*.csv → models/pose_mlp.joblib + labels.json; LOUO report
  app.py         wiring, key handling, CLI flags
run_demo.py      thin entry → app.main()
data/            one csv per person
models/          trained classifier
profiles/        per-user json (fc_min, model path)
```

Benchmark script and `index.html` stay. The old live script is deleted once
`run_demo.py` reproduces it.

Housekeeping before any edit:
- `git init` with a `.gitignore` for `venv/`, `__pycache__/`, `results.js`.
  Move the folder from `~/Downloads/Major Project` to `~/Projects/airpoint`.
- Stop the second Claude session editing this folder.

## 4. Data and training

### Recorder — `run_demo.py --record --user <name>`

- HUD names the pose, 3 s countdown, then captures 10 s (~300 frames) while the
  prompt says "tilt, move closer, move away".
- Keys `1`–`6` jump to a pose; `space` redoes the current one.
- Row: `user, pose, ts, handedness, x0, y0, z0 … x20, y20, z20` — raw MediaPipe
  normalized-image coordinates. Normalization runs at train and inference time, so
  it can change without re-recording.
- 4 teammates × 6 poses ≈ 7k frames, ~1.5 min per person.

### Training — `python -m airpoint.train`

- Normalize → augment (in-plane tilt ±34°, Gaussian landmark noise σ 0.12 palm
  units — lifted from the benchmark) → `MLPClassifier(hidden=(128, 64), relu,
  adam)`, the benchmark's net.
- Reported number is leave-one-user-out: each fold tests on a hand never seen.
  Prints a per-pose precision/recall table and the dataset size.
- Writes `models/pose_mlp.joblib` and `models/labels.json`. Runs in under a second.
- Also writes the summary into `results.js` for the dashboard card (section 7).

### Live inference — `classifier.py`

- `PoseClassifier.predict(landmarks) → (label, confidence)`: normalize →
  `predict_proba` → argmax. ~0.1 ms per frame.
- `VoteSmoother`: majority label over the last 5 frames; if the winner's mean
  confidence < 0.75 → `unknown`; no landmarks → `none`. Label change latency
  ≈ 3 frames (100 ms at 30 fps).
- `RuleClassifier`: today's ratio rules extended with fist, V-sign and thumb
  direction, behind the same interface. Used when no model file exists, and
  toggled live with `k` for the rules-vs-MLP demo beat.

## 5. Control logic — `fsm.py`

`update(pose, ratio, cursor_xy, t) → (state, events)`. No I/O; the clock is a
parameter so tests script pose sequences and assert events.

Events: `Move(x, y)`, `Click`, `RightClick`, `DragStart`, `DragEnd`,
`Scroll(direction)`, `Hotkey(name)`.

**Grip** = `pinch` or `fist`. The FSM treats them as one class so a pinch that
drifts into a fist mid-drag does not drop the drag (hysteresis in pose space).

| State | While in it | Leaves when |
|---|---|---|
| RESTING | nothing | any pose → MOVING |
| any | `unknown` → stay, cursor keeps following | — |
| MOVING | cursor follows; dwell timer if mode on | grip → PRESSING · `v_sign` / hotkey pose → HOLD(pose) · `thumb_up`/`thumb_down` → SCROLLING(dir) · `none` → RESTING |
| PRESSING | cursor still follows | grip released < 0.5 s → `Click` → MOVING · held ≥ 0.5 s → `DragStart` → DRAGGING |
| DRAGGING | button down, cursor follows | grip released → `DragEnd` → MOVING · hand lost → `DragEnd` → RESTING |
| HOLD(pose) | one-shot timer | 0.3 s reached → fires (`RightClick` or `Hotkey`) once, then waits for `palm` before re-arming · released early → MOVING, no event |
| SCROLLING(dir) | cursor frozen; `Scroll(dir)` every 100 ms | pose changes → MOVING |

Double click: two `Click`s inside the OS double-click window. A delayed single
click would add latency to every click; a `doubleClick()` after an already-fired
`Click` makes a triple click. The HUD labels two fast clicks "DOUBLE" for show.

Dwell: in MOVING with dwell mode on, cursor speed < 15 px/s for 1.0 s → `Click`,
then disarmed until the cursor moves 30 px from the click spot. HUD ring fills
over the second.

Cursor mapping: inner box of the frame (`ACTIVE_REGION`) → full screen, as today.

## 6. Calibration — `c` key

Under a minute, no restart:

1. **Still phase, 5 s.** Open palm, hold still. Standard deviation of the raw
   cursor in px maps to `fc_min`: std ≤ 3 px → 1.0 Hz; std ≥ 12 px → 0.3 Hz;
   linear between. β stays 0.010.
2. **Pose phase, 6 × 5 s.** Same recorder code; rows go to `data/<user>_calib.csv`.
3. **Retrain.** Base dataset + this user's rows weighted ×3 (row duplication) →
   new MLP → atomically swapped into the live classifier. HUD: "Learning your
   hand… done, NN% on your frames" (held-out split of the calibration rows).
4. **Profile.** `profiles/<user>.json` holds `fc_min` and the model path.
   `run_demo.py --user <name>` loads it.

Failure: retrain error keeps the old model and shows the error on the HUD.

## 7. HUD and demo surface — `hud.py`

- **Full view** (today's window, upgraded): skeleton, red raw / green filtered
  dots, banner with state, pose label + confidence bar, last event, FPS,
  `rules|mlp` and `mouse on|off` badges, dwell ring, record/calibration prompts
  with countdown.
- **Mini view** (`v`): 320×240, always on top (`cv2.WND_PROP_TOPMOST`), screen
  corner, so the audience sees what the system reads while real apps are driven.
- **Keys:** `m` mouse · `d` dwell · `k` rules↔mlp · `c` calibrate · `v` mini/full
  · `r` record · `q` quit. Printed on launch.
- **Dashboard:** one new card in `index.html` — real dataset size and
  leave-one-user-out accuracy per pose, read from `results.js`. No other
  dashboard work.

## 8. Safety and error handling

- Hand lost, `q`, or any exception → `DragEnd` and `mouseUp` in a `finally`. The
  OS button is never left down.
- Low confidence → `unknown` → no new action, current state kept.
- Moving the folder breaks `venv/` (absolute paths in `pyvenv.cfg` and script
  shebangs); recreate it after the move, before anything else.
- Camera: try index 1 then 0; otherwise exit with the permissions message.
- Missing model file → `RuleClassifier`, banner says so.
- Calibration retrain failure → old model kept, error shown.
- Mouse mode starts OFF on every launch; `m` toggles only with the video window
  focused.

## 9. Tests (pytest, no camera)

- `features`: wrist at origin, palm length 1, invariant to translation and scale.
- `VoteSmoother`: 1–2 frame flicker does not change the label; low confidence →
  `none`.
- `fsm`: scripted `(pose, t)` sequences → exact event lists. Cases: short pinch →
  `Click`; held → `DragStart`/`DragEnd`; pinch→fist mid-drag keeps dragging;
  `v_sign` fires once and re-arms only after `palm`; dwell arms/disarms; hand lost
  mid-drag → `DragEnd`; scroll emits at the fixed rate and stops on pose change.
- `actions`: `FakeMouse` records calls; `DragEnd` without `DragStart` is a no-op.
- `train`: a two-user fixture csv trains and scores ≥ 90% held-out. Guards the
  pipeline, not accuracy.
- Camera, tracker, HUD: manual.

## 10. Order of work

| Week | Build | Demo-able at end |
|---|---|---|
| 1 | git init + move · package split, old script retired · recorder · record 4 teammates · `train.py` · MLP live + vote · FSM with grip/HOLD/SCROLLING · full slide-4 table | all 6 poses drive the real mouse; `k` shows rules vs MLP |
| 2 | dwell mode · calibration (tremor → fc_min, retrain, hot-swap, profile) · mini HUD · tests | "teach my hand" live; drive a browser with the mini HUD visible |
| 3 | 2–3 hotkey poses · two-hand zoom if ahead · dashboard card · rehearsed demo script | full show |

Week 1 is the risk: recording needs the teammates in one room with the laptop.
Schedule that first; everything after depends on the data.

## 11. Decisions log

- MLP over more rules: rules benchmark at 85.8% and conflict as signs grow.
- Store raw landmarks, normalize at use: normalization can change without
  re-recording.
- Leave-one-user-out as the reported accuracy: the 99.9% on augmented copies of 20
  rows is not a generalization number.
- Grip class in the FSM: pinch/fist flicker must not drop a drag.
- No `DoubleClick` event: latency or triple-click, either way worse than letting
  the OS detect it.
- Rules classifier kept behind the same interface: zero-dependency fallback and a
  live comparison on stage.
- Dynamic gestures deferred: slide 10 already places ISL in the 8th semester.
