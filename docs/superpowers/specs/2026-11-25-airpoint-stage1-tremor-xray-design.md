# AirPoint Stage 1 — Tremor switch, tremor canceller, X-ray view

Date: 2026-11-25
Status: design, awaiting user review
Roadmap: `2026-11-25-airpoint-v2-roadmap.md`, Stage 1
Builder: Madhuram + Claude

## 1. Goal

Show, live and honestly, what AirPoint does for a shaking hand, and make it do
more. A presenter with a steady hand presses one key and the real cursor shakes
like a tremor hand. AirPoint measures the shake, locks onto its frequency within
two seconds, and removes it. A second key opens a panel that shows the system
thinking: class probabilities, the state machine, the pointer signal before and
after, the tremor spectrum.

Two enabling changes ride along: one coordinate space for all pixel thresholds,
and a frame loop split into parts that eight more stages can extend.

## 2. What the user sees

| Key | Effect |
|---|---|
| `t` | simulated tremor on/off. A red SIMULATED tag shows in every view while on. |
| `[` `]` | simulated tremor smaller / larger |
| `{` `}` | simulated tremor slower / faster |
| `b` | pointer filter: full → smoothing only → off → full |
| `x` | X-ray panel beside the camera view |

- **full** = tremor canceller + 1-Euro smoothing (new default).
- **smoothing only** = today's behaviour.
- **off** = raw pointer, for the A/B on the real cursor.

Live banner gains: `filter full|smooth|OFF`, a shake readout
(`shake 12 px @ 6.0 Hz → 0.8 px`), and the SIMULATED tag. The mini view label
gains `SIM` when the simulator is on.

X-ray panel (400 px wide, right of the camera view):
1. probability bar per class, winner lit, tick at the 0.75 vote gate
2. the five-frame vote buffer and the smoothed label
3. the six states, active one lit, seconds in state, last event
4. scope: pointer x over the last 4 s, raw in red, output in green
5. spectrum 3–12 Hz, raw vs output, tremor peak marked
6. numbers: tracker ms, classifier ms, fps, native and canonical frame size

Demo flow: `t` → cursor shakes → about 2 s later the tag reads LOCKED 6.0 Hz and
the cursor steadies → `b` twice shows today's smoothing and raw for comparison →
`x` explains why.

## 3. Evidence (simulation, 2026-11-25)

A design spike compared pointer pipelines on synthetic sessions: 30 fps with
timestamp jitter, minimum-jerk reaches and holds, landmark noise 1 px, tremor with
frequency wander ±0.45 Hz, 25% amplitude modulation and a 20% second harmonic.
Units are pixels of a 640-wide frame; error is against the true intended path.
"Dwell works" = the existing `DwellDetector` clicks within 1.6 s of the hand
stopping. `beta` is the 1-Euro speed gain; the real value is fixed in the build
(§4.1), so both are shown.

| Tremor | beta | Today: shake left | Today: dwell works | Canceller: shake left | Canceller: dwell works |
|---|---|---|---|---|---|
| none | .010 | 0.43 px | 100% | 0.43 px (identical) | 100% |
| moderate, 14 px 6 Hz | .010 | 2.5 px | 100% | 0.75 px | 100% |
| moderate, 14 px 6 Hz | .020 | 3.4 px | 100% | 0.86 px | 100% |
| severe, 25 px 5 Hz | .010 | 6.7 px | 56% | 1.3 px | 100% |
| severe, 25 px 5 Hz | .020 | 9.6 px | 15% | 1.6 px | 100% |
| very severe, 40 px 6 Hz | .020 | 16.4 px | 0% | 2.8 px | 98% |
| very severe, 40 px 4.5 Hz | .020 | 21.1 px | 0% | 3.2 px | 98% |

What it showed:
- Plain smoothing cannot remove a periodic tremor: the 1-Euro speed term reads the
  shake as fast movement and opens the filter. Lowering `fc_min` from measured
  jitter (the Week 2 calibration mapping, and the roadmap's "adaptive smoothing")
  barely helps: 3.4 → 2.7 px moderate, 9.6 → 8.5 px severe.
- Removing only the tremor band does: 3–7× less shake, and dwell clicking goes from
  failing to working under severe tremor.
- Cost: in fast reaches the cursor trails the hand about a third more (10.5 → 14.2
  px RMS at beta .010). Arrival on target is not delayed.
- Lock-on takes about 1.7 s, release about 3 s. On a noisy signal with no tremor
  the meter was falsely locked under 1% of the time; a false lock only adds that
  small reach lag.
- Tried and dropped: an LMS sine canceller (worse on holds), easing the canceller
  off during fast motion (tremor fools the speed gate), a narrow notch for slow
  tremor (far worse at large amplitude), smoothing driven by leftover jitter (no gain).

All of this is simulation. Real hands are checked in the camera session (§7).

## 4. Design

### 4.1 One coordinate space

Finding: the camera's real frame size was never measured. Week 2's pixel
thresholds (dwell 15 / 30 px, jitter 3 → 12 px) were written for a 640-wide frame,
and the 1-Euro speed gain was tuned by feel at whatever the camera delivers.

- The tracker keeps reading the native frame, so detection is unchanged.
- Everything after it works in a canonical space 640 wide (height keeps the camera's
  aspect): pointer pixels, thresholds, filters, HUD drawing. The view is the native
  frame resized to that size. Thresholds then mean the same on any camera, and the
  Stage 9 browser port can share them.
- `beta` is rescaled once so the Week 1 feel is unchanged: `0.010 × native_width /
  640`, hard-coded after the user reports the native width (build task 1). If the
  camera is already 640 wide nothing changes.
- The full-view window becomes resizable; the HUD is laid out for the 640 canvas.
  A test measures every HUD string with `cv2.getTextSize` so none overflows.
- Launch line prints native and canonical frame size.

### 4.2 Frame loop in parts (no behaviour change)

```
app.py            args, camera, tracker, window, the loop; nothing else
session.py        Session: shared state, tick(frame, pts, hand, t) -> image to show,
                  global keys, mode switching, classifier hot-swap
pointer.py        PointerPipeline: canceller → 1-Euro, filter mode, meters, trails
modes/live.py     classify → vote → state machine → pointer → dwell → effects → banner
modes/record.py   RecordSession + Recorder + prompt
modes/calibrate.py CalibSession + Recorder + Trainer + prompt
```

- A mode has `tick(session, obs, view)`, `on_key(session, key, t) -> bool`, `close()`.
- `Session` receives its parts (runner, classifiers, profile, root folder, a
  window callback, a trainer factory), so tests build one with `FakeMouse`, a blank
  image and the synthetic hands, with no camera and no real OS calls.
- This is the first build task. The 55 existing tests stay green, and new headless
  tests drive whole flows: palm → pinch → palm gives one click; dwell; record writes
  rows; calibration walks its phases and hot-swaps; mode switches release the button.

### 4.3 Tremor simulator — `tremor.py`

- Adds one offset to all 21 landmarks before anything reads them, so the skeleton
  and the pointer both show a shaking hand. Features are
  wrist-centred, so sign recognition is unaffected (a test asserts `normalize_63`
  is unchanged).
- Signal: fundamental plus 20% second harmonic, y at 0.6× with a phase offset,
  slow frequency wander of about ±0.45 Hz, 25% amplitude modulation (the model the
  evidence in §3 used). Phase is integrated
  step by step, so changing the frequency live is smooth.
- Amplitude is in palm lengths (default 0.15, range 0.05–0.45), so it looks the
  same at any distance. Frequency default 6.0 Hz, range 3.5–10.
- Live mode only. Entering record or calibrate turns it off: simulated shake never
  reaches recorded data or a profile.

### 4.4 Tremor meter — `tremor.py`

- Keeps the last 3 s of raw pointer samples with their timestamps. Every 10th
  frame: resample to a uniform grid, zero-phase Butterworth band-pass 3–12 Hz
  (upper edge capped at 0.42 × frame rate), amplitude = RMS across both axes,
  frequency = strongest peak of the windowed spectrum, prominence = peak over the
  band median.
- An estimate counts as tremor when amplitude ≥ 1.5 px and prominence ≥ 6.
- Locked after three consecutive tremor estimates agreeing within 1 Hz; released
  after three consecutive without. Frequency is smoothed while locked.
- Below 20 fps it reports "camera too slow" and nothing engages.
- A second meter on the pipeline output gives the leftover shake for the readout.
- Uses `scipy.signal`, already installed with scikit-learn.

### 4.5 Tremor canceller and filter modes — `tremor.py`, `pointer.py`

- A second-order notch per axis at the locked frequency, Q 1.5, coefficients from
  the measured frame rate. It runs on every sample so its state is always warm; its
  output is blended in over 0.3 s when the meter locks and out when it releases.
- Chain: raw → canceller → 1-Euro (constants unchanged apart from §4.1) → cursor.
  The pre-click slowdown stays.
- With no lock the chain is exactly today's. A test asserts identical output.
- `b` cycles full / smoothing only / off. `off` bypasses both filters but keeps
  feeding them, so switching back is seamless.
- The Week 2 calibration still sets `fc_min` from still-hand jitter. It stays, for
  non-periodic jitter; §3 shows it is not the answer to periodic tremor.

### 4.6 X-ray view and HUD — `hud.py`

- `draw_xray(panel, taps)` draws the panel from a plain snapshot: probabilities,
  vote buffer, state and time in state, last event, 4 s of raw and output pointer,
  two spectra, timings, frame sizes. The session composes `[view | panel]`.
- Mini view wins over X-ray while it is on.
- Banner and prompts are re-laid for the 640 canvas (§4.1).

### 4.7 Classifier taps — `classifier.py`

- `PoseClassifier.proba(landmarks)` returns probabilities in label order; `predict`
  is derived from it. `RuleClassifier.proba` returns a one-hot vector. One model
  call per frame, as today.

## 5. Safety and honesty

- The simulator is always labelled, in full, mini and X-ray views.
- Simulated shake cannot enter recordings, calibration or profiles (§4.3).
- Mouse still starts OFF. Mode switches still release a held button.
- The readout is "shake", in pixels and Hz. It is not a medical measurement, and a
  30 fps webcam cannot see tremor above about 12 Hz. The simulator shakes the whole
  hand; it does not shake fingers against each other, so it exercises the pointer
  path, not sign recognition.

## 6. Tests (pytest, no camera)

- Session flows (§4.2), including: filter mode cycle; simulator forced off on
  entering record or calibrate; classifier swap resets the vote buffer.
- Simulator: displacement scales with palm length; features unchanged; off = identity.
- Meter, on synthetic sessions from a shared helper (`tests/pointer_sim.py`, the
  spike's generator): locks within 2.5 s on 14 px 6 Hz, frequency within 0.4 Hz of
  the true value, amplitude within 25%; never locks on a no-tremor session; releases within 4 s;
  "camera too slow" under 20 fps.
- Canceller: on the moderate session, shake left is under half of smoothing only;
  with no tremor the output equals smoothing only exactly; under the severe session
  dwell works ≥ 95% with the canceller and under 70% without.
- `proba` sums to 1 and agrees with `predict`; rules give one-hot.
- HUD: every string fits the 640 canvas; X-ray panel draws at 360 and 480 high.

Expected suite: about 85 tests, from 55.

## 7. Camera session (user)

Before the build, once:
- Run the camera check script and report its line (native frame size, capture fps).

After the build:
1. Week 2 checks never run: calibration end to end, dwell click, mini view on top.
2. Feel check: cursor speed and smoothness match Week 1.
3. `t`: cursor shakes, SIMULATED shows, lock within about 2 s, cursor steadies.
4. `b`: the three filter modes look clearly different on the real cursor.
5. Shake your own hand on purpose, simulator off: does the meter lock on a real shake?
6. Two minutes of normal use: note any false lock.
7. `x`: panel readable; fps not lower than before.

## 8. Docs

README (keys, a short "how the tremor canceller works", references), `run.txt`,
the roadmap's Stage 1 text. Dashboard numbers move in Stage 2.

## 9. Decisions

- Canceller instead of "smoothing adapts by itself": the simulation showed the
  second barely works (§3).
- A fixed-Q notch driven by the meter, over an LMS canceller: same steady state,
  simpler, better with wandering tremor.
- Default filter mode is full: with no lock it is identical to today. The camera
  session decides whether it stays the default.
- Tracker keeps the native frame; only the maths and the view move to 640.
- Simulator is live-mode only.
- No new command-line flags, no profile changes (profile v2 is Stage 3).

## 10. Limits to state in the report

- Results in §3 are simulated. No patient data exists for this project.
- Real tremor is less regular than the simulator; the canceller will remove less
  of it, and smoothing remains the fallback.
- Related work to cite after verifying each reference: Casiez et al. (2012) 1-Euro
  filter; Widrow et al. (1975) adaptive noise cancelling and Riviere et al. (1998)
  tremor cancelling for the notch idea.
