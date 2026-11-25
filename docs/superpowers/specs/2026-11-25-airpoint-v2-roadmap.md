# AirPoint v2 — Roadmap: every feature, in stages

Date: 2026-11-25
Status: order approved by the user 2026-11-25; each stage still needs its own spec approval
Decision behind it (user, 2026-11-25): "time is not an issue, do all properly".

This is not a build spec. It splits the full feature list into stages. Each stage
gets its own design spec, implementation plan, build, and camera check. The system
stays demo-able after every stage.

Relation to the 2026-11-13 spec:
- Replaces its Week 3 row (§10): hotkey poses become user-taught signs (Stage 4);
  the dashboard card moves to Stage 2.
- Reopens part of its "Out of scope" list (§2): the Fitts target harness, a user
  test, head mode and voice mode are now in scope.
- Still out: ISL fingerspelling, dynamic (sequence) gestures, Windows/Linux testing,
  two-hand zoom (target users rarely have two free hands), webcam eye-gaze (too
  coarse), an on-screen keyboard of our own (the OS one is opened instead).

## 1. Feature inventory → stage

| Asked for | Stage | Note |
|---|---|---|
| Tremor switch | 1 | simulated tremor, filter A/B, live gauge, adaptive smoothing |
| X-ray view | 1 | live confidence bars, state machine, trails, timings |
| Practice game | 2 | web pages driven by the real cursor: tutorial + standard test |
| Real numbers on the dashboard | 2 | measured values replace the typed-in pilot values |
| Comfort box | 3 | reach step in calibration |
| Choose your poses | 3 | roles in the profile |
| Either hand | 3 | mirrored to one canonical hand |
| Pause sign | 3 | lock state; trigger is NOT a fist hold (that is drag) |
| Feedback and hints | 3 | sounds, "move closer", "too dark", "hand lost" |
| Teach a sign live | 4 | new class, retrain, hot-swap, bind to an action |
| Speak for me | 4 | action = speak a phrase |
| Glow at the cursor | 5 | screen overlay |
| Point-only menu | 5 | the wheel on the overlay; merged with the row above |
| Face mode | 6 | nose pointer, expressions as signs |
| Hold to talk / talk to type | 7 | merged; local speech-to-text, plus spoken commands |
| Voice plus AI that clicks | 8 | the only feature that needs internet |
| Try it by QR in a browser | 9 | in-browser hand tracking on a hosted page |

## 2. Stages

Status (kept current): all ten stages built 2026-11-25. Stage 7's on-laptop speech engine is written
but not installed (that download is the user's call). Stage 8's model was measured on drawn screens
only. Stage 9's page is not published (switching on GitHub Pages is the user's call). Stage 10's
documents and the final dashboard pass are done (`docs/endsem/`); its measurement day and the backup
video are the user's. From Stage 2 on the user approved every stage in advance and tests them all in
one session; that session's checklist is `docs/camera-checklist.md`.

Size is relative: S, M, L. "You" = work only the user can do (camera, people).

### Stage 1 — Tremor switch, tremor canceller, X-ray view (M)
Design: `2026-11-25-airpoint-stage1-tremor-xray-design.md`.
- First task: restructure the frame loop in `app.py` into a session, a pointer
  pipeline and one object per mode (live, record, calibrate). No behaviour change;
  the existing 55 tests plus new headless flow tests guard it.
- One coordinate space: everything after the tracker works in a 640-wide canonical
  frame, so pixel thresholds mean the same on any camera. Measured: the camera is
  1920 × 1080 (Week 2's thresholds assumed 640 wide, so they are three times too
  strict as shipped) and idles at 15 fps in dim light unless asked for 30. The
  camera is pinned to 30 fps and the 1-Euro speed gain is rescaled once.
- `tremor.py`: a simulator (whole-hand shake added to the landmarks, live mode
  only, always labelled), a meter (frequency and amplitude of the shake, with
  lock-on), and a canceller (a notch at the locked frequency, before the 1-Euro
  filter).
- Changed from the first draft: "smoothing adapts by itself" became the canceller.
  A simulation study showed that lowering the smoothing cutoff barely touches a
  periodic tremor, while removing only the tremor band leaves 4–6× less shake and
  makes dwell clicking work under severe tremor.
- Keys: simulator on/off, size, speed; filter mode full / smoothing only / off;
  X-ray panel.
- X-ray panel: probability bars, vote buffer, state machine, pointer scope,
  spectrum, timings.
- You: one camera session after the build, which also covers the three Week 2
  checks never run.
- Limits to state: results are simulated; a 30 fps webcam caps measurable tremor
  near 12 Hz; this is not a medical measurement.

### Stage 2 — Practice pages + real numbers (M)
- `python -m airpoint.lab`: a local server (standard library) that serves two pages
  and saves raw trial data to `results/trials.jsonl`.
- Tutorial page: move, click, double-click, drag a box, scroll to a line,
  right-click. Teaches a new user in about two minutes; also the safe demo sandbox.
- Test page: ISO 9241-411 multi-directional tapping (ring of targets, three
  difficulty levels). The page records raw events only. Python computes effective
  throughput (We = 4.133·SD of endpoints, IDe = log2(De/We + 1), TP = IDe/MT) and
  error rate, so the maths is unit-tested.
- Works for every input because AirPoint moves the real OS cursor: mouse, trackpad,
  hand + pinch, hand + dwell, simulated tremor with filter off/on, later wheel and
  face. The condition is a label chosen at the start.
- `python -m airpoint.results` merges trials and `models/train_report.json` into
  `results.js`. Dashboard: dataset size and leave-one-user-out per pose; measured
  throughput and error per condition with participant count. The typed-in pilot
  movement times in the benchmark script and their dashboard card are removed.
- You: run the test yourself for pilot numbers. Teammates on measurement day
  (Stage 10).

### Stage 3 — Fits your body (M)
- Profile v2: one JSON per user for everything personal (model, `fc_min`, comfort
  box, hand, roles, bindings, dwell settings, sounds). Versioned; v1 upgrades on load.
- Choose your poses: the FSM takes roles (move, grip, right click, scroll up/down)
  from the profile instead of module constants. A role can be remapped or switched
  off; calibration offers "skip" for poses the user could not hold.
- Comfort box: a reach step in calibration ("move as far as is comfortable") gives
  a robust bounding box → `active_region`. Higher gain amplifies shake, so the
  smoothing floor scales with it.
- Either hand: left hands are mirrored to one canonical hand before features, with
  mirrored copies in training. One model serves both.
- Pause: a lock state in the runner; while locked nothing reaches the OS. Unlock by
  holding an open palm still (ring shown) or a key. Triggers: a key now, a wheel
  item and a bindable sign in Stages 4–5.
- Hints: a pure function from frame and hand statistics to messages (too dark, move
  closer, too close, move to centre, show your hand, sign not clear, slow computer).
- Sounds: click, drag, lock (system sounds through `afplay`), with a toggle.

### Stage 4 — Teach a sign + Speak for me (M)
- Teach mode: name it, hold it for a short capture, pick an action, personal retrain
  with the new class, hot-swap, show hold-out accuracy for the new sign, warn if it
  collides with an existing sign. Forget and list signs.
- Custom rows live in `data/<user>_signs.csv` (ignored by git); bindings live in
  the profile. The capture guard passes labels it has no rule for.
- Actions: key combo, type text, speak a phrase (`say`; Indian English and Hindi
  voices are installed on this Mac), open an app or URL, open the OS on-screen
  keyboard, pause. One dispatch table behind the same master switch as the mouse.
- Custom signs fire through the FSM's existing one-shot hold path (`hold_events`).
- Risk: a random hand shape read as a custom sign. Longer hold for custom signs
  plus the confidence gate; false fires per minute measured on an idle clip.

### Stage 5 — Glow at the cursor + point-only wheel (L)
- Overlay: a separate process (AppKit is already installed with pyautogui) drawing
  a click-through, always-on-top glow that follows the real cursor: dwell progress,
  click ripple, state tag (DRAG, SCROLL, LOCKED, SIMULATED), hint pill. If it cannot
  start, AirPoint runs without it.
- Wheel: a pure state machine. In point-only mode a dwell opens it at the anchor:
  click (default if you stay still), double click, right click, drag (held until
  the next dwell), scroll, pause. Pick by direction plus a short hold.
- Mac-only. Day one of the stage is a spike: window level over full-screen apps,
  Retina coordinates, second display, process lifecycle.

### Stage 6 — Face mode (M)
- A face source (MediaPipe FaceLandmarker with blendshapes; present in the
  installed MediaPipe; small model download). Head turn drives the pointer through
  the same comfort-box mapping. Expressions map to the same roles (mouth open =
  grip, brows up = right click), thresholds calibrated per user and stored in the
  profile.
- Same smoother, FSM, dwell, wheel, filter and effects: one brain, another body part.
- Measured with the Stage 2 test against hand and mouse.
- Risks: glasses, lighting, talking opens the mouth (lock or hold-to-talk must
  suppress it), neck fatigue.

### Stage 7 — Hold to talk (M)
As built (see the stage spec): tap to talk instead of hold to talk; macOS dictation is the default
engine and local Whisper the optional one, since its package and model need a download.
- A speech service on its own thread records only while the talk sign or wheel item
  is held, transcribes on the laptop with Whisper, then either runs a command from a
  short fixed list (click, double click, right click, scroll, enter, undo, select
  all) or types the text.
- Engine chosen by a spike on this M5 (faster-whisper vs mlx-whisper). English
  first; a multilingual model is the option for Hindi.
- Risks: room noise at the demo, accent accuracy. Push-to-talk limits both; the OS
  on-screen keyboard action from Stage 4 is the fallback.

### Stage 8 — AI helper (L)
- Hold the talk sign and say a task. The request text and a screenshot go to an AI
  model that proposes one step at a time. Each step is shown on the overlay, then
  runs through the same effect runner. Risky steps (send, delete, pay, anything off
  an allow-list) wait for a confirm. Any hand movement or the stop sign takes
  control back. Step cap per task.
- The only feature that sends anything off the laptop (screenshots and request
  text). Off by default, switched on per session, with a visible mark. The key comes
  from the environment and never enters the repo.
- Needs internet, an API key, Screen Recording permission. Provider chosen by the
  user on 2026-11-25: OpenAI, key read from `OPENAI_API_KEY` in the environment
  (the variable may hold several keys separated by `;`). The exact model id, cost
  and the tool loop are settled in this stage's own spec. The key is never written
  to this repo.

### Stage 9 — Try it in your browser (L)
- A hosted page (GitHub Pages on the public repo): MediaPipe hand tracking in the
  browser, a JavaScript port of the features, filter and a subset of the state
  machine, MLP weights exported to JSON, a virtual cursor inside the page running
  the Stage 2 tutorial and test. The visitor sees their own score. Nothing is uploaded.
- A parity fixture exported from Python keeps the two runtimes giving the same answers.
- Needs Pages enabled on the repo (user's OK at that point) and a QR code on the slides.

### Stage 10 — Wrap up (M, mostly you)
- Measurement day: teammates record their hands (base model retrain, true
  leave-one-user-out) and run every mode through the Stage 2 test.
- Final dashboard, report with measured numbers and honest limits, slides, viva
  guide, three-minute demo script, backup video, performa.

## 3. Architecture direction

```
sources     hand | face  → observation: pointer, sign label, confidence, probabilities
            (tremor simulator wraps the pointer signal)
decide      vote smoother → state machine (+ dwell, wheel, lock) → events
pointer     1-Euro filter (cutoff from the tremor meter) → comfort box → screen
effects     mouse | keys | type | speak | sound | open     one master switch
profile     one JSON per user: everything personal
modes       live | record | calibrate | teach               one object each
surfaces    camera window (full, mini, X-ray) | screen overlay process | dashboard
services    speech-in thread | AI helper thread             both emit events
lab, web    practice pages (local server) | hosted try-it page
```

Rules that carry through every stage:
- Timing and decision logic lives in pure classes with pytest, no camera.
- Mouse and every effect that touches the OS start OFF; held buttons are released
  on any exit; the AI helper has its own extra gates.
- Camera frames and audio never leave the laptop. Stage 8 is the single, opt-in
  exception for screenshots and request text.
- macOS first. Overlay, speech out and sounds are Mac-only and guarded; the core
  stays portable.
- Per stage: spec → plan → test-first build → camera checklist → README, run.txt,
  viva guide → push. Commits carry no co-author trailer.

## 4. Decisions made here
- The practice game is a web page, not a camera-window game: one test measures every
  input fairly, and Stage 9 reuses it.
- Raw trials are recorded in the page; all maths is in Python.
- Pause is not a fist hold: a held fist is already drag.
- The overlay is a separate process, so it never fights OpenCV's own window loop.
- Speech-to-text is local Whisper. Apple's Speech framework is avoided unless the
  Stage 7 spike shows it works from a terminal-launched script. (Stage 7 changed this:
  macOS dictation, driven by its shortcut, is the default; local Whisper is optional.)
- The AI helper comes after voice and the overlay, because it needs both.

## 5. Honest positioning (for the viva)
macOS already ships Dwell Control, Head Pointer and Voice Control. AirPoint differs
by: hand signs learned per person, signs taught in seconds, tremor measured and
filtered adaptively, one pipeline across hand and face, measured results, open
source on any webcam.

Related work to cite; verify each reference before it enters the report:
ISO 9241-411 and Soukoreff & MacKenzie (2004) for throughput; Casiez et al. (2012)
1-Euro filter; Google Project Gameface (2023) for face-gesture pointing on
MediaPipe; Bolt (1980) "Put-That-There" for voice plus pointing; Wobbrock et al.
(2009) Angle Mouse and Trewin et al. (2006) Steady Clicks for pointing aids.

## 6. Owned by the user
- Camera sessions per stage (Stage 1 also covers the Week 2 checks).
- Teammates' recordings and measurement day.
- End-sem date and the department's required format; performa answers.
- Stage 8: API key. Stage 9: OK to enable GitHub Pages.
