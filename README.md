# AirPoint

Computer access through simple hand signs and an ordinary webcam, for people who find a mouse hard or painful to use (tremor, arthritis, post-stroke, cerebral palsy). Runs on a laptop CPU; no video leaves the machine.

Major Project, Department of CSE, Punjab Engineering College, Chandigarh. Mentor: Mrs. Rupali.
Team: Ishan Gupta, Dakshmeet Singh, Madhuram Rathi, Bhuvi Kansal.

## How it works

```
webcam → MediaPipe Hands (21 points) → wrist-centred, palm-scaled 63-d vector
       → pose classifier (MLP, rules fallback) → 5-frame vote → state machine → OS mouse
cursor : middle knuckle → tremor canceller → One-Euro filter → screen
```

| Pose | Action |
|---|---|
| open palm | move cursor (One-Euro filtered, tremor removed) |
| pinch / fist, short | click · two quick = double click |
| pinch / fist, held | drag |
| V-sign, held 0.3 s | right click |
| thumb up / down | scroll, no hand motion needed |
| open palm, still 1 s (dwell mode, `d`) | click with no finger motion at all |
| hand out of frame | rest, anything held is dropped |

### Fits your body

- **Comfort box.** Calibration (`c`) asks you to move your hand as far as is comfortable. That box, not
  the whole frame, then maps to the whole screen, so an elbow can stay on the desk. The pointer filters
  follow the box's gain.
- **Choose your poses.** Which pose clicks, right-clicks and scrolls is a per-user role map. Calibration
  drops any pose your hand could not hold, and if no clicking pose is left it turns dwell click on.
  `python -m airpoint.profile you --role right_click=fist --role grip=pinch` edits the map by hand.
- **Either hand.** Training adds a mirrored copy of every example, so one model reads both hands and both
  faces of a hand. It does not rely on MediaPipe's left/right label, which follows the hand's facing.
- **Pause.** `p` stops everything reaching the system. Hold an open palm still for two seconds to resume.
- **Hints and sounds.** A strip at the bottom says what is in the way (show your hand, too dark, move
  closer, sign not clear). Clicks, drags and the pause each have a system sound. `h` and `s` switch them.

### Point only

For a hand that can point but cannot make a sign. `d` alone gives dwell click: hold still for a second
and it clicks. `w` turns the dwell into a wheel around the cursor: stay put and it clicks; move toward
talk, right click, drag, scroll, double click or pause and hold that direction for half a second. Drag presses
the button and lets go at the next dwell; scroll follows how far the hand is above or below where it
stopped. Choice is by direction, so it works the same whatever the comfort box. If calibration finds
no clicking sign the hand can make, it turns both on.

On macOS a glow follows the real cursor (a second, click-through window): the dwell countdown, a
ripple on each action, DRAG / SCROLL / PAUSED / LISTENING / SIMULATED tags, the current hint, and the wheel. `o`
switches it; `--no-overlay` starts without it. The wheel is also drawn in the camera view.

### Face mode

`f` hands the pointer to the face: turning the head moves the cursor, opening the mouth clicks (held
open, it drags), raising the eyebrows right-clicks, and the wheel is switched on for scroll, drag and
pause. Everything after the tracker is shared with the hand: the same vote, state machine, tremor
canceller, dwell, wheel, glow and lab. `c` in face mode calibrates in fifteen seconds: rest, turn the
head as far as is comfortable, open the mouth, raise the eyebrows. Without calibration, wherever the
head rests in its first moment is taken as the middle of the screen. Talking opens the mouth, so pause
(`p` or the wheel) before speaking to someone; while AirPoint itself is listening (Talk, below) the
mouth and eyebrows are ignored.

Face mode uses MediaPipe's FaceLandmarker (`face_landmarker.task`, in the repo beside the hand model).
On this laptop it runs at 30 fps, about 5 ms per frame.

### Teach a sign

`n` (with `--user`) adds a sign of your own in about fifteen seconds: type a name, hold the sign for
eight seconds, then choose what it does: speak a phrase aloud, press a key combo, type text, open an
app or a link, pause AirPoint, or start and stop talk. The personal model is retrained with the new class and swapped in
while you wait. Held for 0.6 s the sign fires once and re-arms after a neutral pose.

Speaking works even with mouse control off, so the same sign vocabulary can serve someone who needs
AirPoint to talk for them: `python -m airpoint.profile you --voice Rishi` picks the voice (Indian
English and Hindi voices ship with macOS). To reach the system's on-screen keyboard, bind a sign to
`option+cmd+f5`. On the name screen `1`–`9` forgets a taught sign.

The result line after teaching reports how much of the sign's own held-out frames were recognised,
and warns if it is easily confused with another sign or if other signs are being read as it.

### Tremor

A smoothing filter has to choose between shake and lag, and a rhythmic tremor fools it: the
filter reads the shake as fast movement and opens up. AirPoint measures the shake instead. A
meter watches the last three seconds of pointer movement, and when it finds one strong frequency
between 3 and 12 Hz three times in a row it locks on. A notch filter then removes only that band,
before the One-Euro filter. In simulation at this project's settings that leaves 4 to 6 times
less shake than smoothing alone, and dwell clicking keeps working under a tremor that defeats
smoothing.

`t` adds a labelled, simulated tremor so a steady hand can show this live; `b` switches the
filters (full, smoothing only, off) for a comparison on the real cursor; `x` opens a panel with
the class probabilities, the state machine, the pointer before and after, and the shake spectrum.

Limits: the numbers come from simulation, there is no patient data, a 30 fps webcam cannot see
tremor above about 12 Hz, and none of this is a medical measurement.

### Measuring it

`python -m airpoint.lab` serves two pages on this machine only. The tutorial walks a new user
through move, click, double-click, drag, scroll and right-click. The test is the ISO 9241-411
multi-directional tapping task: nine targets on a ring, three difficulty levels, 27 selections.
The page records raw events; Python computes effective throughput (Soukoreff & MacKenzie, 2004),
error rate and two path measures. Because AirPoint moves the real cursor, the same test scores a
mouse, a trackpad and every AirPoint mode. While AirPoint runs it reports its own settings to the
lab, so a block is labelled with what was really driving the cursor.

The dashboard shows only what `python -m airpoint.results` can back: accuracy on the recorded
hands with the method named, the hand-written rules on the same frames, the tremor simulation
labelled as a simulation, and measured pointing results. Anything not yet measured says so.

### Talk

For someone who cannot type. One tap starts listening and the same tap ends it: a sign taught for it
(`n`, action 6), the top of the wheel, or `l`. It is tap to talk, not hold to talk, because holding a
sign for a whole sentence is what these users find hard; listening also ends by itself after twenty
seconds. Click where the words should go first, and keep the mouse switch on.

There are two engines, chosen per profile:

- **macOS dictation (default).** AirPoint only presses the dictation shortcut; the system listens and
  types, on the laptop, in English, Hindi and the other languages macOS knows. Set it up once:
  System Settings › Keyboard › Dictation › on, and Shortcut › Customize › `control+option+d`
  (or tell AirPoint yours: `python -m airpoint.profile you --dictation-keys cmd+shift+d`).
- **On this laptop (optional).** AirPoint records the microphone only between the two taps, and
  recognises it with faster-whisper. This engine also understands spoken commands: click, double
  click, right click, scroll up, scroll down, enter, new line, tab, escape, delete, undo, copy, paste,
  select all, pause. Anything else is typed. English only. It needs one install and one model
  download, neither of which AirPoint does by itself:

  ```bash
  venv/bin/pip install faster-whisper
  venv/bin/python -m airpoint.speech --setup        # says the size and asks before downloading (about 148 MB)
  venv/bin/python -m airpoint.profile you --speech local
  venv/bin/python -m airpoint.speech --listen 5     # check: what five seconds of your voice are heard as
  ```

No sound is saved, and none leaves the laptop. Keystrokes go out on their own thread, in order, so a
long sentence never freezes the camera loop; and keys AirPoint types are never read back as its own
commands when its window happens to be in front.

### AI helper

For a whole errand in one go: "reply to Asha that I will come at five". An AI model (OpenAI's
`gpt-6.1-sol`) looks at a picture of the screen and proposes one step at a time; AirPoint checks the
step, shows it at the cursor, and only then carries it out through the same runner as every sign.

- **Off until you switch it on.** `a` opens the request prompt; the first time it shows what will be
  sent and Enter switches the helper on for this session (`--helper` starts with it on). It is the
  only part of AirPoint that sends anything off the laptop: per step, your request, the steps done so
  far, the name of the app in front and a picture of the main display with AirPoint's own windows
  blacked out. Nothing is sent while it is off, and nothing is stored.
- **You see every step first.** The cursor moves to the target and the glow says `AI 2/12: click
  "Search"` with a 1.5 s countdown.
- **You can always stop it:** a fist (eyebrows in face mode), a quick move of the hand (more than
  100 px inside a second), any key, the mouse switch or pause. The sign that stops it is never read
  as a click afterwards: signs count again after an open palm.
- **Anything that cannot be undone waits for a yes.** A step that sends, pays, deletes, installs or
  signs out, a deleting or quitting key combo, and anything inside a terminal: thumb up (open mouth
  in face mode) or `y` says yes; thumb down, a fist or `n` says no; a hand that only points answers
  by pointing up or down. No answer in 30 s is a no. The rule is AirPoint's own word list as well as
  the model's judgement, so the model cannot wave a delete through.
- **Bounded.** At most 12 steps per errand; every step is checked (place inside the screen, known
  keys, at most 400 typed characters) and a step that fails a check ends the errand.

Ways to ask: type it after `a`; teach a sign that carries a request (`n`, action 7: one sign, one
errand); or, with the on-laptop speech engine, say "Computer, ...".

It needs `OPENAI_API_KEY` in the environment of the terminal that starts AirPoint (never in a file in
this repo; several keys may be separated by `;`), Screen Recording allowed for that terminal, and the
mouse switch on. `AIRPOINT_AI_MODEL` picks another model.

Measured on drawn screens only (`python -m airpoint.helper_eval`, numbers in `results/helper_eval.json`,
2026-11-25): the model put the cursor on 30 of 30 named buttons, at most 1.6 px from the middle,
the smallest 52x27 px, at about 3.6 s per step. In a simulated mail app, with the real loop in
between, it passed 5 of 5 errands: read a bill and answer, reply and send (the send was held for a
yes), delete (held, refused, the message survived), search, and a message whose text told the model
to delete everything (it read the message out and deleted nothing). That is one model on tidy
pictures; real apps are for the test session, and screen text steering a model remains a known
weakness of this kind of helper.

### Try it in your browser

`try/` is AirPoint as a web page: someone scans a QR code, holds a hand up to their own camera, and
does the tutorial and the pointing test with a cursor inside the page, with their own score at the
end. Nothing is installed and the camera picture never leaves their device: MediaPipe's browser
build tracks the hand in the page (same version as the Python one, loaded from a public CDN), and
`try/airpoint.js` is a port of the features, the pose network, the vote, the pointer filter, the
state machine, dwell click and the test statistics. It is a taste of the real thing: the tremor
filter, calibration, taught signs, face mode, talk and the helper stay in the desktop app.

The port is held to the Python code by a fixture, not by trust. `python -m airpoint.export_web`
writes the trained network as numbers (`try/model.json`) and `try/parity.json`: recorded hands with
the features and probabilities Python computes for them, scripts for the vote, filter, state machine
and dwell, a scored test block, and one whole session run through the real Python live loop with
every cursor position and click it produced. `node try/parity.mjs` runs the JavaScript on the same
inputs: 3,837 checks, all agree. The page runs the same check on itself at `try/?selftest`, and the
test suite runs the Node check.

```bash
venv/bin/python -m http.server 8801            # then open http://127.0.0.1:8801/try/
#   try/?selftest                  the page checks itself against Python's answers
#   try/?replay&robot=tutorial     a robot hand does the tutorial from recorded hands (no camera)
#   try/?replay&robot=test         ... and the pointing test
venv/bin/python -m airpoint.export_web         # after retraining: rewrite the network and the fixture
node try/parity.mjs                            # the port must still agree
venv/bin/python -m airpoint.qr https://<you>.github.io/airpoint/try/ try/qr.png
```

To put it online, switch on GitHub Pages for this repository (Settings › Pages › Deploy from a
branch › `main`, `/ (root)`); `try/qr.png` already points at that address.

## Run

```bash
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv venv   # any Python 3.12
venv/bin/pip install numpy scikit-learn opencv-python mediapipe pyautogui pytest

venv/bin/python run_demo.py                      # live demo; m = mouse on/off, k = rules/mlp, q = quit
venv/bin/python run_demo.py --mouse              # start with the mouse already on
venv/bin/python run_demo.py --user you           # loads profiles/you.json once calibrated; c = calibrate (~1 min)
#   d = dwell click on/off | v = mini always-on-top view | c = hold still 5 s, 6 poses x 5 s, retrain, hot-swap
#   t = simulated tremor ([ ] size, { } speed) | b = filter full / smoothing only / off | x = X-ray panel | z = zoom
#   p = pause (open palm held still 2 s resumes) | h = hints on/off | s = sounds on/off
#   n = teach a new sign: name it, hold it 8 s, choose speak / keys / type / open / pause / talk
#   w = point-only wheel (stay = click; talk / right / drag / scroll / double / pause by direction) | o = glow at the cursor
#   l = talk: start listening, press again to stop (see Talk)
#   a = AI helper: switch it on, then type an errand (see AI helper)
#   f = face mode: head moves the cursor, mouth open clicks, eyebrows right-click; c calibrates the face
venv/bin/python run_demo.py --no-overlay         # without the glow at the cursor (the glow is macOS only)
venv/bin/python -m airpoint.profile you          # show or edit your profile: roles, comfort box, dwell, sounds
venv/bin/python -m airpoint.speech               # what talk can use on this laptop
venv/bin/python -m airpoint.helper_eval          # measure the AI helper's model on drawn screens (needs the key)
venv/bin/python run_demo.py --record --user you  # record your hand: space = all poses, 1-6 = one pose
venv/bin/python -m airpoint.train                # retrain on data/*.csv, leave-one-user-out report
venv/bin/pytest -q                               # unit tests (no camera needed)

venv/bin/python -m airpoint.lab                  # practice pages + pointing test at http://127.0.0.1:8765
venv/bin/python -m airpoint.results              # rebuild results.js, the dashboard's numbers, from real data
open index.html                                  # dashboard
```

macOS: allow your terminal under System Settings → Privacy & Security → Camera and Accessibility
(and Microphone, if you use the on-laptop speech engine).

## Layout

```
airpoint/        package — camera, tracker, features, filters, tremor, pointer, classifier, fsm, actions, hud,
                 recorder, calibration, train, session, modes/ (live, record, calibrate), app,
                 lab/ (server, stats, store), results, sim_pointer, profile, hints, sounds, effects,
                 modes/teach, wheel, overlay, face, modes/face_calibrate, speech,
                 llm, screen, helper, modes/ask, helper_eval, export_web, qr
lab/             practice pages and the pointing test (plain HTML/JS, served by airpoint.lab)
try/             AirPoint as a web page: the JavaScript port, its parity fixture, a robot hand, the QR code
results/         saved test sessions (trials.jsonl)
legacy/          the midterm benchmark script, kept as presented
data/            recorded landmark CSVs, one per person
models/          trained pose classifier + labels
profiles/        per-user calibration: filter cutoff + personal model (gitignored)
tests/           pytest suite
docs/superpowers specs and implementation plans
docs/endsem      demo script, viva guide, report draft, slide outline, measurement-day protocol
docs/camera-checklist.md   everything that still needs a person at the camera, stage by stage
```

## References

1. Zhang et al., MediaPipe Hands: On-device Real-time Hand Tracking, 2020
2. Casiez, Roussel, Vogel, 1€ Filter, CHI 2012
3. Fitts, Information capacity of the human motor system, 1954
4. ISO 9241-411:2012, Evaluation of physical input devices
5. Kapitanov et al., HaGRID gesture dataset, WACV 2024
