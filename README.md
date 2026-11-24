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

### Teach a sign

`n` (with `--user`) adds a sign of your own in about fifteen seconds: type a name, hold the sign for
eight seconds, then choose what it does: speak a phrase aloud, press a key combo, type text, open an
app or a link, or pause AirPoint. The personal model is retrained with the new class and swapped in
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
#   n = teach a new sign: name it, hold it 8 s, choose speak / keys / type / open / pause
venv/bin/python -m airpoint.profile you          # show or edit your profile: roles, comfort box, dwell, sounds
venv/bin/python run_demo.py --record --user you  # record your hand: space = all poses, 1-6 = one pose
venv/bin/python -m airpoint.train                # retrain on data/*.csv, leave-one-user-out report
venv/bin/pytest -q                               # unit tests (no camera needed)

venv/bin/python -m airpoint.lab                  # practice pages + pointing test at http://127.0.0.1:8765
venv/bin/python -m airpoint.results              # rebuild results.js, the dashboard's numbers, from real data
open index.html                                  # dashboard
```

macOS: allow your terminal under System Settings → Privacy & Security → Camera and Accessibility.

## Layout

```
airpoint/        package — camera, tracker, features, filters, tremor, pointer, classifier, fsm, actions, hud,
                 recorder, calibration, train, session, modes/ (live, record, calibrate), app,
                 lab/ (server, stats, store), results, sim_pointer, profile, hints, sounds, effects,
                 modes/teach
lab/             practice pages and the pointing test (plain HTML/JS, served by airpoint.lab)
results/         saved test sessions (trials.jsonl)
legacy/          the midterm benchmark script, kept as presented
data/            recorded landmark CSVs, one per person
models/          trained pose classifier + labels
profiles/        per-user calibration: filter cutoff + personal model (gitignored)
tests/           pytest suite
docs/superpowers specs and implementation plans
```

## References

1. Zhang et al., MediaPipe Hands: On-device Real-time Hand Tracking, 2020
2. Casiez, Roussel, Vogel, 1€ Filter, CHI 2012
3. Fitts, Information capacity of the human motor system, 1954
4. ISO 9241-411:2012, Evaluation of physical input devices
5. Kapitanov et al., HaGRID gesture dataset, WACV 2024
