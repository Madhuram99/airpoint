# AirPoint

Computer access through simple hand signs and an ordinary webcam, for people who find a mouse hard or painful to use (tremor, arthritis, post-stroke, cerebral palsy). Runs on a laptop CPU; no video leaves the machine.

Major Project, Department of CSE, Punjab Engineering College, Chandigarh. Mentor: Mrs. Rupali.
Team: Ishan Gupta, Dakshmeet Singh, Madhuram Rathi, Bhuvi Kansal.

## How it works

```
webcam → MediaPipe Hands (21 points) → wrist-centred, palm-scaled 63-d vector
       → pose classifier (MLP, rules fallback) → 5-frame vote → state machine → OS mouse
```

| Pose | Action |
|---|---|
| open palm | move cursor (One-Euro filtered, tremor removed) |
| pinch / fist, short | click · two quick = double click |
| pinch / fist, held | drag |
| V-sign, held 0.3 s | right click |
| thumb up / down | scroll, no hand motion needed |
| hand out of frame | rest, anything held is dropped |

## Run

```bash
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv venv   # any Python 3.12
venv/bin/pip install numpy scikit-learn opencv-python mediapipe pyautogui pytest

venv/bin/python run_demo.py                      # live demo; m = mouse on/off, k = rules/mlp, q = quit
venv/bin/python run_demo.py --record --user you  # record your hand: space = all poses, 1-6 = one pose
venv/bin/python -m airpoint.train                # retrain on data/*.csv, leave-one-user-out report
venv/bin/pytest -q                               # unit tests (no camera needed)

venv/bin/python airpoint_offline_training_filter_fitts_law_benchmark_suite.py   # benchmarks -> results.js
open index.html                                  # dashboard
```

macOS: allow your terminal under System Settings → Privacy & Security → Camera and Accessibility.

## Layout

```
airpoint/        package — camera, tracker, features, filters, classifier, fsm, actions, hud, recorder, train, app
data/            recorded landmark CSVs, one per person
models/          trained pose classifier + labels
tests/           pytest suite
docs/superpowers specs and implementation plans
```

## References

1. Zhang et al., MediaPipe Hands: On-device Real-time Hand Tracking, 2020
2. Casiez, Roussel, Vogel, 1€ Filter, CHI 2012
3. Fitts, Information capacity of the human motor system, 1954
4. ISO 9241-411:2012, Evaluation of physical input devices
5. Kapitanov et al., HaGRID gesture dataset, WACV 2024
