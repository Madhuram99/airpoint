# AirPoint — Mentor Demonstration Guide

## 0. Setup (already done on this Mac)
A Python 3.12 virtual environment lives in `venv/` with numpy, scikit-learn, opencv-python, mediapipe and pyautogui.

```bash
cd "/Users/madhuramrathi/Downloads/Major Project"
source venv/bin/activate
```

## 1. Show the Normalized 63-D Dataset (`hagrid_airpoint_landmarks_sample.csv`)
- Each row is a **63-dimensional skeletal hand vector** (`x0, y0, z0` … `x20, y20, z20`) across 5 gestures, from HaGRID (Kapitanov et al., WACV 2024) and local webcam recordings.
- `x0, y0, z0` (wrist) is always `0, 0, 0` and `x9, y9, z9` (middle knuckle) is `0, -1, 0` — **wrist-centred, palm-scaled normalization**, so the classifier works at any camera distance.

## 2. Run the Experiment Suite
```bash
python airpoint_offline_training_filter_fitts_law_benchmark_suite.py
```
Every number printed is computed live and saved to `results.js`, which the dashboard reads.

1. **Experiment 1 — Rules vs. MLP:** 4-fold *leave-one-recording-out* cross-validation (test hands are never seen in training), with ±34° hand tilt and landmark noise. Geometric rules ≈ **85.8%**, 63-input MLP ≈ **99.9%**, inference ≈ **0.06 ms/frame** on CPU. Rules fail most on Pinch (≈64%) because noise pushes the thumb–index gap across the fixed threshold.
2. **Experiment 2 — 1-Euro Filter:** at rest, jitter drops from ±9.1 px to ±2.2 px (**−76%**), as smooth as a heavy fixed low-pass filter — but during a fast reach the 1-Euro filter lags only 21 px vs. 185 px (**89% less lag**). That trade-off is exactly the point of Casiez et al. (CHI 2012).
3. **Experiment 3 — Fitts's Law (ISO 9241-411):** ID = log₂(D/W + 1), throughput = ID / MT. With the configured pilot movement times, throughput rises from 1.40 to 2.58 bits/s (**+83.5%**). The movement times are inputs in `run_fitts_law_benchmark()` — replace them with measured trial means once the user study is done.
4. **Hysteresis stress test:** a relaxed hand hovering near the pinch threshold for 5 simulated minutes — a single 0.30 threshold fires **76 false clicks/min**, dual 0.28/0.36 hysteresis fires **4.4/min** (−94%).

## 3. Open the Dashboard
```bash
open index.html
```
Walk through: pipeline → 5 gestures → 1-Euro filter animation → **click the FSM buttons** (Hand Detected → Pinch → Hold > 500 ms → Release Drag) → results tables (filled from `results.js`).

## 4. Live Webcam Prototype
```bash
python airpoint_live_webcam_prototype_1_euro_filter_fsm_pinch.py
```
- Raise your hand: **red dot** = raw landmark, **green circle** = 1-Euro filtered cursor (anchored to the middle knuckle so pinching doesn't move it).
- Quick pinch (< 0.5 s): HUD goes `MOVING → PRESSING → LEFT CLICK!` (hysteresis 0.28 / 0.36).
- Hold the pinch (> 0.5 s): FSM enters `DRAGGING`; release → `DROP`.
- Thumb out, other fingers curled: `SCROLLING`.
- **Press `m` to control the real mouse** — the rectangle on screen maps to your whole display. Pinch clicks, hold-pinch drags (try dragging a Finder window), thumb pose + moving your hand up/down scrolls. Press `m` again to stop, `q` to quit.
- First time with `m`: macOS asks to allow your terminal under **System Settings → Privacy & Security → Accessibility**. Allow it, then restart the script.
- Camera: uses the MacBook camera automatically (works with or without an iPhone Continuity Camera nearby).
