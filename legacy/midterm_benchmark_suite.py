"""
KEPT AS THE MIDTERM DELIVERABLE. The dashboard no longer reads this script's output: since
Stage 2 its numbers come from `python -m airpoint.results`, which uses the real recordings and
measured test sessions. Two things to know before quoting anything from here: the accuracy below
is measured on augmented copies of 20 seed rows, and the Fitts movement times are configured
pilot values, not measurements. This script now writes `legacy/midterm_results.js`.

AirPoint Midterm Evaluation — Offline Training & Research Benchmark Pipeline
Department of Computer Science & Engineering, PEC Chandigarh
Authors: Madhuram Rathi, Ishan Gupta, Dakshmeet Singh, Bhuvi Kansal
Mentor: Mrs. Rupali

What this script does when you run it in Antigravity:
1. Loads `hagrid_airpoint_landmarks_sample.csv` (21 3D landmarks = 63 features)
   and augments it with realistic hand rotation + motor tremor across all
   5 Slide-4 accessibility gestures.
2. Evaluates the Slide-9 Geometric Rule-Based Baseline vs. the Slide-8
   63-Input MLP Classifier (63 -> 128 -> 64 -> 5) with 4-fold
   leave-one-recording-out cross-validation (test hands never seen in training).
3. Runs the Casiez et al. (CHI 2012) One-Euro Filter experiment on simulated
   6 Hz hand tremor + ballistic reach, proving jitter reduction without lag.
4. Computes ISO 9241-411 Fitts's Law Index of Difficulty (ID) and Throughput
   (ID / MT) to validate Slide 10 evaluation metrics, and simulates the
   false-click rate of single-threshold vs. hysteresis pinch detection.
5. Writes every number to `results.js`, which the index.html dashboard loads.
"""

import csv
import json
import math
import time
import os
import numpy as np

# Optional scikit-learn import with clean fallback so the script NEVER crashes
try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ==============================================================================
# 1. SCALE-FREE LANDMARK NORMALIZATION (Zhang et al., 2020 + Slide 5 & 9)
# ==============================================================================
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_TIP = 5, 8
MIDDLE_MCP, MIDDLE_TIP = 9, 12
RING_MCP, RING_TIP = 13, 16
PINKY_MCP, PINKY_TIP = 17, 20

GESTURE_NAMES = [
    "0: Move (Open Palm)",
    "1: Click (Pinch)",
    "2: Alt Click (Fist)",
    "3: Right Click (V-Sign)",
    "4: Scroll (Thumb)"
]


def normalize_63_landmarks(vec_63):
    """
    Translates wrist (point 0) to (0,0,0) and divides all 21 points by the
    Euclidean palm size ||P_9 - P_0|| so gestures are scale- and position-free.
    """
    pts = np.array(vec_63, dtype=np.float64).reshape(21, 3)
    wrist = pts[WRIST].copy()
    pts = pts - wrist
    palm_size = np.linalg.norm(pts[MIDDLE_MCP]) + 1e-6
    pts = pts / palm_size
    return pts.flatten()


TILT_MAX_RAD = 0.60    # +/- 34 deg in-plane hand rotation
NOISE_SIGMA = 0.12     # landmark jitter in palm units (MediaPipe noise + tremor)


def load_seed_dataset(csv_path=None):
    """Loads the seed HaGRID + Local PEC webcam CSV as normalized 63-D vectors."""
    csv_path = csv_path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        "hagrid_airpoint_landmarks_sample.csv")
    seed_X, seed_y = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            coords = [float(row[f"{axis}{i}"]) for i in range(21) for axis in ("x", "y", "z")]
            seed_X.append(normalize_63_landmarks(coords))
            seed_y.append(int(row["gesture_id"]))
    print(f"[DATA] Loaded {len(seed_X)} seed landmark profiles from '{os.path.basename(csv_path)}'.")
    return np.array(seed_X), np.array(seed_y)


def augment(seed_X, seed_y, seed_ids, total_samples, rng):
    """
    Expands the chosen seed profiles with realistic in-plane hand tilt
    (+/- 34 deg) and physiological tremor / sensor noise.
    """
    X_out, y_out = [], []
    for i in range(total_samples):
        idx = seed_ids[i % len(seed_ids)]
        base_pts = seed_X[idx].reshape(21, 3)

        theta = rng.uniform(-TILT_MAX_RAD, TILT_MAX_RAD)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        rot_z = np.array([[cos_t, -sin_t, 0.0],
                          [sin_t,  cos_t, 0.0],
                          [0.0,    0.0,   1.0]])
        noisy_pts = base_pts @ rot_z.T + rng.normal(0.0, NOISE_SIGMA, size=(21, 3))
        X_out.append(normalize_63_landmarks(noisy_pts.flatten()))
        y_out.append(seed_y[idx])
    return np.array(X_out), np.array(y_out)


def leave_one_seed_out_folds(seed_y, n_folds=4):
    """
    Each fold holds out one seed recording per gesture, so the test set only
    contains hands the classifier never saw during training (no leakage).
    """
    per_class = {c: np.where(seed_y == c)[0] for c in np.unique(seed_y)}
    for k in range(n_folds):
        test_ids = [ids[k % len(ids)] for ids in per_class.values()]
        train_ids = [i for i in range(len(seed_y)) if i not in test_ids]
        yield k, train_ids, test_ids


# ==============================================================================
# 2. GEOMETRIC RULE BASELINE (Slide 8 & Slide 9 Implementation)
# ==============================================================================
def geometric_rule_classifier(vec_63):
    """
    Implements the deterministic scale-free geometric rule baseline from Slide 8 & 9.
    Works well for upright hands, but struggles when the hand tilts > 18 degrees.
    """
    pts = vec_63.reshape(21, 3)
    palm_len = np.linalg.norm(pts[MIDDLE_MCP] - pts[WRIST]) + 1e-6

    # Slide 9 formula: pinch_ratio = dist(THUMB_TIP, INDEX_TIP) / palm
    pinch_gap = np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP]) / palm_len

    # Extension ratios of fingers from wrist
    idx_ext = np.linalg.norm(pts[INDEX_TIP] - pts[WRIST]) / palm_len
    mid_ext = np.linalg.norm(pts[MIDDLE_TIP] - pts[WRIST]) / palm_len
    ring_ext = np.linalg.norm(pts[RING_TIP] - pts[WRIST]) / palm_len
    pinky_ext = np.linalg.norm(pts[PINKY_TIP] - pts[WRIST]) / palm_len
    thumb_ext = np.linalg.norm(pts[THUMB_TIP] - pts[WRIST]) / palm_len

    # Rule 1: Pinch (Slide 9 threshold < 0.30, with middle finger extended)
    if pinch_gap < 0.30 and mid_ext > 1.35:
        return 1  # Click (Pinch)

    # Rule 2: V-Sign (Index and Middle extended, Ring and Pinky curled)
    if idx_ext > 1.55 and mid_ext > 1.55 and ring_ext < 1.05 and pinky_ext < 1.00:
        return 3  # Right Click (V-Sign)

    # Rule 3: Scroll Thumb (Thumb strongly extended, other 4 fingers curled)
    if thumb_ext > 1.35 and idx_ext < 0.95 and mid_ext < 0.95 and ring_ext < 0.95:
        return 4  # Scroll (Thumb)

    # Rule 4: Closed Fist (All 4 fingers curled tightly toward palm)
    if idx_ext < 0.92 and mid_ext < 0.92 and ring_ext < 0.92 and pinky_ext < 0.92:
        return 2  # Alt Click (Fist)

    # Default: Open Palm (Move)
    return 0


# ==============================================================================
# 3. ONE-EURO FILTER (Casiez, Roussel, Vogel — CHI 2012, Slide 7)
# ==============================================================================
class OneEuroFilter:
    """
    Exact implementation of the 1-Euro Filter equations on Slide 7:
      fc = fc_min + beta * |speed|
      alpha = 1 / (1 + 1 / (2 * pi * fc * dt))
      x_out = alpha * x + (1 - alpha) * x_prev
    """
    def __init__(self, fc_min=0.80, beta=0.010, d_cutoff=1.0):
        self.fc_min = fc_min
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(fc, dt):
        return 1.0 / (1.0 + 1.0 / (2.0 * math.pi * fc * dt))

    def filter(self, x, t):
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x

        dt = max(t - self.t_prev, 1e-5)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        edx = a_d * dx + (1.0 - a_d) * self.dx_prev

        fc = self.fc_min + self.beta * abs(edx)
        alpha = self._alpha(fc, dt)
        x_out = alpha * x + (1.0 - alpha) * self.x_prev

        self.x_prev, self.dx_prev, self.t_prev = x_out, edx, t
        return x_out


def run_signal_processing_benchmark():
    """
    Simulates a user holding their hand steady with a 6 Hz tremor (Frames 0-24),
    making a fast ballistic movement across the screen (Frames 25-34),
    and holding steady over a target button (Frames 35-89). Resting jitter is
    measured once each filter has settled (Frames 0-24 and 55-89).
    """
    print("\n" + "=" * 78)
    print("EXPERIMENT 2: ONE-EURO FILTER TREMOR SUPPRESSION BENCHMARK (Casiez et al. 2012)")
    print("=" * 78)

    fps = 30.0
    dt = 1.0 / fps
    rng = np.random.default_rng(101)

    one_euro = OneEuroFilter(fc_min=0.80, beta=0.010)
    fixed_alpha = 0.15  # Traditional fixed low-pass filter
    fixed_prev = None

    resting_errors_raw = []
    resting_errors_fixed = []
    resting_errors_euro = []

    fast_lag_fixed = []
    fast_lag_euro = []

    for frame in range(90):
        t = frame * dt
        # Ideal ground-truth trajectory (pixels)
        if frame < 25:
            ideal_x = 200.0
        elif frame < 35:
            # Fast move from X=200px to X=800px in 10 frames (0.33 sec)
            ideal_x = 200.0 + (800.0 - 200.0) * ((frame - 25) / 10.0)
        else:
            ideal_x = 800.0

        # Add 6 Hz physiological hand tremor + camera sensor noise
        tremor = 14.0 * math.sin(2.0 * math.pi * 6.0 * t) + rng.normal(0, 3.2)
        raw_x = ideal_x + tremor

        # Fixed filter
        fixed_x = raw_x if fixed_prev is None else (fixed_alpha * raw_x + (1.0 - fixed_alpha) * fixed_prev)
        fixed_prev = fixed_x

        # 1-Euro filter
        euro_x = one_euro.filter(raw_x, t)

        if frame < 25 or frame >= 55:
            resting_errors_raw.append(abs(raw_x - ideal_x))
            resting_errors_fixed.append(abs(fixed_x - ideal_x))
            resting_errors_euro.append(abs(euro_x - ideal_x))
        elif 26 <= frame <= 35:
            fast_lag_fixed.append(abs(fixed_x - ideal_x))
            fast_lag_euro.append(abs(euro_x - ideal_x))

    raw_j, fixed_j, euro_j = (float(np.mean(e)) for e in
                              (resting_errors_raw, resting_errors_fixed, resting_errors_euro))
    lag_fixed, lag_euro = float(np.mean(fast_lag_fixed)), float(np.mean(fast_lag_euro))
    reduction = (1.0 - euro_j / raw_j) * 100.0

    print(f"Resting Cursor Jitter (Mean Abs Dispersion in Pixels):")
    print(f"  - Raw Unfiltered Landmarks : +/- {raw_j:.2f} px  (Unusable for small icons)")
    print(f"  - Fixed Low-Pass Filter    : +/- {fixed_j:.2f} px")
    print(f"  - AirPoint 1-Euro Filter   : +/- {euro_j:.2f} px  (-{reduction:.1f}% jitter reduction)")
    print(f"\nBallistic Movement Tracking Error (Dynamic Lag During Fast Reach):")
    print(f"  - Fixed Low-Pass Filter    : {lag_fixed:.2f} px lag (Feels sluggish/heavy)")
    print(f"  - AirPoint 1-Euro Filter   : {lag_euro:.2f} px lag ({(1 - lag_euro / lag_fixed) * 100:.0f}% less lag)")

    return {"raw_jitter_px": raw_j, "fixed_jitter_px": fixed_j, "euro_jitter_px": euro_j,
            "jitter_reduction_pct": reduction, "lag_fixed_px": lag_fixed, "lag_euro_px": lag_euro}


# ==============================================================================
# 4. FITTS'S LAW & ISO 9241-411 EVALUATION (Slide 10 & Fitts 1954)
# ==============================================================================
def run_fitts_law_benchmark():
    """
    Computes Fitts's Law Index of Difficulty: ID = log2(D/W + 1)
    and Throughput: TP = ID / MT across three standard target regimes.
    """
    print("\n" + "=" * 78)
    print("EXPERIMENT 3: ISO 9241-411 FITTS'S LAW THROUGHPUT & HYSTERESIS CLICK STUDY")
    print("=" * 78)

    # Movement times (seconds) per target regime. Replace with your own
    # measured trial means when you run the user study.
    trials = [
        {"D": 240, "W": 80, "mt_raw": 1.45, "mt_airpoint": 0.78},
        {"D": 480, "W": 60, "mt_raw": 2.15, "mt_airpoint": 1.18},
        {"D": 600, "W": 40, "mt_raw": 2.94, "mt_airpoint": 1.61},
    ]

    print("Movement times (MT) are configured pilot values in run_fitts_law_benchmark().")
    print(f"{'Distance (D)':<14} {'Width (W)':<12} {'ID (bits)':<12} {'Raw TP (bps)':<15} {'AirPoint TP (bps)':<18}")
    print("-" * 78)

    tp_raw_list, tp_air_list = [], []
    for tr in trials:
        id_bits = math.log2(tr["D"] / tr["W"] + 1.0)
        tp_raw = id_bits / tr["mt_raw"]
        tp_air = id_bits / tr["mt_airpoint"]
        tp_raw_list.append(tp_raw)
        tp_air_list.append(tp_air)
        tr.update(ID=id_bits, tp_raw=tp_raw, tp_airpoint=tp_air)
        print(f"{tr['D']:<14} {tr['W']:<12} {id_bits:<12.2f} {tp_raw:<15.2f} {tp_air:<18.2f}")

    mean_raw, mean_air = float(np.mean(tp_raw_list)), float(np.mean(tp_air_list))
    gain = (mean_air / mean_raw - 1.0) * 100.0
    print("-" * 78)
    print(f"Mean Fitts Throughput : Raw = {mean_raw:.2f} bits/s  -->  AirPoint = {mean_air:.2f} bits/s (+{gain:.1f}%)")

    single, hyst = run_false_click_simulation()
    print(f"False Clicks / Minute (stress test, hand hovering near threshold): Single (0.30) = {single:.1f}/min --> "
          f"Hysteresis (0.28/0.36) = {hyst:.1f}/min")

    return {"trials": trials, "tp_raw": mean_raw, "tp_airpoint": mean_air, "tp_gain_pct": gain,
            "false_clicks_single": single, "false_clicks_hysteresis": hyst}


def run_false_click_simulation(minutes=5.0, fps=30.0):
    """
    Simulates a user moving the cursor with a relaxed half-open hand (pinch
    ratio drifting around 0.30-0.40) and never intending to click, then counts
    how many pinch-enter events each detector fires per minute.
    """
    rng = np.random.default_rng(7)
    n = int(minutes * 60 * fps)
    t = np.arange(n) / fps
    # Slow posture drift + 6 Hz tremor + per-frame landmark noise
    ratio = (0.345 + 0.035 * np.sin(2 * math.pi * 0.15 * t)
             + 0.012 * np.sin(2 * math.pi * 6.0 * t) + rng.normal(0, 0.012, n))

    def count_clicks(enter, release):
        active, clicks = False, 0
        for r in ratio:
            if not active and r < enter:
                active, clicks = True, clicks + 1
            elif active and r > release:
                active = False
        return clicks / minutes

    return count_clicks(0.30, 0.30), count_clicks(0.28, 0.36)


# ==============================================================================
# MAIN EXECUTION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    print("=" * 78)
    print("AIRPOINT MIDTERM RESEARCH & TRAINING SUITE (PEC CHANDIGARH - OCT 2026)")
    print("=" * 78)

    results = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    # 1. Load & prepare 63-D skeletal landmark dataset
    seed_X, seed_y = load_seed_dataset()

    # 2. Evaluate Rule-Based Baseline vs. 63-Input MLP Classifier (Slide 8)
    print("\n" + "=" * 78)
    print("EXPERIMENT 1: GEOMETRIC RULES BASELINE vs. 63-INPUT LANDMARK MLP (Slide 8)")
    print("=" * 78)
    print(f"Protocol: 4-fold leave-one-recording-out | tilt +/-{math.degrees(TILT_MAX_RAD):.0f} deg | "
          f"noise sigma {NOISE_SIGMA} palm units")

    if SKLEARN_AVAILABLE:
        all_true, all_rule, all_mlp = [], [], []
        train_ms, inf_ms = [], []
        for k, train_ids, test_ids in leave_one_seed_out_folds(seed_y):
            rng = np.random.default_rng(42 + k)
            X_train, y_train = augment(seed_X, seed_y, train_ids, 1200, rng)
            X_test, y_test = augment(seed_X, seed_y, test_ids, 300, rng)

            t0 = time.perf_counter()
            mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                                solver="adam", max_iter=400, random_state=42)
            mlp.fit(X_train, y_train)
            train_ms.append((time.perf_counter() - t0) * 1000.0)

            # Single-frame CPU inference latency (one 63-D vector at a time, like the live loop)
            t_inf = time.perf_counter()
            for sample in X_test[:100]:
                mlp.predict(sample.reshape(1, -1))
            inf_ms.append((time.perf_counter() - t_inf) * 1000.0 / 100)

            fold_mlp = mlp.predict(X_test)
            fold_rule = np.array([geometric_rule_classifier(s) for s in X_test])
            print(f"  Fold {k + 1}: rules {np.mean(fold_rule == y_test) * 100:5.1f}% | "
                  f"MLP {np.mean(fold_mlp == y_test) * 100:5.1f}%")
            all_true.extend(y_test); all_rule.extend(fold_rule); all_mlp.extend(fold_mlp)

        y_true, y_rule, y_mlp = map(np.array, (all_true, all_rule, all_mlp))
        rule_acc = accuracy_score(y_true, y_rule) * 100.0
        mlp_acc = accuracy_score(y_true, y_mlp) * 100.0
        print(f"\n[BASELINE]  Scale-Free Geometric Rules Accuracy (held-out hands): {rule_acc:.2f}%")
        print(f"[MLP MODEL] 63-Input Landmark MLP Accuracy (held-out hands)      : {mlp_acc:.2f}%")
        print(f"[LATENCY]   MLP Training Time: {np.mean(train_ms):.1f} ms | "
              f"Inference Latency: {np.mean(inf_ms):.3f} ms/frame on CPU\n")
        print(classification_report(y_true, y_mlp, target_names=GESTURE_NAMES, digits=3))

        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_mlp, labels=range(5), zero_division=0)
        results["classifier"] = {
            "rule_acc": rule_acc, "mlp_acc": mlp_acc,
            "train_ms": float(np.mean(train_ms)), "inference_ms": float(np.mean(inf_ms)),
            "per_class": [{
                "name": GESTURE_NAMES[c],
                "rule_acc": float(np.mean(y_rule[y_true == c] == c) * 100.0),
                "mlp_acc": float(np.mean(y_mlp[y_true == c] == c) * 100.0),
                "precision": float(prec[c]), "recall": float(rec[c]), "f1": float(f1[c]),
            } for c in range(5)],
            "weighted": dict(zip(("precision", "recall", "f1"), map(float, precision_recall_fscore_support(
                y_true, y_mlp, average="weighted", zero_division=0)[:3]))),
        }
    else:
        print("[INFO] Install scikit-learn (`pip install scikit-learn`) to view live MLP training metrics.")

    # 3. Run Signal Processing Benchmark (Slide 7)
    results["filter"] = run_signal_processing_benchmark()

    # 4. Run Fitts's Law Evaluation Benchmark (Slide 10)
    results["fitts"] = run_fitts_law_benchmark()

    # 5. Publish results for the web dashboard (index.html loads results.js)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "midterm_results.js")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("window.AIRPOINT_MIDTERM_RESULTS = " + json.dumps(results, indent=2) + ";\n")
    print(f"\n[MIDTERM] Results written to legacy/{os.path.basename(out_path)}. The dashboard reads results.js, "
          "built by `python -m airpoint.results`.")
    print("[SUCCESS] All AirPoint Midterm experiments completed successfully.")