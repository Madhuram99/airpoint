window.AIRPOINT_RESULTS = {
  "generated_at": "2026-11-25 02:44",
  "commit": "aef3263",
  "dataset": {
    "rows": 1508,
    "users": [
      "madhuram"
    ],
    "per_pose": {
      "fist": 168,
      "palm": 297,
      "pinch": 279,
      "thumb_down": 214,
      "thumb_up": 275,
      "v_sign": 275
    }
  },
  "classifier": {
    "method": "25% row hold-out of one user (not a generalization number)",
    "generalizes": false,
    "accuracy": 1.0,
    "rule_acc": 0.5086,
    "users": [
      "madhuram"
    ],
    "trained_rows": 1508,
    "trained_at": "2026-11-24 23:38:10",
    "stale": false,
    "per_pose": [
      {
        "pose": "fist",
        "rows": 168,
        "rule_acc": 0.8333,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0
      },
      {
        "pose": "palm",
        "rows": 297,
        "rule_acc": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0
      },
      {
        "pose": "pinch",
        "rows": 279,
        "rule_acc": 0.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0
      },
      {
        "pose": "thumb_down",
        "rows": 214,
        "rule_acc": 0.0748,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0
      },
      {
        "pose": "thumb_up",
        "rows": 275,
        "rule_acc": 0.2582,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0
      },
      {
        "pose": "v_sign",
        "rows": 275,
        "rule_acc": 0.8836,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0
      }
    ]
  },
  "latency": {
    "classifier_ms": 0.0407,
    "features_ms": 0.0031,
    "frames": 300
  },
  "tremor": {
    "note": "Simulation: synthetic reaches and holds with a tremor of wandering frequency and size, 30 fps, 640-wide frame. No patient data.",
    "smooth": "1-Euro smoothing only",
    "full": "tremor canceller + 1-Euro smoothing",
    "fc_min": 0.8,
    "beta": 0.03,
    "sessions": 3,
    "seconds": 60,
    "levels": [
      {
        "name": "moderate",
        "hz": 6.0,
        "amp_px": 14.0,
        "smooth": {
          "shake_px": 4.208,
          "dwell_ok": 0.9556
        },
        "full": {
          "shake_px": 0.9622,
          "dwell_ok": 1.0
        }
      },
      {
        "name": "severe",
        "hz": 5.0,
        "amp_px": 25.0,
        "smooth": {
          "shake_px": 12.0416,
          "dwell_ok": 0.0559
        },
        "full": {
          "shake_px": 1.9032,
          "dwell_ok": 1.0
        }
      },
      {
        "name": "very severe",
        "hz": 6.0,
        "amp_px": 40.0,
        "smooth": {
          "shake_px": 20.5994,
          "dwell_ok": 0.0
        },
        "full": {
          "shake_px": 3.2815,
          "dwell_ok": 1.0
        }
      }
    ]
  },
  "pointing": {
    "conditions": [],
    "participants": 0,
    "blocks": 0
  }
};
