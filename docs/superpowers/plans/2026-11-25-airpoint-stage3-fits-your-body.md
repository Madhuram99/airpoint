# AirPoint Stage 3 — Fits your body: task list

Spec: `docs/superpowers/specs/2026-11-25-airpoint-stage3-fits-your-body-design.md`.
Code lives in the commits, one per task, suite green each time. Baseline `128 passed`.

- [x] **1. Profile v2** — `airpoint/profile.py`: `Profile`, `load`, `save`, v1 upgrade,
  `check_roles`, CLI. `calibration.py` profile helpers removed; callers and tests moved over.
- [x] **2. Roles** — `AirPointFSM(grip, scroll, hold_events)` with a neutral-pose rule;
  `profile.fsm_parts(roles)`; `Session` builds its state machine from the profile.
- [x] **3. Either hand** — mirrored copies in `train.augment`; `mirrored_acc` in the report
  and in `results.js`; shipped model retrained; a mirrored fixture classifies the same.
- [x] **4. Comfort box** — `calibration.comfort_box`; reach phase in `CalibSession` and
  `CalibrateMode`; `PointerPipeline.set_gain`; region applied to the runner.
- [x] **5. Pause** — `ActionRunner.locked`; `p`; still-palm unlock with its ring; banner tag.
- [x] **6. Hints and sounds** — `hints.py`, `sounds.py`, HUD strip, keys `h` and `s`.
- [x] **7. Calibration saves it all** — skipped poses, dwell fallback, roles, box; launch
  applies the profile; live state carries lock and gain.
- [x] **8. Docs and push** — README, `run.txt`, roadmap status, `docs/camera-checklist.md`.
