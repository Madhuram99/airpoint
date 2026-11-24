# AirPoint Stage 4 — Teach a sign, speak for me: task list

Spec: `docs/superpowers/specs/2026-11-25-airpoint-stage4-teach-speak-design.md`.
Code lives in the commits, one per task, suite green each time. Baseline `152 passed`.

- [ ] **1. Training with personal files** — guard passes unknown labels; `load_rows` skips
  `*_signs.csv`; `train_personal(data_dir, models_dir, user)` gathers calibration and sign
  rows, reports per-label recall, confusion and false reads; callers and tests updated.
- [ ] **2. Bindings and effects** — `Profile.signs`, `Profile.voice`; `fsm_parts(roles, signs)`;
  per-pose hold time in the state machine; `effects.py` (say, open); `type` through the
  mouse object; `pause` event; speaking allowed with the mouse off, nothing while paused.
- [ ] **3. Teach mode** — `modes/teach.py` with text entry, capture, action menu, training,
  result line, forgetting; key `n`; the quit key is text while typing.
- [ ] **4. X-ray with many classes; docs; checklist; push.**
