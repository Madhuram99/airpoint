# AirPoint Stage 6 — Face mode: task list

Spec: `docs/superpowers/specs/2026-11-25-airpoint-stage6-face-mode-design.md`.
Code lives in the commits, one per task, suite green each time. Baseline `190 passed`.

- [ ] **1. `airpoint/face.py`** — `head_angles`, `FacePointer`, `FaceSigns`, `FaceCalib`,
  `FaceTracker`, defaults; tests on synthetic matrices, scores and recordings.
- [ ] **2. Face in the live loop** — `Session.source`, `set_source`, `Obs.face`, reading
  split in `LiveMode`, X-ray labels, hints wording, key `f`, hand-only keys refuse.
- [ ] **3. Face calibration mode** — `modes/face_calibrate.py`, profile `face`, `c` in face
  mode; app loop runs the face tracker when the source is face.
- [ ] **4. Docs, checklist, model file tracked, push.**
