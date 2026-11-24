# AirPoint Stage 5 — Glow at the cursor, point-only wheel: task list

Spec: `docs/superpowers/specs/2026-11-25-airpoint-stage5-overlay-wheel-design.md`.
Code lives in the commits, one per task, suite green each time. Baseline `169 passed`.

- [x] **1. Wheel** — `airpoint/wheel.py` (`Wheel.update`, `frozen`, `view`); `double_click`
  event; tests for every path in the spec.
- [x] **2. Wheel in the live loop** — key `w`, `Profile.wheel`, cursor frozen while open,
  drawn in the camera view, calibration fallback, lab label.
- [x] **3. Overlay process** — `airpoint/overlay.py`: painter, window, pipe reader, client;
  off-screen render test; a launch check against the window list.
- [x] **4. Overlay in the live loop** — state per frame, key `o`, `--no-overlay`.
- [x] **5. Docs, checklist, push.**
