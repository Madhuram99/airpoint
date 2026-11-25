# AirPoint Stage 9 — Try it in your browser: task list

Spec: `docs/superpowers/specs/2026-11-25-airpoint-stage9-browser-design.md`.
Code lives in the commits, suite green each time. Baseline `319 passed`.

- [ ] **1. Export and port** — `airpoint/export_web.py`, `try/model.json`, `try/parity.json`, `try/airpoint.js`,
  `try/parity.mjs`; pytest runs the Node check.
- [ ] **2. The page** — `try/index.html`, `app.js`, `pointer.js`, styles; static hook in `lab/lab.js`.
- [ ] **3. In a real browser** — engine loads, replay drives the tutorial and the test, self-test passes.
- [ ] **4. QR code, docs, checklist, push.** Pages itself stays off until the user says so.
