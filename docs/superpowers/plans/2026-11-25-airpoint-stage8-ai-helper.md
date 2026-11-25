# AirPoint Stage 8 — AI helper: task list

Spec: `docs/superpowers/specs/2026-11-25-airpoint-stage8-ai-helper-design.md`.
Code lives in the commits, suite green each time. Baseline `247 passed`.

- [ ] **1. `llm.py`** — Responses API over HTTPS, key from the environment, scrubbed errors.
- [ ] **2. `helper.py`** — step format, checks, risk rule, `Helper` state machine.
- [ ] **3. `screen.py`** — picture of the screen without AirPoint's own windows; permission; app in front.
- [ ] **4. Yes/no wheel and pointing the cursor** — `Wheel` with custom items; runner `point`.
- [ ] **5. In the live loop** — `a`, the request prompt, signs and speech that ask, stop and answer
  gestures, banner and glow, `--helper`.
- [ ] **6. `helper_eval.py`** — drawn screens, simulated mail app; run with the real model; save numbers.
- [ ] **7. Docs, checklist, push.**
