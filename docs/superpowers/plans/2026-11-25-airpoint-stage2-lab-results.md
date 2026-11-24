# AirPoint Stage 2 — Practice pages and measured numbers: task list

Spec: `docs/superpowers/specs/2026-11-25-airpoint-stage2-lab-results-design.md`.

From this stage on the user has approved all stages in advance and tests them in one
session at the end. The plan is therefore a task list with interfaces, tests and
checks; the code is in the commits, one per task, each with the suite green.
Baseline: `venv/bin/pytest -q` → `106 passed`.

- [ ] **1. `airpoint/sim_pointer.py`** — move `tests/pointer_sim.py` into the package (the
  results generator needs it); update the three test imports. Suite unchanged.
- [ ] **2. `airpoint/lab/stats.py`** — `sequence_stats(seq)`, `block_summary(block)`,
  `aggregate(blocks)`. Tests: hand-computed sequence (dx, We, Ae, IDe, TP), error rate,
  flat-spread fallback flagged, path ratio and re-entries, aggregation over participants.
- [ ] **3. `airpoint/lab/store.py`, `server.py`, `__main__.py`** — `results/trials.jsonl`;
  routes `/`, `/lab/*`, `/index.html`, `/results.js`, `/api/state`, `/api/blocks`,
  `/api/summary`. Tests on an ephemeral port: static, traversal refused, save + score,
  oversized refused, summary.
- [ ] **4. Live state** — `Session.state()` and a once-a-second write of
  `results/live.json`; `server` reads it, stale after 3 s; `condition_label(state, device)`.
  Tests: throttle, contents, fresh/stale, labels.
- [ ] **5. Pages** — `lab/index.html`, `test.html`, `tutorial.html`, `lab.css`, `lab.js`,
  `test.js`, `tutorial.js`. Checks: `node --check`; a browser run of both pages against
  the real server, saving one block and one tutorial.
- [ ] **6. `airpoint/results.py`** — `build(root) -> dict`, `write(root)`; dataset,
  classifier on real frames (with the rule baseline), latency, tremor simulation table,
  pointing summary. Tests on a temporary project, including the empty state.
- [ ] **7. Dashboard** — `index.html` reads only `results.js`; invented fallbacks removed;
  six poses; pointing table by condition; tremor table; links to the lab. `results.js`
  tracked. Midterm benchmark moved to `legacy/`, writing `legacy/midterm_results.js`.
- [ ] **8. Docs and push** — README, `run.txt`, roadmap status, `docs/camera-checklist.md`
  (one list for the single test session, starting with Stage 1's items).
