# Stage 11 plan: fixes from the first look

Spec: `docs/superpowers/specs/2026-11-25-airpoint-stage11-first-look-design.md`. Tests first, one commit per task.

1. **Helper programs without fork.** `spawn()` in `actions.py` (Popen with `close_fds=False`, which is
   `posix_spawn` on macOS); used by sounds, effects, the overlay and the screen capture. Test: `fork_exec`
   made to fail, the sound still starts.
2. **Speech in its own process.** `speech.py`: `hear(recognizer, audio)` (the checks and the recognition,
   one place), `serve()` (the process: blocking reads, answer before closing, watchdog), `Ear` (the client:
   non-blocking pipe, reader thread, time limits, restart). `Talker` takes an engine; `make_talker` gives
   it an `Ear`. Tests: a fake engine inside the real process (start, stop, heard, problem, a hung process
   is replaced, a dead app ends the process); the old in-process tests stay green.
3. **Words reach the app used last.** `Focus` in `actions.py`; `settle` event; live mode brings that app
   forward before typing. Tests with a fake workspace.
4. **Drawing kit.** `ui.py`: palette, font, `Canvas` (canonical units on an image `k` times larger).
   Tests: same layout at k=1 and k=2, text fits its box, colours land where expected.
5. **The window.** Camera overlays scale-aware; `panel_model` / `bar_model` (pure) in live mode; panel and
   message line drawn from the models and cached; X-ray takes the panel's place; mini and prompts restyled;
   every key writes a sentence to the message line. `app.py` sizes the window in points. Tests: models,
   sizes at each zoom, pixel checks.
6. **Glow.** `tone` and `word` in the overlay state; halo, colours, words. Tests on the state; renders
   looked at.
7. **Wheel.** Larger, icons, lit slice, caption, preview on `w`; the in-window wheel to match. Tests on
   the state; renders looked at.
8. **Docs.** README, run.txt, camera checklist section 11, demo script and viva guide wording, roadmap.
