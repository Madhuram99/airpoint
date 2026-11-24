# AirPoint Stage 7 — Talk: task list

Spec: `docs/superpowers/specs/2026-11-25-airpoint-stage7-talk-design.md`.
Code lives in the commits, suite green each time. Baseline `209 passed`.

- [ ] **1. `airpoint/speech.py`** — `interpret`, `Talker`, `MicRecorder`, `WhisperRecognizer`;
  tests with a fake recorder and recognizer.
- [ ] **2. In the live loop** — `talk` action for taught signs, key `l`, `Profile.dictation_keys`,
  LISTENING / THINKING in the banner, mini label and glow.
- [ ] **3. Docs, checklist, push.** Local engine install left as a documented, optional step.
