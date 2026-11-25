# AirPoint Stage 7 — Talk: task list

Spec: `docs/superpowers/specs/2026-11-25-airpoint-stage7-talk-design.md`.
Code lives in the commits, suite green each time. Baseline `209 passed`, end `247 passed`.

- [x] **1. Keystrokes** — worker thread, `control` fix, typable text (`2768c14`).
- [x] **2. `airpoint/speech.py`** — `interpret`, `Talker`, `MicRecorder`, `WhisperRecognizer`, `make_talker`,
  command line; tests with a stand-in recorder and recognizer.
- [x] **3. In the live loop** — `talk` action for taught signs, wheel top slot, key `l`, profile fields,
  face mode ignores the mouth while listening, LISTENING / THINKING in banner, mini label and glow,
  own-keystroke guard in the app loop (`8a1f780`).
- [x] **4. Docs, checklist, push.**
- [ ] **5. On-laptop engine on this machine** — waits for the user's yes to
  `pip install faster-whisper` and the 148 MB model; then checklist items 7.9 to 7.12.
