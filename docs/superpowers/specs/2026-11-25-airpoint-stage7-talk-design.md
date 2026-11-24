# AirPoint Stage 7 — Talk

Date: 2026-11-25
Status: approved in advance (user, 2026-11-25). Local speech engine NOT installed: waiting for the
user's yes on that download.
Roadmap: `2026-11-25-airpoint-v2-roadmap.md`, Stage 7

## 1. Goal

A user who cannot type gets text and a few spoken commands. One sign (or the `l`
key) starts listening; the same sign stops it.

## 2. Two changes from the roadmap, with reasons

- **Tap to talk, not hold to talk.** Holding a sign for the length of a sentence is
  exactly what this project's users find hard. One sign starts, the same sign ends,
  and listening also ends by itself after twenty seconds.
- **Two engines.** The roadmap named local Whisper. It needs a package and a model
  of 75–145 MB from the internet, which is the user's call and has not been given.
  So the default engine is the dictation built into macOS, which is on-device on
  Apple silicon and handles Indian English and Hindi; the local engine is written
  but stays off until installed.

## 3. Design — `speech.py`

- `Talker` is the tap-to-talk state machine: idle, listening, thinking. `toggle(t)`
  and `poll(t)` return events for the frame loop, so speech can never block a frame.
- **System engine (default).** `toggle` sends the dictation shortcut stored in the
  profile (`control+option+d` by default); macOS shows its microphone and types into
  the focused app. Stopping sends it again. No audio touches AirPoint. Setup, once:
  System Settings › Keyboard › Dictation › on, Shortcut › Customize › control+option+d.
- **Local engine (optional).** `MicRecorder` captures the default microphone at
  16 kHz only between start and stop. On stop the audio goes to `WhisperRecognizer`
  on a worker thread. The text is then interpreted: a phrase that is exactly a
  command runs it (click, double click, right click, scroll up/down, enter, tab,
  escape, delete, undo, copy, paste, select all, pause); anything else is typed.
  Nothing is recorded outside the listening window and nothing leaves the laptop.
- Typing and commands go through the same runner as every other action, so they
  need the mouse switch on and do nothing while paused.
- A taught sign can be bound to `talk` (item 6 on the teach menu).
- The banner, the mini label and the cursor glow show LISTENING and THINKING.

## 4. Tests

Interpretation of commands against free text, punctuation and case; the talker with
a fake recorder and recognizer through start, stop, result, timeout, empty audio
and a failing recognizer; the system engine's two shortcut presses; a session where
a sign bound to talk toggles listening and the result is typed only with the mouse on.

## 5. Not verified

The local engine has never run on this machine: it is written against the
faster-whisper interface and is exercised only through a stand-in. The system
engine sends the shortcut correctly, but whether dictation starts depends on the
one-time setting above, which only the user can make.
