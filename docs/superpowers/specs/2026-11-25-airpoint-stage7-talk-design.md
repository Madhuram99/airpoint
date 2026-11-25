# AirPoint Stage 7 — Talk

Date: 2026-11-25
Status: approved in advance (user, 2026-11-25); built the same day. The on-laptop speech engine is
written but NOT installed: its package and model are a download, which waits for the user's yes.
Roadmap: `2026-11-25-airpoint-v2-roadmap.md`, Stage 7

## 1. Goal

A user who cannot type gets text and a few spoken commands. One tap starts listening and the same
tap ends it: a taught sign, the top of the wheel, or the `l` key.

## 2. Two changes from the roadmap, with reasons

- **Tap to talk, not hold to talk.** Holding a sign for the length of a sentence is exactly what this
  project's users find hard. One tap starts, the same tap ends, and listening also ends by itself
  after twenty seconds.
- **Two engines.** The roadmap named local Whisper. It needs a package and a model of about 148 MB
  from the internet, which is the user's call and has not been given. So the default engine is the
  dictation built into macOS, which runs on the laptop and knows Indian English and Hindi; the
  local engine is written but stays off until installed.

## 3. Design

`speech.py`:
- `Talker` is the tap-to-talk state machine: idle, listening, thinking. `toggle(t)` starts or ends;
  `poll(t)`, called every frame, returns the events of a finished recognition and ends an overlong
  listen. Nothing in it blocks a frame.
- **System engine (default).** A tap presses the dictation shortcut from the profile
  (`control+option+d` unless changed) through the action runner; macOS shows its microphone and types
  into the app in front. The next tap presses it again. No audio touches AirPoint.
- **Local engine (optional).** `MicRecorder` captures the default microphone at 16 kHz only between
  the two taps. On the second tap the audio goes to `WhisperRecognizer` on a worker thread. A phrase
  that is exactly a command runs it; anything else is typed. A recording that is too short, silent
  (microphone access refused) or never louder than -40 dB is not sent to the model at all, and the
  model runs with voice-activity filtering, because Whisper invents words for silence.
- The recognizer never downloads: it loads with `local_files_only`. The model is fetched only by
  `python -m airpoint.speech --setup`, which states the size and asks first. `make_talker(profile)`
  falls back to system dictation, with a line saying what is missing.
- `python -m airpoint.speech` reports what is set up; `--file x.wav` and `--listen N` print what a
  recording is heard as and what it would do, without typing anything.

In the live loop:
- A taught sign can be bound to `talk` (item 6 on the teach menu). The wheel's top slot is talk; its
  centre was already click, so nothing was lost and the six directions keep their size.
- Talk needs the mouse switch on and does not start during a pause. Turning the mouse off, pausing,
  leaving the live view or quitting ends listening first, so macOS dictation is never left running.
- In face mode the mouth and eyebrows are ignored while listening and for one second after:
  speaking is not a click.
- The banner's third line, the mini label and the cursor glow show LISTENING (with the seconds
  left) and THINKING; after a local recognition the banner shows what was heard for four seconds.

Keystrokes (found while building this):
- pyautogui has no key called `control`; it was silently dropped, so every taught combo with control
  pressed the wrong keys. Our names are now translated, and a test checks every key name a sign may
  use against pyautogui's table.
- Typing a sentence takes seconds. All keystrokes now go through one worker thread: in order, never
  two at once, never holding up a frame.
- Text is reduced to what a US keyboard can type (curly quotes and dashes straightened, accents
  dropped); pyautogui would skip the rest silently.
- With AirPoint's own window in front, what it types arrives at its own key reader. Those keys are
  dropped (while typing, and until five frames pass without a key), and the banner says the words
  had nowhere to go. A dictated "q" cannot quit the app.

## 4. Tests

Commands against free text, punctuation and case; the talker with a stand-in recorder and
recognizer through start, stop, result, timeout, short, silent, quiet and failing audio, and a
microphone that will not open; the system engine's shortcut presses; engine choice and fallback;
WAV reading; the command line, including that `--setup` never downloads unasked. Through `Session`:
the key, a bound sign, the wheel, the mouse switch, pause, mode changes and quit, the twenty-second
end, the local engine typing and running commands, face mode, overlay tag and banner. The keystroke
worker: order, a failing keystroke, the typing flag; and the app loop dropping its own keys.

## 5. Checked without a person, and not checked

Checked: the microphone accepts 16 kHz mono (asked of the audio system, nothing recorded);
faster-whisper's interface (`download_model`, `WhisperModel`, `transcribe`) read from its source;
model sizes read from the model pages; every screen rendered off-screen and looked at.

Not checked: no keystroke has been sent to the real system in this stage, so that macOS dictation
starts on the shortcut, and that typing lands in another app, are for the test session. The local
engine has never run on this machine: no install, no model. The -40 dB silence gate is a standard
figure, not measured in this room. How macOS dictation behaves when it stops on its own (and so
falls out of step with AirPoint's idea of it) is unknown; the twenty-second limit is there to end it
first, and `l` puts the two back in step.
