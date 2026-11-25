# AirPoint Stage 7 — Talk

Date: 2026-11-25
Status: approved in advance (user, 2026-11-25); built the same day. The on-laptop speech engine was
installed and measured later that day, on the user's yes (section 6).
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
starts on the shortcut, and that typing lands in another app, are for the test session. The
microphone has not been recorded from: the local engine was measured on synthesized speech
(section 6). The -40 dB silence gate is a standard figure, not measured in this room. How macOS dictation behaves when it stops on its own (and so
falls out of step with AirPoint's idea of it) is unknown; the twenty-second limit is there to end it
first, and `l` puts the two back in step.

## 6. The on-laptop engine, installed and measured (2026-11-25)

On the user's yes: `pip install faster-whisper` (1.2.1, with ctranslate2 4.9.1 and onnxruntime 1.32.0;
twenty new packages, none of the existing ones changed) and the `base.en` model (145 MB) through
`python -m airpoint.speech --setup`. The model loads in 0.3 s.

Measured with macOS's own speech synthesizer, not with people and not through the microphone: five
voices (Samantha, Daniel, Rishi, Tara, Aman), all seventeen command phrases and eight sentences,
two passes.

| Setting | Commands as intended | Sentences as intended |
|---|---|---|
| plain decoding | 64 of 75 (first run, fifteen phrases) | 24 of 25 |
| + the command words given as hotwords | 71 of 75 | 24 of 25 |
| + a token cap by clip length (as shipped) | 163 of 170 | 78 of 80 |
| + a repetition penalty of 1.3 (dropped) | 164 of 170 | 50 of 80: sentences cut short |

Median 0.22 s per phrase, slowest 1.3 s. Hiss, a tone and silence produced no event.

Two things this found, both fixed:
- One-word commands are misheard without help ("paste" as "peace", "pause" as "cause"). The command
  list is now passed to the recognizer as hotwords. An initial prompt did as well on accuracy but
  once came back as the transcript, so it was not used.
- A half-second clip once sent the model into a loop: "pause, pause, pause..." a hundred times,
  taking eight seconds, and that text would have been typed. Decoding is now capped at ten tokens a
  second of audio plus sixteen, and the talker refuses any transcript with more words than five a
  second plus three: it types nothing and says "could not make that out".

Left as it is: the bare word "tab" is heard as "time" in three of five voices, so "tab key" and
"next field" were added and are the names to use.
