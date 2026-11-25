"""Tap to talk: say a command or some text; it is carried out or typed.

Two engines. The default is the dictation built into macOS: AirPoint only presses its shortcut and
the system listens and types. The optional local engine records the microphone between start and
stop and recognises it on this laptop with faster-whisper; it also understands spoken commands.

    venv/bin/python -m airpoint.speech                 # what is set up on this laptop
    venv/bin/python -m airpoint.speech --setup         # fetch the local speech model (asks first)
    venv/bin/python -m airpoint.speech --file hi.wav   # what a recording is heard as
    venv/bin/python -m airpoint.speech --listen 5      # the same, from the microphone
"""
import argparse
import importlib.util
import re
import threading
import time
import wave

import numpy as np

from .fsm import Event

RATE, MAX_LISTEN_S, MIN_S = 16000, 20.0, 0.25
QUIET_PEAK = 0.01              # a recording that never gets louder than this (-40 dB) holds no speech
MOUTH_REST_S = 1.0             # after listening ends, the face still ignores the mouth for this long
DICTATION_KEYS = ("control", "option", "d")
MODEL_MB = {"tiny.en": 78, "base.en": 148, "small.en": 487}      # download sizes on huggingface.co/Systran
COMMANDS = {
    "click": Event("click"), "double click": Event("double_click"), "right click": Event("right_click"),
    "scroll up": Event("scroll", 5), "scroll down": Event("scroll", -5),
    "enter": Event("hotkey", ("enter",)), "new line": Event("hotkey", ("enter",)),
    "tab": Event("hotkey", ("tab",)), "escape": Event("hotkey", ("escape",)),
    "delete": Event("hotkey", ("backspace",)), "undo": Event("hotkey", ("command", "z")),
    "copy": Event("hotkey", ("command", "c")), "paste": Event("hotkey", ("command", "v")),
    "select all": Event("hotkey", ("command", "a")), "pause": Event("pause"),
}


def interpret(text):
    """Recognised speech -> one Event, or None for silence. A phrase that is exactly a command runs it
    ("Click." and "scroll DOWN" count); anything else is typed as it was heard."""
    spoken = re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", text.lower())).strip()
    if not spoken:
        return None
    return COMMANDS.get(spoken) or Event("type", text.strip() + " ")


class MicRecorder:
    """The default microphone at 16 kHz mono, only between start() and stop()."""

    def __init__(self):
        self.stream, self.chunks = None, []

    def start(self):
        import sounddevice as sd
        self.chunks = []
        self.stream = sd.InputStream(samplerate=RATE, channels=1, dtype="float32",
                                     callback=lambda data, frames, when, status: self.chunks.append(data[:, 0].copy()))
        self.stream.start()

    def stop(self):
        """-> the recording as float32 samples."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, dtype=np.float32)


class WhisperRecognizer:
    """Speech to text on this laptop with faster-whisper, English. It never downloads by itself: the
    model is fetched once, on purpose, by `python -m airpoint.speech --setup`."""

    def __init__(self, size="base.en"):
        self.size, self.model, self.lock = size, None, threading.Lock()

    @staticmethod
    def missing(size):
        """-> what stands between this laptop and local recognition, or None when it is ready."""
        if importlib.util.find_spec("faster_whisper") is None:
            return "faster-whisper is not installed (venv/bin/pip install faster-whisper)"
        try:
            from faster_whisper.utils import download_model
            download_model(size, local_files_only=True)
        except Exception:
            return f"the '{size}' speech model is not on this laptop (venv/bin/python -m airpoint.speech --setup)"
        return None

    def load(self):
        with self.lock:
            if self.model is None:
                from faster_whisper import WhisperModel
                self.model = WhisperModel(self.size, device="cpu", compute_type="int8", local_files_only=True)
        return self.model

    def warm(self):
        """Load ahead of the first sentence, on a thread of its own. A failure waits for transcribe to report."""
        try:
            self.load()
        except Exception:
            pass

    def transcribe(self, audio):
        segments, _ = self.load().transcribe(audio, language="en", beam_size=1, vad_filter=True,
                                             condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()


class Talker:
    """Tap to talk: toggle() starts listening, toggle() again ends it. state: idle | listening | thinking.

    With a `recognizer` and a `recorder` (the local engine) the microphone is recorded between the
    two taps, recognised on a worker thread, and poll() hands back what was said as events.
    Without them (the system engine) each tap presses the dictation shortcut through `press`, and
    macOS does the listening and the typing itself."""

    def __init__(self, recognizer=None, recorder=None, dictation_keys=DICTATION_KEYS, press=None):
        self.recognizer, self.recorder = recognizer, recorder
        self.keys, self.press = tuple(dictation_keys), press or (lambda keys: None)
        self.state, self.t_start, self.t_end = "idle", 0.0, None
        self.heard, self.problem, self.ready = "", "", []

    @property
    def local(self):
        return self.recognizer is not None and self.recorder is not None

    def left(self, t):
        """Seconds of listening left before it ends by itself."""
        return max(MAX_LISTEN_S - (t - self.t_start), 0.0)

    def mouth_busy(self, t):
        """True while a moving mouth is speech, not a sign (face mode): listening, and a moment after."""
        return self.state != "idle" or (self.t_end is not None and t - self.t_end < MOUTH_REST_S)

    def toggle(self, t):
        if self.state == "idle":
            self.start(t)
        elif self.state == "listening":
            self.stop(t)
        return self.state

    def start(self, t):
        self.heard = self.problem = ""
        if self.local:
            try:
                self.recorder.start()
            except Exception as e:                   # no microphone, or the system refused it
                self.problem, self.t_end = f"the microphone did not open ({type(e).__name__})", t
                return
        else:
            self.press(self.keys)
        self.state, self.t_start = "listening", t

    def stop(self, t):
        if self.state != "listening":
            return
        self.t_end = t
        if not self.local:
            self.press(self.keys)
            self.state = "idle"
            return
        audio = self.recorder.stop()
        self.state = "thinking"
        threading.Thread(target=self._recognise, args=(audio,), daemon=True).start()

    def _recognise(self, audio):
        text, problem = "", ""
        if len(audio) < MIN_S * RATE:
            problem = "too short to be speech"
        elif not audio.any():
            problem = "the microphone gave no sound: allow it in System Settings > Privacy & Security > Microphone"
        elif float(np.abs(audio).max()) < QUIET_PEAK:
            problem = "heard nothing"
        else:
            try:
                text = self.recognizer.transcribe(audio)
            except Exception as e:                   # a failed recognition must not take the app down
                problem = f"could not recognise ({type(e).__name__}: {e})"
        event = interpret(text)
        self.ready.append((text, problem or ("" if event else "heard nothing"), [event] if event else []))

    def poll(self, t):
        """Call every frame. -> the events of a finished recognition; also ends an overlong listen."""
        if self.state == "listening" and t - self.t_start >= MAX_LISTEN_S:
            self.stop(t)
        if self.state == "thinking" and self.ready:
            self.heard, self.problem, events = self.ready.pop(0)
            self.state, self.t_end = "idle", t
            return events
        return []

    def close(self):
        """Drop whatever is going on. The caller ends a system dictation first, with stop()."""
        if self.state == "listening" and self.local:
            self.recorder.stop()
        self.state = "idle"


def make_talker(profile, press=None):
    """The profile picks the engine; anything the local one lacks falls back to system dictation."""
    if profile.speech == "local":
        why = WhisperRecognizer.missing(profile.speech_model)
        if why is None:
            ears = WhisperRecognizer(profile.speech_model)
            threading.Thread(target=ears.warm, daemon=True).start()
            return Talker(ears, MicRecorder(), profile.dictation_keys, press)
        print(f"[AirPoint] local speech is off: {why}. Using macOS dictation instead.")
    return Talker(dictation_keys=profile.dictation_keys, press=press)


def read_wav(path):
    """A 16-bit WAV file -> float32 mono samples at 16 kHz."""
    with wave.open(path) as f:
        if f.getsampwidth() != 2:
            raise ValueError("need a 16-bit WAV file (say -o x.wav --data-format=LEI16@16000 \"...\")")
        rate, channels = f.getframerate(), f.getnchannels()
        audio = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != RATE:
        from scipy.signal import resample_poly
        audio = resample_poly(audio, RATE, rate).astype(np.float32)
    return audio


def main(argv=None, recognizer=None, recorder=None, ask=input, sleep=time.sleep):
    ap = argparse.ArgumentParser(description="AirPoint speech: what is set up, and what a recording is heard as")
    ap.add_argument("--model", default="base.en", help="tiny.en (fastest), base.en, small.en (most accurate)")
    ap.add_argument("--setup", action="store_true", help="download the local speech model")
    ap.add_argument("--yes", action="store_true", help="with --setup: do not ask before downloading")
    ap.add_argument("--file", help="a WAV file to recognise")
    ap.add_argument("--listen", type=float, metavar="SECONDS", help="record the microphone, then recognise")
    args = ap.parse_args(argv)

    if args.setup:
        if importlib.util.find_spec("faster_whisper") is None:
            raise SystemExit("faster-whisper is not installed. First: venv/bin/pip install faster-whisper")
        size = f"about {MODEL_MB[args.model]} MB" if args.model in MODEL_MB else "size unknown"
        print(f"This downloads the '{args.model}' speech model ({size}) from huggingface.co into ~/.cache/huggingface.")
        if not args.yes and ask("Download it now? [y/N] ").strip().lower() not in ("y", "yes"):
            raise SystemExit("nothing downloaded")
        from faster_whisper.utils import download_model
        print("model is at", download_model(args.model))
        return
    if args.file or args.listen:
        if recognizer is None:
            why = WhisperRecognizer.missing(args.model)
            if why:
                raise SystemExit(f"local speech is not ready: {why}")
            recognizer = WhisperRecognizer(args.model)
        if args.file:
            audio = read_wav(args.file)
        else:
            recorder = recorder or MicRecorder()
            print(f"listening for {args.listen:.0f} s ... (nothing is saved)")
            recorder.start()
            sleep(args.listen)
            audio = recorder.stop()
        peak = float(np.abs(audio).max()) if len(audio) else 0.0
        t0 = time.perf_counter()
        text = recognizer.transcribe(audio)
        print(f"{len(audio) / RATE:.1f} s of sound, peak {peak:.3f} | recognised in {time.perf_counter() - t0:.2f} s")
        print(f"heard: {text!r}")
        print(f"does:  {interpret(text)}")
        return
    why = WhisperRecognizer.missing(args.model)
    print("local engine :", "ready" if why is None else f"off: {why}")
    print("system engine: macOS dictation. Once: System Settings > Keyboard > Dictation > on, and set its")
    print("               Shortcut to Customize > control+option+d (or tell AirPoint yours: "
          "python -m airpoint.profile <user> --dictation-keys ...)")
    print("spoken commands (local engine):", ", ".join(COMMANDS))


if __name__ == "__main__":
    main()
