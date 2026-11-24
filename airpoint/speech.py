"""Tap to talk: say a command or some text; it is carried out or typed.

Two engines. The default is the dictation built into macOS: AirPoint only presses its shortcut and
the system listens and types. The optional local engine records the microphone between start and
stop and recognises it on this laptop with faster-whisper (not installed by default):
    venv/bin/pip install faster-whisper
"""
import re
import threading

import numpy as np

from .fsm import Event

RATE, MAX_LISTEN_S = 16000, 20.0
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
    """Speech to text on this laptop. Raises ImportError until faster-whisper is installed."""

    def __init__(self, size="base.en"):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(size, device="cpu", compute_type="int8")

    def transcribe(self, audio):
        segments, _ = self.model.transcribe(audio, language="en", beam_size=1)
        return " ".join(s.text.strip() for s in segments).strip()


class Talker:
    """Tap to talk. toggle() starts, toggle() again stops; both return events for the frame loop.

    With a `recognizer`: records between the two taps, recognises on a worker thread, and poll()
    returns the result when it is ready. Without one: each tap presses the system's dictation
    shortcut and macOS does the listening and the typing. state: idle | listening | thinking."""

    def __init__(self, recognizer=None, recorder=None, dictation_keys=("control", "option", "d")):
        self.recognizer, self.recorder, self.keys = recognizer, recorder, tuple(dictation_keys)
        self.state, self.t_start, self.heard = "idle", 0.0, ""
        self.worker, self.ready = None, []

    @property
    def local(self):
        return self.recognizer is not None and self.recorder is not None

    def toggle(self, t):
        if self.state == "thinking":
            return []
        if self.state == "idle":
            self.state, self.t_start = "listening", t
            if self.local:
                self.recorder.start()
                return []
            return [Event("hotkey", self.keys)]
        return self._stop()

    def _stop(self):
        if not self.local:
            self.state = "idle"
            return [Event("hotkey", self.keys)]
        audio = self.recorder.stop()
        self.state = "thinking"
        self.worker = threading.Thread(target=self._recognise, args=(audio,), daemon=True)
        self.worker.start()
        return []

    def _recognise(self, audio):
        try:
            text = self.recognizer.transcribe(audio) if len(audio) >= RATE // 4 else ""
        except Exception as e:                       # a failed recognition must not take the app down
            text = ""
            self.heard = f"could not recognise ({type(e).__name__})"
        else:
            self.heard = text
        event = interpret(text)
        self.ready.append([event] if event is not None else [])

    def poll(self, t):
        """Call every frame. -> events from a finished recognition; also ends an overlong listen."""
        if self.state == "listening" and t - self.t_start >= MAX_LISTEN_S:
            return self._stop()
        if self.state == "thinking" and self.ready:
            self.state = "idle"
            return self.ready.pop(0)
        return []

    def close(self):
        if self.state == "listening" and self.local:
            self.recorder.stop()
        self.state = "idle"
