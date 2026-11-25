"""Tap to talk: say a command or some text; it is carried out or typed.

Two engines. The default is the dictation built into macOS: AirPoint only presses its shortcut and
the system listens and types. The optional local engine records the microphone between start and
stop and recognises it on this laptop with faster-whisper; it also understands spoken commands.

The local engine lives in a process of its own (`--serve`, see Ear). The app never opens the
microphone or loads the recognizer, so nothing they do can stall a frame, and ending that process
always lets go of the microphone.

    venv/bin/python -m airpoint.speech                 # what is set up on this laptop
    venv/bin/python -m airpoint.speech --setup         # fetch the local speech model (asks first)
    venv/bin/python -m airpoint.speech --file hi.wav   # what a recording is heard as
    venv/bin/python -m airpoint.speech --listen 5      # the same, from the microphone
"""
import argparse
import importlib.util
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import wave

import numpy as np

from . import proc
from .fsm import Event

RATE, MAX_LISTEN_S, MIN_S = 16000, 20.0, 0.25
BLOCK = RATE // 10             # the microphone is read a tenth of a second at a time
CLOSE_S = 2.0                  # a microphone that takes longer than this to close has hung: the process ends itself
READY_S, ACK_S, ANSWER_S, ANSWER_PER_S = 40.0, 5.0, 15.0, 2.0     # the app's time limits for the speech process
QUIET_PEAK = 0.01              # a recording that never gets louder than this (-40 dB) holds no speech
MOUTH_REST_S = 1.0             # after listening ends, the face still ignores the mouth for this long
MAX_WORDS_PER_S, WORDS_SLACK = 5.0, 3      # nobody speaks faster: more words than this is the recognizer looping
DICTATION_KEYS = ("control", "option", "d")
MODEL_MB = {"tiny.en": 78, "base.en": 148, "small.en": 487}      # download sizes on huggingface.co/Systran
COMMANDS = {
    "click": Event("click"), "double click": Event("double_click"), "right click": Event("right_click"),
    "scroll up": Event("scroll", 5), "scroll down": Event("scroll", -5),
    "enter": Event("hotkey", ("enter",)), "new line": Event("hotkey", ("enter",)),
    "tab": Event("hotkey", ("tab",)), "tab key": Event("hotkey", ("tab",)), "next field": Event("hotkey", ("tab",)),
    "escape": Event("hotkey", ("escape",)),
    "delete": Event("hotkey", ("backspace",)), "undo": Event("hotkey", ("command", "z")),
    "copy": Event("hotkey", ("command", "c")), "paste": Event("hotkey", ("command", "v")),
    "select all": Event("hotkey", ("command", "a")), "pause": Event("pause"),
}


HOTWORDS = ", ".join(COMMANDS)      # told to the recognizer: one-word commands are otherwise misheard ("paste" as "peace")


def interpret(text):
    """Recognised speech -> one Event, or None for silence. A phrase that is exactly a command runs it
    ("Click." and "scroll DOWN" count); "Computer, ..." is a request for the AI helper; anything else
    is typed as it was heard."""
    spoken = re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", text.lower())).strip()
    if not spoken:
        return None
    errand = re.match(r"\s*(?:hey[\s,]+)?computer[\s,.:;!]+(\S.*)", text, flags=re.I | re.S)
    if errand:
        return Event("ask", errand.group(1).strip())
    return COMMANDS.get(spoken) or Event("type", text.strip() + " ")


class Microphone:
    """The default microphone at 16 kHz mono, read a tenth of a second at a time by the thread that
    owns it. There is no callback: the audio thread never runs Python code."""

    def __init__(self):
        self.stream = None

    def open(self):
        import sounddevice as sd
        self.stream = sd.InputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=BLOCK)
        self.stream.start()

    def read(self):
        data, _ = self.stream.read(BLOCK)
        return data[:, 0].copy()

    def close(self):
        stream, self.stream = self.stream, None
        if stream is not None:
            stream.stop()
            stream.close()


class MicRecorder:
    """start() ... stop() -> everything the microphone gave in between. For the command line."""

    def __init__(self, mic=None):
        self.mic, self.chunks, self.on, self.thread = mic or Microphone(), [], False, None

    def start(self):
        self.mic.open()
        self.chunks, self.on = [], True
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        while self.on:
            self.chunks.append(self.mic.read())

    def stop(self):
        """-> the recording as float32 samples."""
        self.on = False
        if self.thread is not None:
            self.thread.join(1.0)
            self.thread = None
        self.mic.close()
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, dtype=np.float32)


def hear(recognizer, audio):
    """A finished recording -> (text, problem): what was said, or why nothing will be done."""
    if len(audio) < MIN_S * RATE:
        return "", "too short to be speech"
    if not audio.any():
        return "", "the microphone gave no sound: allow it in System Settings > Privacy & Security > Microphone"
    if float(np.abs(audio).max()) < QUIET_PEAK:
        return "", "heard nothing"
    try:
        text = recognizer.transcribe(audio)
    except Exception as e:                       # a failed recognition must not take anything down
        return "", f"could not recognise ({type(e).__name__}: {e})"
    if len(text.split()) > MAX_WORDS_PER_S * len(audio) / RATE + WORDS_SLACK:
        return "", "could not make that out: say it again"      # never type a recognizer's loop
    return (text, "") if interpret(text) else ("", "heard nothing")


class WhisperRecognizer:
    """Speech to text on this laptop with faster-whisper, English. It never downloads by itself: the
    model is fetched once, on purpose, by `python -m airpoint.speech --setup`."""

    def __init__(self, size="base.en"):
        self.size, self.model, self.lock = size, None, threading.Lock()

    @staticmethod
    def missing(size):
        """-> what stands between this laptop and local recognition, or None when it is ready.
        Looks at what is installed and cached without loading the recognizer into this process."""
        if importlib.util.find_spec("faster_whisper") is None:
            return "faster-whisper is not installed (venv/bin/pip install faster-whisper)"
        try:
            from huggingface_hub import try_to_load_from_cache
            found = try_to_load_from_cache(f"Systran/faster-whisper-{size}", "model.bin")
        except Exception:
            found = None
        if not isinstance(found, str):
            return f"the '{size}' speech model is not on this laptop (venv/bin/python -m airpoint.speech --setup)"
        return None

    def load(self):
        with self.lock:
            if self.model is None:
                from faster_whisper import WhisperModel
                self.model = WhisperModel(self.size, device="cpu", compute_type="int8", local_files_only=True)
        return self.model

    def transcribe(self, audio):
        # A short clip can send the model into a loop ("pause, pause, pause ..." for eight seconds). The token
        # cap bounds that; the Talker refuses whatever still comes out too long for the time spoken. A
        # repetition penalty was tried and dropped: it cut real sentences short ("the copy of").
        segments, _ = self.load().transcribe(audio, language="en", beam_size=1, vad_filter=True,
                                             condition_on_previous_text=False, hotwords=HOTWORDS,
                                             max_new_tokens=int(10 * len(audio) / RATE) + 16)
        return " ".join(s.text.strip() for s in segments).strip()


class InProcess:
    """Recorder and recognizer inside this process, the recognition on a thread. For tests and small
    tools; the app talks to an Ear instead."""

    def __init__(self, recognizer, recorder):
        self.recognizer, self.recorder, self.answers, self.recording = recognizer, recorder, [], False

    def start(self):
        self.recorder.start()
        self.recording = True

    def stop(self):
        audio, self.recording = self.recorder.stop(), False
        threading.Thread(target=lambda: self.answers.append(hear(self.recognizer, audio)), daemon=True).start()

    def cancel(self):
        if self.recording:
            self.recorder.stop()
            self.recording = False

    def poll(self):
        return self.answers.pop(0) if self.answers else None

    def close(self):
        pass


class Ear:
    """The speech process and the pipe to it. Nothing here blocks: words go out through a non-blocking
    pipe, answers come back on a reader thread, every wait has a time limit, and a process that
    misses one is ended and replaced at the next start."""

    def __init__(self, model="base.en", cmd=None, clock=time.monotonic, spawn=proc.spawn):
        self.model, self.clock, self.spawn = model, clock, spawn
        own, env = proc.module("airpoint.speech", "--serve", "--model", model)
        self.cmd, self.env = cmd or own, None if cmd else env
        self.proc, self.inbox, self.ready, self.failed = None, queue.Queue(), False, ""
        self.waiting, self.deadline, self.t_listen, self.t_launch = None, 0.0, 0.0, 0.0   # waiting: ack | words | answer
        self._launch()

    @property
    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def _launch(self):
        self.ready, self.failed, self.inbox, self.t_launch = False, "", queue.Queue(), self.clock()
        try:
            self.proc = self.spawn(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   **({"env": self.env} if self.env else {}))
            os.set_blocking(self.proc.stdin.fileno(), False)
        except (OSError, ValueError):
            self.proc = None
            return
        threading.Thread(target=self._read, args=(self.proc.stdout, self.inbox), daemon=True).start()

    @staticmethod
    def _read(pipe, inbox):
        """Each process has its own queue, so one that is being replaced cannot speak for the next."""
        try:
            for line in pipe:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict):
                    inbox.put(event)
        except (OSError, ValueError):
            pass
        inbox.put({"event": "gone"})

    def _send(self, word):
        try:
            os.write(self.proc.stdin.fileno(), (word + "\n").encode())
            return True
        except (OSError, ValueError, AttributeError):
            return False

    def _drop(self):
        proc, self.proc = self.proc, None
        if proc is not None:
            try:
                proc.kill()
                proc.wait(timeout=1.0)
            except (OSError, subprocess.TimeoutExpired):
                pass

    def start(self):
        if not self.alive:
            self._launch()
        if not self.alive or not self._send("start"):
            raise OSError("the speech process did not start")
        now = self.clock()
        loading = 0.0 if self.ready else max(READY_S - (now - self.t_launch), 0.0)
        self.waiting, self.t_listen, self.deadline = "ack", now, now + loading + ACK_S

    def stop(self):
        if self.waiting in ("ack", "words"):
            now = self.clock()
            self._send("stop")
            late = max(self.deadline - now, 0.0) if self.waiting == "ack" else 0.0
            self.waiting, self.deadline = "answer", now + late + ANSWER_S + ANSWER_PER_S * (now - self.t_listen)

    def cancel(self):
        if self.waiting is not None:
            self._send("cancel")
            self.waiting = None

    def poll(self):
        """-> (text, problem) once for each listen, when it is known; None until then."""
        answer = None
        while True:
            try:
                event = self.inbox.get_nowait()
            except queue.Empty:
                break
            kind = event.get("event")
            if kind == "ready":
                self.ready = True
            elif kind == "failed":
                self.failed = str(event.get("problem", "the speech model did not load"))
            elif kind == "listening" and self.waiting == "ack":
                self.waiting = "words"
            elif kind in ("heard", "problem") and self.waiting is not None:
                answer, self.waiting = (str(event.get("text", "")), str(event.get("problem", ""))), None
            elif kind == "gone":
                self.proc = None
                if self.waiting is not None:
                    answer, self.waiting = ("", self.failed or "the speech process stopped: try again"), None
        if answer is None and self.waiting in ("ack", "answer") and self.clock() > self.deadline:
            late = "the microphone did not answer" if self.waiting == "ack" else "recognising took too long"
            self._drop()
            answer, self.waiting = ("", f"{late}: try again"), None
        return answer

    def close(self):
        proc, self.proc, self.waiting = self.proc, None, None
        if proc is None:
            return
        try:
            proc.stdin.close()                  # end of input ends the process, and with it the microphone
            proc.wait(timeout=1.0)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            proc.kill()


class Talker:
    """Tap to talk: toggle() starts listening, toggle() again ends it. state: idle | listening | thinking.

    With an `engine` (the local one) the microphone is recorded between the two taps and recognised
    away from the frame loop; poll() hands back what was said as events. Without one (the system
    engine) each tap presses the dictation shortcut through `press`, and macOS listens and types."""

    def __init__(self, recognizer=None, recorder=None, dictation_keys=DICTATION_KEYS, press=None, engine=None):
        self.recognizer, self.recorder = recognizer, recorder
        both = recognizer is not None and recorder is not None
        self.engine = engine or (InProcess(recognizer, recorder) if both else None)
        self.keys, self.press = tuple(dictation_keys), press or (lambda keys: None)
        self.state, self.t_start, self.t_end = "idle", 0.0, None
        self.heard, self.problem = "", ""

    @property
    def local(self):
        return self.engine is not None

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
                self.engine.start()
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
        self.engine.stop()
        self.state = "thinking"

    def poll(self, t):
        """Call every frame. -> the events of a finished recognition; also ends an overlong listen."""
        if self.state == "listening" and t - self.t_start >= MAX_LISTEN_S:
            self.stop(t)
        if not self.local or self.state == "idle":
            return []
        answer = self.engine.poll()              # also how a microphone that would not open reports back
        if answer is None:
            return []
        self.heard, self.problem = answer
        self.state, self.t_end = "idle", t
        event = interpret(self.heard) if self.heard else None
        return [event] if event else []

    def close(self):
        """Drop whatever is going on and let go of the speech process. The caller ends a system
        dictation first, with stop()."""
        if self.local:
            if self.state != "idle":
                self.engine.cancel()
            self.engine.close()
        self.state = "idle"


def make_talker(profile, press=None, ear=None):
    """The profile picks the engine; anything the local one lacks falls back to system dictation."""
    if profile.speech == "local":
        why = WhisperRecognizer.missing(profile.speech_model)
        if why is None:
            return Talker(dictation_keys=profile.dictation_keys, press=press,
                          engine=(ear or Ear)(profile.speech_model))
        print(f"[AirPoint] local speech is off: {why}. Using macOS dictation instead.")
    return Talker(dictation_keys=profile.dictation_keys, press=press)


def serve(recognizer=None, mic=None, model="base.en", lines=None, out=None, max_s=MAX_LISTEN_S,
          close_s=CLOSE_S, die=os._exit):
    """The speech process. Words come in on stdin (`start`, `stop`, `cancel`, `quit`); one JSON line
    goes out for each thing that happens (`ready`, `listening`, `heard`, `problem`, `failed`, `stuck`).

    The microphone is read on this thread. What was said goes out before the stream is closed, and
    closing has a time limit: a hung audio call ends this process after the words are safe. A listen
    ends by itself after `max_s`, and end of input ends everything, so the microphone cannot stay
    open behind an app that has stopped or gone."""
    out = out or sys.stdout
    inbox = queue.Queue()

    def tell(event, **more):
        out.write(json.dumps(dict(event=event, **more)) + "\n")
        out.flush()

    def read():
        for line in (sys.stdin if lines is None else lines):
            inbox.put(line.strip())
        inbox.put("quit")

    threading.Thread(target=read, daemon=True).start()
    try:
        recognizer = recognizer or WhisperRecognizer(model)
        getattr(recognizer, "load", lambda: None)()          # the model is in memory before the first word
    except Exception as e:
        tell("failed", problem=f"the speech model did not load ({type(e).__name__}: {e})"[:200])
        return 1
    tell("ready")
    mic = mic or Microphone()
    while True:
        word = inbox.get()
        if word == "quit":
            return 0
        if word != "start":
            continue
        try:
            mic.open()
        except Exception as e:
            tell("problem", problem=f"the microphone did not open ({type(e).__name__})")
            continue
        tell("listening")
        chunks, word, t0 = [], None, time.monotonic()
        while word is None:
            try:
                chunks.append(mic.read())
            except Exception as e:
                word = f"broke:{type(e).__name__}"
                break
            try:
                word = inbox.get_nowait()
            except queue.Empty:
                word = "stop" if time.monotonic() - t0 >= max_s else None
            if word == "start":
                word = None
        closed = threading.Event()

        def close():
            try:
                mic.close()
            except Exception:
                pass
            closed.set()

        threading.Thread(target=close, daemon=True).start()   # closing never delays the answer
        if word == "stop":
            audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
            text, problem = hear(recognizer, audio)
            tell("heard", text=text, problem=problem, seconds=round(len(audio) / RATE, 2))
        elif word.startswith("broke:"):
            tell("heard", text="", problem=f"the microphone stopped ({word[6:]})", seconds=0.0)
        if not closed.wait(close_s):
            tell("stuck")
            die(3)                                             # the words are out; the app starts a fresh process
            return 3
        if word == "quit":
            return 0


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
    ap.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)       # the speech process, started by the app
    args = ap.parse_args(argv)

    if args.serve:
        out = os.fdopen(os.dup(1), "w")
        os.dup2(2, 1)                          # whatever a library prints goes to stderr, never into the answers
        raise SystemExit(serve(model=args.model, out=out))

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
