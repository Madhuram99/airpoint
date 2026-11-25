"""The speech process and the app's side of the pipe. No microphone and no model here: both are stood in
for, in a thread for the loop itself and inside a real second process for everything about processes."""
import json
import queue
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from airpoint import speech
from airpoint.fsm import Event
from airpoint.speech import ACK_S, BLOCK, RATE, READY_S, Ear, Talker, WhisperRecognizer, serve
from tests.test_speech import Ears, _say, voice


class Mic:
    def __init__(self, audio=None, fail=False, close_takes=0.0):
        self.audio, self.fail, self.close_takes = voice(2.0) if audio is None else audio, fail, close_takes
        self.events, self.sent = [], 0

    def open(self):
        if self.fail:
            raise OSError("no input device")
        self.events.append("open")
        self.sent = 0

    def read(self):
        time.sleep(0.002)
        chunk = self.audio[self.sent:self.sent + BLOCK]
        self.sent += BLOCK
        return chunk if len(chunk) == BLOCK else np.zeros(BLOCK, dtype=np.float32)

    def close(self):
        time.sleep(self.close_takes)
        self.events.append("close")


class Wire:
    """Both ends of the process's pipes, in this test's hands."""

    def __init__(self):
        self.lines, self.events, self.result, self.died = queue.Queue(), [], [], []

    def __iter__(self):
        return iter(self.lines.get, None)

    def say(self, word):
        self.lines.put(word + "\n")

    def hang_up(self):
        self.lines.put(None)

    def write(self, text):
        self.events.append(json.loads(text))

    def flush(self):
        pass

    def kinds(self):
        return [e["event"] for e in self.events]

    def wait(self, kind, n=1, limit=5.0):
        deadline = time.time() + limit
        while time.time() < deadline:
            if self.kinds().count(kind) >= n:
                return [e for e in self.events if e["event"] == kind][n - 1]
            time.sleep(0.005)
        raise AssertionError(f"no '{kind}' in {self.kinds()}")

    def ended(self, limit=5.0):
        deadline = time.time() + limit
        while time.time() < deadline and not self.result:
            time.sleep(0.005)
        return self.result[0] if self.result else None


def running(**kw):
    wire = Wire()
    kw.setdefault("die", wire.died.append)
    threading.Thread(target=lambda: wire.result.append(serve(lines=wire, out=wire, **kw)), daemon=True).start()
    return wire


def test_the_process_listens_between_start_and_stop_and_answers_with_what_was_said():
    mic, ears = Mic(), Ears("Scroll down.")
    w = running(recognizer=ears, mic=mic)
    w.wait("ready")
    w.say("start")
    w.wait("listening")
    while mic.sent < 8 * BLOCK:
        time.sleep(0.002)
    w.say("stop")
    heard = w.wait("heard")
    assert heard["text"] == "Scroll down." and heard["problem"] == "" and heard["seconds"] >= 0.8
    assert ears.heard and ears.heard[0] >= 8 * BLOCK
    w.say("quit")
    assert w.ended() == 0 and mic.events == ["open", "close"] and w.died == []


def test_a_microphone_that_will_not_open_is_a_problem_and_the_next_start_still_works():
    mic = Mic(fail=True)
    w = running(recognizer=Ears("click"), mic=mic)
    w.say("start")
    assert "microphone did not open (OSError)" in w.wait("problem")["problem"] and "listening" not in w.kinds()
    mic.fail = False
    w.say("start")
    w.wait("listening")
    while mic.sent < 5 * BLOCK:
        time.sleep(0.002)
    w.say("stop")
    assert w.wait("heard")["text"] == "click"


def test_a_listen_ends_by_itself_and_what_is_not_speech_says_why():
    w = running(recognizer=Ears("click"), mic=Mic(), max_s=0.05)
    w.say("start")
    assert w.wait("heard")["text"] == "click"                    # nobody said stop
    quiet = running(recognizer=Ears("Thank you."), mic=Mic(voice(2.0, level=0.002)), max_s=0.05)
    quiet.say("start")
    heard = quiet.wait("heard")
    assert (heard["text"], heard["problem"]) == ("", "heard nothing")      # room noise never reaches the recognizer


def test_cancel_and_the_app_going_away_both_let_go_of_the_microphone_without_an_answer():
    mic = Mic()
    w = running(recognizer=Ears("click"), mic=mic)
    w.say("start")
    w.wait("listening")
    w.say("cancel")
    w.say("start")
    w.wait("listening", 2)
    assert mic.events[:2] == ["open", "close"]
    w.hang_up()                                                   # end of input while listening: the app has gone
    assert w.ended() == 0 and mic.events == ["open", "close", "open", "close"] and "heard" not in w.kinds()


def test_a_microphone_that_hangs_on_closing_ends_the_process_after_the_words_are_out():
    w = running(recognizer=Ears("Scroll down."), mic=Mic(close_takes=3.0), close_s=0.1)
    w.say("start")
    w.wait("listening")
    w.say("stop")
    assert w.wait("heard")["text"] == "Scroll down."
    w.wait("stuck")
    assert w.ended() == 3 and w.died == [3] and w.kinds().index("heard") < w.kinds().index("stuck")


def test_a_model_that_does_not_load_is_said_once_and_the_process_ends():
    class Broken:
        def load(self):
            raise RuntimeError("no model file")

    w = running(recognizer=Broken(), mic=Mic())
    assert "did not load (RuntimeError: no model file)" in w.wait("failed")["problem"] and w.ended() == 1


# ---- a real second process, with a stand-in microphone and recognizer inside it

CHILD = """
import os, sys, time
import numpy as np
from airpoint import speech

class Mic:
    def open(self):
        pass
    def read(self):
        time.sleep(0.01)
        return (0.2 * np.sin(2 * np.pi * 200 * np.arange(speech.BLOCK) / speech.RATE)).astype(np.float32)
    def close(self):
        time.sleep(CLOSE_TAKES)

class Ears:
    def transcribe(self, audio):
        return "scroll down"

out = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)
print("a library printing to stdout must not break the answers")
raise SystemExit(speech.serve(recognizer=Ears(), mic=Mic(), out=out, close_s=0.3))
"""


def child(close_takes=0.0):
    return [sys.executable, "-c", CHILD.replace("CLOSE_TAKES", str(close_takes))]


def answer(ear, limit=10.0):
    deadline = time.time() + limit
    while time.time() < deadline:
        got = ear.poll()
        if got is not None:
            return got
        time.sleep(0.01)
    raise AssertionError("the speech process never answered")


def test_the_app_side_starts_the_process_and_gets_the_words_back():
    ear = Ear(cmd=child())
    proc = ear.proc
    try:
        k = Talker(engine=ear)
        assert k.local and k.toggle(0.0) == "listening"
        time.sleep(0.6)
        assert k.poll(0.6) == [] and k.toggle(0.7) == "thinking"
        deadline = time.time() + 10
        events = []
        while k.state != "idle" and time.time() < deadline:
            events = k.poll(1.0)
            time.sleep(0.01)
        assert events == [Event("scroll", -5)] and k.heard == "scroll down" and k.problem == ""
        assert ear.ready and ear.alive
    finally:
        k.close()
    assert proc.wait(timeout=5) == 0 and ear.proc is None          # closing the talker ended the process


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_a_process_that_never_answers_is_ended_and_the_next_start_gets_a_new_one():
    clock = Clock()
    ear = Ear(cmd=[sys.executable, "-c", "import time; time.sleep(60)"], clock=clock)
    stuck = ear.proc
    try:
        ear.start()
        clock.t += READY_S + ACK_S - 1.0
        assert ear.poll() is None and ear.alive                    # still inside the time allowed for loading
        clock.t += 2.0
        assert ear.poll() == ("", "the microphone did not answer: try again")
        assert stuck.wait(timeout=5) != 0 and ear.proc is None and ear.poll() is None
        ear.cmd = child()
        ear.start()                                                # a new process, the real clock no longer matters
        assert ear.proc is not stuck and ear.alive
        time.sleep(0.5)
        ear.stop()
        clock.t += 1.0
        assert answer(ear) == ("scroll down", "")
    finally:
        ear.close()


def test_a_process_whose_microphone_hangs_still_delivers_and_is_replaced():
    ear = Ear(cmd=child(close_takes=30.0))
    first = ear.proc
    try:
        ear.start()
        time.sleep(0.5)
        ear.stop()
        assert answer(ear) == ("scroll down", "")                  # the words came before the hang mattered
        assert first.wait(timeout=5) == 3                          # then the process ended itself
        deadline = time.time() + 5
        while ear.proc is not None and time.time() < deadline:
            ear.poll()
            time.sleep(0.01)
        ear.start()
        assert ear.proc is not first and ear.alive
    finally:
        ear.close()


def test_the_process_ends_when_the_app_does():
    p = subprocess.Popen(child(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert json.loads(p.stdout.readline())["event"] == "ready"
    p.stdin.write(b"start\n")
    p.stdin.flush()
    assert json.loads(p.stdout.readline())["event"] == "listening"
    p.stdin.close()                                                # what the operating system does when the app dies
    assert p.wait(timeout=5) == 0


def test_a_process_that_cannot_be_started_is_a_microphone_that_did_not_open():
    k = Talker(engine=Ear(cmd=["/no/such/program"]))
    assert k.toggle(0.0) == "idle" and "microphone did not open (OSError)" in k.problem


FILE_CHILD = """
import os, sys, time
import numpy as np
from airpoint import speech

class FileMic:
    def open(self):
        self.audio, self.at = speech.read_wav(sys.argv[1]), 0
    def read(self):
        time.sleep(0.005)
        chunk = self.audio[self.at:self.at + speech.BLOCK]
        self.at += speech.BLOCK
        return chunk if len(chunk) == speech.BLOCK else np.zeros(speech.BLOCK, dtype=np.float32)
    def close(self):
        pass

out = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)
raise SystemExit(speech.serve(mic=FileMic(), model="base.en", out=out))
"""


def test_the_real_recognizer_inside_the_real_process_hears_a_command(tmp_path):
    """Runs only where the on-laptop engine and its model are installed. No microphone: a synthesized file."""
    if WhisperRecognizer.missing("base.en"):
        pytest.skip(WhisperRecognizer.missing("base.en"))
    if _say("scroll down", tmp_path / "a.wav") is None:
        pytest.skip("no speech synthesizer here")
    ear = Ear(cmd=[sys.executable, "-c", FILE_CHILD, str(tmp_path / "a.wav")])
    try:
        k = Talker(engine=ear)
        k.toggle(0.0)
        deadline = time.time() + 60
        while ear.waiting != "words" and time.time() < deadline:   # the model loads first
            assert k.poll(0.0) == []
            time.sleep(0.05)
        time.sleep(0.5)
        k.toggle(2.0)
        events = []
        while k.state != "idle" and time.time() < deadline:
            events = k.poll(3.0)
            time.sleep(0.02)
        assert events == [Event("scroll", -5)], (k.heard, k.problem)
    finally:
        k.close()
