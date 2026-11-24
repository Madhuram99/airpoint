import time

import numpy as np

from airpoint.fsm import Event
from airpoint.speech import MAX_LISTEN_S, RATE, Talker, interpret


def test_exact_commands_run_and_everything_else_is_typed():
    assert interpret("click") == Event("click")
    assert interpret(" Click. ") == Event("click")
    assert interpret("Scroll DOWN!") == Event("scroll", -5)
    assert interpret("select all") == Event("hotkey", ("command", "a"))
    assert interpret("pause") == Event("pause")
    assert interpret("Click the blue button.") == Event("type", "Click the blue button. ")
    assert interpret("Hello, I will be late.") == Event("type", "Hello, I will be late. ")
    assert interpret("   ") is None and interpret("...") is None


class Mic:
    def __init__(self, seconds=1.0):
        self.seconds, self.events = seconds, []

    def start(self):
        self.events.append("start")

    def stop(self):
        self.events.append("stop")
        return np.zeros(int(self.seconds * RATE), dtype=np.float32)


class Ears:
    def __init__(self, text="scroll down", fail=False):
        self.text, self.fail, self.heard = text, fail, []

    def transcribe(self, audio):
        self.heard.append(len(audio))
        if self.fail:
            raise RuntimeError("model blew up")
        return self.text


def wait(talker, t=0.0):
    deadline = time.time() + 5
    while time.time() < deadline:
        events = talker.poll(t)
        if events or talker.state == "idle":
            return events
        time.sleep(0.01)
    raise AssertionError("recognition never finished")


def test_local_engine_records_between_two_taps_and_returns_the_result_later():
    mic, ears = Mic(), Ears("Scroll down.")
    k = Talker(ears, mic)
    assert k.local and k.toggle(0.0) == [] and k.state == "listening" and mic.events == ["start"]
    assert k.poll(1.0) == []
    assert k.toggle(2.0) == [] and k.state == "thinking" and mic.events == ["start", "stop"]
    assert k.toggle(2.1) == []                                    # a tap while thinking is ignored
    assert wait(k) == [Event("scroll", -5)] and k.state == "idle" and k.heard == "Scroll down."
    assert ears.heard == [RATE]


def test_free_speech_is_typed_and_silence_or_a_failure_does_nothing():
    k = Talker(Ears("I will be late."), Mic())
    k.toggle(0.0)
    k.toggle(1.0)
    assert wait(k) == [Event("type", "I will be late. ")]
    quiet = Talker(Ears("anything"), Mic(seconds=0.1))            # a slip of the hand: too short to be speech
    quiet.toggle(0.0)
    quiet.toggle(0.1)
    assert wait(quiet) == [] and quiet.recognizer.heard == []
    broken = Talker(Ears(fail=True), Mic())
    broken.toggle(0.0)
    broken.toggle(1.0)
    assert wait(broken) == [] and "could not recognise" in broken.heard


def test_listening_ends_by_itself():
    mic = Mic()
    k = Talker(Ears("click"), mic)
    k.toggle(0.0)
    assert k.poll(MAX_LISTEN_S - 0.1) == [] and k.state == "listening"
    assert k.poll(MAX_LISTEN_S) == [] and k.state == "thinking" and mic.events == ["start", "stop"]
    assert wait(k, MAX_LISTEN_S + 1) == [Event("click")]


def test_system_engine_presses_the_dictation_shortcut_to_start_and_to_stop():
    k = Talker(dictation_keys=["control", "option", "d"])
    assert not k.local
    assert k.toggle(0.0) == [Event("hotkey", ("control", "option", "d"))] and k.state == "listening"
    assert k.toggle(3.0) == [Event("hotkey", ("control", "option", "d"))] and k.state == "idle"
    k.toggle(10.0)
    assert k.poll(10.0 + MAX_LISTEN_S) == [Event("hotkey", ("control", "option", "d"))] and k.state == "idle"
