import time
import wave

import numpy as np
import pytest

from airpoint import speech
from airpoint.fsm import Event
from airpoint.profile import Profile
from airpoint.speech import MAX_LISTEN_S, RATE, Talker, WhisperRecognizer, interpret, make_talker, read_wav


def test_exact_commands_run_and_everything_else_is_typed():
    assert interpret("click") == Event("click")
    assert interpret(" Click. ") == Event("click")
    assert interpret("Scroll DOWN!") == Event("scroll", -5)
    assert interpret("select all") == Event("hotkey", ("command", "a"))
    assert interpret("pause") == Event("pause")
    assert interpret("Click the blue button.") == Event("type", "Click the blue button. ")
    assert interpret("Hello, I will be late.") == Event("type", "Hello, I will be late. ")
    assert interpret("   ") is None and interpret("...") is None


def test_computer_at_the_start_makes_it_a_request_for_the_ai_helper():
    assert interpret("Computer, open the calculator.") == Event("ask", "open the calculator.")
    assert interpret("hey computer reply to Asha that I will come at five") == Event("ask", "reply to Asha that I will come at five")
    assert interpret("Computer.") == Event("type", "Computer. ")                  # no request after it: just a word
    assert interpret("My computer is slow.") == Event("type", "My computer is slow. ")


def voice(seconds=1.0, level=0.2):
    """Something loud enough to be speech: a 200 Hz tone."""
    return (level * np.sin(2 * np.pi * 200 * np.arange(int(seconds * RATE)) / RATE)).astype(np.float32)


class Mic:
    def __init__(self, audio=None, fail=False):
        self.audio, self.fail, self.events = voice() if audio is None else audio, fail, []

    def start(self):
        if self.fail:
            raise OSError("no input device")
        self.events.append("start")

    def stop(self):
        self.events.append("stop")
        return self.audio


class Ears:
    def __init__(self, text="scroll down", fail=False):
        self.text, self.fail, self.heard = text, fail, []

    def transcribe(self, audio):
        self.heard.append(len(audio))
        if self.fail:
            raise RuntimeError("model blew up")
        return self.text


def wait(talker, t=0.0):
    """Poll until the recognition thread has answered. -> its events."""
    deadline = time.time() + 5
    while time.time() < deadline:
        events = talker.poll(t)
        if talker.state == "idle":
            return events
        time.sleep(0.01)
    raise AssertionError("recognition never finished")


def test_local_engine_records_between_two_taps_and_answers_later():
    mic, ears = Mic(), Ears("Scroll down.")
    k = Talker(ears, mic)
    assert k.local and k.toggle(0.0) == "listening" and mic.events == ["start"]
    assert k.poll(1.0) == [] and 18.9 < k.left(1.0) <= 19.0
    assert k.toggle(2.0) == "thinking" and mic.events == ["start", "stop"]
    assert k.toggle(2.1) == "thinking"                            # a tap while thinking changes nothing
    assert wait(k, 2.5) == [Event("scroll", -5)] and k.heard == "Scroll down." and k.problem == ""
    assert ears.heard == [RATE]


def test_free_speech_is_typed():
    k = Talker(Ears("I will be late."), Mic())
    k.toggle(0.0)
    k.toggle(1.0)
    assert wait(k) == [Event("type", "I will be late. ")]


@pytest.mark.parametrize("mic, ears, problem, asked", [
    (Mic(voice(0.1)), Ears("anything"), "too short", False),                  # a slip of the hand
    (Mic(np.zeros(RATE, dtype=np.float32)), Ears("anything"), "allow it in System Settings", False),
    (Mic(voice(level=0.002)), Ears("Thank you."), "heard nothing", False),    # room noise never reaches the model
    (Mic(), Ears(""), "heard nothing", True),
    (Mic(), Ears(fail=True), "could not recognise (RuntimeError: model blew up)", True),
])
def test_what_is_not_speech_does_nothing_and_says_why(mic, ears, problem, asked):
    k = Talker(ears, mic)
    k.toggle(0.0)
    k.toggle(1.0)
    assert wait(k) == [] and problem in k.problem and k.heard == ""
    assert bool(ears.heard) is asked


def test_a_microphone_that_will_not_open_leaves_the_talker_idle():
    k = Talker(Ears(), Mic(fail=True))
    assert k.toggle(0.0) == "idle" and "microphone did not open (OSError)" in k.problem
    k.recorder.fail = False
    assert k.toggle(1.0) == "listening" and k.problem == ""


def test_listening_ends_by_itself():
    mic = Mic()
    k = Talker(Ears("click"), mic)
    k.toggle(0.0)
    assert k.poll(MAX_LISTEN_S - 0.1) == [] and k.state == "listening"
    events = k.poll(MAX_LISTEN_S)                                 # the answer may already be there in the same poll
    assert k.state != "listening" and mic.events == ["start", "stop"]
    assert (events or wait(k, MAX_LISTEN_S + 1)) == [Event("click")]


def test_system_engine_presses_the_dictation_shortcut_to_start_and_to_stop():
    pressed = []
    k = Talker(dictation_keys=["control", "option", "d"], press=pressed.append)
    assert not k.local
    assert k.toggle(0.0) == "listening" and pressed == [("control", "option", "d")]
    assert k.toggle(3.0) == "idle" and len(pressed) == 2
    k.toggle(10.0)
    assert k.poll(10.0 + MAX_LISTEN_S) == [] and k.state == "idle" and len(pressed) == 4      # ended by itself
    k.stop(40.0)
    assert len(pressed) == 4                                                                  # nothing to end


def test_the_mouth_is_speech_while_listening_and_for_a_moment_after():
    k = Talker()
    assert not k.mouth_busy(0.0)
    k.toggle(1.0)
    assert k.mouth_busy(5.0)
    k.toggle(6.0)
    assert k.mouth_busy(6.5) and not k.mouth_busy(7.5)


def test_closing_mid_sentence_lets_go_of_the_microphone():
    mic = Mic()
    k = Talker(Ears(), mic)
    k.toggle(0.0)
    k.close()
    assert mic.events == ["start", "stop"] and k.state == "idle"


def test_the_profile_picks_the_engine_and_a_missing_local_one_falls_back(monkeypatch, capsys):
    assert not make_talker(Profile(user="ana")).local                         # the default is system dictation
    p = Profile(user="ana", speech="local", dictation_keys=["command", "shift", "d"])
    monkeypatch.setattr(WhisperRecognizer, "missing", staticmethod(lambda size: "faster-whisper is not installed"))
    k = make_talker(p)
    assert not k.local and k.keys == ("command", "shift", "d")
    assert "local speech is off: faster-whisper is not installed" in capsys.readouterr().out
    monkeypatch.setattr(WhisperRecognizer, "missing", staticmethod(lambda size: None))
    monkeypatch.setattr(WhisperRecognizer, "warm", lambda self: None)
    k = make_talker(p)
    assert k.local and k.recognizer.size == "base.en"


def test_on_this_laptop_the_local_engine_reports_what_it_lacks():
    why = WhisperRecognizer.missing("base.en")
    assert why is None or "faster-whisper is not installed" in why or "not on this laptop" in why


def _wav(path, audio, rate):
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes((audio * 32767).astype("<i2").tobytes())


def test_wav_files_come_back_as_16_khz_float(tmp_path):
    _wav(tmp_path / "a.wav", voice(0.5), RATE)
    a = read_wav(str(tmp_path / "a.wav"))
    assert a.dtype == np.float32 and len(a) == RATE // 2 and 0.19 < np.abs(a).max() < 0.21
    tone = (0.2 * np.sin(2 * np.pi * 200 * np.arange(48000) / 48000)).astype(np.float32)
    _wav(tmp_path / "b.wav", tone, 48000)
    b = read_wav(str(tmp_path / "b.wav"))
    assert b.dtype == np.float32 and len(b) == RATE and 0.18 < np.abs(b).max() < 0.22


def test_command_line_says_what_a_recording_does(tmp_path, capsys):
    _wav(tmp_path / "a.wav", voice(1.0), RATE)
    speech.main(["--file", str(tmp_path / "a.wav")], recognizer=Ears("Scroll down."))
    out = capsys.readouterr().out
    assert "1.0 s of sound, peak 0.200" in out and "heard: 'Scroll down.'" in out and "kind='scroll', arg=-5" in out
    mic = Mic()
    speech.main(["--listen", "3"], recognizer=Ears("hello"), recorder=mic, sleep=lambda s: None)
    assert mic.events == ["start", "stop"] and "kind='type', arg='hello '" in capsys.readouterr().out


def test_command_line_status_and_setup_never_download_unasked(capsys, monkeypatch):
    speech.main([])
    out = capsys.readouterr().out
    assert "local engine" in out and "control+option+d" in out and "scroll down" in out
    if WhisperRecognizer.missing("base.en") and "not installed" in WhisperRecognizer.missing("base.en"):
        with pytest.raises(SystemExit, match="not installed"):
            speech.main(["--setup"])
    monkeypatch.setattr(speech.importlib.util, "find_spec", lambda name: object())
    with pytest.raises(SystemExit, match="nothing downloaded"):
        speech.main(["--setup"], ask=lambda prompt: "n")
    assert "about 148 MB" in capsys.readouterr().out
