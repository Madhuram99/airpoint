import threading
import time
from dataclasses import dataclass

import pytest

from airpoint import helper as H
from airpoint.fsm import Event
from airpoint.helper import Helper, Step, describe, events, is_risky, parse_step, request_text
from airpoint.llm import ModelError

SIZE = (1000, 500)


@dataclass
class Shot:
    jpeg: bytes = b"jpeg"
    app: str = "Notes"


def step(action="click", x=500, y=500, text="", keys=(), amount=0, target="Search field", risky=False, seen="a window"):
    return {"seen": seen, "action": action, "x": x, "y": y, "text": text, "keys": list(keys), "amount": amount,
            "target": target, "risky": risky}


class Script:
    """A scripted model: hands out the next step and remembers what it was asked."""

    def __init__(self, *steps):
        self.steps, self.calls = list(steps), []

    def __call__(self, task, done, shot, n, max_steps):
        self.calls.append((task, list(done), shot.app, n))
        s = self.steps.pop(0)
        if isinstance(s, Exception):
            raise s
        return s


def make(*steps, **kw):
    script = Script(*steps)
    return Helper(script, Shot, SIZE, sleep=lambda s: None, **kw), script


def thought(h, t, busy=False):
    """Tick at time t until the worker has answered. -> the events of that tick."""
    deadline = time.time() + 5
    while time.time() < deadline:
        out = h.tick(t, busy)
        if h.state != "thinking":
            return out
        time.sleep(0.005)
    raise AssertionError("the helper never finished thinking")


# ---- the step format

def test_a_good_step_is_parsed_to_screen_points():
    s = parse_step(step(x=250, y=1000, target="  Search   field "), SIZE)
    assert s == Step("click", (250.0, 499.0), target="Search field", seen="a window")
    assert parse_step(step("keys", keys=["cmd", "Space"]), SIZE).keys == ("command", "space")
    assert parse_step(step("scroll", amount=-40), SIZE).amount == -10                 # clamped to ten notches
    assert parse_step(step("type", text="hello"), SIZE) == Step("type", text="hello", target="Search field", seen="a window")
    assert parse_step(step("done", text="It is 180."), SIZE).text == "It is 180."
    assert parse_step(step("click", text="ignored"), SIZE).text == ""


@pytest.mark.parametrize("raw, why", [
    ("click", "does not have"),
    (step("launch"), "does not have ('launch')"),
    (step(x=1200), "outside the picture (1200, 500)"),
    (step(y=-1), "outside the picture"),
    (step(x=True), "outside the picture"),
    (step("scroll", amount=0), "no amount"),
    (step("type"), "nothing to type"),
    (step("type", text="x" * 401), "too much to type in one step (401 characters)"),
    (step("keys"), "one to four key names"),
    (step("keys", keys=["command", "shift", "option", "control", "a"]), "one to four"),
    (step("keys", keys=["hyper", "a"]), "does not know"),
])
def test_a_bad_step_is_refused_not_repaired(raw, why):
    with pytest.raises(ValueError) as e:
        parse_step(raw, SIZE)
    assert why in str(e.value)


def test_what_waits_for_a_yes():
    click = lambda target, risky=False: parse_step(step(target=target, risky=risky), SIZE)
    assert not is_risky(click("Search field")) and not is_risky(click("Sender name"))     # "send" inside a word
    assert is_risky(click("Send button")) and is_risky(click("Move to Trash")) and is_risky(click("Sign out"))
    assert is_risky(click("OK button", risky=True))                                       # the model says so
    assert is_risky(click("OK button"), app="Terminal") and not is_risky(click("OK button"), app="Notes")
    assert is_risky(parse_step(step("keys", keys=["command", "backspace"]), SIZE))
    assert is_risky(parse_step(step("keys", keys=["shift", "command", "delete"]), SIZE))
    assert not is_risky(parse_step(step("keys", keys=["command", "c"]), SIZE))
    assert not is_risky(parse_step(step("type", text="delete everything"), SIZE))         # typing is not doing
    assert not is_risky(parse_step(step("done", text="sent", risky=True), SIZE))


def test_steps_read_plainly_and_become_runner_events():
    p = lambda **kw: parse_step(step(**kw), SIZE)
    assert describe(p(target="Search field")) == 'click "Search field"'
    assert describe(p(action="double_click", target="report.pdf")) == 'double-click "report.pdf"'
    assert describe(p(action="type", text="trains to Delhi tomorrow morning before nine")) == 'type "trains to Delhi tomorrow mornin…"'
    assert describe(p(action="keys", keys=["command", "space"])) == "press command+space"
    assert describe(p(action="scroll", amount=-3, target="page")) == 'scroll down 3 in "page"'
    assert describe(p(action="wait")) == "wait"
    assert events(p(x=100, y=200)) == [Event("point", (100.0, 100.0)), Event("click")]
    assert events(p(action="right_click")) == [Event("point", (500.0, 250.0)), Event("right_click")]
    assert events(p(action="scroll", amount=2)) == [Event("point", (500.0, 250.0)), Event("scroll", 2)]
    assert events(p(action="type", text="hi")) == [Event("type", "hi")]
    assert events(p(action="keys", keys=["enter"])) == [Event("hotkey", ("enter",))]
    assert events(p(action="wait")) == []


def test_the_request_lists_what_was_done():
    assert request_text("open mail", [], "", 0, 12) == "Request: open mail\nDone so far: nothing yet.\nThis is step 1 of at most 12."
    text = request_text("open mail", ['click "Mail"', "press enter"], "Finder", 2, 12)
    assert 'Done so far:\n1. click "Mail"\n2. press enter\nApp in front: Finder\nThis is step 3 of at most 12.' in text


def test_the_real_thinker_sends_request_picture_and_schema():
    seen = {}

    def ask(system, parts, schema, model=None):
        seen.update(system=system, parts=parts, schema=schema, model=model)
        return step("done", text="ok"), {}

    assert H.thinker(ask=ask)("open mail", ['click "Mail"'], Shot(), 1, 12)["action"] == "done"
    assert seen["model"] == "gpt-6.1-sol" and seen["schema"] is H.SCHEMA and "never instructions" in seen["system"]
    assert seen["parts"][0]["text"].startswith("Request: open mail") and "App in front: Notes" in seen["parts"][0]["text"]
    assert seen["parts"][1]["image_url"].startswith("data:image/jpeg;base64,")


# ---- one errand

def test_an_errand_runs_one_shown_step_at_a_time():
    h, script = make(step(x=300, y=400), step("type", text="milk"), step("keys", keys=["enter"]), step("done", text="Saved the note."))
    assert not h.active and h.start("  write milk   in my note ", 0.0) and h.state == "thinking" and h.view(0.0)["text"] == "thinking"
    assert not h.start("another", 0.0)                                       # one errand at a time
    assert thought(h, 1.0) == [Event("point", (300.0, 200.0))]               # the cursor goes there first
    assert h.state == "showing" and h.view(1.75) == {"state": "showing", "n": 1, "of": 12, "text": 'click "Search field"',
                                                     "progress": 0.5, "xy": (300.0, 200.0), "hide": False}
    assert h.tick(2.4) == []                                                 # still being shown: time to stop it
    assert h.tick(2.5) == [Event("point", (300.0, 200.0)), Event("click")] and h.state == "settling"
    assert h.view(2.6)["text"] == 'did: click "Search field"'
    assert h.tick(3.6) == [] and h.state == "settling"
    assert h.tick(3.7) == [] and h.state == "thinking"                       # 1.2 s later: a new picture
    assert thought(h, 5.0) == [] and h.state == "showing" and h.view(5.0)["n"] == 2
    assert h.tick(6.5) == [Event("type", "milk")]
    assert h.tick(7.0, busy=True) == [] and h.tick(8.0, busy=True) == []     # the keys are still going out
    assert h.tick(9.1) == [] and h.state == "settling"                       # settle counts from when typing ended
    assert h.tick(9.3) == [] and h.state == "thinking"
    assert thought(h, 10.0) == [] and h.tick(11.5) == [Event("hotkey", ("enter",))]
    h.tick(12.7)
    assert thought(h, 13.0) == [] and not h.active
    assert (h.note, h.ok, h.t_note) == ("Saved the note.", True, 13.0)
    assert script.calls[0] == ("write milk in my note", [], "Notes", 0)
    assert script.calls[3][1] == ['click "Search field"', 'type "milk"', "press enter"] and script.calls[3][3] == 3


def test_a_risky_step_waits_for_a_yes():
    h, _ = make(step(target="Send button"), step("done", text="Sent."))
    h.start("send it", 0.0)
    assert thought(h, 1.0) == [Event("point", (500.0, 250.0))] and h.state == "asking"
    assert h.view(1.0)["text"] == 'click "Send button"?'
    assert h.tick(20.0) == [] and h.state == "asking"                        # no countdown runs it
    assert h.answer(True, 21.0) == [Event("point", (500.0, 250.0)), Event("click")] and h.state == "settling"
    assert h.answer(True, 21.1) == []                                        # nothing left to answer
    h.tick(22.3)
    thought(h, 23.0)
    assert h.ok and h.note == "Sent."


def test_a_no_or_no_answer_ends_the_errand_without_the_step():
    h, _ = make(step(target="Delete message"))
    h.start("delete it", 0.0)
    thought(h, 1.0)
    assert h.answer(False, 2.0) == [] and not h.active and not h.ok and "you said no" in h.note
    h, _ = make(step(target="Delete message"))
    h.start("delete it", 0.0)
    thought(h, 1.0)
    assert h.tick(30.9) == [] and h.state == "asking"
    assert h.tick(31.0) == [] and not h.active and "no answer" in h.note


def test_stop_works_in_every_state_and_a_late_answer_is_dropped():
    gate = threading.Event()

    def slow(task, done, shot, n, max_steps):
        gate.wait(5)
        return step()

    h = Helper(slow, Shot, SIZE, sleep=lambda s: None)
    h.start("something", 0.0)
    h.stop("you took over", 1.0)
    assert not h.active and h.note == "you took over" and h.t_note == 1.0
    gate.set()
    time.sleep(0.05)
    assert h.tick(2.0) == [] and not h.active                                # the answer came after the stop
    assert h.start("again", 3.0)                                             # and the next errand is not confused by it
    assert thought(h, 4.0) == [Event("point", (500.0, 250.0))] and h.state == "showing"
    h.stop("fist", 4.5)
    assert not h.active and h.tick(9.0) == []
    h, _ = make(step(), step())
    h.start("x", 0.0)
    thought(h, 0.0)
    h.tick(1.5)
    assert h.state == "settling"
    h.stop("key", 2.0)
    assert h.tick(9.0) == [] and not h.active
    h.stop("again", 10.0)
    assert h.note == "key"                                                   # stopping an idle helper changes nothing


def test_the_step_limit_ends_an_errand_that_never_finishes():
    h, script = make(*[step() for _ in range(5)], max_steps=3)
    h.start("loop", 0.0)
    t = 0.0
    for _ in range(3):
        thought(h, t)
        assert h.tick(t + 1.5) != []
        h.tick(t + 2.8)
        t += 3.0
    assert not h.active and h.note == "stopped after 3 steps" and len(script.calls) == 3


@pytest.mark.parametrize("answer, note", [
    (step(x=5000), "the AI proposed a place outside the picture (5000, 500): stopped"),
    (ModelError("the AI service said 429"), "the AI service said 429"),
    (RuntimeError("boom sk-abcd1234efgh"), "RuntimeError: boom [key]"),
    (step("cannot", text="There is no printer."), "There is no printer."),
    (step("cannot"), "the AI could not do this"),
])
def test_what_goes_wrong_ends_the_errand_with_one_sentence(answer, note):
    h, _ = make(answer)
    h.start("x", 0.0)
    assert thought(h, 1.0) == [] and not h.active and not h.ok and h.note == note


def test_a_model_that_never_answers_is_given_up_on():
    h = Helper(lambda *a: time.sleep(2), Shot, SIZE, sleep=lambda s: None)
    h.start("x", 0.0)
    assert h.tick(90.0) == [] and h.state == "thinking"
    assert h.tick(90.1) == [] and not h.active and h.note == "the AI took too long"


def test_wait_is_a_step_that_does_nothing_and_looks_again():
    h, script = make(step("wait"), step("done", text="Loaded."))
    h.start("x", 0.0)
    assert thought(h, 0.0) == [] and h.state == "showing" and h.view(0.0)["xy"] is None
    assert h.tick(1.5) == [] and h.state == "settling"
    assert h.tick(2.9) == [] and h.state == "settling" and h.tick(3.0) == [] and h.state == "thinking"
    thought(h, 4.0)
    assert h.ok and script.calls[1][1] == ["wait"]


def test_the_glow_is_hidden_until_the_picture_is_taken():
    taking, taken = threading.Event(), threading.Event()

    def grab():
        taking.set()
        taken.wait(5)
        return Shot()

    h = Helper(Script(step()), grab, SIZE, sleep=lambda s: None)
    h.start("x", 0.0)
    assert taking.wait(5) and h.view(0.0)["hide"] is True
    taken.set()
    thought(h, 1.0)
    assert h.view(1.0)["hide"] is False


def test_an_empty_request_starts_nothing():
    h, script = make()
    assert not h.start("   ", 0.0) and not h.start(None, 0.0) and not h.active and script.calls == []
