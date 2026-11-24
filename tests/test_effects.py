from airpoint.effects import Effects


class Proc:
    def __init__(self, cmd):
        self.cmd, self.alive, self.terminated = cmd, True, False

    def poll(self):
        return None if self.alive else 0

    def terminate(self):
        self.terminated, self.alive = True, False


def make(voice=None):
    procs = []

    def run(cmd):
        procs.append(Proc(cmd))
        return procs[-1]
    return Effects(voice=voice, run=run), procs


def test_say_uses_the_profile_voice_and_a_new_phrase_cuts_off_the_last():
    fx, procs = make(voice="Rishi")
    fx.say("I need water")
    fx.say("thank you")
    assert [p.cmd for p in procs] == [["say", "-v", "Rishi", "I need water"], ["say", "-v", "Rishi", "thank you"]]
    assert procs[0].terminated and not procs[1].terminated
    fx2, procs2 = make()
    fx2.say("hello")
    assert procs2[0].cmd == ["say", "hello"]


def test_open_tells_links_and_paths_from_application_names():
    fx, procs = make()
    for target in ("https://example.org", "/Applications/Notes.app", "~/notes.txt", "Notes"):
        fx.open(target)
    assert [p.cmd for p in procs] == [["open", "https://example.org"], ["open", "/Applications/Notes.app"],
                                      ["open", "~/notes.txt"], ["open", "-a", "Notes"]]
