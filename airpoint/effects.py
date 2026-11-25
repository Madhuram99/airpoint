"""What a sign can do besides drive the mouse: speak a phrase, open an app or a link. macOS commands."""
from .proc import spawn as _quiet


class Effects:
    """`run` is injectable so tests record commands instead of executing them."""

    def __init__(self, voice=None, run=_quiet):
        self.voice, self.run, self.speaking = voice, run, None

    def say(self, text):
        """Speak `text`. A new phrase cuts off one still being spoken."""
        if self.speaking is not None and self.speaking.poll() is None:
            self.speaking.terminate()
        self.speaking = self.run(["say"] + (["-v", self.voice] if self.voice else []) + [text])

    def open(self, target):
        """A link or a path opens as itself; anything else is taken as an application name."""
        is_place = "://" in target or target.startswith(("/", "~", "."))
        self.run(["open", target] if is_place else ["open", "-a", target])
