"""Starting helper programs (a sound, `say`, the screen capture, the glow, the speech process).

The frame loop shares its process with tracker, audio and recognizer threads. `fork` copies a process
while those threads may hold locks; `posix_spawn` starts a program without copying anything. Python
uses it when the program is given by its full path and `close_fds` is off; what this process has open
is close-on-exec already, so nothing leaks into the program.
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # the folder that holds the airpoint package


def _args(cmd, kw):
    kw.setdefault("stdout", subprocess.DEVNULL)
    kw.setdefault("stderr", subprocess.DEVNULL)
    kw["close_fds"] = False
    return [shutil.which(cmd[0]) or cmd[0], *cmd[1:]]


def spawn(cmd, **kw):
    """Start `cmd` and return at once. -> the Popen. Quiet unless told otherwise."""
    return subprocess.Popen(_args(cmd, kw), **kw)


def run(cmd, **kw):
    """Start `cmd` and wait for it, as subprocess.run does."""
    return subprocess.run(_args(cmd, kw), **kw)


def module(name, *args):
    """-> (cmd, env) that run one of AirPoint's own modules in a new Python, whatever folder AirPoint was
    started from: the new process is told where the package is instead of being started in its folder
    (a working directory would mean a fork)."""
    path = os.environ.get("PYTHONPATH")
    env = dict(os.environ, PYTHONPATH=ROOT + (os.pathsep + path if path else ""))
    return [sys.executable, "-m", name, *args], env
