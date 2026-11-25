import subprocess
import sys

import pytest

from airpoint import proc
from airpoint.sounds import _afplay


def no_fork(monkeypatch):
    def forked(*args, **kw):
        raise AssertionError("this process was forked")
    monkeypatch.setattr(subprocess, "_fork_exec", forked)


@pytest.mark.skipif(not subprocess._USE_POSIX_SPAWN, reason="no posix_spawn on this platform")
def test_helper_programs_start_without_forking_this_process(monkeypatch):
    no_fork(monkeypatch)
    assert proc.spawn(["true"]).wait(timeout=5) == 0                       # a bare name is looked up first
    child = proc.spawn([sys.executable, "-c", "import sys; sys.exit(int(sys.stdin.read()))"], stdin=subprocess.PIPE)
    child.stdin.write(b"7")
    child.stdin.close()
    assert child.wait(timeout=10) == 7                                     # a pipe to the program still works


@pytest.mark.skipif(not subprocess._USE_POSIX_SPAWN, reason="no posix_spawn on this platform")
def test_a_sound_and_a_run_to_completion_do_not_fork_either(monkeypatch, tmp_path):
    no_fork(monkeypatch)
    play = _afplay()
    if play is not None:
        play(str(tmp_path / "no-such-sound.aiff"))                         # starts afplay; it fails quietly on its own
    assert proc.run(["true"], check=True, timeout=5).returncode == 0


def test_a_program_that_does_not_exist_is_an_oserror():
    with pytest.raises(OSError):
        proc.spawn(["no-such-program-anywhere"])


@pytest.mark.skipif(not subprocess._USE_POSIX_SPAWN, reason="no posix_spawn on this platform")
def test_airpoints_own_modules_start_from_any_folder_without_a_fork(monkeypatch, tmp_path):
    cmd, env = proc.module("airpoint.speech", "--serve", "--model", "base.en")
    assert cmd[1:] == ["-m", "airpoint.speech", "--serve", "--model", "base.en"] and env["PYTHONPATH"].startswith(proc.ROOT)
    monkeypatch.chdir(tmp_path)                                    # not the project folder
    no_fork(monkeypatch)
    found = proc.spawn([sys.executable, "-c", "import airpoint.proc, sys; sys.exit(0)"], env=env)
    assert found.wait(timeout=20) == 0
    lost = subprocess.Popen([sys.executable, "-c", "import airpoint.proc"], close_fds=False, stderr=subprocess.DEVNULL,
                            env={k: v for k, v in env.items() if k != "PYTHONPATH"})
    assert lost.wait(timeout=20) != 0                              # without being told, it would not find the package
