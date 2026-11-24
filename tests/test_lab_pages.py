"""The lab pages are plain files; these checks keep them loadable."""
import os
import shutil
import subprocess

import pytest

LAB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lab")
SCRIPTS = ("lab.js", "test.js", "tutorial.js")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("name", SCRIPTS)
def test_script_parses(name):
    r = subprocess.run(["node", "--check", os.path.join(LAB, name)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_pages_load_their_scripts_and_nothing_from_the_network():
    for page, scripts in (("index.html", ["lab.js"]), ("test.html", ["lab.js", "test.js"]),
                          ("tutorial.html", ["lab.js", "tutorial.js"])):
        html = open(os.path.join(LAB, page)).read()
        for s in scripts:
            assert f'<script src="{s}"></script>' in html
        assert "http://" not in html and "https://" not in html      # works offline


def test_the_test_page_uses_the_levels_the_spec_names():
    js = open(os.path.join(LAB, "test.js")).read()
    assert "const N = 9;" in js and "[[320, 96], [480, 64], [640, 48]]" in js
