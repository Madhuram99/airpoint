import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from airpoint.lab import server, store
from tests.test_lab_stats import hand_sequence


@pytest.fixture
def lab(tmp_path):
    (tmp_path / "lab").mkdir()
    (tmp_path / "lab" / "index.html").write_text("<h1>lab home</h1>")
    (tmp_path / "lab" / "lab.js").write_text("// js")
    (tmp_path / "index.html").write_text("<h1>dashboard</h1>")
    (tmp_path / "secret.txt").write_text("not served")
    httpd = server.make_server(str(tmp_path), port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield tmp_path, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.headers.get("Content-Type"), r.read().decode()


def post(url, obj, raw=None):
    req = urllib.request.Request(url, data=raw if raw is not None else json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def status(url):
    try:
        return get(url)[0]
    except urllib.error.HTTPError as e:
        return e.code


def test_serves_the_lab_pages_and_the_dashboard_and_nothing_else(lab):
    _, base = lab
    assert get(base + "/lab/index.html")[2] == "<h1>lab home</h1>"
    assert get(base + "/")[2] == "<h1>lab home</h1>"                      # / redirects to the lab home
    assert get(base + "/lab/lab.js")[1].startswith("text/javascript")
    assert get(base + "/index.html")[2] == "<h1>dashboard</h1>"
    assert status(base + "/secret.txt") == 404
    assert status(base + "/lab/../secret.txt") == 404
    assert status(base + "/lab/%2e%2e/secret.txt") == 404
    assert status(base + "/lab/missing.html") == 404


def test_state_is_not_running_without_a_fresh_live_file(lab):
    root, base = lab
    assert json.loads(get(base + "/api/state")[2]) == {"running": False, "state": None}
    (root / "results").mkdir()
    live = {"ts": time.time(), "mouse": True, "dwell": False, "filter": "full",
            "sim": {"on": False, "hz": 6.0, "amp": 0.2}, "user": "ana"}
    (root / "results" / "live.json").write_text(json.dumps(live))
    got = json.loads(get(base + "/api/state")[2])
    assert got["running"] and got["state"]["user"] == "ana"
    live["ts"] = time.time() - 10                                            # AirPoint quit a while ago
    (root / "results" / "live.json").write_text(json.dumps(live))
    assert json.loads(get(base + "/api/state")[2])["running"] is False


def test_condition_labels_name_what_was_actually_driving_the_cursor():
    live = {"mouse": True, "dwell": False, "filter": "full", "sim": {"on": False, "hz": 6.0, "amp": 0.2}}
    assert server.condition_label("mouse", live) == "mouse"
    assert server.condition_label("trackpad", None) == "trackpad"
    assert server.condition_label("airpoint", live) == "airpoint sign-click"
    assert server.condition_label("airpoint", dict(live, dwell=True)) == "airpoint dwell-click"
    assert server.condition_label("airpoint", dict(live, filter="off")) == "airpoint sign-click, filter off"
    shaky = dict(live, sim={"on": True, "hz": 6.0, "amp": 0.2}, filter="smooth")
    assert server.condition_label("airpoint", shaky) == "airpoint sign-click, simulated tremor 0.20 palm @ 6.0 Hz, filter smooth"
    assert server.condition_label("airpoint", dict(live, wheel=True, dwell=True)) == "airpoint wheel"
    assert server.condition_label("airpoint", None) == "airpoint (settings unknown)"
    assert server.condition_label("airpoint", dict(live, mouse=False)) == "airpoint (settings unknown)"


def test_a_test_block_is_saved_scored_and_summarised(lab):
    root, base = lab
    live = {"mouse": True, "dwell": True, "filter": "full", "sim": {"on": False, "hz": 6.0, "amp": 0.2}}
    block = {"kind": "test", "participant": "P1", "device": "airpoint", "hand": "right",
             "airpoint_start": live, "airpoint_end": live, "sequences": [hand_sequence()]}
    code, out = post(base + "/api/blocks", block)
    assert code == 200 and out["summary"]["trials"] == 4 and out["summary"]["tp"] > 0
    saved = store.read(str(root / "results" / "trials.jsonl"))
    assert len(saved) == 1 and saved[0]["id"] == out["id"] and saved[0]["condition"] == "airpoint dwell-click"
    assert "saved_at" in saved[0]

    post(base + "/api/blocks", {"kind": "tutorial", "participant": "P1", "device": "mouse",
                                "steps": [{"name": "move", "seconds": 3.5}]})
    summary = json.loads(get(base + "/api/summary")[2])
    assert [c["condition"] for c in summary["conditions"]] == ["airpoint dwell-click"]
    assert [b["kind"] for b in summary["blocks"]] == ["tutorial", "test"]     # newest first
    assert summary["blocks"][1]["tp"] == pytest.approx(out["summary"]["tp"])


def test_bad_blocks_are_refused_and_nothing_is_written(lab):
    root, base = lab
    assert post(base + "/api/blocks", {"kind": "test", "participant": "P1", "device": "mouse"})[0] == 400      # no sequences
    assert post(base + "/api/blocks", {"kind": "nope", "participant": "P1", "device": "mouse"})[0] == 400
    assert post(base + "/api/blocks", {"kind": "tutorial", "participant": "x" * 80, "device": "mouse", "steps": []})[0] == 400
    assert post(base + "/api/blocks", None, raw=b"{not json")[0] == 400
    assert post(base + "/api/blocks", None, raw=b"[" + b"1," * (server.MAX_BODY // 2) + b"1]")[0] == 413
    assert not (root / "results" / "trials.jsonl").exists()


def test_store_skips_a_damaged_line(tmp_path):
    path = str(tmp_path / "results" / "trials.jsonl")
    store.append(path, {"a": 1})
    with open(path, "a") as f:
        f.write("{broken\n")
    store.append(path, {"a": 2})
    assert [r["a"] for r in store.read(path)] == [1, 2]
    assert store.read(str(tmp_path / "missing.jsonl")) == []
