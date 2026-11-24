"""The lab's local web server (standard library only): static pages plus a small JSON API."""
import json
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

from . import store
from .stats import aggregate, block_summary

PORT = 8765
MAX_BODY = 8 * 1024 * 1024
STALE_S = 3.0                       # a live file older than this means AirPoint is not running
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png"}
ROOT_FILES = ("index.html", "results.js")   # the dashboard, served beside the lab


def read_state(root):
    """AirPoint's live settings, or None when it is not running."""
    try:
        with open(os.path.join(root, "results", "live.json")) as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return state if time.time() - state.get("ts", 0) <= STALE_S else None


def condition_label(device, state):
    """The name a block is grouped under. For AirPoint it comes from what AirPoint reported,
    not from what the participant remembers pressing."""
    if device != "airpoint":
        return device
    if not state or not state.get("mouse"):
        return "airpoint (settings unknown)"
    label = "airpoint dwell-click" if state.get("dwell") else "airpoint sign-click"
    sim = state.get("sim") or {}
    if sim.get("on"):
        return f"{label}, simulated tremor {sim['amp']:.2f} palm @ {sim['hz']:.1f} Hz, filter {state.get('filter')}"
    return label if state.get("filter") == "full" else f"{label}, filter {state.get('filter')}"


def _check(block):
    """-> error text, or None if the block is fit to save."""
    if not isinstance(block, dict) or block.get("kind") not in ("test", "tutorial"):
        return "kind must be test or tutorial"
    who = block.get("participant")
    if not isinstance(who, str) or not 1 <= len(who) <= 40:
        return "participant must be 1-40 characters"
    if block.get("device") not in ("airpoint", "mouse", "trackpad"):
        return "device must be airpoint, mouse or trackpad"
    if block["kind"] == "test":
        seqs = block.get("sequences")
        if not isinstance(seqs, list) or not seqs or not all(isinstance(s, dict) and s.get("trials") for s in seqs):
            return "a test block needs sequences with trials"
    elif not isinstance(block.get("steps"), list):
        return "a tutorial block needs steps"
    return None


class Handler(BaseHTTPRequestHandler):
    root = "."

    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _file(self, base, rel):
        base = os.path.realpath(base)
        full = os.path.realpath(os.path.join(base, rel))
        ext = os.path.splitext(full)[1]
        if not full.startswith(base + os.sep) or ext not in TYPES or not os.path.isfile(full):
            return self._send(404, {"error": "not found"})
        with open(full, "rb") as f:
            self._send(200, f.read(), TYPES[ext])

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        trials = os.path.join(self.root, "results", "trials.jsonl")
        if path == "/":
            self.send_response(302)
            self.send_header("Location", "/lab/index.html")
            self.end_headers()
        elif path == "/api/state":
            state = read_state(self.root)
            self._send(200, {"running": state is not None, "state": state})
        elif path == "/api/summary":
            blocks = store.read(trials)
            recent = [dict(block_summary(b), id=b.get("id"), saved_at=b.get("saved_at"),
                           participant=b.get("participant"), condition=b.get("condition")) for b in blocks[::-1][:30]]
            for r in recent:
                r.pop("sequences", None)
            self._send(200, {"conditions": aggregate(blocks), "blocks": recent})
        elif path.startswith("/lab/"):
            self._file(os.path.join(self.root, "lab"), path[len("/lab/"):])
        elif path.lstrip("/") in ROOT_FILES:
            self._file(self.root, path.lstrip("/"))
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/blocks":
            return self._send(404, {"error": "not found"})
        size = int(self.headers.get("Content-Length") or 0)
        if size > MAX_BODY:
            left = min(size, 8 * MAX_BODY)      # drain what was sent, so the refusal reaches the page
            while left > 0:
                chunk = self.rfile.read(min(65536, left))
                if not chunk:
                    break
                left -= len(chunk)
            self.close_connection = True
            return self._send(413, {"error": "block too large"})
        try:
            block = json.loads(self.rfile.read(size))
        except json.JSONDecodeError:
            return self._send(400, {"error": "not JSON"})
        problem = _check(block)
        if problem:
            return self._send(400, {"error": problem})
        block["id"] = uuid.uuid4().hex[:12]
        block["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        block["condition"] = condition_label(block["device"], block.get("airpoint_start"))
        try:
            summary = block_summary(block)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            return self._send(400, {"error": f"block could not be scored: {type(e).__name__}"})
        store.append(os.path.join(self.root, "results", "trials.jsonl"), block)
        self._send(200, {"id": block["id"], "condition": block["condition"], "summary": summary})


def make_server(root, port=PORT):
    """Bound to this machine only. port=0 picks a free one (tests)."""
    handler = type("LabHandler", (Handler,), {"root": root})
    return ThreadingHTTPServer(("127.0.0.1", port), handler)
