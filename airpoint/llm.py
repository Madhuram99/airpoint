"""The one place AirPoint talks to an AI model: OpenAI's Responses API over plain HTTPS.

The key comes from the environment (OPENAI_API_KEY; several keys may be separated by ';') at the
moment of the call. It is never stored, logged or written anywhere, and anything key-shaped is
scrubbed from error text before it can reach the screen.
"""
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request

API_URL = "https://api.openai.com/v1/responses"
MODEL = "gpt-6.1-sol"
TIMEOUT_S, RETRY_AFTER_S = 60.0, 2.0


class ModelError(Exception):
    """A call that did not produce a usable answer. The message is safe to show."""


def keys(env=None):
    """-> the API keys in OPENAI_API_KEY, in order. Empty when the variable is not set."""
    raw = (os.environ if env is None else env).get("OPENAI_API_KEY", "")
    return [k.strip().strip("'\"") for k in raw.split(";") if k.strip().strip("'\"")]


def scrub(text):
    """Text with anything that looks like a key removed."""
    return re.sub(r"\b(sk|rk|pk)-[A-Za-z0-9_\-\*\.]{4,}", "[key]", str(text))


def image_part(jpeg_bytes):
    return {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode()}


def text_part(text):
    return {"type": "input_text", "text": text}


def _post(url, headers, body, timeout):
    """-> (HTTP status, parsed JSON body). Network trouble raises OSError."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {}


def _text(data):
    return "".join(c.get("text", "") for item in data.get("output", []) if item.get("type") == "message"
                   for c in item.get("content", []) if c.get("type") == "output_text")


def ask(system, parts, schema, *, model=MODEL, api_keys=None, effort="low", max_tokens=1500,
        timeout=TIMEOUT_S, post=_post, sleep=time.sleep):
    """One question, one JSON answer. -> (dict matching `schema`, usage dict).

    `parts` is the user turn: text_part(...) and image_part(...) items. A refused key moves on to the
    next one; a busy or failing server gets one more try. Anything else raises ModelError."""
    api_keys = keys() if api_keys is None else list(api_keys)
    if not api_keys:
        raise ModelError("no API key: set OPENAI_API_KEY in the environment")
    body = {"model": model, "store": False, "max_output_tokens": max_tokens,
            "input": [{"role": "developer", "content": system}, {"role": "user", "content": parts}],
            "text": {"format": {"type": "json_schema", "name": schema["title"], "strict": True, "schema": schema}}}
    if effort:
        body["reasoning"] = {"effort": effort}
    problem, retried = "no answer", False
    queue = list(api_keys)
    while queue:
        key = queue[0]
        try:
            status, data = post(API_URL, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                                body, timeout)
        except OSError as e:
            raise ModelError(scrub(f"could not reach the AI service ({type(e).__name__}: {e})")) from None
        if status == 200:
            if data.get("status") not in (None, "completed"):
                raise ModelError(f"the answer was cut short ({(data.get('incomplete_details') or {}).get('reason', data.get('status'))})")
            try:
                return json.loads(_text(data)), data.get("usage", {})
            except ValueError:
                raise ModelError("the AI answered with something that is not JSON") from None
        message = scrub((data.get("error") or {}).get("message", ""))[:200]
        problem = f"the AI service said {status}" + (f": {message}" if message else "")
        if status in (401, 403):
            queue.pop(0)                                   # this key is refused: try the next one
        elif status in (429, 500, 502, 503, 504) and not retried:
            retried = True
            sleep(RETRY_AFTER_S)
        elif status == 429 and len(queue) > 1:
            queue.pop(0)                                   # still busy on this key: another may have room
        else:
            break
    raise ModelError(problem)
