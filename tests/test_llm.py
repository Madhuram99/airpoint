import json

import pytest

from airpoint import llm
from airpoint.llm import ModelError, ask, image_part, keys, scrub, text_part

SCHEMA = {"title": "answer", "type": "object", "properties": {"word": {"type": "string"}},
          "required": ["word"], "additionalProperties": False}


def ok(obj):
    return 200, {"status": "completed", "usage": {"input_tokens": 7, "output_tokens": 3},
                 "output": [{"type": "reasoning"}, {"type": "message", "content": [{"type": "output_text", "text": json.dumps(obj)}]}]}


class Server:
    """Stands in for the API: answers from a script and remembers what it was sent."""

    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def __call__(self, url, headers, body, timeout):
        self.seen.append((url, headers, body, timeout))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_keys_come_from_the_environment_and_may_be_several():
    assert keys({}) == [] and keys({"OPENAI_API_KEY": ""}) == []
    assert keys({"OPENAI_API_KEY": "sk-aaaa1111"}) == ["sk-aaaa1111"]
    assert keys({"OPENAI_API_KEY": '"sk-aaaa1111; sk-bbbb2222;"'}) == ["sk-aaaa1111", "sk-bbbb2222"]


def test_anything_key_shaped_is_scrubbed():
    assert scrub("Incorrect API key provided: sk-proj-ab12****************9xYz.") == "Incorrect API key provided: [key]"
    assert scrub("no secrets here") == "no secrets here"


def test_one_question_is_one_strict_json_request():
    server = Server(ok({"word": "yellow"}))
    answer, usage = ask("Answer with one word.", [text_part("banana?"), image_part(b"\xff\xd8jpeg")], SCHEMA,
                        api_keys=["sk-test0001"], post=server)
    assert answer == {"word": "yellow"} and usage["input_tokens"] == 7
    url, headers, body, timeout = server.seen[0]
    assert url == "https://api.openai.com/v1/responses" and headers["Authorization"] == "Bearer sk-test0001"
    assert body["model"] == "gpt-6.1-sol" and body["store"] is False and body["reasoning"] == {"effort": "low"}
    assert body["input"][0] == {"role": "developer", "content": "Answer with one word."}
    assert body["input"][1]["content"][0] == {"type": "input_text", "text": "banana?"}
    assert body["input"][1]["content"][1]["image_url"].startswith("data:image/jpeg;base64,/9hq")
    assert body["text"]["format"] == {"type": "json_schema", "name": "answer", "strict": True, "schema": SCHEMA}


def test_a_refused_key_moves_on_to_the_next_and_a_busy_server_gets_one_more_try():
    server = Server((401, {"error": {"message": "Incorrect API key provided: sk-test0001"}}), ok({"word": "a"}))
    assert ask("s", [text_part("q")], SCHEMA, api_keys=["sk-test0001", "sk-test0002"], post=server)[0] == {"word": "a"}
    assert [h["Authorization"] for _, h, _, _ in server.seen] == ["Bearer sk-test0001", "Bearer sk-test0002"]

    naps = []
    server = Server((503, {}), ok({"word": "b"}))
    assert ask("s", [text_part("q")], SCHEMA, api_keys=["sk-test0001"], post=server, sleep=naps.append)[0] == {"word": "b"}
    assert naps == [llm.RETRY_AFTER_S] and len(server.seen) == 2


@pytest.mark.parametrize("replies, said", [
    ([(401, {"error": {"message": "Incorrect API key provided: sk-test0001"}})], "said 401: Incorrect API key provided: [key]"),
    ([(404, {"error": {"message": "The model `nope` does not exist"}})], "said 404: The model `nope` does not exist"),
    ([(500, {}), (500, {})], "said 500"),
    ([(429, {}), (429, {}), (429, {})], "said 429"),
    ([OSError("timed out")], "could not reach the AI service (OSError: timed out)"),
    ([(200, {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": []})], "cut short (max_output_tokens)"),
    ([(200, {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "sorry"}]}]})], "not JSON"),
])
def test_failures_become_one_safe_sentence(replies, said):
    with pytest.raises(ModelError) as e:
        ask("s", [text_part("q")], SCHEMA, api_keys=["sk-test0001", "sk-test0002"][:1 if len(replies) < 3 else 2],
            post=Server(*replies), sleep=lambda s: None)
    assert said in str(e.value) and "sk-test" not in str(e.value)


def test_no_key_is_said_plainly_and_nothing_is_sent():
    server = Server()
    with pytest.raises(ModelError, match="set OPENAI_API_KEY"):
        ask("s", [text_part("q")], SCHEMA, api_keys=[], post=server)
    assert server.seen == []
