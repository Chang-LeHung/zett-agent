"""A local Gemini ``streamGenerateContent`` server for end-to-end provider tests.

It speaks the Generative Language REST protocol closely enough that the real
``google-genai`` SDK talks to it through ``GoogleProvider(base_url=...)``, and it
enforces the rules the live service enforces and that a mapping bug would break:

* the API key header is required;
* roles are ``user``/``model``, every turn has parts, and the first turn is the user's;
* a model turn with N function calls is answered by one turn holding exactly N
  function responses with the same names (the live service answers 400 otherwise);
* on a ``gemini-3`` model every function-call turn replayed to the server keeps
  the ``thoughtSignature`` it was given.

The model behaviour is scripted by the last user text: ``weather in A and B``
makes the "model" call ``get_weather`` for both cities at once, ``image`` makes
it describe the image it was sent, and anything else is echoed.
"""

from __future__ import annotations

import base64
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

SIGNATURE = base64.b64encode(b"signature-from-the-server").decode()
API_KEY = "test-key"

_ROUTE = re.compile(r"^/v1beta/models/(?P<model>[^/:]+):(?P<method>streamGenerateContent|generateContent)$")


def _error(status: int, message: str, code: str) -> tuple[int, dict[str, Any]]:
    return status, {"error": {"code": status, "message": message, "status": code}}


def validate(model: str, body: dict[str, Any]) -> str | None:
    """Return the message of the 400 the live service would send, or None when valid."""
    contents = body.get("contents")
    if not isinstance(contents, list) or not contents:
        return "contents is not specified"
    if contents[0].get("role") != "user":
        return "Please ensure that multiturn requests start with a user turn."
    previous_calls: list[str] = []
    for index, turn in enumerate(contents):
        if turn.get("role") not in ("user", "model"):
            return f"Invalid role at contents[{index}]: {turn.get('role')!r}"
        parts = turn.get("parts")
        if not parts:
            return f"contents[{index}].parts must not be empty"
        responses = [part["functionResponse"]["name"] for part in parts if "functionResponse" in part]
        if previous_calls:
            if turn["role"] != "user" or sorted(responses) != sorted(previous_calls):
                return (
                    "Please ensure that the number of function response parts is equal to the number of "
                    "function call parts of the function call turn."
                )
        elif responses:
            return "A function response must follow a turn that made a function call."
        previous_calls = [part["functionCall"]["name"] for part in parts if "functionCall" in part]
        if model.startswith("gemini-3") and turn["role"] == "model" and previous_calls:
            first = next(part for part in parts if "functionCall" in part)
            if first.get("thoughtSignature") != SIGNATURE:
                return f"Function call is missing a thought_signature in functionCall parts at contents[{index}]."
    if previous_calls:
        return "The last turn must not be a function call."
    return None


def _decode_bytes(data: str) -> bytes:
    """Decode the base64 the SDK sends, which may be URL-safe and unpadded."""
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def _user_text(contents: list[dict[str, Any]]) -> str:
    for turn in reversed(contents):
        texts = [part["text"] for part in turn["parts"] if "text" in part]
        if turn["role"] == "user" and texts:
            return " ".join(texts)
    return ""


def script(model: str, body: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the streamed chunks for one request, as Gemini candidate dictionaries."""
    contents = body["contents"]
    last = contents[-1]
    responses = [part["functionResponse"] for part in last["parts"] if "functionResponse" in part]
    usage = {"promptTokenCount": 11, "candidatesTokenCount": 5, "totalTokenCount": 16}
    if responses:
        summary = "; ".join(f"{item['name']}={json.dumps(item['response'], sort_keys=True)}" for item in responses)
        return [
            _chunk([{"text": "Comparing the results. ", "thought": True}]),
            _chunk([{"text": "Results: "}]),
            _chunk([{"text": summary}], finish="STOP", usage=usage),
        ]
    text = _user_text(contents)
    # Safety behaviour, selected by a phrase in the prompt, as the live service decides it
    # itself: a blocked prompt has no candidate at all, and a refusal ends a candidate.
    if "blocked prompt" in text:
        return [{"promptFeedback": {"blockReason": "PROHIBITED_CONTENT", "blockReasonMessage": "Prompt blocked"}}]
    if "refused after a call" in text:
        call = {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}
        return [_chunk([call], finish="SAFETY", usage=usage)]
    if "refused halfway" in text:
        chunk = _chunk([{"text": "Here is the start of"}], finish="SAFETY", usage=usage)
        chunk["candidates"][0]["finishMessage"] = "The response was blocked."
        return [chunk]
    if "too long" in text:
        return [_chunk([{"text": "cut"}], finish="MAX_TOKENS", usage=usage)]
    match = re.search(r"weather in (\w+) and (\w+)", text)
    if match and body.get("tools"):
        names = {declaration["name"] for tool in body["tools"] for declaration in tool.get("functionDeclarations", ())}
        assert "get_weather" in names, names
        first = {"functionCall": {"name": "get_weather", "args": {"city": match.group(1)}}}
        if model.startswith("gemini-3"):
            first["thoughtSignature"] = SIGNATURE
        second = {"functionCall": {"name": "get_weather", "args": {"city": match.group(2)}}}
        return [_chunk([first]), _chunk([second], finish="STOP", usage=usage)]
    if any("inlineData" in part for turn in contents for part in turn["parts"]):
        sizes = [
            len(_decode_bytes(part["inlineData"]["data"]))
            for turn in contents
            for part in turn["parts"]
            if "inlineData" in part
        ]
        return [_chunk([{"text": f"I can see {len(sizes)} image(s) of {sizes[0]} bytes."}], finish="STOP", usage=usage)]
    return [_chunk([{"text": "Echo: "}]), _chunk([{"text": text}], finish="STOP", usage=usage)]


def _chunk(
    parts: list[dict[str, Any]], *, finish: str | None = None, usage: dict[str, int] | None = None
) -> dict[str, Any]:
    candidate: dict[str, Any] = {"content": {"role": "model", "parts": parts}, "index": 0}
    if finish:
        candidate["finishReason"] = finish
    chunk: dict[str, Any] = {"candidates": [candidate], "modelVersion": "fake-gemini"}
    if usage:
        chunk["usageMetadata"] = usage
    return chunk


class FakeGemini:
    """Run the server on a free local port; use as a context manager."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        #: Statuses to answer with before behaving, one per request (for retry tests).
        self.failures: list[int] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass

            def do_POST(self) -> None:  # noqa: N802
                try:
                    self._handle()
                except Exception as error:  # a scripting bug must fail the test with its cause
                    self._send(*_error(500, f"fake server bug: {type(error).__name__}: {error}", "INTERNAL"))

            def _handle(self) -> None:
                parsed = urlparse(self.path)
                route = _ROUTE.match(parsed.path)
                length = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append(body)
                outer.headers.append({key.lower(): value for key, value in self.headers.items()})
                if route is None:
                    return self._send(*_error(404, f"no route for {parsed.path}", "NOT_FOUND"))
                if self.headers.get("x-goog-api-key") != API_KEY:
                    return self._send(
                        *_error(403, "API key not valid. Please pass a valid API key.", "PERMISSION_DENIED")
                    )
                if outer.failures:
                    status = outer.failures.pop(0)
                    return self._send(*_error(status, "transient failure", "UNAVAILABLE"))
                model = route["model"]
                problem = validate(model, body)
                if problem:
                    return self._send(*_error(400, problem, "INVALID_ARGUMENT"))
                chunks = script(model, body)
                if parse_qs(parsed.query).get("alt") != ["sse"]:
                    return self._send(200, chunks)
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                for chunk in chunks:
                    self.wfile.write(f"data: {json.dumps(chunk)}\r\n\r\n".encode())
                    self.wfile.flush()

            def _send(self, status: int, payload: Any) -> None:
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self) -> FakeGemini:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
