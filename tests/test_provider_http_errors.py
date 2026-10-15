"""What a provider's HTTP failure actually said, including when it was a web page.

The bodies are what real systems send: an nginx error page, a captive portal's login page, a
gateway's JSON error, a bare line of text, nothing. Each used to collapse into one generic
sentence, and a ``200`` that was not an event stream used to become an empty, successful answer.
"""

from __future__ import annotations

import httpx
import pytest

from zett_agent.messages import UserMessage
from zett_agent.model import ModelRequest, RetryOptions
from zett_agent.providers.anthropic import AnthropicProvider
from zett_agent.providers.base import ProviderAuthError, ProviderError, ProviderHTTPError, ProviderResponseError
from zett_agent.providers.google import GoogleProvider
from zett_agent.providers.http_errors import failure_message, read_failure
from zett_agent.providers.ollama import OllamaProvider
from zett_agent.providers.openai import OpenAIProvider

REQUEST = ModelRequest(messages=(UserMessage(content="hi"),))
NO_RETRY = RetryOptions(max_retries=0)

NGINX = (
    b"<html>\r\n<head><title>502 Bad Gateway</title></head>\r\n<body>\r\n"
    b"<center><h1>502 Bad Gateway</h1></center>\r\n<hr><center>nginx/1.24.0</center>\r\n</body>\r\n</html>\r\n"
)
PORTAL = (
    b"<!DOCTYPE html><html><head><title>Sign in to WiFi</title></head><body><form>Please sign in</form></body></html>"
)


# ------------------------------------------------------------------ reading a body


def test_a_web_page_is_recognized_by_its_content_type_or_by_what_it_starts_with():
    by_type = read_failure(content_type="text/html; charset=utf-8", text="whatever")
    by_start = read_failure(content_type=None, text="  \r\n<!DOCTYPE html><html><title>Oops</title></html>")
    by_tag = read_failure(content_type="text/plain", text="<html><body>x</body></html>")

    assert by_type.is_html and by_start.is_html and by_tag.is_html
    assert not read_failure(content_type="application/json", text='{"error": "x"}').is_html
    assert not read_failure(content_type="text/plain", text="upstream connect error").is_html


def test_the_title_of_a_page_is_found_and_cleaned():
    assert read_failure(content_type="text/html", text=NGINX.decode()).title == "502 Bad Gateway"
    assert read_failure(content_type="text/html", text="<title>  Sign &amp; go\n</title>").title == "Sign & go"
    # With no title the first heading is the next best thing.
    assert read_failure(content_type="text/html", text="<html><h2>Maintenance</h2></html>").title == "Maintenance"
    assert read_failure(content_type="text/html", text="<html><body>plain</body></html>").title is None


def test_a_pages_preview_is_plain_text_without_scripts_or_markup():
    page = (
        "<html><head><style>p{}</style><script>alert(1)</script></head><body><p>Service  <b>down</b></p></body></html>"
    )

    assert read_failure(content_type="text/html", text=page).preview == "Service down"


@pytest.mark.parametrize(
    ("parsed", "expected"),
    [
        ({"error": {"message": "model not found", "type": "invalid_request_error"}}, "model not found"),
        ({"message": "model not found", "type": "invalid_request_error"}, "model not found"),  # the SDK's inner object
        ({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}, "Overloaded"),
        ({"error": "quota exceeded"}, "quota exceeded"),
        ({"detail": "Not Found"}, "Not Found"),
        ({"msg": "bad"}, "bad"),
        ("a bare string", "a bare string"),
    ],
)
def test_the_providers_own_words_are_found_in_the_shapes_providers_use(parsed, expected):
    assert read_failure(content_type="application/json", text="", parsed=parsed).reason == expected


def test_a_json_body_is_read_even_when_the_caller_did_not_decode_it():
    failure = read_failure(content_type="application/json", text='{"error": {"message": "rate limited"}}')

    assert failure.reason == "rate limited"
    assert read_failure(content_type="application/json", text="{broken").reason is None


def test_a_plain_text_body_is_kept_as_a_preview_and_a_long_one_is_cut():
    short = read_failure(content_type="text/plain", text="upstream connect error or disconnect/reset before headers")
    long = read_failure(content_type="text/plain", text="word " * 400)

    assert short.reason is None and short.preview == "upstream connect error or disconnect/reset before headers"
    assert len(long.preview) <= 300 and long.preview.endswith("…")


def test_an_empty_body_has_nothing_to_show():
    failure = read_failure(content_type=None, text="")

    assert failure.reason is None and failure.preview == "" and not failure.is_html


# ------------------------------------------------------------- the sentence it makes


def test_the_message_for_a_page_says_it_is_a_page_and_what_usually_causes_that():
    message = failure_message("openai", 502, read_failure(content_type="text/html", text=NGINX.decode()))

    assert 'HTTP 502 with a web page ("502 Bad Gateway") instead of an API response' in message
    assert "proxy, gateway, firewall or login page" in message and "base_url" in message
    assert "<" not in message  # never pastes markup


def test_the_message_for_a_json_error_carries_the_providers_words():
    failure = read_failure(content_type="application/json", text="", parsed={"error": {"message": "no such model"}})

    assert failure_message("kimi", 400, failure) == "kimi returned HTTP 400: no such model"


def test_the_message_falls_back_to_a_preview_then_to_the_status_phrase():
    text = read_failure(content_type="text/plain", text="upstream reset")
    nothing = read_failure(content_type=None, text="")

    assert failure_message("p", 500, text) == "p returned HTTP 500: upstream reset"
    assert failure_message("p", 502, nothing) == "p returned HTTP 502 (Bad Gateway)"
    assert failure_message("p", 599, nothing) == "p returned HTTP 599 with an empty body"


# ------------------------------------------------------ every protocol, end to end


def chat(status, ctype, body):
    return OpenAIProvider("m", "k", transport=transport(status, ctype, body), retry=NO_RETRY)


def responses(status, ctype, body):
    return OpenAIProvider("m", "k", transport=transport(status, ctype, body), retry=NO_RETRY, response=True)


def anthropic(status, ctype, body):
    return AnthropicProvider("m", "k", transport=transport(status, ctype, body), retry=NO_RETRY)


def transport(status, ctype, body):
    return httpx.MockTransport(lambda request: httpx.Response(status, content=body, headers={"content-type": ctype}))


PROTOCOLS = {"chat": chat, "responses": responses, "anthropic": anthropic}


async def failure_of(provider):
    try:
        with pytest.raises(ProviderError) as raised:
            [event async for event in provider.stream(REQUEST)]
    finally:
        await provider.aclose()
    return raised.value


@pytest.mark.parametrize("name", PROTOCOLS)
@pytest.mark.parametrize("status", [502, 503, 504, 500])
async def test_an_error_page_in_front_of_the_api_is_reported_as_a_page(name, status):
    error = await failure_of(PROTOCOLS[name](status, "text/html", NGINX))

    assert isinstance(error, ProviderHTTPError) and error.is_html and error.status_code == status
    assert f"HTTP {status} with a web page" in str(error) and "502 Bad Gateway" in str(error)
    assert error.reason is None and "nginx" in error.preview.lower()
    assert error.__cause__ is not None  # the SDK's own error is still there


@pytest.mark.parametrize("name", PROTOCOLS)
async def test_a_login_page_where_the_api_should_be_is_not_an_empty_answer(name):
    """The captive portal case: ``200`` with HTML, which used to read as a successful empty reply."""
    error = await failure_of(PROTOCOLS[name](200, "text/html; charset=utf-8", PORTAL))

    assert isinstance(error, ProviderHTTPError) and error.is_html and error.status_code == 200
    assert "Sign in to WiFi" in str(error) and "instead of an API response" in str(error)


@pytest.mark.parametrize("name", PROTOCOLS)
async def test_json_where_a_stream_was_asked_for_is_reported_with_what_it_said(name):
    body = b'{"error":{"message":"quota exceeded","type":"insufficient_quota"}}'

    error = await failure_of(PROTOCOLS[name](200, "application/json", body))

    assert isinstance(error, ProviderHTTPError) and error.status_code == 200
    assert "application/json instead of an event stream" in str(error) and "quota exceeded" in str(error)


@pytest.mark.parametrize("name", ["chat", "anthropic"])
async def test_a_stream_that_carries_nothing_is_an_error_not_an_empty_answer(name):
    error = await failure_of(PROTOCOLS[name](200, "text/event-stream", b""))

    assert "without sending any data" in str(error) and "HTTP 200" in str(error) and "text/event-stream" in str(error)


async def test_the_responses_endpoint_also_refuses_an_empty_stream():
    error = await failure_of(responses(200, "text/event-stream", b""))

    assert "without a terminal response" in str(error)


@pytest.mark.parametrize("name", PROTOCOLS)
@pytest.mark.parametrize(
    ("status", "body", "words"),
    [
        (400, b'{"error":{"message":"model [x] service info not found"}}', "model [x] service info not found"),
        (404, b'{"error":{"message":"no such route"}}', "no such route"),
        (429, b'{"error":{"message":"Rate limit reached"}}', "Rate limit reached"),
        (500, b"upstream connect error or disconnect/reset before headers", "upstream connect error"),
    ],
)
async def test_a_failing_status_carries_the_providers_own_explanation(name, status, body, words):
    ctype = "text/plain" if not body.startswith(b"{") else "application/json"

    error = await failure_of(PROTOCOLS[name](status, ctype, body))

    assert isinstance(error, ProviderHTTPError) and error.status_code == status
    assert f"HTTP {status}" in str(error) and words in str(error)
    assert (error.reason or error.preview).startswith(words[:10])


@pytest.mark.parametrize("name", PROTOCOLS)
async def test_a_rejected_key_is_still_an_auth_error_and_still_explains(name):
    error = await failure_of(PROTOCOLS[name](401, "application/json", b'{"error":{"message":"Incorrect API key"}}'))

    assert isinstance(error, ProviderAuthError)
    assert "rejected API credential" in str(error) and "HTTP 401: Incorrect API key" in str(error)


async def test_a_login_page_where_a_key_was_expected_says_so_instead_of_blaming_the_key():
    error = await failure_of(chat(401, "text/html", NGINX.replace(b"502 Bad Gateway", b"401 Authorization Required")))

    assert (
        isinstance(error, ProviderAuthError) and "web page" in str(error) and "401 Authorization Required" in str(error)
    )


@pytest.mark.parametrize(
    ("status", "retried"),
    [(500, True), (502, True), (503, True), (429, True), (400, False), (404, False), (401, False)],
)
async def test_which_statuses_are_retried_is_unchanged(status, retried):
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(status, json={"error": {"message": "x"}})

    provider = OpenAIProvider(
        "m",
        "k",
        transport=httpx.MockTransport(handler),
        retry=RetryOptions(max_retries=2, base_delay=0.001, max_delay=0.002),
    )
    try:
        with pytest.raises(ProviderError):
            [event async for event in provider.stream(REQUEST)]
    finally:
        await provider.aclose()

    assert len(attempts) == (3 if retried else 1)


async def test_a_page_with_a_success_status_is_never_retried():
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(200, content=PORTAL, headers={"content-type": "text/html"})

    provider = OpenAIProvider(
        "m",
        "k",
        transport=httpx.MockTransport(handler),
        retry=RetryOptions(max_retries=3, base_delay=0.001, max_delay=0.002),
    )
    try:
        with pytest.raises(ProviderHTTPError):
            [event async for event in provider.stream(REQUEST)]
    finally:
        await provider.aclose()

    assert len(attempts) == 1


@pytest.mark.parametrize("ctype", ["text/plain", "text/plain; charset=utf-8", "application/octet-stream"])
async def test_a_real_stream_sent_with_an_unusual_content_type_is_still_read(ctype):
    """Some gateways send an event stream as text/plain; only a web page or JSON is certainly not one."""
    body = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    provider = chat(200, ctype, body)
    try:
        events = [event async for event in provider.stream(REQUEST)]
    finally:
        await provider.aclose()

    assert events[-1].response.message.content == "ok"


async def test_an_unusual_content_type_with_nothing_in_it_is_still_an_error():
    error = await failure_of(chat(200, "text/plain", b"<html>not a stream</html>"))

    assert "without sending any data" in str(error)


async def test_a_real_event_stream_with_a_charset_is_accepted():
    body = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    provider = chat(200, "text/event-stream; charset=utf-8", body)
    try:
        events = [event async for event in provider.stream(REQUEST)]
    finally:
        await provider.aclose()

    assert events[-1].response.message.content == "ok"


# ------------------------------------------------------------------ Google, Ollama


def google(status, ctype, body):
    return GoogleProvider("gemini-2.5-flash", api_key="k", transport=transport(status, ctype, body), retry=NO_RETRY)


def ollama(status, ctype, body):
    return OllamaProvider("m", transport=transport(status, ctype, body), retry=NO_RETRY)


@pytest.mark.parametrize("make", [google, ollama], ids=["google", "ollama"])
async def test_an_error_page_is_a_page_for_google_and_ollama_too(make):
    error = await failure_of(make(502, "text/html", NGINX))

    assert isinstance(error, ProviderHTTPError) and error.is_html and error.status_code == 502
    assert "502 Bad Gateway" in str(error) and "<" not in str(error)


async def test_google_does_not_leak_an_sdk_exception_for_a_page_with_a_success_status():
    error = await failure_of(google(200, "text/html", PORTAL))

    assert isinstance(error, ProviderHTTPError) and error.is_html and "Sign in to WiFi" in str(error)


async def test_ollama_does_not_leak_a_json_error_for_something_that_is_not_ollama():
    error = await failure_of(ollama(200, "text/html", PORTAL))

    assert type(error) is ProviderResponseError and "not a stream of JSON lines" in str(error)


@pytest.mark.parametrize(
    ("make", "body", "words"),
    [
        (
            google,
            b'{"error":{"code":400,"message":"API key not valid","status":"INVALID_ARGUMENT"}}',
            "API key not valid",
        ),
        (ollama, b'{"error":"model \'x\' not found"}', "model 'x' not found"),
    ],
    ids=["google", "ollama"],
)
async def test_google_and_ollama_carry_the_servers_words(make, body, words):
    status = 400 if make is google else 404

    error = await failure_of(make(status, "application/json", body))

    assert isinstance(error, ProviderHTTPError) and words in str(error) and f"HTTP {status}" in str(error)


@pytest.mark.parametrize("make", [google, ollama], ids=["google", "ollama"])
async def test_a_rejected_key_is_an_auth_error_for_google_and_ollama(make):
    error = await failure_of(make(401, "application/json", b'{"error":{"code":401,"message":"bad key"}}'))

    assert isinstance(error, ProviderAuthError) and "bad key" in str(error)


@pytest.mark.parametrize("make", [google, ollama], ids=["google", "ollama"])
async def test_an_empty_answer_is_an_error_for_google_and_ollama(make):
    error = await failure_of(make(200, "text/event-stream", b""))

    assert "without" in str(error)
