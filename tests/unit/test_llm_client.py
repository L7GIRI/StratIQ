import io
import json
import urllib.error

import pytest

from stratiq.llm import LLMError, OllamaClient


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_payload_is_deterministic_and_carries_the_schema(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout):
        seen["body"] = json.loads(req.data.decode("utf-8"))
        seen["url"] = req.full_url
        return FakeResponse(json.dumps({"message": {"content": "{}"}}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    out = OllamaClient(model="m")([{"role": "user", "content": "hi"}], {"type": "object"})
    assert out == "{}"
    assert seen["url"].endswith("/api/chat")
    assert seen["body"]["options"]["temperature"] == 0
    assert seen["body"]["format"] == {"type": "object"}
    assert seen["body"]["stream"] is False


def test_unreachable_server_raises_llm_error(monkeypatch):
    def boom(req, timeout):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(LLMError):
        OllamaClient()([{"role": "user", "content": "hi"}])


def test_unexpected_response_shape_raises_llm_error(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: FakeResponse(b'{"oops": 1}'))
    with pytest.raises(LLMError):
        OllamaClient()([{"role": "user", "content": "hi"}])


@pytest.mark.ollama
def test_real_model_turns_the_ema_example_into_a_valid_spec():
    from stratiq.llm import nl_to_spec

    res = nl_to_spec(
        "Buy gold when the 20 EMA crosses above the 50 EMA, sell when it crosses below. "
        "Stop 1.5 ATR, target 2R, risk 1% per trade, hourly chart.",
        OllamaClient(),
    )
    assert res.spec is not None or res.questions
