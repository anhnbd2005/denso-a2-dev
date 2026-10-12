"""Offline contract tests for the explicit OpenAI-compatible client."""

import os
from types import SimpleNamespace

from llm_client import LLMStructuredOutputError, UniversalLLMClient


def response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def stream_response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content))]
    )


class FakeCompletions:
    def __init__(self, outputs, reject_json_mode=False):
        self.outputs = list(outputs)
        self.reject_json_mode = reject_json_mode
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.reject_json_mode and "response_format" in kwargs:
            raise RuntimeError("response_format is unsupported")
        output = self.outputs.pop(0)
        if kwargs.get("stream"):
            return [stream_response(chunk) for chunk in output]
        return response(output)


def configured_client(completions):
    client = UniversalLLMClient(api_key="unit-test-secret")
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    return client


def run_llm_client_tests() -> None:
    old_values = {
        key: os.environ.get(key)
        for key in (
            "OPENAI_API_KEY",
            "OPENAI_COMPAT_MODEL",
            "OPENAI_MODEL_NAME",
            "OPENAI_MODEL",
        )
    }
    try:
        for key in old_values:
            os.environ.pop(key, None)
        unconfigured = UniversalLLMClient()
        assert unconfigured.client is None
        assert unconfigured.provider == "unconfigured"
        assert unconfigured.health_check()["ok"] is False
        assert "api_key" not in unconfigured.health_check()
        try:
            unconfigured.generate("x", "y")
        except RuntimeError as exc:
            assert "not configured" in str(exc)
        else:
            raise AssertionError("Missing key must fail explicitly")

        os.environ["OPENAI_MODEL"] = "lowest"
        os.environ["OPENAI_MODEL_NAME"] = "middle"
        os.environ["OPENAI_COMPAT_MODEL"] = "highest"
        assert UniversalLLMClient(api_key="x").model_name == "highest"
        assert (
            UniversalLLMClient(api_key="x", model_name="explicit").model_name
            == "explicit"
        )

        health_calls = FakeCompletions(["OK"])
        health_client = configured_client(health_calls)
        health = health_client.health_check()
        assert health["ok"] is True and health["response"] == "OK"
        assert health_calls.calls[0]["messages"][0]["content"] == (
            "Reply exactly with OK"
        )
        assert "api_key" not in health

        text_calls = FakeCompletions(["Câu trả lời có căn cứ."])
        text_client = configured_client(text_calls)
        assert text_client.generate("prompt", "system", max_tokens=321) == (
            "Câu trả lời có căn cứ."
        )
        assert text_calls.calls[0]["temperature"] == 0
        assert text_calls.calls[0]["max_tokens"] == 321

        stream_calls = FakeCompletions([["Câu ", "trả lời ", "đang stream."]])
        stream_client = configured_client(stream_calls)
        assert "".join(
            stream_client.generate_stream("prompt", "system", max_tokens=222)
        ) == "Câu trả lời đang stream."
        assert stream_calls.calls[0]["stream"] is True
        assert stream_calls.calls[0]["max_tokens"] == 222

        native_calls = FakeCompletions(['{"ok": true}'])
        native = configured_client(native_calls)
        assert native.generate_json("prompt", "system") == {"ok": True}
        assert native.last_json_mode == "response_format"
        assert native_calls.calls[0]["response_format"] == {
            "type": "json_object"
        }

        fallback_calls = FakeCompletions(
            ['```json\n{"ok": true}\n```'], reject_json_mode=True
        )
        fallback = configured_client(fallback_calls)
        assert fallback.generate_json("prompt", "system") == {"ok": True}
        assert fallback.last_json_mode == "prompt_only_fallback"
        assert len(fallback_calls.calls) == 2
        assert "response_format" not in fallback_calls.calls[1]

        invalid = configured_client(FakeCompletions(["not json"]))
        try:
            invalid.generate_json("prompt", "system")
        except LLMStructuredOutputError:
            pass
        else:
            raise AssertionError("Invalid JSON must fail closed")
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    print("[PASS] OPENAI-COMPATIBLE CLIENT CONTRACT")
    print("Unconfigured explicit | model priority | JSON mode/fallback | fail closed")


if __name__ == "__main__":
    run_llm_client_tests()
