"""Explicit client for a user-configured OpenAI-compatible endpoint."""

import json
import os
from typing import Any, Dict, Optional

from openai import OpenAI


DEFAULT_BASE_URL = "http://localhost:20128/v1"
DEFAULT_MODEL = "ag/gemini-3.8-flash-low"


class LLMStructuredOutputError(RuntimeError):
    """Raised when the provider does not return one parseable JSON object."""


class UniversalLLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_seconds: float = 60.0,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = (
            base_url or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        )
        self.model_name = (
            model_name
            or os.getenv("OPENAI_COMPAT_MODEL")
            or os.getenv("OPENAI_MODEL_NAME")
            or os.getenv("OPENAI_MODEL")
            or DEFAULT_MODEL
        )
        self.model = self.model_name
        self.last_json_mode: Optional[str] = None
        self.last_raw_output: Optional[str] = None
        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=timeout_seconds,
                max_retries=1,
            )
            self.provider = "openai_compatible"
        else:
            self.client = None
            self.provider = "unconfigured"

    def _require_client(self) -> OpenAI:
        if self.client is None:
            raise RuntimeError("OpenAI-compatible API is not configured.")
        return self.client

    def health_check(self) -> Dict[str, Any]:
        result = {
            "ok": False,
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model_name,
            "error": None,
        }
        if self.client is None:
            result["error"] = "OpenAI-compatible API is not configured."
            return result
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": "Reply exactly with OK"}
                ],
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            result.update({"ok": True, "response": content.strip()})
        except Exception as exc:
            safe_message = str(exc)
            if self.api_key:
                safe_message = safe_message.replace(self.api_key, "***")
            result["error"] = f"{type(exc).__name__}: {safe_message}"
        return result

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 700,
    ) -> str:
        response = self._require_client().chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("OpenAI-compatible API returned empty content.")
        return content

    @staticmethod
    def _response_format_unsupported(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        message = str(exc).casefold()
        marker = any(
            text in message
            for text in (
                "response_format",
                "json_object",
                "json mode",
                "unsupported parameter",
                "unknown parameter",
            )
        )
        return marker and (status_code in {400, 404, 422} or status_code is None)

    @staticmethod
    def _parse_json_object(content: str) -> Dict[str, Any]:
        cleaned = content.strip()
        if cleaned.startswith("```json") and cleaned.endswith("```"):
            cleaned = cleaned[7:-3].strip()
        elif cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned[3:-3].strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMStructuredOutputError(
                "Provider output is not valid JSON."
            ) from exc
        if not isinstance(parsed, dict):
            raise LLMStructuredOutputError(
                "Provider output must be exactly one JSON object."
            )
        return parsed

    def generate_json(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        client = self._require_client()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
            self.last_json_mode = "response_format"
        except Exception as exc:
            if not self._response_format_unsupported(exc):
                raise
            fallback_instruction = (
                "\n\nReturn exactly one valid JSON object. "
                "No markdown. No code fence. No explanation."
            )
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt + fallback_instruction},
                ],
                temperature=0,
            )
            self.last_json_mode = "prompt_only_fallback"
        content = response.choices[0].message.content or ""
        self.last_raw_output = content
        return self._parse_json_object(content)
