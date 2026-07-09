"""Pluggable LLM + embedding client.

Defaults to a local Ollama server so the app is free and private. The public
interface is small (``complete``, ``complete_json``, ``embed``) so a cloud
backend can be swapped in later without touching callers.
"""

from __future__ import annotations

import json
from typing import cast

import requests

from aiapply_lite.config import settings


class LLMError(RuntimeError):
    """Raised when the LLM backend is unreachable or returns an error."""


class OllamaClient:
    """Thin client over the Ollama HTTP API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        embed_model: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.llm_model
        self.embed_model = embed_model or settings.embed_model
        self.timeout = timeout or settings.llm_timeout

    def available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def complete(self, prompt: str, system: str = "", temperature: float = 0.3) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network
            raise LLMError(f"Ollama request failed: {exc}") from exc
        data = resp.json()
        return str(data.get("response", "")).strip()

    def complete_json(self, prompt: str, system: str = "", temperature: float = 0.1) -> dict[str, object]:
        """Ask the model for JSON and parse it defensively."""

        instruction = "Respond with ONLY valid JSON, no markdown fences, no prose."
        full_system = f"{system}\n{instruction}".strip()
        raw = self.complete(prompt, system=full_system, temperature=temperature)
        return _parse_json_object(raw)

    def embed(self, text: str) -> list[float]:
        try:
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network
            raise LLMError(f"Ollama embedding request failed: {exc}") from exc
        data = resp.json()
        vector = cast("list[float]", data.get("embedding", []))
        return [float(x) for x in vector]


def _parse_json_object(raw: str) -> dict[str, object]:
    """Extract the first JSON object from a model response."""

    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return cast(dict[str, object], parsed)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return cast(dict[str, object], parsed)
        except json.JSONDecodeError:
            pass
    return {}


_default_client: OllamaClient | None = None


def get_client() -> OllamaClient:
    global _default_client
    if _default_client is None:
        _default_client = OllamaClient()
    return _default_client
