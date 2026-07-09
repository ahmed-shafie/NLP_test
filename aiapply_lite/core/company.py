"""Research a company: what they do, history, products, news, tech stack.

Uses a web-search API (Tavily) for current context when a key is configured,
otherwise falls back to the LLM's own knowledge. Either way the LLM produces
the final structured report.
"""

from __future__ import annotations

from typing import cast

import requests

from aiapply_lite.config import settings
from aiapply_lite.core.llm import LLMError, OllamaClient, get_client
from aiapply_lite.models import CompanyReport

_SYSTEM = (
    "You are a company research assistant. Given a company name and optional "
    "web-search context, produce a factual briefing. Return JSON with keys: "
    "what_they_do, history, products, recent_news, tech_stack. Each value is a "
    "short paragraph string. If unsure, say so rather than inventing facts."
)


def research_company(name: str, client: OllamaClient | None = None) -> CompanyReport:
    name = name.strip()
    report = CompanyReport(company=name)
    if not name:
        return report

    context, sources = _web_context(name)
    report.sources = sources

    client = client or get_client()
    try:
        if client.available():
            prompt = f"COMPANY: {name}\n\nWEB CONTEXT:\n{context or '(none)'}"
            data = client.complete_json(prompt, system=_SYSTEM)
            report.what_they_do = str(data.get("what_they_do", ""))
            report.history = str(data.get("history", ""))
            report.products = str(data.get("products", ""))
            report.recent_news = str(data.get("recent_news", ""))
            report.tech_stack = str(data.get("tech_stack", ""))
    except LLMError:
        report.what_they_do = "LLM unavailable - could not generate report."
    return report


def _web_context(name: str) -> tuple[str, list[str]]:
    if not settings.tavily_enabled:
        return "", []
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": settings.tavily_api_key,
                "query": f"{name} company overview products history",
                "max_results": 5,
                "search_depth": "basic",
            },
            timeout=20,
        )
        resp.raise_for_status()
    except requests.RequestException:
        return "", []
    payload = cast(dict[str, object], resp.json())
    results = cast(list[dict[str, object]], payload.get("results", []))
    snippets: list[str] = []
    sources: list[str] = []
    for item in results:
        content = str(item.get("content", ""))
        url = str(item.get("url", ""))
        if content:
            snippets.append(content)
        if url:
            sources.append(url)
    return "\n\n".join(snippets), sources
