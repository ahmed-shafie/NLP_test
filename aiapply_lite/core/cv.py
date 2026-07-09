"""Turn raw CV text into a structured :class:`CVProfile`.

Uses the local LLM when available for rich fields (name, summary, titles) and
always falls back to deterministic keyword extraction so the app degrades
gracefully without a model.
"""

from __future__ import annotations

import re

from aiapply_lite.core import keywords
from aiapply_lite.core.llm import LLMError, OllamaClient, get_client
from aiapply_lite.models import CVProfile

_CV_SYSTEM = (
    "You are a resume parser. Extract structured fields from the CV text. "
    "Return JSON with keys: name, title, summary, skills (array of strings), "
    "years_experience (number), titles_held (array), education (array)."
)


def parse_cv(text: str, client: OllamaClient | None = None) -> CVProfile:
    text = text.strip()
    profile = CVProfile(raw_text=text)
    profile.skills = keywords.extract_skills(text)
    profile.years_experience = _guess_years(text)

    client = client or get_client()
    try:
        if client.available():
            _enrich_with_llm(text, profile, client)
    except LLMError:
        pass
    return profile


def _enrich_with_llm(text: str, profile: CVProfile, client: OllamaClient) -> None:
    prompt = f"CV TEXT:\n{text[:6000]}"
    data = client.complete_json(prompt, system=_CV_SYSTEM)
    if not data:
        return
    profile.name = str(data.get("name", profile.name) or profile.name)
    profile.title = str(data.get("title", profile.title) or profile.title)
    profile.summary = str(data.get("summary", profile.summary) or profile.summary)

    llm_skills = data.get("skills")
    if isinstance(llm_skills, list):
        merged = set(profile.skills) | {str(s).lower() for s in llm_skills}
        profile.skills = sorted(merged)

    titles = data.get("titles_held")
    if isinstance(titles, list):
        profile.titles_held = [str(t) for t in titles]

    education = data.get("education")
    if isinstance(education, list):
        profile.education = [str(e) for e in education]

    years = data.get("years_experience")
    if isinstance(years, (int, float)) and years > 0:
        profile.years_experience = float(years)


def _guess_years(text: str) -> float:
    match = re.search(r"(\d+)\+?\s*years?\s+(?:of\s+)?experience", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 0.0
