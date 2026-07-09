"""Score jobs against a CV.

The final score blends two signals:
  - semantic similarity between CV and job text (embeddings, when available)
  - hard-skill keyword overlap (always available, deterministic)

If embeddings are unavailable the score falls back to keyword overlap only,
so the feature still works fully offline without an embedding model.
"""

from __future__ import annotations

import math

from aiapply_lite.core import keywords
from aiapply_lite.core.llm import LLMError, OllamaClient, get_client
from aiapply_lite.models import CVProfile, Job, MatchResult

SEMANTIC_WEIGHT = 0.6
KEYWORD_WEIGHT = 0.4


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def keyword_overlap(cv_skills: list[str], job_text: str) -> tuple[list[str], list[str]]:
    """Return (matched, missing) skills between the CV and a job."""

    job_skills = set(keywords.extract_skills(job_text))
    cv_set = {s.lower() for s in cv_skills}
    matched = sorted(job_skills & cv_set)
    missing = sorted(job_skills - cv_set)
    return matched, missing


def score_jobs(
    profile: CVProfile,
    jobs: list[Job],
    client: OllamaClient | None = None,
) -> list[MatchResult]:
    client = client or get_client()
    use_embeddings = False
    cv_vec: list[float] = []
    try:
        if profile.raw_text and client.available():
            cv_vec = client.embed(_cv_text(profile))
            use_embeddings = bool(cv_vec)
    except LLMError:
        use_embeddings = False

    results: list[MatchResult] = []
    for job in jobs:
        job_text = f"{job.title}\n{job.description}\n{' '.join(job.tags)}"
        matched, missing = keyword_overlap(profile.skills, job_text)

        keyword_score = _keyword_score(matched, missing)
        if use_embeddings:
            try:
                job_vec = client.embed(job_text[:4000])
                semantic = max(0.0, cosine_similarity(cv_vec, job_vec))
            except LLMError:
                semantic = keyword_score
            blended = SEMANTIC_WEIGHT * semantic + KEYWORD_WEIGHT * keyword_score
        else:
            blended = keyword_score

        results.append(
            MatchResult(
                job=job,
                score=int(round(blended * 100)),
                matched_keywords=matched,
                missing_keywords=missing,
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results


def _keyword_score(matched: list[str], missing: list[str]) -> float:
    total = len(matched) + len(missing)
    if total == 0:
        return 0.0
    return len(matched) / total


def _cv_text(profile: CVProfile) -> str:
    parts = [profile.title, profile.summary, " ".join(profile.skills), profile.raw_text]
    return "\n".join(p for p in parts if p)[:4000]
