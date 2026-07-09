"""Fetch job listings from pluggable, ToS-friendly sources.

Sources (configured via ``AIAPPLY_JOB_SOURCES``):
  - ``remotive``  : Remotive public API, no key required
  - ``remoteok``  : Remote OK public API, no key required
  - ``adzuna``    : Adzuna official API, requires free app id/key
  - ``mock``      : offline sample data (used automatically as a fallback)

Direct scraping of LinkedIn / Indeed is intentionally NOT implemented as it
violates their terms of service.
"""

from __future__ import annotations

import html
import re
from typing import cast

import requests

from aiapply_lite.config import settings
from aiapply_lite.models import Job

_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    return html.unescape(_TAG_RE.sub(" ", text)).strip()


def fetch_jobs(query: str, limit: int = 25, sources: list[str] | None = None) -> list[Job]:
    """Fetch and merge jobs from all configured sources."""

    sources = sources or settings.job_source_list
    jobs: list[Job] = []
    errors: list[str] = []

    for source in sources:
        try:
            if source == "remotive":
                jobs.extend(_fetch_remotive(query, limit))
            elif source == "remoteok":
                jobs.extend(_fetch_remoteok(query, limit))
            elif source == "adzuna" and settings.adzuna_enabled:
                jobs.extend(_fetch_adzuna(query, limit))
            elif source == "mock":
                jobs.extend(_fetch_mock(query, limit))
        except requests.RequestException as exc:  # pragma: no cover - network
            errors.append(f"{source}: {exc}")

    if not jobs:
        jobs = _fetch_mock(query, limit)

    return _dedupe(jobs)[:limit]


def _fetch_remotive(query: str, limit: int) -> list[Job]:
    params: dict[str, str | int] = {"search": query, "limit": limit}
    resp = requests.get(
        "https://remotive.com/api/remote-jobs",
        params=params,
        timeout=20,
    )
    resp.raise_for_status()
    payload = cast(dict[str, object], resp.json())
    raw_jobs = payload.get("jobs", [])
    jobs: list[Job] = []
    for item in cast(list[dict[str, object]], raw_jobs)[:limit]:
        jobs.append(
            Job(
                source="remotive",
                external_id=str(item.get("id", "")),
                title=str(item.get("title", "")),
                company=str(item.get("company_name", "")),
                location=str(item.get("candidate_required_location", "Remote")),
                url=str(item.get("url", "")),
                description=_strip_html(str(item.get("description", ""))),
                tags=[str(t) for t in cast(list[object], item.get("tags", []))],
                salary=str(item.get("salary", "")),
                posted_at=str(item.get("publication_date", "")),
            )
        )
    return jobs


def _fetch_remoteok(query: str, limit: int) -> list[Job]:
    resp = requests.get(
        "https://remoteok.com/api",
        headers={"User-Agent": "aiapply_lite/1.0"},
        timeout=20,
    )
    resp.raise_for_status()
    rows = cast(list[dict[str, object]], resp.json())
    q = query.lower()
    jobs: list[Job] = []
    for item in rows:
        if "id" not in item:  # first element is metadata
            continue
        title = str(item.get("position", item.get("title", "")))
        desc = _strip_html(str(item.get("description", "")))
        tags = [str(t) for t in cast(list[object], item.get("tags", []))]
        haystack = f"{title} {desc} {' '.join(tags)}".lower()
        if q and q not in haystack:
            continue
        jobs.append(
            Job(
                source="remoteok",
                external_id=str(item.get("id", "")),
                title=title,
                company=str(item.get("company", "")),
                location=str(item.get("location", "Remote")) or "Remote",
                url=str(item.get("url", "")),
                description=desc,
                tags=tags,
                posted_at=str(item.get("date", "")),
            )
        )
        if len(jobs) >= limit:
            break
    return jobs


def _fetch_adzuna(query: str, limit: int) -> list[Job]:
    country = settings.adzuna_country
    params: dict[str, str | int] = {
        "app_id": settings.adzuna_app_id,
        "app_key": settings.adzuna_app_key,
        "what": query,
        "results_per_page": limit,
        "content-type": "application/json",
    }
    resp = requests.get(
        f"https://api.adzuna.com/v1/api/jobs/{country}/search/1",
        params=params,
        timeout=20,
    )
    resp.raise_for_status()
    payload = cast(dict[str, object], resp.json())
    results = cast(list[dict[str, object]], payload.get("results", []))
    jobs: list[Job] = []
    for item in results[:limit]:
        company = cast(dict[str, object], item.get("company", {}))
        location = cast(dict[str, object], item.get("location", {}))
        salary_min = item.get("salary_min")
        salary = f"{salary_min}" if salary_min else ""
        jobs.append(
            Job(
                source="adzuna",
                external_id=str(item.get("id", "")),
                title=str(item.get("title", "")),
                company=str(company.get("display_name", "")),
                location=str(location.get("display_name", "")),
                url=str(item.get("redirect_url", "")),
                description=_strip_html(str(item.get("description", ""))),
                salary=salary,
                posted_at=str(item.get("created", "")),
            )
        )
    return jobs


def _fetch_mock(query: str, limit: int) -> list[Job]:
    samples = [
        Job(
            source="mock",
            external_id="mock-1",
            title="Senior Python Engineer",
            company="Acme Data",
            location="Remote",
            url="https://example.com/jobs/mock-1",
            description=(
                "Build data pipelines in Python. Experience with FastAPI, "
                "PostgreSQL, AWS, Docker and CI/CD required. NLP a plus."
            ),
            tags=["python", "fastapi", "aws", "docker"],
        ),
        Job(
            source="mock",
            external_id="mock-2",
            title="Machine Learning Engineer",
            company="Nimbus AI",
            location="Remote (EU)",
            url="https://example.com/jobs/mock-2",
            description=(
                "Work on NLP models with PyTorch and Hugging Face transformers. "
                "Deploy with Kubernetes. Strong Python and machine learning."
            ),
            tags=["python", "pytorch", "nlp", "kubernetes"],
        ),
        Job(
            source="mock",
            external_id="mock-3",
            title="Full Stack Developer",
            company="BrightApps",
            location="Remote",
            url="https://example.com/jobs/mock-3",
            description=(
                "React + TypeScript frontend, Node.js backend. REST and GraphQL "
                "APIs. Docker, GitHub Actions. Agile team."
            ),
            tags=["react", "typescript", "node.js", "graphql"],
        ),
    ]
    q = query.lower()
    filtered = [j for j in samples if not q or q in f"{j.title} {j.description} {' '.join(j.tags)}".lower()]
    return (filtered or samples)[:limit]


def _dedupe(jobs: list[Job]) -> list[Job]:
    seen: set[str] = set()
    unique: list[Job] = []
    for job in jobs:
        key = f"{job.company.lower()}|{job.title.lower()}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(job)
    return unique
