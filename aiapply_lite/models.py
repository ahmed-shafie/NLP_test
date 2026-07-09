"""Shared data structures used across the app."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Job:
    """A normalized job listing from any source."""

    source: str
    external_id: str
    title: str
    company: str
    location: str
    url: str
    description: str
    tags: list[str] = field(default_factory=list)
    salary: str = ""
    posted_at: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "external_id": self.external_id,
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "url": self.url,
            "description": self.description,
            "tags": self.tags,
            "salary": self.salary,
            "posted_at": self.posted_at,
        }


@dataclass
class MatchResult:
    """A job scored against a CV."""

    job: Job
    score: int  # 0-100
    matched_keywords: list[str] = field(default_factory=list)
    missing_keywords: list[str] = field(default_factory=list)

    @property
    def explanation(self) -> str:
        parts: list[str] = []
        if self.matched_keywords:
            parts.append("Matches: " + ", ".join(self.matched_keywords[:10]))
        if self.missing_keywords:
            parts.append("Missing: " + ", ".join(self.missing_keywords[:10]))
        return " | ".join(parts) if parts else "No keyword overlap details."


@dataclass
class CVProfile:
    """Structured representation of a user's CV."""

    raw_text: str = ""
    name: str = ""
    title: str = ""
    summary: str = ""
    skills: list[str] = field(default_factory=list)
    years_experience: float = 0.0
    titles_held: list[str] = field(default_factory=list)
    education: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_text": self.raw_text,
            "name": self.name,
            "title": self.title,
            "summary": self.summary,
            "skills": self.skills,
            "years_experience": self.years_experience,
            "titles_held": self.titles_held,
            "education": self.education,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CVProfile:
        return cls(
            raw_text=str(data.get("raw_text", "")),
            name=str(data.get("name", "")),
            title=str(data.get("title", "")),
            summary=str(data.get("summary", "")),
            skills=[str(s) for s in _as_list(data.get("skills"))],
            years_experience=_as_float(data.get("years_experience")),
            titles_held=[str(s) for s in _as_list(data.get("titles_held"))],
            education=[str(s) for s in _as_list(data.get("education"))],
        )


@dataclass
class CompanyReport:
    company: str
    what_they_do: str = ""
    history: str = ""
    products: str = ""
    recent_news: str = ""
    tech_stack: str = ""
    sources: list[str] = field(default_factory=list)


def _as_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    return []


def _as_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0
