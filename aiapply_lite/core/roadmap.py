"""Generate a personalized 'roadmap to succeed' for a target job + company."""

from __future__ import annotations

from aiapply_lite.core.llm import LLMError, OllamaClient, get_client
from aiapply_lite.core.matching import keyword_overlap
from aiapply_lite.models import CompanyReport, CVProfile, Job

_SYSTEM = (
    "You are a senior career coach. Using the candidate's CV, a target job, and "
    "a company briefing, write a concrete, honest roadmap for the candidate to "
    "land and succeed in this role. Cover: (1) gap analysis of missing skills, "
    "(2) a prioritized 30/60/90-day learning plan, (3) portfolio/projects to "
    "build, (4) how to align with the company's focus, (5) interview prep tips. "
    "Be specific and actionable. Use markdown with clear headings."
)


def generate_roadmap(
    profile: CVProfile,
    job: Job,
    company: CompanyReport | None = None,
    client: OllamaClient | None = None,
) -> str:
    client = client or get_client()
    matched, missing = keyword_overlap(profile.skills, f"{job.title} {job.description}")

    company_block = ""
    if company:
        company_block = (
            f"\nCOMPANY BRIEFING ({company.company}):\n"
            f"- What they do: {company.what_they_do}\n"
            f"- Products: {company.products}\n"
            f"- Tech stack: {company.tech_stack}\n"
            f"- Recent news: {company.recent_news}\n"
        )

    prompt = (
        f"CANDIDATE:\n"
        f"- Title: {profile.title}\n"
        f"- Years experience: {profile.years_experience}\n"
        f"- Skills: {', '.join(profile.skills) or '(none detected)'}\n"
        f"- Summary: {profile.summary}\n\n"
        f"TARGET JOB: {job.title} at {job.company} ({job.location})\n"
        f"JOB DESCRIPTION:\n{job.description[:3000]}\n\n"
        f"SKILLS ALREADY MATCHED: {', '.join(matched) or 'none'}\n"
        f"SKILLS TO ACQUIRE: {', '.join(missing) or 'none obvious'}\n"
        f"{company_block}"
    )

    try:
        if client.available():
            return client.complete(prompt, system=_SYSTEM, temperature=0.4)
    except LLMError:
        pass
    return _fallback_roadmap(matched, missing, job)


def _fallback_roadmap(matched: list[str], missing: list[str], job: Job) -> str:
    strengths = [f"- {s}" for s in matched] or ["- (none detected)"]
    gaps = [f"- {s}" for s in missing] or ["- (no obvious gaps detected)"]
    lines = [
        f"# Roadmap for {job.title} at {job.company}",
        "",
        "_(LLM unavailable - showing a keyword-based summary.)_",
        "",
        "## Strengths (already matched)",
        *strengths,
        "",
        "## Skills to acquire",
        *gaps,
        "",
        "## Suggested plan",
        "1. Prioritize the missing skills above, hardest-first.",
        "2. Build one portfolio project combining them.",
        "3. Prepare STAR interview stories mapping your experience to the role.",
    ]
    return "\n".join(lines)
