"""Generate a tailored resume and cover letter for a target job."""

from __future__ import annotations

from aiapply_lite.core.llm import LLMError, OllamaClient, get_client
from aiapply_lite.models import CVProfile, Job

_RESUME_SYSTEM = (
    "You are an expert resume writer. Rewrite the candidate's resume tailored to "
    "the target job, emphasizing relevant experience and matching keywords "
    "honestly (never invent experience). Output clean markdown."
)

_COVER_SYSTEM = (
    "You are an expert cover-letter writer. Write a concise, specific cover "
    "letter (3-4 short paragraphs) for the candidate and target job. Reference "
    "the company where relevant. Do not fabricate achievements."
)


def generate_resume(profile: CVProfile, job: Job, client: OllamaClient | None = None) -> str:
    client = client or get_client()
    prompt = (
        f"CANDIDATE CV:\n{profile.raw_text[:5000]}\n\n"
        f"TARGET JOB: {job.title} at {job.company}\n"
        f"JOB DESCRIPTION:\n{job.description[:3000]}"
    )
    try:
        if client.available():
            return client.complete(prompt, system=_RESUME_SYSTEM, temperature=0.4)
    except LLMError:
        pass
    return profile.raw_text or "(no CV provided)"


def generate_cover_letter(profile: CVProfile, job: Job, client: OllamaClient | None = None) -> str:
    client = client or get_client()
    prompt = (
        f"CANDIDATE:\n"
        f"- Name: {profile.name or '(candidate)'}\n"
        f"- Title: {profile.title}\n"
        f"- Skills: {', '.join(profile.skills)}\n"
        f"- Summary: {profile.summary}\n\n"
        f"TARGET JOB: {job.title} at {job.company}\n"
        f"JOB DESCRIPTION:\n{job.description[:3000]}"
    )
    try:
        if client.available():
            return client.complete(prompt, system=_COVER_SYSTEM, temperature=0.5)
    except LLMError:
        pass
    return (
        f"Dear Hiring Manager,\n\nI am excited to apply for the {job.title} role at "
        f"{job.company}.\n\n(LLM unavailable - draft could not be generated.)\n\n"
        f"Best regards,\n{profile.name}"
    )
