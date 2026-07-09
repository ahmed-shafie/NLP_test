"""Streamlit UI for aiapply_lite - a personal, local job-search assistant."""

from __future__ import annotations

import streamlit as st

from aiapply_lite.config import settings
from aiapply_lite.core import cv, generate, jobs, matching, roadmap, storage
from aiapply_lite.core.company import research_company
from aiapply_lite.core.llm import get_client
from aiapply_lite.models import CompanyReport, CVProfile, Job, MatchResult


def _get_profile() -> CVProfile | None:
    if "profile" not in st.session_state:
        st.session_state["profile"] = storage.load_profile()
    return st.session_state.get("profile")


def _set_profile(profile: CVProfile) -> None:
    st.session_state["profile"] = profile
    storage.save_profile(profile)


def sidebar_status() -> None:
    st.sidebar.header("Status")
    client = get_client()
    if client.available():
        st.sidebar.success(f"Ollama online ({client.model})")
    else:
        st.sidebar.error("Ollama offline - features degrade gracefully")
    st.sidebar.caption(f"Job sources: {', '.join(settings.job_source_list)}")
    st.sidebar.caption(f"Web search: {'on' if settings.tavily_enabled else 'off'}")


def tab_cv() -> None:
    st.header("My CV")
    profile = _get_profile()
    default_text = profile.raw_text if profile else ""
    text = st.text_area("Paste your CV / resume text", value=default_text, height=300)
    if st.button("Parse & save CV", type="primary"):
        if not text.strip():
            st.warning("Please paste your CV text first.")
            return
        with st.spinner("Parsing CV..."):
            parsed = cv.parse_cv(text)
            _set_profile(parsed)
        st.success("CV parsed and saved.")

    profile = _get_profile()
    if profile:
        st.subheader("Parsed profile")
        st.write(f"**Name:** {profile.name or '-'}")
        st.write(f"**Title:** {profile.title or '-'}")
        st.write(f"**Years experience:** {profile.years_experience}")
        st.write(f"**Skills:** {', '.join(profile.skills) or '-'}")
        if profile.summary:
            st.write(f"**Summary:** {profile.summary}")


def tab_search() -> None:
    st.header("Job Search + Fit Score")
    profile = _get_profile()
    if not profile or not profile.raw_text:
        st.info("Add your CV in the 'My CV' tab first for accurate scoring.")
    query = st.text_input("Search jobs (role, skill, keyword)", value="python")
    limit = st.slider("Max results", 5, 50, 20)
    if st.button("Search", type="primary"):
        with st.spinner("Fetching and scoring jobs..."):
            found = jobs.fetch_jobs(query, limit=limit)
            results = matching.score_jobs(profile or CVProfile(), found)
            st.session_state["results"] = results
    _render_results(st.session_state.get("results", []))


def _render_results(results: list[MatchResult]) -> None:
    if not results:
        return
    st.subheader(f"{len(results)} jobs")
    for idx, res in enumerate(results):
        job = res.job
        with st.expander(f"[{res.score}%] {job.title} - {job.company} ({job.source})"):
            st.write(f"**Location:** {job.location}")
            if job.url:
                st.write(f"**Link:** {job.url}")
            st.progress(min(res.score, 100) / 100)
            st.caption(res.explanation)
            st.write(job.description[:1500] + ("..." if len(job.description) > 1500 else ""))
            cols = st.columns(3)
            if cols[0].button("Research company", key=f"co_{idx}"):
                st.session_state["selected_company"] = job.company
                st.session_state["active_tab_hint"] = "Company"
            if cols[1].button("Success roadmap", key=f"rm_{idx}"):
                st.session_state["selected_job"] = job
            if cols[2].button("Tailor resume", key=f"cv_{idx}"):
                st.session_state["tailor_job"] = job

    _render_inline_roadmap()
    _render_inline_tailor()


def _render_inline_roadmap() -> None:
    job = st.session_state.get("selected_job")
    if not isinstance(job, Job):
        return
    profile = _get_profile()
    if not profile:
        st.warning("Add your CV first to generate a roadmap.")
        return
    st.subheader(f"Roadmap: {job.title} @ {job.company}")
    with st.spinner("Generating roadmap..."):
        company = research_company(job.company) if job.company else None
        text = roadmap.generate_roadmap(profile, job, company)
    st.markdown(text)
    storage.save_output(f"roadmap_{job.company}_{job.title}.md", text)
    st.session_state.pop("selected_job", None)


def _render_inline_tailor() -> None:
    job = st.session_state.get("tailor_job")
    if not isinstance(job, Job):
        return
    profile = _get_profile()
    if not profile:
        st.warning("Add your CV first.")
        return
    st.subheader(f"Tailored application: {job.title} @ {job.company}")
    with st.spinner("Writing resume + cover letter..."):
        resume = generate.generate_resume(profile, job)
        cover = generate.generate_cover_letter(profile, job)
    st.markdown("### Tailored resume")
    st.markdown(resume)
    st.markdown("### Cover letter")
    st.markdown(cover)
    st.session_state.pop("tailor_job", None)


def tab_company() -> None:
    st.header("Company Deep-Dive")
    default = st.session_state.get("selected_company", "")
    name = st.text_input("Company name", value=str(default))
    if st.button("Research", type="primary") and name.strip():
        with st.spinner("Researching company..."):
            report: CompanyReport = research_company(name)
        st.subheader(report.company)
        st.markdown(f"**What they do**\n\n{report.what_they_do or '-'}")
        st.markdown(f"**History**\n\n{report.history or '-'}")
        st.markdown(f"**Products**\n\n{report.products or '-'}")
        st.markdown(f"**Tech stack**\n\n{report.tech_stack or '-'}")
        st.markdown(f"**Recent news**\n\n{report.recent_news or '-'}")
        if report.sources:
            st.caption("Sources: " + ", ".join(report.sources))


def main() -> None:
    st.set_page_config(page_title="AIApply Lite", page_icon="briefcase", layout="wide")
    st.title("AIApply Lite - personal job-search assistant")
    sidebar_status()

    tabs = st.tabs(["My CV", "Job Search + Score", "Company Deep-Dive"])
    with tabs[0]:
        tab_cv()
    with tabs[1]:
        tab_search()
    with tabs[2]:
        tab_company()


if __name__ == "__main__":
    main()
