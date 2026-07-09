from aiapply_lite.core import matching
from aiapply_lite.core.llm import OllamaClient
from aiapply_lite.models import CVProfile, Job


class _OfflineClient(OllamaClient):
    def available(self) -> bool:
        return False


def _job(title: str, desc: str) -> Job:
    return Job(
        source="mock",
        external_id="x",
        title=title,
        company="Co",
        location="Remote",
        url="",
        description=desc,
    )


def test_cosine_similarity_bounds() -> None:
    assert matching.cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert matching.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert matching.cosine_similarity([], [1.0]) == 0.0


def test_keyword_overlap() -> None:
    matched, missing = matching.keyword_overlap(["python", "aws"], "Python and Docker role")
    assert "python" in matched
    assert "docker" in missing
    assert "aws" not in matched  # not in the job text


def test_score_jobs_offline_uses_keyword_score() -> None:
    profile = CVProfile(raw_text="python aws docker", skills=["python", "aws", "docker"])
    jobs = [
        _job("Python Eng", "python docker aws role"),
        _job("Sales", "cold calling and crm"),
    ]
    results = matching.score_jobs(profile, jobs, client=_OfflineClient())
    assert results[0].job.title == "Python Eng"
    assert results[0].score >= results[1].score
    assert results[0].score > 0
