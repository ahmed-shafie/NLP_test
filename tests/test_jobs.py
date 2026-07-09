from aiapply_lite.core import jobs
from aiapply_lite.models import Job


def test_strip_html() -> None:
    assert jobs._strip_html("<p>Hello &amp; bye</p>") == "Hello & bye"


def test_mock_jobs_filter() -> None:
    results = jobs._fetch_mock("react", limit=10)
    assert all(isinstance(j, Job) for j in results)
    assert any("react" in j.description.lower() or "react" in j.tags for j in results)


def test_dedupe() -> None:
    a = Job("s", "1", "Engineer", "Acme", "Remote", "", "")
    b = Job("s", "2", "engineer", "acme", "Remote", "", "")
    c = Job("s", "3", "Manager", "Acme", "Remote", "", "")
    unique = jobs._dedupe([a, b, c])
    assert len(unique) == 2


def test_fetch_jobs_falls_back_to_mock(monkeypatch) -> None:
    # No sources configured -> should still return mock data, never empty.
    result = jobs.fetch_jobs("python", limit=5, sources=[])
    assert len(result) > 0
