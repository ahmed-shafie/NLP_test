from pathlib import Path

from aiapply_lite.core import storage
from aiapply_lite.models import CVProfile


def test_profile_roundtrip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(storage.settings, "data_dir", tmp_path)
    profile = CVProfile(
        raw_text="hi",
        name="Ada",
        title="Engineer",
        skills=["python", "sql"],
        years_experience=5.0,
    )
    storage.save_profile(profile)
    loaded = storage.load_profile()
    assert loaded is not None
    assert loaded.name == "Ada"
    assert loaded.skills == ["python", "sql"]
    assert loaded.years_experience == 5.0


def test_load_profile_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(storage.settings, "data_dir", tmp_path)
    assert storage.load_profile() is None


def test_save_output_sanitizes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(storage.settings, "data_dir", tmp_path)
    path = storage.save_output("road map/../x.md", "hello")
    assert path.exists()
    assert path.read_text() == "hello"


def test_cvprofile_from_dict_defaults() -> None:
    p = CVProfile.from_dict({"name": "Bob"})
    assert p.name == "Bob"
    assert p.skills == []
    assert p.years_experience == 0.0
