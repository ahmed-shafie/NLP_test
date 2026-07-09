"""Local file storage for profile, jobs, and generated outputs."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from aiapply_lite.config import settings
from aiapply_lite.models import CVProfile


def _data_dir() -> Path:
    d = settings.data_dir
    d.mkdir(parents=True, exist_ok=True)
    (d / "jobs").mkdir(exist_ok=True)
    (d / "outputs").mkdir(exist_ok=True)
    return d


def profile_path() -> Path:
    return _data_dir() / "profile.yaml"


def save_profile(profile: CVProfile) -> Path:
    path = profile_path()
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(profile.to_dict(), fh, sort_keys=False, allow_unicode=True)
    return path


def load_profile() -> CVProfile | None:
    path = profile_path()
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        return None
    return CVProfile.from_dict(data)


def save_output(name: str, content: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    path = _data_dir() / "outputs" / safe
    path.write_text(content, encoding="utf-8")
    return path


def save_json(subdir: str, name: str, data: dict[str, object]) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    path = _data_dir() / subdir / f"{safe}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
