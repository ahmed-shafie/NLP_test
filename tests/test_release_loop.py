"""The release loop: held-out gating, hard negatives, approval, rollback.

These tests are about the *process*, not the model: a threshold graded by the
rows it was fitted on, a release promoted with a failing gate, or a live version
whose data files silently changed are all ways a measured layer stops being
measured.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.eval.harness import load_gold
from app.eval.splits import (
    Split,
    assign_splits,
    load_hard_negatives,
    load_split,
    money_flow_breaches,
)
from app.release import registry
from app.release.build import (
    RUNTIME_DATA_FILES,
    THRESHOLD_FIELDS,
    build_candidate,
    shipped_files,
    thresholds_snapshot,
)
from app.release.manifest import (
    FileDigest,
    LeakyManifest,
    Manifest,
    Metrics,
    drifted_files,
    sha256_file,
)


@pytest.fixture(autouse=True)
def _isolated_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    from app.config import settings

    monkeypatch.setattr(settings, "release_dir", str(tmp_path / "releases"))
    yield


def _metrics(breaches: int = 0) -> Metrics:
    return Metrics(
        intent_accuracy=1.0,
        rows=121,
        slot_f1={"amount": 1.0},
        over_blocks=0,
        wrong_flow_starts=0,
        money_flow_breaches=breaches,
    )


def _manifest(
    version: str = "2026.06.1",
    *,
    files: list[FileDigest] | None = None,
    breaches: int = 0,
    calibration_split: str = "dev",
    gate_split: str = "test",
) -> Manifest:
    return Manifest(
        version=version,
        created_at="2026-06-22T21:00:00+00:00",
        manifest_version=1,
        thresholds={"topic_head_threshold": 0.999},
        files=files if files is not None else [],
        metrics=_metrics(breaches),
        calibration_split=calibration_split,
        gate_split=gate_split,
        gold_sha256="0" * 64,
        hard_negatives_sha256="1" * 64,
        hard_negative_count=36,
    )


# ---- the split ------------------------------------------------------------


def test_dev_and_test_splits_partition_the_gold_set_without_overlap() -> None:
    rows = load_gold()
    dev = load_split(Split.DEV)
    test = load_split(Split.TEST)
    assert len(dev) + len(test) == len(rows)
    assert not {row.text for row in dev} & {row.text for row in test}
    # Both halves have to be big enough to mean anything.
    assert len(test) >= 80
    assert len(dev) >= 80


def test_the_split_is_stable_across_runs_and_stratified_by_intent() -> None:
    rows = load_gold()
    assert assign_splits(rows) == assign_splits(list(reversed(rows)))
    # Every intent is represented on the held-out side; otherwise its recall is
    # measured only on rows the thresholds were allowed to see.
    held_out = {row.intent for row in load_split(Split.TEST)}
    assert held_out == {row.intent for row in rows}


def test_adding_a_gold_row_cannot_move_an_existing_row_between_splits() -> None:
    rows = load_gold()
    before = assign_splits(rows)
    # A brand-new row lands wherever its own digest puts it; the rest of the
    # stratum keeps its side, so yesterday's held-out rows stay held out.
    stable = sum(
        1 for text, split in assign_splits(rows[:-3]).items() if before[text] is split
    )
    assert stable >= len(rows[:-3]) - 8


# ---- hard negatives -------------------------------------------------------


def test_no_hard_negative_opens_a_money_flow() -> None:
    # The zero-tolerance property: none of these sentences may make the
    # assistant start asking for an amount or a bill reference.
    assert money_flow_breaches() == []


def test_hard_negatives_are_bilingual_and_documented() -> None:
    negatives = load_hard_negatives()
    assert len(negatives) >= 30
    assert len({negative.language for negative in negatives if negative.language}) == 2
    assert all(negative.reason.strip() for negative in negatives)
    assert len({negative.text for negative in negatives}) == len(negatives)


# ---- the manifest ---------------------------------------------------------


def test_a_manifest_calibrated_on_its_own_gate_split_is_rejected() -> None:
    # The leak in the imported evaluation service: thresholds fitted on the gold
    # rows that then grade them. Such a manifest cannot exist here.
    with pytest.raises(LeakyManifest):
        _manifest(calibration_split="test", gate_split="test")


def test_a_manifest_round_trips_through_json() -> None:
    manifest = _manifest()
    assert Manifest.from_json(manifest.to_json()) == manifest


def test_thresholds_snapshot_names_every_calibrated_setting() -> None:
    snapshot = thresholds_snapshot()
    assert set(snapshot) == set(THRESHOLD_FIELDS)
    assert snapshot["topic_head_threshold"] == pytest.approx(0.999)


def test_drift_is_reported_when_a_pinned_file_changes(tmp_path: Path) -> None:
    data = tmp_path / "app" / "nlu" / "data"
    data.mkdir(parents=True)
    pinned = data / "topic_head.npz"
    pinned.write_bytes(b"weights")
    manifest = _manifest(
        files=[
            FileDigest(
                path="app/nlu/data/topic_head.npz",
                sha256=sha256_file(pinned),
                bytes=pinned.stat().st_size,
            )
        ]
    )
    assert drifted_files(manifest, tmp_path) == []
    pinned.write_bytes(b"other weights")
    assert drifted_files(manifest, tmp_path) == ["CHANGED app/nlu/data/topic_head.npz"]
    pinned.unlink()
    assert drifted_files(manifest, tmp_path) == ["MISSING app/nlu/data/topic_head.npz"]


def test_every_runtime_data_file_the_release_pins_exists_in_the_tree() -> None:
    # A pin that silently drops a file pins nothing: the corpus, the vectors and
    # the trained head all change answers without changing code.
    pinned = {digest.path for digest in shipped_files()}
    assert pinned == set(RUNTIME_DATA_FILES)


# ---- promote / rollback ---------------------------------------------------


def test_promotion_refuses_a_failing_gate_and_an_unnamed_approver() -> None:
    manifest = _manifest()
    registry.store(manifest)
    with pytest.raises(registry.ReleaseRefused):
        registry.promote(
            manifest, approved_by="Ahmed", violations=["intent accuracy 0.81 < 0.95"]
        )
    with pytest.raises(registry.ReleaseRefused):
        registry.promote(manifest, approved_by="   ", violations=[])
    assert registry.current() is None


def test_promotion_refuses_a_hard_negative_breach() -> None:
    manifest = _manifest(breaches=1)
    with pytest.raises(registry.ReleaseRefused, match="money flow"):
        registry.promote(manifest, approved_by="Ahmed", violations=[])


def test_promotion_refuses_when_the_measured_files_changed(tmp_path: Path) -> None:
    pinned = tmp_path / "app" / "nlu" / "data" / "topic_head.npz"
    pinned.parent.mkdir(parents=True)
    pinned.write_bytes(b"weights")
    manifest = _manifest(
        files=[FileDigest(path="app/nlu/data/topic_head.npz", sha256="0" * 64, bytes=7)]
    )
    with pytest.raises(registry.ReleaseRefused, match="changed since the build"):
        registry.promote(manifest, approved_by="Ahmed", violations=[], root=tmp_path)


def test_a_promoted_release_records_who_approved_it() -> None:
    manifest = _manifest()
    live = registry.promote(
        manifest, approved_by="Ahmed N", violations=[], reason="first cut"
    )
    assert live.version == "2026.06.1"
    stored = registry.load("2026.06.1")
    assert stored.approved_by == "Ahmed N"
    assert stored.approved_at is not None
    assert [(e.action, e.version) for e in registry.history()] == [
        ("promote", "2026.06.1")
    ]


def test_rollback_makes_an_earlier_version_live_and_needs_a_reason() -> None:
    registry.promote(_manifest("2026.06.1"), approved_by="Ahmed", violations=[])
    registry.promote(_manifest("2026.06.2"), approved_by="Ahmed", violations=[])
    with pytest.raises(registry.ReleaseRefused, match="reason"):
        registry.rollback("2026.06.1", actor="ops", reason="  ")
    live = registry.rollback("2026.06.1", actor="ops", reason="wrong biller answers")
    assert live.version == "2026.06.1"
    assert live.action == "rollback"
    assert registry.current() == live
    # The log keeps both promotions and the rollback, in order.
    assert [entry.action for entry in registry.history()] == [
        "promote",
        "promote",
        "rollback",
    ]


def test_rollback_to_an_unknown_version_is_refused() -> None:
    with pytest.raises(registry.UnknownRelease):
        registry.rollback("2099.01.1", actor="ops", reason="panic")


def test_versions_are_listed_oldest_first() -> None:
    registry.store(_manifest("2026.06.2"))
    older = Manifest(
        version="2026.06.1",
        created_at="2026-05-01T00:00:00+00:00",
        manifest_version=1,
        thresholds={},
        files=[],
        metrics=_metrics(),
        calibration_split="dev",
        gate_split="test",
        gold_sha256="0" * 64,
        hard_negatives_sha256="1" * 64,
    )
    registry.store(older)
    assert registry.versions() == ["2026.06.1", "2026.06.2"]


# ---- the real build ------------------------------------------------------


def test_building_the_current_tree_gates_on_held_out_rows_and_passes() -> None:
    manifest, violations = build_candidate("test-candidate")
    assert violations == []
    assert manifest.gate_split == Split.TEST.value
    assert manifest.calibration_split == Split.DEV.value
    assert manifest.metrics.rows == len(load_split(Split.TEST))
    assert manifest.metrics.money_flow_breaches == 0
    assert manifest.dev_metrics is not None
    assert manifest.files


# ---- the operations read --------------------------------------------------


def test_the_release_read_is_authenticated_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.config import settings
    from app.main import app

    registry.promote(_manifest(), approved_by="Ahmed N", violations=[])
    client = TestClient(app)

    monkeypatch.setattr(settings, "ops_api_key", None)
    assert client.get("/ops/release").status_code == 503

    monkeypatch.setattr(settings, "ops_api_key", "s3cret")
    assert client.get("/ops/release").status_code == 401
    assert client.get("/ops/release", headers={"x-ops-key": "wrong"}).status_code == 401

    response = client.get("/ops/release", headers={"x-ops-key": "s3cret"})
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == "2026.06.1"
    assert body["approved_by"] == "Ahmed N"
    assert body["gate_split"] == "test"
    assert body["calibration_split"] == "dev"

    history = client.get("/ops/release/history", headers={"x-ops-key": "s3cret"})
    assert [entry["action"] for entry in history.json()] == ["promote"]
