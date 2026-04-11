from __future__ import annotations

from pathlib import Path

from dr_wandb import (
    ExportEngine,
    ExportMode,
    ExportRequest,
    FetchMode,
    iter_history_rows,
    load_manifest,
    load_run_snapshots,
)

from tests.helpers import FakeApi, history_run, metadata_run


def test_metadata_export_writes_named_store(tmp_path: Path) -> None:
    runs = [
        metadata_run(
            "run-1",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:10:00+00:00",
            state="finished",
        )
    ]
    engine = ExportEngine(api_factory=lambda timeout: FakeApi(runs))
    summary = engine.export(
        ExportRequest(
            entity="ml-moe",
            project="moe",
            name="moe_runs",
            data_root=tmp_path,
            mode=ExportMode.METADATA,
            fetch_mode=FetchMode.FULL_RECONCILE,
        )
    )

    assert summary.run_count == 1
    manifest = load_manifest("moe_runs", tmp_path)
    assert manifest.history_path is None
    snapshots = load_run_snapshots("moe_runs", tmp_path)
    assert len(snapshots) == 1
    assert snapshots[0].raw_run["config"]["lr"] == 0.001


def test_history_export_incremental_merges_rows(tmp_path: Path) -> None:
    first_runs = [
        history_run(
            "run-1",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:10:00+00:00",
            state="running",
            steps=[1, 2],
        )
    ]
    engine = ExportEngine(api_factory=lambda timeout: FakeApi(first_runs))
    engine.export(
        ExportRequest(
            entity="ml-moe",
            project="moe",
            name="moe_history",
            data_root=tmp_path,
            mode=ExportMode.HISTORY,
            fetch_mode=FetchMode.FULL_RECONCILE,
        )
    )

    second_runs = [
        history_run(
            "run-1",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:20:00+00:00",
            state="running",
            steps=[1, 2, 3],
        )
    ]
    engine = ExportEngine(api_factory=lambda timeout: FakeApi(second_runs))
    summary = engine.export(
        ExportRequest(
            entity="ml-moe",
            project="moe",
            name="moe_history",
            data_root=tmp_path,
            mode=ExportMode.HISTORY,
            fetch_mode=FetchMode.INCREMENTAL,
        )
    )

    assert summary.history_count == 3
    rows = list(iter_history_rows("moe_history", tmp_path))
    assert [row.step for row in rows] == [1, 2, 3]
