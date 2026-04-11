from __future__ import annotations

from pathlib import Path
from typing import Any

import wandb

from dr_wandb import (
    ExportEngine,
    ExportMode,
    ExportRequest,
    FetchMode,
    RecordStore,
)

from tests.helpers import FakeApi, history_run, metadata_run


def test_metadata_export_writes_named_store(
    tmp_path: Path, monkeypatch: Any
) -> None:
    runs = [
        metadata_run(
            "run-1",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:10:00+00:00",
            state="finished",
        )
    ]
    monkeypatch.setattr(wandb, "Api", lambda **kwargs: FakeApi(runs))
    engine = ExportEngine()
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
    store = RecordStore.from_name_and_root("moe_runs", tmp_path)
    manifest = store.load_manifest()
    assert manifest is not None
    assert manifest.history_path is None
    snapshots = store.load_run_snapshots()
    assert len(snapshots) == 1
    assert snapshots[0].run.config["lr"] == 0.001


def test_history_export_incremental_merges_rows(
    tmp_path: Path, monkeypatch: Any
) -> None:
    first_runs = [
        history_run(
            "run-1",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:10:00+00:00",
            state="running",
            steps=[1, 2],
        )
    ]
    monkeypatch.setattr(wandb, "Api", lambda **kwargs: FakeApi(first_runs))
    engine = ExportEngine()
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
    monkeypatch.setattr(wandb, "Api", lambda **kwargs: FakeApi(second_runs))
    engine = ExportEngine()
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
    store = RecordStore.from_name_and_root("moe_history", tmp_path)
    rows = list(store.iter_history_rows())
    assert [row.step for row in rows] == [1, 2, 3]
