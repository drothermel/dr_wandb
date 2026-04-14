from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from dr_wandb import (
    ExportEngine,
    ExportMode,
    ExportRequest,
    ExportStore,
    SyncMode,
)

from tests.helpers import history_run, metadata_run


def test_metadata_export_writes_named_store(
    tmp_path: Path, install_fake_wandb_api
) -> None:
    install_fake_wandb_api(
        [
            metadata_run(
                "run-1",
                created_at="2024-01-01T00:00:00+00:00",
                updated_at="2024-01-01T00:10:00+00:00",
                state="finished",
            )
        ]
    )
    summary = ExportEngine(
        ExportRequest(
            entity="ml-moe",
            project="moe",
            name="moe_runs",
            data_root=tmp_path,
            mode=ExportMode.METADATA,
            sync_mode=SyncMode.FULL_RECONCILE,
        )
    ).export()

    assert summary.run_count == 1
    store = ExportStore(name="moe_runs", data_root=tmp_path)
    manifest = store.load_manifest()
    assert manifest is not None
    assert manifest.history_path is None
    snapshots = store.load_run_snapshots()
    assert len(snapshots) == 1
    assert snapshots[0].run.config["lr"] == 0.001


def test_history_export_merges_existing_and_new_rows(
    tmp_path: Path, install_fake_wandb_api
) -> None:
    install_fake_wandb_api(
        [
            history_run(
                "run-1",
                created_at="2024-01-01T00:00:00+00:00",
                updated_at="2024-01-01T00:10:00+00:00",
                state="running",
                steps=[1, 2],
            )
        ]
    )
    ExportEngine(
        ExportRequest(
            entity="ml-moe",
            project="moe",
            name="moe_history",
            data_root=tmp_path,
            mode=ExportMode.HISTORY,
            sync_mode=SyncMode.FULL_RECONCILE,
        )
    ).export()

    install_fake_wandb_api(
        [
            history_run(
                "run-1",
                created_at="2024-01-01T00:00:00+00:00",
                updated_at="2024-01-01T00:20:00+00:00",
                state="running",
                steps=[1, 2, 3],
            )
        ]
    )
    summary = ExportEngine(
        ExportRequest(
            entity="ml-moe",
            project="moe",
            name="moe_history",
            data_root=tmp_path,
            mode=ExportMode.HISTORY,
            sync_mode=SyncMode.INCREMENTAL,
        )
    ).export()

    assert summary.history_count == 3
    store = ExportStore(name="moe_history", data_root=tmp_path)
    rows = list(store.iter_history_rows())
    assert [row.step for row in rows] == [1, 2, 3]


def test_state_is_not_advanced_when_manifest_write_fails(
    tmp_path: Path,
    install_fake_wandb_api,
    monkeypatch: Any,
) -> None:
    install_fake_wandb_api(
        [
            metadata_run(
                "run-1",
                created_at="2024-01-01T00:00:00+00:00",
                updated_at="2024-01-01T00:10:00+00:00",
                state="finished",
            )
        ]
    )

    from dr_wandb.store import ExportStore as _Store

    original_save_manifest = _Store.save_manifest

    def failing_save_manifest(self, manifest):  # noqa: ARG001
        raise OSError("simulated manifest write failure")

    monkeypatch.setattr(_Store, "save_manifest", failing_save_manifest)

    with pytest.raises(OSError, match="simulated manifest write failure"):
        ExportEngine(
            ExportRequest(
                entity="ml-moe",
                project="moe",
                name="moe_runs",
                data_root=tmp_path,
                mode=ExportMode.METADATA,
                sync_mode=SyncMode.FULL_RECONCILE,
            )
        ).export()

    monkeypatch.setattr(_Store, "save_manifest", original_save_manifest)

    store = ExportStore(name="moe_runs", data_root=tmp_path)
    assert not store.state_path.exists(), (
        "state.json must not be saved when artifact writes fail"
    )
    assert store.load_manifest() is None


def test_export_logs_first_run_and_each_five_percent_bucket(
    tmp_path: Path, install_fake_wandb_api, caplog: pytest.LogCaptureFixture
) -> None:
    install_fake_wandb_api(
        [
            metadata_run(
                f"run-{index}",
                created_at="2024-01-01T00:00:00+00:00",
                updated_at="2024-01-01T00:10:00+00:00",
                state="finished",
            )
            for index in range(1, 41)
        ]
    )

    caplog.set_level(logging.INFO, logger="dr_wandb.engine")

    ExportEngine(
        ExportRequest(
            entity="ml-moe",
            project="moe",
            name="moe_runs",
            data_root=tmp_path,
            mode=ExportMode.METADATA,
            sync_mode=SyncMode.FULL_RECONCILE,
        )
    ).export()

    progress_messages = [
        record.getMessage()
        for record in caplog.records
        if "Processed " in record.getMessage()
    ]
    assert progress_messages == [
        "Processed 1/40 runs (2% complete)",
        "Processed 2/40 runs (5% complete)",
        "Processed 4/40 runs (10% complete)",
        "Processed 6/40 runs (15% complete)",
        "Processed 8/40 runs (20% complete)",
        "Processed 10/40 runs (25% complete)",
        "Processed 12/40 runs (30% complete)",
        "Processed 14/40 runs (35% complete)",
        "Processed 16/40 runs (40% complete)",
        "Processed 18/40 runs (45% complete)",
        "Processed 20/40 runs (50% complete)",
        "Processed 22/40 runs (55% complete)",
        "Processed 24/40 runs (60% complete)",
        "Processed 26/40 runs (65% complete)",
        "Processed 28/40 runs (70% complete)",
        "Processed 30/40 runs (75% complete)",
        "Processed 32/40 runs (80% complete)",
        "Processed 34/40 runs (85% complete)",
        "Processed 36/40 runs (90% complete)",
        "Processed 38/40 runs (95% complete)",
        "Processed 40/40 runs (100% complete)",
    ]
