from __future__ import annotations

from pathlib import Path

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
