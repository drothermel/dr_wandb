from __future__ import annotations

from dr_wandb import (
    ExportMode,
    ExportRequest,
    ExportState,
    FetchMode,
    HistoryRow,
    HistoryWindow,
    RecordStore,
    RunSnapshot,
    WandbRun,
)
from dr_wandb.export.export_state import RunTrackingState

from tests.helpers import history_run, metadata_run


def test_wandb_run_from_fake_run() -> None:
    run = metadata_run(
        "run-1",
        created_at="2024-01-01T00:00:00+00:00",
        updated_at="2024-01-01T00:10:00+00:00",
        state="finished",
    )

    wandb_run = WandbRun.from_wandb_run(
        run,
        entity="ml-moe",
        project="moe",
        include_metadata=True,
    )

    assert wandb_run.run_id == "run-1"
    assert wandb_run.name == "run-1"
    assert wandb_run.entity == "ml-moe"
    assert wandb_run.project == "moe"
    assert wandb_run.state == "finished"
    assert wandb_run.created_at == "2024-01-01T00:00:00+00:00"
    assert wandb_run.updated_at == "2024-01-01T00:10:00+00:00"
    assert wandb_run.config == {"lr": 0.001}
    assert wandb_run.summary_metrics == {"loss": 1.23}
    assert wandb_run.tags == ["baseline"]
    assert wandb_run.url == "https://wandb.ai/ml-moe/moe/runs/run-1"


def test_run_tracking_state_from_wandb_run() -> None:
    run = metadata_run(
        "run-1",
        created_at="2024-01-01T00:00:00+00:00",
        updated_at="2024-01-01T00:10:00+00:00",
        state="finished",
    )
    wandb_run = WandbRun.from_wandb_run(
        run,
        entity="ml-moe",
        project="moe",
        include_metadata=False,
    )

    tracking = RunTrackingState.from_wandb_run(wandb_run, last_history_step=12)

    assert tracking.run_id == "run-1"
    assert tracking.created_at == "2024-01-01T00:00:00+00:00"
    assert tracking.updated_at == "2024-01-01T00:10:00+00:00"
    assert tracking.run_state == "finished"
    assert tracking.last_history_step == 12


def test_history_row_from_history_entry() -> None:
    run = history_run(
        "run-1",
        created_at="2024-01-01T00:00:00+00:00",
        updated_at="2024-01-01T00:10:00+00:00",
        state="running",
        steps=[1],
    )

    row = HistoryRow.from_history_entry(
        run_id="run-1", entry=run.scan_history()[0]
    )

    assert row.run_id == "run-1"
    assert row.step == 1
    assert row.timestamp == "2024-01-01T00:00:01+00:00"
    assert row.runtime == 10
    assert row.metrics == {"eval/loss": 1.0}
    assert row.wandb_metadata == {"runtime": 10}


def test_run_snapshot_sort_key() -> None:
    snapshot = RunSnapshot(
        run=WandbRun(
            run_id="run-2",
            name="run-2",
            entity="ml-moe",
            project="moe",
            created_at="2024-01-01T00:00:00+00:00",
        ),
        exported_at="2024-01-01T01:00:00+00:00",
    )

    assert snapshot.sort_key == ("2024-01-01T00:00:00+00:00", "run-2")


def test_top_level_reexports_remain_available() -> None:
    request = ExportRequest(entity="e", project="p", name="n")
    state = ExportState(name="n", entity="e", project="p")
    window = HistoryWindow(min_step=1)

    assert request.mode == ExportMode.METADATA
    assert request.fetch_mode == FetchMode.INCREMENTAL
    assert state.name == "n"
    assert window.min_step == 1
    assert RecordStore is not None
