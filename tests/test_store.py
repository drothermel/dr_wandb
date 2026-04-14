from __future__ import annotations

from pathlib import Path

import pytest

from dr_wandb import (
    ExportEngine,
    ExportManifest,
    ExportMode,
    ExportRequest,
    ExportState,
    ExportStore,
    iter_history_rows,
    load_manifest,
    load_run_snapshots,
)

from tests.helpers import FakeUser, metadata_run


def test_store_path_layout(tmp_path: Path) -> None:
    store = ExportStore(name="moe_runs", data_root=tmp_path)

    assert store.export_dir == tmp_path / "moe_runs"
    assert store.manifest_path == tmp_path / "moe_runs" / "manifest.json"
    assert store.state_path == tmp_path / "moe_runs" / "state.json"
    assert store.runs_path == tmp_path / "moe_runs" / "runs.jsonl"
    assert store.history_path == tmp_path / "moe_runs" / "history.jsonl"


def test_store_load_and_save_state(tmp_path: Path) -> None:
    store = ExportStore(name="moe_runs", data_root=tmp_path)
    state = ExportState(name="moe_runs", entity="ml-moe", project="moe")

    store.save_state(state)

    loaded = store.load_state(entity="ml-moe", project="moe")
    assert loaded == state


def test_store_load_state_rejects_mismatched_identity(tmp_path: Path) -> None:
    store = ExportStore(name="moe_runs", data_root=tmp_path)
    other = ExportState(name="other_runs", entity="ml-moe", project="other")

    store.save_state(other)

    with pytest.raises(ValueError, match="State identity does not match"):
        store.load_state(entity="ml-moe", project="moe")


def test_store_load_and_save_manifest(tmp_path: Path) -> None:
    store = ExportStore(name="moe_runs", data_root=tmp_path)
    manifest = ExportManifest(
        name="moe_runs",
        entity="ml-moe",
        project="moe",
        mode=ExportMode.METADATA,
        created_at="2024-01-01T00:00:00+00:00",
        updated_at="2024-01-01T00:00:00+00:00",
        runs_path=str(store.runs_path),
        history_path=None,
        run_count=1,
        history_count=0,
        selected_history_keys=None,
        history_window=None,
    )

    store.save_manifest(manifest)

    loaded = store.load_manifest()
    assert loaded == manifest


def test_store_round_trips_wandb_run(
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
    ExportEngine(
        ExportRequest(
            entity="ml-moe",
            project="moe",
            name="moe_runs",
            data_root=tmp_path,
            mode=ExportMode.METADATA,
        )
    ).export()

    store = ExportStore(name="moe_runs", data_root=tmp_path)
    snapshots = store.load_run_snapshots()
    assert snapshots[0].run.summary_metrics["loss"] == 1.23


def test_store_round_trips_non_json_user_objects(
    tmp_path: Path, install_fake_wandb_api
) -> None:
    install_fake_wandb_api(
        [
            metadata_run(
                "run-1",
                created_at="2024-01-01T00:00:00+00:00",
                updated_at="2024-01-01T00:10:00+00:00",
                state="finished",
                user=FakeUser(user_id="user-1", username="danielle"),
            )
        ]
    )
    ExportEngine(
        ExportRequest(
            entity="ml-moe",
            project="moe",
            name="moe_runs",
            data_root=tmp_path,
            mode=ExportMode.METADATA,
        )
    ).export()

    store = ExportStore(name="moe_runs", data_root=tmp_path)
    snapshots = store.load_run_snapshots()
    assert snapshots[0].run.user == {
        "id": "user-1",
        "username": "danielle",
    }


def test_top_level_loader_helpers(
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
    ExportEngine(
        ExportRequest(
            entity="ml-moe",
            project="moe",
            name="moe_runs",
            data_root=tmp_path,
            mode=ExportMode.METADATA,
        )
    ).export()

    manifest = load_manifest("moe_runs", tmp_path)
    assert manifest is not None
    assert manifest.run_count == 1

    snapshots = load_run_snapshots("moe_runs", tmp_path)
    assert len(snapshots) == 1

    rows = list(iter_history_rows("moe_runs", tmp_path))
    assert rows == []
