from __future__ import annotations

from pathlib import Path

from dr_wandb.export.models import ExportManifest, ExportMode, ExportState
from dr_wandb.export.export_paths import ExportPaths


def test_export_paths_from_name_and_root(tmp_path: Path) -> None:
    paths = ExportPaths.from_name_and_root("moe_runs", tmp_path)

    assert paths.name == "moe_runs"
    assert paths.data_root == tmp_path
    assert paths.export_dir == tmp_path / "moe_runs"
    assert paths.manifest_path == tmp_path / "moe_runs" / "manifest.json"
    assert paths.state_path == tmp_path / "moe_runs" / "state.json"


def test_export_paths_build_run_and_history_paths(tmp_path: Path) -> None:
    paths = ExportPaths.from_name_and_root("moe_runs", tmp_path)

    assert paths.runs_path("jsonl") == tmp_path / "moe_runs" / "runs.jsonl"
    assert (
        paths.history_path("parquet")
        == tmp_path / "moe_runs" / "history.parquet"
    )


def test_export_paths_load_and_save_state(tmp_path: Path) -> None:
    paths = ExportPaths.from_name_and_root("moe_runs", tmp_path)
    state = ExportState(name="moe_runs", entity="ml-moe", project="moe")

    paths.save_state(state)

    loaded = paths.load_state(entity="ml-moe", project="moe")
    assert loaded == state


def test_export_paths_load_and_save_manifest(tmp_path: Path) -> None:
    paths = ExportPaths.from_name_and_root("moe_runs", tmp_path)
    manifest = ExportManifest(
        name="moe_runs",
        entity="ml-moe",
        project="moe",
        mode=ExportMode.METADATA,
        output_format="jsonl",
        created_at="2024-01-01T00:00:00+00:00",
        updated_at="2024-01-01T00:00:00+00:00",
        runs_path=str(paths.runs_path("jsonl")),
        history_path=None,
        run_count=1,
        history_count=0,
        selected_history_keys=None,
        history_window=None,
    )

    paths.save_manifest(manifest)

    loaded = paths.load_manifest()
    assert loaded == manifest
