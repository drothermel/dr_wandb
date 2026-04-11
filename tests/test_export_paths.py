from __future__ import annotations

from pathlib import Path

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
