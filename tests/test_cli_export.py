from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

from dr_wandb.cli.export import export_main
from dr_wandb.sync_types import ExportSummary


def test_export_main_writes_output_json(tmp_path: Path):
    output_json = tmp_path / "summary.json"

    summary = ExportSummary(
        entity="entity",
        project="project",
        state_path=str(tmp_path / "state.json"),
        output_format="jsonl",
        runs_output_path=str(tmp_path / "runs.jsonl"),
        history_output_path=str(tmp_path / "history.jsonl"),
        manifest_output_path=str(tmp_path / "manifest.json"),
        run_count=1,
        history_count=2,
        exported_at="2026-03-05T12:00:00+00:00",
    )

    with patch("dr_wandb.cli.export.load_policy") as mock_load_policy:
        with patch("dr_wandb.cli.export.SyncEngine") as mock_engine_cls:
            mock_load_policy.return_value = Mock()
            mock_engine = Mock()
            mock_engine.export_project.return_value = summary
            mock_engine_cls.return_value = mock_engine

            rc = export_main(
                [
                    "entity",
                    "project",
                    str(tmp_path),
                    "--output-format",
                    "jsonl",
                    "--output-json",
                    str(output_json),
                ]
            )

    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["entity"] == "entity"
    assert payload["run_count"] == 1
    assert payload["history_count"] == 2


def test_export_main_passes_config_to_engine(tmp_path: Path):
    summary = ExportSummary(
        entity="entity",
        project="project",
        state_path=str(tmp_path / "state.json"),
        output_format="parquet",
        runs_output_path=str(tmp_path / "runs.parquet"),
        history_output_path=str(tmp_path / "history.parquet"),
        manifest_output_path=str(tmp_path / "manifest.json"),
        run_count=0,
        history_count=0,
        exported_at="2026-03-05T12:00:00+00:00",
    )

    with patch("dr_wandb.cli.export.load_policy") as mock_load_policy:
        with patch("dr_wandb.cli.export.SyncEngine") as mock_engine_cls:
            mock_load_policy.return_value = Mock()
            mock_engine = Mock()
            mock_engine.export_project.return_value = summary
            mock_engine_cls.return_value = mock_engine

            rc = export_main(
                [
                    "entity",
                    "project",
                    str(tmp_path / "out"),
                    "--runs-per-page",
                    "123",
                    "--save-every",
                    "7",
                    "--checkpoint-every-runs",
                    "11",
                    "--no-finalize-compact",
                ]
            )

    assert rc == 0
    cfg = mock_engine.export_project.call_args.args[0]
    assert cfg.entity == "entity"
    assert cfg.project == "project"
    assert cfg.output_format == "parquet"
    assert cfg.runs_per_page == 123
    assert cfg.save_every == 7
    assert cfg.incremental is True
    assert cfg.checkpoint_every_runs == 11
    assert cfg.finalize_compact is False
    assert cfg.inspection_sample_rows == 5


def test_export_main_supports_no_incremental(tmp_path: Path):
    summary = ExportSummary(
        entity="entity",
        project="project",
        state_path=str(tmp_path / "state.json"),
        output_format="parquet",
        runs_output_path=str(tmp_path / "runs.parquet"),
        history_output_path=str(tmp_path / "history.parquet"),
        manifest_output_path=str(tmp_path / "manifest.json"),
        run_count=0,
        history_count=0,
        exported_at="2026-03-05T12:00:00+00:00",
    )

    with patch("dr_wandb.cli.export.load_policy") as mock_load_policy:
        with patch("dr_wandb.cli.export.SyncEngine") as mock_engine_cls:
            mock_load_policy.return_value = Mock()
            mock_engine = Mock()
            mock_engine.export_project.return_value = summary
            mock_engine_cls.return_value = mock_engine

            rc = export_main(
                [
                    "entity",
                    "project",
                    str(tmp_path / "out"),
                    "--no-incremental",
                ]
            )

    assert rc == 0
    cfg = mock_engine.export_project.call_args.args[0]
    assert cfg.incremental is False
