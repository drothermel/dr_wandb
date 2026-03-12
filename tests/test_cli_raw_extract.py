from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from dr_wandb.cli.raw_extract import app
from dr_wandb.raw_extract import RawExtractSummary


def test_cli_raw_extract_uses_profile_mode_output_path(tmp_path: Path):
    runner = CliRunner()
    summary = RawExtractSummary(
        entity="entity",
        project="project",
        output_dir=str(tmp_path / "my_export" / "wandb_raw_extract"),
        runs_output_path=str(tmp_path / "my_export" / "wandb_raw_extract" / "runs_raw.jsonl"),
        run_count=3,
        exported_at="2026-03-11T12:00:00+00:00",
        runs_per_page=100,
        include_metadata=False,
        postprocess_dedup=True,
        final_run_count=3,
    )

    with patch("dr_wandb.cli.raw_extract.extract_runs_raw", return_value=summary) as mock_extract:
        result = runner.invoke(
            app,
            [
                "entity",
                "project",
                "--export-name",
                "my export",
                "--data-root",
                str(tmp_path),
            ],
        )

    assert result.exit_code == 0
    cfg = mock_extract.call_args.args[0]
    assert cfg.output_dir == tmp_path / "my_export" / "wandb_raw_extract"
    assert cfg.runs_raw_path == tmp_path / "my_export" / "wandb_raw_extract" / "runs_raw.jsonl"
    assert (
        cfg.runs_raw_deduping_path
        == tmp_path / "my_export" / "wandb_raw_extract" / "runs_raw__deduping.jsonl"
    )
    assert cfg.postprocess_dedup is True
    payload = json.loads(result.stdout)
    assert payload["run_count"] == 3
    assert payload["final_run_count"] == 3


def test_cli_raw_extract_passes_include_metadata_and_timeout(tmp_path: Path):
    runner = CliRunner()
    summary = RawExtractSummary(
        entity="entity",
        project="project",
        output_dir=str(tmp_path / "export" / "wandb_raw_extract"),
        runs_output_path=str(tmp_path / "export" / "wandb_raw_extract" / "runs_raw.jsonl"),
        run_count=0,
        exported_at="2026-03-11T12:00:00+00:00",
        runs_per_page=25,
        include_metadata=True,
        postprocess_dedup=False,
        final_run_count=None,
    )

    with patch("dr_wandb.cli.raw_extract.extract_runs_raw", return_value=summary) as mock_extract:
        result = runner.invoke(
            app,
            [
                "entity",
                "project",
                "--export-name",
                "export",
                "--data-root",
                str(tmp_path),
                "--runs-per-page",
                "25",
                "--timeout-seconds",
                "123",
                "--include-metadata",
                "--no-postprocess-dedup",
            ],
        )

    assert result.exit_code == 0
    cfg = mock_extract.call_args.args[0]
    assert cfg.runs_per_page == 25
    assert cfg.timeout_seconds == 123
    assert cfg.include_metadata is True
    assert cfg.postprocess_dedup is False
