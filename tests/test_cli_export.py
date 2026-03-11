from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

from dr_wandb.cli.export import bootstrap_export_main, export_main
from dr_wandb.sync_types import BootstrapSummary, ExportSummary, FetchMode, RefreshScope


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
    assert cfg.fetch_mode == FetchMode.INCREMENTAL
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


def test_export_main_supports_full_reconcile_fetch_mode(tmp_path: Path):
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
                    "--fetch-mode",
                    "full_reconcile",
                ]
            )

    assert rc == 0
    cfg = mock_engine.export_project.call_args.args[0]
    assert cfg.fetch_mode == FetchMode.FULL_RECONCILE


def test_export_main_passes_refresh_scope(tmp_path: Path):
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
                    "--refresh-scope",
                    "unfinished_or_missing_eval",
                ]
            )

    assert rc == 0
    cfg = mock_engine.export_project.call_args.args[0]
    assert cfg.refresh_scope == RefreshScope.UNFINISHED_OR_MISSING_EVAL


def test_export_main_passes_policy_kwargs(tmp_path: Path):
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
                    "--policy-module",
                    "example.policy",
                    "--policy-class",
                    "ExamplePolicy",
                    "--policy-kwargs-json",
                    '{"revisit_missing_eval": true}',
                ]
            )

    assert rc == 0
    mock_load_policy.assert_called_once_with(
        "example.policy",
        "ExamplePolicy",
        policy_kwargs={"revisit_missing_eval": True},
    )
    cfg = mock_engine.export_project.call_args.args[0]
    assert cfg.policy_kwargs == {"revisit_missing_eval": True}


def test_bootstrap_export_main_passes_config_to_engine(tmp_path: Path):
    summary = BootstrapSummary(
        entity="entity",
        project="project",
        source_dir=str(tmp_path / "source"),
        output_dir=str(tmp_path / "out"),
        state_path=str(tmp_path / "state.json"),
        output_format="parquet",
        runs_output_path=str(tmp_path / "out" / "runs.parquet"),
        history_output_path=str(tmp_path / "out" / "history.parquet"),
        manifest_output_path=str(tmp_path / "out" / "manifest.json"),
        checkpoint_manifest_path=str(tmp_path / "out" / "_checkpoints" / "manifest.json"),
        run_count=1,
        history_count=2,
        checkpoint_count=1,
        bootstrapped_at="2026-03-10T12:00:00+00:00",
    )

    with patch("dr_wandb.cli.export.load_policy") as mock_load_policy:
        with patch("dr_wandb.cli.export.SyncEngine") as mock_engine_cls:
            mock_load_policy.return_value = Mock()
            mock_engine = Mock()
            mock_engine.bootstrap_export.return_value = summary
            mock_engine_cls.return_value = mock_engine

            rc = bootstrap_export_main(
                [
                    "entity",
                    "project",
                    str(tmp_path / "source"),
                    str(tmp_path / "out"),
                    "--output-format",
                    "jsonl",
                ]
            )

    assert rc == 0
    cfg = mock_engine.bootstrap_export.call_args.args[0]
    assert cfg.entity == "entity"
    assert cfg.project == "project"
    assert cfg.source_dir == tmp_path / "source"
    assert cfg.output_dir == tmp_path / "out"
    assert cfg.output_format == "jsonl"
    assert cfg.overwrite_output is False


def test_bootstrap_export_main_supports_overwrite_output(tmp_path: Path):
    summary = BootstrapSummary(
        entity="entity",
        project="project",
        source_dir=str(tmp_path / "source"),
        output_dir=str(tmp_path / "out"),
        state_path=str(tmp_path / "state.json"),
        output_format="parquet",
        runs_output_path=str(tmp_path / "out" / "runs.parquet"),
        history_output_path=str(tmp_path / "out" / "history.parquet"),
        manifest_output_path=str(tmp_path / "out" / "manifest.json"),
        checkpoint_manifest_path=str(tmp_path / "out" / "_checkpoints" / "manifest.json"),
        run_count=1,
        history_count=2,
        checkpoint_count=1,
        bootstrapped_at="2026-03-10T12:00:00+00:00",
    )

    with patch("dr_wandb.cli.export.load_policy") as mock_load_policy:
        with patch("dr_wandb.cli.export.SyncEngine") as mock_engine_cls:
            mock_load_policy.return_value = Mock()
            mock_engine = Mock()
            mock_engine.bootstrap_export.return_value = summary
            mock_engine_cls.return_value = mock_engine

            rc = bootstrap_export_main(
                [
                    "entity",
                    "project",
                    str(tmp_path / "source"),
                    str(tmp_path / "out"),
                    "--overwrite-output",
                ]
            )

    assert rc == 0
    cfg = mock_engine.bootstrap_export.call_args.args[0]
    assert cfg.overwrite_output is True


def test_export_main_profile_mode_derives_paths_and_writes_summary(tmp_path: Path):
    summary = ExportSummary(
        entity="entity",
        project="project",
        state_path=str(tmp_path / ".sync" / "demo__state.json"),
        output_format="parquet",
        runs_output_path=str(tmp_path / "demo" / "runs.parquet"),
        history_output_path=str(tmp_path / "demo" / "history.parquet"),
        manifest_output_path=str(tmp_path / "demo" / "manifest.json"),
        run_count=3,
        history_count=4,
        exported_at="2026-03-11T12:00:00+00:00",
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
                    "--export-name",
                    "demo",
                    "--data-root",
                    str(tmp_path),
                ]
            )

    assert rc == 0
    cfg = mock_engine.export_project.call_args.args[0]
    assert cfg.output_dir == tmp_path / "demo"
    assert cfg.state_path == tmp_path / ".sync" / "demo__state.json"
    payload = json.loads(
        (tmp_path / ".sync" / "demo__last_export_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["run_count"] == 3


def test_export_main_profile_mode_rejects_explicit_paths(tmp_path: Path):
    with patch("dr_wandb.cli.export.load_policy"):
        with patch("dr_wandb.cli.export.SyncEngine"):
            try:
                export_main(
                    [
                        "entity",
                        "project",
                        str(tmp_path / "out"),
                        "--export-name",
                        "demo",
                        "--data-root",
                        str(tmp_path),
                    ]
                )
            except ValueError as exc:
                assert "Profile mode cannot be combined" in str(exc)
            else:
                raise AssertionError("expected ValueError")


def test_bootstrap_export_main_profile_mode_archives_existing_paths(tmp_path: Path):
    canonical_output_dir = tmp_path / "demo"
    canonical_output_dir.mkdir()
    (canonical_output_dir / "placeholder.txt").write_text("old", encoding="utf-8")

    sync_dir = tmp_path / ".sync"
    sync_dir.mkdir()
    state_path = sync_dir / "demo__state.json"
    state_path.write_text("{}", encoding="utf-8")
    export_summary_path = sync_dir / "demo__last_export_summary.json"
    export_summary_path.write_text("{}", encoding="utf-8")
    bootstrap_summary_path = sync_dir / "demo__last_bootstrap_summary.json"
    bootstrap_summary_path.write_text("{}", encoding="utf-8")

    summary = BootstrapSummary(
        entity="entity",
        project="project",
        source_dir=str(tmp_path / "_archive" / "demo__20260311T120000Z"),
        output_dir=str(canonical_output_dir),
        state_path=str(state_path),
        output_format="parquet",
        runs_output_path=str(canonical_output_dir / "runs.parquet"),
        history_output_path=str(canonical_output_dir / "history.parquet"),
        manifest_output_path=str(canonical_output_dir / "manifest.json"),
        checkpoint_manifest_path=str(canonical_output_dir / "_checkpoints" / "manifest.json"),
        run_count=1,
        history_count=2,
        checkpoint_count=1,
        bootstrapped_at="2026-03-10T12:00:00+00:00",
    )

    with patch("dr_wandb.cli.export.load_policy") as mock_load_policy:
        with patch("dr_wandb.cli.export.SyncEngine") as mock_engine_cls:
            mock_load_policy.return_value = Mock()
            mock_engine = Mock()
            mock_engine.bootstrap_export.return_value = summary
            mock_engine_cls.return_value = mock_engine

            rc = bootstrap_export_main(
                [
                    "entity",
                    "project",
                    "--export-name",
                    "demo",
                    "--data-root",
                    str(tmp_path),
                ]
            )

    assert rc == 0
    cfg = mock_engine.bootstrap_export.call_args.args[0]
    assert cfg.output_dir == canonical_output_dir
    assert cfg.state_path == state_path
    assert cfg.source_dir.parent == tmp_path / "_archive"
    assert cfg.source_dir.name.startswith("demo__")
    assert not canonical_output_dir.exists()
    assert cfg.source_dir.exists()
    assert (cfg.source_dir / "placeholder.txt").read_text(encoding="utf-8") == "old"
    assert not state_path.exists()
    archived_sync_dir = sync_dir / "_archive"
    archived_files = sorted(path.name for path in archived_sync_dir.iterdir())
    assert any(name.startswith("demo__state__") for name in archived_files)
    assert any(name.startswith("demo__last_export_summary__") for name in archived_files)
    assert any(name.startswith("demo__last_bootstrap_summary__") for name in archived_files)
    payload = json.loads(bootstrap_summary_path.read_text(encoding="utf-8"))
    assert payload["checkpoint_count"] == 1
