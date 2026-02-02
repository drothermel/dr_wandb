from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
import pytest

from dr_wandb.cli.download import ProjDownloadConfig, download_project


class TestProjDownloadConfigFilenames:
    def test_runs_filename_pickle_format(self):
        cfg = ProjDownloadConfig(
            entity="my-team",
            project="my-project",
            output_format="pkl",
        )
        assert cfg.runs_output_filename == "my-team_my-project_runs.pkl"

    def test_runs_filename_parquet_format(self):
        cfg = ProjDownloadConfig(
            entity="my-team",
            project="my-project",
            output_format="parquet",
        )
        assert cfg.runs_output_filename == "my-team_my-project_runs.parquet"

    def test_histories_filename_pickle_format(self):
        cfg = ProjDownloadConfig(
            entity="my-team",
            project="my-project",
            output_format="pkl",
        )
        assert cfg.histories_output_filename == "my-team_my-project_histories.pkl"

    def test_histories_filename_parquet_format(self):
        cfg = ProjDownloadConfig(
            entity="my-team",
            project="my-project",
            output_format="parquet",
        )
        assert cfg.histories_output_filename == "my-team_my-project_histories.parquet"

    def test_filenames_with_special_characters_in_names(self):
        cfg = ProjDownloadConfig(
            entity="team-name",
            project="project_v2",
            output_format="pkl",
        )
        assert cfg.runs_output_filename == "team-name_project_v2_runs.pkl"
        assert cfg.histories_output_filename == "team-name_project_v2_histories.pkl"


class TestProjDownloadConfigFetchRunsCfg:
    def test_fetch_runs_cfg_includes_history_by_default(self):
        cfg = ProjDownloadConfig(
            entity="my-team",
            project="my-project",
        )
        assert cfg.fetch_runs_cfg["include_history"] is True

    def test_fetch_runs_cfg_excludes_history_when_runs_only(self):
        cfg = ProjDownloadConfig(
            entity="my-team",
            project="my-project",
            runs_only=True,
        )
        assert cfg.fetch_runs_cfg["include_history"] is False

    def test_fetch_runs_cfg_contains_entity_and_project(self):
        cfg = ProjDownloadConfig(
            entity="my-team",
            project="my-project",
        )
        assert cfg.fetch_runs_cfg["entity"] == "my-team"
        assert cfg.fetch_runs_cfg["project"] == "my-project"

    def test_fetch_runs_cfg_contains_runs_per_page(self):
        cfg = ProjDownloadConfig(
            entity="my-team",
            project="my-project",
            runs_per_page=100,
        )
        assert cfg.fetch_runs_cfg["runs_per_page"] == 100


class TestDownloadProjectParquetHistories:
    """Test the incremental parquet saving logic for histories."""

    def test_parquet_saves_histories_incrementally(self):
        """Test that histories from multiple runs are combined correctly."""
        runs = [
            {"run_id": "run_1", "name": "run1"},
            {"run_id": "run_2", "name": "run2"},
        ]
        histories = [
            [{"run_id": "run_1", "step": i, "loss": 0.5 - i * 0.1} for i in range(3)],
            [{"run_id": "run_2", "step": i, "loss": 0.4 - i * 0.1} for i in range(2)],
        ]

        with TemporaryDirectory() as tmpdir:
            with patch("dr_wandb.cli.download.fetch_project_runs") as mock_fetch:
                mock_fetch.return_value = (runs, histories)
                download_project(
                    entity="test_entity",
                    project="test_project",
                    output_dir=tmpdir,
                    output_format="parquet",
                    runs_only=False,
                )

            # Verify histories file was created and contains all entries
            histories_file = Path(tmpdir) / "test_entity_test_project_histories.parquet"
            assert histories_file.exists()

            df = pd.read_parquet(histories_file)
            assert len(df) == 5  # 3 + 2 entries
            assert set(df["run_id"].unique()) == {"run_1", "run_2"}

    def test_parquet_skips_empty_histories(self):
        """Test that empty histories are not saved."""
        runs = [
            {"run_id": "run_1", "name": "run1"},
            {"run_id": "run_2", "name": "run2"},
        ]
        histories = [
            [],  # Empty history
            [{"run_id": "run_2", "step": 0, "loss": 0.5}],
        ]

        with TemporaryDirectory() as tmpdir:
            with patch("dr_wandb.cli.download.fetch_project_runs") as mock_fetch:
                mock_fetch.return_value = (runs, histories)
                download_project(
                    entity="test_entity",
                    project="test_project",
                    output_dir=tmpdir,
                    output_format="parquet",
                    runs_only=False,
                )

            # Verify histories file was created with only non-empty entries
            histories_file = Path(tmpdir) / "test_entity_test_project_histories.parquet"
            assert histories_file.exists()

            df = pd.read_parquet(histories_file)
            assert len(df) == 1
            assert df["run_id"].iloc[0] == "run_2"

    def test_parquet_no_histories_file_when_all_empty(self):
        """Test that no histories file is created when all histories are empty."""
        runs = [{"run_id": "run_1", "name": "run1"}]
        histories = [[]]  # All empty

        with TemporaryDirectory() as tmpdir:
            with patch("dr_wandb.cli.download.fetch_project_runs") as mock_fetch:
                mock_fetch.return_value = (runs, histories)
                download_project(
                    entity="test_entity",
                    project="test_project",
                    output_dir=tmpdir,
                    output_format="parquet",
                    runs_only=False,
                )

            # Verify histories file was NOT created
            histories_file = Path(tmpdir) / "test_entity_test_project_histories.parquet"
            assert not histories_file.exists()
