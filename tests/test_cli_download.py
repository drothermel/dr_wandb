from __future__ import annotations

from dr_wandb.cli.download import ProjDownloadConfig


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
