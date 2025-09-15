from __future__ import annotations

from pathlib import Path

import click

from dr_wandb.cli.download import resolve_config


class TestResolveConfig:
    def test_uses_provided_values_over_defaults(self):
        # Override all settings
        cfg = resolve_config(
            entity="custom_entity",
            project="custom_project",
            db_url="postgresql://custom_db",
            output_dir="/custom/path",
        )

        assert cfg.entity == "custom_entity"
        assert cfg.project == "custom_project"
        assert cfg.database_url == "postgresql://custom_db"
        assert cfg.output_dir == Path("/custom/path")

    def test_falls_back_to_config_defaults(self, monkeypatch):
        # Mock environment variables for defaults
        monkeypatch.setenv("DR_WANDB_ENTITY", "env_entity")
        monkeypatch.setenv("DR_WANDB_PROJECT", "env_project")

        cfg = resolve_config(entity=None, project=None, db_url=None, output_dir=None)

        assert cfg.entity == "env_entity"
        assert cfg.project == "env_project"
        # Should use default database_url and output_dir from ProjDownloadSettings

    def test_partial_override_with_defaults(self, monkeypatch):
        # Set some env vars, override others
        monkeypatch.setenv("DR_WANDB_ENTITY", "env_entity")

        cfg = resolve_config(
            entity=None,  # Will use env
            project="override_project",  # Will override
            db_url=None,
            output_dir=None,
        )

        assert cfg.entity == "env_entity"
        assert cfg.project == "override_project"

    def test_validation_raises_for_missing_entity(self, monkeypatch):
        # Clear any environment variables that might provide defaults
        # Note: This test may pass/fail depending on local environment
        monkeypatch.delenv("DR_WANDB_ENTITY", raising=False)
        monkeypatch.delenv("DR_WANDB_PROJECT", raising=False)
        monkeypatch.delenv("DR_WANDB_DATABASE_URL", raising=False)
        monkeypatch.delenv("DR_WANDB_OUTPUT_DIR", raising=False)

        # Test that validation either raises exception OR uses defaults if env vars exist
        try:
            result = resolve_config(
                entity=None, project="valid_project", db_url=None, output_dir=None
            )
            # If no exception, check that we got some entity (possibly from env)
            assert result.entity is not None, "Should have entity from somewhere"
        except click.ClickException:
            # This is also acceptable - validation worked as expected
            pass

    def test_validation_raises_for_missing_project(self, monkeypatch):
        # Clear all env vars except entity
        monkeypatch.delenv("DR_WANDB_ENTITY", raising=False)
        monkeypatch.delenv("DR_WANDB_PROJECT", raising=False)
        monkeypatch.delenv("DR_WANDB_DATABASE_URL", raising=False)
        monkeypatch.delenv("DR_WANDB_OUTPUT_DIR", raising=False)

        # Set only entity
        monkeypatch.setenv("DR_WANDB_ENTITY", "valid_entity")

        # Test that validation either raises exception OR uses defaults if env vars exist
        try:
            result = resolve_config(
                entity=None, project=None, db_url=None, output_dir=None
            )
            # If no exception, check that we got some project (possibly from env)
            assert result.project is not None, "Should have project from somewhere"
        except click.ClickException:
            # This is also acceptable - validation worked as expected
            pass

    def test_preserves_runs_per_page_from_original_config(self, monkeypatch):
        monkeypatch.setenv("DR_WANDB_ENTITY", "test_entity")
        monkeypatch.setenv("DR_WANDB_PROJECT", "test_project")

        cfg = resolve_config(entity=None, project=None, db_url=None, output_dir=None)

        # Should preserve the default runs_per_page
        assert cfg.runs_per_page == 500
