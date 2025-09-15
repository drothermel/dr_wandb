from __future__ import annotations

from unittest.mock import Mock, patch

from click.testing import CliRunner

from dr_wandb.cli.download import download_project


class TestCLIContract:
    """Test CLI behavior without requiring WandB API or real database."""

    def test_cli_requires_entity_and_project(self):
        """Test that CLI properly validates required arguments."""
        runner = CliRunner()

        # Missing both entity and project (will fail validation)
        result = runner.invoke(download_project, [])
        # CLI should exit with error due to validation failure
        assert result.exit_code != 0

    def test_cli_accepts_valid_arguments(self):
        """Test that CLI accepts valid argument combinations."""
        runner = CliRunner()

        with (
            patch("dr_wandb.cli.download.ProjectStore") as mock_store_class,
            patch("dr_wandb.cli.download.Downloader") as mock_downloader_class,
        ):
            # Mock the store and downloader
            mock_store = Mock()
            mock_downloader = Mock()
            mock_store_class.return_value = mock_store
            mock_downloader_class.return_value = mock_downloader

            # Mock successful download
            mock_stats = Mock()
            mock_stats.__str__ = Mock(return_value="Download complete")
            mock_downloader.download_project.return_value = mock_stats
            mock_downloader.write_downloaded_to_parquet.return_value = None

            # Test with required arguments
            result = runner.invoke(
                download_project,
                ["--entity", "test_entity", "--project", "test_project"],
            )

            # Should not crash
            assert result.exit_code == 0
            assert "Beginning Dr. Wandb" in result.output

            # Verify mocks were called appropriately
            mock_store_class.assert_called_once()
            mock_downloader_class.assert_called_once()
            mock_downloader.download_project.assert_called_once()

    def test_cli_handles_optional_flags(self):
        """Test that CLI handles optional flags without crashing."""
        runner = CliRunner()

        with (
            patch("dr_wandb.cli.download.ProjectStore") as mock_store_class,
            patch("dr_wandb.cli.download.Downloader") as mock_downloader_class,
        ):
            mock_store = Mock()
            mock_downloader = Mock()
            mock_store_class.return_value = mock_store
            mock_downloader_class.return_value = mock_downloader

            mock_stats = Mock()
            mock_stats.__str__ = Mock(return_value="Download complete")
            mock_downloader.download_project.return_value = mock_stats

            # Test with all optional flags
            result = runner.invoke(
                download_project,
                [
                    "--entity",
                    "test_entity",
                    "--project",
                    "test_project",
                    "--runs-only",
                    "--force-refresh",
                    "--db-url",
                    "sqlite:///:memory:",
                    "--output-dir",
                    "/tmp/test",
                ],
            )

            assert result.exit_code == 0

            # Verify download was called with correct flags
            call_args = mock_downloader.download_project.call_args
            assert call_args[1]["runs_only"] is True
            assert call_args[1]["force_refresh"] is True

    def test_cli_uses_environment_variables(self, monkeypatch):
        """Test that CLI respects environment variables."""
        runner = CliRunner()

        # Set environment variables
        monkeypatch.setenv("DR_WANDB_ENTITY", "env_entity")
        monkeypatch.setenv("DR_WANDB_PROJECT", "env_project")

        with (
            patch("dr_wandb.cli.download.ProjectStore") as mock_store_class,
            patch("dr_wandb.cli.download.Downloader") as mock_downloader_class,
        ):
            mock_store = Mock()
            mock_downloader = Mock()
            mock_store_class.return_value = mock_store
            mock_downloader_class.return_value = mock_downloader

            mock_stats = Mock()
            mock_stats.__str__ = Mock(return_value="Download complete")
            mock_downloader.download_project.return_value = mock_stats

            # Don't provide entity/project as CLI args
            result = runner.invoke(download_project, [])

            assert result.exit_code == 0

            # Verify the environment values were used
            call_args = mock_downloader.download_project.call_args
            assert call_args[1]["entity"] == "env_entity"
            assert call_args[1]["project"] == "env_project"

    def test_cli_shows_configuration_info(self):
        """Test that CLI displays configuration information."""
        runner = CliRunner()

        with (
            patch("dr_wandb.cli.download.ProjectStore") as mock_store_class,
            patch("dr_wandb.cli.download.Downloader") as mock_downloader_class,
        ):
            mock_store = Mock()
            mock_downloader = Mock()
            mock_store_class.return_value = mock_store
            mock_downloader_class.return_value = mock_downloader

            mock_stats = Mock()
            mock_stats.__str__ = Mock(return_value="Stats: 5 runs downloaded")
            mock_downloader.download_project.return_value = mock_stats

            result = runner.invoke(
                download_project,
                ["--entity", "test_entity", "--project", "test_project"],
            )

            # Should show configuration info
            assert "test_entity/test_project" in result.output
            assert "Database:" in result.output
            assert "Output directory:" in result.output
            assert "Force refresh:" in result.output

    def test_cli_displays_stats_output(self):
        """Test that CLI displays download statistics."""
        runner = CliRunner()

        with (
            patch("dr_wandb.cli.download.ProjectStore") as mock_store_class,
            patch("dr_wandb.cli.download.Downloader") as mock_downloader_class,
        ):
            mock_store = Mock()
            mock_downloader = Mock()
            mock_store_class.return_value = mock_store
            mock_downloader_class.return_value = mock_downloader

            # Create realistic stats output
            mock_stats = Mock()
            mock_stats.__str__ = Mock(
                return_value="""
:: Downloader Stats ::
 - # WandB runs: 100
 - # Stored runs: 95
 - # New runs: 5
 - # Updated runs: 0
"""
            )
            mock_downloader.download_project.return_value = mock_stats

            result = runner.invoke(
                download_project,
                ["--entity", "test_entity", "--project", "test_project"],
            )

            # Should display the stats
            assert "Downloader Stats" in result.output
            assert "WandB runs: 100" in result.output


class TestCLIErrorHandling:
    """Test CLI error handling behavior."""

    def test_cli_handles_store_initialization_errors(self):
        """Test CLI behavior when ProjectStore initialization fails."""
        runner = CliRunner()

        with patch("dr_wandb.cli.download.ProjectStore") as mock_store_class:
            # Make ProjectStore raise an exception
            mock_store_class.side_effect = Exception("Database connection failed")

            result = runner.invoke(
                download_project,
                ["--entity", "test_entity", "--project", "test_project"],
            )

            # CLI should not crash silently
            assert result.exit_code != 0

    def test_cli_handles_downloader_errors(self):
        """Test CLI behavior when download operation fails."""
        runner = CliRunner()

        with (
            patch("dr_wandb.cli.download.ProjectStore") as mock_store_class,
            patch("dr_wandb.cli.download.Downloader") as mock_downloader_class,
        ):
            mock_store = Mock()
            mock_downloader = Mock()
            mock_store_class.return_value = mock_store
            mock_downloader_class.return_value = mock_downloader

            # Make download operation fail
            mock_downloader.download_project.side_effect = Exception("WandB API error")

            result = runner.invoke(
                download_project,
                ["--entity", "test_entity", "--project", "test_project"],
            )

            # CLI should not crash silently
            assert result.exit_code != 0
