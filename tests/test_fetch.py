from __future__ import annotations

from unittest.mock import Mock, patch

from dr_wandb.fetch import fetch_project_runs


class MockRuns:
    """Mock Runs object that is iterable and has a total attribute."""

    _SENTINEL = object()

    def __init__(self, runs: list, total: int | None = _SENTINEL):
        self._runs = runs
        if total is MockRuns._SENTINEL:
            # No argument provided, default to len(runs)
            self.total = len(runs)
        else:
            # Explicit value provided (including None), preserve it
            self.total = total

    def __iter__(self):
        return iter(self._runs)


def test_fetch_project_runs_with_history(mock_wandb_run, sample_history_entries):
    mock_wandb_run.scan_history.return_value = sample_history_entries

    # Create a mock Runs object that is iterable
    mock_runs_obj = MockRuns([mock_wandb_run], total=1)

    with patch("dr_wandb.fetch.wandb.Api") as mock_api:
        mock_api_instance = Mock()
        mock_api_instance.runs.return_value = mock_runs_obj
        mock_api.return_value = mock_api_instance

        runs, histories = fetch_project_runs(
            "test_entity",
            "test_project",
            runs_per_page=50,
            include_history=True,
            progress_callback=lambda *args: None,
        )

    assert len(runs) == 1
    run_payload = runs[0]
    assert run_payload["run_id"] == mock_wandb_run.id
    assert run_payload["config"]["learning_rate"] == 0.001

    assert len(histories) == 1
    history_entries = histories[0]
    assert history_entries[0]["run_id"] == mock_wandb_run.id
    assert history_entries[0]["metrics"]["loss"] == sample_history_entries[0]["loss"]
    assert history_entries[0]["timestamp"].isoformat().startswith("2024-01-")


def test_fetch_project_runs_without_history(mock_wandb_run):
    progress_calls: list[tuple[int, int | None, str]] = []

    def progress(idx: int, total: int | None, name: str) -> None:
        progress_calls.append((idx, total, name))

    # Create a mock Runs object that is iterable
    mock_runs_obj = MockRuns([mock_wandb_run], total=1)

    with patch("dr_wandb.fetch.wandb.Api") as mock_api:
        mock_api_instance = Mock()
        mock_api_instance.runs.return_value = mock_runs_obj
        mock_api.return_value = mock_api_instance

        runs, histories = fetch_project_runs(
            "test_entity",
            "test_project",
            include_history=False,
            progress_callback=progress,
        )

    assert len(runs) == 1
    assert runs[0]["run_id"] == mock_wandb_run.id
    assert histories == []
    assert progress_calls == [(1, 1, mock_wandb_run.name)]


def test_fetch_project_runs_progress_with_unknown_total(mock_wandb_run):
    """Test that progress callback uses None when total is not available."""
    progress_calls: list[tuple[int, int | None, str]] = []

    def progress(idx: int, total: int | None, name: str) -> None:
        progress_calls.append((idx, total, name))

    # Create a mock Runs object and set total to None to simulate API not providing it
    mock_runs_obj = MockRuns([mock_wandb_run], total=None)

    with patch("dr_wandb.fetch.wandb.Api") as mock_api:
        mock_api_instance = Mock()
        mock_api_instance.runs.return_value = mock_runs_obj
        mock_api.return_value = mock_api_instance

        runs, _histories = fetch_project_runs(
            "test_entity",
            "test_project",
            include_history=False,
            progress_callback=progress,
        )

    assert len(runs) == 1
    # Verify progress callback was called with None for unknown total
    assert progress_calls == [(1, None, mock_wandb_run.name)]
