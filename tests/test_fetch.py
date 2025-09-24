from __future__ import annotations

from unittest.mock import patch

from dr_wandb.fetch import fetch_project_runs


def test_fetch_project_runs_with_history(mock_wandb_run, sample_history_entries):
    mock_wandb_run.scan_history.return_value = sample_history_entries

    with patch("dr_wandb.fetch._iterate_runs", return_value=iter([mock_wandb_run])):
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
    progress_calls: list[tuple[int, int, str]] = []

    def progress(idx: int, total: int, name: str) -> None:
        progress_calls.append((idx, total, name))

    with patch("dr_wandb.fetch._iterate_runs", return_value=iter([mock_wandb_run])):
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

