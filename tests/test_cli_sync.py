from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

from dr_wandb.cli.sync import inspect_state_main
from dr_wandb.sync_types import StateInspectionSummary


def test_inspect_state_main_passes_args_to_engine(tmp_path: Path):
    summary = StateInspectionSummary(
        entity="entity",
        project="project",
        state_path=str(tmp_path / "state.json"),
        tracked_runs=3,
        terminal_count=1,
        ignore_count=1,
        non_terminal_count=1,
        status_counts={"finished": 1, "ignore": 1, "unfinished": 1},
        selected_view="non_terminal",
    )

    with patch("dr_wandb.cli.sync.SyncEngine") as mock_engine_cls:
        mock_engine = Mock()
        mock_engine.inspect_state.return_value = summary
        mock_engine_cls.return_value = mock_engine

        rc = inspect_state_main(
            [
                "entity",
                "project",
                "--state-path",
                str(tmp_path / "state.json"),
                "--show-runs",
                "non_terminal",
                "--limit",
                "5",
            ]
        )

    assert rc == 0
    kwargs = mock_engine.inspect_state.call_args.kwargs
    assert kwargs["entity"] == "entity"
    assert kwargs["project"] == "project"
    assert kwargs["state_path"] == tmp_path / "state.json"
    assert kwargs["show_runs"] == "non_terminal"
    assert kwargs["limit"] == 5


def test_inspect_state_main_writes_output_json(tmp_path: Path):
    summary = StateInspectionSummary(
        entity="entity",
        project="project",
        state_path=str(tmp_path / "state.json"),
        tracked_runs=3,
        terminal_count=1,
        ignore_count=1,
        non_terminal_count=1,
        status_counts={"finished": 1, "ignore": 1, "unfinished": 1},
    )
    output_json = tmp_path / "inspection.json"

    with patch("dr_wandb.cli.sync.SyncEngine") as mock_engine_cls:
        mock_engine = Mock()
        mock_engine.inspect_state.return_value = summary
        mock_engine_cls.return_value = mock_engine

        rc = inspect_state_main(
            [
                "entity",
                "project",
                "--output-json",
                str(output_json),
            ]
        )

    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["tracked_runs"] == 3
    assert payload["ignore_count"] == 1
