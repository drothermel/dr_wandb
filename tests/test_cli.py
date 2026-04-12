from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import typer

from dr_wandb import ExportMode, ExportRequest, ExportSummary, SyncMode
from dr_wandb.cli import export_command


class _FakeEngine:
    captured: dict[str, ExportRequest] = {}

    def __init__(self, request: ExportRequest) -> None:
        _FakeEngine.captured["request"] = request
        self._request = request

    def export(self) -> ExportSummary:
        request = self._request
        return ExportSummary(
            name=request.name,
            entity=request.entity,
            project=request.project,
            mode=request.mode,
            sync_mode=request.sync_mode,
            output_dir=f"/tmp/{request.name}",
            state_path=f"/tmp/{request.name}/state.json",
            manifest_path=f"/tmp/{request.name}/manifest.json",
            runs_path=f"/tmp/{request.name}/runs.jsonl",
            history_path=f"/tmp/{request.name}/history.jsonl",
            run_count=2,
            history_count=5,
            exported_at="2024-01-01T00:00:00+00:00",
        )


def test_cli_export_builds_history_request(
    monkeypatch: Any, capsys: Any, tmp_path: Path
) -> None:
    _FakeEngine.captured = {}
    monkeypatch.setattr("dr_wandb.cli.ExportEngine", _FakeEngine)

    export_command(
        "ml-moe",
        "moe",
        name="moe_history",
        data_root=tmp_path,
        mode=ExportMode.HISTORY,
        sync_mode=SyncMode.INCREMENTAL,
        runs_per_page=500,
        timeout_seconds=120,
        include_metadata=False,
        history_key=["eval/loss"],
        min_step=100,
        max_step=None,
        max_records=50,
    )

    request = _FakeEngine.captured["request"]
    assert request.mode == ExportMode.HISTORY
    assert request.sync_mode == SyncMode.INCREMENTAL
    assert request.history_selection is not None
    assert request.history_selection.keys == ["eval/loss"]
    assert request.history_selection.window is not None
    assert request.history_selection.window.min_step == 100
    assert request.history_selection.window.max_records == 50

    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "moe_history"
    assert payload["history_count"] == 5


def test_cli_rejects_history_flags_without_history_mode(
    tmp_path: Path,
) -> None:
    with pytest.raises(typer.BadParameter):
        export_command(
            "ml-moe",
            "moe",
            name="moe_runs",
            data_root=tmp_path,
            mode=ExportMode.METADATA,
            sync_mode=SyncMode.INCREMENTAL,
            runs_per_page=500,
            timeout_seconds=120,
            include_metadata=False,
            history_key=["eval/loss"],
            min_step=None,
            max_step=None,
            max_records=None,
        )
