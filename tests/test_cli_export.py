from __future__ import annotations

import json
from pathlib import Path

from dr_wandb.cli.export import export_command
from dr_wandb.export.models import ExportMode, ExportSummary, FetchMode


def test_cli_export_builds_history_request(monkeypatch, capsys, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_export(self, request):  # type: ignore[no-untyped-def]
        captured["request"] = request
        return ExportSummary(
            name=request.name,
            entity=request.entity,
            project=request.project,
            mode=request.mode,
            fetch_mode=request.fetch_mode,
            output_dir=str(tmp_path / request.name),
            state_path=str(tmp_path / request.name / "state.json"),
            manifest_path=str(tmp_path / request.name / "manifest.json"),
            runs_path=str(tmp_path / request.name / "runs.jsonl"),
            history_path=str(tmp_path / request.name / "history.jsonl"),
            run_count=2,
            history_count=5,
            exported_at="2024-01-01T00:00:00+00:00",
        )

    monkeypatch.setattr("dr_wandb.export.engine.ExportEngine.export", fake_export)

    export_command(
        "ml-moe",
        "moe",
        name="moe_history",
        data_root=tmp_path,
        mode=ExportMode.HISTORY,
        output_format="jsonl",
        fetch_mode=FetchMode.INCREMENTAL,
        history_key=["eval/loss"],
        min_step=100,
        max_step=None,
        max_records=50,
    )

    request = captured["request"]
    assert request is not None
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert payload["name"] == "moe_history"
    assert payload["history_count"] == 5
