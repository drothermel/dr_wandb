from __future__ import annotations

import json
from pathlib import Path

import typer

from dr_wandb.export.engine import ExportEngine
from dr_wandb.export.export_modes import ExportMode, FetchMode
from dr_wandb.export.export_request import ExportRequest
from dr_wandb.export.policy import HistoryWindow, StaticHistoryPolicy


def export_command(
    entity: str,
    project: str,
    name: str = typer.Option(..., "--name"),
    data_root: Path = typer.Option(Path("./data"), "--data-root"),
    mode: ExportMode = typer.Option(ExportMode.METADATA, "--mode"),
    fetch_mode: FetchMode = typer.Option(
        FetchMode.INCREMENTAL, "--fetch-mode"
    ),
    runs_per_page: int = typer.Option(500, "--runs-per-page"),
    timeout_seconds: int = typer.Option(120, "--timeout-seconds"),
    include_metadata: bool = typer.Option(
        False, "--include-metadata/--no-include-metadata"
    ),
    history_key: list[str] | None = typer.Option(None, "--history-key"),
    min_step: int | None = typer.Option(None, "--min-step"),
    max_step: int | None = typer.Option(None, "--max-step"),
    max_records: int | None = typer.Option(None, "--max-records"),
) -> None:
    has_history_selection = any(
        value is not None and value != []
        for value in [history_key, min_step, max_step, max_records]
    )
    assert mode == ExportMode.HISTORY or not has_history_selection, (
        "History selection flags require --mode history"
    )

    history_policy = None
    if mode == ExportMode.HISTORY and has_history_selection:
        history_policy = StaticHistoryPolicy(
            keys=list(history_key) if history_key is not None else None,
            window=HistoryWindow(
                min_step=min_step,
                max_step=max_step,
                max_records=max_records,
            ),
        )

    summary = ExportEngine(
        ExportRequest(
            entity=entity,
            project=project,
            name=name,
            data_root=data_root,
            mode=mode,
            fetch_mode=fetch_mode,
            runs_per_page=runs_per_page,
            timeout_seconds=timeout_seconds,
            include_metadata=include_metadata,
            history_policy=history_policy,
        )
    ).export()
    typer.echo(
        json.dumps(summary.model_dump(mode="python"), indent=2, sort_keys=True)
    )


def main() -> None:
    typer.run(export_command)


if __name__ == "__main__":
    main()
