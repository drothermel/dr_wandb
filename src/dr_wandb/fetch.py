from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import Any

import wandb

from dr_wandb.history_entry_record import HistoryEntryRecord
from dr_wandb.run_record import RunRecord
from dr_wandb.utils import default_progress_callback

ProgressFn = Callable[[int, int, str], None]


def _iterate_runs(
    entity: str,
    project: str,
    *,
    runs_per_page: int,
) -> Iterator[wandb.apis.public.Run]:
    api = wandb.Api()
    yield from api.runs(f"{entity}/{project}", per_page=runs_per_page)


def serialize_run(run: wandb.apis.public.Run) -> dict[str, Any]:
    record = RunRecord.from_wandb_run(run)
    return record.model_dump()


def serialize_history_entry(
    run: wandb.apis.public.Run, history_entry: dict[str, Any]
) -> dict[str, Any]:
    record = HistoryEntryRecord.from_wandb_history(history_entry, run.id)
    return record.model_dump()


def fetch_project_runs(
    entity: str,
    project: str,
    *,
    runs_per_page: int = 500,
    include_history: bool = True,
    progress_callback: ProgressFn | None = None,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    progress = progress_callback or default_progress_callback

    runs: list[dict[str, Any]] = []
    histories: list[list[dict[str, Any]]] = []

    logging.info(">> Downloading runs, this will take a while (minutes)")
    run_iter = list(_iterate_runs(entity, project, runs_per_page=runs_per_page))
    total = len(run_iter)
    logging.info(f"  - total runs found: {total}")

    logging.info(f">> Serializing runs and maybe getting histories: {include_history}")
    for index, run in enumerate(run_iter, start=1):
        runs.append(serialize_run(run))
        if include_history:
            history_payloads = [
                serialize_history_entry(run, entry) for entry in run.scan_history()
            ]
            histories.append(history_payloads)
        progress(index, total, run.name)

    if not include_history:
        histories = []

    return runs, histories
