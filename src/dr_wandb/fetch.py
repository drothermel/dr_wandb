from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import Any

import wandb

from dr_wandb.history_entry_record import HistoryEntryRecord
from dr_wandb.run_record import RunRecord
from dr_wandb.utils import default_progress_callback

ProgressFn = Callable[[int, int | None, str], None]


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

    logging.info(">> Downloading and processing runs, this will take a while (minutes)")

    # Get Runs object to check for total count without materializing all runs
    api = wandb.Api()
    runs_obj = api.runs(f"{entity}/{project}", per_page=runs_per_page)
    total: int | None = None

    # Check if the Runs object has a total attribute (some API versions provide this)
    if hasattr(runs_obj, "total") and runs_obj.total is not None:
        total = runs_obj.total
        logging.info(f"  - total runs found: {total}")
    else:
        # If total is not available, we'll count as we go
        logging.info("  - streaming runs (total count not available upfront)")

    logging.info(f">> Serializing runs and maybe getting histories: {include_history}")

    # Stream runs directly from the Runs object without materializing
    run_count = 0
    for run in runs_obj:
        run_count += 1
        runs.append(serialize_run(run))
        if include_history:
            history_payloads = [
                serialize_history_entry(run, entry) for entry in run.scan_history()
            ]
            histories.append(history_payloads)

        # Use None to indicate unknown total when total is not available
        progress(run_count, total, run.name)

    # Update total if we didn't have it initially
    if total is None:
        total = run_count
        logging.info(f"  - total runs processed: {total}")

    if not include_history:
        histories = []

    return runs, histories
