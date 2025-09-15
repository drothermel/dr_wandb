import logging
from datetime import datetime
from typing import Any

import wandb

from dr_wandb.constants import RunId, RunState


def extract_as_datetime(data: dict[str, Any], key: str) -> datetime | None:
    return datetime.fromtimestamp(data.get(key)) if data.get(key) else None


def select_updated_runs(
    all_runs: list[wandb.apis.public.Run],
    existing_run_states: dict[RunId, RunState],
) -> list[wandb.apis.public.Run]:
    return [
        run
        for run in all_runs
        if run.id not in existing_run_states
        or existing_run_states[run.id] != "finished"
    ]


def default_progress_callback(run_index: int, total_runs: int, message: str) -> None:
    logging.info(f">> {run_index}/{total_runs}: {message}")
