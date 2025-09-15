import json
import logging
from datetime import datetime
from typing import Any

import pandas as pd
import wandb

from dr_wandb.constants import MAX_INT, RunId, RunState


def extract_as_datetime(data: dict[str, Any], key: str) -> datetime | None:
    timestamp = data.get(key)
    return datetime.fromtimestamp(timestamp) if timestamp is not None else None


def select_updated_runs(
    all_runs: list[wandb.apis.public.Run],
    existing_run_states: dict[RunId, RunState],
) -> list[wandb.apis.public.Run]:
    return [
        run
        for run in all_runs
        if run.id not in existing_run_states or existing_run_states[run.id] == "running"
    ]


def default_progress_callback(run_index: int, total_runs: int, message: str) -> None:
    logging.info(f">> {run_index}/{total_runs}: {message}")


def convert_large_ints_in_data(data: Any, max_int: int = MAX_INT) -> Any:
    if isinstance(data, dict):
        return {k: convert_large_ints_in_data(v, max_int) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_large_ints_in_data(item, max_int) for item in data]
    elif isinstance(data, int) and abs(data) > max_int:
        return float(data)
    return data


def safe_convert_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "int64":
            mask = df[col].abs() > MAX_INT
            if mask.any():
                df[col] = df[col].astype("float64")
        elif df[col].dtype == "object":
            df[col] = df[col].apply(
                lambda x: json.dumps(convert_large_ints_in_data(x), default=str)
                if isinstance(x, dict | list)
                else str(x)
                if x is not None
                else None
            )
    return df
