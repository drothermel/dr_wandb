from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

MAX_INT = 2**31 - 1


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
