from __future__ import annotations

import json
from typing import Any

import pandas as pd

from dr_wandb.shared.serialization import parse_jsonish, to_jsonable

MAX_INT = 2**31 - 1


def _convert_large_ints(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _convert_large_ints(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_convert_large_ints(nested) for nested in value]
    if isinstance(value, int) and abs(value) > MAX_INT:
        return float(value)
    return value


def records_to_parquet_frame(
    records: list[dict[str, Any]], *, json_columns: set[str]
) -> pd.DataFrame:
    normalized: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {}
        for key, value in record.items():
            if key in json_columns:
                row[key] = json.dumps(
                    to_jsonable(_convert_large_ints(value)),
                    sort_keys=True,
                )
            elif isinstance(value, int) and abs(value) > MAX_INT:
                row[key] = float(value)
            else:
                row[key] = value
        normalized.append(row)
    return pd.DataFrame(normalized)


def parquet_frame_to_records(
    frame: pd.DataFrame, *, json_columns: set[str]
) -> list[dict[str, Any]]:
    records = frame.to_dict(orient="records")
    normalized: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {}
        for key, value in record.items():
            if key in json_columns and value is not None:
                row[key] = parse_jsonish(value)
            else:
                row[key] = value
        normalized.append(row)
    return normalized
