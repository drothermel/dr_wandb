from __future__ import annotations

import hashlib
import json
from typing import Any

from dr_ds.serialization import serialize_timestamp, to_jsonable
from pydantic import BaseModel, Field

from dr_wandb.config import (
    ExportMode,
    ExportRequest,
    HistoryWindow,
    SyncMode,
)
from dr_wandb.wandb_run import WandbRun


class HistoryRow(BaseModel):
    run_id: str
    step: int | None = None
    timestamp: str | None = None
    runtime: int | float | None = None
    wandb_metadata: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_history_entry(
        cls, *, run_id: str, entry: dict[str, Any]
    ) -> HistoryRow:
        wandb_value = entry.get("_wandb")
        return cls(
            run_id=run_id,
            step=entry.get("_step")
            if isinstance(entry.get("_step"), int)
            else None,
            timestamp=serialize_timestamp(entry.get("_timestamp")),
            runtime=entry.get("_runtime"),
            wandb_metadata=dict(wandb_value)
            if isinstance(wandb_value, dict)
            else {},
            metrics={
                str(key): value
                for key, value in entry.items()
                if not str(key).startswith("_")
            },
            extra={
                str(key): value
                for key, value in entry.items()
                if str(key).startswith("_")
                and str(key)
                not in {"_step", "_timestamp", "_runtime", "_wandb"}
            },
        )


def scan_history_for_export(
    *,
    request: ExportRequest,
    wandb_run: WandbRun,
    raw_run: Any,
    run_last_history_step: int | None,
) -> list[HistoryRow]:
    if request.mode != ExportMode.HISTORY:
        return []
    selection = request.history_selection
    keys = selection.keys if selection is not None else None
    window = selection.window if selection is not None else None
    if request.sync_mode == SyncMode.INCREMENTAL and window is None:
        window = _default_incremental_window(run_last_history_step)
    entries = _scan_history(run=raw_run, keys=keys, window=window)
    return [
        HistoryRow.from_history_entry(run_id=wandb_run.run_id, entry=entry)
        for entry in entries
    ]


def merge_history_rows(
    *,
    existing_history_rows: list[HistoryRow],
    new_history_rows: list[HistoryRow],
) -> list[HistoryRow]:
    by_key: dict[tuple[Any, ...], HistoryRow] = {}
    for row in [*existing_history_rows, *new_history_rows]:
        by_key[_history_row_key(row)] = row
    merged = list(by_key.values())
    merged.sort(
        key=lambda row: (
            row.run_id,
            row.step if row.step is not None else float("inf"),
            str(row.timestamp or ""),
        )
    )
    return merged


def max_history_step(rows: list[HistoryRow]) -> int | None:
    step_values = [row.step for row in rows if row.step is not None]
    if len(step_values) == 0:
        return None
    return max(step_values)


def observed_last_history_step(
    *, wandb_run: WandbRun, raw_run: Any
) -> int | None:
    typed = wandb_run.history_keys_last_step
    if typed is not None:
        return typed
    value = getattr(raw_run, "last_history_step", None)
    if isinstance(value, int):
        return value
    return None


def _default_incremental_window(
    run_last_history_step: int | None,
) -> HistoryWindow | None:
    if run_last_history_step is None:
        return None
    return HistoryWindow(min_step=run_last_history_step + 1)


def _scan_history(
    *,
    run: Any,
    keys: list[str] | None,
    window: HistoryWindow | None,
) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {}
    if keys is not None:
        kwargs["keys"] = keys
    if window is not None:
        if window.min_step is not None:
            kwargs["min_step"] = window.min_step
        if window.max_step is not None:
            kwargs["max_step"] = window.max_step
    try:
        entries = list(run.scan_history(**kwargs))
    except TypeError:
        kwargs.pop("max_step", None)
        entries = list(run.scan_history(**kwargs))
    if window is not None and window.max_records is not None:
        entries = entries[-window.max_records :]
    return entries


def _history_row_key(row: HistoryRow) -> tuple[Any, ...]:
    payload_hash = hashlib.sha256(
        json.dumps(
            to_jsonable(
                {
                    "step": row.step,
                    "timestamp": row.timestamp,
                    "runtime": row.runtime,
                    "wandb_metadata": row.wandb_metadata,
                    "metrics": row.metrics,
                    "extra": row.extra,
                }
            ),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if row.step is not None and row.timestamp is not None:
        return (row.run_id, row.step, row.timestamp, payload_hash)
    if row.step is not None:
        return (row.run_id, row.step, payload_hash)
    if row.timestamp is not None:
        return (row.run_id, row.timestamp, payload_hash)
    return (row.run_id, payload_hash)
