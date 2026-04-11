from __future__ import annotations

import hashlib
import json
from typing import Any

from dr_ds.serialization import to_jsonable

from dr_wandb.export.export_request import ExportRequest
from dr_wandb.export.export_modes import ExportMode, FetchMode
from dr_wandb.export.policy import (
    HistoryPolicyContext,
    HistoryRow,
    HistoryWindow,
)
from dr_wandb.export.wandb_run import WandbRun


def scan_history_for_export(
    *,
    request: ExportRequest,
    ctx: HistoryPolicyContext,
) -> list[HistoryRow]:
    if request.mode != ExportMode.HISTORY:
        return []
    keys = (
        request.history_policy.select_history_keys(ctx)
        if request.history_policy is not None
        else None
    )
    window = (
        request.history_policy.select_history_window(ctx)
        if request.history_policy is not None
        else None
    )
    if request.fetch_mode == FetchMode.INCREMENTAL and window is None:
        window = default_history_window(ctx)
    entries = scan_history(run=ctx.raw_run, keys=keys, window=window)
    return [
        HistoryRow.from_history_entry(run_id=ctx.run_id, entry=entry)
        for entry in entries
    ]


def default_history_window(ctx: HistoryPolicyContext) -> HistoryWindow | None:
    if ctx.run_last_history_step is None:
        return None
    return HistoryWindow(min_step=ctx.run_last_history_step + 1)


def scan_history(
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


def max_history_step(rows: list[HistoryRow]) -> int | None:
    step_values = [row.step for row in rows if row.step is not None]
    if len(step_values) == 0:
        return None
    return max(step_values)


def merge_history_rows(
    *,
    existing_history_rows: list[HistoryRow],
    new_history_rows: list[HistoryRow],
) -> list[HistoryRow]:
    by_key: dict[tuple[Any, ...], HistoryRow] = {}
    for row in [*existing_history_rows, *new_history_rows]:
        by_key[history_row_key(row)] = row
    merged = list(by_key.values())
    merged.sort(
        key=lambda row: (
            row.run_id,
            row.step if row.step is not None else float("inf"),
            str(row.timestamp or ""),
        )
    )
    return merged


def history_row_key(row: HistoryRow) -> tuple[Any, ...]:
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
