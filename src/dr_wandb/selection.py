from __future__ import annotations

from enum import StrEnum
from typing import Any

from dr_wandb.config import ExportRequest, SyncMode
from dr_wandb.state import ExportState, RunTrackingState


class TerminalRunState(StrEnum):
    FINISHED = "finished"
    FAILED = "failed"
    CRASHED = "crashed"
    KILLED = "killed"


RUN_BATCH_SIZE = 100


def select_runs(
    *,
    api: Any,
    request: ExportRequest,
    state: ExportState,
) -> list[Any]:
    if (
        request.sync_mode == SyncMode.FULL_RECONCILE
        or state.max_created_at is None
    ):
        return list(
            _iter_runs(
                api=api,
                entity=request.entity,
                project=request.project,
                runs_per_page=request.runs_per_page,
            )
        )

    selected_runs: list[Any] = []
    seen_run_ids: set[str] = set()
    for run in _iter_runs(
        api=api,
        entity=request.entity,
        project=request.project,
        runs_per_page=request.runs_per_page,
        filters={"createdAt": {"$gte": state.max_created_at}},
    ):
        run_id = str(getattr(run, "id", ""))
        if run_id == "" or run_id in seen_run_ids or run_id in state.runs:
            continue
        seen_run_ids.add(run_id)
        selected_runs.append(run)

    refresh_ids = [
        run_id
        for run_id, tracking in state.runs.items()
        if _should_refresh(tracking) and run_id not in seen_run_ids
    ]
    for batch in _chunked(refresh_ids, RUN_BATCH_SIZE):
        for run in _iter_runs(
            api=api,
            entity=request.entity,
            project=request.project,
            runs_per_page=request.runs_per_page,
            filters={"name": {"$in": batch}},
        ):
            run_id = str(getattr(run, "id", ""))
            if run_id == "" or run_id in seen_run_ids:
                continue
            seen_run_ids.add(run_id)
            selected_runs.append(run)
    return selected_runs


def _iter_runs(
    *,
    api: Any,
    entity: str,
    project: str,
    runs_per_page: int,
    filters: dict[str, Any] | None = None,
) -> Any:
    return api.runs(
        f"{entity}/{project}",
        filters=filters or {},
        order="+created_at",
        per_page=runs_per_page,
        lazy=False,
    )


def _chunked(values: list[str], size: int) -> list[list[str]]:
    return [
        values[index : index + size] for index in range(0, len(values), size)
    ]


def _should_refresh(tracking: RunTrackingState) -> bool:
    return tracking.run_state not in TerminalRunState
