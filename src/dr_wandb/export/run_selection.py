from __future__ import annotations

from typing import Any

from dr_wandb.export.export_request import ExportRequest
from dr_wandb.export.export_state import ExportState, RunTrackingState
from dr_wandb.export.export_modes import FetchMode

TERMINAL_RUN_STATES = {"finished", "failed", "crashed", "killed"}
RUN_BATCH_SIZE = 100


def select_runs(
    *,
    api: Any,
    request: ExportRequest,
    state: ExportState,
) -> list[Any]:
    if (
        request.fetch_mode == FetchMode.FULL_RECONCILE
        or state.max_created_at is None
    ):
        return list(
            iter_runs(
                api=api,
                entity=request.entity,
                project=request.project,
                runs_per_page=request.runs_per_page,
            )
        )

    selected_runs: list[Any] = []
    seen_run_ids: set[str] = set()
    for run in iter_runs(
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
        if should_refresh_tracking_state(tracking)
        and run_id not in seen_run_ids
    ]
    for batch in chunked(refresh_ids, RUN_BATCH_SIZE):
        for run in iter_runs(
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


def iter_runs(
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


def chunked(values: list[str], size: int) -> list[list[str]]:
    return [
        values[index : index + size] for index in range(0, len(values), size)
    ]


def should_refresh_tracking_state(tracking: RunTrackingState) -> bool:
    return tracking.run_state not in TERMINAL_RUN_STATES


def run_state(run: Any) -> str | None:
    value = getattr(run, "state", None)
    return str(value) if value is not None else None
