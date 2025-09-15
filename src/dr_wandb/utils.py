from datetime import datetime
from typing import Any

import wandb
from sqlalchemy import Select, select, text
from sqlalchemy.orm import Session

from dr_wandb.constants import SUPPORTED_FILTER_FIELDS, QueryType, RunId, RunState
from dr_wandb.store import HistoryRecord, RunRecord


def build_query(
    base_query: QueryType,
    kwargs: dict[str, Any] | None = None,
) -> Select[RunRecord]:
    assert base_query in ["runs", "history"]
    query = select(RunRecord)
    if base_query == "history":
        query = select(HistoryRecord, RunRecord.run_name, RunRecord.project).join(
            RunRecord, HistoryRecord.run_id == RunRecord.run_id
        )
    if kwargs is not None:
        assert all(k in SUPPORTED_FILTER_FIELDS for k in kwargs)
        assert all(v is not None for v in kwargs.values())
        if "project" in kwargs:
            query = query.where(RunRecord.project == kwargs["project"])
        if "entity" in kwargs:
            query = query.where(RunRecord.entity == kwargs["entity"])
        if "state" in kwargs:
            query = query.where(RunRecord.state == kwargs["state"])
        if "run_ids" in kwargs:
            query = query.where(RunRecord.run_id.in_(kwargs["run_ids"]))
    return query


def delete_history_for_run(session: Session, run_id: str) -> None:
    session.execute(
        text("DELETE FROM wandb_history WHERE run_id = :run_id"),
        {"run_id": run_id},
    )


def extract_as_datetime(data: dict[str, Any], key: str) -> datetime | None:
    return datetime.fromtimestamp(data.get(key)) if data.get(key) else None


def select_new_and_unfinished_runs(
    all_runs: list[wandb.apis.public.Run],
    existing_run_states: dict[RunId, RunState],
) -> list[wandb.apis.public.Run]:
    return [
        run
        for run in all_runs
        if run.id not in existing_run_states
        or existing_run_states[run.id] != "finished"
    ]
