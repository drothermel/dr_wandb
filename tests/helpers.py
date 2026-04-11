from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


class FakeRun:
    def __init__(
        self,
        *,
        run_id: str,
        created_at: str,
        updated_at: str,
        state: str,
        attrs: dict[str, Any],
        history: list[dict[str, Any]] | None = None,
    ) -> None:
        self.id = run_id
        self.name = attrs.get("displayName", run_id)
        self.created_at = _dt(created_at)
        self.updated_at = _dt(updated_at)
        self.state = state
        self.config = attrs.get("config", {})
        self.summary_metrics = attrs.get("summaryMetrics", {})
        self.summary = attrs.get("summaryMetrics", {})
        self.tags = attrs.get("tags", [])
        self.group = attrs.get("group")
        self.history_keys = attrs.get("historyKeys")
        self.job_type = attrs.get("jobType")
        self.sweep_name = attrs.get("sweepName")
        self.user = attrs.get("user")
        self.read_only = attrs.get("readOnly")
        self.heartbeat_at = None
        self.storage_id = attrs.get("storageId")
        self.url = attrs.get("url")
        self.path = attrs.get("path")
        self.display_name = attrs.get("displayName")
        self.metadata = attrs.get("metadata")
        self.system_metrics = attrs.get("systemMetrics")
        self._attrs = attrs
        self._history = history or []

    def scan_history(
        self,
        *,
        keys: list[str] | None = None,
        min_step: int | None = None,
        max_step: int | None = None,
    ) -> list[dict[str, Any]]:
        rows = list(self._history)
        if min_step is not None:
            rows = [
                row
                for row in rows
                if not isinstance(row.get("_step"), int)
                or row["_step"] >= min_step
            ]
        if max_step is not None:
            rows = [
                row
                for row in rows
                if not isinstance(row.get("_step"), int)
                or row["_step"] <= max_step
            ]
        if keys is None:
            return rows
        selected: list[dict[str, Any]] = []
        for row in rows:
            selected.append(
                {
                    key: value
                    for key, value in row.items()
                    if key.startswith("_") or key in set(keys)
                }
            )
        return selected


class FakeApi:
    def __init__(self, runs: list[FakeRun]) -> None:
        self._runs = runs

    def runs(
        self,
        project_path: str,
        *,
        filters: dict[str, Any],
        order: str,
        per_page: int,
        lazy: bool,
    ) -> list[FakeRun]:
        _ = project_path
        _ = order
        _ = per_page
        _ = lazy
        runs = list(self._runs)
        created_filter = filters.get("createdAt", {})
        if "$gte" in created_filter:
            runs = [
                run
                for run in runs
                if run._attrs.get("createdAt", "") >= created_filter["$gte"]
            ]
        name_filter = filters.get("name", {})
        if "$in" in name_filter:
            allowed = set(name_filter["$in"])
            runs = [run for run in runs if run.id in allowed]
        return runs


def metadata_run(
    run_id: str, *, created_at: str, updated_at: str, state: str
) -> FakeRun:
    attrs = {
        "id": run_id,
        "name": run_id,
        "displayName": run_id,
        "state": state,
        "createdAt": created_at,
        "updatedAt": updated_at,
        "config": {"lr": 0.001},
        "summaryMetrics": {"loss": 1.23},
        "tags": ["baseline"],
        "url": f"https://wandb.ai/ml-moe/moe/runs/{run_id}",
    }
    return FakeRun(
        run_id=run_id,
        created_at=created_at,
        updated_at=updated_at,
        state=state,
        attrs=attrs,
    )


def history_run(
    run_id: str,
    *,
    created_at: str,
    updated_at: str,
    state: str,
    steps: list[int],
) -> FakeRun:
    attrs = {
        "id": run_id,
        "name": run_id,
        "displayName": run_id,
        "state": state,
        "createdAt": created_at,
        "updatedAt": updated_at,
        "config": {"width": 512},
        "summaryMetrics": {"eval/loss": 0.5},
        "historyKeys": {"lastStep": max(steps)},
    }
    history = [
        {
            "_step": step,
            "_timestamp": f"2024-01-01T00:00:0{step}+00:00",
            "_runtime": step * 10,
            "_wandb": {"runtime": step * 10},
            "eval/loss": 1.0 / step,
        }
        for step in steps
    ]
    return FakeRun(
        run_id=run_id,
        created_at=created_at,
        updated_at=updated_at,
        state=state,
        attrs=attrs,
        history=history,
    )
