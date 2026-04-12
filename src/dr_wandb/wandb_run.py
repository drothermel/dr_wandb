from __future__ import annotations

from typing import Any

from dr_ds.serialization import serialize_timestamp, to_jsonable
from pydantic import BaseModel, ConfigDict, Field


class WandbRun(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str
    name: str
    display_name: str | None = None
    entity: str
    project: str

    state: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    heartbeat_at: str | None = None

    storage_id: str | None = None
    url: str | None = None
    path: Any | None = None

    tags: list[str] = Field(default_factory=list)
    group: str | None = None
    job_type: str | None = None
    sweep_name: str | None = None
    user: Any | None = None
    read_only: bool | None = None

    config: dict[str, Any] = Field(default_factory=dict)
    summary_metrics: dict[str, Any] = Field(default_factory=dict)
    system_metrics: dict[str, Any] | None = None
    history_keys: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_wandb_run(
        cls,
        run: Any,
        *,
        entity: str,
        project: str,
        include_metadata: bool,
    ) -> WandbRun:
        summary = getattr(run, "summary_metrics", None)
        if summary is None:
            summary = getattr(run, "summary", None)
        payload: dict[str, Any] = {
            "run_id": str(getattr(run, "id", "")),
            "name": str(getattr(run, "name", "")),
            "display_name": getattr(run, "display_name", None),
            "entity": entity,
            "project": project,
            "state": _optional_str(getattr(run, "state", None)),
            "created_at": serialize_timestamp(
                getattr(run, "created_at", None)
            ),
            "updated_at": serialize_timestamp(
                getattr(run, "updated_at", None)
            ),
            "heartbeat_at": serialize_timestamp(
                getattr(run, "heartbeat_at", None)
            ),
            "storage_id": getattr(run, "storage_id", None),
            "url": getattr(run, "url", None),
            "path": getattr(run, "path", None),
            "tags": list(getattr(run, "tags", None) or []),
            "group": getattr(run, "group", None),
            "job_type": getattr(run, "job_type", None),
            "sweep_name": getattr(run, "sweep_name", None),
            "user": getattr(run, "user", None),
            "read_only": getattr(run, "read_only", None),
            "config": getattr(run, "config", None) or {},
            "summary_metrics": summary or {},
            "system_metrics": getattr(run, "system_metrics", None),
            "history_keys": getattr(run, "history_keys", None),
            "metadata": (
                getattr(run, "metadata", None) if include_metadata else None
            ),
        }
        return cls.model_validate(to_jsonable(payload))

    @property
    def history_keys_last_step(self) -> int | None:
        if isinstance(self.history_keys, dict):
            value = self.history_keys.get("lastStep")
            if isinstance(value, int):
                return value
        return None


class RunSnapshot(BaseModel):
    run: WandbRun
    exported_at: str

    @property
    def sort_key(self) -> tuple[str, str]:
        return (self.run.created_at or "", self.run.run_id)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
