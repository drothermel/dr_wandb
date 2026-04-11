from __future__ import annotations

from typing import Any, Protocol

from dr_ds.serialization import serialize_timestamp
from pydantic import BaseModel, ConfigDict, Field

from dr_wandb.export.wandb_run import WandbRun


class HistoryWindow(BaseModel):
    min_step: int | None = None
    max_step: int | None = None
    max_records: int | None = None


class HistoryPolicyContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    entity: str
    project: str
    run_id: str
    run_name: str
    run_state: str | None
    run_updated_at: str | None
    run_last_history_step: int | None = None
    raw_run: Any

    @classmethod
    def from_wandb_run(
        cls,
        *,
        wandb_run: WandbRun,
        raw_run: Any,
        run_last_history_step: int | None = None,
    ) -> HistoryPolicyContext:
        return cls(
            entity=wandb_run.entity,
            project=wandb_run.project,
            run_id=wandb_run.run_id,
            run_name=wandb_run.name,
            run_state=wandb_run.state,
            run_updated_at=wandb_run.updated_at,
            run_last_history_step=run_last_history_step,
            raw_run=raw_run,
        )


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


class HistoryPolicy(Protocol):
    def select_history_keys(
        self, ctx: HistoryPolicyContext
    ) -> list[str] | None: ...

    def select_history_window(
        self, ctx: HistoryPolicyContext
    ) -> HistoryWindow | None: ...


class StaticHistoryPolicy:
    def __init__(
        self,
        *,
        keys: list[str] | None = None,
        window: HistoryWindow | None = None,
    ) -> None:
        self.keys = list(keys) if keys is not None else None
        self.window = window

    def select_history_keys(
        self, ctx: HistoryPolicyContext
    ) -> list[str] | None:
        _ = ctx
        return list(self.keys) if self.keys is not None else None

    def select_history_window(
        self, ctx: HistoryPolicyContext
    ) -> HistoryWindow | None:
        _ = ctx
        return self.window
