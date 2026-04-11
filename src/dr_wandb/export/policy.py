from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


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
    run: Any


class HistoryRow(BaseModel):
    run_id: str
    step: int | None = None
    timestamp: str | None = None
    runtime: int | float | None = None
    wandb_metadata: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)


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
