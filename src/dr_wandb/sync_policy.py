from __future__ import annotations

from typing import Any, Protocol

from dr_wandb.sync_types import (
    ErrorAction,
    HistoryWindow,
    PatchPlan,
    RunDecision,
    SyncContext,
)


class SyncPolicy(Protocol):
    def select_history_keys(self, ctx: SyncContext) -> list[str] | None:
        ...

    def select_history_window(self, ctx: SyncContext) -> HistoryWindow | None:
        ...

    def classify_run(self, ctx: SyncContext, history_tail: list[dict[str, Any]]) -> RunDecision:
        ...

    def infer_patch(self, ctx: SyncContext, history_tail: list[dict[str, Any]]) -> PatchPlan | None:
        ...

    def should_update(self, ctx: SyncContext, patch: PatchPlan) -> bool:
        ...

    def is_terminal(self, ctx: SyncContext, decision: RunDecision) -> bool:
        ...

    def on_error(self, ctx: SyncContext, exc: Exception) -> ErrorAction:
        ...


class NoopPolicy:
    def select_history_keys(self, ctx: SyncContext) -> list[str] | None:
        return None

    def select_history_window(self, ctx: SyncContext) -> HistoryWindow | None:
        return None

    def classify_run(self, ctx: SyncContext, history_tail: list[dict[str, Any]]) -> RunDecision:
        _ = history_tail
        return RunDecision(status="unknown", reason="NoopPolicy")

    def infer_patch(self, ctx: SyncContext, history_tail: list[dict[str, Any]]) -> PatchPlan | None:
        _ = ctx
        _ = history_tail
        return None

    def should_update(self, ctx: SyncContext, patch: PatchPlan) -> bool:
        _ = ctx
        return not patch.is_empty()

    def is_terminal(self, ctx: SyncContext, decision: RunDecision) -> bool:
        _ = ctx
        return decision.status == "finished"

    def on_error(self, ctx: SyncContext, exc: Exception) -> ErrorAction:
        _ = ctx
        _ = exc
        return ErrorAction.SKIP
