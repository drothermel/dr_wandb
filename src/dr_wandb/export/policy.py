from __future__ import annotations

from typing import Protocol

from dr_wandb.export.models import HistoryPolicyContext, HistoryWindow


class HistoryPolicy(Protocol):
    def select_history_keys(
        self, ctx: HistoryPolicyContext
    ) -> list[str] | None:
        ...

    def select_history_window(
        self, ctx: HistoryPolicyContext
    ) -> HistoryWindow | None:
        ...


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
