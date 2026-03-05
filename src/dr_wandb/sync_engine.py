from __future__ import annotations

import importlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

from dr_wandb.patch_ops import apply_run_patch
from dr_wandb.sync_policy import NoopPolicy, SyncPolicy
from dr_wandb.sync_state import default_state_path, load_state, save_state
from dr_wandb.sync_types import (
    ApplyResult,
    ErrorAction,
    HistoryWindow,
    PatchPlan,
    PlannedPatch,
    ProjectSyncState,
    RunCursor,
    RunDecision,
    RunEvaluation,
    SyncContext,
    SyncSummary,
)


def load_policy(policy_module: str, policy_class: str) -> SyncPolicy:
    module = importlib.import_module(policy_module)
    klass = getattr(module, policy_class)
    return cast(SyncPolicy, klass())


_wandb_api: Any = None


def _default_wandb_api() -> Any:
    global _wandb_api
    if _wandb_api is None:
        import wandb

        _wandb_api = wandb.Api()
    return _wandb_api


def write_patch_jsonl(plans: list[PlannedPatch], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for plan in plans:
            f.write(json.dumps(plan.model_dump(mode="python"), sort_keys=True) + "\n")


def read_patch_jsonl(input_path: Path) -> list[PlannedPatch]:
    plans: list[PlannedPatch] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            plans.append(PlannedPatch.model_validate(json.loads(stripped)))
    return plans


class SyncEngine:
    def __init__(
        self,
        policy: SyncPolicy | None = None,
        *,
        api_factory: Callable[[], Any] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        self.policy = policy or NoopPolicy()
        self._api_factory = api_factory or _default_wandb_api
        self._sleep = sleep_fn or time.sleep
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def _with_retry(self, ctx: SyncContext, fn: Callable[[], Any], fallback: Any) -> Any:
        attempts = 0
        while True:
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                action = self.policy.on_error(ctx, exc)
                if action == ErrorAction.ABORT:
                    raise
                attempts += 1
                if action == ErrorAction.RETRY and attempts <= self.max_retries:
                    backoff = self.retry_backoff_seconds * attempts
                    logging.warning(
                        "Retrying run %s after error (%s): %s",
                        ctx.run_id,
                        type(exc).__name__,
                        exc,
                    )
                    self._sleep(backoff)
                    continue
                logging.warning(
                    "Skipping operation for run %s after error (%s): %s",
                    ctx.run_id,
                    type(exc).__name__,
                    exc,
                )
                return fallback

    def _coerce_run_updated_at(self, run: Any) -> str | None:
        updated_at = getattr(run, "updated_at", None)
        if updated_at is None:
            return None
        if isinstance(updated_at, datetime):
            return updated_at.astimezone(timezone.utc).isoformat()
        return str(updated_at)

    def _default_window(self, ctx: SyncContext) -> HistoryWindow | None:
        if ctx.cursor and ctx.cursor.last_step is not None:
            return HistoryWindow(min_step=ctx.cursor.last_step + 1)
        return None

    def _scan_history(
        self,
        ctx: SyncContext,
        *,
        keys: list[str] | None,
        window: HistoryWindow | None,
    ) -> list[dict[str, Any]]:
        if keys is None and window is None:
            return []

        def do_scan() -> list[dict[str, Any]]:
            kwargs: dict[str, Any] = {}
            if keys:
                kwargs["keys"] = keys
            if window is not None:
                if window.min_step is not None:
                    kwargs["min_step"] = window.min_step
                if window.max_step is not None:
                    kwargs["max_step"] = window.max_step
            try:
                entries = list(ctx.run.scan_history(**kwargs))
            except TypeError:
                kwargs.pop("max_step", None)
                entries = list(ctx.run.scan_history(**kwargs))

            if window is not None and window.max_records is not None:
                entries = entries[-window.max_records :]
            return entries

        return cast(list[dict[str, Any]], self._with_retry(ctx, do_scan, []))

    def _max_history_step(self, history_tail: list[dict[str, Any]], existing: int | None) -> int | None:
        max_step = existing
        for entry in history_tail:
            step = entry.get("_step")
            if isinstance(step, int):
                max_step = step if max_step is None else max(max_step, step)
        return max_step

    def _evaluate_run(self, ctx: SyncContext) -> tuple[RunDecision, PatchPlan | None, list[dict[str, Any]]]:
        selected_keys = self._with_retry(
            ctx,
            lambda: self.policy.select_history_keys(ctx),
            None,
        )
        selected_window = self._with_retry(
            ctx,
            lambda: self.policy.select_history_window(ctx),
            None,
        )

        if selected_window is None:
            selected_window = self._default_window(ctx)

        history_tail = self._scan_history(ctx, keys=selected_keys, window=selected_window)

        decision = cast(
            RunDecision,
            self._with_retry(
                ctx,
                lambda: self.policy.classify_run(ctx, history_tail),
                RunDecision(status="error", reason="classify_run failed"),
            ),
        )
        patch = cast(
            PatchPlan | None,
            self._with_retry(
                ctx,
                lambda: self.policy.infer_patch(ctx, history_tail),
                None,
            ),
        )
        return decision, patch, history_tail

    def _update_cursor(
        self,
        state: ProjectSyncState,
        ctx: SyncContext,
        decision: RunDecision,
        history_tail: list[dict[str, Any]],
    ) -> None:
        history_seen = (ctx.cursor.history_seen if ctx.cursor else 0) + len(history_tail)
        state.runs[ctx.run_id] = RunCursor(
            run_id=ctx.run_id,
            updated_at=ctx.run_updated_at,
            last_step=self._max_history_step(
                history_tail,
                existing=(ctx.cursor.last_step if ctx.cursor else None),
            ),
            history_seen=history_seen,
            decision_status=decision.status,
            decision_reason=decision.reason,
            metadata=decision.metadata,
        )

    def sync_project(
        self,
        entity: str,
        project: str,
        *,
        state_path: Path | None = None,
        runs_per_page: int = 500,
        save_every: int = 25,
    ) -> SyncSummary:
        resolved_state_path = state_path or default_state_path(entity, project)
        state = load_state(resolved_state_path, entity=entity, project=project)

        api = self._api_factory()
        runs_obj = api.runs(f"{entity}/{project}", per_page=runs_per_page)

        evaluations: list[RunEvaluation] = []
        processed_runs = 0
        planned_patches = 0

        for run in runs_obj:
            processed_runs += 1
            cursor = state.runs.get(run.id)
            ctx = SyncContext(
                entity=entity,
                project=project,
                run_id=run.id,
                run_name=run.name,
                run_state=getattr(run, "state", None),
                run_updated_at=self._coerce_run_updated_at(run),
                run=run,
                cursor=cursor,
            )

            decision, patch, history_tail = self._evaluate_run(ctx)
            is_planned = bool(
                patch
                and not patch.is_empty()
                and self._with_retry(
                    ctx,
                    lambda _ctx=ctx, _patch=patch: self.policy.should_update(_ctx, _patch),
                    False,
                )
            )
            if is_planned:
                planned_patches += 1

            evaluations.append(
                RunEvaluation(
                    run_id=ctx.run_id,
                    run_name=ctx.run_name,
                    run_state=ctx.run_state,
                    history_records=len(history_tail),
                    decision=decision,
                    patch_planned=is_planned,
                )
            )
            self._update_cursor(state, ctx, decision, history_tail)

            if processed_runs % save_every == 0:
                state.last_synced_at = datetime.now(timezone.utc).isoformat()
                save_state(state, resolved_state_path)

        state.last_synced_at = datetime.now(timezone.utc).isoformat()
        save_state(state, resolved_state_path)

        return SyncSummary(
            entity=entity,
            project=project,
            state_path=str(resolved_state_path),
            processed_runs=processed_runs,
            planned_patches=planned_patches,
            run_evaluations=evaluations,
        )

    def plan_patches(
        self,
        entity: str,
        project: str,
        *,
        state_path: Path | None = None,
        runs_per_page: int = 500,
        save_every: int = 25,
    ) -> list[PlannedPatch]:
        resolved_state_path = state_path or default_state_path(entity, project)
        state = load_state(resolved_state_path, entity=entity, project=project)

        api = self._api_factory()
        runs_obj = api.runs(f"{entity}/{project}", per_page=runs_per_page)

        planned: list[PlannedPatch] = []
        processed_runs = 0

        for run in runs_obj:
            processed_runs += 1
            cursor = state.runs.get(run.id)
            ctx = SyncContext(
                entity=entity,
                project=project,
                run_id=run.id,
                run_name=run.name,
                run_state=getattr(run, "state", None),
                run_updated_at=self._coerce_run_updated_at(run),
                run=run,
                cursor=cursor,
            )
            decision, patch, history_tail = self._evaluate_run(ctx)
            self._update_cursor(state, ctx, decision, history_tail)

            if patch and not patch.is_empty():
                should_update = bool(
                    self._with_retry(
                        ctx,
                        lambda _ctx=ctx, _patch=patch: self.policy.should_update(
                            _ctx, _patch
                        ),
                        False,
                    )
                )
                if should_update:
                    planned.append(
                        PlannedPatch(
                            entity=entity,
                            project=project,
                            run_id=ctx.run_id,
                            run_name=ctx.run_name,
                            patch=patch,
                        )
                    )

            if processed_runs % save_every == 0:
                state.last_synced_at = datetime.now(timezone.utc).isoformat()
                save_state(state, resolved_state_path)

        state.last_synced_at = datetime.now(timezone.utc).isoformat()
        save_state(state, resolved_state_path)
        return planned

    def apply_patch_plans(
        self,
        plans: list[PlannedPatch],
        *,
        dry_run: bool = True,
    ) -> list[ApplyResult]:
        api = self._api_factory()
        results: list[ApplyResult] = []

        for plan in plans:
            run_path = f"{plan.entity}/{plan.project}/{plan.run_id}"
            run = api.run(run_path)
            result = apply_run_patch(run, plan.patch, dry_run=dry_run)
            results.append(result)

        return results
