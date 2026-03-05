from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from dr_wandb.sync_engine import SyncEngine
from dr_wandb.sync_policy import NoopPolicy, SyncPolicy
from dr_wandb.sync_types import (
    ErrorAction,
    ExportConfig,
    HistoryWindow,
    PatchPlan,
    RunDecision,
    SyncContext,
)


class FakeRun:
    def __init__(
        self,
        run_id: str,
        name: str,
        *,
        state: str = "running",
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> None:
        self.id = run_id
        self.name = name
        self.state = state
        self.config = config or {}
        self.tags = tags or []
        self.updated_at = datetime.now(UTC)
        self.created_at = datetime.now(UTC)
        self.metadata = {"source": "test"}
        self.system_metrics = {"gpu": 1}
        self.summary_metrics = {"loss": 0.1}
        self._history = history or []
        self.update_calls = 0

    def scan_history(self, **kwargs: Any):
        min_step = kwargs.get("min_step")
        max_step = kwargs.get("max_step")
        keys = kwargs.get("keys")
        for entry in self._history:
            step = entry.get("_step")
            if isinstance(min_step, int) and isinstance(step, int) and step < min_step:
                continue
            if isinstance(max_step, int) and isinstance(step, int) and step > max_step:
                continue
            if keys:
                filtered = {k: v for k, v in entry.items() if k in set(keys) or k == "_step"}
                yield filtered
            else:
                yield entry

    def update(self):
        self.update_calls += 1


class FakeApi:
    def __init__(self, runs: list[FakeRun], project_names: list[str] | None = None) -> None:
        self._runs = runs
        self._run_lookup = {run.id: run for run in runs}
        self._project_names = project_names if project_names is not None else ["project"]

    def runs(self, path: str, per_page: int = 500):
        _ = path
        _ = per_page
        return list(self._runs)

    def run(self, path: str):
        run_id = path.rsplit("/", 1)[-1]
        return self._run_lookup[run_id]

    def project(self, name: str, entity: str):
        _ = entity
        for project_name in self._project_names:
            if project_name == name:
                return type("Project", (), {"name": project_name})
        raise ValueError(f"Could not find project {name}")

    def projects(self, entity: str):
        _ = entity
        names = self._project_names
        return [type("Project", (), {"name": name}) for name in names]


class RecordingPolicy(NoopPolicy):
    def __init__(self) -> None:
        self.history_keys_calls: list[str] = []
        self.classify_calls: list[tuple[str, int]] = []

    def select_history_keys(self, ctx: SyncContext) -> list[str] | None:
        self.history_keys_calls.append(ctx.run_id)
        return ["lr"]

    def select_history_window(self, ctx: SyncContext) -> HistoryWindow | None:
        _ = ctx
        return HistoryWindow(min_step=1)

    def classify_run(
        self, ctx: SyncContext, history_tail: list[dict[str, Any]]
    ) -> RunDecision:
        self.classify_calls.append((ctx.run_id, len(history_tail)))
        return RunDecision(status="ok", metadata={"history": len(history_tail)})

    def infer_patch(
        self, ctx: SyncContext, history_tail: list[dict[str, Any]]
    ) -> PatchPlan | None:
        _ = history_tail
        if ctx.run_id == "r1":
            return PatchPlan(set_config={"field": "value"})
        return None

    def should_update(self, ctx: SyncContext, patch: PatchPlan) -> bool:
        _ = ctx
        return not patch.is_empty()


class RetryPolicy(RecordingPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.failures = 0

    def classify_run(self, ctx, history_tail):
        if self.failures == 0:
            self.failures += 1
            raise RuntimeError("transient")
        return super().classify_run(ctx, history_tail)

    def on_error(self, ctx, exc):
        _ = ctx
        _ = exc
        return ErrorAction.RETRY


class AbortPolicy(RecordingPolicy):
    def classify_run(self, ctx, history_tail):
        _ = ctx
        _ = history_tail
        raise RuntimeError("fatal")

    def on_error(self, ctx, exc):
        _ = ctx
        _ = exc
        return ErrorAction.ABORT


class SkipPolicy(RecordingPolicy):
    def classify_run(self, ctx, history_tail):
        _ = ctx
        _ = history_tail
        raise RuntimeError("bad")

    def on_error(self, ctx, exc):
        _ = ctx
        _ = exc
        return ErrorAction.SKIP


def _engine_with(
    runs: list[FakeRun], policy: SyncPolicy | NoopPolicy, state_path: Path
) -> tuple[SyncEngine, Path]:
    return SyncEngine(
        policy=policy,
        api_factory=lambda: FakeApi(runs),
        sleep_fn=lambda _: None,
        max_retries=2,
        retry_backoff_seconds=0.0,
    ), state_path


def _engine_with_projects(
    runs: list[FakeRun],
    policy: SyncPolicy | NoopPolicy,
    state_path: Path,
    project_names: list[str],
) -> tuple[SyncEngine, Path]:
    return SyncEngine(
        policy=policy,
        api_factory=lambda: FakeApi(runs, project_names=project_names),
        sleep_fn=lambda _: None,
        max_retries=2,
        retry_backoff_seconds=0.0,
    ), state_path


def _engine_with_api(
    policy: SyncPolicy | NoopPolicy,
    state_path: Path,
    api_factory: Any,
) -> tuple[SyncEngine, Path]:
    return SyncEngine(
        policy=policy,
        api_factory=api_factory,
        sleep_fn=lambda _: None,
        max_retries=2,
        retry_backoff_seconds=0.0,
    ), state_path


def test_engine_noop_policy_syncs_without_history(tmp_path: Path):
    runs = [FakeRun("r1", "run-1"), FakeRun("r2", "run-2")]
    engine, state_path = _engine_with(runs, NoopPolicy(), tmp_path / "state.json")

    summary = engine.sync_project("entity", "project", state_path=state_path)

    assert summary.processed_runs == 2
    assert summary.planned_patches == 0
    assert all(evaluation.history_records == 0 for evaluation in summary.run_evaluations)


def test_engine_policy_hooks_and_patch_planning(tmp_path: Path):
    runs = [
        FakeRun("r1", "run-1", history=[{"_step": 0, "lr": 0.1}, {"_step": 1, "lr": 0.2}]),
        FakeRun("r2", "run-2", history=[{"_step": 1, "lr": 0.3}]),
    ]
    policy = RecordingPolicy()
    engine, state_path = _engine_with(runs, policy, tmp_path / "state.json")

    summary = engine.sync_project("entity", "project", state_path=state_path)

    assert summary.processed_runs == 2
    assert summary.planned_patches == 1
    assert policy.history_keys_calls == ["r1", "r2"]
    assert policy.classify_calls[0][0] == "r1"


def test_plan_and_apply_patch_plans_respects_dry_run(tmp_path: Path):
    runs = [FakeRun("r1", "run-1", history=[{"_step": 1, "lr": 0.2}])]
    policy = RecordingPolicy()
    engine, state_path = _engine_with(runs, policy, tmp_path / "state.json")

    plans = engine.plan_patches("entity", "project", state_path=state_path)
    assert len(plans) == 1

    dry_run_results = engine.apply_patch_plans(plans, dry_run=True)
    assert dry_run_results[0].applied is False
    assert runs[0].update_calls == 0

    apply_results = engine.apply_patch_plans(plans, dry_run=False)
    assert apply_results[0].applied is True
    assert runs[0].update_calls == 1


def test_retry_action_retries_then_succeeds(tmp_path: Path):
    runs = [FakeRun("r1", "run-1", history=[{"_step": 1, "lr": 0.2}])]
    policy = RetryPolicy()
    engine, state_path = _engine_with(runs, policy, tmp_path / "state.json")

    summary = engine.sync_project("entity", "project", state_path=state_path)

    assert summary.run_evaluations[0].decision.status == "ok"
    assert policy.failures == 1


def test_skip_action_uses_fallback_decision(tmp_path: Path):
    runs = [FakeRun("r1", "run-1", history=[{"_step": 1, "lr": 0.2}])]
    policy = SkipPolicy()
    engine, state_path = _engine_with(runs, policy, tmp_path / "state.json")

    summary = engine.sync_project("entity", "project", state_path=state_path)

    assert summary.run_evaluations[0].decision.status == "error"


def test_abort_action_raises(tmp_path: Path):
    runs = [FakeRun("r1", "run-1", history=[{"_step": 1, "lr": 0.2}])]
    policy = AbortPolicy()
    engine, state_path = _engine_with(runs, policy, tmp_path / "state.json")

    with pytest.raises(RuntimeError, match="fatal"):
        engine.sync_project("entity", "project", state_path=state_path)


def test_export_project_jsonl_writes_manifest_and_rows(tmp_path: Path):
    runs = [
        FakeRun("r1", "run-1", history=[{"_step": 0, "lr": 0.1}, {"_step": 1, "lr": 0.2}]),
        FakeRun("r2", "run-2", history=[{"_step": 0, "lr": 0.3}]),
    ]
    engine, _state_path = _engine_with(runs, NoopPolicy(), tmp_path / "state.json")
    cfg = ExportConfig(
        entity="entity",
        project="project",
        output_dir=tmp_path / "out-jsonl",
        output_format="jsonl",
        state_path=tmp_path / "state.json",
    )

    summary = engine.export_project(cfg)

    assert summary.run_count == 2
    assert summary.history_count == 3
    assert Path(summary.runs_output_path).exists()
    assert Path(summary.history_output_path).exists()
    assert Path(summary.manifest_output_path).exists()

    manifest = json.loads(Path(summary.manifest_output_path).read_text(encoding="utf-8"))
    assert manifest["runs_count"] == 2
    assert manifest["history_count"] == 3
    assert manifest["policy_class"] == "NoopPolicy"


def test_export_project_parquet_and_incremental_history(tmp_path: Path):
    runs = [FakeRun("r1", "run-1", history=[{"_step": 0, "lr": 0.1}, {"_step": 1, "lr": 0.2}])]
    state_path = tmp_path / "state.json"
    engine, _ = _engine_with(runs, NoopPolicy(), state_path)
    cfg = ExportConfig(
        entity="entity",
        project="project",
        output_dir=tmp_path / "out-parquet",
        output_format="parquet",
        state_path=state_path,
    )

    first = engine.export_project(cfg)
    second = engine.export_project(cfg)

    assert first.history_count == 2
    assert second.history_count == 0

    runs_df = pd.read_parquet(first.runs_output_path)
    history_df = pd.read_parquet(first.history_output_path)
    assert len(runs_df) == 1
    assert len(history_df) == 0  # second export overwrites files with incremental slice
    assert runs_df["decision_status"].iloc[0] == "unknown"


class ExportWindowPolicy(NoopPolicy):
    def select_history_keys(self, ctx: SyncContext) -> list[str] | None:
        _ = ctx
        return ["lr"]

    def select_history_window(self, ctx: SyncContext) -> HistoryWindow | None:
        _ = ctx
        return HistoryWindow(min_step=1)

    def classify_run(
        self, ctx: SyncContext, history_tail: list[dict[str, Any]]
    ) -> RunDecision:
        _ = ctx
        return RunDecision(status="filtered", metadata={"rows": len(history_tail)})


def test_export_project_respects_policy_keys_window_and_decision(tmp_path: Path):
    runs = [FakeRun("r1", "run-1", history=[{"_step": 0, "lr": 0.1}, {"_step": 1, "lr": 0.2}])]
    engine, _ = _engine_with(runs, ExportWindowPolicy(), tmp_path / "state.json")
    cfg = ExportConfig(
        entity="entity",
        project="project",
        output_dir=tmp_path / "out-filtered",
        output_format="jsonl",
        state_path=tmp_path / "state.json",
    )

    summary = engine.export_project(cfg)
    runs_lines = Path(summary.runs_output_path).read_text(encoding="utf-8").strip().splitlines()
    history_lines = Path(summary.history_output_path).read_text(encoding="utf-8").strip().splitlines()

    run_row = json.loads(runs_lines[0])
    history_row = json.loads(history_lines[0])
    assert summary.history_count == 1
    assert run_row["decision_status"] == "filtered"
    assert run_row["decision_metadata"]["rows"] == 1
    assert history_row["_step"] == 1
    assert "lr" in history_row["metrics"]


def test_sync_project_resolves_project_case_and_punctuation(tmp_path: Path):
    runs = [FakeRun("r1", "run-1")]
    engine, state_path = _engine_with_projects(
        runs,
        NoopPolicy(),
        tmp_path / "state.json",
        project_names=["MoE"],
    )

    summary = engine.sync_project("ml-moe", "moe", state_path=state_path)

    assert summary.entity == "ml-moe"
    assert summary.project == "MoE"
    assert summary.processed_runs == 1


def test_sync_project_raises_clear_error_for_missing_project(tmp_path: Path):
    runs = [FakeRun("r1", "run-1")]
    engine, state_path = _engine_with_projects(
        runs,
        NoopPolicy(),
        tmp_path / "state.json",
        project_names=["alpha", "beta"],
    )

    with pytest.raises(ValueError, match="Could not find project"):
        engine.sync_project("ml-moe", "moe", state_path=state_path)


def test_sync_project_raises_resolver_error_when_lookup_and_listing_fail(tmp_path: Path):
    runs = [FakeRun("r1", "run-1")]

    class BrokenApi(FakeApi):
        def project(self, name: str, entity: str):
            _ = name
            _ = entity
            raise RuntimeError("lookup failed")

        def projects(self, entity: str):
            _ = entity
            raise RuntimeError("listing failed")

    engine, state_path = _engine_with_api(
        NoopPolicy(),
        tmp_path / "state.json",
        api_factory=lambda: BrokenApi(runs),
    )

    with pytest.raises(ValueError, match="Direct lookup failed"):
        engine.sync_project("ml-moe", "moe", state_path=state_path)
