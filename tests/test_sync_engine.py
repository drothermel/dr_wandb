from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from dr_wandb.sync_engine import SyncEngine
from dr_wandb.sync_policy import NoopPolicy, SyncPolicy
from dr_wandb.sync_types import (
    BootstrapConfig,
    CheckpointManifest,
    ErrorAction,
    ExportConfig,
    FetchMode,
    HistoryWindow,
    PatchPlan,
    ProjectSyncState,
    RunDecision,
    RunCursor,
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
        created_at: datetime | None = None,
    ) -> None:
        self.id = run_id
        self.name = name
        self.state = state
        self.config = config or {}
        self.tags = tags or []
        self.updated_at = datetime.now(UTC)
        self.created_at = created_at or datetime.now(UTC)
        self.metadata = {"source": "test"}
        self.system_metrics = {"gpu": 1}
        self.summary_metrics = {"loss": 0.1}
        self._history = history or []
        self.update_calls = 0
        self.scan_history_calls = 0
        last_step = max(
            [entry["_step"] for entry in self._history if isinstance(entry.get("_step"), int)],
            default=-1,
        )
        self._attrs = {"historyKeys": {"lastStep": last_step}}

    def scan_history(self, **kwargs: Any):
        self.scan_history_calls += 1
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
        self.runs_calls: list[dict[str, Any]] = []

    def runs(
        self,
        path: str,
        filters: dict[str, Any] | None = None,
        order: str = "+created_at",
        per_page: int = 500,
        include_sweeps: bool = True,
        lazy: bool = True,
    ):
        _ = path
        _ = order
        _ = per_page
        _ = include_sweeps
        _ = lazy
        filters = filters or {}
        self.runs_calls.append({"filters": filters, "lazy": lazy})
        selected_runs = list(self._runs)
        if created_at_filter := filters.get("createdAt"):
            gte_value = created_at_filter.get("$gte")
            if gte_value is not None:
                selected_runs = [
                    run
                    for run in selected_runs
                    if run.created_at.isoformat() >= str(gte_value)
                ]
        if name_filter := filters.get("name"):
            names = set(name_filter.get("$in", []))
            selected_runs = [run for run in selected_runs if run.id in names]
        return selected_runs

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
    assert summary.checkpoint_count == 1
    assert summary.finalized is True
    assert Path(summary.runs_output_path).exists()
    assert Path(summary.history_output_path).exists()
    assert Path(summary.manifest_output_path).exists()
    assert Path(summary.checkpoint_manifest_path).exists()

    manifest = json.loads(Path(summary.manifest_output_path).read_text(encoding="utf-8"))
    assert manifest["runs_count"] == 2
    assert manifest["history_count"] == 3
    assert manifest["policy_class"] == "NoopPolicy"

    checkpoint_manifest = CheckpointManifest.model_validate(
        json.loads(Path(summary.checkpoint_manifest_path).read_text(encoding="utf-8"))
    )
    assert checkpoint_manifest.total_run_rows == 2
    assert checkpoint_manifest.total_history_rows == 3
    assert checkpoint_manifest.status == "completed"


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
    assert second.history_count == 2

    runs_df = pd.read_parquet(first.runs_output_path)
    history_df = pd.read_parquet(first.history_output_path)
    assert len(runs_df) == 1
    assert len(history_df) == 2
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


class TerminalOnStepPolicy(NoopPolicy):
    def classify_run(
        self, ctx: SyncContext, history_tail: list[dict[str, Any]]
    ) -> RunDecision:
        _ = history_tail
        status = "finished" if ctx.run_last_history_step == 10 else "running"
        return RunDecision(status=status)


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


def test_export_project_logs_processed_runs_progress_with_percent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    runs = [
        FakeRun("r1", "run-1", history=[{"_step": 0, "lr": 0.1}]),
        FakeRun("r2", "run-2", history=[{"_step": 0, "lr": 0.2}]),
    ]
    engine, _ = _engine_with(runs, NoopPolicy(), tmp_path / "state.json")
    cfg = ExportConfig(
        entity="entity",
        project="project",
        output_dir=tmp_path / "out-logs",
        output_format="jsonl",
        state_path=tmp_path / "state.json",
        checkpoint_every_runs=1,
    )

    caplog.set_level("INFO")
    engine.export_project(cfg)

    assert "Starting run iteration (selected_runs=2)" in caplog.text
    assert "Checkpoint 1 committed: processed_runs=1/2 (50.0%)" in caplog.text
    assert "Checkpoint 2 committed: processed_runs=2/2 (100.0%)" in caplog.text


def test_export_project_logs_first_run_before_checkpoint_boundary(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    runs = [
        FakeRun("r1", "run-1", history=[{"_step": 0, "lr": 0.1}]),
        FakeRun("r2", "run-2", history=[{"_step": 0, "lr": 0.2}]),
    ]
    engine, _ = _engine_with(runs, NoopPolicy(), tmp_path / "state.json")
    cfg = ExportConfig(
        entity="entity",
        project="project",
        output_dir=tmp_path / "out-first-run-log",
        output_format="jsonl",
        state_path=tmp_path / "state.json",
        checkpoint_every_runs=25,
        save_every=25,
    )

    caplog.set_level("INFO")
    engine.export_project(cfg)

    assert "Export progress: processed_runs=1/2 (50.0%) buffered_run_rows=1 buffered_history_rows=1" in caplog.text


def test_sync_project_incremental_fetches_only_new_and_active_runs(tmp_path: Path):
    old_created_at = datetime(2026, 3, 1, tzinfo=UTC)
    new_created_at = datetime(2026, 3, 2, tzinfo=UTC)
    runs = [
        FakeRun("sealed", "sealed-run", history=[{"_step": 10}], created_at=old_created_at),
        FakeRun("active", "active-run", history=[{"_step": 7}], created_at=old_created_at),
        FakeRun("new", "new-run", history=[{"_step": 1}], created_at=new_created_at),
    ]
    api = FakeApi(runs)
    engine, state_path = _engine_with_api(
        TerminalOnStepPolicy(),
        tmp_path / "state.json",
        api_factory=lambda: api,
    )
    state = ProjectSyncState(
        entity="entity",
        project="project",
        max_created_at=old_created_at.isoformat(),
        runs={
            "sealed": RunCursor(run_id="sealed", last_step=10, terminal=True),
            "active": RunCursor(run_id="active", last_step=7, terminal=False),
        },
    )
    state_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")

    summary = engine.sync_project("entity", "project", state_path=state_path)

    assert summary.processed_runs == 2
    assert {evaluation.run_id for evaluation in summary.run_evaluations} == {"active", "new"}
    assert api.runs_calls[0]["filters"] == {"createdAt": {"$gte": old_created_at.isoformat()}}
    assert api.runs_calls[1]["filters"] == {"name": {"$in": ["active"]}}


def test_sync_project_incremental_metadata_only_policy_skips_history_scan(tmp_path: Path):
    run = FakeRun("r1", "run-1", history=[{"_step": 10, "lr": 0.2}])
    api = FakeApi([run])
    engine, state_path = _engine_with_api(
        TerminalOnStepPolicy(),
        tmp_path / "state.json",
        api_factory=lambda: api,
    )

    summary = engine.sync_project("entity", "project", state_path=state_path)

    assert summary.run_evaluations[0].history_records == 0
    assert run.scan_history_calls == 0
    state = ProjectSyncState.model_validate_json(state_path.read_text(encoding="utf-8"))
    assert state.runs["r1"].terminal is True
    assert state.runs["r1"].last_step == 10


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


def test_export_project_writes_checkpoint_stats_and_inspection(tmp_path: Path):
    runs = [
        FakeRun("r1", "run-1", history=[{"_step": 0, "lr": 0.1}]),
        FakeRun("r2", "run-2", history=[{"_step": 0, "lr": 0.2}]),
        FakeRun("r3", "run-3", history=[{"_step": 0, "lr": 0.3}]),
    ]
    engine, _ = _engine_with(runs, NoopPolicy(), tmp_path / "state.json")
    cfg = ExportConfig(
        entity="entity",
        project="project",
        output_dir=tmp_path / "out-checkpoint",
        output_format="parquet",
        state_path=tmp_path / "state.json",
        checkpoint_every_runs=2,
    )

    summary = engine.export_project(cfg)

    checkpoint_manifest = CheckpointManifest.model_validate(
        json.loads(Path(summary.checkpoint_manifest_path).read_text(encoding="utf-8"))
    )
    assert summary.checkpoint_count == 2
    assert checkpoint_manifest.total_run_rows == 3
    assert checkpoint_manifest.total_history_rows == 3
    assert checkpoint_manifest.total_run_rows == sum(
        record.run_rows for record in checkpoint_manifest.checkpoints
    )
    assert checkpoint_manifest.total_history_rows == sum(
        record.history_rows for record in checkpoint_manifest.checkpoints
    )

    inspection_path = Path(checkpoint_manifest.inspection_path)
    lines = inspection_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == summary.checkpoint_count
    payload = json.loads(lines[0])
    assert payload["checkpoint_id"] == 1
    assert payload["cumulative_run_rows"] >= 1


def test_export_project_resume_after_state_save_failure_dedupes_final_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runs = [
        FakeRun("r1", "run-1", history=[{"_step": 0, "lr": 0.1}]),
        FakeRun("r2", "run-2", history=[{"_step": 0, "lr": 0.2}]),
    ]
    state_path = tmp_path / "state.json"
    engine, _ = _engine_with(runs, NoopPolicy(), state_path)
    cfg = ExportConfig(
        entity="entity",
        project="project",
        output_dir=tmp_path / "out-resume",
        output_format="parquet",
        state_path=state_path,
        checkpoint_every_runs=1,
    )

    from dr_wandb import sync_engine as sync_engine_module

    original_save_state = sync_engine_module.save_state
    calls = {"count": 0}

    def flaky_save_state(state, path):  # noqa: ANN001
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("simulated save failure")
        return original_save_state(state, path)

    monkeypatch.setattr(sync_engine_module, "save_state", flaky_save_state)
    with pytest.raises(RuntimeError, match="simulated save failure"):
        engine.export_project(cfg)

    monkeypatch.setattr(sync_engine_module, "save_state", original_save_state)
    summary = engine.export_project(cfg)

    runs_df = pd.read_parquet(summary.runs_output_path)
    history_df = pd.read_parquet(summary.history_output_path)
    assert summary.partial_run_count >= 3
    assert summary.run_count == 2
    assert summary.history_count == 2
    assert len(runs_df) == 2
    assert len(history_df) == 2


def test_export_project_no_finalize_compact_keeps_checkpoint_outputs(tmp_path: Path):
    runs = [FakeRun("r1", "run-1", history=[{"_step": 0, "lr": 0.1}])]
    engine, _ = _engine_with(runs, NoopPolicy(), tmp_path / "state.json")
    cfg = ExportConfig(
        entity="entity",
        project="project",
        output_dir=tmp_path / "out-no-finalize",
        output_format="parquet",
        state_path=tmp_path / "state.json",
        finalize_compact=False,
    )

    summary = engine.export_project(cfg)

    checkpoint_manifest = CheckpointManifest.model_validate(
        json.loads(Path(summary.checkpoint_manifest_path).read_text(encoding="utf-8"))
    )
    assert summary.finalized is False
    assert summary.run_count == 1
    assert summary.history_count == 1
    assert checkpoint_manifest.status == "completed_no_compact"
    assert Path(summary.runs_output_path).is_dir()
    assert Path(summary.history_output_path).is_dir()


@pytest.mark.parametrize("source_format", ["jsonl", "parquet"])
def test_bootstrap_export_rebuilds_state_and_merges_to_single_checkpoint(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    source_format: str,
):
    runs = [
        FakeRun(
            "r1",
            "run-1",
            history=[{"_step": 0, "lr": 0.1}, {"_step": 10, "lr": 0.01}],
            created_at=datetime(2026, 3, 1, tzinfo=UTC),
        ),
        FakeRun(
            "r2",
            "run-2",
            history=[{"_step": 0, "lr": 0.2}, {"_step": 5, "lr": 0.05}],
            created_at=datetime(2026, 3, 2, tzinfo=UTC),
        ),
    ]
    source_state_path = tmp_path / "source-state.json"
    source_engine, _ = _engine_with(runs, NoopPolicy(), source_state_path)
    source_cfg = ExportConfig(
        entity="entity",
        project="project",
        output_dir=tmp_path / "source-export",
        output_format=cast(Any, source_format),
        state_path=source_state_path,
        checkpoint_every_runs=1,
    )
    source_summary = source_engine.export_project(source_cfg)
    source_manifest = CheckpointManifest.model_validate(
        json.loads(Path(source_summary.checkpoint_manifest_path).read_text(encoding="utf-8"))
    )
    assert len(source_manifest.checkpoints) == 2

    bootstrap_engine = SyncEngine(policy=TerminalOnStepPolicy())
    caplog.set_level("INFO")
    bootstrap_summary = bootstrap_engine.bootstrap_export(
        BootstrapConfig(
            entity="entity",
            project="project",
            source_dir=tmp_path / "source-export",
            output_dir=tmp_path / "bootstrapped-export",
            state_path=tmp_path / "bootstrapped-state.json",
        )
    )

    checkpoint_manifest = CheckpointManifest.model_validate(
        json.loads(Path(bootstrap_summary.checkpoint_manifest_path).read_text(encoding="utf-8"))
    )
    assert bootstrap_summary.run_count == 2
    assert bootstrap_summary.history_count == 4
    assert bootstrap_summary.checkpoint_count == 1
    assert len(checkpoint_manifest.checkpoints) == 1
    assert checkpoint_manifest.total_run_rows == 2
    assert checkpoint_manifest.total_history_rows == 4

    state = ProjectSyncState.model_validate_json(
        Path(bootstrap_summary.state_path).read_text(encoding="utf-8")
    )
    assert state.max_created_at == "2026-03-02T00:00:00+00:00"
    assert state.runs["r1"].terminal is True
    assert state.runs["r2"].terminal is False
    assert state.runs["r1"].last_step == 10
    assert state.runs["r2"].last_step == 5

    if source_format == "jsonl":
        run_rows = [
            json.loads(line)
            for line in Path(bootstrap_summary.runs_output_path)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
    else:
        run_rows = pd.read_parquet(bootstrap_summary.runs_output_path).to_dict(orient="records")
    run_status_by_id = {row["run_id"]: row["decision_status"] for row in run_rows}
    assert run_status_by_id == {"r1": "finished", "r2": "running"}
    assert "Bootstrap source loaded: run_rows=2 history_rows=4" in caplog.text
    assert "Bootstrap history stream complete: processed_history_rows=4 matched_history_rows=4" in caplog.text
    assert "Starting bootstrap run rebuild (selected_runs=2)" in caplog.text
    assert "Bootstrap progress: processed_runs=1/2 (50.0%) buffered_run_rows=1" in caplog.text
    assert "Bootstrap checkpoint committed: checkpoint_id=1 processed_runs=2/2 (100.0%)" in caplog.text
    assert "Bootstrap complete: processed_runs=2/2 (100.0%) run_count=2 history_rows=4 checkpoints=1" in caplog.text


def test_bootstrap_export_requires_overwrite_for_existing_targets(tmp_path: Path):
    runs = [FakeRun("r1", "run-1", history=[{"_step": 0, "lr": 0.1}])]
    source_state_path = tmp_path / "source-state.json"
    source_engine, _ = _engine_with(runs, NoopPolicy(), source_state_path)
    source_cfg = ExportConfig(
        entity="entity",
        project="project",
        output_dir=tmp_path / "source-export",
        output_format="jsonl",
        state_path=source_state_path,
    )
    source_engine.export_project(source_cfg)
    output_dir = tmp_path / "bootstrapped-export"
    output_dir.mkdir()
    state_path = tmp_path / "bootstrapped-state.json"
    state_path.write_text("{}", encoding="utf-8")

    engine = SyncEngine(policy=TerminalOnStepPolicy())

    with pytest.raises(ValueError, match="already exists"):
        engine.bootstrap_export(
            BootstrapConfig(
                entity="entity",
                project="project",
                source_dir=tmp_path / "source-export",
                output_dir=output_dir,
                state_path=state_path,
            )
        )


def test_bootstrap_export_overwrite_replaces_existing_targets(tmp_path: Path):
    runs = [FakeRun("r1", "run-1", history=[{"_step": 0, "lr": 0.1}, {"_step": 10, "lr": 0.01}])]
    source_state_path = tmp_path / "source-state.json"
    source_engine, _ = _engine_with(runs, NoopPolicy(), source_state_path)
    source_cfg = ExportConfig(
        entity="entity",
        project="project",
        output_dir=tmp_path / "source-export",
        output_format="jsonl",
        state_path=source_state_path,
    )
    source_engine.export_project(source_cfg)
    output_dir = tmp_path / "bootstrapped-export"
    output_dir.mkdir()
    (output_dir / "stale.txt").write_text("stale", encoding="utf-8")
    state_path = tmp_path / "bootstrapped-state.json"
    state_path.write_text("{}", encoding="utf-8")

    engine = SyncEngine(policy=TerminalOnStepPolicy())
    summary = engine.bootstrap_export(
        BootstrapConfig(
            entity="entity",
            project="project",
            source_dir=tmp_path / "source-export",
            output_dir=output_dir,
            state_path=state_path,
            overwrite_output=True,
        )
    )

    assert Path(summary.runs_output_path).exists()
    assert not (output_dir / "stale.txt").exists()


def test_bootstrap_export_rejects_mismatched_source_manifest(tmp_path: Path):
    runs = [FakeRun("r1", "run-1", history=[{"_step": 0, "lr": 0.1}])]
    source_state_path = tmp_path / "source-state.json"
    source_engine, _ = _engine_with(runs, NoopPolicy(), source_state_path)
    source_cfg = ExportConfig(
        entity="entity",
        project="project",
        output_dir=tmp_path / "source-export",
        output_format="jsonl",
        state_path=source_state_path,
    )
    source_summary = source_engine.export_project(source_cfg)
    manifest_path = Path(source_summary.manifest_output_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["entity"] = "other-entity"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    engine = SyncEngine(policy=TerminalOnStepPolicy())

    with pytest.raises(ValueError, match="manifest mismatch"):
        engine.bootstrap_export(
            BootstrapConfig(
                entity="entity",
                project="project",
                source_dir=tmp_path / "source-export",
                output_dir=tmp_path / "bootstrapped-export",
                state_path=tmp_path / "bootstrapped-state.json",
            )
        )
