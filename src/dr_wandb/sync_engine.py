from __future__ import annotations

import importlib
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

import pandas as pd

from dr_wandb.patch_ops import apply_run_patch
from dr_wandb.sync_policy import NoopPolicy, SyncPolicy
from dr_wandb.sync_state import default_state_path, load_state, save_state
from dr_wandb.sync_types import (
    ApplyResult,
    CheckpointManifest,
    CheckpointRecord,
    ErrorAction,
    ExportConfig,
    ExportSummary,
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
from dr_wandb.utils import safe_convert_for_parquet


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


def _serialize_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _serialize_run_row(
    run: Any,
    *,
    entity: str,
    project: str,
    decision: RunDecision,
    cursor: RunCursor,
) -> dict[str, Any]:
    summary = getattr(run, "summary", None)
    if summary is None:
        summary = getattr(run, "summary_metrics", None)
    summary_dict = dict(summary) if summary else {}

    config_dict = dict(getattr(run, "config", {}) or {})
    metadata_dict = dict(getattr(run, "metadata", {}) or {})
    system_metrics_dict = dict(getattr(run, "system_metrics", {}) or {})
    system_attrs: dict[str, Any] = {}
    if config_dict:
        system_attrs["config"] = config_dict
    if summary_dict:
        system_attrs["summary_metrics"] = summary_dict
    if metadata_dict:
        system_attrs["metadata"] = metadata_dict
    if system_metrics_dict:
        system_attrs["system_metrics"] = system_metrics_dict

    row = {
        "run_id": getattr(run, "id", ""),
        "run_name": getattr(run, "name", ""),
        "state": getattr(run, "state", None),
        "entity": entity,
        "project": project,
        "created_at": _serialize_timestamp(getattr(run, "created_at", None)),
        "updated_at": _serialize_timestamp(getattr(run, "updated_at", None)),
        "config": config_dict,
        "summary": summary_dict,
        "wandb_metadata": metadata_dict,
        "system_metrics": system_metrics_dict,
        "system_attrs": system_attrs,
        "sweep_info": {
            "sweep_id": getattr(run, "sweep_id", None),
            "sweep_url": getattr(run, "sweep_url", None),
        },
        "decision_status": decision.status,
        "decision_reason": decision.reason,
        "decision_metadata": decision.metadata,
        "history_seen": cursor.history_seen,
        "last_step": cursor.last_step,
    }
    return _to_jsonable(row)


def _serialize_history_row(run_id: str, history_entry: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        str(key): value
        for key, value in history_entry.items()
        if not str(key).startswith("_")
    }
    row = {
        "run_id": run_id,
        "_step": history_entry.get("_step"),
        "_timestamp": history_entry.get("_timestamp"),
        "_runtime": history_entry.get("_runtime"),
        "_wandb": history_entry.get("_wandb", {}),
        "metrics": metrics,
    }
    return _to_jsonable(row)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name,
        suffix=".tmp",
        delete=False,
    ) as tmp_file:
        json.dump(payload, tmp_file, indent=2, sort_keys=True)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        tmp_name = tmp_file.name
    os.replace(tmp_name, path)


def _atomic_write_rows(
    *,
    rows: list[dict[str, Any]],
    path: Path,
    output_format: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "parquet":
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=path.name,
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
        df = safe_convert_for_parquet(pd.DataFrame(rows))
        df.to_parquet(tmp_path)
        with open(tmp_path, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        return

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name,
        suffix=".tmp",
        delete=False,
    ) as tmp_file:
        for row in rows:
            tmp_file.write(json.dumps(_to_jsonable(row), sort_keys=True, default=str) + "\n")
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        tmp_name = tmp_file.name
    os.replace(tmp_name, path)


def _state_hash(state: ProjectSyncState) -> str:
    payload = json.dumps(state.model_dump(mode="python"), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _history_metrics_hash(value: Any) -> str:
    payload = json.dumps(_to_jsonable(value), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _IncrementalExportWriter:
    def __init__(self, *, config: ExportConfig, policy: SyncPolicy) -> None:
        self.config = config
        self.policy = policy
        self.output_dir = Path(config.output_dir)
        self.checkpoint_root = self.output_dir / config.checkpoint_dirname
        self.runs_dir = self.checkpoint_root / "runs"
        self.history_dir = self.checkpoint_root / "history"
        self.manifest_path = self.checkpoint_root / "manifest.json"
        self.inspection_path = self.checkpoint_root / "inspection.jsonl"
        self._manifest: CheckpointManifest | None = None

    @property
    def manifest(self) -> CheckpointManifest:
        if self._manifest is None:
            raise RuntimeError("begin_export() must be called before accessing manifest")
        return self._manifest

    def _chunk_extension(self) -> str:
        return ".parquet" if self.config.output_format == "parquet" else ".jsonl"

    def _save_manifest(self) -> None:
        _atomic_write_json(
            self.manifest_path,
            self.manifest.model_dump(mode="python"),
        )

    def begin_export(self) -> CheckpointManifest:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

        now = _utc_now_iso()
        if self.manifest_path.exists():
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            manifest = CheckpointManifest.model_validate(raw)
            if manifest.entity != self.config.entity or manifest.project != self.config.project:
                raise ValueError(
                    "Checkpoint manifest entity/project mismatch: "
                    f"found {manifest.entity}/{manifest.project}, expected "
                    f"{self.config.entity}/{self.config.project}"
                )
            if manifest.output_format != self.config.output_format:
                raise ValueError(
                    "Checkpoint manifest output format mismatch: "
                    f"found {manifest.output_format}, expected {self.config.output_format}"
                )
            manifest.status = "in_progress"
            manifest.updated_at = now
            manifest.policy_module = self.policy.__class__.__module__
            manifest.policy_class = self.policy.__class__.__name__
            self._manifest = manifest
            self._save_manifest()
            return manifest

        manifest = CheckpointManifest(
            entity=self.config.entity,
            project=self.config.project,
            output_format=self.config.output_format,
            policy_module=self.policy.__class__.__module__,
            policy_class=self.policy.__class__.__name__,
            created_at=now,
            updated_at=now,
            checkpoint_dir=str(self.checkpoint_root),
            runs_dir=str(self.runs_dir),
            history_dir=str(self.history_dir),
            inspection_path=str(self.inspection_path),
        )
        self._manifest = manifest
        self._save_manifest()
        return manifest

    def write_checkpoint(
        self,
        *,
        run_rows_batch: list[dict[str, Any]],
        history_rows_batch: list[dict[str, Any]],
        run_start_index: int,
        run_end_index: int,
        state_hash: str,
    ) -> CheckpointRecord:
        checkpoint_id = self.manifest.last_checkpoint_id + 1
        extension = self._chunk_extension()
        runs_file = self.runs_dir / f"chunk-{checkpoint_id:06d}{extension}"
        history_file = self.history_dir / f"chunk-{checkpoint_id:06d}{extension}"
        record_file = self.checkpoint_root / f"checkpoint-{checkpoint_id:06d}.json"

        _atomic_write_rows(
            rows=run_rows_batch,
            path=runs_file,
            output_format=self.config.output_format,
        )
        _atomic_write_rows(
            rows=history_rows_batch,
            path=history_file,
            output_format=self.config.output_format,
        )

        step_values = [
            entry.get("_step")
            for entry in history_rows_batch
            if isinstance(entry.get("_step"), int)
        ]
        metric_keys: set[str] = set()
        for entry in history_rows_batch:
            metrics = entry.get("metrics", {})
            if isinstance(metrics, dict):
                metric_keys.update(str(key) for key in metrics)

        record = CheckpointRecord(
            checkpoint_id=checkpoint_id,
            created_at=_utc_now_iso(),
            run_rows=len(run_rows_batch),
            history_rows=len(history_rows_batch),
            cumulative_run_rows=self.manifest.total_run_rows + len(run_rows_batch),
            cumulative_history_rows=self.manifest.total_history_rows + len(history_rows_batch),
            run_start_index=run_start_index,
            run_end_index=run_end_index,
            run_ids=[
                str(row.get("run_id", ""))
                for row in run_rows_batch[: self.config.inspection_sample_rows]
            ],
            run_names=[
                str(row.get("run_name", ""))
                for row in run_rows_batch[: self.config.inspection_sample_rows]
            ],
            metric_keys_sample=sorted(metric_keys)[: self.config.inspection_sample_rows],
            step_min=min(step_values) if step_values else None,
            step_max=max(step_values) if step_values else None,
            runs_file=str(runs_file),
            history_file=str(history_file),
            record_file=str(record_file),
            state_hash=state_hash,
        )
        _atomic_write_json(record_file, record.model_dump(mode="python"))
        return record

    def commit_checkpoint(
        self,
        record: CheckpointRecord,
        *,
        processed_runs: int,
        elapsed_seconds: float,
        checkpoint_elapsed_seconds: float,
    ) -> CheckpointManifest:
        manifest = self.manifest
        manifest.checkpoints.append(record)
        manifest.last_checkpoint_id = record.checkpoint_id
        manifest.total_run_rows = record.cumulative_run_rows
        manifest.total_history_rows = record.cumulative_history_rows
        manifest.updated_at = _utc_now_iso()
        manifest.status = "in_progress"
        self._save_manifest()

        runs_per_second = (
            record.run_rows / checkpoint_elapsed_seconds if checkpoint_elapsed_seconds > 0 else 0.0
        )
        history_rows_per_second = (
            record.history_rows / checkpoint_elapsed_seconds
            if checkpoint_elapsed_seconds > 0
            else 0.0
        )
        total_runs_per_second = processed_runs / elapsed_seconds if elapsed_seconds > 0 else 0.0
        total_history_per_second = (
            manifest.total_history_rows / elapsed_seconds if elapsed_seconds > 0 else 0.0
        )
        inspection_payload = {
            "checkpoint_id": record.checkpoint_id,
            "created_at": record.created_at,
            "run_rows": record.run_rows,
            "history_rows": record.history_rows,
            "cumulative_run_rows": record.cumulative_run_rows,
            "cumulative_history_rows": record.cumulative_history_rows,
            "run_start_index": record.run_start_index,
            "run_end_index": record.run_end_index,
            "run_ids": record.run_ids,
            "run_names": record.run_names,
            "metric_keys_sample": record.metric_keys_sample,
            "step_min": record.step_min,
            "step_max": record.step_max,
            "checkpoint_runs_per_second": runs_per_second,
            "checkpoint_history_rows_per_second": history_rows_per_second,
            "total_runs_per_second": total_runs_per_second,
            "total_history_rows_per_second": total_history_per_second,
        }
        with open(self.inspection_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(inspection_payload, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return manifest

    def _load_chunk_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        run_frames: list[pd.DataFrame] = []
        history_frames: list[pd.DataFrame] = []

        for record in self.manifest.checkpoints:
            runs_file = Path(record.runs_file)
            history_file = Path(record.history_file)
            if self.config.output_format == "parquet":
                if runs_file.exists():
                    run_frames.append(pd.read_parquet(runs_file))
                if history_file.exists():
                    history_frames.append(pd.read_parquet(history_file))
                continue

            if runs_file.exists():
                if runs_file.stat().st_size > 0:
                    run_frames.append(pd.read_json(runs_file, lines=True))
                else:
                    run_frames.append(pd.DataFrame())
            if history_file.exists():
                if history_file.stat().st_size > 0:
                    history_frames.append(pd.read_json(history_file, lines=True))
                else:
                    history_frames.append(pd.DataFrame())

        runs_df = pd.concat(run_frames, ignore_index=True) if run_frames else pd.DataFrame()
        history_df = (
            pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
        )
        return runs_df, history_df

    def _dedupe_runs(self, runs_df: pd.DataFrame) -> pd.DataFrame:
        if runs_df.empty:
            return runs_df
        if "run_id" not in runs_df.columns:
            return runs_df
        return runs_df.drop_duplicates(subset=["run_id"], keep="last").reset_index(drop=True)

    def _dedupe_history(self, history_df: pd.DataFrame) -> pd.DataFrame:
        if history_df.empty:
            return history_df
        if "metrics" in history_df.columns:
            history_df = history_df.copy()
            history_df["_metrics_hash"] = history_df["metrics"].apply(_history_metrics_hash)

        dedupe_subset: list[str]
        has_run_id = "run_id" in history_df.columns
        has_step = "_step" in history_df.columns
        has_timestamp = "_timestamp" in history_df.columns
        has_metrics_hash = "_metrics_hash" in history_df.columns

        if has_run_id and has_step and has_timestamp and has_metrics_hash:
            dedupe_subset = ["run_id", "_step", "_timestamp", "_metrics_hash"]
        elif has_run_id and has_step:
            dedupe_subset = ["run_id", "_step"]
        elif has_run_id and has_timestamp and has_metrics_hash:
            dedupe_subset = ["run_id", "_timestamp", "_metrics_hash"]
        else:
            return history_df.reset_index(drop=True)

        deduped = history_df.drop_duplicates(subset=dedupe_subset, keep="last").reset_index(
            drop=True
        )
        if "_metrics_hash" in deduped.columns:
            deduped = deduped.drop(columns=["_metrics_hash"])
        return deduped

    def finalize_outputs(
        self,
        *,
        write_outputs_fn: Callable[
            [list[dict[str, Any]], list[dict[str, Any]], str], tuple[Path, Path, Path]
        ],
    ) -> tuple[Path, Path, Path, int, int, bool]:
        if not self.config.finalize_compact:
            self.manifest.status = "completed_no_compact"
            self.manifest.updated_at = _utc_now_iso()
            self._save_manifest()
            return (
                self.runs_dir,
                self.history_dir,
                self.manifest_path,
                self.manifest.total_run_rows,
                self.manifest.total_history_rows,
                False,
            )

        runs_df, history_df = self._load_chunk_frames()
        deduped_runs = self._dedupe_runs(runs_df)
        deduped_history = self._dedupe_history(history_df)

        run_rows = deduped_runs.to_dict(orient="records")
        history_rows = deduped_history.to_dict(orient="records")
        exported_at = _utc_now_iso()
        runs_path, history_path, manifest_path = write_outputs_fn(
            run_rows,
            history_rows,
            exported_at,
        )

        self.manifest.status = "completed"
        self.manifest.updated_at = exported_at
        self._save_manifest()
        return (
            runs_path,
            history_path,
            manifest_path,
            len(run_rows),
            len(history_rows),
            True,
        )


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

    def _normalize_project_token(self, value: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "", value).lower()

    def _resolve_project_ref(self, api: Any, entity: str, project: str) -> tuple[str, str]:
        direct_lookup_error: Exception | None = None
        if hasattr(api, "project"):
            try:
                project_obj = api.project(project, entity=entity)
                resolved_name = getattr(project_obj, "name", None)
                if resolved_name:
                    return entity, str(resolved_name)
                return entity, project
            except Exception as exc:  # noqa: BLE001
                direct_lookup_error = exc

        if not hasattr(api, "projects"):
            if direct_lookup_error is not None:
                raise ValueError(
                    f"Could not resolve project {project!r} in entity {entity!r}. "
                    f"Direct lookup failed with {type(direct_lookup_error).__name__}: "
                    f"{direct_lookup_error}"
                ) from direct_lookup_error
            return entity, project

        try:
            projects = list(api.projects(entity=entity))
        except Exception as exc:  # noqa: BLE001
            if direct_lookup_error is not None:
                raise ValueError(
                    f"Could not resolve project {project!r} in entity {entity!r}. "
                    f"Direct lookup failed with {type(direct_lookup_error).__name__}: "
                    f"{direct_lookup_error}; project listing failed with "
                    f"{type(exc).__name__}: {exc}"
                ) from direct_lookup_error
            raise ValueError(
                f"Could not resolve project {project!r} in entity {entity!r}. "
                f"Project listing failed with {type(exc).__name__}: {exc}"
            ) from exc

        if not projects:
            if direct_lookup_error is not None:
                raise ValueError(
                    f"Could not find project {project!r} in entity {entity!r}. "
                    "Direct lookup failed and project listing returned no projects."
                ) from direct_lookup_error
            return entity, project

        names: list[str] = []
        for item in projects:
            name = getattr(item, "name", None)
            if name:
                names.append(str(name))

        if project in names:
            return entity, project

        requested_norm = self._normalize_project_token(project)
        normalized_matches = [
            name for name in names if self._normalize_project_token(name) == requested_norm
        ]
        if len(normalized_matches) == 1:
            resolved = normalized_matches[0]
            logging.info(
                "Resolved project %r to %r in entity %r",
                project,
                resolved,
                entity,
            )
            return entity, resolved

        if normalized_matches:
            raise ValueError(
                "Ambiguous project name "
                f"{project!r} for entity {entity!r}; matching projects: {sorted(normalized_matches)}"
            )

        preview = ", ".join(sorted(names)[:15])
        raise ValueError(
            f"Could not find project {project!r} in entity {entity!r}. "
            f"Available projects include: [{preview}]"
        )

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

    def _default_export_window(self, ctx: SyncContext) -> HistoryWindow:
        if ctx.cursor and ctx.cursor.last_step is not None:
            return HistoryWindow(min_step=ctx.cursor.last_step + 1)
        return HistoryWindow()

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

    def _evaluate_run(
        self,
        ctx: SyncContext,
        *,
        include_history_when_unspecified: bool = False,
    ) -> tuple[RunDecision, PatchPlan | None, list[dict[str, Any]]]:
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

        if include_history_when_unspecified and selected_window is None and selected_keys is None:
            selected_window = self._default_export_window(ctx)

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
        api = self._api_factory()
        resolved_entity, resolved_project = self._resolve_project_ref(api, entity, project)
        resolved_state_path = state_path or default_state_path(
            resolved_entity, resolved_project
        )
        state = load_state(
            resolved_state_path, entity=resolved_entity, project=resolved_project
        )
        runs_obj = api.runs(
            f"{resolved_entity}/{resolved_project}", per_page=runs_per_page
        )

        evaluations: list[RunEvaluation] = []
        processed_runs = 0
        planned_patches = 0

        for run in runs_obj:
            processed_runs += 1
            cursor = state.runs.get(run.id)
            ctx = SyncContext(
                entity=resolved_entity,
                project=resolved_project,
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
            entity=resolved_entity,
            project=resolved_project,
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
        api = self._api_factory()
        resolved_entity, resolved_project = self._resolve_project_ref(api, entity, project)
        resolved_state_path = state_path or default_state_path(
            resolved_entity, resolved_project
        )
        state = load_state(
            resolved_state_path, entity=resolved_entity, project=resolved_project
        )
        runs_obj = api.runs(
            f"{resolved_entity}/{resolved_project}", per_page=runs_per_page
        )

        planned: list[PlannedPatch] = []
        processed_runs = 0

        for run in runs_obj:
            processed_runs += 1
            cursor = state.runs.get(run.id)
            ctx = SyncContext(
                entity=resolved_entity,
                project=resolved_project,
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
                            entity=resolved_entity,
                            project=resolved_project,
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

    def _write_export_outputs(
        self,
        *,
        config: ExportConfig,
        run_rows: list[dict[str, Any]],
        history_rows: list[dict[str, Any]],
        exported_at: str,
    ) -> tuple[Path, Path, Path]:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        base = f"{config.entity}_{config.project}"
        if config.output_format == "parquet":
            runs_path = output_dir / f"{base}_runs.parquet"
            history_path = output_dir / f"{base}_history.parquet"

            runs_df = safe_convert_for_parquet(pd.DataFrame(run_rows))
            history_df = safe_convert_for_parquet(pd.DataFrame(history_rows))
            runs_df.to_parquet(runs_path)
            history_df.to_parquet(history_path)
        else:
            runs_path = output_dir / f"{base}_runs.jsonl"
            history_path = output_dir / f"{base}_history.jsonl"
            with open(runs_path, "w", encoding="utf-8") as f:
                for row in run_rows:
                    f.write(
                        json.dumps(_to_jsonable(row), sort_keys=True, default=str) + "\n"
                    )
            with open(history_path, "w", encoding="utf-8") as f:
                for row in history_rows:
                    f.write(
                        json.dumps(_to_jsonable(row), sort_keys=True, default=str)
                        + "\n"
                    )

        manifest_path = output_dir / f"{base}_manifest.json"
        manifest = {
            "schema_version": 1,
            "entity": config.entity,
            "project": config.project,
            "output_format": config.output_format,
            "policy_module": self.policy.__class__.__module__,
            "policy_class": self.policy.__class__.__name__,
            "exported_at": exported_at,
            "runs_count": len(run_rows),
            "history_count": len(history_rows),
            "files": {
                "runs": str(runs_path),
                "history": str(history_path),
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        return runs_path, history_path, manifest_path

    def _create_export_writer(self, config: ExportConfig) -> _IncrementalExportWriter:
        return _IncrementalExportWriter(config=config, policy=self.policy)

    def export_project(self, config: ExportConfig) -> ExportSummary:
        started_at = time.monotonic()
        logging.info(
            "Resolving project reference for export: entity=%s project=%s",
            config.entity,
            config.project,
        )
        api = self._api_factory()
        resolved_entity, resolved_project = self._resolve_project_ref(
            api, config.entity, config.project
        )
        logging.info(
            "Export target resolved to %s/%s",
            resolved_entity,
            resolved_project,
        )
        resolved_state_path = config.state_path or default_state_path(
            resolved_entity, resolved_project
        )
        state = load_state(
            resolved_state_path, entity=resolved_entity, project=resolved_project
        )
        logging.info(
            "Loaded sync state from %s (tracked_runs=%s)",
            resolved_state_path,
            len(state.runs),
        )
        logging.info(
            "Building runs iterator for %s/%s (runs_per_page=%s)",
            resolved_entity,
            resolved_project,
            config.runs_per_page,
        )
        runs_obj = api.runs(
            f"{resolved_entity}/{resolved_project}", per_page=config.runs_per_page
        )
        logging.info("Starting run iteration")
        processed_runs = 0
        resolved_config = config.model_copy(
            update={"entity": resolved_entity, "project": resolved_project}
        )

        if not resolved_config.incremental:
            run_rows: list[dict[str, Any]] = []
            history_rows: list[dict[str, Any]] = []

            for run in runs_obj:
                processed_runs += 1
                cursor = state.runs.get(run.id)
                ctx = SyncContext(
                    entity=resolved_entity,
                    project=resolved_project,
                    run_id=run.id,
                    run_name=run.name,
                    run_state=getattr(run, "state", None),
                    run_updated_at=self._coerce_run_updated_at(run),
                    run=run,
                    cursor=cursor,
                )

                decision, _patch, history_tail = self._evaluate_run(
                    ctx,
                    include_history_when_unspecified=True,
                )
                self._update_cursor(state, ctx, decision, history_tail)
                updated_cursor = state.runs[ctx.run_id]
                run_rows.append(
                    _serialize_run_row(
                        run,
                        entity=resolved_entity,
                        project=resolved_project,
                        decision=decision,
                        cursor=updated_cursor,
                    )
                )
                history_rows.extend(
                    _serialize_history_row(ctx.run_id, entry) for entry in history_tail
                )

                if processed_runs % resolved_config.save_every == 0:
                    state.last_synced_at = _utc_now_iso()
                    save_state(state, resolved_state_path)
                    logging.info(
                        "Export progress: runs=%s history_rows=%s (state saved)",
                        processed_runs,
                        len(history_rows),
                    )

            exported_at = _utc_now_iso()
            state.last_synced_at = exported_at
            save_state(state, resolved_state_path)

            runs_path, history_path, manifest_path = self._write_export_outputs(
                config=resolved_config,
                run_rows=run_rows,
                history_rows=history_rows,
                exported_at=exported_at,
            )
            elapsed_seconds = time.monotonic() - started_at
            logging.info(
                "Export complete: runs=%s history_rows=%s elapsed=%.2fs runs_file=%s history_file=%s",
                len(run_rows),
                len(history_rows),
                elapsed_seconds,
                runs_path,
                history_path,
            )
            return ExportSummary(
                entity=resolved_entity,
                project=resolved_project,
                state_path=str(resolved_state_path),
                output_format=resolved_config.output_format,
                runs_output_path=str(runs_path),
                history_output_path=str(history_path),
                manifest_output_path=str(manifest_path),
                run_count=len(run_rows),
                history_count=len(history_rows),
                exported_at=exported_at,
                checkpoint_count=0,
                checkpoint_manifest_path="",
                finalized=True,
                partial_run_count=len(run_rows),
                partial_history_count=len(history_rows),
            )

        writer = self._create_export_writer(resolved_config)
        checkpoint_manifest = writer.begin_export()
        logging.info(
            "Checkpoint export initialized at %s (existing_checkpoints=%s, total_run_rows=%s, total_history_rows=%s)",
            writer.manifest_path,
            len(checkpoint_manifest.checkpoints),
            checkpoint_manifest.total_run_rows,
            checkpoint_manifest.total_history_rows,
        )

        run_rows_batch: list[dict[str, Any]] = []
        history_rows_batch: list[dict[str, Any]] = []
        checkpoint_started_at = time.monotonic()

        def flush_checkpoint() -> None:
            nonlocal run_rows_batch, history_rows_batch, checkpoint_started_at, checkpoint_manifest
            if not run_rows_batch and not history_rows_batch:
                return

            batch_runs = len(run_rows_batch)
            run_start_index = processed_runs - batch_runs + 1
            run_end_index = processed_runs
            record = writer.write_checkpoint(
                run_rows_batch=run_rows_batch,
                history_rows_batch=history_rows_batch,
                run_start_index=run_start_index,
                run_end_index=run_end_index,
                state_hash=_state_hash(state),
            )
            total_elapsed = time.monotonic() - started_at
            checkpoint_elapsed = time.monotonic() - checkpoint_started_at
            checkpoint_manifest = writer.commit_checkpoint(
                record,
                processed_runs=processed_runs,
                elapsed_seconds=total_elapsed,
                checkpoint_elapsed_seconds=checkpoint_elapsed,
            )

            logging.info(
                "Checkpoint %s committed: run_rows=%s history_rows=%s total_run_rows=%s total_history_rows=%s",
                record.checkpoint_id,
                record.run_rows,
                record.history_rows,
                checkpoint_manifest.total_run_rows,
                checkpoint_manifest.total_history_rows,
            )

            state.last_synced_at = _utc_now_iso()
            save_state(state, resolved_state_path)
            run_rows_batch = []
            history_rows_batch = []
            checkpoint_started_at = time.monotonic()

        for run in runs_obj:
            processed_runs += 1
            cursor = state.runs.get(run.id)
            ctx = SyncContext(
                entity=resolved_entity,
                project=resolved_project,
                run_id=run.id,
                run_name=run.name,
                run_state=getattr(run, "state", None),
                run_updated_at=self._coerce_run_updated_at(run),
                run=run,
                cursor=cursor,
            )

            decision, _patch, history_tail = self._evaluate_run(
                ctx,
                include_history_when_unspecified=True,
            )
            self._update_cursor(state, ctx, decision, history_tail)
            updated_cursor = state.runs[ctx.run_id]
            run_rows_batch.append(
                _serialize_run_row(
                    run,
                    entity=resolved_entity,
                    project=resolved_project,
                    decision=decision,
                    cursor=updated_cursor,
                )
            )
            history_rows_batch.extend(
                _serialize_history_row(ctx.run_id, entry) for entry in history_tail
            )

            if processed_runs % resolved_config.checkpoint_every_runs == 0:
                flush_checkpoint()
            elif processed_runs % resolved_config.save_every == 0:
                logging.info(
                    "Export progress: runs=%s buffered_run_rows=%s buffered_history_rows=%s",
                    processed_runs,
                    len(run_rows_batch),
                    len(history_rows_batch),
                )

        flush_checkpoint()
        state.last_synced_at = _utc_now_iso()
        save_state(state, resolved_state_path)

        runs_path, history_path, manifest_path, run_count, history_count, finalized = (
            writer.finalize_outputs(
                write_outputs_fn=lambda run_rows, history_rows, exported_at: self._write_export_outputs(
                    config=resolved_config,
                    run_rows=run_rows,
                    history_rows=history_rows,
                    exported_at=exported_at,
                )
            )
        )
        checkpoint_manifest = writer.manifest
        exported_at = _utc_now_iso()
        elapsed_seconds = time.monotonic() - started_at
        logging.info(
            "Export complete: runs=%s history_rows=%s checkpoints=%s finalized=%s elapsed=%.2fs runs_path=%s history_path=%s",
            run_count,
            history_count,
            len(checkpoint_manifest.checkpoints),
            finalized,
            elapsed_seconds,
            runs_path,
            history_path,
        )

        return ExportSummary(
            entity=resolved_entity,
            project=resolved_project,
            state_path=str(resolved_state_path),
            output_format=resolved_config.output_format,
            runs_output_path=str(runs_path),
            history_output_path=str(history_path),
            manifest_output_path=str(manifest_path),
            run_count=run_count,
            history_count=history_count,
            exported_at=exported_at,
            checkpoint_count=len(checkpoint_manifest.checkpoints),
            checkpoint_manifest_path=str(writer.manifest_path),
            finalized=finalized,
            partial_run_count=checkpoint_manifest.total_run_rows,
            partial_history_count=checkpoint_manifest.total_history_rows,
        )
