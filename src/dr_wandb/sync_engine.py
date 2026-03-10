from __future__ import annotations

import ast
import importlib
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from dr_wandb.patch_ops import apply_run_patch
from dr_wandb.sync_policy import NoopPolicy, SyncPolicy
from dr_wandb.sync_state import default_state_path, load_state, save_state
from dr_wandb.sync_types import (
    ApplyResult,
    BootstrapConfig,
    BootstrapSummary,
    CheckpointManifest,
    CheckpointRecord,
    ErrorAction,
    ExportConfig,
    ExportSummary,
    FetchMode,
    HistoryWindow,
    PatchPlan,
    PlannedPatch,
    ProjectSyncState,
    RunCursor,
    RunDecision,
    RunEvaluation,
    RunSelectionSummary,
    StateInspectionRun,
    StateInspectionSummary,
    SyncContext,
    SyncSummary,
)
from dr_wandb.utils import safe_convert_for_parquet


def load_policy(policy_module: str, policy_class: str) -> SyncPolicy:
    module = importlib.import_module(policy_module)
    klass = getattr(module, policy_class)
    return cast(SyncPolicy, klass())


_wandb_api: Any = None
_ACTIVE_RUN_BATCH_SIZE = 100
_BOOTSTRAP_RUN_LOG_EVERY = 100
_BOOTSTRAP_HISTORY_LOG_EVERY = 250_000
_BOOTSTRAP_HISTORY_BATCH_SIZE = 25_000


def _default_wandb_api() -> Any:
    global _wandb_api
    if _wandb_api is None:
        import wandb

        _wandb_api = wandb.Api()
    return _wandb_api


def _isoformat_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def _progress_percent(processed_runs: int, total_runs: int) -> float:
    if total_runs <= 0:
        return 100.0
    return (processed_runs / total_runs) * 100.0


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (dict, list)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _parse_jsonish(value: Any) -> Any:
    if _is_missing_value(value) or isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(value)
        except (json.JSONDecodeError, SyntaxError, ValueError):
            continue
    return value


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


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


class _BootstrapRun(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    name: str
    state: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    summary_metrics: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    system_metrics: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    sweep_id: str | None = None
    sweep_url: str | None = None


class _BootstrapSourceInfo(BaseModel):
    runs_path: Path
    history_path: Path
    output_format: str
    manifest_path: Path | None = None
    run_count: int | None = None
    history_count: int | None = None


class _BootstrapRunState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run: _BootstrapRun
    cursor: RunCursor
    selected_keys: list[str] | None = None
    selected_window: HistoryWindow | None = None
    history_tail: list[dict[str, Any]] = Field(default_factory=list)
    history_seen_count: int = 0
    max_step_seen: int | None = None


class _StreamingRowWriter:
    def __init__(self, *, path: Path, output_format: str, batch_size: int) -> None:
        self.path = path
        self.output_format = output_format
        self.batch_size = batch_size
        self._rows: list[dict[str, Any]] = []
        self._jsonl_handle: Any | None = None
        self._parquet_writer: pq.ParquetWriter | None = None
        self._parquet_schema: pa.Schema | None = None
        self._row_count = 0
        self._initialized = False

    def write_row(self, row: dict[str, Any]) -> None:
        self._rows.append(row)
        self._row_count += 1
        if len(self._rows) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_format == "jsonl":
            if self._jsonl_handle is None:
                self._jsonl_handle = open(self.path, "w", encoding="utf-8")
            for row in self._rows:
                self._jsonl_handle.write(
                    json.dumps(_to_jsonable(row), sort_keys=True, default=str) + "\n"
                )
            self._jsonl_handle.flush()
            self._rows = []
            self._initialized = True
            return

        df = safe_convert_for_parquet(pd.DataFrame(self._rows))
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self._parquet_writer is None:
            self._parquet_schema = table.schema
            self._parquet_writer = pq.ParquetWriter(self.path, table.schema)
        self._parquet_writer.write_table(table)
        self._rows = []
        self._initialized = True

    def close(self) -> None:
        self.flush()
        if self._jsonl_handle is not None:
            self._jsonl_handle.flush()
            os.fsync(self._jsonl_handle.fileno())
            self._jsonl_handle.close()
            self._jsonl_handle = None
            return
        if self._parquet_writer is not None:
            self._parquet_writer.close()
            self._parquet_writer = None
            with open(self.path, "rb") as f:
                os.fsync(f.fileno())
            return
        if not self._initialized:
            _atomic_write_rows(rows=[], path=self.path, output_format=self.output_format)

    @property
    def row_count(self) -> int:
        return self._row_count


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
        return _isoformat_or_none(getattr(run, "updated_at", None))

    def _coerce_run_created_at(self, run: Any) -> str | None:
        return _isoformat_or_none(getattr(run, "created_at", None))

    def _run_last_history_step(self, run: Any) -> int | None:
        attrs = getattr(run, "_attrs", None)
        if isinstance(attrs, dict):
            history_keys = attrs.get("historyKeys")
            if isinstance(history_keys, dict):
                last_step = history_keys.get("lastStep")
                if isinstance(last_step, int):
                    return last_step
        explicit_last_step = getattr(run, "last_history_step", None)
        if isinstance(explicit_last_step, int):
            return explicit_last_step
        history = getattr(run, "_history", None)
        if not isinstance(history, list):
            return None
        step_values = [
            entry.get("_step")
            for entry in history
            if isinstance(entry, dict) and isinstance(entry.get("_step"), int)
        ]
        if not step_values:
            return None
        return max(step_values)

    def _default_export_window(self, ctx: SyncContext) -> HistoryWindow:
        if ctx.cursor and ctx.cursor.last_step is not None:
            return HistoryWindow(min_step=ctx.cursor.last_step + 1)
        return HistoryWindow()

    def _chunked(self, values: list[str], size: int) -> list[list[str]]:
        return [values[i : i + size] for i in range(0, len(values), size)]

    def _iter_runs_query(
        self,
        api: Any,
        *,
        entity: str,
        project: str,
        runs_per_page: int,
        filters: dict[str, Any] | None = None,
    ) -> Any:
        return api.runs(
            f"{entity}/{project}",
            filters=filters or {},
            order="+created_at",
            per_page=runs_per_page,
            lazy=False,
        )

    def _select_runs_for_processing(
        self,
        api: Any,
        *,
        entity: str,
        project: str,
        state: ProjectSyncState,
        runs_per_page: int,
        fetch_mode: FetchMode,
    ) -> list[Any]:
        if fetch_mode == FetchMode.FULL_RECONCILE or state.max_created_at is None:
            return list(
                self._iter_runs_query(
                    api,
                    entity=entity,
                    project=project,
                    runs_per_page=runs_per_page,
                )
            )

        selected_runs: list[Any] = []
        seen_run_ids: set[str] = set()
        new_run_filters = {"createdAt": {"$gte": state.max_created_at}}
        for run in self._iter_runs_query(
            api,
            entity=entity,
            project=project,
            runs_per_page=runs_per_page,
            filters=new_run_filters,
        ):
            run_id = getattr(run, "id", "")
            if not run_id or run_id in seen_run_ids or run_id in state.runs:
                continue
            seen_run_ids.add(run_id)
            selected_runs.append(run)

        active_run_ids = [
            run_id
            for run_id, cursor in state.runs.items()
            if not cursor.terminal and run_id not in seen_run_ids
        ]
        for batch in self._chunked(active_run_ids, _ACTIVE_RUN_BATCH_SIZE):
            batch_filters = {"name": {"$in": batch}}
            for run in self._iter_runs_query(
                api,
                entity=entity,
                project=project,
                runs_per_page=runs_per_page,
                filters=batch_filters,
            ):
                run_id = getattr(run, "id", "")
                if not run_id or run_id in seen_run_ids:
                    continue
                seen_run_ids.add(run_id)
                selected_runs.append(run)

        return selected_runs

    def _summarize_selected_runs(
        self,
        *,
        state: ProjectSyncState,
        runs_obj: list[Any],
        fetch_mode: FetchMode,
    ) -> RunSelectionSummary:
        ignore_runs = 0
        terminal_runs = 0
        non_terminal_runs = 0
        for cursor in state.runs.values():
            if cursor.decision_status == "ignore":
                ignore_runs += 1
            elif cursor.terminal:
                terminal_runs += 1
            else:
                non_terminal_runs += 1

        selected_new_runs = 0
        selected_active_runs = 0
        for run in runs_obj:
            if getattr(run, "id", "") in state.runs:
                selected_active_runs += 1
            else:
                selected_new_runs += 1

        return RunSelectionSummary(
            fetch_mode=fetch_mode,
            tracked_runs=len(state.runs),
            terminal_runs=terminal_runs,
            ignore_runs=ignore_runs,
            non_terminal_runs=non_terminal_runs,
            selected_runs=len(runs_obj),
            selected_new_runs=selected_new_runs,
            selected_active_runs=selected_active_runs,
            skipped_tracked_runs=max(
                len(state.runs) - selected_active_runs,
                0,
            ),
        )

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

    def _is_terminal(self, ctx: SyncContext, decision: RunDecision) -> bool:
        return bool(
            self._with_retry(
                ctx,
                lambda: self.policy.is_terminal(ctx, decision),
                False,
            )
        )

    def _update_cursor(
        self,
        state: ProjectSyncState,
        ctx: SyncContext,
        decision: RunDecision,
        history_tail: list[dict[str, Any]],
        *,
        is_terminal: bool,
    ) -> None:
        history_seen = (ctx.cursor.history_seen if ctx.cursor else 0) + len(history_tail)
        last_step = self._max_history_step(
            history_tail,
            existing=(ctx.cursor.last_step if ctx.cursor else None),
        )
        if ctx.run_last_history_step is not None:
            last_step = (
                ctx.run_last_history_step
                if last_step is None
                else max(last_step, ctx.run_last_history_step)
            )
        state.runs[ctx.run_id] = RunCursor(
            run_id=ctx.run_id,
            updated_at=ctx.run_updated_at,
            last_step=last_step,
            history_seen=history_seen,
            terminal=is_terminal,
            decision_status=decision.status,
            decision_reason=decision.reason,
            metadata=decision.metadata,
        )
        created_at = self._coerce_run_created_at(ctx.run)
        if created_at is not None and (
            state.max_created_at is None or created_at > state.max_created_at
        ):
            state.max_created_at = created_at

    def sync_project(
        self,
        entity: str,
        project: str,
        *,
        fetch_mode: FetchMode = FetchMode.INCREMENTAL,
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
        runs_obj = self._select_runs_for_processing(
            api,
            entity=resolved_entity,
            project=resolved_project,
            state=state,
            runs_per_page=runs_per_page,
            fetch_mode=fetch_mode,
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
                run_last_history_step=self._run_last_history_step(run),
                run=run,
                cursor=cursor,
            )

            decision, patch, history_tail = self._evaluate_run(ctx)
            terminal = self._is_terminal(ctx, decision)
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
            self._update_cursor(
                state,
                ctx,
                decision,
                history_tail,
                is_terminal=terminal,
            )

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
        fetch_mode: FetchMode = FetchMode.INCREMENTAL,
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
        runs_obj = self._select_runs_for_processing(
            api,
            entity=resolved_entity,
            project=resolved_project,
            state=state,
            runs_per_page=runs_per_page,
            fetch_mode=fetch_mode,
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
                run_last_history_step=self._run_last_history_step(run),
                run=run,
                cursor=cursor,
            )
            decision, patch, history_tail = self._evaluate_run(ctx)
            self._update_cursor(
                state,
                ctx,
                decision,
                history_tail,
                is_terminal=self._is_terminal(ctx, decision),
            )

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

    def inspect_state(
        self,
        entity: str,
        project: str,
        *,
        state_path: Path | None = None,
        show_runs: str | None = None,
        limit: int = 20,
    ) -> StateInspectionSummary:
        resolved_state_path = state_path or default_state_path(entity, project)
        state = load_state(resolved_state_path, entity=entity, project=project)

        status_counts: dict[str, int] = {}
        terminal_count = 0
        ignore_count = 0
        non_terminal_count = 0

        for cursor in state.runs.values():
            status = cursor.decision_status or "unknown"
            status_counts[status] = status_counts.get(status, 0) + 1
            if status == "ignore":
                ignore_count += 1
            elif cursor.terminal:
                terminal_count += 1
            else:
                non_terminal_count += 1

        selected_runs: list[StateInspectionRun] = []
        if show_runs is not None:
            def matches_view(cursor: RunCursor) -> bool:
                if show_runs == "ignore":
                    return cursor.decision_status == "ignore"
                if show_runs == "terminal":
                    return cursor.terminal and cursor.decision_status != "ignore"
                return not cursor.terminal

            for run_id, cursor in sorted(state.runs.items()):
                if not matches_view(cursor):
                    continue
                selected_runs.append(
                    StateInspectionRun(
                        run_id=run_id,
                        updated_at=cursor.updated_at,
                        last_step=cursor.last_step,
                        history_seen=cursor.history_seen,
                        terminal=cursor.terminal,
                        decision_status=cursor.decision_status,
                        decision_reason=cursor.decision_reason,
                        metadata=cursor.metadata,
                    )
                )
                if len(selected_runs) >= limit:
                    break

        return StateInspectionSummary(
            entity=state.entity or entity,
            project=state.project or project,
            state_path=str(resolved_state_path),
            tracked_runs=len(state.runs),
            terminal_count=terminal_count,
            ignore_count=ignore_count,
            non_terminal_count=non_terminal_count,
            status_counts=dict(sorted(status_counts.items())),
            max_created_at=state.max_created_at,
            last_synced_at=state.last_synced_at,
            selected_view=cast(Any, show_runs),
            runs=selected_runs,
        )

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

    def _write_export_manifest(
        self,
        *,
        config: ExportConfig,
        runs_path: Path,
        history_path: Path,
        exported_at: str,
        run_count: int,
        history_count: int,
    ) -> Path:
        manifest_path = Path(config.output_dir) / f"{config.entity}_{config.project}_manifest.json"
        manifest = {
            "schema_version": 1,
            "entity": config.entity,
            "project": config.project,
            "output_format": config.output_format,
            "policy_module": self.policy.__class__.__module__,
            "policy_class": self.policy.__class__.__name__,
            "exported_at": exported_at,
            "runs_count": run_count,
            "history_count": history_count,
            "files": {
                "runs": str(runs_path),
                "history": str(history_path),
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return manifest_path

    def _read_export_frame(self, path: Path) -> pd.DataFrame:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".jsonl":
            if not path.exists() or path.stat().st_size == 0:
                return pd.DataFrame()
            return pd.read_json(path, lines=True)
        raise ValueError(f"Unsupported export file format for bootstrap: {path}")

    def _resolve_bootstrap_source_files(
        self,
        *,
        entity: str,
        project: str,
        source_dir: Path,
    ) -> _BootstrapSourceInfo:
        base = f"{entity}_{project}"
        manifest_path = source_dir / f"{base}_manifest.json"
        if manifest_path.exists():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_entity = payload.get("entity")
            manifest_project = payload.get("project")
            if manifest_entity not in (None, entity) or manifest_project not in (None, project):
                raise ValueError(
                    "Bootstrap source manifest mismatch: "
                    f"found {manifest_entity}/{manifest_project}, expected {entity}/{project}"
                )
            output_format = str(payload.get("output_format", "parquet"))
            files = payload.get("files", {})
            runs_ref = files.get("runs")
            history_ref = files.get("history")
            if isinstance(runs_ref, str) and isinstance(history_ref, str):
                runs_path = Path(runs_ref)
                history_path = Path(history_ref)
                if runs_path.exists() and history_path.exists():
                    return _BootstrapSourceInfo(
                        runs_path=runs_path,
                        history_path=history_path,
                        output_format=output_format,
                        manifest_path=manifest_path,
                        run_count=cast(int | None, payload.get("runs_count")),
                        history_count=cast(int | None, payload.get("history_count")),
                    )

        for output_format in ("parquet", "jsonl"):
            runs_path = source_dir / f"{base}_runs.{output_format}"
            history_path = source_dir / f"{base}_history.{output_format}"
            if runs_path.exists() and history_path.exists():
                return _BootstrapSourceInfo(
                    runs_path=runs_path,
                    history_path=history_path,
                    output_format=output_format,
                )

        raise ValueError(
            f"Could not find compact export files for {entity}/{project} in {source_dir}"
        )

    def _prepare_bootstrap_target(
        self,
        *,
        output_dir: Path,
        state_path: Path,
        overwrite_output: bool,
    ) -> None:
        if output_dir.exists():
            if not overwrite_output:
                raise ValueError(
                    f"Bootstrap output_dir already exists: {output_dir}. "
                    "Use --overwrite-output to replace it."
                )
            _remove_path(output_dir)
        if state_path.exists():
            if not overwrite_output:
                raise ValueError(
                    f"Bootstrap state_path already exists: {state_path}. "
                    "Use --overwrite-output to replace it."
                )
            _remove_path(state_path)

    def _create_bootstrap_staging_dir(self, output_dir: Path) -> Path:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        return Path(
            tempfile.mkdtemp(
                dir=output_dir.parent,
                prefix=f".{output_dir.name}.bootstrap-",
            )
        )

    def _coerce_export_dict(self, value: Any) -> dict[str, Any]:
        parsed = _parse_jsonish(value)
        if _is_missing_value(parsed):
            return {}
        if isinstance(parsed, dict):
            return dict(parsed)
        return {}

    def _coerce_export_list(self, value: Any) -> list[Any]:
        parsed = _parse_jsonish(value)
        if _is_missing_value(parsed):
            return []
        if isinstance(parsed, list):
            return list(parsed)
        return []

    def _coerce_export_str(self, value: Any) -> str | None:
        if _is_missing_value(value):
            return None
        return _isoformat_or_none(value)

    def _coerce_export_int(self, value: Any) -> int | None:
        if _is_missing_value(value):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _build_bootstrap_run(self, row: dict[str, Any]) -> _BootstrapRun:
        summary = self._coerce_export_dict(row.get("summary"))
        summary_metrics = self._coerce_export_dict(row.get("summary_metrics"))
        return _BootstrapRun(
            id=str(row.get("run_id", "")),
            name=str(row.get("run_name", "")),
            state=self._coerce_export_str(row.get("state")),
            created_at=self._coerce_export_str(row.get("created_at")),
            updated_at=self._coerce_export_str(row.get("updated_at")),
            config=self._coerce_export_dict(row.get("config")),
            summary_metrics=summary_metrics or summary,
            summary=summary or summary_metrics,
            metadata=self._coerce_export_dict(row.get("wandb_metadata")),
            system_metrics=self._coerce_export_dict(row.get("system_metrics")),
            tags=self._coerce_export_list(row.get("tags")),
            sweep_id=self._coerce_export_str(
                self._coerce_export_dict(row.get("sweep_info")).get("sweep_id")
            ),
            sweep_url=self._coerce_export_str(
                self._coerce_export_dict(row.get("sweep_info")).get("sweep_url")
            ),
        )

    def _deserialize_history_entry(self, row: dict[str, Any]) -> dict[str, Any]:
        entry: dict[str, Any] = {}
        for key in ("_step", "_timestamp", "_runtime"):
            value = row.get(key)
            if not _is_missing_value(value):
                entry[key] = value
        wandb_value = self._coerce_export_dict(row.get("_wandb"))
        if wandb_value:
            entry["_wandb"] = wandb_value
        metrics = self._coerce_export_dict(row.get("metrics"))
        if metrics:
            entry.update(metrics)
        return entry

    def _normalize_bootstrap_history_row(self, row: dict[str, Any]) -> dict[str, Any]:
        normalized = {
            "run_id": self._coerce_export_str(row.get("run_id")) or "",
            "_step": self._coerce_export_int(row.get("_step")),
            "_timestamp": self._coerce_export_str(row.get("_timestamp")),
            "_runtime": row.get("_runtime"),
            "_wandb": self._coerce_export_dict(row.get("_wandb")),
            "metrics": self._coerce_export_dict(row.get("metrics")),
        }
        return normalized

    def _history_row_matches_window(
        self, row: dict[str, Any], window: HistoryWindow | None
    ) -> bool:
        if window is None:
            return True
        step = self._coerce_export_int(row.get("_step"))
        if window.min_step is not None and step is not None and step < window.min_step:
            return False
        if window.max_step is not None and step is not None and step > window.max_step:
            return False
        return True

    def _append_bootstrap_history_entry(
        self,
        run_state: _BootstrapRunState,
        row: dict[str, Any],
    ) -> None:
        if run_state.selected_keys is None and run_state.selected_window is None:
            return
        if not self._history_row_matches_window(row, run_state.selected_window):
            return
        entry = self._deserialize_history_entry(row)
        if run_state.selected_keys is not None:
            selected_keys = set(run_state.selected_keys)
            entry = {
                key: value
                for key, value in entry.items()
                if key.startswith("_") or key in selected_keys
            }
        run_state.history_tail.append(entry)
        if (
            run_state.selected_window is not None
            and run_state.selected_window.max_records is not None
            and len(run_state.history_tail) > run_state.selected_window.max_records
        ):
            run_state.history_tail = run_state.history_tail[
                -run_state.selected_window.max_records :
            ]

    def _prepare_bootstrap_run_state(
        self,
        *,
        entity: str,
        project: str,
        row: dict[str, Any],
    ) -> _BootstrapRunState:
        run = self._build_bootstrap_run(row)
        last_step = self._coerce_export_int(row.get("last_step"))
        history_seen = self._coerce_export_int(row.get("history_seen")) or 0
        cursor = RunCursor(
            run_id=run.id,
            updated_at=run.updated_at,
            last_step=last_step,
            history_seen=history_seen,
        )
        ctx = SyncContext(
            entity=entity,
            project=project,
            run_id=run.id,
            run_name=run.name,
            run_state=run.state,
            run_updated_at=run.updated_at,
            run_last_history_step=last_step,
            run=run,
            cursor=cursor,
        )
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
        return _BootstrapRunState(
            run=run,
            cursor=cursor,
            selected_keys=selected_keys,
            selected_window=selected_window,
            history_seen_count=history_seen,
            max_step_seen=last_step,
        )

    def _iter_bootstrap_history_rows(
        self,
        source: _BootstrapSourceInfo,
    ) -> Any:
        history_path = source.history_path
        if source.output_format == "parquet":
            parquet_file = pq.ParquetFile(history_path)
            for batch in parquet_file.iter_batches(batch_size=_BOOTSTRAP_HISTORY_BATCH_SIZE):
                for row in batch.to_pylist():
                    yield row
            return

        with open(history_path, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                yield cast(dict[str, Any], json.loads(stripped))

    def bootstrap_export(self, config: BootstrapConfig) -> BootstrapSummary:
        started_at = time.monotonic()
        source_dir = Path(config.source_dir)
        output_dir = Path(config.output_dir)
        source = self._resolve_bootstrap_source_files(
            entity=config.entity,
            project=config.project,
            source_dir=source_dir,
        )
        output_format = config.output_format or cast(str, source.output_format)
        resolved_state_path = config.state_path or default_state_path(
            config.entity, config.project
        )
        if source_dir.resolve() == output_dir.resolve():
            raise ValueError("Bootstrap output_dir must differ from source_dir")
        self._prepare_bootstrap_target(
            output_dir=output_dir,
            state_path=resolved_state_path,
            overwrite_output=config.overwrite_output,
        )
        staging_output_dir = self._create_bootstrap_staging_dir(output_dir)
        staged_state_path = resolved_state_path.parent / f".{resolved_state_path.name}.bootstrap"
        staged_config = ExportConfig(
            entity=config.entity,
            project=config.project,
            output_dir=staging_output_dir,
            output_format=cast(Any, output_format),
            state_path=resolved_state_path,
            checkpoint_dirname=config.checkpoint_dirname,
            finalize_compact=config.finalize_compact,
            inspection_sample_rows=config.inspection_sample_rows,
            policy_module=config.policy_module,
            policy_class=config.policy_class,
        )
        writer = self._create_export_writer(staged_config)
        state = ProjectSyncState(entity=config.entity, project=config.project)

        logging.info(
            "Bootstrap source resolved: runs_path=%s history_path=%s source_format=%s manifest_path=%s",
            source.runs_path,
            source.history_path,
            source.output_format,
            source.manifest_path,
        )
        if source.history_count is None and source.output_format == "parquet":
            source.history_count = pq.ParquetFile(source.history_path).metadata.num_rows
        source_runs_df = self._read_export_frame(source.runs_path)
        logging.info(
            "Bootstrap source loaded: run_rows=%s history_rows=%s",
            len(source_runs_df),
            source.history_count if source.history_count is not None else "?",
        )
        deduped_runs_df = writer._dedupe_runs(source_runs_df)
        logging.info(
            "Bootstrap source deduped: run_rows=%s",
            len(deduped_runs_df),
        )
        deduped_run_rows = deduped_runs_df.to_dict(orient="records")
        total_runs = len(deduped_run_rows)
        run_states = {
            run_state.run.id: run_state
            for run_state in (
                self._prepare_bootstrap_run_state(
                    entity=config.entity,
                    project=config.project,
                    row=row,
                )
                for row in deduped_run_rows
            )
        }
        checkpoint_manifest = writer.begin_export()
        final_checkpoint_root = output_dir / config.checkpoint_dirname
        writer.manifest.checkpoint_dir = str(final_checkpoint_root)
        writer.manifest.runs_dir = str(final_checkpoint_root / "runs")
        writer.manifest.history_dir = str(final_checkpoint_root / "history")
        writer.manifest.inspection_path = str(final_checkpoint_root / "inspection.jsonl")
        writer._save_manifest()
        checkpoint_id = checkpoint_manifest.last_checkpoint_id + 1
        extension = writer._chunk_extension()
        runs_checkpoint_path = writer.runs_dir / f"chunk-{checkpoint_id:06d}{extension}"
        history_checkpoint_path = writer.history_dir / f"chunk-{checkpoint_id:06d}{extension}"
        physical_record_path = writer.checkpoint_root / f"checkpoint-{checkpoint_id:06d}.json"
        final_record_path = final_checkpoint_root / f"checkpoint-{checkpoint_id:06d}.json"
        history_writer = _StreamingRowWriter(
            path=history_checkpoint_path,
            output_format=staged_config.output_format,
            batch_size=_BOOTSTRAP_HISTORY_BATCH_SIZE,
        )
        history_rows_processed = 0
        matched_history_rows = 0
        step_min: int | None = None
        step_max: int | None = None
        metric_keys_sample: set[str] = set()
        logging.info(
            "Starting bootstrap history stream (history_rows=%s)",
            source.history_count if source.history_count is not None else "?",
        )
        for row in self._iter_bootstrap_history_rows(source):
            normalized_row = self._normalize_bootstrap_history_row(row)
            history_writer.write_row(normalized_row)
            history_rows_processed += 1
            step = self._coerce_export_int(normalized_row.get("_step"))
            if step is not None:
                step_min = step if step_min is None else min(step_min, step)
                step_max = step if step_max is None else max(step_max, step)
            metrics = self._coerce_export_dict(normalized_row.get("metrics"))
            if metrics:
                metric_keys_sample.update(str(key) for key in metrics)
            run_id = normalized_row["run_id"]
            run_state = run_states.get(run_id)
            if run_state is not None:
                matched_history_rows += 1
                run_state.history_seen_count += 1
                if step is not None:
                    run_state.max_step_seen = (
                        step
                        if run_state.max_step_seen is None
                        else max(run_state.max_step_seen, step)
                    )
                self._append_bootstrap_history_entry(run_state, normalized_row)
            if (
                history_rows_processed == 1
                or history_rows_processed % _BOOTSTRAP_HISTORY_LOG_EVERY == 0
            ):
                if source.history_count is None:
                    logging.info(
                        "Bootstrap history progress: processed_history_rows=%s matched_runs=%s",
                        history_rows_processed,
                        matched_history_rows,
                    )
                else:
                    logging.info(
                        "Bootstrap history progress: processed_history_rows=%s/%s (%.1f%%) matched_runs=%s",
                        history_rows_processed,
                        source.history_count,
                        _progress_percent(history_rows_processed, source.history_count),
                        matched_history_rows,
                    )
        history_writer.close()
        logging.info(
            "Bootstrap history stream complete: processed_history_rows=%s matched_history_rows=%s",
            history_rows_processed,
            matched_history_rows,
        )

        run_rows: list[dict[str, Any]] = []
        logging.info("Starting bootstrap run rebuild (selected_runs=%s)", total_runs)
        for processed_runs, run_state in enumerate(run_states.values(), start=1):
            last_step = run_state.cursor.last_step
            if last_step is None:
                last_step = run_state.max_step_seen
            cursor = run_state.cursor.model_copy(
                update={
                    "last_step": last_step,
                    "history_seen": run_state.history_seen_count,
                }
            )
            ctx = SyncContext(
                entity=config.entity,
                project=config.project,
                run_id=run_state.run.id,
                run_name=run_state.run.name,
                run_state=run_state.run.state,
                run_updated_at=run_state.run.updated_at,
                run_last_history_step=last_step,
                run=run_state.run,
                cursor=cursor,
            )
            decision = cast(
                RunDecision,
                self._with_retry(
                    ctx,
                    lambda: self.policy.classify_run(ctx, run_state.history_tail),
                    RunDecision(status="error", reason="classify_run failed"),
                ),
            )
            self._update_cursor(
                state,
                ctx,
                decision,
                run_state.history_tail,
                is_terminal=self._is_terminal(ctx, decision),
            )
            run_rows.append(
                _serialize_run_row(
                    run_state.run,
                    entity=config.entity,
                    project=config.project,
                    decision=decision,
                    cursor=state.runs[run_state.run.id],
                )
            )
            if processed_runs == 1 or processed_runs % _BOOTSTRAP_RUN_LOG_EVERY == 0:
                logging.info(
                    "Bootstrap progress: processed_runs=%s/%s (%.1f%%) buffered_run_rows=%s",
                    processed_runs,
                    total_runs,
                    _progress_percent(processed_runs, total_runs),
                    len(run_rows),
                )

        logging.info(
            "Writing merged bootstrap checkpoint: run_rows=%s history_rows=%s",
            len(run_rows),
            history_rows_processed,
        )
        _atomic_write_rows(
            rows=run_rows,
            path=runs_checkpoint_path,
            output_format=staged_config.output_format,
        )
        record = CheckpointRecord(
            checkpoint_id=checkpoint_id,
            created_at=_utc_now_iso(),
            run_rows=len(run_rows),
            history_rows=history_rows_processed,
            cumulative_run_rows=len(run_rows),
            cumulative_history_rows=history_rows_processed,
            run_start_index=1 if run_rows else 0,
            run_end_index=len(run_rows),
            run_ids=[
                str(row.get("run_id", ""))
                for row in run_rows[: staged_config.inspection_sample_rows]
            ],
            run_names=[
                str(row.get("run_name", ""))
                for row in run_rows[: staged_config.inspection_sample_rows]
            ],
            metric_keys_sample=sorted(metric_keys_sample)[: staged_config.inspection_sample_rows],
            step_min=step_min,
            step_max=step_max,
            runs_file=str(final_checkpoint_root / "runs" / runs_checkpoint_path.name),
            history_file=str(final_checkpoint_root / "history" / history_checkpoint_path.name),
            record_file=str(final_record_path),
            state_hash=_state_hash(state),
        )
        _atomic_write_json(physical_record_path, record.model_dump(mode="python"))
        checkpoint_manifest = writer.commit_checkpoint(
            record,
            processed_runs=len(run_rows),
            elapsed_seconds=max(time.monotonic() - started_at, 0.0001),
            checkpoint_elapsed_seconds=max(time.monotonic() - started_at, 0.0001),
        )
        logging.info(
            "Bootstrap checkpoint committed: checkpoint_id=%s processed_runs=%s/%s (%.1f%%)",
            record.checkpoint_id,
            len(run_rows),
            total_runs,
            _progress_percent(len(run_rows), total_runs),
        )
        bootstrapped_at = _utc_now_iso()
        state.last_synced_at = bootstrapped_at
        save_state(state, staged_state_path)

        if staged_config.finalize_compact:
            logging.info("Publishing compact bootstrap outputs from baseline checkpoint")
            base = f"{staged_config.entity}_{staged_config.project}"
            runs_output_path = staging_output_dir / f"{base}_runs.{staged_config.output_format}"
            history_output_path = staging_output_dir / f"{base}_history.{staged_config.output_format}"
            final_runs_output_path = output_dir / runs_output_path.name
            final_history_output_path = output_dir / history_output_path.name
            _copy_file(runs_checkpoint_path, runs_output_path)
            _copy_file(history_checkpoint_path, history_output_path)
            manifest_output_path = self._write_export_manifest(
                config=staged_config,
                runs_path=final_runs_output_path,
                history_path=final_history_output_path,
                exported_at=bootstrapped_at,
                run_count=len(run_rows),
                history_count=history_rows_processed,
            )
            writer.manifest.status = "completed"
            writer.manifest.updated_at = bootstrapped_at
            writer._save_manifest()
            finalized = True
        else:
            logging.info("Skipping compact bootstrap outputs because finalize_compact=False")
            writer.manifest.status = "completed_no_compact"
            writer.manifest.updated_at = bootstrapped_at
            writer._save_manifest()
            runs_output_path = writer.runs_dir
            history_output_path = writer.history_dir
            manifest_output_path = writer.manifest_path
            finalized = False

        os.replace(staging_output_dir, output_dir)
        os.replace(staged_state_path, resolved_state_path)
        checkpoint_manifest = writer.manifest
        if finalized:
            final_manifest_output_path = output_dir / manifest_output_path.name
        else:
            final_runs_output_path = output_dir / config.checkpoint_dirname / "runs"
            final_history_output_path = output_dir / config.checkpoint_dirname / "history"
            final_manifest_output_path = output_dir / config.checkpoint_dirname / "manifest.json"
        final_checkpoint_manifest_path = output_dir / config.checkpoint_dirname / "manifest.json"
        elapsed_seconds = time.monotonic() - started_at
        logging.info(
            "Bootstrap complete: processed_runs=%s/%s (%.1f%%) run_count=%s history_rows=%s checkpoints=%s finalized=%s elapsed=%.2fs runs_path=%s history_path=%s",
            len(run_rows),
            total_runs,
            _progress_percent(len(run_rows), total_runs),
            len(run_rows),
            history_rows_processed,
            len(checkpoint_manifest.checkpoints),
            finalized,
            elapsed_seconds,
            final_runs_output_path,
            final_history_output_path,
        )
        return BootstrapSummary(
            entity=config.entity,
            project=config.project,
            source_dir=str(source_dir),
            output_dir=str(output_dir),
            state_path=str(resolved_state_path),
            output_format=cast(Any, output_format),
            runs_output_path=str(final_runs_output_path),
            history_output_path=str(final_history_output_path),
            manifest_output_path=str(final_manifest_output_path),
            checkpoint_manifest_path=str(final_checkpoint_manifest_path),
            run_count=len(run_rows),
            history_count=history_rows_processed,
            checkpoint_count=len(checkpoint_manifest.checkpoints),
            bootstrapped_at=bootstrapped_at,
        )

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
            "Building runs iterator for %s/%s (runs_per_page=%s, fetch_mode=%s)",
            resolved_entity,
            resolved_project,
            config.runs_per_page,
            config.fetch_mode,
        )
        runs_obj = self._select_runs_for_processing(
            api,
            entity=resolved_entity,
            project=resolved_project,
            state=state,
            runs_per_page=config.runs_per_page,
            fetch_mode=config.fetch_mode,
        )
        selection_summary = self._summarize_selected_runs(
            state=state,
            runs_obj=runs_obj,
            fetch_mode=config.fetch_mode,
        )
        total_runs = len(runs_obj)
        if config.fetch_mode == FetchMode.INCREMENTAL:
            logging.info(
                "Incremental selection summary: tracked_runs=%s non_terminal_runs=%s "
                "terminal_runs=%s ignore_runs=%s selected_active_runs=%s "
                "selected_new_runs=%s skipped_tracked_runs=%s",
                selection_summary.tracked_runs,
                selection_summary.non_terminal_runs,
                selection_summary.terminal_runs,
                selection_summary.ignore_runs,
                selection_summary.selected_active_runs,
                selection_summary.selected_new_runs,
                selection_summary.skipped_tracked_runs,
            )
        logging.info("Starting run iteration (selected_runs=%s)", total_runs)
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
                    run_last_history_step=self._run_last_history_step(run),
                    run=run,
                    cursor=cursor,
                )

                decision, _patch, history_tail = self._evaluate_run(
                    ctx,
                    include_history_when_unspecified=True,
                )
                self._update_cursor(
                    state,
                    ctx,
                    decision,
                    history_tail,
                    is_terminal=self._is_terminal(ctx, decision),
                )
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
                        "Export progress: processed_runs=%s/%s (%.1f%%) history_rows=%s (state saved)",
                        processed_runs,
                        total_runs,
                        _progress_percent(processed_runs, total_runs),
                        len(history_rows),
                    )
                elif processed_runs == 1:
                    logging.info(
                        "Export progress: processed_runs=%s/%s (%.1f%%) history_rows=%s",
                        processed_runs,
                        total_runs,
                        _progress_percent(processed_runs, total_runs),
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
                "Export complete: processed_runs=%s/%s (%.1f%%) run_count=%s history_rows=%s elapsed=%.2fs runs_file=%s history_file=%s",
                processed_runs,
                total_runs,
                _progress_percent(processed_runs, total_runs),
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
                "Checkpoint %s committed: processed_runs=%s/%s (%.1f%%) run_rows=%s history_rows=%s total_run_rows=%s total_history_rows=%s",
                record.checkpoint_id,
                processed_runs,
                total_runs,
                _progress_percent(processed_runs, total_runs),
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
                run_last_history_step=self._run_last_history_step(run),
                run=run,
                cursor=cursor,
            )

            decision, _patch, history_tail = self._evaluate_run(
                ctx,
                include_history_when_unspecified=True,
            )
            self._update_cursor(
                state,
                ctx,
                decision,
                history_tail,
                is_terminal=self._is_terminal(ctx, decision),
            )
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
            elif processed_runs == 1:
                logging.info(
                    "Export progress: processed_runs=%s/%s (%.1f%%) buffered_run_rows=%s buffered_history_rows=%s",
                    processed_runs,
                    total_runs,
                    _progress_percent(processed_runs, total_runs),
                    len(run_rows_batch),
                    len(history_rows_batch),
                )
            elif processed_runs % resolved_config.save_every == 0:
                logging.info(
                    "Export progress: processed_runs=%s/%s (%.1f%%) buffered_run_rows=%s buffered_history_rows=%s",
                    processed_runs,
                    total_runs,
                    _progress_percent(processed_runs, total_runs),
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
            "Export complete: processed_runs=%s/%s (%.1f%%) run_count=%s history_rows=%s checkpoints=%s finalized=%s elapsed=%.2fs runs_path=%s history_path=%s",
            processed_runs,
            total_runs,
            _progress_percent(processed_runs, total_runs),
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
