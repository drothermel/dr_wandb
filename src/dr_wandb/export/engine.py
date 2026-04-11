from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from dr_wandb.export.models import (
    ExportManifest,
    ExportMode,
    ExportRequest,
    ExportState,
    ExportSummary,
    FetchMode,
    HistoryPolicyContext,
    HistoryRow,
    HistoryWindow,
    RunSnapshot,
    RunTrackingState,
)
from dr_wandb.export.store import (
    HISTORY_ROW_JSON_COLUMNS,
    RUN_SNAPSHOT_JSON_COLUMNS,
    history_path,
    load_manifest,
    load_state,
    read_records,
    remove_if_exists,
    resolve_export_paths,
    runs_path,
    save_manifest,
    save_state,
    write_records,
)
from dr_ds.serialization import serialize_timestamp, to_jsonable, utc_now_iso

TERMINAL_RUN_STATES = {"finished", "failed", "crashed", "killed"}
RUN_BATCH_SIZE = 100


def _build_default_api(timeout_seconds: int) -> Any:
    import wandb

    return wandb.Api(timeout=timeout_seconds)


class ExportEngine:
    def __init__(
        self,
        *,
        api_factory: Callable[[int], Any] | None = None,
    ) -> None:
        self.api_factory = api_factory or _build_default_api

    def export(self, request: ExportRequest) -> ExportSummary:
        paths = resolve_export_paths(name=request.name, data_root=request.data_root)
        state = load_state(paths, entity=request.entity, project=request.project)
        state.name = request.name
        state.entity = request.entity
        state.project = request.project
        existing_manifest = load_manifest(paths)
        existing_snapshots = self._load_existing_snapshots(
            request=request,
            manifest=existing_manifest,
        )
        existing_history_rows = self._load_existing_history_rows(
            request=request,
            manifest=existing_manifest,
        )

        if request.fetch_mode == FetchMode.FULL_RECONCILE:
            state = ExportState(
                name=request.name,
                entity=request.entity,
                project=request.project,
            )
            existing_snapshots = {}
            existing_history_rows = []

        api = self.api_factory(request.timeout_seconds)
        runs = self._select_runs(api=api, request=request, state=state)
        exported_at = utc_now_iso()

        snapshot_by_id = dict(existing_snapshots)
        new_history_rows: list[HistoryRow] = []

        for run in runs:
            run_id = str(getattr(run, "id", ""))
            if run_id == "":
                continue
            tracking = state.runs.get(run_id)
            ctx = HistoryPolicyContext(
                entity=request.entity,
                project=request.project,
                run_id=run_id,
                run_name=str(getattr(run, "name", "")),
                run_state=self._run_state(run),
                run_updated_at=serialize_timestamp(getattr(run, "updated_at", None)),
                run_last_history_step=(
                    tracking.last_history_step if tracking is not None else None
                ),
                run=run,
            )
            raw_run = self._build_raw_run_payload(
                run=run,
                include_metadata=request.include_metadata,
            )
            snapshot_by_id[run_id] = RunSnapshot(
                run_id=run_id,
                entity=request.entity,
                project=request.project,
                exported_at=exported_at,
                raw_run=raw_run,
            )

            tracking_state = RunTrackingState(
                run_id=run_id,
                created_at=serialize_timestamp(getattr(run, "created_at", None)),
                updated_at=serialize_timestamp(getattr(run, "updated_at", None)),
                run_state=self._run_state(run),
                last_history_step=(
                    tracking.last_history_step if tracking is not None else None
                ),
            )

            if request.mode == ExportMode.HISTORY:
                history_rows = self._scan_history_for_export(
                    request=request,
                    ctx=ctx,
                )
                new_history_rows.extend(history_rows)
                max_step = self._max_history_step(history_rows)
                observed_last_step = self._observed_last_history_step(run)
                if observed_last_step is not None:
                    max_step = (
                        observed_last_step
                        if max_step is None
                        else max(max_step, observed_last_step)
                    )
                tracking_state.last_history_step = max_step

            state.runs[run_id] = tracking_state
            if tracking_state.created_at is not None and (
                state.max_created_at is None
                or tracking_state.created_at > state.max_created_at
            ):
                state.max_created_at = tracking_state.created_at

        snapshots = self._sorted_snapshots(snapshot_by_id.values())
        if request.mode == ExportMode.HISTORY:
            history_rows = self._merge_history_rows(
                existing_history_rows=existing_history_rows,
                new_history_rows=new_history_rows,
            )
        else:
            history_rows = []

        state.last_exported_at = exported_at
        save_state(paths, state)

        runs_output_path = runs_path(paths, request.output_format)
        history_output_path = history_path(paths, request.output_format)
        write_records(
            runs_output_path,
            [snapshot.model_dump(mode="python") for snapshot in snapshots],
            json_columns=RUN_SNAPSHOT_JSON_COLUMNS,
        )
        if request.mode == ExportMode.HISTORY:
            write_records(
                history_output_path,
                [row.model_dump(mode="python") for row in history_rows],
                json_columns=HISTORY_ROW_JSON_COLUMNS,
            )
        else:
            remove_if_exists(history_output_path)
        for other_format in {"jsonl", "parquet"} - {request.output_format}:
            remove_if_exists(runs_path(paths, other_format))
            remove_if_exists(history_path(paths, other_format))

        selected_history_keys = self._selected_history_keys(
            request=request,
            runs=runs,
        )
        selected_history_window = self._selected_history_window(
            request=request,
            runs=runs,
        )
        manifest = ExportManifest(
            name=request.name,
            entity=request.entity,
            project=request.project,
            mode=request.mode,
            output_format=request.output_format,
            created_at=(
                existing_manifest.created_at
                if existing_manifest is not None
                else exported_at
            ),
            updated_at=exported_at,
            runs_path=str(runs_output_path),
            history_path=(
                str(history_output_path) if request.mode == ExportMode.HISTORY else None
            ),
            run_count=len(snapshots),
            history_count=len(history_rows),
            selected_history_keys=selected_history_keys,
            history_window=selected_history_window,
        )
        save_manifest(paths, manifest)
        return ExportSummary(
            name=request.name,
            entity=request.entity,
            project=request.project,
            mode=request.mode,
            fetch_mode=request.fetch_mode,
            output_dir=str(paths.export_dir),
            state_path=str(paths.state_path),
            manifest_path=str(paths.manifest_path),
            runs_path=str(runs_output_path),
            history_path=(
                str(history_output_path) if request.mode == ExportMode.HISTORY else None
            ),
            run_count=len(snapshots),
            history_count=len(history_rows),
            exported_at=exported_at,
        )

    def _load_existing_snapshots(
        self,
        *,
        request: ExportRequest,
        manifest: ExportManifest | None,
    ) -> dict[str, RunSnapshot]:
        if request.fetch_mode == FetchMode.FULL_RECONCILE or manifest is None:
            return {}
        records = read_records(
            path=Path(manifest.runs_path),
            json_columns=RUN_SNAPSHOT_JSON_COLUMNS,
        )
        return {
            snapshot.run_id: snapshot
            for snapshot in (RunSnapshot.model_validate(record) for record in records)
        }

    def _load_existing_history_rows(
        self,
        *,
        request: ExportRequest,
        manifest: ExportManifest | None,
    ) -> list[HistoryRow]:
        if (
            request.fetch_mode == FetchMode.FULL_RECONCILE
            or request.mode != ExportMode.HISTORY
            or manifest is None
            or manifest.history_path is None
        ):
            return []
        records = read_records(
            path=Path(manifest.history_path),
            json_columns=HISTORY_ROW_JSON_COLUMNS,
        )
        return [HistoryRow.model_validate(record) for record in records]

    def _select_runs(
        self,
        *,
        api: Any,
        request: ExportRequest,
        state: ExportState,
    ) -> list[Any]:
        if (
            request.fetch_mode == FetchMode.FULL_RECONCILE
            or state.max_created_at is None
        ):
            return list(
                self._iter_runs(
                    api=api,
                    entity=request.entity,
                    project=request.project,
                    runs_per_page=request.runs_per_page,
                )
            )

        selected_runs: list[Any] = []
        seen_run_ids: set[str] = set()
        for run in self._iter_runs(
            api=api,
            entity=request.entity,
            project=request.project,
            runs_per_page=request.runs_per_page,
            filters={"createdAt": {"$gte": state.max_created_at}},
        ):
            run_id = str(getattr(run, "id", ""))
            if run_id == "" or run_id in seen_run_ids or run_id in state.runs:
                continue
            seen_run_ids.add(run_id)
            selected_runs.append(run)

        refresh_ids = [
            run_id
            for run_id, tracking in state.runs.items()
            if self._should_refresh_tracking_state(tracking)
            and run_id not in seen_run_ids
        ]
        for batch in self._chunked(refresh_ids, RUN_BATCH_SIZE):
            for run in self._iter_runs(
                api=api,
                entity=request.entity,
                project=request.project,
                runs_per_page=request.runs_per_page,
                filters={"name": {"$in": batch}},
            ):
                run_id = str(getattr(run, "id", ""))
                if run_id == "" or run_id in seen_run_ids:
                    continue
                seen_run_ids.add(run_id)
                selected_runs.append(run)
        return selected_runs

    def _iter_runs(
        self,
        *,
        api: Any,
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

    def _chunked(self, values: list[str], size: int) -> list[list[str]]:
        return [values[index : index + size] for index in range(0, len(values), size)]

    def _should_refresh_tracking_state(self, tracking: RunTrackingState) -> bool:
        return tracking.run_state not in TERMINAL_RUN_STATES

    def _run_state(self, run: Any) -> str | None:
        value = getattr(run, "state", None)
        return str(value) if value is not None else None

    def _scan_history_for_export(
        self,
        *,
        request: ExportRequest,
        ctx: HistoryPolicyContext,
    ) -> list[HistoryRow]:
        if request.mode != ExportMode.HISTORY:
            return []
        keys = (
            request.history_policy.select_history_keys(ctx)
            if request.history_policy is not None
            else None
        )
        window = (
            request.history_policy.select_history_window(ctx)
            if request.history_policy is not None
            else None
        )
        if request.fetch_mode == FetchMode.INCREMENTAL and window is None:
            window = self._default_history_window(ctx)
        entries = self._scan_history(run=ctx.run, keys=keys, window=window)
        return [self._build_history_row(ctx.run_id, entry) for entry in entries]

    def _selected_history_keys(
        self, *, request: ExportRequest, runs: list[Any]
    ) -> list[str] | None:
        if request.history_policy is None or len(runs) == 0:
            return None
        first_run = runs[0]
        ctx = HistoryPolicyContext(
            entity=request.entity,
            project=request.project,
            run_id=str(getattr(first_run, "id", "")),
            run_name=str(getattr(first_run, "name", "")),
            run_state=self._run_state(first_run),
            run_updated_at=serialize_timestamp(getattr(first_run, "updated_at", None)),
            run_last_history_step=self._observed_last_history_step(first_run),
            run=first_run,
        )
        keys = request.history_policy.select_history_keys(ctx)
        return list(keys) if keys is not None else None

    def _selected_history_window(
        self, *, request: ExportRequest, runs: list[Any]
    ) -> HistoryWindow | None:
        if request.history_policy is None or len(runs) == 0:
            return None
        first_run = runs[0]
        ctx = HistoryPolicyContext(
            entity=request.entity,
            project=request.project,
            run_id=str(getattr(first_run, "id", "")),
            run_name=str(getattr(first_run, "name", "")),
            run_state=self._run_state(first_run),
            run_updated_at=serialize_timestamp(getattr(first_run, "updated_at", None)),
            run_last_history_step=self._observed_last_history_step(first_run),
            run=first_run,
        )
        return request.history_policy.select_history_window(ctx)

    def _default_history_window(
        self, ctx: HistoryPolicyContext
    ) -> HistoryWindow | None:
        if ctx.run_last_history_step is None:
            return None
        return HistoryWindow(min_step=ctx.run_last_history_step + 1)

    def _scan_history(
        self,
        *,
        run: Any,
        keys: list[str] | None,
        window: HistoryWindow | None,
    ) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {}
        if keys is not None:
            kwargs["keys"] = keys
        if window is not None:
            if window.min_step is not None:
                kwargs["min_step"] = window.min_step
            if window.max_step is not None:
                kwargs["max_step"] = window.max_step
        try:
            entries = list(run.scan_history(**kwargs))
        except TypeError:
            kwargs.pop("max_step", None)
            entries = list(run.scan_history(**kwargs))
        if window is not None and window.max_records is not None:
            entries = entries[-window.max_records :]
        return entries

    def _observed_last_history_step(self, run: Any) -> int | None:
        attrs = getattr(run, "_attrs", None)
        if isinstance(attrs, dict):
            history_keys = attrs.get("historyKeys")
            if isinstance(history_keys, dict):
                last_step = history_keys.get("lastStep")
                if isinstance(last_step, int):
                    return last_step
        value = getattr(run, "last_history_step", None)
        if isinstance(value, int):
            return value
        return None

    def _build_history_row(self, run_id: str, entry: dict[str, Any]) -> HistoryRow:
        wandb_value = entry.get("_wandb")
        return HistoryRow(
            run_id=run_id,
            step=entry.get("_step") if isinstance(entry.get("_step"), int) else None,
            timestamp=serialize_timestamp(entry.get("_timestamp")),
            runtime=entry.get("_runtime"),
            wandb_metadata=dict(wandb_value) if isinstance(wandb_value, dict) else {},
            metrics={
                str(key): value
                for key, value in entry.items()
                if not str(key).startswith("_")
            },
            extra={
                str(key): value
                for key, value in entry.items()
                if str(key).startswith("_")
                and str(key) not in {"_step", "_timestamp", "_runtime", "_wandb"}
            },
        )

    def _max_history_step(self, rows: list[HistoryRow]) -> int | None:
        step_values = [row.step for row in rows if row.step is not None]
        if len(step_values) == 0:
            return None
        return max(step_values)

    def _merge_history_rows(
        self,
        *,
        existing_history_rows: list[HistoryRow],
        new_history_rows: list[HistoryRow],
    ) -> list[HistoryRow]:
        by_key: dict[tuple[Any, ...], HistoryRow] = {}
        for row in [*existing_history_rows, *new_history_rows]:
            by_key[self._history_row_key(row)] = row
        merged = list(by_key.values())
        merged.sort(
            key=lambda row: (
                row.run_id,
                row.step if row.step is not None else float("inf"),
                str(row.timestamp or ""),
            )
        )
        return merged

    def _history_row_key(self, row: HistoryRow) -> tuple[Any, ...]:
        payload_hash = hashlib.sha256(
            json.dumps(
                to_jsonable(
                    {
                        "step": row.step,
                        "timestamp": row.timestamp,
                        "runtime": row.runtime,
                        "wandb_metadata": row.wandb_metadata,
                        "metrics": row.metrics,
                        "extra": row.extra,
                    }
                ),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if row.step is not None and row.timestamp is not None:
            return (row.run_id, row.step, row.timestamp, payload_hash)
        if row.step is not None:
            return (row.run_id, row.step, payload_hash)
        if row.timestamp is not None:
            return (row.run_id, row.timestamp, payload_hash)
        return (row.run_id, payload_hash)

    def _sorted_snapshots(self, snapshots: Any) -> list[RunSnapshot]:
        return sorted(
            snapshots,
            key=lambda snapshot: (
                str(snapshot.raw_run.get("createdAt", "")),
                snapshot.run_id,
            ),
            reverse=True,
        )

    def _build_raw_run_payload(
        self, *, run: Any, include_metadata: bool
    ) -> dict[str, Any]:
        payload = self._attrs_payload(run)
        if not include_metadata:
            payload.pop("metadata", None)
        payload = self._drop_duplicate_aliases(payload)
        self._fill_if_missing(payload, "config", self._public_attr(run, "config"))
        self._fill_summary(payload, run)
        self._fill_if_missing(
            payload,
            "systemMetrics",
            self._public_attr(run, "system_metrics"),
        )
        if include_metadata:
            self._fill_if_missing(
                payload,
                "metadata",
                self._public_attr(run, "metadata"),
            )
        self._fill_if_missing(payload, "tags", self._public_attr(run, "tags"))
        self._fill_if_missing(payload, "group", self._public_attr(run, "group"))
        self._fill_if_missing(
            payload,
            "historyKeys",
            self._public_attr(run, "history_keys"),
        )
        self._fill_if_missing(
            payload,
            "jobType",
            self._public_attr(run, "job_type"),
        )
        self._fill_if_missing(
            payload,
            "sweepName",
            self._public_attr(run, "sweep_name"),
        )
        self._fill_if_missing(payload, "user", self._public_attr(run, "user"))
        self._fill_if_missing(
            payload,
            "readOnly",
            self._public_attr(run, "read_only"),
        )
        self._fill_if_missing(
            payload,
            "createdAt",
            self._public_attr(run, "created_at"),
        )
        self._fill_if_missing(
            payload,
            "updatedAt",
            self._public_attr(run, "updated_at"),
        )
        self._fill_if_missing(
            payload,
            "heartbeatAt",
            self._public_attr(run, "heartbeat_at"),
        )
        self._fill_if_missing(
            payload,
            "storageId",
            self._public_attr(run, "storage_id"),
        )
        self._fill_if_missing(payload, "url", self._public_attr(run, "url"))
        self._fill_if_missing(payload, "path", self._public_attr(run, "path"))
        self._fill_if_missing(
            payload,
            "displayName",
            self._public_attr(run, "display_name"),
        )
        self._fill_if_missing(payload, "name", self._public_attr(run, "name"))
        self._fill_if_missing(payload, "state", self._public_attr(run, "state"))
        payload.pop("summary", None)
        payload.pop("summary_metrics", None)
        payload.pop("system_metrics", None)
        payload.pop("history_keys", None)
        payload.pop("job_type", None)
        payload.pop("sweep_name", None)
        payload.pop("read_only", None)
        payload.pop("created_at", None)
        payload.pop("updated_at", None)
        payload.pop("heartbeat_at", None)
        payload.pop("storage_id", None)
        return to_jsonable(payload)

    def _attrs_payload(self, run: Any) -> dict[str, Any]:
        raw_attrs = getattr(run, "_attrs", None)
        if not isinstance(raw_attrs, dict):
            return {}
        return to_jsonable(raw_attrs)

    def _drop_duplicate_aliases(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = dict(payload)
        if "systemMetrics" in result:
            result.pop("system_metrics", None)
        if "historyKeys" in result:
            result.pop("history_keys", None)
        if "jobType" in result:
            result.pop("job_type", None)
        if "sweepName" in result:
            result.pop("sweep_name", None)
        if "readOnly" in result:
            result.pop("read_only", None)
        if "createdAt" in result:
            result.pop("created_at", None)
        if "updatedAt" in result:
            result.pop("updated_at", None)
        if "heartbeatAt" in result:
            result.pop("heartbeat_at", None)
        if "storageId" in result:
            result.pop("storage_id", None)
        return result

    def _fill_if_missing(self, payload: dict[str, Any], key: str, value: Any) -> None:
        if key in payload and payload[key] is not None:
            return
        if value is None:
            return
        payload[key] = value

    def _fill_summary(self, payload: dict[str, Any], run: Any) -> None:
        if payload.get("summaryMetrics") is not None:
            return
        summary_metrics = self._public_attr(run, "summary_metrics")
        if summary_metrics is not None:
            payload["summaryMetrics"] = summary_metrics
            return
        summary = self._public_attr(run, "summary")
        if summary is not None:
            payload["summaryMetrics"] = summary

    def _public_attr(self, run: Any, attr_name: str) -> Any:
        if not hasattr(run, attr_name):
            return None
        return self._coerce_public_value(getattr(run, attr_name))

    def _coerce_public_value(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return {
                str(key): self._coerce_public_value(nested)
                for key, nested in value.items()
            }
        if isinstance(value, list):
            return [self._coerce_public_value(nested) for nested in value]
        if isinstance(value, tuple):
            return [self._coerce_public_value(nested) for nested in value]
        if isinstance(value, set):
            return sorted(self._coerce_public_value(nested) for nested in value)
        raw_attrs = getattr(value, "_attrs", None)
        if isinstance(raw_attrs, dict):
            return {
                str(key): self._coerce_public_value(nested)
                for key, nested in raw_attrs.items()
            }
        if isinstance(value, Mapping):
            return {
                str(key): self._coerce_public_value(nested)
                for key, nested in value.items()
            }
        if hasattr(value, "items") and callable(getattr(value, "items", None)):
            return {
                str(key): self._coerce_public_value(nested)
                for key, nested in value.items()
            }
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            return [self._coerce_public_value(nested) for nested in value]
        return str(value)
