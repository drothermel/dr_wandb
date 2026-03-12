from __future__ import annotations

from collections.abc import Mapping, Sized
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import srsly
import wandb
from pydantic import BaseModel, Field


class RawExtractConfig(BaseModel):
    entity: str
    project: str
    output_dir: Path
    runs_raw_path: Path
    runs_raw_deduping_path: Path
    runs_per_page: int = 100
    timeout_seconds: int = 300
    include_metadata: bool = False
    postprocess_dedup: bool = True


class RawRunRecord(BaseModel):
    run_id: str
    entity: str
    project: str
    exported_at: str
    raw_run_hash: str
    raw_run: dict[str, Any] = Field(default_factory=dict)


class RawExtractSummary(BaseModel):
    entity: str
    project: str
    output_dir: str
    runs_output_path: str
    run_count: int
    exported_at: str
    runs_per_page: int
    include_metadata: bool = False
    postprocess_dedup: bool = True
    final_run_count: int | None = None


def extract_runs_raw(config: RawExtractConfig) -> RawExtractSummary:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=config.timeout_seconds)
    runs_iter = api.runs(
        f"{config.entity}/{config.project}",
        order="-created_at",
        per_page=config.runs_per_page,
        lazy=False,
    )
    total_runs = _maybe_total_runs(runs_iter)

    run_count = 0
    with config.runs_raw_path.open("a", encoding="utf-8") as handle:
        for run in runs_iter:
            record = build_raw_run_record(
                run=run,
                entity=config.entity,
                project=config.project,
                include_metadata=config.include_metadata,
            )
            handle.write(srsly.json_dumps(record.model_dump(mode="python")) + "\n")
            handle.flush()
            run_count += 1
            if run_count % config.runs_per_page == 0:
                _log_progress(
                    run_count=run_count,
                    total_runs=total_runs,
                    entity=config.entity,
                    project=config.project,
                )

    if run_count % config.runs_per_page != 0:
        _log_progress(
            run_count=run_count,
            total_runs=total_runs,
            entity=config.entity,
            project=config.project,
        )

    final_run_count: int | None = None
    if config.postprocess_dedup:
        logging.info("Starting postprocess dedupe for %s", config.runs_raw_path)
        final_run_count = postprocess_dedupe_runs_raw(
            input_path=config.runs_raw_path,
            temp_output_path=config.runs_raw_deduping_path,
        )

    exported_at = _utc_now_iso()
    return RawExtractSummary(
        entity=config.entity,
        project=config.project,
        output_dir=str(config.output_dir),
        runs_output_path=str(config.runs_raw_path),
        run_count=run_count,
        exported_at=exported_at,
        runs_per_page=config.runs_per_page,
        include_metadata=config.include_metadata,
        postprocess_dedup=config.postprocess_dedup,
        final_run_count=final_run_count,
    )


def build_raw_run_record(
    *,
    run: Any,
    entity: str,
    project: str,
    include_metadata: bool,
) -> RawRunRecord:
    raw_run = build_raw_run_payload(run=run, include_metadata=include_metadata)
    exported_at = _utc_now_iso()
    return RawRunRecord(
        run_id=str(getattr(run, "id")),
        entity=entity,
        project=project,
        exported_at=exported_at,
        raw_run_hash=hash_raw_run_payload(raw_run),
        raw_run=raw_run,
    )


def build_raw_run_payload(*, run: Any, include_metadata: bool) -> dict[str, Any]:
    payload = _attrs_payload(run)

    if not include_metadata:
        payload.pop("metadata", None)

    payload = _drop_duplicate_aliases(payload)

    _fill_if_missing(payload, "config", _public_attr(run, "config"))
    _fill_summary(payload, run)
    _fill_if_missing(payload, "systemMetrics", _public_attr(run, "system_metrics"))
    if include_metadata:
        _fill_if_missing(payload, "metadata", _public_attr(run, "metadata"))
    _fill_if_missing(payload, "tags", _public_attr(run, "tags"))
    _fill_if_missing(payload, "group", _public_attr(run, "group"))
    _fill_if_missing(payload, "historyKeys", _public_attr(run, "history_keys"))
    _fill_if_missing(payload, "jobType", _public_attr(run, "job_type"))
    _fill_if_missing(payload, "sweepName", _public_attr(run, "sweep_name"))
    _fill_if_missing(payload, "user", _public_attr(run, "user"))
    _fill_if_missing(payload, "readOnly", _public_attr(run, "read_only"))
    _fill_if_missing(payload, "createdAt", _public_attr(run, "created_at"))
    _fill_if_missing(payload, "updatedAt", _public_attr(run, "updated_at"))
    _fill_if_missing(payload, "heartbeatAt", _public_attr(run, "heartbeat_at"))
    _fill_if_missing(payload, "storageId", _public_attr(run, "storage_id"))
    _fill_if_missing(payload, "url", _public_attr(run, "url"))
    _fill_if_missing(payload, "path", _public_attr(run, "path"))
    _fill_if_missing(payload, "displayName", _public_attr(run, "display_name"))
    _fill_if_missing(payload, "name", _public_attr(run, "name"))
    _fill_if_missing(payload, "state", _public_attr(run, "state"))

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
    return _to_jsonable(payload)


def hash_raw_run_payload(raw_run: dict[str, Any]) -> str:
    payload = json.dumps(raw_run, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def postprocess_dedupe_runs_raw(input_path: Path, temp_output_path: Path) -> int:
    records = _load_records_for_postprocess(input_path)

    seen: set[tuple[str, str, str, str]] = set()
    kept_records: list[RawRunRecord] = []
    for record in reversed(records):
        dedupe_key = (
            record.project,
            record.entity,
            record.run_id,
            record.raw_run_hash,
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        kept_records.append(record)

    kept_records.reverse()
    temp_output_path.parent.mkdir(parents=True, exist_ok=True)
    with temp_output_path.open("w", encoding="utf-8") as handle:
        for record in kept_records:
            handle.write(srsly.json_dumps(record.model_dump(mode="python")) + "\n")
    os.replace(temp_output_path, input_path)
    logging.info(
        "Deduped %s from %s valid rows to %s rows",
        input_path.name,
        len(records),
        len(kept_records),
    )
    return len(kept_records)


def _fill_summary(payload: dict[str, Any], run: Any) -> None:
    if _has_value(payload, "summaryMetrics"):
        return
    summary_metrics = _public_attr(run, "summary_metrics")
    if summary_metrics is not None:
        payload["summaryMetrics"] = summary_metrics
        return
    summary = _public_attr(run, "summary")
    if summary is not None:
        payload["summaryMetrics"] = summary


def _load_records_for_postprocess(input_path: Path) -> list[RawRunRecord]:
    lines = input_path.read_text(encoding="utf-8").splitlines()
    nonempty_lines = [
        (line_number, line.strip())
        for line_number, line in enumerate(lines, start=1)
        if line.strip()
    ]
    if not nonempty_lines:
        return []

    records: list[RawRunRecord] = []
    last_nonempty_index = len(nonempty_lines) - 1
    for index, (line_number, line) in enumerate(nonempty_lines):
        try:
            payload = json.loads(line)
            records.append(RawRunRecord.model_validate(payload))
        except Exception as exc:
            if index == last_nonempty_index:
                logging.warning(
                    "Removed malformed trailing line from %s at line %s",
                    input_path,
                    line_number,
                )
                continue
            if isinstance(exc, json.JSONDecodeError):
                raise ValueError(
                    f"Invalid JSON in {input_path} at line {line_number}"
                ) from exc
            raise ValueError(
                f"Invalid raw run record in {input_path} at line {line_number}"
            ) from exc
    return records


def _log_progress(*, run_count: int, total_runs: int | None, entity: str, project: str) -> None:
    if total_runs is None or total_runs <= 0:
        logging.info("Fetched %s runs from %s/%s", run_count, entity, project)
        return
    percent = (run_count / total_runs) * 100.0
    logging.info(
        "Fetched %s/%s runs from %s/%s: %.1f%%",
        run_count,
        total_runs,
        entity,
        project,
        percent,
    )


def _maybe_total_runs(runs_iter: Any) -> int | None:
    if not isinstance(runs_iter, Sized):
        return None
    total_runs = len(runs_iter)
    return total_runs if total_runs >= 0 else None


def _fill_if_missing(payload: dict[str, Any], key: str, value: Any) -> None:
    if _has_value(payload, key) or value is None:
        return
    payload[key] = value


def _has_value(payload: dict[str, Any], key: str) -> bool:
    return key in payload and payload[key] is not None


def _attrs_payload(run: Any) -> dict[str, Any]:
    raw_attrs = getattr(run, "_attrs", None)
    if not isinstance(raw_attrs, dict):
        return {}
    return _to_jsonable(raw_attrs)


def _drop_duplicate_aliases(payload: dict[str, Any]) -> dict[str, Any]:
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


def _public_attr(run: Any, attr_name: str) -> Any:
    if not hasattr(run, attr_name):
        return None
    value = getattr(run, attr_name)
    return _coerce_public_value(value)


def _coerce_public_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, dict):
        return {str(k): _coerce_public_value(v) for k, v in value.items()}
    if isinstance(value, list):
        coerced_seq = [_coerce_public_value(v) for v in value]
        pairs = []
        for elem in value:
            if not (isinstance(elem, (list, tuple)) and len(elem) == 2):
                return coerced_seq
            pairs.append((elem[0], elem[1]))
        return {str(k): _coerce_public_value(v) for k, v in pairs}
    if isinstance(value, tuple):
        coerced_seq = [_coerce_public_value(v) for v in value]
        pairs = []
        for elem in value:
            if not (isinstance(elem, (list, tuple)) and len(elem) == 2):
                return coerced_seq
            pairs.append((elem[0], elem[1]))
        return {str(k): _coerce_public_value(v) for k, v in pairs}
    if isinstance(value, set):
        coerced_sorted = sorted(_coerce_public_value(v) for v in value)
        pairs = []
        for elem in value:
            if not (isinstance(elem, (list, tuple)) and len(elem) == 2):
                return coerced_sorted
            pairs.append((elem[0], elem[1]))
        return {str(k): _coerce_public_value(v) for k, v in pairs}
    raw_attrs = getattr(value, "_attrs", None)
    if isinstance(raw_attrs, dict):
        return {str(k): _coerce_public_value(v) for k, v in raw_attrs.items()}
    if isinstance(value, Mapping) or (
        hasattr(value, "items") and callable(getattr(value, "items", None))
    ):
        return {str(k): _coerce_public_value(v) for k, v in value.items()}
    if not isinstance(value, (str, bytes, Mapping)) and hasattr(value, "__iter__"):
        pairs = []
        for elem in iter(value):
            if not (isinstance(elem, (list, tuple)) and len(elem) == 2):
                return str(value)
            pairs.append((elem[0], elem[1]))
        return {str(k): _coerce_public_value(v) for k, v in pairs}
    return str(value)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_to_jsonable(v) for v in value)
    return value


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
