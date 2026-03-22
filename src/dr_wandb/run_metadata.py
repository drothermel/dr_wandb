from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field

from dr_wandb.export_profile import build_profile, resolve_raw_extract_profile_paths
from dr_wandb.raw_extract import RawRunRecord


class ParsedSummaryMetrics(BaseModel):
    runtime: int | float | None = None
    step: int | None = None
    timestamp: int | float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_raw(cls, value: Any) -> ParsedSummaryMetrics:
        raw = _coerce_mapping(value)
        extra: dict[str, Any] = {}

        runtime = _promote_number(raw, "_runtime", extra)
        step = _promote_int(raw, "_step", extra)
        timestamp = _promote_number(raw, "_timestamp", extra)

        wandb_payload = raw.get("_wandb")
        if isinstance(wandb_payload, dict):
            runtime = _merge_wandb_promoted(
                current=runtime,
                incoming=wandb_payload.get("runtime"),
                parser=_coerce_number,
                extra=extra,
                extra_key="wandb.runtime",
            )
            step = _merge_wandb_promoted(
                current=step,
                incoming=wandb_payload.get("step"),
                parser=_coerce_int,
                extra=extra,
                extra_key="wandb.step",
            )
            timestamp = _merge_wandb_promoted(
                current=timestamp,
                incoming=wandb_payload.get("timestamp"),
                parser=_coerce_number,
                extra=extra,
                extra_key="wandb.timestamp",
            )
            for key, nested_value in wandb_payload.items():
                if key in {"runtime", "step", "timestamp"}:
                    continue
                extra[f"wandb.{key}"] = nested_value
        elif wandb_payload is not None:
            extra["_wandb"] = wandb_payload

        metrics: dict[str, Any] = {}
        for key, nested_value in raw.items():
            if key in {"_runtime", "_step", "_timestamp", "_wandb"}:
                continue
            if str(key).startswith("_"):
                extra[str(key)] = nested_value
                continue
            if not _insert_nested_metric(metrics, str(key), nested_value):
                extra[str(key)] = nested_value

        return cls(
            runtime=runtime,
            step=step,
            timestamp=timestamp,
            metrics=metrics,
            extra=extra,
        )

    def flatten_metrics(self) -> dict[str, Any]:
        flattened: dict[str, Any] = {}
        _flatten_nested_metrics(self.metrics, flattened)
        return flattened

    def to_flat_dict(self, *, include_extra: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.runtime is not None:
            payload["runtime"] = self.runtime
        if self.step is not None:
            payload["step"] = self.step
        if self.timestamp is not None:
            payload["timestamp"] = self.timestamp
        payload.update(self.flatten_metrics())
        if include_extra:
            payload.update(self.extra)
        return payload


class ParsedHistoryField(BaseModel):
    key: str | None = None
    type: str | None = None
    count: int | None = None
    monotonic: bool | None = None
    previousValue: Any = None
    extra: dict[str, Any] = Field(default_factory=dict)


class UnparsedHistoryField(BaseModel):
    key: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None


class ParsedHistoryKeys(BaseModel):
    fields: dict[str, ParsedHistoryField | UnparsedHistoryField] = Field(
        default_factory=dict
    )
    lastStep: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_raw(cls, value: Any) -> ParsedHistoryKeys:
        raw = _coerce_mapping(value)
        raw_fields = raw.get("keys")
        fields: dict[str, ParsedHistoryField | UnparsedHistoryField] = {}
        if isinstance(raw_fields, dict):
            for key, field_value in raw_fields.items():
                fields[str(key)] = _parse_history_field(str(key), field_value)
        elif raw_fields is not None:
            fields["__unparsed__"] = UnparsedHistoryField(
                key="__unparsed__",
                raw={"keys": raw_fields},
                reason="keys_not_mapping",
            )

        extra = {
            str(key): nested_value
            for key, nested_value in raw.items()
            if str(key) not in {"keys", "lastStep"}
        }
        return cls(
            fields=fields,
            lastStep=_coerce_int(raw.get("lastStep")),
            extra=extra,
        )


class CanonicalRunMetadata(BaseModel):
    run_id: str
    entity: str
    project: str
    exported_at: str
    raw_run_hash: str
    name: str | None = None
    state: str | None = None
    group: str | None = None
    createdAt: str | None = None
    heartbeatAt: str | None = None
    tags: list[str] = Field(default_factory=list)
    url: str | None = None
    historyLineCount: int | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    summaryMetrics: ParsedSummaryMetrics = Field(default_factory=ParsedSummaryMetrics)
    historyKeys: ParsedHistoryKeys = Field(default_factory=ParsedHistoryKeys)
    extra: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_raw_run_record(cls, record: RawRunRecord) -> CanonicalRunMetadata:
        raw_run = dict(record.raw_run)
        name = _coerce_str(raw_run.get("displayName")) or _coerce_str(raw_run.get("name"))
        tags = _coerce_string_list(raw_run.get("tags"))
        config = _coerce_mapping(raw_run.get("config"))
        summary_metrics = ParsedSummaryMetrics.from_raw(raw_run.get("summaryMetrics"))
        history_keys = ParsedHistoryKeys.from_raw(raw_run.get("historyKeys"))

        promoted_keys = {
            "displayName",
            "name",
            "state",
            "group",
            "createdAt",
            "heartbeatAt",
            "tags",
            "url",
            "historyLineCount",
            "config",
            "summaryMetrics",
            "historyKeys",
        }
        extra = {
            str(key): value
            for key, value in raw_run.items()
            if str(key) not in promoted_keys
        }
        return cls(
            run_id=record.run_id,
            entity=record.entity,
            project=record.project,
            exported_at=record.exported_at,
            raw_run_hash=record.raw_run_hash,
            name=name,
            state=_coerce_str(raw_run.get("state")),
            group=_coerce_str(raw_run.get("group")),
            createdAt=_coerce_str(raw_run.get("createdAt")),
            heartbeatAt=_coerce_str(raw_run.get("heartbeatAt")),
            tags=tags,
            url=_coerce_str(raw_run.get("url")),
            historyLineCount=_coerce_int(raw_run.get("historyLineCount")),
            config=config,
            summaryMetrics=summary_metrics,
            historyKeys=history_keys,
            extra=extra,
        )


def resolve_runs_raw_path(*, export_name: str, data_root: Path) -> Path:
    profile = build_profile(
        export_name=export_name,
        data_root=str(data_root),
        sync_root=None,
    )
    if profile is None:
        raise ValueError("export_name and data_root are required")
    return resolve_raw_extract_profile_paths(profile).runs_raw_path


def _iter_raw_run_records(*, export_name: str, data_root: Path) -> Iterator[RawRunRecord]:
    runs_raw_path = resolve_runs_raw_path(export_name=export_name, data_root=data_root)
    if not runs_raw_path.exists():
        raise FileNotFoundError(f"Missing raw extract file: {runs_raw_path}")
    with runs_raw_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {runs_raw_path} at line {line_number}"
                ) from exc
            try:
                yield RawRunRecord.model_validate(payload)
            except Exception as exc:
                raise ValueError(
                    f"Invalid raw run record in {runs_raw_path} at line {line_number}"
                ) from exc


def load_canonical_run_metadata(
    *,
    export_name: str,
    data_root: Path,
) -> list[CanonicalRunMetadata]:
    raw_records = _load_latest_raw_run_records(
        export_name=export_name,
        data_root=data_root,
    )
    return [
        CanonicalRunMetadata.from_raw_run_record(record) for record in raw_records
    ]


def load_raw_run_record_dicts(
    *,
    export_name: str,
    data_root: Path,
) -> list[dict[str, Any]]:
    raw_records = _load_latest_raw_run_records(
        export_name=export_name,
        data_root=data_root,
    )
    return [record.model_dump(mode="python") for record in raw_records]


def _load_latest_raw_run_records(
    *,
    export_name: str,
    data_root: Path,
) -> list[RawRunRecord]:
    latest_by_run_id: dict[str, RawRunRecord] = {}
    for record in _iter_raw_run_records(export_name=export_name, data_root=data_root):
        current = latest_by_run_id.get(record.run_id)
        if current is None or _record_sort_key(record) > _record_sort_key(current):
            latest_by_run_id[record.run_id] = record

    raw_records = sorted(latest_by_run_id.values(), key=lambda record: record.run_id)
    raw_records = sorted(
        raw_records,
        key=lambda record: _coerce_str(record.raw_run.get("createdAt")) or "",
        reverse=True,
    )
    return raw_records


def _record_sort_key(record: RawRunRecord) -> tuple[str, str]:
    return record.exported_at, record.raw_run_hash


def _parse_history_field(key: str, value: Any) -> ParsedHistoryField | UnparsedHistoryField:
    if not isinstance(value, dict):
        return UnparsedHistoryField(
            key=key,
            raw={"value": value},
            reason="field_not_mapping",
        )

    type_counts = value.get("typeCounts")
    if not isinstance(type_counts, list) or len(type_counts) != 1:
        return UnparsedHistoryField(
            key=key,
            raw=dict(value),
            reason="unexpected_type_counts",
        )

    type_count = type_counts[0]
    if not isinstance(type_count, dict):
        return UnparsedHistoryField(
            key=key,
            raw=dict(value),
            reason="type_count_not_mapping",
        )

    parsed_type = _coerce_str(type_count.get("type"))
    parsed_count = _coerce_int(type_count.get("count"))
    if parsed_type is None or parsed_count is None:
        return UnparsedHistoryField(
            key=key,
            raw=dict(value),
            reason="invalid_type_count_entry",
        )

    extra = {
        str(key): nested_value
        for key, nested_value in value.items()
        if str(key) not in {"typeCounts", "monotonic", "previousValue"}
    }
    return ParsedHistoryField(
        key=key,
        type=parsed_type,
        count=parsed_count,
        monotonic=_coerce_bool(value.get("monotonic")),
        previousValue=value.get("previousValue"),
        extra=extra,
    )


def _promote_number(raw: dict[str, Any], key: str, extra: dict[str, Any]) -> int | float | None:
    value = raw.get(key)
    parsed = _coerce_number(value)
    if value is not None and parsed is None:
        extra[key] = value
    return parsed


def _promote_int(raw: dict[str, Any], key: str, extra: dict[str, Any]) -> int | None:
    value = raw.get(key)
    parsed = _coerce_int(value)
    if value is not None and parsed is None:
        extra[key] = value
    return parsed


def _merge_wandb_promoted(
    *,
    current: int | float | None,
    incoming: Any,
    parser: Any,
    extra: dict[str, Any],
    extra_key: str,
) -> Any:
    parsed_incoming = parser(incoming)
    if incoming is not None and parsed_incoming is None:
        extra[extra_key] = incoming
        return current
    if current is None:
        return parsed_incoming
    if parsed_incoming is None or parsed_incoming == current:
        return current
    extra[extra_key] = incoming
    return current


def _insert_nested_metric(metrics: dict[str, Any], key: str, value: Any) -> bool:
    parts = key.split("/")
    if len(parts) == 1:
        existing = metrics.get(parts[0])
        if isinstance(existing, dict):
            return False
        metrics[parts[0]] = value
        return True

    current: dict[str, Any] = metrics
    for part in parts[:-1]:
        existing = current.get(part)
        if existing is None:
            next_node: dict[str, Any] = {}
            current[part] = next_node
            current = next_node
            continue
        if not isinstance(existing, dict):
            return False
        current = existing

    leaf = parts[-1]
    existing_leaf = current.get(leaf)
    if isinstance(existing_leaf, dict):
        return False
    current[leaf] = value
    return True


def _flatten_nested_metrics(metrics: dict[str, Any], flattened: dict[str, Any], prefix: str = "") -> None:
    for key, value in metrics.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            _flatten_nested_metrics(value, flattened, path)
            continue
        flattened[path] = value


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): nested for key, nested in value.items()}
    return {}


def _coerce_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_number(value: Any) -> int | float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None
