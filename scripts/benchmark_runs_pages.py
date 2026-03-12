from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import srsly
import typer
import wandb
from pydantic import BaseModel, Field


class PageTiming(BaseModel):
    page_index: int
    runs_seen: int
    elapsed_since_iteration_start_seconds: float
    elapsed_since_previous_page_seconds: float


class BenchmarkSummary(BaseModel):
    entity: str
    project: str
    order: str
    page_size: int
    num_pages_requested: int
    lazy: bool
    timeout_seconds: int
    api_init_seconds: float
    iterator_setup_seconds: float
    time_to_first_run_seconds: float | None = None
    runs_seen: int = 0
    pages_completed: int = 0
    page_timings: list[PageTiming] = Field(default_factory=list)
    first_page_total_seconds: float | None = None
    first_page_per_run_seconds: float | None = None
    later_pages_mean_seconds: float | None = None
    later_pages_per_run_mean_seconds: float | None = None
    amortized_total_per_run_seconds: float | None = None
    predicted_time_to_3600_runs_seconds: float | None = None
    deep_normalize: bool = True
    include_metadata: bool = True
    dump_elapsed_seconds: float = 0.0
    dump_per_run_seconds: float | None = None
    snapshot_build_elapsed_seconds: float = 0.0
    snapshot_build_per_run_seconds: float | None = None
    serialization_elapsed_seconds: float = 0.0
    serialization_per_run_seconds: float | None = None
    dump_output_bytes: int = 0
    dump_output_mebibytes: float | None = None
    dump_fallback_normalization_count: int = 0


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_to_jsonable(item) for item in value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    return value


def _get_public_attr(run: Any, attr_name: str) -> Any:
    try:
        return getattr(run, attr_name)
    except AttributeError:
        return None


def _build_raw_run_snapshot_record(
    *,
    run: Any,
    entity: str,
    project: str,
    deep_normalize: bool,
    include_metadata: bool,
) -> dict[str, Any]:
    raw_attrs = dict(getattr(run, "_attrs", {}) or {})
    raw_run = _to_jsonable(raw_attrs) if deep_normalize else raw_attrs
    fill_map = {
        "url": "url",
        "path": "path",
        "storageId": "storage_id",
        "config": "config",
        "rawconfig": "rawconfig",
        "summaryMetrics": "summary_metrics",
        "systemMetrics": "system_metrics",
        "tags": "tags",
        "user": "user",
        "createdAt": "created_at",
        "updatedAt": "updated_at",
        "heartbeatAt": "heartbeat_at",
        "displayName": "display_name",
        "name": "name",
        "state": "state",
        "group": "group",
        "jobType": "job_type",
        "sweepName": "sweep_name",
        "readOnly": "read_only",
        "historyKeys": "history_keys",
        "notes": "notes",
        "description": "description",
    }
    if include_metadata:
        fill_map["metadata"] = "metadata"
    for raw_key, public_attr in fill_map.items():
        if raw_key in raw_run and raw_run[raw_key] is not None:
            continue
        value = _get_public_attr(run, public_attr)
        if value is not None:
            raw_run[raw_key] = _to_jsonable(value) if deep_normalize else value

    return {
        "run_id": str(getattr(run, "id", "")),
        "entity": entity,
        "project": project,
        "exported_at": _iso_now(),
        "raw_run": raw_run,
    }


def _serialize_snapshot_record(
    *,
    snapshot_record: dict[str, Any],
    deep_normalize: bool,
) -> tuple[str, bool]:
    if deep_normalize:
        return srsly.json_dumps(snapshot_record), False
    try:
        return srsly.json_dumps(snapshot_record), False
    except TypeError:
        return srsly.json_dumps(_to_jsonable(snapshot_record)), True


def _validate_args(*, page_size: int, num_pages: int, timeout_seconds: int) -> None:
    if page_size <= 0:
        raise ValueError("--page-size must be positive")
    if num_pages <= 0:
        raise ValueError("--num-pages must be positive")
    if timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")


def _benchmark_runs(
    *,
    entity: str,
    project: str,
    page_size: int,
    num_pages: int,
    order: str,
    timeout_seconds: int,
    lazy: bool,
    dump_snapshot_path: Path | None,
    deep_normalize: bool,
    include_metadata: bool,
) -> BenchmarkSummary:
    started_at = time.perf_counter()
    api = wandb.Api(timeout=timeout_seconds)
    api_ready_at = time.perf_counter()

    runs = api.runs(
        f"{entity}/{project}",
        order=order,
        per_page=page_size,
        lazy=lazy,
    )
    iterator = iter(runs)
    iterator_ready_at = time.perf_counter()

    summary = BenchmarkSummary(
        entity=entity,
        project=project,
        order=order,
        page_size=page_size,
        num_pages_requested=num_pages,
        lazy=lazy,
        timeout_seconds=timeout_seconds,
        deep_normalize=deep_normalize,
        include_metadata=include_metadata,
        api_init_seconds=api_ready_at - started_at,
        iterator_setup_seconds=iterator_ready_at - api_ready_at,
    )

    target_runs = page_size * num_pages
    previous_page_elapsed = 0.0
    if dump_snapshot_path is not None:
        dump_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    dump_handle = (
        dump_snapshot_path.open("w", encoding="utf-8") if dump_snapshot_path is not None else None
    )

    try:
        for run_index, run in enumerate(iterator, start=1):
            _ = getattr(run, "id", None)

            if dump_handle is not None:
                dump_started_at = time.perf_counter()
                build_started_at = dump_started_at
                snapshot_record = _build_raw_run_snapshot_record(
                    run=run,
                    entity=entity,
                    project=project,
                    deep_normalize=deep_normalize,
                    include_metadata=include_metadata,
                )
                summary.snapshot_build_elapsed_seconds += (
                    time.perf_counter() - build_started_at
                )
                serialization_started_at = time.perf_counter()
                serialized_record, used_fallback_normalization = _serialize_snapshot_record(
                    snapshot_record=snapshot_record,
                    deep_normalize=deep_normalize,
                )
                if used_fallback_normalization:
                    summary.dump_fallback_normalization_count += 1
                dump_handle.write(serialized_record + "\n")
                summary.serialization_elapsed_seconds += (
                    time.perf_counter() - serialization_started_at
                )
                summary.dump_elapsed_seconds += time.perf_counter() - dump_started_at

            now = time.perf_counter()
            iteration_elapsed = now - iterator_ready_at

            if summary.time_to_first_run_seconds is None:
                summary.time_to_first_run_seconds = iteration_elapsed

            summary.runs_seen = run_index

            if run_index % page_size == 0:
                page_index = run_index // page_size
                summary.page_timings.append(
                    PageTiming(
                        page_index=page_index,
                        runs_seen=run_index,
                        elapsed_since_iteration_start_seconds=iteration_elapsed,
                        elapsed_since_previous_page_seconds=(
                            iteration_elapsed - previous_page_elapsed
                        ),
                    )
                )
                summary.pages_completed = page_index
                previous_page_elapsed = iteration_elapsed

            if run_index >= target_runs:
                break
    finally:
        if dump_handle is not None:
            dump_handle.flush()
            dump_handle.close()
        if dump_snapshot_path is not None and dump_snapshot_path.exists():
            summary.dump_output_bytes = dump_snapshot_path.stat().st_size
            summary.dump_output_mebibytes = summary.dump_output_bytes / (1024 * 1024)

    if summary.runs_seen > 0 and summary.runs_seen % page_size != 0:
        final_elapsed = time.perf_counter() - iterator_ready_at
        summary.page_timings.append(
            PageTiming(
                page_index=math.ceil(summary.runs_seen / page_size),
                runs_seen=summary.runs_seen,
                elapsed_since_iteration_start_seconds=final_elapsed,
                elapsed_since_previous_page_seconds=final_elapsed - previous_page_elapsed,
            )
        )
        summary.pages_completed = len(summary.page_timings)

    _populate_derived_metrics(summary)
    return summary


def _print_human(summary: BenchmarkSummary) -> None:
    print(
        f"Project: {summary.entity}/{summary.project}\n"
        f"Order: {summary.order}\n"
        f"Page size: {summary.page_size}\n"
        f"Requested pages: {summary.num_pages_requested}\n"
        f"Lazy: {summary.lazy}\n"
        f"Timeout seconds: {summary.timeout_seconds}\n"
        f"Include metadata: {summary.include_metadata}\n"
        f"API init seconds: {summary.api_init_seconds:.3f}\n"
        f"Iterator setup seconds: {summary.iterator_setup_seconds:.3f}\n"
        f"Time to first run seconds: "
        f"{summary.time_to_first_run_seconds:.3f}"
        if summary.time_to_first_run_seconds is not None
        else "Time to first run seconds: n/a"
    )
    print(f"Runs seen: {summary.runs_seen}")
    print(f"Pages completed: {summary.pages_completed}")
    if summary.first_page_total_seconds is not None:
        print(f"First page total seconds: {summary.first_page_total_seconds:.3f}")
    if summary.first_page_per_run_seconds is not None:
        print(f"First page per run seconds: {summary.first_page_per_run_seconds:.6f}")
    if summary.later_pages_mean_seconds is not None:
        print(f"Later pages mean seconds: {summary.later_pages_mean_seconds:.3f}")
    if summary.later_pages_per_run_mean_seconds is not None:
        print(
            "Later pages per run mean seconds: "
            f"{summary.later_pages_per_run_mean_seconds:.6f}"
        )
    if summary.amortized_total_per_run_seconds is not None:
        print(
            "Amortized total per run seconds: "
            f"{summary.amortized_total_per_run_seconds:.6f}"
        )
    if summary.predicted_time_to_3600_runs_seconds is not None:
        print(
            "Predicted time to 3600 runs seconds: "
            f"{summary.predicted_time_to_3600_runs_seconds:.3f}"
        )
    print(f"Dump elapsed seconds: {summary.dump_elapsed_seconds:.3f}")
    if summary.dump_per_run_seconds is not None:
        print(f"Dump per run seconds: {summary.dump_per_run_seconds:.6f}")
    print(f"Snapshot build elapsed seconds: {summary.snapshot_build_elapsed_seconds:.3f}")
    if summary.snapshot_build_per_run_seconds is not None:
        print(
            "Snapshot build per run seconds: "
            f"{summary.snapshot_build_per_run_seconds:.6f}"
        )
    print(f"Serialization elapsed seconds: {summary.serialization_elapsed_seconds:.3f}")
    if summary.serialization_per_run_seconds is not None:
        print(
            "Serialization per run seconds: "
            f"{summary.serialization_per_run_seconds:.6f}"
        )
    print(
        f"Dump fallback normalization count: "
        f"{summary.dump_fallback_normalization_count}"
    )
    print(f"Dump output bytes: {summary.dump_output_bytes}")
    if summary.dump_output_mebibytes is not None:
        print(f"Dump output MiB: {summary.dump_output_mebibytes:.3f}")
    for page_timing in summary.page_timings:
        print(
            f"Page {page_timing.page_index}: runs={page_timing.runs_seen} "
            f"elapsed_total={page_timing.elapsed_since_iteration_start_seconds:.3f}s "
            f"elapsed_page={page_timing.elapsed_since_previous_page_seconds:.3f}s"
        )


def _append_summary_jsonl(*, output_path: Path, summary: BenchmarkSummary) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(srsly.json_dumps(summary.model_dump(mode="python")) + "\n")


def _populate_derived_metrics(summary: BenchmarkSummary) -> None:
    if not summary.page_timings or summary.page_size <= 0:
        return

    first_page_total_seconds = summary.page_timings[0].elapsed_since_previous_page_seconds
    summary.first_page_total_seconds = first_page_total_seconds
    summary.first_page_per_run_seconds = first_page_total_seconds / summary.page_size

    later_page_seconds = [
        page_timing.elapsed_since_previous_page_seconds
        for page_timing in summary.page_timings[1:]
    ]
    if later_page_seconds:
        later_pages_mean_seconds = sum(later_page_seconds) / len(later_page_seconds)
        summary.later_pages_mean_seconds = later_pages_mean_seconds
        summary.later_pages_per_run_mean_seconds = (
            later_pages_mean_seconds / summary.page_size
        )

    total_elapsed_seconds = summary.page_timings[-1].elapsed_since_iteration_start_seconds
    if summary.runs_seen > 0:
        summary.amortized_total_per_run_seconds = total_elapsed_seconds / summary.runs_seen
        summary.predicted_time_to_3600_runs_seconds = (
            summary.amortized_total_per_run_seconds * 3600
        )
        summary.dump_per_run_seconds = summary.dump_elapsed_seconds / summary.runs_seen
        summary.snapshot_build_per_run_seconds = (
            summary.snapshot_build_elapsed_seconds / summary.runs_seen
        )
        summary.serialization_per_run_seconds = (
            summary.serialization_elapsed_seconds / summary.runs_seen
        )


app = typer.Typer(add_completion=False, help="Benchmark W&B run pagination.")


@app.command()
def main(
    entity: Annotated[str, typer.Argument(help="W&B entity/team name")],
    project: Annotated[str, typer.Argument(help="W&B project name")],
    page_size: Annotated[int, typer.Option(help="Runs per API page.")] = ...,
    num_pages: Annotated[int, typer.Option(help="Number of pages to iterate.")] = ...,
    order: Annotated[str, typer.Option(help="W&B run ordering.")] = "-created_at",
    timeout_seconds: Annotated[
        int, typer.Option(help="W&B API timeout in seconds.")
    ] = 300,
    lazy: Annotated[
        bool, typer.Option(help="Whether to use lazy loading for Api.runs(...).")
    ] = False,
    dump_snapshot_path: Annotated[
        Path | None,
        typer.Option(
            help=(
                "If set, write one raw-style JSONL record per fetched run to this path. "
                "This approximates the future raw run wrapper shape without deduplication."
            )
        ),
    ] = None,
    deep_normalize: Annotated[
        bool,
        typer.Option(
            help=(
                "Whether to recursively normalize nested values before dumping. "
                "Disable to compare rawer output size and dump speed."
            )
        ),
    ] = True,
    include_metadata: Annotated[
        bool,
        typer.Option(
            help=(
                "Whether to include the expensive run.metadata fill-in when dumping "
                "snapshot records."
            )
        ),
    ] = True,
    results_jsonl: Annotated[
        Path | None,
        typer.Option(
            help="If set, append one JSON record with the benchmark results to this JSONL file."
        ),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit JSON instead of human-readable text.")
    ] = False,
) -> None:
    _validate_args(
        page_size=page_size,
        num_pages=num_pages,
        timeout_seconds=timeout_seconds,
    )
    summary = _benchmark_runs(
        entity=entity,
        project=project,
        page_size=page_size,
        num_pages=num_pages,
        order=order,
        timeout_seconds=timeout_seconds,
        lazy=lazy,
        dump_snapshot_path=dump_snapshot_path,
        deep_normalize=deep_normalize,
        include_metadata=include_metadata,
    )

    if json_output:
        print(json.dumps(summary.model_dump(mode="python"), indent=2, sort_keys=True))
    else:
        _print_human(summary)

    if results_jsonl is not None:
        _append_summary_jsonl(output_path=results_jsonl, summary=summary)


if __name__ == "__main__":
    app()
