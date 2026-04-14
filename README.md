# dr-wandb

Unified export tooling for Weights & Biases.

## Installation

```bash
uv tool install dr-wandb
```

Or as a library:

```bash
uv add dr-wandb
```

## Authentication

```bash
wandb login
```

## Design Goals

- Export W&B runs into a durable local layout that downstream repos can load
  without talking to the W&B API directly.
- Keep export behavior deterministic so repeated syncs and diffs stay readable.
- Separate current run snapshots from optional history rows while letting both
  live under one named export.
- Hide raw W&B SDK object quirks behind JSON-safe stored models.

## CLI

There is one public command:

```bash
wandb-export ENTITY PROJECT --name EXPORT_NAME [OPTIONS]
```

The command always writes into `data_root / name`, where `name` identifies one
logical export. That directory is self-contained and can hold:

- `manifest.json`: what was exported and where its artifacts live
- `state.json`: incremental tracking state used for future syncs
- `runs.jsonl`: the current latest snapshot for each run
- `history.jsonl`: optional history rows when `--mode history` is used

Example metadata-only export:

```bash
wandb-export ml-moe moe --name moe_runs --data-root data
```

This writes:

```text
data/moe_runs/
  manifest.json
  state.json
  runs.jsonl
```

Metadata mode writes one current snapshot per run. It does not export history
rows.

Example history export:

```bash
wandb-export ml-moe moe \
  --name moe_history \
  --data-root data \
  --mode history
```

This writes:

```text
data/moe_history/
  manifest.json
  state.json
  runs.jsonl
  history.jsonl
```

History mode writes the same current snapshots plus normalized history rows
selected from `scan_history`.

History export can be limited with selection flags:

```bash
wandb-export ml-moe moe \
  --name moe_eval_tail \
  --data-root data \
  --mode history \
  --history-key eval/loss \
  --history-key eval/accuracy \
  --max-records 100
```

History selection rules to rely on:

- history selection flags only apply in `--mode history`
- `--history-key` narrows the metric keys requested from W&B
- `--min-step` and `--max-step` constrain the scan window when supported by the
  underlying W&B API call
- `--max-records` trims the final result to the most recent rows after scanning

## Sync Semantics

`dr-wandb` supports two sync modes:

- `SyncMode.INCREMENTAL`
  - fetches newly created runs since the export's tracked `max_created_at`
  - refreshes previously seen runs that are not yet terminal
  - reuses prior snapshots and history rows where possible
- `SyncMode.FULL_RECONCILE`
  - ignores prior incremental state
  - rebuilds the named export from scratch using the current W&B project state

Terminal run states are treated as complete enough to stop refreshing unless a
full reconcile is requested.

## Data Contracts

### Run snapshots

`runs.jsonl` stores `RunSnapshot` records, each of which contains:

- one normalized `WandbRun`
- the `exported_at` timestamp for that snapshot

Raw W&B SDK objects are normalized into JSON-safe data before being written.
That includes nested config-like payloads and object-valued fields such as
`user`.

### History rows

`history.jsonl` stores `HistoryRow` records. Each row keeps:

- `run_id`
- step, timestamp, and runtime when available
- `wandb_metadata` from `_wandb`
- `metrics` for non-underscore keys
- `extra` for underscore-prefixed fields that are not promoted into the typed
  top-level columns

History rows are merged by a stable deduplication key so repeated incremental
exports do not blindly append duplicates.

## Library Usage

```python
from pathlib import Path

from dr_wandb import ExportEngine, ExportMode, ExportRequest

summary = ExportEngine(
    ExportRequest(
        entity="ml-moe",
        project="moe",
        name="moe_runs",
        data_root=Path("data"),
        mode=ExportMode.METADATA,
    )
).export()

print(summary.run_count)
```

`ExportEngine.export()` is the high-level entrypoint when you want the same
workflow the CLI runs and a typed `ExportSummary` describing the artifacts it
wrote.

## Loaders

Read an existing export back in with the top-level helper functions:

```python
from pathlib import Path

from dr_wandb import load_manifest, load_run_snapshots

manifest = load_manifest("moe_runs", Path("data"))
snapshots = load_run_snapshots("moe_runs", Path("data"))

print(manifest.mode, len(snapshots))
```

If the export includes history rows:

```python
from pathlib import Path

from dr_wandb import iter_history_rows

rows = list(iter_history_rows("moe_history", Path("data")))
print(len(rows))
```

`ExportStore` is also available if you want direct access to the per-export
directory layout and I/O. The top-level helpers are the better default when you
just want to read a completed export.

## Core Concepts

- `ExportMode.METADATA` exports current run snapshots only.
- `ExportMode.HISTORY` exports current run snapshots plus history rows.
- `SyncMode.INCREMENTAL` fetches newly created runs and refreshes tracked non-terminal runs.
- `SyncMode.FULL_RECONCILE` rebuilds the export from scratch.
- Each named export is self-contained inside `data_root / name`.

## Development

For local iteration, run the smallest targeted test command that covers the
code you changed.

Before committing:

```bash
uv run ruff format .
uv run ruff check .
uv run ty check
uv run pytest
```

Before publishing:

```bash
uv build
```

## License

MIT
