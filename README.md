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

## CLI

There is one public command:

```bash
wandb-export ENTITY PROJECT --name EXPORT_NAME [OPTIONS]
```

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

## Library Usage

```python
from pathlib import Path

from dr_wandb import ExportEngine, ExportRequest, ExportMode

summary = ExportEngine().export(
    ExportRequest(
        entity="ml-moe",
        project="moe",
        name="moe_runs",
        data_root=Path("data"),
        mode=ExportMode.METADATA,
    )
)

print(summary.run_count)
```

## Loaders

Use the public loaders to read an existing export back in:

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

## Core Concepts

- `metadata` mode exports current run snapshots only.
- `history` mode exports current run snapshots plus history rows.
- `incremental` mode fetches newly created runs and refreshes tracked non-terminal runs.
- `full_reconcile` rebuilds the export from scratch.
- Each named export is self-contained inside `data_root / name`.

## License

MIT
