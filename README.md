# dr-wandb

Policy-driven sync, export, and update tooling for Weights & Biases.

## Installation

```bash
# CLI tool
uv tool install dr-wandb

# Or as a library
uv add dr-wandb
```

## Authentication

```bash
wandb login
# or
export WANDB_API_KEY=your_api_key_here
```

## CLI

### Export canonical project data

```bash
wandb-export ENTITY PROJECT OUTPUT_DIR [OPTIONS]

Options:
  --output-format  [parquet|jsonl]  Output format (default: parquet)
  --runs-per-page  INTEGER          Runs fetched per API call (default: 500)
  --state-path     TEXT             Optional explicit sync state path
  --save-every     INTEGER          Persist state every N runs (default: 25)
  --checkpoint-every-runs INTEGER   Write checkpoint chunk every N runs (default: 25)
  --no-incremental                  Disable checkpointed export (legacy single-shot output)
  --no-finalize-compact             Keep checkpoint chunks only (skip compact final tables)
  --inspection-sample-rows INTEGER  Sample size for per-checkpoint inspection stats (default: 5)
  --policy-module  TEXT             Policy module (default: dr_wandb.sync_policy)
  --policy-class   TEXT             Policy class (default: NoopPolicy)
  --output-json    TEXT             Optional summary output path
```

`wandb-export` now uses incremental checkpointing by default. During export it writes:
- `OUTPUT_DIR/_checkpoints/runs/chunk-*.parquet`
- `OUTPUT_DIR/_checkpoints/history/chunk-*.parquet`
- `OUTPUT_DIR/_checkpoints/manifest.json`
- `OUTPUT_DIR/_checkpoints/inspection.jsonl`

The job can resume after interruption using the same `--state-path`, and final compact outputs are deduplicated from checkpoint chunks.

### Sync + patch workflows

```bash
wandb-sync ENTITY PROJECT --policy-module my_pkg.my_policy --policy-class MyPolicy
wandb-plan-patches ENTITY PROJECT ./patches.jsonl --policy-module my_pkg.my_policy --policy-class MyPolicy
wandb-apply-patches ./patches.jsonl            # dry-run
wandb-apply-patches ./patches.jsonl --apply    # writes updates
```

## Library usage

```python
from pathlib import Path

from dr_wandb.sync_engine import SyncEngine
from dr_wandb.sync_policy import NoopPolicy
from dr_wandb.sync_types import ExportConfig

engine = SyncEngine(policy=NoopPolicy())
summary = engine.export_project(
    ExportConfig(
        entity="my-team",
        project="my-project",
        output_dir=Path("./data"),
        output_format="parquet",
    )
)

print(summary.run_count, summary.history_count)
```

## Core concepts

### `SyncPolicy`
A policy controls data retrieval and decision logic:
- `select_history_keys(ctx)`
- `select_history_window(ctx)`
- `classify_run(ctx, history_tail)`
- `infer_patch(ctx, history_tail)`
- `should_update(ctx, patch)`
- `on_error(ctx, exc)`

### Canonical export outputs
`wandb-export` writes:
- runs table: one row per run with run payload + policy/cursor fields
- history table: one row per history event with `_step`/`_timestamp`/`_runtime`/`_wandb` + metric payload
- manifest JSON: schema/version, policy identity, counts, and file paths

## License

MIT
