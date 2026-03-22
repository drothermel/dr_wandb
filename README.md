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
  --fetch-mode     [incremental|full_reconcile]
                                      Run selection mode (default: incremental)
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

`--fetch-mode incremental` is now the default for `wandb-export`, `wandb-sync`, and `wandb-plan-patches`. In that mode `dr_wandb`:
- fetches newly created runs with a `createdAt >= last_seen_created_at` filter;
- revisits only runs that are still marked non-terminal in the saved state;
- avoids history scans unless the active policy explicitly requests them.

Use `--fetch-mode full_reconcile` to force a full project rescan.

To iteratively update the existing `ml-moe/moe` export on this machine, rerun:

```bash
uv run wandb-export \
  ml-moe moe \
  /Users/daniellerothermel/drotherm/repos/ml-moe/data/wandb_export \
  --state-path /Users/daniellerothermel/drotherm/repos/ml-moe/data/.sync/ml_moe_moe_state.json \
  --output-json /Users/daniellerothermel/drotherm/repos/ml-moe/data/.sync/last_export_summary.json
```

### Sync + patch workflows

```bash
wandb-sync ENTITY PROJECT --policy-module my_pkg.my_policy --policy-class MyPolicy
wandb-bootstrap-export ENTITY PROJECT ./old_export ./new_export --policy-module my_pkg.my_policy --policy-class MyPolicy
wandb-inspect-state ENTITY PROJECT --state-path ./state.json
wandb-plan-patches ENTITY PROJECT ./patches.jsonl --policy-module my_pkg.my_policy --policy-class MyPolicy
wandb-apply-patches ./patches.jsonl            # dry-run
wandb-apply-patches ./patches.jsonl --apply    # writes updates
```

`wandb-sync` and `wandb-plan-patches` also default to `--fetch-mode incremental`. Pass `--fetch-mode full_reconcile` when you need a full project rescan.

`wandb-bootstrap-export` reads an existing compact export (`*_runs.*`, `*_history.*`), rebuilds sync state locally, reapplies the active policy, and seeds a fresh output directory with a single merged checkpoint baseline. It now streams large history tables instead of materializing the whole history export in memory. Use `--overwrite-output` when you want to replace an existing bootstrap target directory or state file.

`wandb-inspect-state` reads the saved sync state and reports tracked run counts by status, including terminal, ignore, and non-terminal runs. Use `--show-runs non_terminal|ignore|terminal --limit N` when you want a small sample of the matching runs.

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

### Loading extracted run metadata

Use the loader functions in `dr_wandb.run_metadata` when reading `runs_raw.jsonl` exports back in. Both loaders return only the latest raw snapshot per `run_id`, ordered by `createdAt` descending.

```python
from pathlib import Path

from dr_wandb.run_metadata import (
    load_canonical_run_metadata,
    load_raw_run_record_dicts,
)

data_root = Path("./data")

canonical_runs = load_canonical_run_metadata(
    export_name="moe_runs",
    data_root=data_root,
)

raw_run_dicts = load_raw_run_record_dicts(
    export_name="moe_runs",
    data_root=data_root,
)
```

Choose the loader based on the shape you want:
- `load_canonical_run_metadata(...)` returns `CanonicalRunMetadata` models with promoted fields like `name`, `config`, `summaryMetrics`, and `historyKeys`.
- `load_raw_run_record_dicts(...)` returns `RawRunRecord`-shaped dictionaries with the original outer record fields: `run_id`, `entity`, `project`, `exported_at`, `raw_run_hash`, and `raw_run`.

These are the supported read paths for run metadata exports. They intentionally deduplicate multiple raw rows for the same run and should be preferred over line-by-line access to `runs_raw.jsonl`.

## Core concepts

### `SyncPolicy`
A policy controls data retrieval and decision logic:
- `select_history_keys(ctx)`
- `select_history_window(ctx)`
- `classify_run(ctx, history_tail)`
- `infer_patch(ctx, history_tail)`
- `should_update(ctx, patch)`
- `is_terminal(ctx, decision)`
- `on_error(ctx, exc)`

### Canonical export outputs
`wandb-export` writes:
- runs table: one row per run with run payload + policy/cursor fields
- history table: one row per history event with `_step`/`_timestamp`/`_runtime`/`_wandb` + metric payload
- manifest JSON: schema/version, policy identity, counts, and file paths

## License

MIT
