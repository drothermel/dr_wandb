# dr-wandb

Download Weights & Biases experiment data to local pickle or parquet files for offline analysis.

## Installation

```bash
# CLI tool
uv tool install dr-wandb

# Or as a library
uv add dr-wandb
```

## Authentication

Configure Weights & Biases authentication:

```bash
wandb login
```

Or set the API key as an environment variable:

```bash
export WANDB_API_KEY=your_api_key_here
```

## CLI Usage

```bash
wandb-download ENTITY PROJECT OUTPUT_DIR [OPTIONS]

Options:
  --output-format  [pkl|parquet]  Output format (default: pkl)
  --runs-only                     Download only run metadata, skip histories
  --runs-per-page  INTEGER        Runs to fetch per API call (default: 500)
  --log-every      INTEGER        Log progress every N runs (default: 20)
```

### Examples

Download runs and histories as pickle files:
```bash
wandb-download my-team my-project ./data
```

Download as parquet files:
```bash
wandb-download my-team my-project ./data --output-format parquet
```

Download only run metadata (skip histories):
```bash
wandb-download my-team my-project ./data --runs-only
```

### Output Files

**Pickle format (default):**
- `{entity}_{project}_runs.pkl` - List of run dictionaries
- `{entity}_{project}_histories.pkl` - List of history entry lists per run

**Parquet format:**
- `{entity}_{project}_runs.parquet` - Run metadata as DataFrame
- `{entity}_{project}_histories.parquet` - All history entries flattened

## Library Usage

```python
from dr_wandb import fetch_project_runs, serialize_run, serialize_history_entry

# Fetch all runs and histories
runs, histories = fetch_project_runs(
    entity="my-team",
    project="my-project",
    include_history=True,
)

# Each run is a dict with keys:
# run_id, run_name, state, project, entity, created_at,
# config, summary, wandb_metadata, system_metrics, system_attrs, sweep_info

# Each history entry is a dict with keys:
# run_id, step, timestamp, runtime, wandb_metadata, metrics
```

### Pydantic Models

The library uses pydantic models for type-safe data handling:

```python
from dr_wandb.run_record import RunRecord
from dr_wandb.history_entry_record import HistoryEntryRecord

# Convert a wandb run to a typed record
record = RunRecord.from_wandb_run(wandb_run)
data = record.model_dump()  # Get as dict
json_str = record.model_dump_json()  # Get as JSON string
```

## Data Schema

**RunRecord fields:**
| Field | Type | Description |
|-------|------|-------------|
| run_id | str | Unique run identifier |
| run_name | str | Human-readable run name |
| state | str | finished, running, crashed, failed, killed |
| project | str | Project name |
| entity | str | Entity (user/team) name |
| created_at | datetime | Run creation timestamp |
| config | dict | Experiment configuration |
| summary | dict | Final metrics and outputs |
| wandb_metadata | dict | Platform metadata |
| system_metrics | dict | Hardware/system info |
| system_attrs | dict | Additional system attributes |
| sweep_info | dict | Hyperparameter sweep info |

**HistoryEntryRecord fields:**
| Field | Type | Description |
|-------|------|-------------|
| run_id | str | Parent run identifier |
| step | int | Training step number |
| timestamp | datetime | Time of logging |
| runtime | int | Seconds since run start |
| wandb_metadata | dict | Platform logging metadata |
| metrics | dict | All logged metrics |

## License

MIT
