# dr_wandb

A command-line utility for downloading and archiving Weights & Biases experiment data to local storage formats optimized for offline analysis. 


## Installation

CLI Tool Install: `wandb-downloader`
```
uv tool install dr_wandb
```

Or, to use the library functions
```bash
# To use the library functions
uv add dr_wandb
# Optionally
uv add dr_wandb[postgres]
uv sync
```

### Authentication

Configure Weights & Biases authentication using one of these methods:

```bash
wandb login
```

Or set the API key as an environment variable:

```bash
export WANDB_API_KEY=your_api_key_here
```

## Quickstart

The default approach doesn't involve postgres. It fetches the runs, and optionally histories, and dumps them to local pkl files.  

```bash
» wandb-download --help

 Usage: wandb-download [OPTIONS] ENTITY PROJECT OUTPUT_DIR

╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    entity          TEXT  [required]                                                                                                                            │
│ *    project         TEXT  [required]                                                                                                                            │
│ *    output_dir      TEXT  [required]                                                                                                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --runs-only             --no-runs-only             [default: no-runs-only]                                                                                       │
│ --runs-per-page                           INTEGER  [default: 500]                                                                                                │
│ --log-every                               INTEGER  [default: 20]                                                                                                 │
│ --install-completion                               Install completion for the current shell.                                                                     │
│ --show-completion                                  Show completion for the current shell, to copy it or customize the installation.                              │
│ --help                                             Show this message and exit.                                                                                   │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

An example:
```bash
» wandb-download --runs-only "ml-moe" "ft-scaling" "./data"                                          1 ↵
2025-11-10 21:47:54 - INFO -
:: Beginning Dr. Wandb Project Downloading Tool ::

2025-11-10 21:47:54 - INFO - {
    "entity": "ml-me",
    "project": "scaling",
    "output_dir": "data",
    "runs_only": true,
    "runs_per_page": 500,
    "log_every": 20,
    "runs_output_filename": "ml-me_scaling_runs.pkl",
    "histories_output_filename": "ml-me_scaling_histories.pkl"
}
2025-11-10 21:47:54 - INFO -
2025-11-10 21:47:54 - INFO - >> Downloading runs, this will take a while (minutes)
wandb: Currently logged in as: danielle-rothermel (ml-moe) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
2025-11-10 21:48:00 - INFO -   - total runs found: 517
2025-11-10 21:48:00 - INFO - >> Serializing runs and maybe getting histories: False
2025-11-10 21:48:07 - INFO - >> 20/517: 2025_08_21-08_24_43_test_finetune_DD-dolma1_7-10M_main_1Mtx1_--learning_rate=5e-05
2025-11-10 21:48:12 - INFO - >> 40/517: 2025_08_21-08_24_43_test_finetune_DD-dolma1_7-150M_main_10Mtx1_--learning_rate=5e-06
...
2025-11-10 21:50:46 - INFO - >> Dumped runs data to: ./data/ml-moe_ft-scaling_runs.pkl
2025-11-10 21:50:46 - INFO - >> Runs only, not dumping histories to: ./data/ml-moe_ft-scaling_histories.pkl
```



## Very Alpha: Postgres Version

**Its very likely this won't currently work.**  Download all runs from a Weights & Biases project:

```bash
uv run python src/dr_wandb/cli/postres_download.py --entity your_entity --project your_project

Options:
  --entity TEXT        WandB entity (username or team name)
  --project TEXT       WandB project name
  --runs-only          Download only run metadata, skip training history
  --force-refresh      Download all data, ignoring existing records
  --db-url TEXT        PostgreSQL connection string
  --output-dir TEXT    Directory for exported Parquet files
  --help              Show help message and exit
```

The tool creates a PostgreSQL database, downloads experiment data, and exports Parquet files to the configured output directory. It tool tracks existing data and downloads only new or updated runs by default. A run is considered for update if:

- It does not exist in the local database
- Its state is "running" (indicating potential new data)

Use `--force-refresh` to download all runs regardless of existing data.

### Environment Variables

The tool reads configuration from environment variables with the `DR_WANDB_` prefix and supports `.env` files:

| Variable | Description | Default |
|----------|-------------|---------|
| `DR_WANDB_ENTITY` | Weights & Biases entity name | None |
| `DR_WANDB_PROJECT` | Weights & Biases project name | None |
| `DR_WANDB_DATABASE_URL` | PostgreSQL connection string | `postgresql+psycopg2://localhost/wandb` |
| `DR_WANDB_OUTPUT_DIR` | Directory for exported files | `./data` |

### Database Configuration

The PostgreSQL connection string follows the standard format:

```
postgresql+psycopg2://username:password@host:port/database_name
```

If the specified database does not exist, the tool will attempt to create it automatically.

### Data Schema


The tool generates the following files in the output directory:

- `runs_metadata.parquet` - Complete run metadata including configurations, summaries, and system information
- `runs_history.parquet` - Training metrics and logged values over time
- `runs_metadata_{component}.parquet` - Component-specific files for config, summary, wandb_metadata, system_metrics, system_attrs, and sweep_info


**Run Records**
- **run_id**: Unique identifier for the experiment run
- **run_name**: Human-readable name assigned to the run
- **state**: Current state (finished, running, crashed, failed, killed)
- **project**: Project name
- **entity**: Entity name
- **created_at**: Timestamp of run creation
- **config**: Experiment configuration parameters (JSONB)
- **summary**: Final metrics and outputs (JSONB)
- **wandb_metadata**: Platform-specific metadata (JSONB)
- **system_metrics**: Hardware and system information (JSONB)
- **system_attrs**: Additional system attributes (JSONB)
- **sweep_info**: Hyperparameter sweep information (JSONB)

**Training History Records**
- **run_id**: Reference to the parent run
- **step**: Training step number
- **timestamp**: Time of metric logging
- **runtime**: Elapsed time since run start
- **wandb_metadata**: Platform logging metadata (JSONB)
- **metrics**: All logged metrics and values (JSONB, flattened in Parquet export)
