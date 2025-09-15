# dr_wandb

A command-line utility for downloading and archiving Weights & Biases experiment data to local storage formats optimized for offline analysis. Stores to PostgreSQL db + Parquet files, supports incremental updates and selective data retrieval.

## Installation

```bash
uv add dr_wandb
```

### Prerequisites

- Python 3.12 or higher
- PostgreSQL database server
- Weights & Biases account with API access
- PyArrow for Parquet file operations

### Authentication

Configure Weights & Biases authentication using one of these methods:

```bash
wandb login
```

Or set the API key as an environment variable:

```bash
export WANDB_API_KEY=your_api_key_here
```

## Basic Usage

Download all runs from a Weights & Biases project:

```bash
wandb-download --entity your_entity --project your_project

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

## Data Schema


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

