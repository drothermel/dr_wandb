from pathlib import Path
from typing import Literal

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data"
DEFAULT_RUNS_FILENAME = "runs_metadata.parquet"
DEFAULT_HISTORY_FILENAME = "runs_history.parquet"
SUPPORTED_FILTER_FIELDS = ["project", "entity", "state", "run_ids"]

type RunId = str
WANDB_RUN_STATES = ["finished", "running", "crashed", "failed", "killed"]
type RunState = Literal["finished", "running", "crashed", "failed", "killed"]

RUN_DATA_COMPONENTS = [
    "config",
    "summary",
    "metadata",
    "system_metrics",
    "system_attrs",
    "sweep_info",
]
type All = Literal["all"]
type RunDataComponent = Literal[
    "config",
    "summary",
    "metadata",
    "system_metrics",
    "system_attrs",
    "sweep_info",
]

QUERY_TYPES = ["runs", "history"]
type QueryType = Literal["runs", "history"]
