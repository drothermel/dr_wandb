from collections.abc import Callable
from typing import Literal

from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


MAX_INT = 2**31 - 1

SUPPORTED_FILTER_FIELDS = ["project", "entity", "state", "run_ids"]
type FilterField = Literal["project", "entity", "state", "run_ids"]

WANDB_RUN_STATES = ["finished", "running", "crashed", "failed", "killed"]
type RunState = Literal["finished", "running", "crashed", "failed", "killed"]
type RunId = str

Base.type_annotation_map = {RunId: String}

type ProgressCallback = Callable[[int, int, str], None]
