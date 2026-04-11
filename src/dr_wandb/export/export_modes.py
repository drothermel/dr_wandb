from __future__ import annotations

from enum import StrEnum


class ExportMode(StrEnum):
    METADATA = "metadata"
    HISTORY = "history"


class FetchMode(StrEnum):
    INCREMENTAL = "incremental"
    FULL_RECONCILE = "full_reconcile"
