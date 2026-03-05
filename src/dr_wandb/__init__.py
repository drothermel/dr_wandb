from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "ErrorAction",
    "ExportConfig",
    "ExportSummary",
    "HistoryWindow",
    "NoopPolicy",
    "PatchPlan",
    "PlannedPatch",
    "ProjectSyncState",
    "RunCursor",
    "RunDecision",
    "RunEvaluation",
    "SyncContext",
    "SyncEngine",
    "SyncPolicy",
    "SyncSummary",
]

_ATTR_TO_MODULE = {
    "SyncEngine": "dr_wandb.sync_engine",
    "SyncPolicy": "dr_wandb.sync_policy",
    "NoopPolicy": "dr_wandb.sync_policy",
    "ErrorAction": "dr_wandb.sync_types",
    "ExportConfig": "dr_wandb.sync_types",
    "ExportSummary": "dr_wandb.sync_types",
    "HistoryWindow": "dr_wandb.sync_types",
    "PatchPlan": "dr_wandb.sync_types",
    "PlannedPatch": "dr_wandb.sync_types",
    "ProjectSyncState": "dr_wandb.sync_types",
    "RunCursor": "dr_wandb.sync_types",
    "RunDecision": "dr_wandb.sync_types",
    "RunEvaluation": "dr_wandb.sync_types",
    "SyncContext": "dr_wandb.sync_types",
    "SyncSummary": "dr_wandb.sync_types",
}


def __getattr__(name: str) -> Any:
    module_name = _ATTR_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module 'dr_wandb' has no attribute {name!r}")
    module = importlib.import_module(module_name)
    return getattr(module, name)
