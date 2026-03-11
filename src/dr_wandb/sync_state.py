from __future__ import annotations

import getpass
import json
import os
import re
import socket
import tempfile
from pathlib import Path

from dr_wandb.sync_types import ProjectSyncState

_SAFE_TOKEN = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize(value: str) -> str:
    return _SAFE_TOKEN.sub("_", value).strip("_") or "unknown"


def default_state_path(
    entity: str,
    project: str,
    *,
    user: str | None = None,
    host: str | None = None,
    root_dir: Path | None = None,
) -> Path:
    resolved_user = _sanitize(user or getpass.getuser())
    resolved_host = _sanitize((host or socket.gethostname()).split(".")[0])
    cache_dir = root_dir or (Path.home() / ".cache" / "dr_wandb")
    filename = f"{_sanitize(entity)}__{_sanitize(project)}__{resolved_user}__{resolved_host}.json"
    return cache_dir / filename


def load_state(path: Path, *, entity: str, project: str) -> ProjectSyncState:
    if not path.exists():
        return ProjectSyncState(entity=entity, project=project)

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    state = ProjectSyncState.model_validate(raw)
    if not state.entity:
        state.entity = entity
    if not state.project:
        state.project = project
    return state


def save_state(state: ProjectSyncState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name,
        suffix=".tmp",
        delete=False,
    ) as tmp_file:
        json.dump(state.model_dump(mode="python"), tmp_file, indent=2, sort_keys=True)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        tmp_name = tmp_file.name

    os.replace(tmp_name, path)
