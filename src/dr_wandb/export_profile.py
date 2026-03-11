from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field


_SAFE_TOKEN = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize(value: str) -> str:
    return _SAFE_TOKEN.sub("_", value).strip("_") or "unknown"


class ExportProfile(BaseModel):
    export_name: str
    data_root: Path
    sync_root: Path | None = None

    @property
    def token(self) -> str:
        return _sanitize(self.export_name)

    @property
    def resolved_sync_root(self) -> Path:
        if self.sync_root is not None:
            return Path(self.sync_root)
        return Path(self.data_root) / ".sync"


class ExportProfilePaths(BaseModel):
    profile: ExportProfile
    output_dir: Path
    state_path: Path
    export_summary_path: Path
    bootstrap_summary_path: Path
    archive_data_root: Path
    archive_sync_root: Path

    @classmethod
    def from_profile(cls, profile: ExportProfile) -> ExportProfilePaths:
        token = profile.token
        data_root = Path(profile.data_root)
        sync_root = profile.resolved_sync_root
        return cls(
            profile=profile,
            output_dir=data_root / token,
            state_path=sync_root / f"{token}__state.json",
            export_summary_path=sync_root / f"{token}__last_export_summary.json",
            bootstrap_summary_path=sync_root / f"{token}__last_bootstrap_summary.json",
            archive_data_root=data_root / "_archive",
            archive_sync_root=sync_root / "_archive",
        )


class ExportProfileArchivePaths(BaseModel):
    output_dir: Path | None = None
    state_path: Path | None = None
    export_summary_path: Path | None = None
    bootstrap_summary_path: Path | None = None


class ExportProfileBootstrapPlan(BaseModel):
    source_dir: Path
    output_dir: Path
    state_path: Path
    export_summary_path: Path
    bootstrap_summary_path: Path
    archives: ExportProfileArchivePaths = Field(default_factory=ExportProfileArchivePaths)
    archive_timestamp: str | None = None


def build_profile(
    *,
    export_name: str | None,
    data_root: str | None,
    sync_root: str | None,
) -> ExportProfile | None:
    if export_name is None and data_root is None and sync_root is None:
        return None
    if export_name is None or data_root is None:
        raise ValueError("--export-name and --data-root are required together")
    return ExportProfile(
        export_name=export_name,
        data_root=Path(data_root),
        sync_root=Path(sync_root) if sync_root is not None else None,
    )


def resolve_profile_paths(profile: ExportProfile) -> ExportProfilePaths:
    return ExportProfilePaths.from_profile(profile)


def resolve_bootstrap_plan(
    *,
    profile_paths: ExportProfilePaths,
    explicit_source_dir: Path | None = None,
) -> ExportProfileBootstrapPlan:
    source_dir = explicit_source_dir or profile_paths.output_dir
    if not source_dir.exists():
        raise ValueError(f"Bootstrap source_dir does not exist: {source_dir}")

    archives = ExportProfileArchivePaths()
    archive_timestamp: str | None = None
    if (
        profile_paths.output_dir.exists()
        or profile_paths.state_path.exists()
        or profile_paths.export_summary_path.exists()
        or profile_paths.bootstrap_summary_path.exists()
    ):
        archive_timestamp = _utc_timestamp_token()
        archives = ExportProfileArchivePaths(
            output_dir=(
                profile_paths.archive_data_root
                / f"{profile_paths.profile.token}__{archive_timestamp}"
                if profile_paths.output_dir.exists()
                else None
            ),
            state_path=(
                profile_paths.archive_sync_root
                / f"{profile_paths.profile.token}__state__{archive_timestamp}.json"
                if profile_paths.state_path.exists()
                else None
            ),
            export_summary_path=(
                profile_paths.archive_sync_root
                / f"{profile_paths.profile.token}__last_export_summary__{archive_timestamp}.json"
                if profile_paths.export_summary_path.exists()
                else None
            ),
            bootstrap_summary_path=(
                profile_paths.archive_sync_root
                / f"{profile_paths.profile.token}__last_bootstrap_summary__{archive_timestamp}.json"
                if profile_paths.bootstrap_summary_path.exists()
                else None
            ),
        )

    should_repoint_source = False
    if archives.output_dir is not None:
        if explicit_source_dir is None:
            should_repoint_source = True
        else:
            should_repoint_source = (
                source_dir.resolve() == profile_paths.output_dir.resolve()
            )
    archived_source_dir = archives.output_dir if should_repoint_source else source_dir
    return ExportProfileBootstrapPlan(
        source_dir=archived_source_dir,
        output_dir=profile_paths.output_dir,
        state_path=profile_paths.state_path,
        export_summary_path=profile_paths.export_summary_path,
        bootstrap_summary_path=profile_paths.bootstrap_summary_path,
        archives=archives,
        archive_timestamp=archive_timestamp,
    )


def apply_bootstrap_archives(plan: ExportProfileBootstrapPlan) -> None:
    _archive_path(plan.output_dir, plan.archives.output_dir)
    _archive_path(plan.state_path, plan.archives.state_path)
    _archive_path(plan.export_summary_path, plan.archives.export_summary_path)
    _archive_path(plan.bootstrap_summary_path, plan.archives.bootstrap_summary_path)


def validate_explicit_path_mode(
    *,
    profile: ExportProfile | None,
    explicit_output_dir: str | None = None,
    explicit_state_path: str | None = None,
    explicit_output_json: str | None = None,
) -> None:
    if profile is None:
        return
    conflicts: list[str] = []
    if explicit_output_dir is not None:
        conflicts.append("output_dir")
    if explicit_state_path is not None:
        conflicts.append("--state-path")
    if explicit_output_json is not None:
        conflicts.append("--output-json")
    if conflicts:
        joined = ", ".join(conflicts)
        raise ValueError(
            f"Profile mode cannot be combined with explicit path arguments: {joined}"
        )


def _archive_path(source: Path, destination: Path | None) -> None:
    if destination is None or not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)


def _utc_timestamp_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
