from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from dr_wandb.export_profile import build_profile, resolve_raw_extract_profile_paths
from dr_wandb.raw_extract import RawExtractConfig, extract_runs_raw

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@app.command()
def main(
    entity: str,
    project: str,
    export_name: str = typer.Option(..., "--export-name"),
    data_root: Path = typer.Option(..., "--data-root"),
    runs_per_page: int = typer.Option(100, "--runs-per-page"),
    timeout_seconds: int = typer.Option(300, "--timeout-seconds"),
    include_metadata: bool = typer.Option(False, "--include-metadata/--no-include-metadata"),
    postprocess_dedup: bool = typer.Option(True, "--postprocess-dedup/--no-postprocess-dedup"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    _setup_logging(log_level)
    profile = build_profile(
        export_name=export_name,
        data_root=str(data_root),
        sync_root=None,
    )
    assert profile is not None, "--export-name and --data-root are required"
    raw_paths = resolve_raw_extract_profile_paths(profile)
    config = RawExtractConfig(
        entity=entity,
        project=project,
        output_dir=raw_paths.output_dir,
        runs_raw_path=raw_paths.runs_raw_path,
        runs_raw_deduping_path=raw_paths.runs_raw_deduping_path,
        runs_per_page=runs_per_page,
        timeout_seconds=timeout_seconds,
        include_metadata=include_metadata,
        postprocess_dedup=postprocess_dedup,
    )
    logging.info(
        "Starting wandb raw extract for %s/%s (runs_per_page=%s, timeout_seconds=%s, include_metadata=%s, postprocess_dedup=%s)",
        entity,
        project,
        runs_per_page,
        timeout_seconds,
        include_metadata,
        postprocess_dedup,
    )
    summary = extract_runs_raw(config)
    typer.echo(json.dumps(summary.model_dump(mode="python"), indent=2, sort_keys=True))


def extract_raw_main() -> None:
    app()


if __name__ == "__main__":
    extract_raw_main()
