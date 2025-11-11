import logging
from pathlib import Path

import click
from pydantic_settings import BaseSettings, SettingsConfigDict

from dr_wandb.downloader import Downloader
from dr_wandb.store import ProjectStore


class ProjDownloadSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="DR_WANDB_")

    entity: str | None = None
    project: str | None = None
    database_url: str = "postgresql+psycopg2://localhost/wandb"
    output_dir: Path = Path(__file__).parent.parent / "data"
    runs_per_page: int = 500


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def validate_settings(entity: str | None, project: str | None) -> None:
    if not entity:
        raise click.ClickException(
            "--entity is required, or set DR_WANDB_ENTITY in .env"
        )
    if not project:
        raise click.ClickException(
            "--project is required, or set DR_WANDB_PROJECT in .env"
        )


def resolve_config(
    entity: str | None,
    project: str | None,
    db_url: str | None,
    output_dir: str | None,
) -> ProjDownloadSettings:
    cfg = ProjDownloadSettings()
    final_entity = entity if entity else cfg.entity
    final_project = project if project else cfg.project
    final_db_url = db_url if db_url else cfg.database_url
    final_output_dir = output_dir if output_dir else cfg.output_dir
    validate_settings(final_entity, final_project)
    return ProjDownloadSettings(
        entity=final_entity,
        project=final_project,
        database_url=final_db_url,
        output_dir=final_output_dir,
        runs_per_page=cfg.runs_per_page,
    )


def execute_download(
    cfg: ProjDownloadSettings, runs_only: bool, force_refresh: bool
) -> None:
    store = ProjectStore(
        cfg.database_url,
        output_dir=cfg.output_dir,
    )
    downloader = Downloader(store, runs_per_page=cfg.runs_per_page)
    click.echo(">> Beginning download:")
    stats = downloader.download_project(
        entity=cfg.entity,
        project=cfg.project,
        runs_only=runs_only,
        force_refresh=force_refresh,
    )
    click.echo(str(stats))
    return downloader


@click.command()
@click.option(
    "--entity",
    envvar="DR_WANDB_ENTITY",
    help="WandB entity (username or team name)",
)
@click.option("--project", envvar="DR_WANDB_PROJECT", help="WandB project name")
@click.option(
    "--runs-only",
    is_flag=True,
    help="Only download runs, don't download history",
)
@click.option(
    "--force-refresh",
    is_flag=True,
    help="Force refresh, download all data",
)
@click.option(
    "--db-url",
    envvar="DR_WANDB_DATABASE_URL",
    help="PostgreSQL connection string",
)
@click.option(
    "--output-dir",
    envvar="DR_WANDB_OUTPUT_DIR",
    help="Output directory",
)
def download_project(
    entity: str | None,
    project: str | None,
    runs_only: bool,
    force_refresh: bool,
    db_url: str | None,
    output_dir: str | None,
) -> None:
    setup_logging()
    click.echo("\n:: Beginning Dr. Wandb Project Downloading Tool ::\n")
    cfg = resolve_config(entity, project, db_url, output_dir)
    click.echo(f">> Downloading project {cfg.entity}/{cfg.project}")
    click.echo(f">> Database: {cfg.database_url}")
    click.echo(f">> Output directory: {cfg.output_dir}")
    click.echo(f">> Force refresh: {force_refresh} Runs only: {runs_only}")
    click.echo()
    downloader = execute_download(cfg, runs_only, force_refresh)
    downloader.write_downloaded_to_parquet()


if __name__ == "__main__":
    download_project()
