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


@click.command()
@click.option(
    "--entity",
    envvar="DR_WANDB_ENTITY",
    help="WandB entity (username or team name)",
)
@click.option("--project", envvar="DR_WANDB_PROJECT", help="WandB project name")
@click.option(
    "--runs_only",
    is_flag=True,
    help="Only download runs, don't download history",
)
@click.option(
    "--force_refresh",
    is_flag=True,
    help="Force refresh, download all data",
)
@click.option(
    "--db_url",
    envvar="DR_WANDB_DATABASE_URL",
    help="PostgreSQL connection string",
)
@click.option(
    "--output_dir",
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
    cfg = ProjDownloadSettings()
    final_entity = entity if entity else cfg.entity
    final_project = project if project else cfg.project
    final_db_url = db_url if db_url else cfg.database_url
    final_output_dir = output_dir if output_dir else cfg.output_dir
    if not final_entity or not final_project:
        raise click.ClickException("--entity and --project are required")

    store = ProjectStore(
        final_db_url,
        output_dir=final_output_dir,
    )
    downloader = Downloader(store, runs_per_page=cfg.runs_per_page)
    logging.info(f"Downloading project {final_entity}/{final_project}")
    stats = downloader.download_project(
        entity=final_entity,
        project=final_project,
        runs_only=runs_only,
        force_refresh=force_refresh,
    )
    logging.info(str(stats))
    downloader.write_downloaded_to_parquet()


if __name__ == "__main__":
    download_project()
