from __future__ import annotations

from pydantic import BaseModel

from dr_wandb.export.wandb_run import WandbRun


class RunSnapshot(BaseModel):
    run: WandbRun
    exported_at: str

    @property
    def sort_key(self) -> tuple[str, str]:
        return (self.run.created_at or "", self.run.run_id)
