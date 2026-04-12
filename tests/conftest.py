from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import wandb

from tests.helpers import FakeApi, FakeRun

FakeApiInstaller = Callable[[list[FakeRun]], None]


@pytest.fixture
def install_fake_wandb_api(monkeypatch: Any) -> FakeApiInstaller:
    """Install a FakeApi as `wandb.Api` so ExportEngine fetches from the given run list."""

    def install(runs: list[FakeRun]) -> None:
        monkeypatch.setattr(wandb, "Api", lambda **kwargs: FakeApi(runs))

    return install
