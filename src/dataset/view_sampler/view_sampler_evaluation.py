import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from dacite import Config, from_dict

from ...evaluation.types import IndexEntry
from ...misc.step_tracker import StepTracker
from ..types import Stage
from .view_sampler import ViewIndex, ViewSampler


@dataclass
class ViewSamplerEvaluationCfg:
    name: Literal["evaluation"]
    index_path: Path
    num_context_views: int


class ViewSamplerEvaluation(ViewSampler[ViewSamplerEvaluationCfg]):
    index: dict[str, list[IndexEntry]]

    def __init__(
        self,
        cfg: ViewSamplerEvaluationCfg,
        stage: Stage,
        is_overfitting: bool,
        cameras_are_circular: bool,
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__(cfg, stage, is_overfitting, cameras_are_circular, step_tracker)

        dacite_config = Config(cast=[tuple])
        with cfg.index_path.open("r") as f:
            self.index = {
                k: [from_dict(IndexEntry, v, dacite_config) for v in views]
                for k, views in json.load(f).items()
            }
            self.total_samples = sum(len(views) for views in self.index.values())

    def sample(
        self,
        scene: str,
        num_views: int,
        device: torch.device = torch.device("cpu"),
    ) -> list[ViewIndex]:
        entries = self.index.get(scene)
        if not entries:
            raise ValueError(f"No indices available for scene {scene}.")
        return [
            ViewIndex(
                torch.tensor(entry.context, dtype=torch.int64, device=device),
                torch.tensor(entry.target, dtype=torch.int64, device=device)
            )
            for entry in entries
        ]

    @property
    def num_context_views(self) -> int:
        return 0

    @property
    def num_target_views(self) -> int:
        return 0
