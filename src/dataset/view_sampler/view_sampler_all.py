from dataclasses import dataclass
from typing import Literal

import torch

from .view_sampler import ViewIndex, ViewSampler


@dataclass
class ViewSamplerAllCfg:
    name: Literal["all"]


class ViewSamplerAll(ViewSampler[ViewSamplerAllCfg]):
    def sample(
        self,
        scene: str,
        num_views: int,
        device: torch.device = torch.device("cpu"),
    ) -> list[ViewIndex]:
    
        all_frames = torch.arange(num_views, device=device)
        return [ViewIndex(all_frames, all_frames)]

    @property
    def num_context_views(self) -> int:
        return 0

    @property
    def num_target_views(self) -> int:
        return 0
