from dataclasses import dataclass
from typing import Literal

import torch

from .view_sampler import ViewIndex, ViewSampler


@dataclass
class ViewSamplerArbitraryCfg:
    name: Literal["arbitrary"]
    num_context_views: int
    num_target_views: int
    context_views: list[int] | None
    target_views: list[int] | None


class ViewSamplerArbitrary(ViewSampler[ViewSamplerArbitraryCfg]):
    def sample(
        self,
        scene: str,
        num_views: int,
        device: torch.device = torch.device("cpu"),
    ) -> list[ViewIndex]:
        """Arbitrarily sample context and target views."""

        index_context = torch.randint(
            0,
            num_views,
            size=(self.cfg.num_context_views,),
            device=device,
        )

        # Allow the context views to be fixed.
        if self.cfg.context_views is not None:
            assert len(self.cfg.context_views) == self.cfg.num_context_views
            index_context = torch.tensor(
                self.cfg.context_views, dtype=torch.int64, device=device
            )

        index_target = torch.randint(
            0,
            num_views,
            size=(self.cfg.num_target_views,),
            device=device,
        )

        # Allow the target views to be fixed.
        if self.cfg.target_views is not None:
            assert len(self.cfg.target_views) == self.cfg.num_target_views
            index_target = torch.tensor(
                self.cfg.target_views, dtype=torch.int64, device=device
            )

        return [ViewIndex(index_context, index_target)]

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
