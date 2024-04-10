from dataclasses import dataclass
from typing import Literal

import torch

from .view_sampler import ViewIndex, ViewSampler


@dataclass
class ViewSamplerBoundedCfg:
    name: Literal["bounded"]
    num_context_views: int
    num_target_views: int
    min_distance_between_context_views: int
    max_distance_between_context_views: int
    max_distance_to_context_views: int
    context_gap_warm_up_steps: int
    target_gap_warm_up_steps: int
    initial_min_distance_between_context_views: int
    initial_max_distance_between_context_views: int
    initial_max_distance_to_context_views: int


class ViewSamplerBounded(ViewSampler[ViewSamplerBoundedCfg]):
    def schedule(
        self, 
        initial: int, 
        final: int,
        steps: int
    ) -> int:
        fraction = self.global_step / steps
        return min(initial + int((final - initial) * fraction), final)

    def sample(
        self,
        scene: str,
        num_views: int,
        device: torch.device = torch.device("cpu"),
    ) -> list[ViewIndex]:

        # Compute the context view spacing based on the current global step.
        if self.stage == "test":
            # When testing, always use the full gap.
            max_context_gap = self.cfg.max_distance_between_context_views
            min_context_gap = self.cfg.max_distance_between_context_views
        elif self.cfg.context_gap_warm_up_steps > 0:
            max_context_gap = self.schedule(
                self.cfg.initial_max_distance_between_context_views,
                self.cfg.max_distance_between_context_views,
                self.cfg.context_gap_warm_up_steps
            )
            min_context_gap = self.schedule(
                self.cfg.initial_min_distance_between_context_views,
                self.cfg.min_distance_between_context_views,
                self.cfg.context_gap_warm_up_steps
            )
        else:
            max_context_gap = self.cfg.max_distance_between_context_views
            min_context_gap = self.cfg.min_distance_between_context_views

        if not self.cameras_are_circular:
            max_context_gap = min(num_views - 1, max_context_gap)   # NOTE fixed former bug here

        # Compute the margin from context window to target window based on the current global step
        if self.stage != "test" and self.cfg.target_gap_warm_up_steps > 0:
            max_target_gap = self.schedule(
                self.cfg.initial_max_distance_to_context_views,
                self.cfg.max_distance_to_context_views,
                self.cfg.target_gap_warm_up_steps
            )
        else:
            max_target_gap = self.cfg.max_distance_to_context_views

        # Pick the gap between the context views.
        if max_context_gap < min_context_gap:
            raise ValueError("Example does not have enough frames!")
        context_gap = torch.randint(
            min_context_gap,
            max_context_gap + 1,
            size=tuple(),
            device=device,
        ).item()

        # Pick the left and right context indices.
        index_context_left = torch.randint(
            low=0,
            high=num_views if self.cameras_are_circular else num_views - context_gap,
            size=tuple(),
            device=device,
        ).item()
        if self.stage == "test":
            index_context_left = index_context_left * 0
        index_context_right = index_context_left + context_gap

        if self.is_overfitting:
            index_context_left *= 0
            index_context_right *= 0
            index_context_right += max_context_gap

        index_target_left = index_context_left - max_target_gap
        index_target_right = index_context_right + max_target_gap

        if not self.cameras_are_circular:
            index_target_left = max(0, index_target_left)
            index_target_right = min(num_views-1, index_target_right)

        # Pick the target view indices.
        if self.stage == "test":
            # When testing, pick all.
            index_target = torch.arange(
                index_target_left,
                index_target_right + 1,
                device=device,
            )
        else:
            # When training or validating (visualizing), pick at random.
            index_target = torch.randint(
                index_target_left,
                index_target_right + 1 ,
                size=(self.cfg.num_target_views,),
                device=device,
            )

        # Apply modulo for circular datasets.
        if self.cameras_are_circular:
            index_target %= num_views
            index_context_right %= num_views

        return [ViewIndex(torch.tensor((index_context_left, index_context_right)), index_target)]

    @property
    def num_context_views(self) -> int:
        return 2

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
