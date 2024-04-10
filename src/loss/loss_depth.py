from dataclasses import dataclass
from typing import Literal

import torch
from einops import reduce
from jaxtyping import Float
from torch import Tensor

from ..model.types import Prediction, GroundTruth
from .loss import LossCfg, Loss


@dataclass
class LossDepthCfg(LossCfg):
    name: Literal["depth"] = "depth"
    sigma_image: float | None = None
    use_second_derivative: bool = False


class LossDepth(Loss):
    def unweighted_loss(
        self,
        prediction: Prediction,
        gt: GroundTruth
    ) -> Float[Tensor, ""]:
        # Scale the depth between the near and far planes.
        near = gt.near[..., None, None].log()
        far = gt.far[..., None, None].log()
        depth = prediction.depth.minimum(far).maximum(near)
        depth = (depth - near) / (far - near)

        # Compute the difference between neighboring pixels in each direction.
        depth_dx = depth.diff(dim=-1)
        depth_dy = depth.diff(dim=-2)

        # If desired, compute a 2nd derivative.
        if self.cfg.use_second_derivative:
            depth_dx = depth_dx.diff(dim=-1)
            depth_dy = depth_dy.diff(dim=-2)

        # If desired, add bilateral filtering.
        if self.cfg.sigma_image is not None:
            color_gt = gt.image
            color_dx = reduce(color_gt.diff(dim=-1), "b v c h w -> b v h w", "max")
            color_dy = reduce(color_gt.diff(dim=-2), "b v c h w -> b v h w", "max")
            if self.cfg.use_second_derivative:
                color_dx = color_dx[..., :, 1:].maximum(color_dx[..., :, :-1])
                color_dy = color_dy[..., 1:, :].maximum(color_dy[..., :-1, :])
            depth_dx = depth_dx * torch.exp(-color_dx * self.cfg.sigma_image)
            depth_dy = depth_dy * torch.exp(-color_dy * self.cfg.sigma_image)

        return depth_dx.abs().mean() + depth_dy.abs().mean()
