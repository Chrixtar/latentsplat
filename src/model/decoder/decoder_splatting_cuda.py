from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ..diagonal_gaussian_distribution import DiagonalGaussianDistribution
from ..types import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda, RenderOutput, render_depth_cuda
from .decoder import Decoder, DecoderOutput


@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
        background_color: list[float] = [0., 0., 0.],
        variational: bool = False
    ) -> None:
        super().__init__(cfg)
        self.register_buffer(
            "background_color",
            torch.tensor(background_color, dtype=torch.float32),
            persistent=False,
        )
        self.variational = variational

    def render_to_decoder_output(
        self,
        render_output: RenderOutput,
        b: int,
        v: int
    ) -> DecoderOutput:
        if render_output.feature is not None:
            features = rearrange(render_output.feature, "(b v) c h w -> b v c h w", b=b, v=v)
            # NOTE background feature = 0 = mean = logvar (of normal distribution)
            mean, logvar = features.chunk(2, dim=2) if self.variational \
                else (features, (1-rearrange(render_output.mask.detach(), "(b v) h w -> b v () h w", b=b, v=v)).log().expand_as(features))
            feature_posterior = DiagonalGaussianDistribution(mean, logvar)
        else:
            feature_posterior = None
        return DecoderOutput(
            color=rearrange(render_output.color, "(b v) c h w -> b v c h w", b=b, v=v) if render_output.color is not None else None,
            feature_posterior=feature_posterior,
            mask=rearrange(render_output.mask, "(b v) h w -> b v h w", b=b, v=v),
            depth=rearrange(render_output.depth, "(b v) h w -> b v h w", b=b, v=v)
        )

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        return_colors: bool = True,
        return_features: bool = True
    ) -> DecoderOutput:
        b, v, _, _ = extrinsics.shape
        color_sh = repeat(gaussians.color_harmonics, "b g c d_sh -> (b v) g c d_sh", v=v) \
            if return_colors and gaussians.color_harmonics is not None else None
        feature_sh = repeat(gaussians.feature_harmonics, "b g c d_sh -> (b v) g c d_sh", v=v) \
            if return_features and gaussians.feature_harmonics is not None else None
        rendered: RenderOutput = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(self.background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            color_sh,
            feature_sh
        )
        out = self.render_to_decoder_output(rendered, b, v)
        if depth_mode is not None and depth_mode != "depth":
            out.depth = self.render_depth(gaussians, extrinsics, intrinsics, near, far, image_shape, depth_mode)
        return out

    def render_depth(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        mode: DepthRenderingMode = "depth",
    ) -> Float[Tensor, "batch view height width"]:
        b, v, _, _ = extrinsics.shape
        result = render_depth_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            mode=mode,
        )
        return rearrange(result, "(b v) h w -> b v h w", b=b, v=v)


    def last_layer_weights(self) -> None:
        return None
