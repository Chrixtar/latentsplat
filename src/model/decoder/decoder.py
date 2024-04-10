from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from jaxtyping import Float
from torch import Tensor, nn

from ..diagonal_gaussian_distribution import DiagonalGaussianDistribution
from ..types import Gaussians

DepthRenderingMode = Literal[
    "depth",
    "log",
    "disparity",
    "relative_disparity",
]


@dataclass
class DecoderOutput:
    color: Float[Tensor, "batch view 3 height width"] | None
    feature_posterior: DiagonalGaussianDistribution | None
    mask: Float[Tensor, "batch view height width"]
    depth: Float[Tensor, "batch view height width"]

T = TypeVar("T")


class Decoder(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
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
        pass

    @property
    @abstractmethod
    def last_layer_weights(self) -> Tensor | None:
        pass