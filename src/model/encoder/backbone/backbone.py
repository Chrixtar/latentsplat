from abc import ABC, abstractmethod
from fractions import Fraction
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor, nn


T = TypeVar("T")


class Backbone(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(
        self, 
        cfg: T, 
        d_in: int, 
        d_out: int,
        scale_factor: Fraction
    ) -> None:
        super(Backbone, self).__init__()
        self.cfg = cfg
        self.d_in = d_in
        self.d_out = d_out
        self.scale_factor = scale_factor

    @abstractmethod
    def forward(
        self,
        x: Float[Tensor, "batch d_in height width"],
    ) -> Float[Tensor, "batch d_out h w"]:
        """
        Where h == height // downscale_factor, w == width // downscale_factor
        """
        pass
