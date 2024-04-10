from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor, nn


T = TypeVar("T")


class Discriminator(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self, 
        input: Float[Tensor, "batch in_dim height width"]
    ) -> Float[Tensor, "batch 1 down_h down_w"]:
        """
        Where down_h == height // self.downscale_factor,
              down_w == width // self.downscale_factor
        """
        pass
    
    @property
    @abstractmethod
    def downscale_factor(self) -> int:
        pass
