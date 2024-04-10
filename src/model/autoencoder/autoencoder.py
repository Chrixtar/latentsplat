from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

from jaxtyping import Float
from torch import Tensor, nn

from ..diagonal_gaussian_distribution import DiagonalGaussianDistribution


T = TypeVar("T")

class Autoencoder(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(
        self, 
        cfg: T
    ) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def encode(
        self,
        images: Float[Tensor, "*#batch d_img height width"]
    ) -> DiagonalGaussianDistribution:
        pass

    @abstractmethod
    def decode(
        self, 
        z: Float[Tensor, "*#batch d_latent latent_height latent_width"],
        skip_z: Optional[Float[Tensor, "*#batch d_skip height width"]] = None,
    ) -> Float[Tensor, "*#batch d_img height width"]:
        pass
    
    @property
    @abstractmethod
    def downscale_factor(self) -> int:
        pass
    
    @property
    @abstractmethod
    def d_latent(self) -> int:
        pass
    
    @property
    @abstractmethod
    def last_layer_weights(self) -> Tensor | None:
        pass

    @property
    @abstractmethod
    def expects_skip(self) -> bool:
        pass

    @property
    @abstractmethod
    def expects_skip_extra(self) -> bool:
        pass
