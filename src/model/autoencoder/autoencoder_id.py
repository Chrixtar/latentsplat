from dataclasses import dataclass
from typing import Literal, Optional

from jaxtyping import Float
from torch import Tensor

from .autoencoder import Autoencoder
from ..diagonal_gaussian_distribution import DiagonalGaussianDistribution


@dataclass
class AutoencoderIdCfg:
    name: Literal["id"]
    skip_connections: bool = False


class AutoencoderId(Autoencoder[AutoencoderIdCfg]):
    def __init__(
        self, 
        cfg: AutoencoderIdCfg, 
        d_in: int = 3,
        d_skip_extra: int = 0,
        sample_size: int = 32
    ) -> None:
        super().__init__(cfg)
        self.d_in = d_in
        
    def encode(
        self, 
        images: Float[Tensor, "*#batch d_img height width"]
    ) -> DiagonalGaussianDistribution:
        return DiagonalGaussianDistribution(images)
    
    def decode(
        self, 
        z: Float[Tensor, "*#batch d_latent latent_height latent_width"],
        skip_z: Optional[Float[Tensor, "*#batch d_skip height width"]] = None,
    ) -> Float[Tensor, "*#batch d_img height width"]:
        return z

    @property
    def downscale_factor(self) -> int:
        return 1

    @property
    def d_latent(self) -> int:
        return self.d_in

    @property
    def last_layer_weights(self) -> None:
        return None

    @property
    def expects_skip(self) -> bool:
        return False
    
    @property
    def expects_skip_extra(self) -> bool:
        return False
