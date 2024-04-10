from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from jaxtyping import Float
from torch import Tensor, nn

from .backbone import Backbone, BackboneCfg
from . import get_backbone


@dataclass
class BackboneEnsembleCfg(BackboneCfg):
    components: list[BackboneCfg]
    name: Literal["ensemble"] = "ensemble"


class BackboneEnsemble(Backbone):
    components: nn.ModuleList

    def __init__(
        self, 
        cfg: BackboneEnsembleCfg, 
        d_in: int,
        d_out: int,
        scale_factor: Fraction
    ) -> None:
        super().__init__(cfg, d_in, d_out, scale_factor)
        self.components = nn.ModuleList(get_backbone(c, d_in, d_out, scale_factor) for c in self.cfg.components)

    def forward(
        self,
        x: Float[Tensor, "batch d_in height width"],
    ) -> Float[Tensor, "batch d_out h w"]:
        """
        Where h == height // downscale_factor, w == width // downscale_factor
        """
        features = self.components[0](x)
        for component in self.components[1:]:
            features = features + component(x)
        return features
