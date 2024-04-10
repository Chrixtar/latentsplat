from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

from jaxtyping import Float
from torch import nn, Tensor

from .backbone import Backbone
from .backbone_dino import BackboneDino, BackboneDinoCfg   # TODO remove BackboneDino (moved into BackboneViT)
from .backbone_resnet import BackboneResnet, BackboneResnetCfg
from .backbone_vit import BackboneViT, BackboneViTCfg


BACKBONES: dict[str, Backbone] = {
    "dino": BackboneDino,
    "resnet": BackboneResnet,
    "vit": BackboneViT,
}


SingleBackboneCfg = BackboneResnetCfg | BackboneViTCfg | BackboneDinoCfg
BackboneCfg = SingleBackboneCfg | list[SingleBackboneCfg]


@dataclass
class BackboneEnsembleCfg:
    name: Literal["ensemble"]
    components: list[SingleBackboneCfg]


class BackboneEnsemble(Backbone[BackboneEnsembleCfg]):
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


def get_backbone(
    cfg: BackboneCfg, 
    d_in: int, 
    d_out: int, 
    scale_factor: Fraction
) -> Backbone:
    if isinstance(cfg, list):
        cfg = BackboneEnsembleCfg(name="ensemble", components=cfg)
        return BackboneEnsemble(cfg, d_in, d_out, scale_factor)
    return BACKBONES[cfg.name](cfg, d_in, d_out, scale_factor)
