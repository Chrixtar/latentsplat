from dataclasses import dataclass
from fractions import Fraction
import functools
from typing import Literal

from timm.models import resnet
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn
from transformers import ResNetBackbone, ResNetConfig

from .backbone import Backbone
from ....misc.fraction_utils import get_integer


@dataclass
class BackboneResnetCfg:
    name: Literal["resnet"]
    model: ResNetConfig | Literal[
        "microsoft/resnet-18", 
        "microsoft/resnet-34", 
        "microsoft/resnet-50", 
        "microsoft/resnet-101", 
        "microsoft/resnet-152", 
        "Ramos-Ramos/dino-resnet-50"
    ]
    num_layers: int | None = None
    use_first_pool: bool = False


class BackboneResnet(Backbone[BackboneResnetCfg]):
    model: resnet.ResNet

    def __init__(
        self, 
        cfg: BackboneResnetCfg, 
        d_in: int, 
        d_out: int,
        scale_factor: Fraction
    ) -> None:
        super().__init__(cfg, d_in, d_out, scale_factor)
        # out_indices = list(range(cfg.num_layers)) if cfg.num_layers is not None else None
        if isinstance(cfg.model, str):
            assert d_in == 3
            self.model = ResNetBackbone.from_pretrained(cfg.model)# , out_indices=out_indices)
        else:
            # cfg.out_indices = out_indices
            self.model = ResNetBackbone(cfg)

        # Optionally remove first pooling
        if not self.cfg.use_first_pool:
            self.model.embedder.pooler = nn.Identity()

        # Replace BatchNorm with InstanceNorm
        norm_layer = functools.partial(
            nn.InstanceNorm2d,
            affine=False,
            track_running_stats=False,
        )
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                bn = getattr(self.model, name)
                setattr(self.model, name, norm_layer(bn.num_features))

        # Set up projections
        self.projections = nn.ModuleList([nn.Conv2d(self.model.config.embedding_size, self.d_out, 1)])
        for i in range(cfg.num_layers):
            self.projections.append(nn.Conv2d(self.model.config.hidden_sizes[i], self.d_out, 1))

    def forward(
        self,
        x: Float[Tensor, "batch d_in height width"],
    ) -> Float[Tensor, "batch d_out h w"]:
        """
        Where h == height // downscale_factor, w == width // downscale_factor
        """
        spatial = x.shape[-2:]

        x = self.model.embedder(x)
        features = [self.projections[0](x)]
        for index in range(self.cfg.num_layers):
            x = self.model.encoder.stages[index](x)
            features.append(self.projections[index+1](x))

        # Upscale the features.
        features = [
            F.interpolate(
                f, 
                tuple(get_integer(self.scale_factor * s) for s in spatial), 
                mode="bilinear", 
                align_corners=True
            )
            for f in features
        ]
        features = torch.stack(features).sum(dim=0)
        return features
