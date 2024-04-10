from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn
from transformers import ViTModel, ViTConfig

from .backbone import Backbone
from ....misc.fraction_utils import get_integer


@dataclass
class BackboneViTCfg:
    name: Literal["vit"]
    model: ViTConfig | Literal[
        "facebook/dino-vits16", 
        "facebook/dino-vits8", 
        "facebook/dino-vitb16", 
        "facebook/dino-vitb8"
    ]
    upscale_mode: Literal["interpolate", "repeat"] = "interpolate"


class BackboneViT(Backbone[BackboneViTCfg]):
    def __init__(
        self, 
        cfg: BackboneViTCfg, 
        d_in: int,
        d_out: int,
        scale_factor: Fraction
    ) -> None:
        super().__init__(cfg, d_in, d_out, scale_factor)
        if isinstance(self.cfg.model, str):
            assert d_in == 3 
            self.model = ViTModel.from_pretrained(self.cfg.model, add_pooling_layer=False)
        else:
            self.model = ViTModel(self.cfg.model, add_pooling_layer=False)

        if self.cfg.upscale_mode == "repeat":
            self.n_repeats = get_integer(self.patch_size * self.scale_factor)

        self.global_token_mlp = nn.Sequential(
                nn.Linear(768, 768),
                nn.ReLU(),
                nn.Linear(768, self.d_out),
        )
        self.local_token_mlp = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, self.d_out),
        )

    def forward(
        self,
        x: Float[Tensor, "batch d_in height width"],
    ) -> Float[Tensor, "batch d_out h w"]:
        """
        Where h == height // downscale_factor, w == width // downscale_factor
        """
        # Compute features from the DINO-pretrained ViT.
        b, _, h, w = x.shape
        assert h % self.patch_size == 0 and w % self.patch_size == 0
        tokens = self.model(x, interpolate_pos_encoding=True).last_hidden_state

        global_token = self.global_token_mlp(tokens[:, 0])
        local_tokens = self.local_token_mlp(tokens[:, 1:])
        # Broadcast global_token to have spatial dimensions
        global_token = global_token.view(b, -1, 1, 1)
        
        if self.cfg.upscale_mode == "interpolate":
            local_tokens = rearrange(
                local_tokens, 
                "b (h w) c -> b c h w",
                h=h // self.patch_size,
                w=w // self.patch_size
            )
            local_tokens = F.interpolate(
                local_tokens, 
                tuple(get_integer(self.scale_factor * s) for s in (h, w)), 
                mode="bilinear", 
                align_corners=True
            )
        elif self.cfg.upscale_mode == "repeat":
            # repeat as in pixelSplat
            local_tokens = repeat(
                local_tokens,
                "b (h w) c -> b c (h hps) (w wps)",
                h=h // self.patch_size,
                hps=self.n_repeats,
                w=w // self.patch_size,
                wps=self.n_repeats,
            )
        else:
            raise ValueError(f"Unknown upscale_mode {self.cfg.upscale_mode}")

        return local_tokens + global_token

    @property
    def patch_size(self) -> int:
        return self.model.config.patch_size
