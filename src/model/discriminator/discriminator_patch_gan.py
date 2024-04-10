from dataclasses import dataclass
from functools import partial
import os
from typing import Literal

from jaxtyping import Float
import torch
from torch import nn, Tensor

from src.constants import PRETRAINED_DISCRIMINATOR_PATH
from .discriminator import Discriminator


@dataclass
class DiscriminatorPatchGanCfg:
    name: Literal["patch_gan"]
    model: Literal["kl_f8", "kl_f16", "kl_f32"]
    base_dim: int = 64
    max_dim_mult: int = 8
    n_layers: int = 3
    downscale_factor: int = 2
    kernel_size: int = 4
    padding: int = 1
    leaky_relu_neg_slope: float = 0.2
    pretrained: bool = True


class DiscriminatorPatchGan(Discriminator[DiscriminatorPatchGanCfg]):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(
        self,
        cfg: DiscriminatorPatchGanCfg,
        d_in: int = 3
    ):
        super(DiscriminatorPatchGan, self).__init__(cfg)
        norm_layer = nn.BatchNorm2d
        use_bias = False    # no need to use bias as BatchNorm2d has affine parameters

        act_layer = partial(nn.LeakyReLU, negative_slope=self.cfg.leaky_relu_neg_slope, inplace=True)
        layers = [
            nn.Conv2d(
                d_in, 
                self.cfg.base_dim, 
                kernel_size=self.cfg.kernel_size, 
                stride=self.cfg.downscale_factor, 
                padding=self.cfg.padding, 
                bias=True
            ),
            act_layer()
        ]
        dim_mult = 1
        dim_mult_prev = 1
        conv_layer = partial(nn.Conv2d, kernel_size=self.cfg.kernel_size, stride=self.cfg.downscale_factor, padding=self.cfg.padding, bias=use_bias)
        for n in range(1, self.cfg.n_layers):
            dim_mult_prev = dim_mult
            dim_mult = min(self.cfg.downscale_factor ** n, self.cfg.max_dim_mult)
            cur_channels = self.cfg.base_dim * dim_mult
            layers += [
                conv_layer(self.cfg.base_dim * dim_mult_prev, cur_channels),
                norm_layer(cur_channels),
                act_layer()
            ]

        dim_mult_prev = dim_mult
        dim_mult = min(self.cfg.downscale_factor ** self.cfg.n_layers, self.cfg.max_dim_mult)
        layers += [
            nn.Conv2d(
                self.cfg.base_dim * dim_mult_prev, 
                self.cfg.base_dim * dim_mult, 
                kernel_size=self.cfg.kernel_size, 
                stride=1, 
                padding=self.cfg.padding, 
                bias=use_bias
            ),
            norm_layer(self.cfg.base_dim * dim_mult),
            act_layer(),
            # output 1 channel prediction map
            nn.Conv2d(
                self.cfg.base_dim * dim_mult, 
                1, 
                kernel_size=self.cfg.kernel_size, 
                stride=1, 
                padding=self.cfg.padding,
                bias=True
            )
        ]
        self.main = nn.Sequential(*layers)

        if self.cfg.pretrained:
            state_dict = torch.load(os.path.join(PRETRAINED_DISCRIMINATOR_PATH, self.cfg.model + ".pt"), map_location="cpu")
            self.load_state_dict(state_dict)

    def forward(
        self, 
        input: Float[Tensor, "batch in_dim height width"]
    ) -> Float[Tensor, "batch 1 down_h down_w"]:
        """
        Where down_h == height // self.downscale_factor,
              down_w == width // self.downscale_factor
        """
        return self.main(input)

    def init_weights(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    @property
    def downscale_factor(self) -> int:
        return self.cfg.downscale_factor ** self.cfg.n_layers
