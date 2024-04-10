from dataclasses import dataclass
import os
from typing import Literal, Optional, Union

from diffusers import (
    AutoencoderKL as Model,
)
from diffusers.models.autoencoders.vae import (
    DecoderOutput,
    DiagonalGaussianDistribution as LatentDistribution,
)
from diffusers.utils.accelerate_utils import apply_forward_hook
from jaxtyping import Float
import torch
from torch import nn, Tensor
from torch.nn.functional import interpolate

from src.constants import PRETRAINED_AUTOENCODER_PATH
from .autoencoder import Autoencoder
from ..diagonal_gaussian_distribution import DiagonalGaussianDistribution
from ...misc.nn_module_tools import zero_module


@dataclass
class AutoencoderKLCfg:
    name: Literal["kl"]
    model: Literal["kl_f8", "kl_f16", "kl_f32"]
    down_block_types: list[str]
    up_block_types: list[str]
    block_out_channels: list[int]
    layers_per_block: int
    latent_channels: int
    skip_connections: bool = False
    skip_extra: bool = True
    skip_zero: bool = True
    pretrained: bool = True


class AutoencoderKL(Autoencoder[AutoencoderKLCfg]):
    def __init__(
        self, 
        cfg: AutoencoderKLCfg, 
        d_in: int = 3,
        d_skip_extra: int = 0,
        sample_size: int = 32
    ) -> None:
        super().__init__(cfg)
        self.model = Model(
            d_in,
            d_in,
            down_block_types=self.cfg.down_block_types,
            up_block_types=self.cfg.up_block_types,
            block_out_channels=self.cfg.block_out_channels,
            layers_per_block=self.cfg.layers_per_block,
            latent_channels=self.cfg.latent_channels,
            sample_size=sample_size
        )
        if self.cfg.pretrained:
            state_dict = torch.load(os.path.join(PRETRAINED_AUTOENCODER_PATH, self.cfg.model + ".pt"), map_location="cpu")
            self.model.load_state_dict(state_dict)
        if self.cfg.skip_connections:
            # Add zero convs for high-resolution skip connections
            self.d_skip = self.d_latent
            if self.cfg.skip_extra:
                self.d_skip += d_skip_extra
            skip = nn.Conv2d(self.d_skip, self.cfg.block_out_channels[-1], kernel_size=1)
            if self.cfg.skip_zero:
                skip = zero_module(skip)
            self.skip_convs = nn.ModuleList([skip])
            for dec_d_in in reversed(self.cfg.block_out_channels):
                skip = nn.Conv2d(self.d_skip, dec_d_in, kernel_size=1)
                if self.cfg.skip_zero:
                    skip = zero_module(skip)
                self.skip_convs.append(skip)
        
    def encode(
        self, 
        images: Float[Tensor, "*#batch d_img height width"]
    ) -> DiagonalGaussianDistribution:
        """
        Expects channels to be in range [0, 1]
        Returns distribution with original number of batch dimensions
        """
        batch_dims = images.shape[:-3]
        images = 2 * images - 1         # normalize to [-1, 1]
        images = images.flatten(0, -4)  # make sure to have just one batch dimension
        latent_dist: LatentDistribution = self.model.encode(images).latent_dist
        return DiagonalGaussianDistribution(
            mean=latent_dist.mean.reshape(*batch_dims, *latent_dist.mean.shape[1:]),
            logvar=latent_dist.logvar.reshape(*batch_dims, *latent_dist.logvar.shape[1:]),
        )
    
    def _decoder_foward(
        self,
        z: torch.Tensor,
        skip_z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""The forward method of the `Decoder` class."""
        decoder = self.model.decoder
        z = decoder.conv_in(z)

        upscale_dtype = next(iter(decoder.up_blocks.parameters())).dtype
        # middle
        z = decoder.mid_block(z)
        z = z.to(upscale_dtype)

        # up
        for i, up_block in enumerate(decoder.up_blocks):
            if self.cfg.skip_connections:
                # Apply skip conv layer
                z = z + self.skip_convs[i](interpolate(
                    skip_z, 
                    size=z.shape[-2:], 
                    mode="bilinear", 
                    align_corners=True
                ))
            z = up_block(z)

        # post-process
        z = decoder.conv_norm_out(z)
        z = decoder.conv_act(z)
        z = decoder.conv_out(z)

        return z

    def _vae_decode(
        self, 
        z: torch.Tensor,
        skip_z: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[DecoderOutput, torch.Tensor]:
        z = self.model.post_quant_conv(z)
        dec = self._decoder_foward(z, skip_z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def vae_decode(
        self, 
        z: torch.Tensor,
        skip_z: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[DecoderOutput, torch.Tensor]:
        """
        Decode a batch of images.

        Args:
            z (`torch.FloatTensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.

        """
        decoded = self._vae_decode(z, skip_z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def decode(
        self, 
        z: Float[Tensor, "*#batch d_latent latent_height latent_width"],
        skip_z: Optional[Float[Tensor, "*#batch d_skip height width"]] = None,
    ) -> Float[Tensor, "*#batch d_img height width"]:
        batch_dims = z.shape[:-3]
        z = z.flatten(0, -4)
        if skip_z is not None:
            skip_z = skip_z.flatten(0, -4)
        sample = self.vae_decode(z, skip_z).sample
        sample = (sample + 1) / 2
        sample = sample.reshape(*batch_dims, *sample.shape[1:])
        return sample

    @property
    def downscale_factor(self) -> int:
        return 2 ** (len(self.cfg.block_out_channels)-1)

    @property
    def d_latent(self) -> int:
        return self.cfg.latent_channels
    
    @property
    def last_layer_weights(self) -> Tensor:
        return self.model.decoder.conv_out.weight

    @property
    def expects_skip(self) -> bool:
        return self.cfg.skip_connections
    
    @property
    def expects_skip_extra(self) -> bool:
        return self.cfg.skip_extra
