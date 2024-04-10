from .discriminator import Discriminator
from .discriminator_patch_gan import DiscriminatorPatchGan, DiscriminatorPatchGanCfg

DISCRIMINATORS = {
    "patch_gan": DiscriminatorPatchGan,
}

DiscriminatorCfg = DiscriminatorPatchGanCfg


def get_discriminator(
    discriminator_cfg: DiscriminatorCfg,
    d_in: int = 3
) -> Discriminator:
    return DISCRIMINATORS[discriminator_cfg.name](discriminator_cfg, d_in)
