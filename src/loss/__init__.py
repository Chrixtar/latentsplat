from dataclasses import dataclass

from .loss_group import LossGroup

from .loss_discriminator import LossDiscriminator, LossDiscriminatorCfg
from .loss_depth import LossDepth, LossDepthCfg
from .loss_generator import LossGenerator, LossGeneratorCfg
from .loss_kl import LossKl, LossKlCfg
from .loss_l1 import LossL1, LossL1Cfg
from .loss_lpips import LossLpips, LossLpipsCfg
from .loss_mse import LossMse, LossMseCfg


LOSSES = {
    "depth": LossDepth,
    "kl": LossKl,
    "l1": LossL1,
    "lpips": LossLpips,
    "mse": LossMse,
}


NLLLossCfg = LossDepthCfg | LossKlCfg | LossL1Cfg | LossLpipsCfg | LossMseCfg


@dataclass
class LossGroupCfg:
    nll: list[NLLLossCfg] | None = None
    generator: LossGeneratorCfg | None = None
    discriminator: LossDiscriminatorCfg | None = None

def get_loss_group(
    name: str, 
    group_cfg: LossGroupCfg | None = None
) -> LossGroup:
    if group_cfg is None:
        return LossGroup(name)
    nll_losses = []
    if group_cfg.nll is not None:
        first_lpips = None
        for cfg in group_cfg.nll:
            # make sure to initialize LPIPS feature extractor only once
            if cfg.name == "lpips":
                loss = LossLpips(cfg, lpips=first_lpips)
                if first_lpips is None:
                    first_lpips = loss.lpips
            else:
                loss = LOSSES[cfg.name](cfg)
            nll_losses.append(loss)
    return LossGroup(
        name, 
        nll_losses,
        generator_loss=LossGenerator(group_cfg.generator) if group_cfg.generator is not None else None,
        discriminator_loss=LossDiscriminator(group_cfg.discriminator) if group_cfg.discriminator is not None else None
    )
