from .autoencoder import Autoencoder
from .autoencoder_id import AutoencoderId, AutoencoderIdCfg
from .autoencoder_kl import AutoencoderKL, AutoencoderKLCfg


AUTOENCODERS = {
    "id": AutoencoderId,
    "kl": AutoencoderKL
}

AutoencoderCfg = AutoencoderIdCfg | AutoencoderKLCfg


def get_autoencoder(
    cfg: AutoencoderCfg, 
    d_in: int = 3,
    d_skip_extra: int = 3,
    sample_size: int = 32
) -> Autoencoder:
    return AUTOENCODERS[cfg.name](cfg, d_in, d_skip_extra, sample_size)
