from fractions import Fraction
from typing import Optional

from .encoder import Encoder
from .encoder_epipolar import EncoderEpipolar, EncoderEpipolarCfg
from .visualization.encoder_visualizer import EncoderVisualizer
from .visualization.encoder_visualizer_epipolar import EncoderVisualizerEpipolar

ENCODERS = {
    "epipolar": (EncoderEpipolar, EncoderVisualizerEpipolar),
}

EncoderCfg = EncoderEpipolarCfg


def get_encoder(
    cfg: EncoderCfg, 
    d_in: int, 
    n_feature_channels: int,
    scale_factor: Fraction,
    variational: bool = False
) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(
        cfg, 
        d_in, 
        n_feature_channels, 
        scale_factor,
        variational
    )
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)        # TODO do we need the variational flag here?
    return encoder, visualizer
