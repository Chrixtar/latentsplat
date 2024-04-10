from dataclasses import dataclass

# This is in a separate file to avoid circular imports.


@dataclass
class EncoderVisualizerEpipolarCfg:
    num_samples: int
    min_resolution: int
    export_ply: bool
    vis_epipolar_samples: bool = True
    vis_epipolar_color_samples: bool = True
    vis_gaussians: bool = True
    vis_overlaps: bool = True
    vis_depth: bool = True
