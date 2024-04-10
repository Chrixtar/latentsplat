# from pytorch3d.renderer.cameras import PerspectiveCameras
from typing import Tuple

import torch
from torch import Tensor
# from typing_extensions import Tuple
from dataclasses import dataclass


@dataclass
class PerspectiveCameras:
    R: Tensor
    T: Tensor
    focal_length: Tensor
    principal_point: Tensor

# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/implicitron/dataset/frame_data.py#L708
def _get_pytorch3d_camera(
    entry,
) -> PerspectiveCameras:
    entry_viewpoint = entry.viewpoint
    assert entry_viewpoint is not None
    # principal point and focal length
    principal_point = torch.tensor(entry_viewpoint.principal_point, dtype=torch.float)
    focal_length = torch.tensor(entry_viewpoint.focal_length, dtype=torch.float)

    format = entry_viewpoint.intrinsics_format
    if entry_viewpoint.intrinsics_format == "ndc_norm_image_bounds":
        # legacy PyTorch3D NDC format
        # convert to pixels unequally and convert to ndc equally
        image_size_as_list = list(reversed(entry.image.size))
        image_size_wh = torch.tensor(image_size_as_list, dtype=torch.float)
        per_axis_scale = image_size_wh / image_size_wh.min()
        focal_length = focal_length * per_axis_scale
        principal_point = principal_point * per_axis_scale
    elif entry_viewpoint.intrinsics_format != "ndc_isotropic":
        raise ValueError(f"Unknown intrinsics format: {format}")

    return PerspectiveCameras(
        focal_length=focal_length[None],
        principal_point=principal_point[None],
        R=torch.tensor(entry_viewpoint.R, dtype=torch.float)[None],
        T=torch.tensor(entry_viewpoint.T, dtype=torch.float)[None],
    )

def _opencv_from_cameras_projection(
    cameras: PerspectiveCameras,
    image_size: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    R_pytorch3d = cameras.R.clone()  # pyre-ignore
    T_pytorch3d = cameras.T.clone()  # pyre-ignore
    focal_pytorch3d = cameras.focal_length
    p0_pytorch3d = cameras.principal_point
    T_pytorch3d[:, :2] *= -1
    R_pytorch3d[:, :, :2] *= -1
    tvec = T_pytorch3d
    R = R_pytorch3d.permute(0, 2, 1)

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size.to(R).flip(dims=(1,))

    # NDC to screen conversion.
    scale = image_size_wh.to(R).min(dim=1, keepdim=True)[0] / 2.0
    scale = scale.expand(-1, 2)
    c0 = image_size_wh / 2.0

    principal_point = -p0_pytorch3d * scale + c0
    focal_length = focal_pytorch3d * scale

    camera_matrix = torch.zeros_like(R)
    camera_matrix[:, :2, 2] = principal_point
    camera_matrix[:, 2, 2] = 1.0
    camera_matrix[:, 0, 0] = focal_length[:, 0]
    camera_matrix[:, 1, 1] = focal_length[:, 1]
    return R, tvec, camera_matrix
