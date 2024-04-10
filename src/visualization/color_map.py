import torch
from colorspacious import cspace_convert
from einops import rearrange
from jaxtyping import Float
from matplotlib import cm
from torch import Tensor


def apply_color_map(
    x: Float[Tensor, " *batch"],
    color_map: str = "inferno",
) -> Float[Tensor, "*batch 3"]:
    cmap = cm.get_cmap(color_map)

    # Convert to NumPy so that Matplotlib color maps can be used.
    mapped = cmap(x.detach().clip(min=0, max=1).cpu().numpy())[..., :3]

    # Convert back to the original format.
    return torch.tensor(mapped, device=x.device, dtype=torch.float32)


def apply_color_map_to_image(
    image: Float[Tensor, "*batch height width"],
    color_map: str = "inferno",
) -> Float[Tensor, "*batch 3 height with"]:
    image = apply_color_map(image, color_map)
    return rearrange(image, "... h w c -> ... c h w")


def apply_color_map_2d(
    x: Float[Tensor, "*#batch"],
    y: Float[Tensor, "*#batch"],
) -> Float[Tensor, "*batch 3"]:
    red = cspace_convert((189, 0, 0), "sRGB255", "CIELab")
    blue = cspace_convert((0, 45, 255), "sRGB255", "CIELab")
    white = cspace_convert((255, 255, 255), "sRGB255", "CIELab")
    x_np = x.detach().clip(min=0, max=1).cpu().numpy()[..., None]
    y_np = y.detach().clip(min=0, max=1).cpu().numpy()[..., None]

    # Interpolate between red and blue on the x axis.
    interpolated = x_np * red + (1 - x_np) * blue

    # Interpolate between color and white on the y axis.
    interpolated = y_np * interpolated + (1 - y_np) * white

    # Convert to RGB.
    rgb = cspace_convert(interpolated, "CIELab", "sRGB1")
    return torch.tensor(rgb, device=x.device, dtype=torch.float32).clip(min=0, max=1)


def apply_depth_color_map(
    depth: Float[Tensor, "*batch height width"]
) -> Float[Tensor, "*batch 3 height with"]:
    positive = depth[depth > 0]
    if positive.numel() == 0:
        return apply_color_map_to_image(torch.zeros_like(depth), "turbo")
    near = positive[:16_000_000].quantile(0.01).log()
    far = depth.view(-1)[:16_000_000].quantile(0.99).log()
    depth = depth.log()
    depth = 1 - (depth - near) / (far - near)
    return apply_color_map_to_image(depth, "turbo")
