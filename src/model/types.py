from dataclasses import dataclass, fields
from typing import Literal

from .diagonal_gaussian_distribution import DiagonalGaussianDistribution
from jaxtyping import Float
from torch import Tensor


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    opacities: Float[Tensor, "batch gaussian"]
    color_harmonics: Float[Tensor, "batch gaussian 3 d_color_sh"] | None = None
    feature_harmonics: Float[Tensor, "batch gaussian channels d_feature_sh"] | None = None


@dataclass
class VariationalGaussians(Gaussians):
    feature_harmonics: DiagonalGaussianDistribution | None = None

    def _to_gaussians(self, feature_harmonics: Float[Tensor, "batch gaussian channels d_feature_sh"]) -> Gaussians:
        return Gaussians(self.means, self.covariances, self.opacities, self.color_harmonics, feature_harmonics)

    def flatten(self) -> Gaussians:
        return self._to_gaussians(self.feature_harmonics.params)

    def mode(self) -> Gaussians:
        return self._to_gaussians(self.feature_harmonics.mode())

    def sample(self) -> Gaussians:
        return self._to_gaussians(self.feature_harmonics.sample())


@dataclass
class Prediction:
    image: Float[Tensor, "batch view channels height width"] | None = None
    posterior: DiagonalGaussianDistribution | None = None
    depth: Float[Tensor, "batch view height width"] | None = None
    logits_fake: Float[Tensor, "batch view 1 h w"] | None = None
    logits_real: Float[Tensor, "batch view 1 h w"] | None = None

    # NOTE assumes all fields to be on the same device
    @property
    def device(self):
        for field in fields(self):
            val = getattr(self, field.name)
            if val is not None:
                return val.device


@dataclass
class GroundTruth:
    image: Float[Tensor, "batch view channels height width"] | None = None
    near: Float[Tensor, "batch view"] | None = None
    far: Float[Tensor, "batch view"] | None = None


VariationalMode = Literal["none", "gaussians", "latents"]