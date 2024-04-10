from dataclasses import dataclass
from typing import Literal

from jaxtyping import Float
from torch import Tensor

from ..model.types import Prediction, GroundTruth
from .loss import LossCfg, Loss


@dataclass
class LossKlCfg(LossCfg):
    name: Literal["kl"] = "kl"


class LossKl(Loss):
    def unweighted_loss(
        self,
        prediction: Prediction,
        gt: GroundTruth | None = None
    ) -> Float[Tensor, ""]:
        return prediction.posterior.kl().mean()
