from dataclasses import dataclass
from typing import Literal

from jaxtyping import Float
from torch import Tensor

from ..model.types import Prediction, GroundTruth
from .loss import LossCfg, Loss


@dataclass
class LossL1Cfg(LossCfg):
    name: Literal["l1"] = "l1"


class LossL1(Loss):
    def unweighted_loss(
        self,
        prediction: Prediction,
        gt: GroundTruth
    ) -> Float[Tensor, ""]:
        delta = prediction.image - gt.image
        return delta.abs().mean()
