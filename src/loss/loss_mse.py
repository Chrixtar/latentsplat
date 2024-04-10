from dataclasses import dataclass
from typing import Literal

from jaxtyping import Float
from torch import Tensor

from ..model.types import Prediction, GroundTruth
from .loss import LossCfg, Loss


@dataclass
class LossMseCfg(LossCfg):
    name: Literal["mse"] = "mse"


class LossMse(Loss):
    def unweighted_loss(
        self,
        prediction: Prediction,
        gt: GroundTruth,
    ) -> Float[Tensor, ""]:
        delta = prediction.image - gt.image
        return (delta**2).mean()
