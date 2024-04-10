from dataclasses import dataclass
from typing import Literal

from jaxtyping import Float

from torch import Tensor

from ..model.types import Prediction, GroundTruth
from .loss import Loss, LossCfg


@dataclass
class LossGeneratorCfg(LossCfg):
    name: Literal["generator"] = "generator"


class LossGenerator(Loss):
    """Generator Loss"""
    def __init__(
        self, 
        cfg: LossGeneratorCfg
    ) -> None:
        super().__init__(cfg)

    def unweighted_loss(
        self,
        prediction: Prediction,
        gt: GroundTruth | None = None,
    ) -> Float[Tensor, ""]:
        return -prediction.logits_fake.mean()
