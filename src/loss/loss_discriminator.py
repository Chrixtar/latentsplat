from dataclasses import dataclass
from typing import Literal, Tuple

from jaxtyping import Float
import torch
from torch import Tensor
import torch.nn.functional as F

from ..model.types import Prediction, GroundTruth
from .loss import Loss, LossCfg, LossOutput, LossValue


@dataclass
class LossDiscriminatorCfg(LossCfg):
    name: Literal["discriminator"] = "discriminator"
    loss: Literal["hinge", "vanilla"] = "hinge"


class LossDiscriminator(Loss):
    def __init__(
        self, 
        cfg: LossDiscriminatorCfg
    ) -> None:
        super().__init__(cfg)
        self.loss = getattr(LossDiscriminator, f"{cfg.loss}_loss")
    
    @staticmethod
    def hinge_loss(
        logits: Tensor
    ) -> Float[Tensor, ""]:
        return F.relu(1 + logits).mean()

    @staticmethod
    def vanilla_loss(
        logits: Tensor
    ) -> Float[Tensor, ""]:
        return F.softplus(logits).mean()

    def unweighted_loss(
        self,
        prediction: Prediction,
        gt: GroundTruth,
    ) -> Tuple[
        Float[Tensor, ""],
        Float[Tensor, ""]
    ]:
        loss_fake = self.loss(prediction.logits_fake)
        loss_real = self.loss(-prediction.logits_real)  # NOTE negative
        return loss_fake, loss_real
    
    def forward(
        self, 
        prediction: Prediction, 
        gt: GroundTruth,
        global_step: int
    ) -> LossOutput:
        # Before the specified step, don't apply the loss.
        if global_step < self.cfg.apply_after_step:
            loss_fake, loss_real = torch.tensor(0, dtype=torch.float32, device=prediction.image.device)
        else:
            loss_fake, loss_real = self.unweighted_loss(prediction, gt)
        return dict(
            fake=LossValue(loss_fake, self.cfg.weight/2 * loss_fake),
            real=LossValue(loss_real, self.cfg.weight/2 * loss_real)
        )
