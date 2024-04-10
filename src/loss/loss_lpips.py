from dataclasses import dataclass
from typing import Literal, Optional

from einops import rearrange
from jaxtyping import Float
from lpips import LPIPS
from torch import nn, Tensor

from ..misc.nn_module_tools import convert_to_buffer
from ..model.types import Prediction, GroundTruth
from .loss import LossCfg, Loss


@dataclass
class LossLpipsCfg(LossCfg):
    name: Literal["lpips"] = "lpips"


class LpipsWrapper(nn.Module):
    lpips: LPIPS

    def __init__(self) -> None:
        super(LpipsWrapper, self).__init__()
        self.lpips = LPIPS(net="vgg")
        convert_to_buffer(self.lpips, persistent=False)
    
    def forward(
        self,
        pred: Float[Tensor, "b c h w"], 
        target: Float[Tensor, "b c h w"]
    ) -> Float[Tensor, ""]:
        return self.lpips.forward(
            pred,
            target,
            normalize=True
        ).mean()
    

class LossLpips(Loss):
    lpips: LpipsWrapper

    def __init__(
        self, 
        cfg: LossLpipsCfg, 
        lpips: Optional[LpipsWrapper] = None
    ) -> None:
        super().__init__(cfg)
        if lpips is None:
            lpips = LpipsWrapper()
        self.lpips = lpips

    def unweighted_loss(
        self,
        prediction: Prediction,
        gt: GroundTruth
    ) -> Float[Tensor, ""]:
        return self.lpips(
            rearrange(prediction.image, "b v c h w -> (b v) c h w"),
            rearrange(gt.image, "b v c h w -> (b v) c h w")
        )
