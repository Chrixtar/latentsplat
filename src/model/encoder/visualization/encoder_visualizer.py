from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

from jaxtyping import Float
from torch import Tensor

T_cfg = TypeVar("T_cfg")
T_encoder = TypeVar("T_encoder")


class EncoderVisualizer(ABC, Generic[T_cfg, T_encoder]):
    cfg: T_cfg
    encoder: T_encoder

    def __init__(self, cfg: T_cfg, encoder: T_encoder) -> None:
        self.cfg = cfg
        self.encoder = encoder

    @abstractmethod
    def visualize(
        self,
        context: dict,
        global_step: int,
        features: Optional[Float[Tensor, "bv c h w"]] = None
    ) -> dict[str, Float[Tensor, "3 _ _"]]:
        pass
