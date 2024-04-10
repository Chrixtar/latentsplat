from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

from jaxtyping import Float
from torch import nn, Tensor

from ...dataset.types import BatchedViews, DataShim
from ..types import VariationalGaussians

T = TypeVar("T")


class Encoder(nn.Module, ABC, Generic[T]):
    cfg: T
    variational: bool

    def __init__(
        self, 
        cfg: T,
        variational: bool
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.variational = variational

    @abstractmethod
    def forward(
        self,
        context: BatchedViews,
        features: Optional[Float[Tensor, "bv d_in h w"]] = None,
        deterministic: bool = False
    ) -> VariationalGaussians:
        pass

    def get_data_shim(self) -> DataShim:
        """The default shim doesn't modify the batch."""
        return lambda x: x

    @property
    @abstractmethod
    def last_layer_weights(self) -> Tensor | None:
        pass