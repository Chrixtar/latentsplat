from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor


class DiagonalGaussianDistribution:
    """Custom implementation to allow any shapes"""

    mean: Tensor
    _logvar: Tensor | None
    _std: Tensor | float
    _var: Tensor | float
    logvar_interval: Tuple[float, float]

    def __init__(
        self,
        mean: Tensor | None = None,
        logvar: Tensor | None = None,
        params: Tensor | None = None,
        dim: int = 0,
        logvar_interval: Tuple[float, float] = (-30.0, 20.0)
    ): 
        assert params is None or (mean is None and logvar is None), "If params are given, mean and logvar are not expected"
        assert mean is not None or params is not None, "Either mean or params must be given"
        self.mean = mean
        self.logvar_interval = logvar_interval
        self.logvar = logvar
        self.dim = dim
        self.params = params

    @property
    def params(self) -> Tensor:
        if self._params is None:
            assert self.logvar is not None, "Trying accessing params without params or logvar"
            return torch.cat((self.mean, self.logvar), dim=self.dim)
        return self._params
        
    @params.setter
    def params(self, val: Tensor | None) -> None:
        if val is not None:
            self.mean, self.logvar = val.chunk(2, dim=self.dim)
        self._params = val

    @property
    def logvar(self) -> Tensor | None:
        return self._logvar
    
    @property
    def std(self) -> Tensor | float:
        return self._std
    
    @property
    def var(self) -> Tensor | float:
        return self._var

    @logvar.setter
    def logvar(self, val: Tensor | None) -> None:
        self._logvar = val
        if self._logvar is not None:
            assert self._logvar.shape == self.mean.shape, "Shapes of mean and logvar must be identical"
            self._logvar = torch.clamp(self._logvar, *self.logvar_interval)
            self._std = torch.exp(0.5 * self._logvar)
            self._var = torch.exp(self._logvar)
        else:
            # Default is zero variance
            self._std = 0.
            self._var = 0.

    @property
    def device(self) -> torch.device:
        return self.mean.device

    def sample(self) -> Tensor:
        if isinstance(self.std, float) and self.std == 0:
            return self.mean
        sample = torch.randn_like(self.mean)
        x = self.mean + self.std * sample
        return x

    def kl(self, other: Optional['DiagonalGaussianDistribution'] = None) -> Tensor:
        if self.logvar is None:
            return torch.zeros_like(self.mean)
        return 0.5 * (self.mean ** 2 + self.var - 1.0 - self.logvar if other is None else \
            (self.mean - other.mean) ** 2 / other.var + self.var / other.var - 1.0 - self.logvar + other.logvar)

    def nll(self, sample: Tensor) -> Tensor:
        if self.logvar is None:
            return torch.zeros_like(self.mean, device=self.device)
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * (logtwopi + self.logvar + (sample - self.mean) ** 2 / self.var)

    def mode(self) -> Tensor:
        return self.mean
