"""Where and in what precision a family computes: one device, one dtype, one seed per fit.

``device='auto'`` is CUDA when present (native Windows and Linux alike), else CPU. Physics fits checked
against an adjoint or an FEA golden run ``float64``; data-driven fits may run ``float32``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from learn_lab.errors import refuse

__all__ = ['Runtime']

_DTYPES = {'float32': torch.float32, 'float64': torch.float64}


@dataclass(frozen=True, slots=True)
class Runtime:
    device: str = 'auto'
    dtype: Literal['float32', 'float64'] = 'float64'
    seed: int = 0

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPES:
            refuse(f'a runtime dtype is one of {list(_DTYPES)}, got {self.dtype!r}')
        if self.device not in {'auto', 'cpu'} and not self.device.startswith('cuda'):
            refuse(f"a runtime device is 'auto', 'cpu' or 'cuda[:n]', got {self.device!r}")
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            refuse(f'{self.device} was asked for but this torch sees no CUDA device')

    @property
    def torch_device(self) -> torch.device:
        if self.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.device)

    @property
    def torch_dtype(self) -> torch.dtype:
        return _DTYPES[self.dtype]

    def generator(self) -> torch.Generator:
        """A generator seeded for this fit on its device; the fit's only randomness."""
        return torch.Generator(device=self.torch_device).manual_seed(self.seed)

    def tensor(self, array: np.ndarray, *, grad: bool = False) -> torch.Tensor:
        out = torch.as_tensor(np.asarray(array, dtype=float), dtype=self.torch_dtype, device=self.torch_device)
        return out.requires_grad_(grad)

    def seeded(self) -> None:
        """Seed parameter initialisation (module constructors draw from the global generator)."""
        torch.manual_seed(self.seed)


def numpy(tensor: torch.Tensor) -> np.ndarray:
    """A detached float64 numpy copy -- the only way a tensor leaves a family."""
    return tensor.detach().to('cpu', torch.float64).numpy()
