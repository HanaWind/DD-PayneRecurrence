from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LabelScaler:
    mean: np.ndarray
    scale: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray

    @classmethod
    def fit(cls, labels: np.ndarray) -> LabelScaler:
        values = np.asarray(labels, dtype=np.float64)
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            raise ValueError("Label array must be finite and two-dimensional")
        mean = np.mean(values, axis=0)
        scale = np.std(values, axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        return cls(
            mean=mean,
            scale=scale,
            minimum=np.min(values, axis=0),
            maximum=np.max(values, axis=0),
        )

    def transform(self, labels: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, device=labels.device, dtype=labels.dtype)
        scale = torch.as_tensor(self.scale, device=labels.device, dtype=labels.dtype)
        return (labels - mean) / scale

    def inverse_transform(self, labels: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, device=labels.device, dtype=labels.dtype)
        scale = torch.as_tensor(self.scale, device=labels.device, dtype=labels.dtype)
        return labels * scale + mean

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "mean": self.mean,
            "scale": self.scale,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> LabelScaler:
        return cls(**{key: np.asarray(value) for key, value in state.items()})


class PixelwisePayne(nn.Module):
    """Two-hidden-layer MLP with independent parameters for every output pixel."""

    def __init__(
        self,
        n_labels: int,
        n_pixels: int,
        hidden_size: int = 40,
        activation_slope: float = 0.01,
    ) -> None:
        super().__init__()
        if min(n_labels, n_pixels, hidden_size) <= 0:
            raise ValueError("Model dimensions must be positive")
        self.n_labels = int(n_labels)
        self.n_pixels = int(n_pixels)
        self.hidden_size = int(hidden_size)
        self.activation_slope = float(activation_slope)

        self.weight1 = nn.Parameter(torch.empty(n_pixels, n_labels, hidden_size))
        self.bias1 = nn.Parameter(torch.zeros(n_pixels, hidden_size))
        self.weight2 = nn.Parameter(torch.empty(n_pixels, hidden_size, hidden_size))
        self.bias2 = nn.Parameter(torch.zeros(n_pixels, hidden_size))
        self.weight3 = nn.Parameter(torch.empty(n_pixels, hidden_size))
        self.bias3 = nn.Parameter(torch.ones(n_pixels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight1, std=self.n_labels**-0.5)
        nn.init.normal_(self.weight2, std=self.hidden_size**-0.5)
        nn.init.normal_(self.weight3, std=self.hidden_size**-0.5)
        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)
        nn.init.ones_(self.bias3)

    def forward(self, labels: torch.Tensor, pixel_slice: slice | None = None) -> torch.Tensor:
        if labels.ndim != 2 or labels.shape[1] != self.n_labels:
            raise ValueError(f"Expected labels [batch, {self.n_labels}], got {tuple(labels.shape)}")
        selection = pixel_slice if pixel_slice is not None else slice(None)
        hidden1 = (
            torch.einsum("bl,plh->bph", labels, self.weight1[selection]) + self.bias1[selection]
        )
        hidden1 = F.leaky_relu(hidden1, negative_slope=self.activation_slope)
        hidden2 = (
            torch.einsum("bph,phk->bpk", hidden1, self.weight2[selection]) + self.bias2[selection]
        )
        hidden2 = F.leaky_relu(hidden2, negative_slope=self.activation_slope)
        return torch.einsum("bph,ph->bp", hidden2, self.weight3[selection]) + self.bias3[selection]

    def config(self) -> dict[str, int | float]:
        return {
            "n_labels": self.n_labels,
            "n_pixels": self.n_pixels,
            "hidden_size": self.hidden_size,
            "activation_slope": self.activation_slope,
        }

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
