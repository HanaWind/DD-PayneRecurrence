from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ddpayne.models.payne import LabelScaler, PixelwisePayne


def weighted_reconstruction_sum(
    prediction: torch.Tensor,
    target: torch.Tensor,
    ivar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape or target.shape != ivar.shape:
        raise ValueError("Prediction, target, and ivar must have identical shapes")
    valid = torch.isfinite(target) & torch.isfinite(ivar) & (ivar > 0)
    residual = torch.where(valid, prediction - target, 0.0)
    weights = torch.where(valid, ivar, 0.0)
    numerator = torch.sum(residual.square() * weights)
    count = torch.sum(valid)
    return numerator, count


@dataclass(frozen=True)
class GradientLibrary:
    labels: np.ndarray
    gradients: np.ndarray
    steps: np.ndarray
    wavelength: np.ndarray
    label_names: tuple[str, ...]

    @classmethod
    def load(
        cls,
        path: str | Path,
        expected_label_names: Sequence[str],
        expected_wavelength: np.ndarray,
    ) -> GradientLibrary:
        with np.load(path, allow_pickle=False) as archive:
            required = {"labels", "gradients", "steps", "wavelength", "label_names"}
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"Gradient library is missing arrays: {sorted(missing)}")
            names = tuple(str(value) for value in archive["label_names"].tolist())
            library = cls(
                labels=np.asarray(archive["labels"], dtype=np.float32),
                gradients=np.asarray(archive["gradients"], dtype=np.float32),
                steps=np.asarray(archive["steps"], dtype=np.float32),
                wavelength=np.asarray(archive["wavelength"], dtype=np.float64),
                label_names=names,
            )
        library.validate(expected_label_names, expected_wavelength)
        return library

    def validate(
        self, expected_label_names: Sequence[str], expected_wavelength: np.ndarray
    ) -> None:
        references, labels = self.labels.shape
        pixels = len(self.wavelength)
        if self.gradients.shape != (references, labels, pixels):
            raise ValueError("Gradient array shape is inconsistent with labels/wavelength")
        if self.steps.shape != (labels,) or np.any(self.steps <= 0):
            raise ValueError("Gradient steps must be positive with shape [labels]")
        if tuple(expected_label_names) != self.label_names:
            raise ValueError("Gradient-library label order differs from training dataset")
        expected = np.asarray(expected_wavelength)
        if expected.shape != self.wavelength.shape or not np.allclose(
            expected, self.wavelength, rtol=0.0, atol=1e-6
        ):
            raise ValueError("Gradient-library wavelength grid differs from training dataset")

    def sample_indices(self, count: int, rng: np.random.Generator) -> np.ndarray:
        count = min(int(count), len(self.labels))
        return rng.choice(len(self.labels), size=count, replace=False)


def finite_difference_gradient_sum(
    model: PixelwisePayne,
    scaler: LabelScaler,
    reference_labels: torch.Tensor,
    target_gradients: torch.Tensor,
    steps: torch.Tensor,
    label_weights: torch.Tensor,
    pixel_slice: slice,
) -> torch.Tensor:
    """Return weighted L1 gradient mismatch summed over refs, labels, and pixels."""
    if target_gradients.ndim != 3:
        raise ValueError("Target gradients must have shape [references, labels, pixels]")
    total = reference_labels.new_zeros(())
    for label_index in range(reference_labels.shape[1]):
        half_step = steps[label_index] * 0.5
        plus = reference_labels.clone()
        minus = reference_labels.clone()
        plus[:, label_index] += half_step
        minus[:, label_index] -= half_step
        predicted_plus = model(scaler.transform(plus), pixel_slice=pixel_slice)
        predicted_minus = model(scaler.transform(minus), pixel_slice=pixel_slice)
        predicted_gradient = (predicted_plus - predicted_minus) / steps[label_index]
        target = target_gradients[:, label_index, pixel_slice]
        total = total + label_weights[label_index] * torch.sum(
            torch.abs(predicted_gradient - target)
        )
    return total
