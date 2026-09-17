from __future__ import annotations

import numpy as np
import torch

from ddpayne.inference.fit import fit_label_batch
from ddpayne.models.losses import finite_difference_gradient_sum
from ddpayne.models.payne import LabelScaler, PixelwisePayne


def test_pixelwise_payne_shape_and_gradient():
    model = PixelwisePayne(n_labels=3, n_pixels=7, hidden_size=5)
    labels = torch.randn(4, 3, requires_grad=True)
    output = model(labels)
    assert output.shape == (4, 7)
    output.sum().backward()
    assert labels.grad is not None
    assert torch.isfinite(labels.grad).all()


def test_physical_gradient_loss_is_zero_for_matching_model():
    torch.manual_seed(3)
    model = PixelwisePayne(n_labels=2, n_pixels=5, hidden_size=4)
    physical = np.asarray([[5000.0, 4.0], [6000.0, 3.0]], dtype=np.float32)
    scaler = LabelScaler.fit(np.asarray([[4000, 1], [5000, 3], [6500, 5]], dtype=float))
    steps = torch.tensor([100.0, 0.1])
    references = torch.from_numpy(physical)
    target = torch.empty(2, 2, 5)
    with torch.no_grad():
        for index in range(2):
            plus = references.clone()
            minus = references.clone()
            plus[:, index] += steps[index] * 0.5
            minus[:, index] -= steps[index] * 0.5
            target[:, index] = (
                model(scaler.transform(plus)) - model(scaler.transform(minus))
            ) / steps[index]
    loss = finite_difference_gradient_sum(
        model,
        scaler,
        references,
        target,
        steps,
        torch.ones(2),
        slice(0, 5),
    )
    assert float(loss.detach()) < 1e-6


def test_label_fit_and_fisher_uncertainty_are_finite():
    torch.manual_seed(5)
    model = PixelwisePayne(n_labels=2, n_pixels=6, hidden_size=4)
    scaler = LabelScaler.fit(np.asarray([[-1.0, -2.0], [0.0, 0.0], [1.0, 2.0]]))
    with torch.no_grad():
        flux = model(torch.zeros(1, 2))
    result = fit_label_batch(
        model=model,
        scaler=scaler,
        flux=flux,
        ivar=torch.full_like(flux, 100.0),
        steps=2,
        learning_rate=0.03,
        pixel_chunk_size=3,
        estimate_uncertainty=True,
    )
    assert result["labels"].shape == (1, 2)
    assert np.allclose(result["labels"], 0.0, atol=1e-6)
    assert np.all(np.isfinite(result["uncertainties"]))
    assert np.allclose(result["reduced_chi2"], 0.0, atol=1e-7)
