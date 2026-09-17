from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from ddpayne.models.payne import LabelScaler, PixelwisePayne
from ddpayne.training.train import train_from_config


def run_smoke_test(work_dir: str | Path) -> dict[str, object]:
    output = Path(work_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    n_objects, n_labels, n_pixels = 96, 3, 24
    label_names = ["teff", "logg", "fe_h"]
    labels = np.column_stack(
        [
            rng.uniform(4200, 6800, n_objects),
            rng.uniform(1.0, 4.8, n_objects),
            rng.uniform(-2.0, 0.4, n_objects),
        ]
    ).astype(np.float32)
    scaler = LabelScaler.fit(labels)
    torch.manual_seed(11)
    truth = PixelwisePayne(n_labels, n_pixels, hidden_size=6)
    with torch.no_grad():
        flux = truth(scaler.transform(torch.from_numpy(labels))).numpy()
    noise_sigma = 0.01
    flux = (flux + rng.normal(0, noise_sigma, flux.shape)).astype(np.float32)
    ivar = np.full_like(flux, 1.0 / noise_sigma**2)
    wavelength = np.linspace(4000, 5000, n_pixels)
    split = np.concatenate(
        [np.zeros(70, dtype=np.uint8), np.ones(16, dtype=np.uint8), np.full(10, 2, dtype=np.uint8)]
    )

    dataset_path = output / "synthetic.h5"
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(dataset_path, "w") as handle:
        handle.create_dataset("wavelength", data=wavelength)
        handle.create_dataset("flux", data=flux)
        handle.create_dataset("ivar", data=ivar)
        handle.create_dataset("labels", data=labels)
        handle.create_dataset("split", data=split)
        identifiers = np.asarray([f"synthetic-{index}" for index in range(n_objects)], dtype=object)
        handle.create_dataset("source_id", data=identifiers, dtype=string_dtype)
        handle.create_dataset("obsid", data=identifiers, dtype=string_dtype)
        handle.create_dataset("path", data=identifiers, dtype=string_dtype)
        handle.attrs["label_names"] = json.dumps(label_names)
        handle.attrs["config"] = "{}"

    reference_labels = labels[:6].copy()
    steps = np.asarray([50.0, 0.05, 0.05], dtype=np.float32)
    gradients = np.empty((len(reference_labels), n_labels, n_pixels), dtype=np.float32)
    with torch.no_grad():
        for label_index in range(n_labels):
            plus = reference_labels.copy()
            minus = reference_labels.copy()
            plus[:, label_index] += steps[label_index] * 0.5
            minus[:, label_index] -= steps[label_index] * 0.5
            plus_flux = truth(scaler.transform(torch.from_numpy(plus))).numpy()
            minus_flux = truth(scaler.transform(torch.from_numpy(minus))).numpy()
            gradients[:, label_index] = (plus_flux - minus_flux) / steps[label_index]
    gradient_path = output / "gradients.npz"
    np.savez(
        gradient_path,
        labels=reference_labels,
        gradients=gradients,
        steps=steps,
        wavelength=wavelength,
        label_names=np.asarray(label_names),
    )

    run_dir = output / "run"
    config = {
        "dataset": str(dataset_path),
        "run_dir": str(run_dir),
        "device": "cpu",
        "seed": 19,
        "model": {"hidden_size": 6, "activation_slope": 0.01, "pixel_chunk_size": 12},
        "optimization": {
            "batch_size": 16,
            "max_steps": 20,
            "learning_rate_start": 0.01,
            "learning_rate_end": 0.001,
            "weight_decay": 0.0,
            "gradient_clip_norm": 10.0,
            "mixed_precision": False,
            "num_workers": 0,
            "validation_every": 5,
            "checkpoint_every": 10,
        },
        "gradient_regularization": {
            "path": str(gradient_path),
            "required": True,
            "apply_every_steps": 2,
            "reference_batch_size": 3,
            "weights": {"default": 0.1},
        },
    }
    config_path = output / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    result = train_from_config(config_path)
    result["dataset"] = str(dataset_path)
    result["gradient_library"] = str(gradient_path)
    return result
