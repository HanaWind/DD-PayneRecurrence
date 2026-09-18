from __future__ import annotations

import json
import math
import time
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from ddpayne.config import load_yaml, project_path
from ddpayne.data.dataset import H5SpectraDataset
from ddpayne.models.losses import (
    GradientLibrary,
    finite_difference_gradient_sum,
    weighted_reconstruction_sum,
)
from ddpayne.models.payne import LabelScaler, PixelwisePayne
from ddpayne.utils import set_random_seed


def _cycle(loader: DataLoader) -> Iterator[dict[str, Any]]:
    while True:
        yield from loader


def _pixel_slices(n_pixels: int, chunk_size: int) -> Iterator[slice]:
    for start in range(0, n_pixels, chunk_size):
        yield slice(start, min(start + chunk_size, n_pixels))


def _device(name: str) -> torch.device:
    requested = torch.device(name)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return requested


def _label_weights(label_names: list[str], config: dict[str, Any]) -> np.ndarray:
    values = config.get("weights", {})
    default = float(values.get("default", 1.0))
    return np.asarray([float(values.get(name, default)) for name in label_names], dtype=np.float32)


def _exponential_learning_rate_factor(
    step: int, learning_rate_start: float, learning_rate_end: float, max_steps: int
) -> float:
    if learning_rate_start <= 0 or learning_rate_end <= 0:
        raise ValueError("Learning rates must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    decay = math.log(learning_rate_end / learning_rate_start) / max_steps
    return math.exp(decay * step)


def _save_checkpoint(
    path: Path,
    model: PixelwisePayne,
    optimizer: torch.optim.Optimizer,
    scaler: LabelScaler,
    label_names: list[str],
    wavelength: np.ndarray,
    config: dict[str, Any],
    step: int,
    best_validation: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "model_config": model.config(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "label_scaler": scaler.state_dict(),
            "label_names": label_names,
            "wavelength": wavelength,
            "config": config,
            "step": step,
            "best_validation": best_validation,
        },
        temporary,
    )
    temporary.replace(path)


@torch.no_grad()
def _validate(
    model: PixelwisePayne,
    scaler: LabelScaler,
    loader: DataLoader,
    device: torch.device,
    pixel_chunk_size: int,
    mixed_precision: bool,
) -> float:
    model.eval()
    numerator_total = 0.0
    count_total = 0
    for batch in loader:
        labels = batch["labels"].to(device)
        flux = batch["flux"].to(device)
        ivar = batch["ivar"].to(device)
        scaled_labels = scaler.transform(labels)
        for selection in _pixel_slices(model.n_pixels, pixel_chunk_size):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=mixed_precision and device.type == "cuda",
            ):
                prediction = model(scaled_labels, pixel_slice=selection)
                numerator, count = weighted_reconstruction_sum(
                    prediction, flux[:, selection], ivar[:, selection]
                )
            numerator_total += float(numerator)
            count_total += int(count)
    model.train()
    return numerator_total / max(count_total, 1)


def train_from_config(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    set_random_seed(int(config.get("seed", 42)))
    device = _device(str(config.get("device", "cuda")))
    dataset_path = project_path(config["dataset"])
    run_dir = project_path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.snapshot.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    train_dataset = H5SpectraDataset(dataset_path, split="train")
    validation_dataset = H5SpectraDataset(dataset_path, split="validation")
    with h5py.File(dataset_path, "r") as handle:
        train_labels = handle["labels"][train_dataset.indices]
    scaler = LabelScaler.fit(train_labels)
    label_names = list(train_dataset.label_names)
    wavelength = train_dataset.wavelength

    model_config = config.get("model", {})
    model = PixelwisePayne(
        n_labels=len(label_names),
        n_pixels=len(wavelength),
        hidden_size=int(model_config.get("hidden_size", 40)),
        activation_slope=float(model_config.get("activation_slope", 0.01)),
    ).to(device)

    optimization = config.get("optimization", {})
    batch_size = int(optimization.get("batch_size", 128))
    workers = int(optimization.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train_dataset) >= batch_size,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    batches = _cycle(train_loader)

    learning_rate_start = float(optimization.get("learning_rate_start", 0.01))
    learning_rate_end = float(optimization.get("learning_rate_end", 0.0001))
    max_steps = int(optimization.get("max_steps", 10_000))
    _exponential_learning_rate_factor(0, learning_rate_start, learning_rate_end, max_steps)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate_start,
        weight_decay=float(optimization.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _exponential_learning_rate_factor(
            step, learning_rate_start, learning_rate_end, max_steps
        ),
    )
    mixed_precision = bool(optimization.get("mixed_precision", True)) and device.type == "cuda"
    amp_scaler = torch.amp.GradScaler(device.type, enabled=mixed_precision)
    pixel_chunk_size = int(model_config.get("pixel_chunk_size", model.n_pixels))

    regularization = config.get("gradient_regularization", {})
    gradient_path = project_path(regularization.get("path", ""))
    gradient_library: GradientLibrary | None = None
    if gradient_path.is_file():
        gradient_library = GradientLibrary.load(gradient_path, label_names, wavelength)
    elif regularization.get("required", True):
        raise FileNotFoundError(
            f"Full DD-Payne training requires a gradient library: {gradient_path}"
        )
    else:
        warnings.warn(
            "No physical gradient library: this run is a data-driven baseline, not full DD-Payne",
            stacklevel=2,
        )
    label_weights = torch.as_tensor(_label_weights(label_names, regularization), device=device)
    gradient_steps = None
    rng = np.random.default_rng(int(config.get("seed", 42)))

    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    best_validation = float("inf")
    validation_every = int(optimization.get("validation_every", 100))
    checkpoint_every = int(optimization.get("checkpoint_every", 500))
    gradient_every = int(regularization.get("apply_every_steps", 1))
    reference_batch_size = int(regularization.get("reference_batch_size", 8))
    start_time = time.perf_counter()

    for step in range(1, max_steps + 1):
        batch = next(batches)
        labels = batch["labels"].to(device, non_blocking=True)
        flux = batch["flux"].to(device, non_blocking=True)
        ivar = batch["ivar"].to(device, non_blocking=True)
        scaled_labels = scaler.transform(labels)
        valid_total = torch.sum(torch.isfinite(ivar) & (ivar > 0)).clamp_min(1)
        optimizer.zero_grad(set_to_none=True)
        reconstruction_value = 0.0
        gradient_value = 0.0

        apply_gradient = gradient_library is not None and step % gradient_every == 0
        reference_labels = None
        target_gradients = None
        if apply_gradient and gradient_library is not None:
            reference_indices = gradient_library.sample_indices(reference_batch_size, rng)
            reference_labels = torch.as_tensor(
                gradient_library.labels[reference_indices], device=device
            )
            target_gradients = torch.as_tensor(
                gradient_library.gradients[reference_indices], device=device
            )
            gradient_steps = torch.as_tensor(gradient_library.steps, device=device)

        for selection in _pixel_slices(model.n_pixels, pixel_chunk_size):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=mixed_precision,
            ):
                prediction = model(scaled_labels, pixel_slice=selection)
                reconstruction_sum, _ = weighted_reconstruction_sum(
                    prediction, flux[:, selection], ivar[:, selection]
                )
                loss = reconstruction_sum / valid_total
                reconstruction_value += float((reconstruction_sum / valid_total).detach())
            if (
                apply_gradient
                and reference_labels is not None
                and target_gradients is not None
                and gradient_steps is not None
            ):
                # Small finite differences are vulnerable to FP16 cancellation.
                # Keep the physical-gradient constraint in FP32 even when the
                # observational reconstruction uses mixed precision.
                with torch.autocast(device_type=device.type, enabled=False):
                    gradient_sum = finite_difference_gradient_sum(
                        model=model,
                        scaler=scaler,
                        reference_labels=reference_labels,
                        target_gradients=target_gradients,
                        steps=gradient_steps,
                        label_weights=label_weights,
                        pixel_slice=selection,
                    )
                    gradient_loss = (
                        gradient_sum * gradient_every / (len(reference_labels) * model.n_pixels)
                    )
                loss = loss + gradient_loss
                gradient_value += float(gradient_loss.detach())
            amp_scaler.scale(loss).backward()

        amp_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(optimization.get("gradient_clip_norm", 10.0))
        )
        amp_scaler.step(optimizer)
        amp_scaler.update()
        scheduler.step()

        record: dict[str, Any] = {
            "step": step,
            "train_reconstruction": reconstruction_value,
            "train_gradient": gradient_value if apply_gradient else None,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.perf_counter() - start_time,
        }
        if step == 1 or step % validation_every == 0 or step == max_steps:
            validation = _validate(
                model,
                scaler,
                validation_loader,
                device,
                pixel_chunk_size,
                mixed_precision,
            )
            record["validation_reconstruction"] = validation
            if validation < best_validation:
                best_validation = validation
                _save_checkpoint(
                    run_dir / "best.pt",
                    model,
                    optimizer,
                    scaler,
                    label_names,
                    wavelength,
                    config,
                    step,
                    best_validation,
                )
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        if step % checkpoint_every == 0 or step == max_steps:
            _save_checkpoint(
                run_dir / "last.pt",
                model,
                optimizer,
                scaler,
                label_names,
                wavelength,
                config,
                step,
                best_validation,
            )

    train_dataset.close()
    validation_dataset.close()
    return {
        "run_dir": str(run_dir),
        "best_validation": best_validation,
        "parameters": model.parameter_count,
        "steps": max_steps,
        "device": str(device),
        "full_ddpayne": gradient_library is not None,
    }
