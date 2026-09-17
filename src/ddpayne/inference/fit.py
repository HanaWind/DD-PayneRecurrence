from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ddpayne.config import PROJECT_ROOT, load_yaml, project_path
from ddpayne.data.lamost import read_lamost_spectrum
from ddpayne.data.preprocess import preprocess_lamost_spectrum
from ddpayne.models.payne import LabelScaler, PixelwisePayne


def _pixel_slices(n_pixels: int, chunk_size: int) -> Iterator[slice]:
    for start in range(0, n_pixels, chunk_size):
        yield slice(start, min(start + chunk_size, n_pixels))


def load_model_checkpoint(
    path: str | Path, device: str | torch.device
) -> tuple[PixelwisePayne, LabelScaler, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = PixelwisePayne(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    scaler = LabelScaler.from_state_dict(checkpoint["label_scaler"])
    return model, scaler, checkpoint


def fit_label_batch(
    model: PixelwisePayne,
    scaler: LabelScaler,
    flux: torch.Tensor,
    ivar: torch.Tensor,
    steps: int = 300,
    learning_rate: float = 0.03,
    pixel_chunk_size: int = 512,
    estimate_uncertainty: bool = False,
) -> dict[str, np.ndarray]:
    if flux.ndim != 2 or flux.shape != ivar.shape or flux.shape[1] != model.n_pixels:
        raise ValueError("flux/ivar must have shape [batch, model.n_pixels]")
    device = next(model.parameters()).device
    flux = flux.to(device)
    ivar = ivar.to(device)
    dtype = flux.dtype
    physical_minimum = torch.as_tensor(scaler.minimum, device=device, dtype=dtype)
    physical_maximum = torch.as_tensor(scaler.maximum, device=device, dtype=dtype)
    lower = scaler.transform(physical_minimum)
    upper = scaler.transform(physical_maximum)
    fraction = torch.clamp((torch.zeros_like(lower) - lower) / (upper - lower), 1e-4, 1 - 1e-4)
    initial = torch.log(fraction / (1 - fraction))
    raw = initial.repeat(len(flux), 1).clone().requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=learning_rate)
    valid_per_object = torch.sum(torch.isfinite(ivar) & (ivar > 0), dim=1).clamp_min(1)

    model.requires_grad_(False)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        for selection in _pixel_slices(model.n_pixels, pixel_chunk_size):
            scaled_labels = lower + torch.sigmoid(raw) * (upper - lower)
            prediction = model(scaled_labels, pixel_slice=selection)
            target = flux[:, selection]
            weights = ivar[:, selection]
            valid = torch.isfinite(target) & torch.isfinite(weights) & (weights > 0)
            residual = torch.where(valid, prediction - target, 0.0)
            object_loss = (
                torch.sum(residual.square() * torch.where(valid, weights, 0.0), dim=1)
                / valid_per_object
            )
            object_loss.mean().backward()
        optimizer.step()
    model.requires_grad_(True)

    with torch.no_grad():
        scaled_labels = lower + torch.sigmoid(raw) * (upper - lower)
        physical_labels = scaler.inverse_transform(scaled_labels)
        chi2 = torch.zeros(len(flux), device=device, dtype=dtype)
        for selection in _pixel_slices(model.n_pixels, pixel_chunk_size):
            prediction = model(scaled_labels, pixel_slice=selection)
            valid = torch.isfinite(ivar[:, selection]) & (ivar[:, selection] > 0)
            residual = torch.where(valid, prediction - flux[:, selection], 0.0)
            chi2 += torch.sum(
                residual.square() * torch.where(valid, ivar[:, selection], 0.0), dim=1
            )
        reduced_chi2 = chi2 / valid_per_object
        fractional_position = (physical_labels - physical_minimum) / (
            physical_maximum - physical_minimum
        )
        boundary = torch.any((fractional_position < 0.01) | (fractional_position > 0.99), dim=1)

    result = {
        "labels": physical_labels.detach().cpu().numpy(),
        "reduced_chi2": reduced_chi2.detach().cpu().numpy(),
        "valid_pixels": valid_per_object.detach().cpu().numpy(),
        "boundary_flag": boundary.detach().cpu().numpy(),
    }
    if estimate_uncertainty:
        uncertainties = []
        for object_index in range(len(flux)):
            uncertainties.append(
                _fisher_uncertainty(
                    model,
                    scaler,
                    scaled_labels[object_index].detach(),
                    ivar[object_index],
                )
            )
        result["uncertainties"] = np.asarray(uncertainties)
    return result


def _fisher_uncertainty(
    model: PixelwisePayne,
    scaler: LabelScaler,
    scaled_labels: torch.Tensor,
    ivar: torch.Tensor,
) -> np.ndarray:
    scaled_labels = scaled_labels.detach().requires_grad_(True)

    def predict(value: torch.Tensor) -> torch.Tensor:
        return model(value.unsqueeze(0)).squeeze(0)

    jacobian_scaled = torch.autograd.functional.jacobian(predict, scaled_labels, vectorize=True)
    physical_scale = torch.as_tensor(
        scaler.scale, device=jacobian_scaled.device, dtype=jacobian_scaled.dtype
    )
    jacobian = jacobian_scaled / physical_scale.unsqueeze(0)
    valid = torch.isfinite(ivar) & (ivar > 0)
    weighted = jacobian[valid] * ivar[valid].sqrt().unsqueeze(1)
    fisher = weighted.T @ weighted
    covariance = torch.linalg.pinv(fisher)
    return torch.sqrt(torch.clamp(torch.diag(covariance), min=0)).detach().cpu().numpy()


def _spectrum_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _add_qflag_chi2(table: pd.DataFrame) -> pd.DataFrame:
    accepted = table["status"] == "ok"
    table["qflag_chi2"] = np.nan
    if accepted.sum() < 10:
        return table
    snr = pd.to_numeric(table.loc[accepted, "snrg"], errors="coerce").clip(lower=1e-3)
    bins = pd.qcut(np.log10(snr), q=min(20, int(accepted.sum())), duplicates="drop")
    for _, indices in table.loc[accepted].groupby(bins, observed=True).groups.items():
        values = table.loc[indices, "reduced_chi2"].to_numpy(dtype=float)
        median = np.nanmedian(values)
        mad = np.nanmedian(np.abs(values - median))
        if np.isfinite(mad) and mad > 0:
            table.loc[indices, "qflag_chi2"] = (values - median) / mad
    return table


def infer_from_config(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    device_name = str(config.get("device", "cuda"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(device_name)
    model, scaler, checkpoint = load_model_checkpoint(project_path(config["checkpoint"]), device)
    label_names = list(checkpoint["label_names"])
    wavelength = np.asarray(checkpoint["wavelength"], dtype=np.float64)
    manifest = pd.read_csv(project_path(config["manifest"]), low_memory=False)
    maximum = config.get("max_objects")
    if maximum is not None:
        manifest = manifest.iloc[: int(maximum)]
    preprocess_config = load_yaml(project_path(config["preprocess_config"]))
    quality = preprocess_config.get("quality", {})
    continuum = preprocess_config.get("continuum", {})
    optimizer_config = config.get("optimizer", {})
    batch_size = int(config.get("batch_size", 64))
    rows: list[dict[str, Any]] = []
    pending_flux: list[np.ndarray] = []
    pending_ivar: list[np.ndarray] = []
    pending_metadata: list[dict[str, Any]] = []

    def flush() -> None:
        if not pending_flux:
            return
        result = fit_label_batch(
            model=model,
            scaler=scaler,
            flux=torch.as_tensor(np.asarray(pending_flux), dtype=torch.float32),
            ivar=torch.as_tensor(np.asarray(pending_ivar), dtype=torch.float32),
            steps=int(optimizer_config.get("steps", 300)),
            learning_rate=float(optimizer_config.get("learning_rate", 0.03)),
            pixel_chunk_size=int(
                checkpoint["config"].get("model", {}).get("pixel_chunk_size", 512)
            ),
            estimate_uncertainty=bool(optimizer_config.get("estimate_uncertainty", False)),
        )
        for index, metadata in enumerate(pending_metadata):
            row = dict(metadata)
            row.update(
                {
                    name: float(result["labels"][index, column])
                    for column, name in enumerate(label_names)
                }
            )
            if "uncertainties" in result:
                row.update(
                    {
                        f"{name}_err": float(result["uncertainties"][index, column])
                        for column, name in enumerate(label_names)
                    }
                )
            row["reduced_chi2"] = float(result["reduced_chi2"][index])
            row["valid_pixels"] = int(result["valid_pixels"][index])
            row["boundary_flag"] = bool(result["boundary_flag"][index])
            row["status"] = "ok"
            row["error"] = ""
            rows.append(row)
        pending_flux.clear()
        pending_ivar.clear()
        pending_metadata.clear()

    for _, manifest_row in manifest.iterrows():
        metadata = {
            "path": str(manifest_row.get("path", "")),
            "obsid": str(manifest_row.get("obsid", "")),
            "source_id": str(manifest_row.get("source_id", "")),
            "snrg": manifest_row.get("snrg", np.nan),
        }
        try:
            spectrum = read_lamost_spectrum(_spectrum_path(metadata["path"]))
            processed = preprocess_lamost_spectrum(
                spectrum=spectrum,
                target_wavelength=wavelength,
                continuum_config=continuum,
                quality_config=quality,
            )
            pending_flux.append(processed.flux)
            pending_ivar.append(processed.ivar)
            pending_metadata.append(metadata)
            if len(pending_flux) >= batch_size:
                flush()
        except Exception as exc:
            metadata.update(
                {
                    "status": "rejected",
                    "error": f"{type(exc).__name__}: {exc}",
                    "reduced_chi2": np.nan,
                    "valid_pixels": 0,
                    "boundary_flag": True,
                }
            )
            rows.append(metadata)
    flush()

    output = _add_qflag_chi2(pd.DataFrame(rows))
    output_path = project_path(config["output_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "checkpoint": str(project_path(config["checkpoint"])),
                "checkpoint_step": checkpoint["step"],
                "label_names": label_names,
                "objects": len(output),
                "accepted": int((output["status"] == "ok").sum()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "output": str(output_path),
        "objects": len(output),
        "accepted": int((output["status"] == "ok").sum()),
        "rejected": int((output["status"] != "ok").sum()),
    }
