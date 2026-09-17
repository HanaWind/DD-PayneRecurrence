from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ddpayne.data.lamost import SPEED_OF_LIGHT_KMS, LamostSpectrum, radial_velocity_kms


@dataclass(frozen=True)
class ProcessedSpectrum:
    flux: np.ndarray
    ivar: np.ndarray
    wavelength: np.ndarray
    valid_fraction: float


def common_log_wavelength_grid(
    start_angstrom: float, end_angstrom: float, log10_step: float
) -> np.ndarray:
    if not 0 < start_angstrom < end_angstrom or log10_step <= 0:
        raise ValueError("Invalid wavelength-grid boundaries")
    start = np.log10(start_angstrom)
    end = np.log10(end_angstrom)
    count = int(np.floor((end - start) / log10_step)) + 1
    wavelength = np.power(10.0, start + np.arange(count, dtype=np.float64) * log10_step)
    wavelength[0] = start_angstrom
    return wavelength


def to_rest_frame(wavelength: np.ndarray, rv_kms: float) -> np.ndarray:
    return np.asarray(wavelength, dtype=np.float64) / (1.0 + rv_kms / SPEED_OF_LIGHT_KMS)


def gaussian_pseudo_continuum(
    wavelength: np.ndarray,
    flux: np.ndarray,
    valid: np.ndarray,
    sigma_angstrom: float,
    exclude_ranges_angstrom: Iterable[Iterable[float]] = (),
) -> np.ndarray:
    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    continuum_valid = np.asarray(valid, dtype=bool).copy()
    if len(wavelength) < 3 or np.any(np.diff(wavelength) <= 0):
        raise ValueError("Wavelength must be strictly increasing")
    for bounds in exclude_ranges_angstrom:
        low, high = (float(value) for value in bounds)
        continuum_valid &= ~((wavelength >= low) & (wavelength <= high))

    spacing = float(np.nanmedian(np.diff(wavelength)))
    sigma_pixels = sigma_angstrom / spacing
    if not np.isfinite(sigma_pixels) or sigma_pixels <= 0:
        raise ValueError("Invalid Gaussian continuum width")

    weights = continuum_valid.astype(np.float64)
    numerator = gaussian_filter1d(
        np.where(continuum_valid, flux, 0.0), sigma_pixels, mode="nearest"
    )
    denominator = gaussian_filter1d(weights, sigma_pixels, mode="nearest")
    continuum = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 1e-6,
    )
    return continuum


def normalize_spectrum(
    spectrum: LamostSpectrum,
    continuum_config: dict[str, Any],
    reject_nonzero_andmask: bool = True,
    reject_nonzero_ormask: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flux = spectrum.flux.astype(np.float64, copy=True)
    ivar = spectrum.ivar.astype(np.float64, copy=True)
    wavelength = spectrum.wavelength.astype(np.float64, copy=False)
    valid = np.isfinite(flux) & np.isfinite(ivar) & np.isfinite(wavelength) & (ivar > 0)
    if reject_nonzero_andmask:
        valid &= spectrum.andmask == 0
    if reject_nonzero_ormask:
        valid &= spectrum.ormask == 0

    mode = str(continuum_config.get("mode", "gaussian")).lower()
    if mode == "pipeline":
        if spectrum.normalization is None:
            raise ValueError("Pipeline normalization requested but column is unavailable")
        continuum = spectrum.normalization.astype(np.float64, copy=False)
    elif mode == "gaussian":
        continuum = gaussian_pseudo_continuum(
            wavelength=wavelength,
            flux=flux,
            valid=valid,
            sigma_angstrom=float(continuum_config.get("gaussian_sigma_angstrom", 50.0)),
            exclude_ranges_angstrom=continuum_config.get("exclude_ranges_angstrom", ()),
        )
    else:
        raise ValueError(f"Unknown continuum mode: {mode}")

    valid &= np.isfinite(continuum) & (continuum > 0)
    normalized_flux = np.divide(flux, continuum, out=np.ones_like(flux), where=valid)
    normalized_ivar = np.where(valid, ivar * np.square(continuum), 0.0)
    valid &= np.isfinite(normalized_flux) & np.isfinite(normalized_ivar)
    normalized_flux[~valid] = 1.0
    normalized_ivar[~valid] = 0.0
    return normalized_flux, normalized_ivar, valid


def resample_spectrum(
    source_wavelength: np.ndarray,
    source_flux: np.ndarray,
    source_ivar: np.ndarray,
    target_wavelength: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(source_wavelength)
    wave = np.asarray(source_wavelength, dtype=np.float64)[order]
    flux = np.asarray(source_flux, dtype=np.float64)[order]
    ivar = np.asarray(source_ivar, dtype=np.float64)[order]
    valid = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0)
    if valid.sum() < 3:
        raise ValueError("Fewer than three valid source pixels")

    wave_valid = wave[valid]
    flux_valid = flux[valid]
    target = np.asarray(target_wavelength, dtype=np.float64)
    output_flux = np.interp(target, wave_valid, flux_valid, left=1.0, right=1.0)
    output_ivar = np.interp(target, wave, np.where(valid, ivar, 0.0), left=0.0, right=0.0)
    coverage = np.interp(target, wave, valid.astype(np.float64), left=0.0, right=0.0)
    output_ivar[coverage < 0.5] = 0.0
    output_flux[output_ivar <= 0] = 1.0
    return output_flux.astype(np.float32), output_ivar.astype(np.float32)


def preprocess_lamost_spectrum(
    spectrum: LamostSpectrum,
    target_wavelength: np.ndarray,
    continuum_config: dict[str, Any],
    quality_config: dict[str, Any],
) -> ProcessedSpectrum:
    flux, ivar, _ = normalize_spectrum(
        spectrum,
        continuum_config=continuum_config,
        reject_nonzero_andmask=bool(quality_config.get("reject_nonzero_andmask", True)),
        reject_nonzero_ormask=bool(quality_config.get("reject_nonzero_ormask", False)),
    )
    rest_wavelength = to_rest_frame(spectrum.wavelength, radial_velocity_kms(spectrum.metadata))
    output_flux, output_ivar = resample_spectrum(
        source_wavelength=rest_wavelength,
        source_flux=flux,
        source_ivar=ivar,
        target_wavelength=target_wavelength,
    )
    valid_fraction = float(np.mean(output_ivar > 0))
    minimum = float(quality_config.get("minimum_valid_fraction", 0.8))
    if valid_fraction < minimum:
        raise ValueError(
            f"Valid-pixel fraction {valid_fraction:.3f} is below required {minimum:.3f}"
        )
    return ProcessedSpectrum(
        flux=output_flux,
        ivar=output_ivar,
        wavelength=np.asarray(target_wavelength, dtype=np.float64),
        valid_fraction=valid_fraction,
    )
