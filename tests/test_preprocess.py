from __future__ import annotations

import numpy as np

from ddpayne.data.lamost import LamostSpectrum
from ddpayne.data.preprocess import (
    common_log_wavelength_grid,
    gaussian_pseudo_continuum,
    preprocess_lamost_spectrum,
)


def test_log_wavelength_grid_is_strictly_increasing():
    wavelength = common_log_wavelength_grid(4000, 5000, 0.0001)
    assert np.all(np.diff(wavelength) > 0)
    assert wavelength[0] == 4000
    assert wavelength[-1] <= 5000


def test_gaussian_continuum_ignores_masked_line():
    wavelength = np.linspace(4000, 5000, 1001)
    flux = np.full_like(wavelength, 2.0)
    flux[(wavelength > 4495) & (wavelength < 4505)] = 0.5
    continuum = gaussian_pseudo_continuum(
        wavelength,
        flux,
        np.ones_like(flux, dtype=bool),
        sigma_angstrom=50,
        exclude_ranges_angstrom=[(4490, 4510)],
    )
    assert np.nanmedian(np.abs(continuum - 2.0)) < 1e-4


def test_preprocess_marks_masked_pixels_with_zero_weight():
    wavelength = np.linspace(4000, 5000, 1001)
    andmask = np.zeros_like(wavelength)
    andmask[500] = 1
    spectrum = LamostSpectrum(
        flux=np.full_like(wavelength, 2.0),
        ivar=np.full_like(wavelength, 25.0),
        wavelength=wavelength,
        andmask=andmask,
        ormask=np.zeros_like(wavelength),
        normalization=np.full_like(wavelength, 2.0),
        metadata={"z": 0.0},
    )
    processed = preprocess_lamost_spectrum(
        spectrum,
        target_wavelength=wavelength,
        continuum_config={"mode": "pipeline"},
        quality_config={
            "reject_nonzero_andmask": True,
            "reject_nonzero_ormask": False,
            "minimum_valid_fraction": 0.9,
        },
    )
    assert np.allclose(processed.flux[processed.ivar > 0], 1.0)
    assert processed.ivar[500] == 0
