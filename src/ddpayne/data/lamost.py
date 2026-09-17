from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

SPEED_OF_LIGHT_KMS = 299_792.458


@dataclass(frozen=True)
class LamostSpectrum:
    flux: np.ndarray
    ivar: np.ndarray
    wavelength: np.ndarray
    andmask: np.ndarray
    ormask: np.ndarray
    normalization: np.ndarray | None
    metadata: dict[str, Any]


def _array_from_row(row: Any, name: str) -> np.ndarray:
    return np.asarray(row[name], dtype=np.float64).reshape(-1)


def read_lamost_spectrum(path: str | Path) -> LamostSpectrum:
    """Read the LAMOST DR9 v2.0 COADD binary-table representation."""
    spectrum_path = Path(path)
    with fits.open(spectrum_path, memmap=False, lazy_load_hdus=False) as hdul:
        header = hdul[0].header
        if "COADD" not in hdul:
            raise ValueError(f"COADD extension is missing: {spectrum_path}")
        table = hdul["COADD"].data
        if table is None or len(table) != 1:
            raise ValueError(f"Expected one COADD row: {spectrum_path}")
        row = table[0]
        names = {name.upper() for name in (table.names or [])}
        required = {"FLUX", "IVAR", "WAVELENGTH"}
        missing = required - names
        if missing:
            raise ValueError(f"Missing COADD columns {sorted(missing)}: {spectrum_path}")

        flux = _array_from_row(row, "FLUX")
        ivar = _array_from_row(row, "IVAR")
        wavelength = _array_from_row(row, "WAVELENGTH")
        zeros = np.zeros_like(flux)
        andmask = _array_from_row(row, "ANDMASK") if "ANDMASK" in names else zeros
        ormask = _array_from_row(row, "ORMASK") if "ORMASK" in names else zeros
        normalization = _array_from_row(row, "NORMALIZATION") if "NORMALIZATION" in names else None

        metadata = {
            "path": str(spectrum_path),
            "obsid": str(header.get("OBSID", "")),
            "source_id": str(header.get("DESIG", header.get("OBJNAME", ""))).strip(),
            "designation": str(header.get("DESIG", "")).strip(),
            "ra": _finite_float(header.get("RA")),
            "dec": _finite_float(header.get("DEC")),
            "lmjd": str(header.get("LMJD", "")),
            "planid": str(header.get("PLANID", "")).strip(),
            "fiberid": str(header.get("FIBERID", "")),
            "class": str(header.get("CLASS", "")).strip(),
            "subclass": str(header.get("SUBCLASS", "")).strip(),
            "z": _finite_float(header.get("Z")),
            "z_err": _finite_float(header.get("Z_ERR")),
            "snrg": _finite_float(header.get("SNRG")),
            "snrr": _finite_float(header.get("SNRR")),
            "snri": _finite_float(header.get("SNRI")),
            "fib_mask": str(header.get("FIB_MASK", "")),
        }

    lengths = {len(flux), len(ivar), len(wavelength), len(andmask), len(ormask)}
    if normalization is not None:
        lengths.add(len(normalization))
    if len(lengths) != 1:
        raise ValueError(f"COADD columns have inconsistent lengths: {spectrum_path}")
    return LamostSpectrum(
        flux=flux,
        ivar=ivar,
        wavelength=wavelength,
        andmask=andmask,
        ormask=ormask,
        normalization=normalization,
        metadata=metadata,
    )


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def radial_velocity_kms(metadata: dict[str, Any]) -> float:
    z = _finite_float(metadata.get("z"))
    if not np.isfinite(z) or abs(z) > 0.02:
        raise ValueError(f"Invalid stellar redshift Z={z!r}")
    return z * SPEED_OF_LIGHT_KMS
