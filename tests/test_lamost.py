from __future__ import annotations

import gzip
import shutil

import numpy as np
from astropy.io import fits

from ddpayne.data.lamost import read_lamost_spectrum


def test_read_lamost_binary_table(tmp_path):
    length = 5
    columns = []
    for name, values in (
        ("FLUX", np.linspace(1, 2, length)),
        ("IVAR", np.full(length, 4.0)),
        ("WAVELENGTH", np.linspace(4000, 5000, length)),
        ("ANDMASK", np.zeros(length)),
        ("ORMASK", np.zeros(length)),
        ("NORMALIZATION", np.full(length, 2.0)),
    ):
        columns.append(
            fits.Column(name=name, format=f"{length}E", array=[values.astype(np.float32)])
        )
    primary = fits.PrimaryHDU()
    primary.header["OBSID"] = 123
    primary.header["DESIG"] = "LAMOST TEST"
    primary.header["RA"] = 120.0
    primary.header["DEC"] = 30.0
    primary.header["Z"] = 0.0001
    table = fits.BinTableHDU.from_columns(columns, name="COADD")
    uncompressed = tmp_path / "test.fits"
    compressed = tmp_path / "test.fits.gz"
    fits.HDUList([primary, table]).writeto(uncompressed)
    with uncompressed.open("rb") as source, gzip.open(compressed, "wb") as target:
        shutil.copyfileobj(source, target)

    spectrum = read_lamost_spectrum(compressed)

    assert spectrum.flux.shape == (length,)
    assert np.allclose(spectrum.ivar, 4.0)
    assert spectrum.metadata["obsid"] == "123"
    assert spectrum.metadata["source_id"] == "LAMOST TEST"
