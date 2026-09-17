from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from tqdm import tqdm

MANIFEST_COLUMNS = [
    "path",
    "obsid",
    "source_id",
    "designation",
    "ra",
    "dec",
    "lmjd",
    "planid",
    "fiberid",
    "class",
    "subclass",
    "z",
    "z_err",
    "snrg",
    "snrr",
    "snri",
    "fib_mask",
]


def iter_spectra(root: str | Path) -> Iterable[Path]:
    yield from sorted(Path(root).rglob("*.fits.gz"))


def _header_value(header: fits.Header, key: str, default: Any = "") -> Any:
    value = header.get(key, default)
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, np.generic):
        return value.item()
    return value


def manifest_row(path: Path, base: Path) -> dict[str, Any]:
    header = fits.getheader(path, ext=0)
    designation = str(_header_value(header, "DESIG", ""))
    source_id = designation or str(_header_value(header, "OBJNAME", ""))
    try:
        stored_path = str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        stored_path = str(path.resolve())
    return {
        "path": stored_path,
        "obsid": _header_value(header, "OBSID"),
        "source_id": source_id,
        "designation": designation,
        "ra": _header_value(header, "RA"),
        "dec": _header_value(header, "DEC"),
        "lmjd": _header_value(header, "LMJD"),
        "planid": _header_value(header, "PLANID"),
        "fiberid": _header_value(header, "FIBERID"),
        "class": _header_value(header, "CLASS"),
        "subclass": _header_value(header, "SUBCLASS"),
        "z": _header_value(header, "Z"),
        "z_err": _header_value(header, "Z_ERR"),
        "snrg": _header_value(header, "SNRG"),
        "snrr": _header_value(header, "SNRR"),
        "snri": _header_value(header, "SNRI"),
        "fib_mask": _header_value(header, "FIB_MASK"),
    }


def build_manifest(
    spectra_root: str | Path,
    output: str | Path,
    base: str | Path,
    limit: int | None = None,
) -> tuple[int, int]:
    root = Path(spectra_root)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    successes = 0
    failures = 0
    paths = iter_spectra(root)
    if limit is not None:
        from itertools import islice

        paths = islice(paths, limit)
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS + ["error"])
        writer.writeheader()
        for path in tqdm(paths, desc="Reading FITS headers", unit="spectrum"):
            try:
                row = manifest_row(path, Path(base))
                row["error"] = ""
                successes += 1
            except Exception as exc:  # Keep a complete audit trail for corrupt FITS files.
                row = {column: "" for column in MANIFEST_COLUMNS}
                row["path"] = str(path)
                row["error"] = f"{type(exc).__name__}: {exc}"
                failures += 1
            writer.writerow(row)
    temporary.replace(output_path)
    return successes, failures
