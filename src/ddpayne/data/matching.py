from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord


def match_label_catalog(
    manifest_path: str | Path,
    labels_path: str | Path,
    output_path: str | Path,
    max_separation_arcsec: float = 1.0,
) -> int:
    manifest = pd.read_csv(manifest_path, low_memory=False)
    labels = pd.read_csv(labels_path, low_memory=False)
    for name, table in (("manifest", manifest), ("labels", labels)):
        missing = {"ra", "dec"} - set(table.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if "source_id" not in labels.columns:
        raise ValueError("labels is missing source_id")

    manifest_valid = np.isfinite(manifest["ra"]) & np.isfinite(manifest["dec"])
    labels_valid = np.isfinite(labels["ra"]) & np.isfinite(labels["dec"])
    manifest_work = manifest.loc[manifest_valid].copy()
    labels_work = labels.loc[labels_valid].copy()
    if labels_work.empty:
        raise ValueError("No finite coordinates in label catalog")

    spectrum_coordinates = SkyCoord(
        manifest_work["ra"].to_numpy() * u.deg,
        manifest_work["dec"].to_numpy() * u.deg,
        frame="icrs",
    )
    label_coordinates = SkyCoord(
        labels_work["ra"].to_numpy() * u.deg,
        labels_work["dec"].to_numpy() * u.deg,
        frame="icrs",
    )
    nearest, separation, _ = spectrum_coordinates.match_to_catalog_sky(label_coordinates)
    keep = separation.arcsec <= max_separation_arcsec
    matched_manifest = manifest_work.loc[keep].reset_index(drop=True)
    matched_labels = labels_work.iloc[nearest[keep]].reset_index(drop=True)

    matched_manifest = matched_manifest.rename(columns={"source_id": "lamost_source_id"})
    protected = {"path", "obsid", "lamost_source_id"}
    for column in matched_labels.columns:
        target = column
        if column in matched_manifest.columns and column not in protected:
            target = f"label_{column}" if column in {"ra", "dec"} else column
            if target in matched_manifest.columns:
                matched_manifest = matched_manifest.drop(columns=[target])
        matched_manifest[target] = matched_labels[column].to_numpy()
    matched_manifest["match_separation_arcsec"] = separation.arcsec[keep]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    matched_manifest.to_csv(output, index=False)
    return len(matched_manifest)
