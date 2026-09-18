from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from ddpayne.config import PROJECT_ROOT, load_yaml, project_path
from ddpayne.data.lamost import read_lamost_spectrum
from ddpayne.data.preprocess import common_log_wavelength_grid, preprocess_lamost_spectrum

SPLIT_NAMES = {0: "train", 1: "validation", 2: "test"}
REJECTION_COLUMNS = ("obsid", "source_id", "path", "snrg", "class", "stage", "reason")


def stable_split(group_id: str, split_config: dict[str, Any]) -> int:
    fractions = np.array(
        [
            float(split_config.get("train_fraction", 0.8)),
            float(split_config.get("validation_fraction", 0.1)),
            float(split_config.get("test_fraction", 0.1)),
        ],
        dtype=np.float64,
    )
    if np.any(fractions < 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("Split fractions must be non-negative and sum to 1")
    seed = int(split_config.get("seed", 42))
    digest = hashlib.blake2b(f"{seed}:{group_id}".encode(), digest_size=8).digest()
    draw = int.from_bytes(digest, "big") / float(2**64)
    return int(np.searchsorted(np.cumsum(fractions), draw, side="right"))


def _resolve_spectrum_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _passes_header_quality(row: pd.Series, quality: dict[str, Any]) -> tuple[bool, str]:
    if quality.get("require_class_star", True) and str(row.get("class", "")).upper() != "STAR":
        return False, "class is not STAR"
    minimum_snr = float(quality.get("minimum_snr_g", 0.0))
    try:
        snrg = float(row.get("snrg", np.nan))
    except (TypeError, ValueError):
        snrg = np.nan
    if not np.isfinite(snrg) or snrg < minimum_snr:
        return False, f"SNRG is below {minimum_snr:g}"
    return True, ""


def _rejection_record(row: pd.Series, path: str, stage: str, reason: str) -> dict[str, Any]:
    return {
        "obsid": row.get("obsid", ""),
        "source_id": row.get("source_id", ""),
        "path": path,
        "snrg": row.get("snrg", np.nan),
        "class": row.get("class", ""),
        "stage": stage,
        "reason": reason,
    }


def prepare_hdf5(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    labels_path = project_path(config["matched_labels"])
    output_path = project_path(config["output_h5"])
    rejections_path = project_path(config["rejections_csv"])
    summary_path = project_path(
        config.get("summary_json", rejections_path.with_suffix(".summary.json"))
    )
    label_names = [str(name) for name in config["labels"]]
    rows = pd.read_csv(labels_path, low_memory=False)
    missing = set(label_names + ["path", "source_id"]) - set(rows.columns)
    if missing:
        raise ValueError(f"Matched table is missing columns: {sorted(missing)}")

    wavelength_config = config["wavelength"]
    wavelength = common_log_wavelength_grid(
        wavelength_config["start_angstrom"],
        wavelength_config["end_angstrom"],
        wavelength_config["log10_step"],
    )
    quality = config.get("quality", {})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejections_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    string_dtype = h5py.string_dtype(encoding="utf-8")
    count = len(rows)
    chunk_rows = max(1, min(32, count))
    rejections: list[dict[str, Any]] = []
    written = 0

    with h5py.File(temporary, "w") as output:
        output.create_dataset("wavelength", data=wavelength)
        flux_ds = output.create_dataset(
            "flux",
            shape=(count, len(wavelength)),
            maxshape=(None, len(wavelength)),
            dtype="f4",
            chunks=(chunk_rows, len(wavelength)),
            compression="gzip",
            shuffle=True,
        )
        ivar_ds = output.create_dataset(
            "ivar",
            shape=(count, len(wavelength)),
            maxshape=(None, len(wavelength)),
            dtype="f4",
            chunks=(chunk_rows, len(wavelength)),
            compression="gzip",
            shuffle=True,
        )
        labels_ds = output.create_dataset(
            "labels", shape=(count, len(label_names)), maxshape=(None, len(label_names)), dtype="f4"
        )
        split_ds = output.create_dataset("split", shape=(count,), maxshape=(None,), dtype="u1")
        source_ds = output.create_dataset(
            "source_id", shape=(count,), maxshape=(None,), dtype=string_dtype
        )
        obsid_ds = output.create_dataset(
            "obsid", shape=(count,), maxshape=(None,), dtype=string_dtype
        )
        path_ds = output.create_dataset(
            "path", shape=(count,), maxshape=(None,), dtype=string_dtype
        )

        group_column = str(config.get("split", {}).get("group_column", "source_id"))
        for _, row in tqdm(rows.iterrows(), total=count, desc="Preprocessing", unit="spectrum"):
            accepted, reason = _passes_header_quality(row, quality)
            path_text = str(row.get("path", ""))
            if not accepted:
                rejections.append(_rejection_record(row, path_text, "header_quality", reason))
                continue
            label_values = row[label_names].to_numpy(dtype=np.float64)
            if not np.all(np.isfinite(label_values)):
                rejections.append(
                    _rejection_record(
                        row,
                        path_text,
                        "labels",
                        "one or more labels are missing",
                    )
                )
                continue
            try:
                spectrum = read_lamost_spectrum(_resolve_spectrum_path(path_text))
            except Exception as exc:
                rejections.append(
                    _rejection_record(
                        row,
                        path_text,
                        "read_spectrum",
                        f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            try:
                processed = preprocess_lamost_spectrum(
                    spectrum=spectrum,
                    target_wavelength=wavelength,
                    continuum_config=config.get("continuum", {}),
                    quality_config=quality,
                )
            except Exception as exc:
                rejections.append(
                    _rejection_record(
                        row,
                        path_text,
                        "preprocess_spectrum",
                        f"{type(exc).__name__}: {exc}",
                    )
                )
                continue

            source_id = str(row["source_id"])
            group_id = str(row.get(group_column, source_id))
            flux_ds[written] = processed.flux
            ivar_ds[written] = processed.ivar
            labels_ds[written] = label_values.astype(np.float32)
            split_ds[written] = stable_split(group_id, config.get("split", {}))
            source_ds[written] = source_id
            obsid_ds[written] = str(row.get("obsid", spectrum.metadata.get("obsid", "")))
            path_ds[written] = path_text
            written += 1

        if written == 0:
            raise ValueError("No spectra passed preprocessing")
        for dataset in (flux_ds, ivar_ds, labels_ds, split_ds, source_ds, obsid_ds, path_ds):
            dataset.resize(written, axis=0)
        output.attrs["label_names"] = json.dumps(label_names)
        output.attrs["config"] = json.dumps(config)
        output.attrs["format_version"] = "1"

    temporary.replace(output_path)
    rejection_table = pd.DataFrame(rejections, columns=REJECTION_COLUMNS)
    rejection_table.to_csv(rejections_path, index=False)
    with h5py.File(output_path, "r+") as dataset:
        split_counts = {
            SPLIT_NAMES[index]: int(np.sum(dataset["split"][:] == index)) for index in SPLIT_NAMES
        }
        dataset.attrs["input_rows"] = count
        dataset.attrs["accepted_rows"] = written
        dataset.attrs["rejected_rows"] = len(rejections)

    reason_counts = {
        str(reason): int(reason_count)
        for reason, reason_count in rejection_table["reason"].value_counts().items()
    }
    stage_counts = {
        str(stage): int(stage_count)
        for stage, stage_count in rejection_table["stage"].value_counts().items()
    }
    summary = {
        "output": str(output_path),
        "rejections_csv": str(rejections_path),
        "summary_json": str(summary_path),
        "input": count,
        "accepted": written,
        "rejected": len(rejections),
        "acceptance_fraction": written / count if count else 0.0,
        "pixels": len(wavelength),
        "splits": split_counts,
        "rejection_stages": stage_counts,
        "rejection_reasons": reason_counts,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
