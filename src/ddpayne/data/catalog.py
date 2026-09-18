from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table import Table

LABEL_NAMES = (
    "teff",
    "logg",
    "vmic",
    "fe_h",
    "c_fe",
    "n_fe",
    "o_fe",
    "mg_fe",
    "al_fe",
    "si_fe",
    "ca_fe",
    "ti_fe",
    "cr_fe",
    "mn_fe",
    "ni_fe",
)


def read_catalog(path: str | Path) -> pd.DataFrame:
    """Read a LAMOST/APOGEE/GALAH catalog into a pandas table."""
    catalog_path = Path(path)
    if not catalog_path.is_file():
        message = f"Catalog file does not exist: {catalog_path}"
        placeholder_names = ("lamost_full_lrs_catalog", "lamost_dr9_lrs_general")
        if any(catalog_path.name.lower().startswith(name) for name in placeholder_names):
            message += (
                "\nThis is an example placeholder for a downloaded official LAMOST "
                "catalog. For the local spectra in this workspace, use "
                "data/metadata/spectra.csv (or create it with ddpayne build-manifest). "
                "For a full-survey run, download the official LAMOST catalog and "
                "pass its actual path."
            )
        raise FileNotFoundError(message)
    suffixes = [suffix.lower() for suffix in catalog_path.suffixes]
    if ".fits" in suffixes or ".fit" in suffixes:
        table = Table.read(catalog_path, hdu=1)
        # APOGEE allStar contains vector columns such as FPARAM, FELEM and
        # PARAM_COV. They are not needed for the candidate join, and pandas
        # cannot represent them as ordinary scalar DataFrame columns.
        scalar_names = [name for name in table.colnames if np.asarray(table[name]).ndim <= 1]
        return table[scalar_names].to_pandas()
    if ".parquet" in suffixes:
        return pd.read_parquet(catalog_path)
    if ".csv" in suffixes or ".txt" in suffixes:
        return pd.read_csv(catalog_path, low_memory=False)
    raise ValueError(f"Unsupported catalog format: {catalog_path}")


def _normalized_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _column_lookup(table: pd.DataFrame) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for column in table.columns:
        lookup.setdefault(_normalized_name(column), str(column))
    return lookup


def _first_column(
    table: pd.DataFrame, lookup: dict[str, str], aliases: tuple[str, ...]
) -> pd.Series | None:
    for alias in aliases:
        column = lookup.get(_normalized_name(alias))
        if column is not None:
            return table[column]
    return None


def _clean_identifier(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "-999", "-9999"}:
        return None
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text


def _identifier_series(values: pd.Series | None, length: int) -> pd.Series:
    if values is None:
        return pd.Series([None] * length, dtype="string")
    return values.map(_clean_identifier).astype("string")


def _float_series(values: pd.Series | None, length: int) -> pd.Series:
    if values is None:
        return pd.Series(np.full(length, np.nan), dtype="float64")
    return pd.to_numeric(values, errors="coerce")


def _standardize_lamost(table: pd.DataFrame, gaia_version: str) -> pd.DataFrame:
    lookup = _column_lookup(table)
    result = pd.DataFrame(index=table.index)
    aliases = {
        "obsid": ("obsid", "obs_id", "spectrum_id", "spec_id"),
        "source_id": ("source_id", "uid", "designation", "objname"),
        "ra": ("ra", "ra_input", "ra_obs", "ra_target"),
        "dec": ("dec", "dec_input", "dec_obs", "dec_target"),
        "path": ("path", "local_path", "fitsname", "spectrum_path", "file_path"),
        "spectrum_uri": ("spectrum_uri", "uri", "download_url", "url"),
        "release": ("lamost_release", "release", "data_release"),
        "survey_mode": ("survey_mode", "mode", "spectro_mode"),
        "class": ("class", "objtype", "spec_class", "classification"),
        "subclass": ("subclass", "sub_class"),
        "snrg": ("snrg", "snr_g", "snr_g_band"),
        "snru": ("snru", "snr_u", "snr_u_band"),
        "snrr": ("snrr", "snr_r", "snr_r_band"),
        "snri": ("snri", "snr_i", "snr_i_band"),
        "snrz": ("snrz", "snr_z", "snr_z_band"),
        "z": ("z", "redshift"),
        "z_err": ("z_err", "redshift_err"),
    }
    for name, candidates in aliases.items():
        values = _first_column(table, lookup, candidates)
        if name in {"ra", "dec", "snrg", "snru", "snrr", "snri", "snrz", "z", "z_err"}:
            result[name] = _float_series(values, len(table))
        else:
            result[name] = values.to_numpy() if values is not None else ""

    result["gaia_dr3_source_id"] = _identifier_series(
        _first_column(
            table,
            lookup,
            ("gaia_dr3_source_id", "gaiaedr3_source_id", "dr3_source_id")
            + (("gaia_source_id",) if gaia_version in {"auto", "dr3"} else ()),
        ),
        len(table),
    )
    result["gaia_dr2_source_id"] = _identifier_series(
        _first_column(
            table,
            lookup,
            ("gaia_dr2_source_id", "dr2_source_id")
            + (("gaia_source_id",) if gaia_version == "dr2" else ()),
        ),
        len(table),
    )
    return result.reset_index(drop=True)


def _label_aliases(name: str) -> tuple[str, ...]:
    aliases: dict[str, tuple[str, ...]] = {
        "teff": ("teff", "teff_spec", "teff_cal"),
        "logg": ("logg", "logg_spec", "logg_cal"),
        "vmic": ("vmic", "vmicro", "vmicroturb", "v_micro"),
        "fe_h": ("fe_h", "feh", "fe_h_spec"),
    }
    return aliases.get(name, (name, name.replace("_", "")))


def _standardize_labels(table: pd.DataFrame, gaia_version: str) -> pd.DataFrame:
    lookup = _column_lookup(table)
    result = pd.DataFrame(index=table.index)
    source = _first_column(
        table,
        lookup,
        ("source_id", "apogee_id", "sobject_id", "star_id", "object_id"),
    )
    result["source_id"] = _identifier_series(source, len(table))
    result["gaia_dr3_source_id"] = _identifier_series(
        _first_column(
            table,
            lookup,
            ("gaia_dr3_source_id", "gaiaedr3_source_id", "dr3_source_id")
            + (("gaia_source_id",) if gaia_version in {"auto", "dr3"} else ()),
        ),
        len(table),
    )
    result["gaia_dr2_source_id"] = _identifier_series(
        _first_column(
            table,
            lookup,
            ("gaia_dr2_source_id", "dr2_source_id")
            + (("gaia_source_id",) if gaia_version == "dr2" else ()),
        ),
        len(table),
    )
    result["ra"] = _float_series(_first_column(table, lookup, ("ra", "ra_deg")), len(table))
    result["dec"] = _float_series(_first_column(table, lookup, ("dec", "dec_deg")), len(table))
    for name in LABEL_NAMES:
        result[name] = _float_series(_first_column(table, lookup, _label_aliases(name)), len(table))

    audit_aliases = {
        "aspcapflag": ("aspcapflag", "aspcap_flag"),
        "starflag": ("starflag", "star_flag"),
        "extratarg": ("extratarg", "extra_targ"),
        "snr": ("snr", "snr_combined", "snr_apogee"),
    }
    for name in LABEL_NAMES:
        audit_aliases[f"{name}_err"] = (f"{name}_err",)
        audit_aliases[f"{name}_flag"] = (f"{name}_flag",)
    for name, candidates in audit_aliases.items():
        values = _first_column(table, lookup, candidates)
        if values is not None:
            result[name] = values.to_numpy()

    label_source = _first_column(table, lookup, ("label_source", "survey", "catalog"))
    result["label_source"] = label_source.astype("string") if label_source is not None else ""
    return result.reset_index(drop=True)


def _filter_apogee_quality(table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply a conservative, explicit first-pass APOGEE quality selection."""
    lookup = _column_lookup(table)
    standardized = _standardize_labels(table, "auto")
    keep = np.ones(len(table), dtype=bool)
    rejected: dict[str, int] = {}

    finite_labels = standardized[list(LABEL_NAMES)].notna().all(axis=1).to_numpy()
    rejected["nonfinite_labels"] = int((~finite_labels).sum())
    keep &= finite_labels

    aspcapflag = _first_column(table, lookup, ("aspcapflag", "aspcap_flag"))
    if aspcapflag is not None:
        flags = pd.to_numeric(aspcapflag, errors="coerce").fillna(2**63 - 1).astype("int64")
        bad = (flags & ((1 << 23) | (1 << 24))) != 0
        rejected["aspcap_bad"] = int(bad.sum())
        keep &= ~bad.to_numpy()

    extratarg = _first_column(table, lookup, ("extratarg", "extra_targ"))
    if extratarg is not None:
        extra = pd.to_numeric(extratarg, errors="coerce").fillna(1)
        bad = extra != 0
        rejected["extratarg"] = int(bad.sum())
        keep &= ~bad.to_numpy()

    snr = _first_column(table, lookup, ("snr", "snr_combined", "snr_apogee"))
    if snr is not None:
        low_snr = pd.to_numeric(snr, errors="coerce").fillna(-np.inf) < 70.0
        rejected["snr_below_70"] = int(low_snr.sum())
        keep &= ~low_snr.to_numpy()

    for name in LABEL_NAMES:
        flag = _first_column(table, lookup, (f"{name}_flag",))
        if flag is not None:
            bad = pd.to_numeric(flag, errors="coerce").fillna(1) != 0
            rejected[f"{name}_flag"] = int(bad.sum())
            keep &= ~bad.to_numpy()

    rejected["total_rejected_union"] = int((~keep).sum())
    return table.loc[keep].reset_index(drop=True), rejected


def _finite_coordinates(table: pd.DataFrame, index: int) -> bool:
    return bool(np.isfinite(table.at[index, "ra"]) and np.isfinite(table.at[index, "dec"]))


def build_candidate_table(
    lamost_catalog: str | Path,
    label_catalog: str | Path,
    max_separation_arcsec: float = 1.0,
    gaia_version: str = "auto",
    quality: str = "none",
    limit: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Match an official LAMOST observation catalog to high-resolution labels.

    Gaia IDs are used first. Sky coordinates are only used for rows without a
    usable ID match. The returned table is directly usable by ``prepare`` once
    each selected observation has a local ``path``.
    """
    if gaia_version not in {"auto", "dr2", "dr3"}:
        raise ValueError("gaia_version must be auto, dr2, or dr3")
    if quality not in {"none", "apogee"}:
        raise ValueError("quality must be none or apogee")
    lamost_raw = read_catalog(lamost_catalog)
    labels_raw = read_catalog(label_catalog)
    if limit is not None:
        lamost_raw = lamost_raw.head(int(limit))
    lamost = _standardize_lamost(lamost_raw, gaia_version)
    quality_summary: dict[str, int] = {}
    if quality == "apogee":
        labels_raw, quality_summary = _filter_apogee_quality(labels_raw)
    labels = _standardize_labels(labels_raw, gaia_version)
    if lamost["obsid"].isna().all() or (lamost["obsid"].astype(str).str.strip() == "").all():
        raise ValueError("LAMOST catalog must provide obsid")
    if labels["source_id"].isna().all():
        raise ValueError("Label catalog must provide source_id, APOGEE_ID, or sobject_id")

    valid_label_coordinates = np.isfinite(labels["ra"]) & np.isfinite(labels["dec"])
    coordinate_label_indices = np.flatnonzero(valid_label_coordinates.to_numpy())
    label_coordinates = SkyCoord(
        ra=labels.loc[valid_label_coordinates, "ra"].to_numpy() * u.deg,
        dec=labels.loc[valid_label_coordinates, "dec"].to_numpy() * u.deg,
        frame="icrs",
    )
    label_maps: dict[str, dict[str, list[int]]] = {}
    for key_name in ("gaia_dr3_source_id", "gaia_dr2_source_id"):
        mapping: dict[str, list[int]] = {}
        for index, value in labels[key_name].items():
            key = _clean_identifier(value)
            if key is not None:
                mapping.setdefault(key, []).append(int(index))
        label_maps[key_name] = mapping

    matches: list[tuple[int, str, float, str] | None] = [None] * len(lamost)
    unmatched_indices: list[int] = []
    counts = {"gaia_dr3": 0, "gaia_dr2": 0, "sky": 0, "unmatched": 0, "ambiguous_id": 0}
    for row_index in range(len(lamost)):
        selected: tuple[int, str, float, str] | None = None
        key_order = (
            ("gaia_dr3_source_id", "gaia_dr3_id")
            if gaia_version in {"auto", "dr3"}
            else ("gaia_dr2_source_id", "gaia_dr2_id"),
        )
        if gaia_version == "auto":
            key_order = (
                ("gaia_dr3_source_id", "gaia_dr3_id"),
                ("gaia_dr2_source_id", "gaia_dr2_id"),
            )
        for key_name, method in key_order:
            key = _clean_identifier(lamost.at[row_index, key_name])
            candidates = label_maps[key_name].get(key or "", [])
            if not candidates:
                continue
            if len(candidates) > 1 and _finite_coordinates(lamost, row_index):
                candidate_table = labels.iloc[candidates].reset_index(drop=True)
                candidate_coordinates = SkyCoord(
                    candidate_table["ra"].fillna(0).to_numpy() * u.deg,
                    candidate_table["dec"].fillna(0).to_numpy() * u.deg,
                    frame="icrs",
                )
                nearest, separation, _ = SkyCoord(
                    float(lamost.at[row_index, "ra"]) * u.deg,
                    float(lamost.at[row_index, "dec"]) * u.deg,
                    frame="icrs",
                ).match_to_catalog_sky(candidate_coordinates)
                label_index = candidates[int(nearest)]
                separation_arcsec = float(separation.arcsec)
            elif len(candidates) == 1:
                label_index = candidates[0]
                separation_arcsec = float("nan")
            else:
                counts["ambiguous_id"] += 1
                continue
            selected = (label_index, method, separation_arcsec, key_name)
            counts[method.removesuffix("_id")] += 1
            break
        if selected is None:
            unmatched_indices.append(row_index)
        matches[row_index] = selected

    # Coordinate fallback is vectorized because a full LAMOST catalog can have
    # millions of rows. ID matches above remain authoritative even when their
    # coordinate separation deserves a manual conflict review.
    valid_unmatched = [index for index in unmatched_indices if _finite_coordinates(lamost, index)]
    if valid_unmatched and len(label_coordinates) > 0:
        unmatched_coordinates = SkyCoord(
            ra=lamost.loc[valid_unmatched, "ra"].to_numpy() * u.deg,
            dec=lamost.loc[valid_unmatched, "dec"].to_numpy() * u.deg,
            frame="icrs",
        )
        nearest, separations, _ = unmatched_coordinates.match_to_catalog_sky(label_coordinates)
        for position, row_index in enumerate(valid_unmatched):
            separation_arcsec = float(separations[position].arcsec)
            if separation_arcsec <= max_separation_arcsec:
                matches[row_index] = (
                    int(coordinate_label_indices[int(nearest[position])]),
                    "sky",
                    separation_arcsec,
                    "coordinates",
                )
                counts["sky"] += 1
    counts["unmatched"] = sum(match is None for match in matches)

    output_rows: list[dict[str, Any]] = []
    for row_index, match in enumerate(matches):
        if match is None:
            continue
        label_index, method, separation, match_key = match
        row = lamost.iloc[row_index].to_dict()
        row["lamost_source_id"] = row.get("source_id", "")
        row["source_id"] = labels.at[label_index, "source_id"]
        row["match_method"] = method
        row["match_key"] = match_key
        row["match_separation_arcsec"] = separation
        row["match_coordinate_conflict"] = bool(
            np.isfinite(separation) and separation > max_separation_arcsec
        )
        for name in LABEL_NAMES:
            row[name] = labels.at[label_index, name]
        for name in labels.columns:
            excluded = {
                *LABEL_NAMES,
                "source_id",
                "ra",
                "dec",
                "gaia_dr3_source_id",
                "gaia_dr2_source_id",
                "label_source",
            }
            if name not in excluded:
                row[f"label_{name}"] = labels.at[label_index, name]
        row["label_source"] = labels.at[label_index, "label_source"]
        output_rows.append(row)
    output = pd.DataFrame(output_rows)
    if not output.empty:
        output = output.drop_duplicates(subset=["obsid", "source_id"], keep="first")
    summary = {
        "lamost_rows": len(lamost),
        "label_rows": len(labels),
        "matched_rows": len(output),
        "unmatched_rows": counts["unmatched"],
        "gaia_dr3_matches": counts["gaia_dr3"],
        "gaia_dr2_matches": counts["gaia_dr2"],
        "sky_matches": counts["sky"],
        "ambiguous_id_rows": counts["ambiguous_id"],
        "max_separation_arcsec": max_separation_arcsec,
        "quality": quality,
        "quality_summary": quality_summary,
    }
    return output, summary


def write_candidate_table(
    lamost_catalog: str | Path,
    label_catalog: str | Path,
    output: str | Path,
    max_separation_arcsec: float = 1.0,
    gaia_version: str = "auto",
    quality: str = "none",
    limit: int | None = None,
) -> dict[str, Any]:
    table, summary = build_candidate_table(
        lamost_catalog=lamost_catalog,
        label_catalog=label_catalog,
        max_separation_arcsec=max_separation_arcsec,
        gaia_version=gaia_version,
        quality=quality,
        limit=limit,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    obsid_path = output_path.with_suffix(".obsid.txt")
    if "obsid" in table:
        obsids = table["obsid"].map(_clean_identifier).dropna().drop_duplicates()
        obsids.to_csv(obsid_path, index=False, header=False)
        summary["obsid_list"] = str(obsid_path)
    summary["output"] = str(output_path)
    return summary
