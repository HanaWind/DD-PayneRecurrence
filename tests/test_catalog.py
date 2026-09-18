from __future__ import annotations

import pandas as pd
import pytest

from ddpayne.data.catalog import build_candidate_table, read_catalog


def test_missing_lamost_placeholder_has_actionable_message(tmp_path):
    placeholder = tmp_path / "lamost_full_lrs_catalog.fits"

    with pytest.raises(FileNotFoundError, match="example placeholder"):
        read_catalog(placeholder)


def test_build_candidate_table_prefers_gaia_and_falls_back_to_sky(tmp_path):
    lamost = pd.DataFrame(
        {
            "obsid": [101, 102, 103],
            "gaia_source_id": ["9001", "", "9999"],
            "ra": [10.0, 20.0, 80.0],
            "dec": [30.0, 40.0, 50.0],
            "fitsname": ["a.fits.gz", "b.fits.gz", "c.fits.gz"],
        }
    )
    labels = pd.DataFrame(
        {
            "APOGEE_ID": ["apo-1", "apo-2"],
            "GAIAEDR3_SOURCE_ID": ["9001", "9002"],
            "RA": [10.0, 20.0],
            "DEC": [30.0, 40.0],
            "TEFF": [5000.0, 5100.0],
            "LOGG": [2.5, 3.0],
            "FE_H": [-1.0, -0.5],
            "C_FE": [0.1, 0.2],
        }
    )
    lamost_path = tmp_path / "lamost.csv"
    labels_path = tmp_path / "labels.csv"
    lamost.to_csv(lamost_path, index=False)
    labels.to_csv(labels_path, index=False)

    output, summary = build_candidate_table(lamost_path, labels_path, max_separation_arcsec=1.0)

    assert summary["matched_rows"] == 2
    assert summary["gaia_dr3_matches"] == 1
    assert summary["sky_matches"] == 1
    assert set(output["obsid"].astype(str)) == {"101", "102"}
    assert output.loc[output["obsid"].astype(str) == "101", "match_method"].item() == "gaia_dr3_id"
    assert output.loc[output["obsid"].astype(str) == "102", "match_method"].item() == "sky"
    assert output["path"].tolist() == ["a.fits.gz", "b.fits.gz"]
