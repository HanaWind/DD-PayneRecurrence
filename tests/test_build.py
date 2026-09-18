from __future__ import annotations

import pandas as pd

from ddpayne.data.build import _passes_header_quality, _rejection_record


def test_header_quality_rejects_low_snrg_with_explicit_reason():
    accepted, reason = _passes_header_quality(
        pd.Series({"class": "STAR", "snrg": 19.9}),
        {"require_class_star": True, "minimum_snr_g": 20.0},
    )

    assert not accepted
    assert reason == "SNRG is below 20"


def test_rejection_record_keeps_audit_identifiers():
    row = pd.Series({"obsid": "123", "source_id": "APOGEE-1", "snrg": 10.0, "class": "STAR"})

    record = _rejection_record(row, "spectra/example.fits.gz", "header_quality", "low SNR")

    assert record == {
        "obsid": "123",
        "source_id": "APOGEE-1",
        "path": "spectra/example.fits.gz",
        "snrg": 10.0,
        "class": "STAR",
        "stage": "header_quality",
        "reason": "low SNR",
    }
