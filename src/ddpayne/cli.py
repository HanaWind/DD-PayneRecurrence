from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ddpayne.config import PROJECT_ROOT, project_path
from ddpayne.data.build import prepare_hdf5
from ddpayne.data.catalog import write_candidate_table
from ddpayne.data.lamost import read_lamost_spectrum
from ddpayne.data.manifest import build_manifest, iter_spectra
from ddpayne.data.matching import match_label_catalog
from ddpayne.inference.fit import infer_from_config
from ddpayne.smoke import run_smoke_test
from ddpayne.training.train import train_from_config


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def inspect_data(root: Path, limit: int) -> dict[str, Any]:
    paths = list(iter_spectra(root))
    sampled = paths[:limit]
    failures = []
    pixel_counts = []
    snrg = []
    for path in sampled:
        try:
            spectrum = read_lamost_spectrum(path)
            pixel_counts.append(len(spectrum.flux))
            snrg.append(spectrum.metadata["snrg"])
        except Exception as exc:
            failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    sizes = [path.stat().st_size for path in paths]
    return {
        "spectra_root": str(root.resolve()),
        "files": len(paths),
        "compressed_bytes": int(sum(sizes)),
        "sampled": len(sampled),
        "sample_failures": failures[:20],
        "pixel_counts": sorted(set(pixel_counts)),
        "sample_snrg_median": float(np.nanmedian(snrg)) if snrg else None,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="ddpayne")
    commands = root.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect-data", help="Inspect local LAMOST FITS files")
    inspect.add_argument("--spectra-root", required=True)
    inspect.add_argument("--limit", type=int, default=100)

    manifest = commands.add_parser("build-manifest", help="Build a FITS-header manifest")
    manifest.add_argument("--spectra-root", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--limit", type=int)

    matching = commands.add_parser("match-labels", help="Sky-match high-resolution labels")
    matching.add_argument("--manifest", required=True)
    matching.add_argument("--labels", required=True)
    matching.add_argument("--output", required=True)
    matching.add_argument("--max-separation-arcsec", type=float, default=1.0)

    candidates = commands.add_parser(
        "build-candidates",
        help="Match a LAMOST catalog to high-resolution labels and write candidate obsids",
    )
    candidates.add_argument("--lamost-catalog", required=True)
    candidates.add_argument("--labels", required=True)
    candidates.add_argument("--output", required=True)
    candidates.add_argument("--max-separation-arcsec", type=float, default=1.0)
    candidates.add_argument("--gaia-version", choices=["auto", "dr2", "dr3"], default="auto")
    candidates.add_argument(
        "--quality",
        choices=["none", "apogee"],
        default="none",
        help="Apply an explicit first-pass quality filter to the label catalog",
    )
    candidates.add_argument("--limit", type=int)

    prepare = commands.add_parser("prepare", help="Build a training HDF5 dataset")
    prepare.add_argument("--config", required=True)

    train = commands.add_parser("train", help="Train the DD-Payne spectral model")
    train.add_argument("--config", required=True)

    inference = commands.add_parser("infer", help="Fit stellar labels to a manifest")
    inference.add_argument("--config", required=True)

    smoke = commands.add_parser("smoke-test", help="Run a tiny synthetic end-to-end test")
    smoke.add_argument("--work-dir", default="outputs/smoke")
    return root


def main(argv: list[str] | None = None) -> None:
    arguments = parser().parse_args(argv)
    if arguments.command == "inspect-data":
        _print(inspect_data(project_path(arguments.spectra_root), arguments.limit))
    elif arguments.command == "build-manifest":
        success, failure = build_manifest(
            spectra_root=project_path(arguments.spectra_root),
            output=project_path(arguments.output),
            base=PROJECT_ROOT,
            limit=arguments.limit,
        )
        _print(
            {
                "output": str(project_path(arguments.output)),
                "success": success,
                "failure": failure,
            }
        )
    elif arguments.command == "match-labels":
        count = match_label_catalog(
            project_path(arguments.manifest),
            project_path(arguments.labels),
            project_path(arguments.output),
            arguments.max_separation_arcsec,
        )
        _print({"output": str(project_path(arguments.output)), "matches": count})
    elif arguments.command == "build-candidates":
        _print(
            write_candidate_table(
                lamost_catalog=project_path(arguments.lamost_catalog),
                label_catalog=project_path(arguments.labels),
                output=project_path(arguments.output),
                max_separation_arcsec=arguments.max_separation_arcsec,
                gaia_version=arguments.gaia_version,
                quality=arguments.quality,
                limit=arguments.limit,
            )
        )
    elif arguments.command == "prepare":
        _print(prepare_hdf5(project_path(arguments.config)))
    elif arguments.command == "train":
        _print(train_from_config(project_path(arguments.config)))
    elif arguments.command == "infer":
        _print(infer_from_config(project_path(arguments.config)))
    elif arguments.command == "smoke-test":
        _print(run_smoke_test(project_path(arguments.work_dir)))
    else:
        raise AssertionError(arguments.command)


if __name__ == "__main__":
    main()
