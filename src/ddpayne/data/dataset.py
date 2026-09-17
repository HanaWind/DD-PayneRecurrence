from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}


class H5SpectraDataset(Dataset[dict[str, Any]]):
    def __init__(self, path: str | Path, split: str = "train") -> None:
        if split not in SPLIT_CODES:
            raise ValueError(f"Unknown split: {split}")
        self.path = str(Path(path).resolve())
        self.split = split
        self._handle: h5py.File | None = None
        with h5py.File(self.path, "r") as handle:
            self.indices = np.flatnonzero(handle["split"][:] == SPLIT_CODES[split])
            self.wavelength = handle["wavelength"][:]
            self.label_names = json.loads(handle.attrs["label_names"])
        if len(self.indices) == 0:
            raise ValueError(f"Split {split!r} is empty in {self.path}")

    def _file(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = int(self.indices[index])
        handle = self._file()
        return {
            "flux": torch.from_numpy(handle["flux"][row]),
            "ivar": torch.from_numpy(handle["ivar"][row]),
            "labels": torch.from_numpy(handle["labels"][row]),
            "row": row,
        }

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state
