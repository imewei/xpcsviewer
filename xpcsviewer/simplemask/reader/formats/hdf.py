"""HDF5 scattering loader (APS NeXus-style ``/entry/data/data``)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

import h5py
import hdf5plugin  # noqa: F401  # registers HDF5 compression plugins
import numpy as np

from ..io_utils import average_frames_parallel, decode_counts
from .base import ScatteringDataset

logger = logging.getLogger(__name__)


class HdfDataset(ScatteringDataset):
    """Loader for 2-D or 3-D (frame, y, x) detector data stored in HDF5."""

    def __init__(
        self, fname: str, data_path: str = "/entry/data/data", **kwargs: Any
    ) -> None:
        super().__init__(fname)
        self.data_path: str = data_path
        with h5py.File(self.fname, "r") as f:
            if data_path not in f:
                raise KeyError(f"dataset {data_path!r} not found in {fname}")
            shape = f[data_path].shape
        self.ndim: int = len(shape)
        self.frame_count: int | None = None
        if self.ndim == 2:
            self.det_size = cast(tuple[int, int], tuple(shape))
        elif self.ndim == 3:
            if shape[0] == 0:
                raise ValueError(f"empty frame stack for {data_path!r}")
            self.frame_count = shape[0]
            self.det_size = cast(tuple[int, int], tuple(shape[1:]))
        else:
            raise ValueError(f"unexpected dataset rank {self.ndim} for {data_path!r}")

    def get_scattering(
        self,
        num_frames: int = -1,
        begin_idx: int = 0,
        num_processes: int | None = None,
        *,
        check_cancelled: Callable[[], None] | None = None,
    ) -> np.ndarray:
        """Read a scattering image, checking cancellation before single-process reads."""
        # A 2-D dataset is already a single image; there is nothing to average.
        if self.ndim == 2:
            if begin_idx != 0 or num_frames not in (-1, 0, 1):
                raise ValueError(
                    f"frame range (begin_idx={begin_idx}, num_frames={num_frames}) "
                    "is outside a single 2-D image"
                )
            with h5py.File(self.fname, "r") as f:
                if check_cancelled is not None:
                    check_cancelled()
                data = cast(np.ndarray, f[self.data_path][()])
            data = decode_counts(data)
            return data.astype(np.float32)
        return average_frames_parallel(
            self.fname,
            dataset_name=self.data_path,
            start_frame=begin_idx,
            num_frames=num_frames,
            chunk_size=32,
            num_processes=num_processes,
            check_cancelled=check_cancelled,
        )

    def get_frames_mean(
        self,
        begin_idx: int,
        end_idx: int,
        *,
        check_cancelled: Callable[[], None] | None = None,
    ) -> np.ndarray:
        """Return the mean of zero-based frames up to the exclusive, clamped end."""
        if self.frame_count is None:
            raise ValueError("frame ranges require a 3-D dataset")
        if end_idx <= begin_idx:
            raise ValueError("end_idx must be greater than begin_idx")
        return self.get_scattering(
            num_frames=end_idx - begin_idx,
            begin_idx=begin_idx,
            num_processes=1,
            check_cancelled=check_cancelled,
        )
