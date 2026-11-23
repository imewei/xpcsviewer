"""Read and regroup multi-tau G2 from APS HDF5 files."""

from collections.abc import Callable
from dataclasses import dataclass
from operator import index
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 - registers HDF5 compression filters
import numpy as np

from xpcsviewer.module.g2_regroup import G2Regrouping

from .aps_8idi import key

_NEXUS = key["nexus"]


@dataclass(frozen=True)
class G2RegroupResult:
    """Regrouped curves and the native delays and maps used to compute them."""

    source_path: Path
    qmap_path: Path
    delay_unit: str | bytes | None
    delay_list: np.ndarray
    delay_seconds: np.ndarray
    dynamic_bin_ids: np.ndarray
    normalized_g2: np.ndarray
    normalized_g2_err: np.ndarray
    static_roi_map: np.ndarray
    dynamic_roi_map: np.ndarray


def _numeric_dataset(source: h5py.File, path: str) -> h5py.Dataset:
    dataset = source.get(path)
    if not isinstance(dataset, h5py.Dataset) or dataset.dtype.kind not in "iuf":
        raise ValueError(f"Expected a real numeric dataset at {path}")
    return dataset


def _positive_scalar(
    source: h5py.File, path: str, *, integral: bool = False
) -> np.integer | np.floating:
    dataset = _numeric_dataset(source, path)
    if dataset.size != 1:
        raise ValueError(f"Expected one timing value at {path}")
    value = np.asarray(dataset[()]).reshape(-1)[0]
    if not np.isfinite(value) or value <= 0 or (integral and value % 1 != 0):
        requirement = "a positive integer" if integral else "a finite positive number"
        raise ValueError(f"Expected {requirement} at {path}")
    return value


def _read_delays(
    source: h5py.File, delay_count: int
) -> tuple[np.ndarray, np.ndarray, str | bytes | None]:
    dataset = _numeric_dataset(source, _NEXUS["tau"])
    if not delay_count or dataset.shape not in (
        (delay_count,),
        (1, delay_count),
        (delay_count, 1),
    ):
        raise ValueError(
            "delay_list must be a nonempty vector matching the G2 delay count"
        )
    delays = dataset[()].reshape(-1)
    if not np.isfinite(delays).all() or np.any(delays < 0) or not np.any(delays > 0):
        raise ValueError(
            "delay_list must be finite and nonnegative with a positive delay"
        )

    frame_time = _positive_scalar(source, _NEXUS["t0"])
    stride_path, average_path = _NEXUS["stride_frame"], _NEXUS["avg_frame"]
    if (stride_path in source) != (average_path in source):
        raise ValueError(
            "Incomplete timing metadata: both stride_frame and avg_frame are required together"
        )
    if stride_path in source:
        stride = _positive_scalar(source, stride_path, integral=True)
        average = _positive_scalar(source, average_path, integral=True)
    else:
        stride = average = np.int64(1)
    timing_dtype = np.result_type(np.float64, delays, frame_time, stride, average)
    with np.errstate(over="ignore", invalid="ignore"):
        seconds = delays * (timing_dtype.type(frame_time) * stride * average)
    if not np.isfinite(seconds).all():
        raise ValueError("Elapsed delay seconds must be finite")

    unit = dataset.attrs.get("units")
    return delays, seconds, unit if isinstance(unit, (str, bytes)) else None


def _read_roi_maps(
    qmap: h5py.File, detector_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    static_path, dynamic_path = _NEXUS["sqmap"], _NEXUS["dqmap"]
    for prefix in ("/xpcs", ""):
        static = qmap.get(static_path.removeprefix(prefix))
        dynamic = qmap.get(dynamic_path.removeprefix(prefix))
        if isinstance(static, h5py.Dataset) and isinstance(dynamic, h5py.Dataset):
            if static.shape != detector_shape or dynamic.shape != detector_shape:
                raise ValueError(
                    f"Both ROI maps must match detector shape {detector_shape}"
                )
            return static[()], dynamic[()]
    raise ValueError("Expected both ROI maps in /qmap or /xpcs/qmap")


def read_g2_regroup(
    source_path: Path,
    qmap_path: Path,
    *,
    delays_per_chunk: int = 1,
    check_cancelled: Callable[[], None] | None = None,
) -> G2RegroupResult:
    """Regroup APS multi-tau G2 using bounded reads and an external Q-map."""
    chunk_size = index(delays_per_chunk)
    if chunk_size <= 0:
        raise ValueError("delays_per_chunk must be positive")

    with h5py.File(source_path, "r") as source, h5py.File(qmap_path, "r") as qmap:
        cube = _numeric_dataset(source, _NEXUS["G2"])
        if cube.ndim != 4 or cube.shape[1] != 3:
            raise ValueError("unnormalized_G2 must have shape (delays, 3, y, x)")
        delay_count = cube.shape[0]
        delays, seconds, unit = _read_delays(source, delay_count)
        static_map, dynamic_map = _read_roi_maps(qmap, cube.shape[-2:])
        regrouping = G2Regrouping(static_map, dynamic_map)
        bin_ids = regrouping.dynamic_bin_ids
        g2 = np.empty((delay_count, bin_ids.size), dtype=np.float64)
        error = np.empty_like(g2)
        for start in range(0, delay_count, chunk_size):
            if check_cancelled is not None:
                check_cancelled()
            stop = min(start + chunk_size, delay_count)
            chunk = cube[start:stop]
            for delay_index in range(start, stop):
                try:
                    g2[delay_index], error[delay_index] = regrouping.compute_delay(
                        chunk[delay_index - start]
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid G2 at delay index {delay_index}: {exc}"
                    ) from exc
            del chunk

    return G2RegroupResult(
        source_path=source_path,
        qmap_path=qmap_path,
        delay_unit=unit,
        delay_list=delays,
        delay_seconds=seconds,
        dynamic_bin_ids=bin_ids,
        normalized_g2=g2,
        normalized_g2_err=error,
        static_roi_map=static_map,
        dynamic_roi_map=dynamic_map,
    )
