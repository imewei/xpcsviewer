"""Qt worker that loads a raw detector file off the GUI thread.

Runs in a QThreadPool via xpcsviewer.threading.async_workers.BaseAsyncWorker
-- see xpcsviewer/threading/async_workers.py:152-398 for the run()/signals
contract this relies on (signals.finished / signals.error are emitted
automatically by the base class; this file only implements do_work()).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from xpcsviewer.simplemask.qmap import compute_qmap
from xpcsviewer.simplemask.reader import get_reader
from xpcsviewer.simplemask.reader.adapter import to_kernel_metadata
from xpcsviewer.simplemask.reader.exceptions import RawDataReadError
from xpcsviewer.simplemask.reader.formats import HdfDataset, get_format_loader
from xpcsviewer.threading.async_workers import BaseAsyncWorker


class RawFileLoadResult(TypedDict):
    """Shape of the dict RawFileLoadWorker.do_work() returns on success."""

    scattering: np.ndarray
    metadata: dict[str, Any]
    metadata_is_placeholder: bool
    source_path: str
    frame_count: int | None
    request_id: str
    qmap_result: tuple[dict[str, np.ndarray], dict[str, str]]


class RawFramePreviewResult(TypedDict):
    """Preview pixels and the zero-based, half-open range read from the source."""

    source_path: str
    request_id: int
    begin_idx: int
    end_idx: int
    image: np.ndarray


class RawFramePreviewWorker(BaseAsyncWorker):
    """Read an average of the requested detector frames without loading geometry."""

    def __init__(self, path: str, begin_idx: int, count: int, request_id: int) -> None:
        super().__init__(f"raw_preview_{request_id}")
        self.source_path = path
        self.begin_idx = begin_idx
        self.count = count
        self.request_id = request_id

    def do_work(self) -> RawFramePreviewResult:
        """Read the requested frames up to the last available frame."""
        self.check_cancelled()
        loader = get_format_loader(self.source_path)
        if not isinstance(loader, HdfDataset) or loader.frame_count is None:
            raise ValueError("frame preview requires an HDF5 frame stack")
        end_idx = min(self.begin_idx + self.count, loader.frame_count)
        image = loader.get_frames_mean(
            self.begin_idx, end_idx, check_cancelled=self.check_cancelled
        )
        return {
            "source_path": self.source_path,
            "request_id": self.request_id,
            "begin_idx": self.begin_idx,
            "end_idx": end_idx,
            "image": image,
        }


class RawFileLoadWorker(BaseAsyncWorker):
    """Loads a raw detector file and adapts it to SimpleMaskKernel's schema.

    On success, ``signals.finished`` carries a :class:`RawFileLoadResult`.
    On failure, ``signals.error`` carries the wrapped :class:`RawDataReadError`
    message and traceback (see ``BaseAsyncWorker.run()``).
    """

    def __init__(
        self,
        path: str,
        beamline: str,
        worker_id: str | None = None,
        *,
        begin_idx: int = 0,
        num_frames: int = -1,
    ) -> None:
        super().__init__(worker_id)
        self.path = path
        self.source_path = str(Path(path).absolute())
        self.beamline = beamline
        self.begin_idx = begin_idx
        self.num_frames = num_frames

    def do_work(self) -> RawFileLoadResult:
        """Load the requested average and validate its geometry before installation."""
        self.emit_status(f"Reading {self.path}...")
        try:
            self.check_cancelled()
            reader = get_reader(self.beamline, self.source_path)
            reader.prepare_data(
                begin_idx=self.begin_idx,
                num_frames=self.num_frames,
                num_processes=1,
                check_cancelled=self.check_cancelled,
            )
            self.check_cancelled()
            assert reader.metadata is not None
            scattering = reader.scat
            if (
                not isinstance(scattering, np.ndarray)
                or scattering.ndim != 2
                or not scattering.size
            ):
                raise ValueError("expected a nonempty 2-D scattering image")
            metadata = to_kernel_metadata(reader.metadata, reader.stype)
            if metadata["shape"] != scattering.shape:
                raise ValueError(
                    f"geometry shape {metadata['shape']} does not match "
                    f"scattering image shape {scattering.shape}"
                )
            for key, name in (
                ("det_dist", "detector_distance"),
                ("pix_dim", "pixel_size"),
                ("energy", "energy"),
            ):
                value = metadata[key]
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f"{name} must be finite and positive")
            for key, name in (("bcx", "beam_center_x"), ("bcy", "beam_center_y")):
                if not np.isfinite(metadata[key]):
                    raise ValueError(f"{name} must be finite")
            qmap_result = compute_qmap(reader.stype, metadata)
            frame_count = reader.frame_count
        except InterruptedError:
            raise
        except Exception as exc:
            raise RawDataReadError(self.path, exc) from exc

        self.emit_status("Read complete")
        return {
            "scattering": scattering,
            "metadata": metadata,
            "metadata_is_placeholder": reader.metadata_is_placeholder,
            "source_path": self.source_path,
            "frame_count": frame_count,
            "request_id": self.worker_id,
            "qmap_result": qmap_result,
        }
