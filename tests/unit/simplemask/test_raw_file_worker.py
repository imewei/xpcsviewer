"""Unit tests for raw_file_worker.py -- run do_work() directly, no QThreadPool."""

from pathlib import Path
from unittest.mock import Mock, patch

import h5py
import numpy as np
import pytest

from xpcsviewer.simplemask.raw_file_worker import RawFileLoadWorker
from xpcsviewer.simplemask.reader import io_utils
from xpcsviewer.simplemask.reader.exceptions import RawDataReadError
from xpcsviewer.simplemask.reader.formats import hdf


class _FakeReader:
    """Stand-in for a FileReader subclass, avoiding real file IO in this test."""

    frame_count = 3

    def __init__(self, fname):
        self.fname = fname
        self.stype = "Transmission"
        self.metadata_is_placeholder = False
        self.scat = None
        self.metadata = None

    def prepare_data(self, **kwargs):
        self.scat = np.full((4, 4), 3.0, dtype=np.float32)
        self.metadata = {
            "energy": 10.0,
            "beam_center_x": 2.0,
            "beam_center_y": 2.0,
            "detector_shape_x": 4,
            "detector_shape_y": 4,
            "pixel_size": 75e-6,
            "detector_distance": 5.0,
        }


class _FakeFailingReader(_FakeReader):
    def prepare_data(self, **kwargs):
        raise RuntimeError("simulated read failure")


def test_do_work_returns_scattering_and_adapted_metadata():
    with patch(
        "xpcsviewer.simplemask.raw_file_worker.get_reader",
        return_value=_FakeReader("scan.h5"),
    ):
        worker = RawFileLoadWorker("scan.h5", "APS_8IDI")
        result = worker.do_work()

    assert result["scattering"].shape == (4, 4)
    assert result["metadata"]["bcx"] == pytest.approx(2.0)
    assert result["metadata"]["pix_dim"] == pytest.approx(0.075)
    assert result["metadata_is_placeholder"] is False


def test_do_work_wraps_failure_as_raw_data_read_error():
    with patch(
        "xpcsviewer.simplemask.raw_file_worker.get_reader",
        return_value=_FakeFailingReader("bad.h5"),
    ):
        worker = RawFileLoadWorker("bad.h5", "APS_8IDI")
        with pytest.raises(RawDataReadError) as exc_info:
            worker.do_work()

    assert exc_info.value.path == "bad.h5"


def test_do_work_wraps_adapter_failure_as_raw_data_read_error():
    # to_kernel_metadata() raising (e.g. a missing required field) must also
    # cross the worker boundary as RawDataReadError, not an unwrapped
    # KeyError -- this was a real bug caught in review: adaptation used to
    # run outside the try block.
    class _IncompleteMetadataReader(_FakeReader):
        def prepare_data(self, **kwargs):
            self.scat = np.zeros((2, 2), dtype=np.float32)
            self.metadata = {"energy": 10.0}  # missing every other required field

    with patch(
        "xpcsviewer.simplemask.raw_file_worker.get_reader",
        return_value=_IncompleteMetadataReader("incomplete.h5"),
    ):
        worker = RawFileLoadWorker("incomplete.h5", "APS_8IDI")
        with pytest.raises(RawDataReadError) as exc_info:
            worker.do_work()

    assert exc_info.value.path == "incomplete.h5"
    assert isinstance(exc_info.value.cause, KeyError)


def test_do_work_propagates_placeholder_flag():
    class _PlaceholderReader(_FakeReader):
        def prepare_data(self, **kwargs):
            super().prepare_data(**kwargs)
            self.metadata_is_placeholder = True

    with patch(
        "xpcsviewer.simplemask.raw_file_worker.get_reader",
        return_value=_PlaceholderReader("scan.h5"),
    ):
        worker = RawFileLoadWorker("scan.h5", "APS_8IDI")
        result = worker.do_work()

    assert result["metadata_is_placeholder"] is True


def test_raw_worker_passes_range_and_returns_stack_metadata():
    reader = _FakeReader("scan.h5")
    with (
        patch("xpcsviewer.simplemask.raw_file_worker.get_reader", return_value=reader),
        patch.object(reader, "prepare_data", wraps=reader.prepare_data) as prepare,
    ):
        worker = RawFileLoadWorker(
            "scan.h5", "APS_8IDI", "sample", begin_idx=1, num_frames=20
        )
        result = worker.do_work()

    prepare.assert_called_once_with(
        begin_idx=1,
        num_frames=20,
        num_processes=1,
        check_cancelled=worker.check_cancelled,
    )
    assert result["frame_count"] == 3
    assert Path(result["source_path"]).is_absolute()
    assert result["request_id"] == "sample"
    assert result["qmap_result"][0]["q"].shape == (4, 4)


def test_raw_worker_default_retains_representative_pick():
    reader = _FakeReader("scan.h5")
    with (
        patch("xpcsviewer.simplemask.raw_file_worker.get_reader", return_value=reader),
        patch.object(reader, "prepare_data", wraps=reader.prepare_data) as prepare,
    ):
        worker = RawFileLoadWorker("scan.h5", "APS_8IDI")
        worker.do_work()

    prepare.assert_called_once_with(
        begin_idx=0,
        num_frames=-1,
        num_processes=1,
        check_cancelled=worker.check_cancelled,
    )


@pytest.fixture
def frame_stack(tmp_path):
    path = tmp_path / "scan.H5"
    with h5py.File(path, "w") as file:
        file["/entry/data/data"] = np.arange(1001, dtype=np.uint16).reshape(-1, 1, 1)
    return str(path)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, 499.5),
        ({"num_frames": 0}, 500.0),
        ({"begin_idx": 999, "num_frames": 20}, 999.5),
        ({"begin_idx": 999, "num_frames": 0}, 999.5),
    ],
)
def test_raw_worker_averages_requested_frames(frame_stack, kwargs, expected):
    result = RawFileLoadWorker(frame_stack, "APS_8IDI", **kwargs).do_work()
    np.testing.assert_array_equal(result["scattering"], [[expected]])
    assert result["frame_count"] == 1001


@pytest.mark.parametrize("cancel", [False, True], ids=["all", "cancel"])
def test_raw_worker_all_frames_use_one_bounded_reduction(
    frame_stack, monkeypatch, cancel
):
    worker = RawFileLoadWorker(frame_stack, "APS_8IDI", num_frames=0)
    opened = []
    slices = []
    open_file = h5py.File
    get_slice = h5py.Dataset.__getitem__
    average_frames = hdf.average_frames_parallel

    def record_open(*args, **kwargs):
        handle = open_file(*args, **kwargs)
        opened.append(handle)
        return handle

    def record_slice(dataset, index):
        data = get_slice(dataset, index)
        slices.append(index)
        if cancel:
            worker.cancel()
        return data

    def inspect_reduction(*args, **kwargs):
        with monkeypatch.context() as reduction:
            reduction.setattr(h5py, "File", record_open)
            reduction.setattr(h5py.Dataset, "__getitem__", record_slice)
            reduction.setattr(
                io_utils, "Pool", Mock(side_effect=AssertionError("pool started"))
            )
            return average_frames(*args, **kwargs)

    monkeypatch.setattr(hdf, "average_frames_parallel", inspect_reduction)
    finished, cancelled, errors = [], [], []
    worker.signals.finished.connect(finished.append)
    worker.signals.cancelled.connect(lambda *args: cancelled.append(args))
    worker.signals.error.connect(lambda *args: errors.append(args))
    worker.run()

    assert len(opened) == 1
    assert not opened[0].id.valid
    assert not errors
    if cancel:
        assert slices == [slice(0, 32)]
        assert cancelled[0][0] == worker.worker_id
        assert not finished
    else:
        assert len(slices) == 32
        assert all(0 < interval.stop - interval.start <= 32 for interval in slices)
        assert not cancelled
        np.testing.assert_array_equal(finished[0]["scattering"], [[500.0]])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("detector_distance", 0.0),
        ("pixel_size", -1.0),
        ("energy", np.inf),
        ("beam_center_x", np.nan),
    ],
)
def test_raw_worker_rejects_invalid_geometry_before_qmap(field, value):
    class InvalidGeometryReader(_FakeReader):
        def prepare_data(self, **kwargs):
            super().prepare_data(**kwargs)
            self.metadata[field] = value

    with (
        patch(
            "xpcsviewer.simplemask.raw_file_worker.get_reader",
            return_value=InvalidGeometryReader("bad.h5"),
        ),
        patch("xpcsviewer.simplemask.qmap.compute_transmission_qmap") as compute,
        pytest.raises(RawDataReadError, match=field),
    ):
        RawFileLoadWorker("bad.h5", "APS_8IDI").do_work()

    compute.assert_not_called()


def test_raw_worker_rejects_geometry_shape_mismatch_before_qmap():
    class MismatchedShapeReader(_FakeReader):
        def prepare_data(self, **kwargs):
            super().prepare_data(**kwargs)
            self.metadata["detector_shape_x"] = 3

    with (
        patch(
            "xpcsviewer.simplemask.raw_file_worker.get_reader",
            return_value=MismatchedShapeReader("bad.h5"),
        ),
        patch(
            "xpcsviewer.simplemask.raw_file_worker.compute_qmap",
            side_effect=AssertionError("Q-map should not be computed"),
        ),
        pytest.raises(RawDataReadError, match="geometry shape.*scattering image shape"),
    ):
        RawFileLoadWorker("bad.h5", "APS_8IDI").do_work()
