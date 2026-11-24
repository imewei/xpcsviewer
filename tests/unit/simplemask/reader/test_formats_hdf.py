"""Unit tests for reader/formats/hdf.py and format dispatch."""

import h5py
import numpy as np
import pytest

from xpcsviewer.simplemask.reader import io_utils
from xpcsviewer.simplemask.reader.formats import get_format_loader
from xpcsviewer.simplemask.reader.formats.hdf import HdfDataset


@pytest.fixture
def preview_loader(tmp_path):
    fname = str(tmp_path / "preview.h5")
    with h5py.File(fname, "w") as f:
        f.create_dataset(
            "/entry/data/data",
            data=np.array([[[65535, 2]], [[5, 4]], [[7, 8]]], dtype=np.uint16),
            compression="gzip",
        )
    return HdfDataset(fname)


def test_hdf_preview_preserves_signed_counts_and_clamps_end(preview_loader):
    assert preview_loader.frame_count == 3
    np.testing.assert_array_equal(preview_loader.get_frames_mean(0, 1), [[-1, 2]])
    mean = preview_loader.get_frames_mean(1, 9)
    np.testing.assert_array_equal(mean, [[6, 6]])
    assert mean.dtype == np.float32


@pytest.mark.parametrize("begin_idx, end_idx", [(-1, 1), (3, 4), (1, 1)])
def test_hdf_preview_validates_frame_bounds(preview_loader, begin_idx, end_idx):
    with pytest.raises(ValueError):
        preview_loader.get_frames_mean(begin_idx, end_idx)


def test_hdf_single_image_has_no_frame_browser(tmp_path):
    fname = str(tmp_path / "img.h5")
    with h5py.File(fname, "w") as f:
        f.create_dataset("/entry/data/data", shape=(2, 2))
    loader = HdfDataset(fname)
    assert loader.frame_count is None
    with pytest.raises(ValueError, match="3-D"):
        loader.get_frames_mean(0, 1)


def test_hdf_frame_browser_rejects_empty_stack(tmp_path):
    fname = str(tmp_path / "empty.h5")
    with h5py.File(fname, "w") as f:
        f.create_dataset("/entry/data/data", shape=(0, 2, 2))
    with pytest.raises(ValueError):
        HdfDataset(fname)


def test_hdf_frame_browser_accepts_one_frame(tmp_path):
    fname = str(tmp_path / "one_frame.h5")
    with h5py.File(fname, "w") as f:
        f.create_dataset("/entry/data/data", shape=(1, 2, 2))
    assert HdfDataset(fname).frame_count == 1


def test_hdf_loader_accepts_uppercase_suffix(tmp_path):
    fname = str(tmp_path / "scan.H5")
    with h5py.File(fname, "w") as f:
        f.create_dataset("/entry/data/data", shape=(2, 2))
    assert isinstance(get_format_loader(fname), HdfDataset)


def test_hdf_single_process_reduces_bounded_chunks(tmp_path, monkeypatch):
    fname = str(tmp_path / "stack.h5")
    with h5py.File(fname, "w") as f:
        f.create_dataset("/entry/data/data", data=np.full((65, 1, 1), 5))

    reads = []
    original_getitem = h5py.Dataset.__getitem__

    def record_read(dataset, indices):
        reads.append(indices.stop - indices.start)
        return original_getitem(dataset, indices)

    def reject_pool(*args, **kwargs):
        pytest.fail("a single-process read started a process pool")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", record_read)
    monkeypatch.setattr(io_utils, "Pool", reject_pool)
    mean = io_utils.average_frames_parallel(fname, num_processes=1)
    assert max(reads) <= 32
    assert sum(reads) == 65
    np.testing.assert_array_equal(mean, [[5]])


def test_hdf_single_process_reduces_large_counts_before_rounding(tmp_path):
    fname = str(tmp_path / "large_counts.h5")
    with h5py.File(fname, "w") as f:
        f.create_dataset(
            "/entry/data/data", data=np.array([[[20_000_001]], [[0]], [[0]]], np.int32)
        )
    mean = io_utils.average_frames_parallel(fname, num_processes=1)
    assert mean.dtype == np.float32
    np.testing.assert_array_equal(mean, [[6_666_667]])


def test_hdf_single_process_rejects_negative_chunk_size(preview_loader):
    with pytest.raises(ValueError, match="chunk_size"):
        io_utils.average_frames_parallel(
            preview_loader.fname, chunk_size=-1, num_processes=1
        )


@pytest.mark.parametrize("preview", [False, True])
def test_hdf_preview_and_open_check_cancellation_before_each_read(
    tmp_path, monkeypatch, preview
):
    fname = str(tmp_path / "stack.h5")
    with h5py.File(fname, "w") as f:
        f.create_dataset("/entry/data/data", shape=(65, 1, 1))
    loader = HdfDataset(fname)

    events = []
    original_getitem = h5py.Dataset.__getitem__

    def record_read(dataset, indices):
        events.append(indices)
        return original_getitem(dataset, indices)

    def check_cancelled():
        events.append("check")
        if events.count("check") == 2:
            raise RuntimeError("cancelled")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", record_read)
    with pytest.raises(RuntimeError, match="cancelled"):
        if preview:
            loader.get_frames_mean(0, 65, check_cancelled=check_cancelled)
        else:
            loader.get_scattering(num_processes=1, check_cancelled=check_cancelled)
    assert events == ["check", slice(0, 32), "check"]


class TestHdfDataset:
    def test_2d_dataset_shape_and_dtype(self, tmp_path):
        fname = str(tmp_path / "img.h5")
        with h5py.File(fname, "w") as f:
            f.create_dataset(
                "/entry/data/data", data=np.full((5, 7), 3, dtype=np.uint16)
            )
        loader = HdfDataset(fname)
        assert loader.det_size == (5, 7)
        img = loader.get_scattering()
        assert img.shape == (5, 7)
        assert img.dtype == np.float32
        np.testing.assert_allclose(img, 3.0)

    def test_3d_dataset_shape_is_frame_shape(self, tmp_path):
        fname = str(tmp_path / "stack.h5")
        with h5py.File(fname, "w") as f:
            f.create_dataset(
                "/entry/data/data", data=np.zeros((10, 5, 7), dtype=np.uint16)
            )
        loader = HdfDataset(fname)
        assert loader.det_size == (5, 7)

    def test_3d_dataset_averages_frames(self, tmp_path):
        fname = str(tmp_path / "stack.h5")
        with h5py.File(fname, "w") as f:
            data = np.zeros((4, 2, 2), dtype=np.uint16)
            data[:] = 20
            f.create_dataset("/entry/data/data", data=data)
        loader = HdfDataset(fname)
        img = loader.get_scattering(num_frames=0)
        np.testing.assert_allclose(img, 20.0)

    def test_missing_dataset_path_raises(self, tmp_path):
        fname = str(tmp_path / "empty.h5")
        with h5py.File(fname, "w") as f:
            f.create_dataset("/wrong/path", data=np.zeros((2, 2)))
        with pytest.raises(KeyError):
            HdfDataset(fname)

    def test_custom_data_path(self, tmp_path):
        fname = str(tmp_path / "custom.h5")
        with h5py.File(fname, "w") as f:
            f.create_dataset(
                "/my/custom/path", data=np.full((3, 3), 7, dtype=np.uint16)
            )
        loader = HdfDataset(fname, data_path="/my/custom/path")
        img = loader.get_scattering()
        np.testing.assert_allclose(img, 7.0)

    def test_file_size_mb_positive(self, tmp_path):
        fname = str(tmp_path / "img.h5")
        with h5py.File(fname, "w") as f:
            f.create_dataset(
                "/entry/data/data", data=np.zeros((100, 100), dtype=np.uint16)
            )
        loader = HdfDataset(fname)
        assert loader.file_size_mb > 0


class TestGetFormatLoader:
    def test_h5_extension_returns_hdf_dataset(self, tmp_path):
        fname = str(tmp_path / "data.h5")
        with h5py.File(fname, "w") as f:
            f.create_dataset("/entry/data/data", data=np.zeros((2, 2)))
        loader = get_format_loader(fname)
        assert isinstance(loader, HdfDataset)

    def test_hdf_extension_returns_hdf_dataset(self, tmp_path):
        fname = str(tmp_path / "data.hdf")
        with h5py.File(fname, "w") as f:
            f.create_dataset("/entry/data/data", data=np.zeros((2, 2)))
        loader = get_format_loader(fname)
        assert isinstance(loader, HdfDataset)

    def test_hdf5_extension_returns_hdf_dataset(self, tmp_path):
        # .hdf5 is one of the extensions offered by the "Open Raw File"
        # dialog filter (Task 9) -- it must actually be accepted here.
        fname = str(tmp_path / "data.hdf5")
        with h5py.File(fname, "w") as f:
            f.create_dataset("/entry/data/data", data=np.zeros((2, 2)))
        loader = get_format_loader(fname)
        assert isinstance(loader, HdfDataset)

    def test_unrecognized_extension_raises(self, tmp_path):
        fname = str(tmp_path / "data.xyz")
        with pytest.raises(ValueError, match="unsupported dataset file"):
            get_format_loader(fname)
