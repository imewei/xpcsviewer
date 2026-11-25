"""Unit tests for reader/io_utils.py frame-range resolution and averaging."""

import multiprocessing

import h5py
import numpy as np
import pytest

from xpcsviewer.simplemask.reader.io_utils import (
    _cast_to_signed,
    average_frames_parallel,
    resolve_frame_range,
)


class TestCastToSigned:
    def test_uint16_overflow_value_becomes_negative(self):
        # 65535 is the max uint16 value (detector saturation/overflow marker
        # upstream) -- bit-pattern-preserving cast to int16 must turn it
        # negative (-1), not clip or wrap to something else.
        arr = np.array([65535, 0, 100], dtype=np.uint16)
        result = _cast_to_signed(arr)
        assert result.dtype == np.int16
        np.testing.assert_array_equal(result, [-1, 0, 100])

    def test_signed_array_passthrough(self):
        arr = np.array([1, -2, 3], dtype=np.int32)
        result = _cast_to_signed(arr)
        assert result is arr


class TestResolveFrameRange:
    def test_positive_num_frames_clamped_to_end(self):
        assert (
            resolve_frame_range(total_frames=100, start_frame=90, num_frames=50) == 10
        )

    def test_positive_num_frames_within_range(self):
        assert resolve_frame_range(total_frames=100, start_frame=0, num_frames=20) == 20

    def test_zero_means_all_remaining(self):
        assert resolve_frame_range(total_frames=100, start_frame=30, num_frames=0) == 70

    def test_none_means_all_remaining(self):
        assert (
            resolve_frame_range(total_frames=100, start_frame=30, num_frames=None) == 70
        )

    def test_negative_means_representative_subset_floor(self):
        # total_frames // 5 = 200, max(1000, 200) = 1000
        assert (
            resolve_frame_range(total_frames=1000, start_frame=0, num_frames=-1) == 1000
        )

    def test_negative_means_representative_subset_fraction(self):
        # total_frames // 5 = 2000, max(1000, 2000) = 2000
        assert (
            resolve_frame_range(total_frames=10000, start_frame=0, num_frames=-1)
            == 2000
        )

    def test_negative_subset_clamped_to_remaining(self):
        # representative subset would be 1000, but only 40 frames remain
        assert (
            resolve_frame_range(total_frames=100, start_frame=60, num_frames=-1) == 40
        )

    def test_start_frame_out_of_range_raises(self):
        with pytest.raises(ValueError):
            resolve_frame_range(total_frames=100, start_frame=100, num_frames=10)

    def test_negative_start_frame_raises(self):
        with pytest.raises(ValueError):
            resolve_frame_range(total_frames=100, start_frame=-1, num_frames=10)


class TestAverageFramesParallel:
    def _write_stack(self, path, n_frames=5, shape=(4, 6), fill_value=None):
        with h5py.File(path, "w") as f:
            if fill_value is None:
                data = np.arange(
                    n_frames * shape[0] * shape[1], dtype=np.uint16
                ).reshape((n_frames, *shape))
            else:
                data = np.full((n_frames, *shape), fill_value, dtype=np.uint16)
            f.create_dataset("/entry/data/data", data=data)

    def test_averages_all_frames_by_default(self, tmp_path):
        fname = str(tmp_path / "stack.h5")
        self._write_stack(fname, n_frames=4, shape=(2, 2), fill_value=10)
        result = average_frames_parallel(fname, num_frames=0)
        assert result.shape == (2, 2)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, 10.0)

    def test_averages_frame_subset(self, tmp_path):
        fname = str(tmp_path / "stack.h5")
        with h5py.File(fname, "w") as f:
            # frame 0 is all zeros, frame 1 is all tens; average of just frame 1 is 10
            data = np.zeros((2, 3, 3), dtype=np.uint16)
            data[1] = 10
            f.create_dataset("/entry/data/data", data=data)
        result = average_frames_parallel(fname, start_frame=1, num_frames=1)
        np.testing.assert_allclose(result, 10.0)

    def test_averages_via_multiprocessing_pool_path(self, tmp_path):
        # num_frames (40) >= default chunk_size (32) exercises the
        # multiprocessing.Pool branch, not the single-process shortcut.
        fname = str(tmp_path / "big_stack.h5")
        self._write_stack(fname, n_frames=40, shape=(2, 2), fill_value=10)
        result = average_frames_parallel(fname, num_frames=40)
        assert result.shape == (2, 2)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, 10.0)

    def test_rejects_non_3d_dataset(self, tmp_path):
        fname = str(tmp_path / "flat.h5")
        with h5py.File(fname, "w") as f:
            f.create_dataset("/entry/data/data", data=np.zeros((4, 4)))
        with pytest.raises(ValueError):
            average_frames_parallel(fname)


needs_fork = pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="fork start method unavailable",
)


def _write_stack(path, stack):
    with h5py.File(path, "w") as f:
        f["/entry/data/data"] = stack


class TestAverageFramesPrecision:
    @needs_fork
    def test_multiprocess_matches_single_process(self, tmp_path, monkeypatch):
        from multiprocessing import get_context

        from xpcsviewer.simplemask.reader import io_utils

        # the sandbox forbids the default pool; fork works everywhere we run
        monkeypatch.setattr(io_utils, "Pool", get_context("fork").Pool)
        stack = np.full((64, 1, 1), 1_000_000, dtype=np.uint32)
        stack[31] = 1_000_002
        stack[63] = 1_000_001
        path = str(tmp_path / "stack.h5")
        _write_stack(path, stack)

        one = average_frames_parallel(path, num_frames=64, num_processes=1)
        two = average_frames_parallel(path, num_frames=64, num_processes=2)

        expected = np.float32(stack.mean(dtype=np.float64))  # 1000000.046875
        assert one.dtype == two.dtype == np.float32
        np.testing.assert_array_equal(one, expected)
        np.testing.assert_array_equal(two, expected)


class TestAverageFramesLogsRange:
    def test_subset_is_logged(self, tmp_path, caplog):
        path = str(tmp_path / "stack.h5")
        _write_stack(path, np.ones((6000, 2, 2), dtype=np.uint16))
        caplog.set_level("INFO", logger="xpcsviewer.simplemask.reader.io_utils")
        average_frames_parallel(path, num_frames=-1, num_processes=1)
        assert "Averaging frames [0, 1200) of 6000 (representative subset)" in (
            caplog.text
        )

    def test_full_range_is_logged_without_subset_note(self, tmp_path, caplog):
        path = str(tmp_path / "stack.h5")
        _write_stack(path, np.ones((40, 2, 2), dtype=np.uint16))
        caplog.set_level("INFO", logger="xpcsviewer.simplemask.reader.io_utils")
        average_frames_parallel(path, num_frames=0, num_processes=1)
        assert "Averaging frames [0, 40) of 40" in caplog.text
        assert "representative subset" not in caplog.text

    @needs_fork
    def test_pool_branch_is_logged(self, tmp_path, monkeypatch, caplog):
        from multiprocessing import get_context

        from xpcsviewer.simplemask.reader import io_utils

        monkeypatch.setattr(io_utils, "Pool", get_context("fork").Pool)
        path = str(tmp_path / "stack.h5")
        _write_stack(path, np.ones((64, 2, 2), dtype=np.uint16))
        caplog.set_level("INFO", logger="xpcsviewer.simplemask.reader.io_utils")
        average_frames_parallel(path, num_frames=64, num_processes=2)
        assert "Averaging frames [0, 64) of 64" in caplog.text
