"""Regression tests for review round 2 findings R2-01, R2-02 and R2-04."""

from __future__ import annotations

import os

import h5py
import numpy as np
import pytest

from xpcsviewer.fileIO.hdf_reader import HDF5ConnectionPool
from xpcsviewer.fileIO.hdf_reader_enhanced import EnhancedHDF5Reader
from xpcsviewer.io.hdf5_facade import HDF5Facade

pytestmark = pytest.mark.unit


def _write(path, value, extra=0):
    with h5py.File(path, "w") as f:
        f["x"] = np.full(3 + extra, value, dtype=float)


def _replace(path, value, extra=0):
    """Atomically replace ``path`` with a new file holding ``value``."""
    tmp = str(path) + ".new"
    _write(tmp, value, extra)
    os.replace(tmp, path)


class TestR201ReplacedFile:
    def test_pool_reopens_replaced_file(self, tmp_path):
        path = str(tmp_path / "a.h5")
        _write(path, 1.0)
        pool = HDF5ConnectionPool(max_pool_size=2)
        with pool.get_connection(path) as f:
            assert f["x"][0] == 1.0
        _replace(path, 2.0)
        with pool.get_connection(path) as f:
            assert f["x"][0] == 2.0
        pool.clear_pool()

    def test_batch_read_not_served_from_stale_handle_or_cache(self, tmp_path):
        path = str(tmp_path / "a.h5")
        _write(path, 1.0)
        pool = HDF5ConnectionPool(max_pool_size=2)
        assert pool.batch_read_datasets(path, ["x"])["x"][0] == 1.0
        _replace(path, 2.0)
        assert pool.batch_read_datasets(path, ["x"], use_cache=False)["x"][0] == 2.0
        assert pool.batch_read_datasets(path, ["x"])["x"][0] == 2.0
        pool.clear_pool()

    def test_borrowed_old_handle_survives_replacement(self, tmp_path):
        path = str(tmp_path / "a.h5")
        _write(path, 1.0)
        pool = HDF5ConnectionPool(max_pool_size=2)
        with pool.get_connection(path) as old:
            _replace(path, 2.0)
            with pool.get_connection(path) as new:
                assert new["x"][0] == 2.0
                assert old["x"][0] == 1.0  # still open while borrowed
            assert old["x"][0] == 1.0
        assert not old.id.valid  # closed after last release
        assert not pool._retired
        pool.clear_pool()

    def test_unchanged_file_reuses_connection(self, tmp_path):
        path = str(tmp_path / "a.h5")
        _write(path, 1.0)
        pool = HDF5ConnectionPool(max_pool_size=2)
        with pool.get_connection(path):
            pass
        with pool.get_connection(path):
            pass
        assert pool.stats.total_connections_created == 1
        assert pool.stats.total_connections_reused == 1
        pool.clear_pool()

    def test_missing_file_still_raises_open_error(self, tmp_path):
        path = str(tmp_path / "a.h5")
        _write(path, 1.0)
        pool = HDF5ConnectionPool(max_pool_size=2)
        with pool.get_connection(path):
            pass
        os.remove(path)
        with pytest.raises(OSError), pool.get_connection(path):
            pass
        pool.clear_pool()

    def test_enhanced_reader_cache_follows_file_revision(self, tmp_path):
        path = str(tmp_path / "a.h5")
        _write(path, 1.0)
        reader = EnhancedHDF5Reader(enable_read_ahead=False)
        assert reader.read_dataset(path, "x")[0] == 1.0
        _replace(path, 2.0, extra=1)
        data = reader.read_dataset(path, "x")
        assert data[0] == 2.0
        assert data.shape == (4,)

    def test_enhanced_reader_cache_still_hits_for_unchanged_file(self, tmp_path):
        path = str(tmp_path / "a.h5")
        _write(path, 1.0)
        reader = EnhancedHDF5Reader(enable_read_ahead=False)
        reader.read_dataset(path, "x")
        reader.read_dataset(path, "x")
        assert reader.stats["cache_hits"] == 1

    def test_enhanced_reader_deleted_file_raises(self, tmp_path):
        path = str(tmp_path / "a.h5")
        _write(path, 1.0)
        reader = EnhancedHDF5Reader(enable_read_ahead=False)
        reader.read_dataset(path, "x")
        os.remove(path)
        with pytest.raises(OSError):
            reader.read_dataset(path, "x")


class TestR202BatchCacheBound:
    def test_cache_buckets_are_globally_bounded(self, tmp_path):
        pool = HDF5ConnectionPool(max_pool_size=1)
        n_files = getattr(pool, "_read_cache_max_files", 20) + 10
        paths = []
        for i in range(n_files):
            p = str(tmp_path / f"f{i}.h5")
            _write(p, float(i))
            paths.append(p)
            assert pool.batch_read_datasets(p, ["x"])["x"][0] == float(i)
        assert len(pool._read_cache) == getattr(pool, "_read_cache_max_files", 20)
        # most recent files retained, oldest evicted
        assert paths[-1] in pool._read_cache
        assert paths[0] not in pool._read_cache
        pool.clear_pool()

    def test_recently_hit_bucket_is_kept(self, tmp_path):
        pool = HDF5ConnectionPool(max_pool_size=1)
        first = str(tmp_path / "first.h5")
        _write(first, 7.0)
        pool.batch_read_datasets(first, ["x"])
        for i in range(getattr(pool, "_read_cache_max_files", 20) - 1):
            p = str(tmp_path / f"f{i}.h5")
            _write(p, 0.0)
            pool.batch_read_datasets(p, ["x"])
        pool.batch_read_datasets(first, ["x"])  # refresh recency
        extra = str(tmp_path / "extra.h5")
        _write(extra, 0.0)
        pool.batch_read_datasets(extra, ["x"])
        assert first in pool._read_cache
        pool.clear_pool()

    def test_expired_entry_is_dropped_when_found(self, tmp_path, monkeypatch):
        import xpcsviewer.fileIO.hdf_reader as mod

        path = str(tmp_path / "a.h5")
        _write(path, 1.0)
        pool = HDF5ConnectionPool(max_pool_size=1)
        pool.batch_read_datasets(path, ["x"])
        for entry in pool._read_cache[path].values():
            entry["timestamp"] -= mod.CACHE_ENTRY_TIMEOUT + 1

        def boom(*_a, **_k):
            raise RuntimeError("no read")

        monkeypatch.setattr(pool, "get_connection", boom)
        with pytest.raises(RuntimeError):
            pool.batch_read_datasets(path, ["x"])
        assert not pool._read_cache.get(path)
        pool.clear_pool()


class TestR204FallbackQ:
    def _make(self, tmp_path):
        path = str(tmp_path / "g2.h5")
        with h5py.File(path, "w") as f:
            g = f.create_group("xpcs/g2")
            g["g2"] = np.arange(6, dtype=float).reshape(2, 3) + 1.0
            g["g2_err"] = np.full((2, 3), 0.01)
            g["delay_times"] = np.array([1.0, 2.0])
        return path

    def test_q_idx_with_fallback_q_values(self, tmp_path):
        facade = HDF5Facade(pool=HDF5ConnectionPool(max_pool_size=2))
        data = facade.read_g2_data(self._make(tmp_path), q_idx=1)
        assert data.g2.shape == (2, 1)
        assert list(data.q_values) == [1]

    def test_all_q_fallback_unchanged(self, tmp_path):
        facade = HDF5Facade(pool=HDF5ConnectionPool(max_pool_size=2))
        data = facade.read_g2_data(self._make(tmp_path))
        assert list(data.q_values) == [0, 1, 2]
