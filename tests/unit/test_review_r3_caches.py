"""Regression tests for round 3 cache findings R3-02/03/05/23/31/33.

Invariant: data cached for a path is only served while the file at that path
is the revision it was read from, and no cache keeps data alive without bound.
"""

from __future__ import annotations

import gc
import os

import h5py
import numpy as np
import pytest

from xpcsviewer import file_locator
from xpcsviewer.fileIO import qmap_utils
from xpcsviewer.fileIO.hdf_reader import HDF5ConnectionPool
from xpcsviewer.fileIO.hdf_reader_enhanced import EnhancedHDF5Reader, ReadAheadCache
from xpcsviewer.fileIO.qmap_utils import QMapManager
from xpcsviewer.module import twotime_utils

pytestmark = pytest.mark.unit


def _write(path, value, extra=0):
    with h5py.File(path, "w") as f:
        f["x"] = np.full(3 + extra, value, dtype=float)


def _replace(path, value, extra=0):
    """Atomically replace ``path`` with a new file holding ``value``."""
    tmp = str(path) + ".new"
    _write(tmp, value, extra)
    os.replace(tmp, path)


def _write_c2(path, value, n=8, nq=1):
    """Minimal two-time file: one C2 half-matrix plus G2 datasets."""
    key = twotime_utils.key_map_nexus
    with h5py.File(path, "w") as f:
        f[key["c2_processed_bins"]] = np.arange(1, nq + 1)
        for q in range(nq):
            f[f"{key['c2_prefix']}/c2_{q + 1:05d}"] = np.full(
                (n, n), value, dtype=np.float32
            )
        f[key["c2_g2"]] = np.full((5, nq), value)
        f[key["c2_g2_segments"]] = np.full((5, 2, nq), value)


def _replace_c2(path, value, **kw):
    tmp = str(path) + ".new"
    _write_c2(tmp, value, **kw)
    os.replace(tmp, path)


@pytest.fixture(autouse=True)
def _clean_caches():
    twotime_utils.get_single_c2_from_hdf.cache_clear()
    twotime_utils.get_c2_g2partials_from_hdf.cache_clear()
    yield
    twotime_utils.get_single_c2_from_hdf.cache_clear()
    twotime_utils.get_c2_g2partials_from_hdf.cache_clear()


# --------------------------------------------------------------------- R3-02
class TestR302EnhancedReader:
    def test_borrowed_stale_handle_not_reused_after_replace(self, tmp_path):
        path = str(tmp_path / "a.h5")
        _write(path, 1.0)
        reader = EnhancedHDF5Reader(enable_read_ahead=False)
        with reader.get_file_connection(path) as old:
            assert old["x"][0] == 1.0
            _replace(path, 2.0)
            assert reader.read_dataset(path, "x")[0] == 2.0
            assert old["x"][0] == 1.0  # the borrower keeps its own revision
        assert reader.read_dataset(path, "x")[0] == 2.0

    def test_replace_between_read_and_cache_insert_is_not_cached(
        self, tmp_path, monkeypatch
    ):
        path = str(tmp_path / "a.h5")
        _write(path, 3.0)
        reader = EnhancedHDF5Reader(enable_read_ahead=False)
        orig = ReadAheadCache.cache_data

        def replacing(self, *args, **kwargs):
            _replace(path, 4.0)
            return orig(self, *args, **kwargs)

        monkeypatch.setattr(ReadAheadCache, "cache_data", replacing)
        assert reader.read_dataset(path, "x")[0] == 3.0  # bytes of the old file
        monkeypatch.setattr(ReadAheadCache, "cache_data", orig)
        assert reader.read_dataset(path, "x")[0] == 4.0

    def test_dead_revisions_do_not_accumulate(self, tmp_path):
        path = str(tmp_path / "a.h5")
        _write(path, 0.0)
        reader = EnhancedHDF5Reader(enable_read_ahead=False)
        for i in range(1, 6):
            _replace(path, float(i))
            assert reader.read_dataset(path, "x")[0] == float(i)
        assert len(reader.cache.cache) == 1

    def test_access_pattern_history_is_bounded(self, tmp_path):
        cache = ReadAheadCache()
        for i in range(1000):
            cache.record_access(f"/f{i}.h5", "x", None)
        assert len(cache.access_patterns) <= 256


# --------------------------------------------------------------------- R3-03
class TestR303TwoTimeCaches:
    def test_c2_and_g2_follow_atomic_replacement(self, tmp_path):
        path = str(tmp_path / "c2.h5")
        _write_c2(path, 1.0)
        first = twotime_utils.get_single_c2_from_hdf(path, 0, 512, 1, False)
        assert first["g2_full"][0] == 1.0
        assert twotime_utils.get_c2_g2partials_from_hdf(path)["g2_full"][0, 0] == 1.0
        _replace_c2(path, 2.0)
        second = twotime_utils.get_single_c2_from_hdf(path, 0, 512, 1, False)
        assert second["c2_mat"][0, 1] == 4.0  # half + half.T
        assert second["g2_full"][0] == 2.0
        assert twotime_utils.get_c2_g2partials_from_hdf(path)["g2_full"][0, 0] == 2.0

    def test_unchanged_file_is_served_from_cache(self, tmp_path):
        path = str(tmp_path / "c2.h5")
        _write_c2(path, 1.0)
        a = twotime_utils.get_single_c2_from_hdf(path, 0, 512, 1, False)
        b = twotime_utils.get_single_c2_from_hdf(path, 0, 512, 1, False)
        assert a is b

    def test_old_revisions_are_purged_on_insert(self, tmp_path):
        path = str(tmp_path / "c2.h5")
        _write_c2(path, 1.0)
        twotime_utils.get_single_c2_from_hdf(path, 0, 512, 1, False)
        _replace_c2(path, 2.0)
        twotime_utils.get_single_c2_from_hdf(path, 0, 512, 1, False)
        assert len(twotime_utils._c2_cache) == 1


# --------------------------------------------------------------------- R3-31
class TestR331OwnedDisplayArray:
    def test_downsampled_c2_is_owned_and_display_sized(self, tmp_path):
        path = str(tmp_path / "c2.h5")
        _write_c2(path, 1.0, n=256)
        res = twotime_utils.get_single_c2_from_hdf(path, 0, 32, 1, False)
        c2 = res["c2_mat"]
        assert c2.shape == (32, 32)
        assert c2.base is None  # not a view pinning the 256x256 matrix
        assert c2.nbytes == 32 * 32 * 4

    def test_strided_read_matches_full_reconstruction(self, tmp_path):
        path = str(tmp_path / "c2.h5")
        n = 100
        rng = np.random.default_rng(0)
        half = np.triu(rng.random((n, n), dtype=np.float32))
        key = twotime_utils.key_map_nexus
        with h5py.File(path, "w") as f:
            f[key["c2_processed_bins"]] = np.array([1])
            f[f"{key['c2_prefix']}/c2_00001"] = half
            f[key["c2_g2"]] = np.ones((5, 1))
            f[key["c2_g2_segments"]] = np.ones((5, 2, 1))
        res = twotime_utils.get_single_c2_from_hdf(path, 0, 30, 1, False)
        full = twotime_utils._reconstruct_c2_matrix_vectorized(half)
        rate = (n + 30 - 1) // 30
        np.testing.assert_array_equal(res["c2_mat"], full[::rate, ::rate])
        assert res["delta_t"] == rate

    def test_c2_cache_is_bounded_by_bytes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(twotime_utils._c2_cache, "max_bytes", 3 * 64 * 64 * 4)
        paths = []
        for i in range(6):
            p = str(tmp_path / f"c2_{i}.h5")
            _write_c2(p, float(i), n=64)
            paths.append(p)
            twotime_utils.get_single_c2_from_hdf(p, 0, 512, 1, False)
        assert twotime_utils._c2_cache.nbytes <= 3 * 64 * 64 * 4 + 4096
        assert len(twotime_utils._c2_cache) < 6


# --------------------------------------------------------------------- R3-05
class TestR305ExpiredBatchEntries:
    def test_expired_entries_of_other_files_are_purged_on_insert(self, tmp_path):
        pool = HDF5ConnectionPool(max_pool_size=4)
        a, b = str(tmp_path / "a.h5"), str(tmp_path / "b.h5")
        _write(a, 1.0)
        _write(b, 2.0)
        pool.batch_read_datasets(a, ["x"])
        for entry in pool._read_cache[a].values():
            entry["timestamp"] -= 10_000  # long expired
        pool.batch_read_datasets(b, ["x"])
        assert a not in pool._read_cache
        assert b in pool._read_cache
        pool.clear_pool()


# --------------------------------------------------------------------- R3-23
class _FakeXf:
    def __init__(self, fname):
        self.fname = fname
        with h5py.File(fname, "r") as f:
            self.value = float(f["x"][0])
        self.fit_summary = None
        self.atype = ["Multitau"]


@pytest.fixture
def locator(tmp_path, monkeypatch):
    calls = []

    def fake_create(fname, **kwargs):
        calls.append(fname)
        try:
            return _FakeXf(fname)
        except OSError:
            return None

    monkeypatch.setattr(file_locator, "create_xpcs_dataset", fake_create)
    loc = file_locator.FileLocator(str(tmp_path))
    loc.calls = calls
    return loc


class TestR323Locator:
    def test_replaced_file_gives_new_revision(self, tmp_path, locator):
        path = str(tmp_path / "a.h5")
        _write(path, 1.0)
        locator.build(str(tmp_path))
        locator.add_target(["a.h5"])
        assert locator.get_xf_list()[0].value == 1.0
        _replace(path, 2.0)
        locator.build(str(tmp_path))  # what GUI Reload does
        assert locator.get_xf_list()[0].value == 2.0

    def test_replaced_file_detected_without_build(self, tmp_path, locator):
        path = str(tmp_path / "a.h5")
        _write(path, 1.0)
        locator.add_target(["a.h5"])
        _replace(path, 2.0)
        assert locator.get_xf_list()[0].value == 2.0

    def test_unchanged_file_is_cached(self, tmp_path, locator):
        path = str(tmp_path / "a.h5")
        _write(path, 1.0)
        locator.add_target(["a.h5"])
        locator.get_xf_list()
        locator.get_xf_list()
        assert len(locator.calls) == 1

    def test_failed_load_sentinel_expires_on_new_revision(self, tmp_path, locator):
        path = str(tmp_path / "a.h5")
        with open(path, "wb") as f:
            f.write(b"not hdf5")
        locator.add_target(["a.h5"], preload=False)
        assert locator.get_xf_list() == []
        assert locator.get_xf_list() == []
        assert len(locator.calls) == 1  # broken file is not re-read each time
        _replace(path, 5.0)
        got = locator.get_xf_list()
        assert [x.value for x in got] == [5.0]

    def test_empty_target_fallback_does_not_serve_replaced_file(
        self, tmp_path, locator
    ):
        path = str(tmp_path / "a.h5")
        _write(path, 1.0)
        locator.add_target(["a.h5"])
        locator.target.clear()  # target list temporarily empty, cache populated
        _replace(path, 2.0)
        assert [x.value for x in locator.get_xf_list()] == []


class _FakeQMap:
    created = 0

    def __init__(self, fname=None):
        type(self).created += 1
        self.fname = fname
        with h5py.File(fname, "r") as f:
            self.value = float(f["x"][0])


@pytest.fixture
def fake_qmap(monkeypatch):
    _FakeQMap.created = 0
    monkeypatch.setattr(qmap_utils, "QMap", _FakeQMap)
    return _FakeQMap


def _write_hashed(path, value, hash_value):
    with h5py.File(path, "w") as f:
        f["x"] = np.full(3, value)
        f.create_group("/xpcs/qmap").attrs["hash"] = hash_value


class TestR323QMapKey:
    def test_hashless_key_includes_file_revision(self, tmp_path, fake_qmap):
        path = str(tmp_path / "a.h5")
        _write(path, 1.0)
        mgr = QMapManager()
        first = mgr.get_qmap(path)
        assert mgr.get_qmap(path) is first
        _replace(path, 2.0)
        second = mgr.get_qmap(path)
        assert second is not first
        assert second.value == 2.0

    def test_hashed_maps_are_still_shared_between_files(self, tmp_path, fake_qmap):
        a, b = str(tmp_path / "a.h5"), str(tmp_path / "b.h5")
        _write_hashed(a, 1.0, "geom")
        _write_hashed(b, 1.0, "geom")
        mgr = QMapManager()
        assert mgr.get_qmap(a) is mgr.get_qmap(b)
        assert fake_qmap.created == 1


# --------------------------------------------------------------------- R3-33
class TestR333QMapLifetime:
    def test_maps_die_with_their_files(self, tmp_path, fake_qmap):
        mgr = QMapManager()
        owners = []
        for i in range(8):
            p = str(tmp_path / f"f{i}.h5")
            _write(p, float(i))
            owners.append(mgr.get_qmap(p))
        assert len(mgr.db) == 8
        owners.clear()
        gc.collect()
        assert len(mgr.db) == 0

    def test_shared_map_survives_while_one_owner_remains(self, tmp_path, fake_qmap):
        a, b = str(tmp_path / "a.h5"), str(tmp_path / "b.h5")
        _write_hashed(a, 1.0, "geom")
        _write_hashed(b, 1.0, "geom")
        mgr = QMapManager()
        qa = mgr.get_qmap(a)
        mgr.get_qmap(b)
        gc.collect()
        assert mgr.get_qmap(b) is qa
        assert fake_qmap.created == 1
