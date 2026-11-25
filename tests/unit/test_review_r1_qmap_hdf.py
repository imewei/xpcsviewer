"""Regression tests for review round 1 findings R1-03, R1-07 and R1-16."""

from __future__ import annotations

import threading

import h5py
import numpy as np
import pytest

from xpcsviewer.fileIO.hdf_reader import HDF5ConnectionPool
from xpcsviewer.fileIO.qmap_utils import QMap


def _make_qmap(dqlist, n_phi, dynamic_mapping) -> QMap:
    qmap = QMap.__new__(QMap)
    qmap.fname = "synthetic.hdf"
    qmap.map_names = ["q", "phi"]
    qmap.dqlist = np.asarray(dqlist, dtype=float)
    qmap.dynamic_num_pts = np.array([len(dqlist), n_phi])
    qmap.dynamic_index_mapping = np.asarray(dynamic_mapping)
    qmap._qbin_cache = {}
    return qmap


@pytest.mark.unit
class TestR103SparseQValues:
    def test_q_values_follow_occupied_bins(self):
        qmap = _make_qmap([0.1, 0.2, 0.3], 1, [0, 2])
        bins, qvals = qmap.get_qbin_in_qrange(None)
        assert list(bins) == [0, 1]
        np.testing.assert_allclose(qvals, [0.1, 0.3])

    def test_q_values_with_range_and_multiple_phi(self):
        # 3 q x 2 phi grid, occupied flat positions: (q0,phi1), (q1,phi0), (q2,phi1)
        qmap = _make_qmap([0.1, 0.2, 0.3], 2, [1, 2, 5])
        bins, qvals = qmap.get_qbin_in_qrange((0.15, 0.35), zero_based=False)
        assert list(bins) == [2, 3]
        np.testing.assert_allclose(qvals, [0.2, 0.3])
        assert len(bins) == len(qvals)

    def test_dense_map_unchanged(self):
        qmap = _make_qmap([0.1, 0.2, 0.3], 1, [0, 1, 2])
        bins, qvals = qmap.get_qbin_in_qrange(None)
        assert list(bins) == [0, 1, 2]
        np.testing.assert_allclose(qvals, [0.1, 0.2, 0.3])


def _make_saxs_qmap(static_mapping, sqlist, splist) -> QMap:
    qmap = QMap.__new__(QMap)
    qmap.fname = "synthetic.hdf"
    qmap.static_index_mapping = np.asarray(static_mapping)
    qmap.sqlist = np.asarray(sqlist, dtype=float)
    qmap.splist = np.asarray(splist, dtype=float)
    return qmap


@pytest.mark.unit
class TestR107SingleAngleEmptyBin:
    def test_saxs_1d_expands_empty_q_bin_to_nan(self):
        qmap = _make_saxs_qmap([0, 2], [0.1, 0.2, 0.3], [0.0])
        info = qmap.reshape_phi_analysis(np.array([10.0, 30.0]), "f", mode="saxs_1d")
        assert len(info["q"]) == 3
        assert info["Iq"].shape == (1, 3)
        np.testing.assert_array_equal(info["Iq"], [[10.0, np.nan, 30.0]])

    def test_stability_expands_empty_q_bin_to_nan(self):
        qmap = _make_saxs_qmap([0, 2], [0.1, 0.2, 0.3], [0.0])
        out = qmap.reshape_phi_analysis(
            np.array([[10.0, 30.0], [11.0, 31.0]]), "f", mode="stability"
        )
        np.testing.assert_array_equal(out, [[10.0, np.nan, 30.0], [11.0, np.nan, 31.0]])

    def test_dense_single_angle_unchanged(self):
        qmap = _make_saxs_qmap([0, 1, 2], [0.1, 0.2, 0.3], [0.0])
        info = qmap.reshape_phi_analysis(np.array([1.0, 2.0, 3.0]), "f", mode="saxs_1d")
        np.testing.assert_array_equal(info["Iq"], [[1.0, 2.0, 3.0]])
        assert info["labels"] == ["f"]


def _write_h5(path, value):
    with h5py.File(path, "w") as f:
        f["x"] = value


@pytest.mark.unit
class TestR116PoolBorrowSafety:
    def _pool(self, size):
        return HDF5ConnectionPool(
            max_pool_size=size, enable_memory_pressure_adaptation=False
        )

    def test_borrowed_handle_not_evicted(self, tmp_path):
        a, b = str(tmp_path / "a.h5"), str(tmp_path / "b.h5")
        _write_h5(a, 1)
        _write_h5(b, 2)
        pool = self._pool(1)
        try:
            with pool.get_connection(a) as fa:
                with pool.get_connection(b) as fb:
                    assert fa.id.valid
                    assert fa["x"][()] == 1
                    assert fb["x"][()] == 2
                assert fa["x"][()] == 1
        finally:
            pool.clear_pool()

    def test_idle_handle_still_evicted(self, tmp_path):
        a, b = str(tmp_path / "a.h5"), str(tmp_path / "b.h5")
        _write_h5(a, 1)
        _write_h5(b, 2)
        pool = self._pool(1)
        try:
            with pool.get_connection(a) as fa:
                pass
            with pool.get_connection(b):
                pass
            assert not fa.id.valid
            assert len(pool._pool) == 1
        finally:
            pool.clear_pool()

    def test_error_in_one_borrower_does_not_close_shared_handle(self, tmp_path):
        a = str(tmp_path / "a.h5")
        _write_h5(a, 1)
        pool = self._pool(2)
        try:
            with pool.get_connection(a) as outer:
                with pytest.raises(KeyError), pool.get_connection(a):
                    raise KeyError("boom")
                assert outer.id.valid
                assert outer["x"][()] == 1
            assert not outer.id.valid  # closed once the last borrower released
        finally:
            pool.clear_pool()

    def test_aged_health_check_defers_close_while_borrowed(self, tmp_path):
        a = str(tmp_path / "a.h5")
        _write_h5(a, 1)
        pool = HDF5ConnectionPool(
            max_pool_size=2,
            health_check_interval=0.0,
            enable_memory_pressure_adaptation=False,
        )
        b = str(tmp_path / "b.h5")
        _write_h5(b, 2)
        try:
            with pool.get_connection(a) as fa:
                with pool.get_connection(b):  # health check runs, `a` is "aged"
                    pass
                assert fa.id.valid
        finally:
            pool.clear_pool()

    def test_concurrent_readers_never_see_closed_file(self, tmp_path):
        paths = []
        for i in range(4):
            p = str(tmp_path / f"f{i}.h5")
            _write_h5(p, i)
            paths.append(p)
        pool = self._pool(1)
        errors: list[BaseException] = []

        def work(i):
            try:
                for _ in range(50):
                    with pool.get_connection(paths[i]) as f:
                        assert f["x"][()] == i
                        with pool.get_connection(paths[(i + 1) % 4]) as g:
                            assert g["x"][()] == (i + 1) % 4
                        assert f["x"][()] == i
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        pool.clear_pool()
        assert not errors, errors[:1]
