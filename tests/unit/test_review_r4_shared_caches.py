"""Regression tests for round 4 shared-cache findings R4-05 and R4-16.

Invariant: anything shared between files or cached is keyed by EVERYTHING it was
derived from, and its memory is accounted for honestly.

R4-05  a QMap (calibration + per-pixel q/phi) is shared between files only when
       the stored partition hash AND every per-file calibration input match.
R4-16  the C2 cache stores arrays that own their data and charges the bytes it
       actually retains.

All tests write real NeXus-layout HDF5 files with h5py and load them through the
real ``FileLocator`` / ``XpcsFile`` / ``QMapManager``.
"""

from __future__ import annotations

import gc
import os

import h5py
import numpy as np
import pytest

from tests.unit.test_review_r3_caches import _write_c2
from tests.unit.test_review_r3_maps_roi import _write_nexus
from xpcsviewer.file_locator import FileLocator
from xpcsviewer.fileIO.aps_8idi import key as hdf_key
from xpcsviewer.fileIO.hdf_reader import clear_connection_pool
from xpcsviewer.fileIO.qmap_utils import QMapManager
from xpcsviewer.module import twotime_utils
from xpcsviewer.xpcs_file import XpcsFile

pytestmark = pytest.mark.unit

K = hdf_key["nexus"]
SHARED_HASH = "same_copied_partition"


@pytest.fixture(autouse=True)
def _fresh_pool():
    clear_connection_pool()
    twotime_utils.get_single_c2_from_hdf.cache_clear()
    twotime_utils.get_c2_g2partials_from_hdf.cache_clear()
    yield
    clear_connection_pool()
    twotime_utils.get_single_c2_from_hdf.cache_clear()
    twotime_utils.get_c2_g2partials_from_hdf.cache_clear()


def _write_hashed(path, **overrides):
    """Real NeXus file stored with the shared partition hash and overrides."""
    _write_nexus(path)
    with h5py.File(path, "a") as f:
        for name, value in overrides.items():
            del f[K[name]]
            f[K[name]] = value
        f["/xpcs/qmap"].attrs["hash"] = SHARED_HASH


def _open_pair(tmp_path, a_over, b_over):
    pa, pb = tmp_path / "a.h5", tmp_path / "b.h5"
    _write_hashed(pa, **a_over)
    _write_hashed(pb, **b_over)
    loc = FileLocator(str(tmp_path))
    loc.build()
    loc.add_target([pa.name, pb.name])
    a, b = loc.get_xf_list(rows=[0, 1])
    return loc, a, b


# --------------------------------------------------------------------- R4-05
class TestR405QMapSharingKey:
    def test_different_energy_same_hash_gets_own_geometry(self, tmp_path):
        _, a, b = _open_pair(tmp_path, {"X_energy": 10.0}, {"X_energy": 12.0})
        assert a.qmap is not b.qmap
        assert (a.qmap.X_energy, b.qmap.X_energy) == (10.0, 12.0)
        direct = XpcsFile(str(tmp_path / "b.h5"))
        np.testing.assert_array_equal(b.qmap.qmap["q"], direct.qmap.qmap["q"])
        assert not np.allclose(a.qmap.qmap["q"], b.qmap.qmap["q"])

    def test_roi_output_independent_of_open_order(self, tmp_path):
        _, _, b = _open_pair(tmp_path, {"X_energy": 10.0}, {"X_energy": 12.0})
        direct = XpcsFile(str(tmp_path / "b.h5"))
        roi = {"sl_type": "Pie", "angle_range": (0, 360), "dist": 3.0}
        shared_q, shared_y = b.get_roi_data(roi)
        fresh_q, fresh_y = direct.get_roi_data(roi)
        np.testing.assert_allclose(shared_q, fresh_q)
        np.testing.assert_allclose(shared_y, fresh_y, equal_nan=True)

    @pytest.mark.parametrize(
        ("name", "a_val", "b_val"),
        [
            ("det_dist", 5.0, 4.0),
            ("pixel_size", 75e-6, 172e-6),
            ("bcx", 3.5, 2.5),
            ("bcy", 3.5, 4.5),
            ("dqlist", np.array([0.01, 0.02]), np.array([0.01, 0.03])),
            ("sqlist", np.array([0.01, 0.02, 0.03]), np.array([0.01, 0.02, 0.04])),
        ],
    )
    def test_every_calibration_input_is_part_of_the_key(
        self, tmp_path, name, a_val, b_val
    ):
        _, a, b = _open_pair(tmp_path, {name: a_val}, {name: b_val})
        assert a.qmap is not b.qmap

    def test_identical_calibration_is_still_shared(self, tmp_path):
        _, a, b = _open_pair(tmp_path, {}, {})
        assert a.qmap is b.qmap

    def test_replaced_file_with_same_hash_gets_new_energy(self, tmp_path):
        path = tmp_path / "scan.h5"
        _write_hashed(path, X_energy=10.0)
        mgr = QMapManager()
        old = mgr.get_qmap(str(path))
        tmp = tmp_path / "scan.h5.new"
        _write_hashed(tmp, X_energy=12.0)  # same hash, new energy
        os.replace(tmp, path)
        new = mgr.get_qmap(str(path))
        assert old is not new
        assert new.X_energy == 12.0

    def test_maps_still_die_with_their_files(self, tmp_path):
        loc, a, b = _open_pair(tmp_path, {"X_energy": 10.0}, {"X_energy": 12.0})
        assert len(loc.qmap_manager.db) == 2
        loc.clear_target()
        del a, b
        gc.collect()
        assert len(loc.qmap_manager.db) == 0


# --------------------------------------------------------------------- R4-16
def _retained(value) -> int:
    """Bytes of backing storage kept alive by the arrays in a cached result."""
    roots: dict[int, np.ndarray] = {}
    for arr in value.values():
        if isinstance(arr, np.ndarray):
            root = arr
            while isinstance(root.base, np.ndarray):
                root = root.base
            roots[id(root)] = root
    return sum(r.nbytes for r in roots.values())


class TestR416C2ByteAccounting:
    def test_cached_arrays_own_their_data(self, tmp_path):
        path = str(tmp_path / "c2.h5")
        _write_c2(path, 1.0, n=8, nq=4)
        res = twotime_utils.get_single_c2_from_hdf(path, 2, 512, 1, False)
        part = twotime_utils.get_c2_g2partials_from_hdf(path)
        for value in (res, part):
            for name, arr in value.items():
                if isinstance(arr, np.ndarray):
                    assert arr.base is None, name
        assert res["g2_full"].nbytes < part["g2_full"].nbytes  # one bin, not all

    def test_charged_bytes_cover_retained_bytes(self, tmp_path):
        path = str(tmp_path / "c2.h5")
        _write_c2(path, 1.0, n=8, nq=4)
        retained = 0
        for sel in range(4):
            res = twotime_utils.get_single_c2_from_hdf(path, sel, 512, 1, False)
            retained += _retained(res)
        part = twotime_utils.get_c2_g2partials_from_hdf(path)
        retained += _retained(part)
        assert twotime_utils._c2_cache.nbytes >= retained

    def test_view_results_are_charged_for_their_base(self):
        base = np.zeros((100, 100))
        view = base[:, 0]
        assert twotime_utils._nbytes({"a": view}) >= base.nbytes
        # two views of one base are charged once
        assert twotime_utils._nbytes({"a": view, "b": base[:, 1]}) == base.nbytes
