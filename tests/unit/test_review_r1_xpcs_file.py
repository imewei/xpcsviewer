"""Regression tests for review round 1: R1-04, R1-05, R1-09, R1-10, R1-18."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from xpcsviewer.fitting import legacy
from xpcsviewer.utils.memory_manager import MemoryPressure, get_memory_manager
from xpcsviewer.xpcs_file import XpcsFile

pytestmark = pytest.mark.unit

TWOTIME = "/xpcs/twotime"


def _stub(**attrs) -> XpcsFile:
    """XpcsFile without running __init__ (no HDF5 needed)."""
    xf = XpcsFile.__new__(XpcsFile)
    xf.__dict__.update(attrs)
    return xf


class _QMap:
    """Three dynamic Q bins (IDs 1..3) at q = 0.1, 0.2, 0.3."""

    def get_qbin_in_qrange(self, qrange, zero_based=True):
        return np.array([0, 1, 2]), np.array([0.1, 0.2, 0.3])

    def get_qbin_label(self, qbin, append_qbin=False):
        return f"bin{qbin}"


# --------------------------------------------------------------------- R1-04
def test_get_g2_data_twotime_maps_columns_through_processed_bins():
    # Detector bins 1..3, but only bins 2 and 3 were processed (packed columns).
    xf = _stub(
        atype="Twotime",
        qmap=_QMap(),
        c2_g2=np.array([[20.0, 30.0], [21.0, 31.0]]),
        c2_processed_bins=np.array([2, 3]),
        c2_delay=np.array([1.0, 2.0]),
        label="t",
    )
    qvalues, _t, g2, _err, labels = xf.get_g2_data()
    np.testing.assert_allclose(qvalues, [0.2, 0.3])
    assert labels == ["bin2", "bin3"]
    np.testing.assert_array_equal(g2, [[20.0, 30.0], [21.0, 31.0]])


def test_get_g2_data_twotime_drops_unprocessed_bin_not_its_neighbour():
    xf = _stub(
        atype="Twotime",
        qmap=_QMap(),
        c2_g2=np.array([[20.0, 30.0]]),
        c2_processed_bins=np.array([1, 3]),
        c2_delay=np.array([1.0]),
        label="t",
    )
    qvalues, _t, g2, _err, labels = xf.get_g2_data()
    np.testing.assert_allclose(qvalues, [0.1, 0.3])
    assert labels == ["bin1", "bin3"]
    np.testing.assert_array_equal(g2, [[20.0, 30.0]])


def test_r2_15_imported_partition_does_not_relabel_twotime_bins():
    # The file's partition has bin 1 left, bin 2 right; C2 was computed on it.
    xf = _maps_stub([[1, 2]], [1, 2])
    _disp, _saxs, before = xf.get_twotime_maps(auto_crop=False, highlight_xy=(0, 0))
    # import_partition stores the imported map (swapped IDs) on the display layer
    xf.display_dqmap = np.array([[2, 1]])
    _disp, _saxs, after = xf.get_twotime_maps(auto_crop=False, highlight_xy=(0, 0))
    assert before == after == 0


def _maps_stub(dqmap, processed):
    from types import SimpleNamespace

    return _stub(
        qmap=SimpleNamespace(dqmap=np.asarray(dqmap)),
        saxs_2d=np.ones(np.shape(dqmap)),
        saxs_2d_log=np.ones(np.shape(dqmap)),
        c2_processed_bins=np.asarray(processed),
        label="t",
    )


def test_twotime_click_returns_position_in_processed_bins_zero_free_map():
    # dqmap has no zero pixel: np.unique(dqmap)[1:] used to drop bin 1.
    xf = _maps_stub([[1, 2], [3, 3]], [1, 2, 3])
    # displayed map is flipped vertically, so display (x=1, y=1) is bin 2
    _disp, _saxs, sel = xf.get_twotime_maps(auto_crop=False, highlight_xy=(1, 1))
    assert sel == 1


def test_twotime_click_with_unprocessed_bins_uses_processed_position():
    xf = _maps_stub([[0, 1], [2, 3]], [2, 3])
    # display (x=1, y=0) is bin 3 -> second processed bin
    _disp, _saxs, sel = xf.get_twotime_maps(auto_crop=False, highlight_xy=(1, 0))
    assert sel == 1


def test_twotime_click_on_unprocessed_pixel_is_rejected():
    xf = _maps_stub([[0, 1], [2, 3]], [2, 3])
    # display (x=1, y=1) is bin 1, which was not processed
    _disp, _saxs, sel = xf.get_twotime_maps(auto_crop=False, highlight_xy=(1, 1))
    assert sel is None


def test_twotime_selection_highlights_processed_bin_not_unique_position():
    xf = _maps_stub([[0, 1], [2, 3]], [2, 3])
    disp, _saxs, sel = xf.get_twotime_maps(auto_crop=False, selection=1)
    assert sel == 1
    # bin 3 (processed index 1) is highlighted above the max bin id
    assert np.nanmax(disp) == 4
    assert np.argwhere(disp == 4).tolist() == [[0, 1]]


def _write_twotime_h5(path):
    """processed_bins=[2, 10]; HDF5 name order (c2__10 < c2__2) differs."""
    with h5py.File(path, "w") as f:
        for b in (2, 10):
            f[f"{TWOTIME}/correlation_map/c2__{b}"] = np.full((4, 4), float(b))
        f[f"{TWOTIME}/processed_bins"] = np.array([2, 10])
        # (n_tau, n_bins): column k belongs to the k-th processed bin
        f[f"{TWOTIME}/normalized_g2"] = np.tile(np.array([2.0, 10.0]), (3, 1))
        f[f"{TWOTIME}/normalized_g2_segments"] = np.zeros((3, 2, 2))


@pytest.fixture
def twotime_stub(tmp_path, monkeypatch):
    path = tmp_path / "tt.h5"
    _write_twotime_h5(path)
    mod = sys.modules["xpcsviewer._xpcs_file_module"]
    monkeypatch.setattr(
        mod.MemoryMonitor, "is_memory_pressure_high", lambda *a, **k: False
    )
    return _stub(
        fname=str(path),
        c2_processed_bins=np.array([2, 10]),
        c2_kwargs=None,
        t0=0.001,
        c2_t0=0.006,
        _memory_manager=get_memory_manager(),
    )


@pytest.mark.parametrize("selection", [0, 1])
def test_single_c2_matrix_and_g2_belong_to_same_bin(twotime_stub, selection):
    bin_id = [2, 10][selection]
    res = twotime_stub.get_twotime_c2(selection=selection, correct_diag=False)
    assert res["c2_mat"][0, 1] == 2 * bin_id  # half + half.T
    np.testing.assert_array_equal(res["g2_full"], np.full(3, float(bin_id)))


def test_single_c2_missing_matrix_for_processed_bin_is_reported(twotime_stub):
    with h5py.File(twotime_stub.fname, "r+") as f:
        del f[f"{TWOTIME}/correlation_map/c2__10"]
    with pytest.raises(ValueError, match="processed bin 10"):
        twotime_stub.get_twotime_c2(selection=1, correct_diag=False)


# --------------------------------------------------------------------- R1-05
def test_twotime_c2_uses_effective_c2_t0_for_time_axis(twotime_stub):
    res = twotime_stub.get_twotime_c2(selection=0, correct_diag=False)
    # 1 ms exposure * stride 2 * avg 3 = 6 ms per C2 sample, not 1 ms
    assert res["delta_t"] == pytest.approx(0.006)
    assert res["acquire_period"] == pytest.approx(0.006)


# --------------------------------------------------------------------- R1-09
def test_cache_key_distinguishes_fit_flag_order():
    xf = _stub()
    a = xf._generate_cache_key("fit_g2", [True, False, True, True])
    b = xf._generate_cache_key("fit_g2", [True, True, False, True])
    assert a != b


def test_cache_key_distinguishes_bounds_order_and_keeps_dict_sorted():
    xf = _stub()
    assert xf._generate_cache_key("m", [[1, 2], [3, 4]]) != xf._generate_cache_key(
        "m", [[3, 4], [1, 2]]
    )
    assert xf._generate_cache_key("m", {"a": 1, "b": 2}) == xf._generate_cache_key(
        "m", {"b": 2, "a": 1}
    )


# --------------------------------------------------------------------- R1-10
def _tauq_stub():
    q = np.array([0.1, 0.2, 0.4])
    fit_val = np.zeros((3, 2, 2))
    fit_val[:, 0, 1] = [4.0, 2.0, 1.0]
    fit_val[:, 1, 1] = 0.01
    memory = SimpleNamespace(
        cache_get=lambda *a, **k: None, cache_put=lambda *a, **k: None
    )
    return SimpleNamespace(
        fit_summary={"q_val": q, "fit_val": fit_val},
        _generate_cache_key=lambda *a, **k: "k",
        _cache_prefix="t",
        _memory_manager=memory,
    )


def test_fit_tauq_reports_failure_when_solver_raises(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("solver failure")

    monkeypatch.setattr(legacy, "curve_fit", boom)
    stub = _tauq_stub()
    bounds = np.array([[0.0, -3.0], [2.0, 0.0]])
    out = XpcsFile.fit_tauq(stub, (0.0, 1.0), bounds, np.array([True, True]))
    assert not out["tauq_success"]
    assert out["tauq_fit_line"] is None


def test_fit_with_fixed_status_flags_failed_columns(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("solver failure")

    monkeypatch.setattr(legacy, "curve_fit", boom)
    x = np.array([0.1, 0.2, 0.4])
    y = np.array([[4.0], [2.0], [1.0]])
    out = legacy.fit_with_fixed(
        lambda x, a, b: a * x**b,
        x,
        y,
        np.full_like(y, 0.01),
        np.array([[0.0, -3.0], [2.0, 0.0]]),
        np.array([True, True]),
        x,
        return_status=True,
    )
    assert len(out) == 3
    assert not out[2].any()


# --------------------------------------------------------------------- R1-18
def test_large_saxs_default_path_returns_raw_and_log_images(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    data = rng.uniform(1.0, 100.0, (16, 16)).astype(np.float32)
    path = tmp_path / "saxs.h5"
    with h5py.File(path, "w") as f:
        f["/xpcs/temporal_mean/scattering_2d"] = data

    mod = sys.modules["xpcsviewer._xpcs_file_module"]
    monkeypatch.setattr(
        mod,
        "get_file_info",
        lambda *a, **k: {
            "large_datasets": [
                {
                    "path": "/xpcs/temporal_mean/scattering_2d",
                    "estimated_size_mb": 244.0,
                }
            ]
        },
    )
    monkeypatch.setattr(
        mod,
        "predict_operation_memory",
        lambda *a, **k: SimpleNamespace(
            predicted_mb=500.0,
            pressure_level=MemoryPressure.MODERATE,
            confidence=0.9,
            recommended_actions=[],
        ),
    )
    xf = _stub(
        fname=str(path),
        _saxs_2d_data=None,
        _saxs_2d_log_data=None,
        _saxs_data_loaded=False,
        _use_lazy_loading=True,
        _memory_manager=get_memory_manager(),
        _cache_prefix="r1_18_lazy_path",
    )

    raw = xf.saxs_2d
    assert raw.shape == (16, 16)
    assert raw.ndim == 2  # array interface used by get_twotime_maps
    np.testing.assert_array_equal(np.asarray(raw), data)
    log = xf.saxs_2d_log
    np.testing.assert_allclose(
        np.asarray(log), xf._compute_saxs_log_standard(data), rtol=1e-6
    )
