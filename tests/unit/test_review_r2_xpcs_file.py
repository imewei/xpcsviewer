"""Regression tests for review round 2 (xpcs_file.py and qmap_utils.py).

R2-03, R2-13, R2-21 (xpcs_file part), R2-25, R2-27, R2-30, R2-31, R2-35.
"""

from __future__ import annotations

import sys
import uuid
from types import SimpleNamespace

import numpy as np
import pytest

from xpcsviewer.fileIO.qmap_utils import QMap
from xpcsviewer.fitting import legacy
from xpcsviewer.utils.memory_manager import MemoryPressure, get_memory_manager
from xpcsviewer.xpcs_file import XpcsFile

pytestmark = pytest.mark.unit


def _stub(**attrs) -> XpcsFile:
    """XpcsFile without running __init__ (no HDF5 needed)."""
    xf = XpcsFile.__new__(XpcsFile)
    xf.__dict__.update(attrs)
    return xf


class _DictCache:
    """Minimal stand-in for the unified memory manager cache."""

    def __init__(self) -> None:
        self.store: dict = {}

    def cache_get(self, key, cache_type=None):
        return self.store.get(key)

    def cache_put(self, key, value, cache_type=None):
        self.store[key] = value
        return True


# --------------------------------------------------------------------- R2-03
def _geometry_qmap() -> QMap:
    qmap = QMap.__new__(QMap)
    qmap.mask = np.ones((5, 5))
    qmap.bcx = 2.0
    qmap.bcy = 2.0
    qmap.pixel_size = 1.0
    qmap.det_dist = 2.0  # pixel (x=4, y=2) sits at r == det_dist
    qmap.k0 = 3.0
    qmap._qmap_cache = None
    return qmap


def test_alpha_map_is_in_degrees_and_q_still_uses_radians():
    qmap = _geometry_qmap()
    maps, units = qmap.compute_qmap()
    assert units["alpha"] == "°"
    assert maps["alpha"][2, 4] == pytest.approx(45.0, abs=1e-4)
    # trigonometry keeps using radians: q = k0 * sin(alpha_rad)
    assert maps["q"][2, 4] == pytest.approx(3.0 * np.sin(np.pi / 4), rel=1e-6)
    assert "alpha=45.000 °" in _readout(qmap, 4, 2)


def _readout(qmap: QMap, x: int, y: int) -> str:
    qmap.qmap, qmap.qmap_units = qmap.compute_qmap()
    return qmap.get_qmap_at_pos(x, y)


# --------------------------------------------------------------------- R2-35
def _dynamic_qmap(dqlist=(0.1, 0.2)) -> QMap:
    n = len(dqlist)
    qmap = QMap.__new__(QMap)
    qmap.fname = "synthetic.hdf"
    qmap.map_names = ["q", "phi"]
    qmap.dqlist = np.asarray(dqlist, dtype=float)
    qmap.dynamic_num_pts = np.array([n, 1])
    qmap.dynamic_index_mapping = np.arange(n)
    qmap._qbin_cache = {}
    return qmap


def test_qrange_matching_no_bin_is_empty_not_all_bins():
    qmap = _dynamic_qmap()
    bins, qvals = qmap.get_qbin_in_qrange((0.5, 0.6))
    assert len(bins) == 0
    assert len(qvals) == 0


def test_qrange_none_still_selects_all_bins():
    bins, qvals = _dynamic_qmap().get_qbin_in_qrange(None)
    assert list(bins) == [0, 1]
    np.testing.assert_allclose(qvals, [0.1, 0.2])


class _EmptyQMap:
    def get_qbin_in_qrange(self, qrange, zero_based=True):
        return np.array([], dtype=int), np.array([])

    def get_qbin_label(self, qbin, append_qbin=False):
        return f"bin{qbin}"


def _multitau_stub(**extra) -> XpcsFile:
    t = np.logspace(-3, 1, 30)
    g2 = np.ones((30, 2))
    return _stub(
        atype="Multitau",
        qmap=_EmptyQMap(),
        g2=g2,
        g2_err=np.full_like(g2, 0.01),
        t_el=t,
        label="t",
        **extra,
    )


def test_get_g2_data_empty_q_selection_returns_empty_columns():
    qvalues, t_el, g2, g2_err, labels = _multitau_stub().get_g2_data(qrange=(5, 6))
    assert len(qvalues) == 0
    assert labels == []
    assert g2.shape == (30, 0)
    assert g2_err.shape == (30, 0)
    assert t_el.shape == (30,)


def test_get_g2_stability_data_empty_q_selection_returns_empty_columns():
    g2_partial = np.ones((3, 30, 2))
    xf = _multitau_stub(
        g2_partial=g2_partial, g2_partial_err=None, g2_partial_labels=None
    )
    q, _t, g2, g2_err, qbin_labels, _labels = xf.get_g2_stability_data(qrange=(5, 6))
    assert len(q) == 0
    assert qbin_labels == []
    assert g2.shape == (3, 30, 0)
    assert g2_err.shape == g2.shape


def test_fit_g2_with_empty_q_selection_raises_clear_error():
    xf = _multitau_stub(_cache_prefix="r2_35", _memory_manager=_DictCache())
    bounds = np.array([[0.0, 1e-3, 0.5, 0.9], [1.0, 10.0, 2.0, 1.1]])
    with pytest.raises(ValueError, match="no Q bins"):
        xf.fit_g2(
            (5, 6), None, bounds, np.array([True] * 4), "single", force_refit=True
        )


# --------------------------------------------------------------------- R2-13
def test_fit_g2_high_performance_builds_summary_from_solver_arrays():
    t = np.logspace(-3, 1, 30)
    taus = np.array([0.1, 0.4])  # tau ~ q^-2 for q = 0.2, 0.1
    g2 = 1.0 + 0.2 * np.exp(-2 * (t[:, None] / taus[None, :]))

    class _TwoQ:
        def get_qbin_in_qrange(self, qrange, zero_based=True):
            return np.array([0, 1]), np.array([0.2, 0.1])

        def get_qbin_label(self, qbin, append_qbin=False):
            return f"bin{qbin}"

    xf = _stub(
        atype="Multitau",
        qmap=_TwoQ(),
        g2=g2,
        g2_err=np.full_like(g2, 0.005),
        t_el=t,
        label="t",
        _cache_prefix="r2_13",
        _memory_manager=_DictCache(),
    )
    # contrast, tau, stretch, baseline: the layout every fit_val reader assumes
    bounds = np.array([[0.0, 1e-3, 0.5, 0.9], [1.0, 10.0, 2.0, 1.1]])
    out = xf.fit_g2_high_performance(
        None, None, bounds, np.array([True] * 4), "single", use_parallel=False
    )
    assert out is xf.fit_summary
    assert out["high_performance_fit"] is True
    assert out["fit_val"].shape == (2, 2, 4)
    assert out["fit_line"].shape == (2, out["fit_x"].size)
    assert np.all(np.isfinite(out["fit_line"]))
    np.testing.assert_allclose(out["q_val"], [0.2, 0.1])
    np.testing.assert_allclose(out["fit_val"][:, 0, 1], taus, rtol=0.05)
    # downstream readers accept the summary
    assert "a = " in xf.get_fitting_info("g2_fitting")["fit_val"][0]
    xf._cache_prefix = "r2_13_tauq"
    tq = xf.fit_tauq(
        (0.0, 1.0), np.array([[1e-8, -4.0], [1e2, 0.0]]), np.array([True, False])
    )
    assert tq["tauq_success"]


# --------------------------------------------------------------------- R2-25
def _saxs_stub() -> XpcsFile:
    return _stub(
        fname="r2_25.h5",
        _cache_prefix=f"r2_25_{uuid.uuid4().hex}",
        _saxs_data_loaded=False,
        _saxs_2d_data=None,
        _saxs_2d_log_data=None,
        _memory_manager=get_memory_manager(),
        qmap=SimpleNamespace(get_qmap_at_pos=lambda x, y: f"x={x}"),
        label="saxs",
    )


class _FakeReader:
    def __init__(self, arr):
        self.arr = arr

    def read_dataset(self, *args, **kwargs):
        return self.arr


def _patch_saxs_io(monkeypatch, arr, size_mb, enhanced_ok=True):
    import xpcsviewer.fileIO.hdf_reader_enhanced as enhanced

    # xpcsviewer.xpcs_file is a package; the class lives in a loaded sibling module
    mod = sys.modules[XpcsFile.__module__]

    monkeypatch.setattr(
        mod,
        "get_file_info",
        lambda *a, **k: {
            "large_datasets": [
                {
                    "path": "/xpcs/temporal_mean/scattering_2d",
                    "estimated_size_mb": size_mb,
                }
            ]
        },
    )
    if enhanced_ok:
        monkeypatch.setattr(enhanced, "get_enhanced_reader", lambda: _FakeReader(arr))
    else:

        def boom():
            raise RuntimeError("enhanced reader unavailable")

        monkeypatch.setattr(enhanced, "get_enhanced_reader", boom)
        monkeypatch.setattr(mod, "get_chunked_dataset", lambda *a, **k: arr)
        monkeypatch.setattr(mod, "batch_read_fields", lambda *a, **k: {"saxs_2d": arr})


@pytest.mark.parametrize(
    ("size_mb", "enhanced_ok", "mapped"),
    [
        (244.0, True, None),
        (1100.0, True, True),
        (1100.0, False, True),
        (10.0, False, False),
    ],
)
def test_saxs_load_normalises_leading_singleton_for_every_strategy(
    monkeypatch, size_mb, enhanced_ok, mapped
):
    raw = np.arange(1.0, 65.0).reshape(1, 8, 8)
    _patch_saxs_io(monkeypatch, raw, size_mb, enhanced_ok)
    xf = _saxs_stub()
    xf._load_saxs_data_batch(use_memory_mapping=mapped)
    assert xf.saxs_2d_data.shape == (8, 8)
    assert xf.saxs_2d_log_data.shape == (8, 8)
    assert xf.get_info_at_position(1, 0).startswith("I=")  # x=1,y=0 (pixel readout)


def test_saxs_load_keeps_genuine_multi_frame_data(monkeypatch):
    raw = np.ones((3, 8, 8))
    _patch_saxs_io(monkeypatch, raw, 10.0)
    xf = _saxs_stub()
    xf._load_saxs_data_batch()
    assert xf.saxs_2d_data.shape == (3, 8, 8)


# --------------------------------------------------------------------- R2-27
_TAUQ_BOUNDS = np.array([[1e-8, -4.0], [1e2, 0.0]])
_Q = np.array([0.05, 0.1, 0.2, 0.4])


def _g2_fit_summary(tau_scale: float) -> dict:
    fit_val = np.zeros((_Q.size, 2, 2))
    fit_val[:, 0, 1] = tau_scale / _Q**2
    fit_val[:, 1, 1] = 0.01
    return {"q_val": _Q.copy(), "fit_val": fit_val}


def test_tauq_cache_is_not_reused_after_the_g2_fit_changes():
    xf = _stub(
        _cache_prefix="r2_27",
        _memory_manager=_DictCache(),
        fit_summary=_g2_fit_summary(1e-3),
    )
    flag = np.array([True, False])
    first = xf.fit_tauq((0.0, 1.0), _TAUQ_BOUNDS, flag)
    assert first["tauq_success"]
    a_old = float(first["tauq_fit_val"][0, 0])

    xf.fit_summary = _g2_fit_summary(1e-2)  # re-fit G2: taus 10x larger
    second = xf.fit_tauq((0.0, 1.0), _TAUQ_BOUNDS, flag)
    assert float(second["tauq_fit_val"][0, 0]) == pytest.approx(10 * a_old, rel=1e-3)


def test_tauq_cache_still_hits_for_identical_g2_fit():
    cache = _DictCache()
    xf = _stub(
        _cache_prefix="r2_27b", _memory_manager=cache, fit_summary=_g2_fit_summary(1e-3)
    )
    flag = np.array([True, False])
    xf.fit_tauq((0.0, 1.0), _TAUQ_BOUNDS, flag)
    n_entries = len(cache.store)
    # same numbers in a fresh summary object -> same key, no new entry
    xf.fit_summary = _g2_fit_summary(1e-3)
    xf.fit_tauq((0.0, 1.0), _TAUQ_BOUNDS, flag)
    assert len(cache.store) == n_entries


# ----------------------------------------------------- R2-21 (xpcs_file part)
def test_failed_tauq_fit_does_not_leave_fit_looking_numbers(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("solver failure")

    monkeypatch.setattr(legacy, "curve_fit", boom)
    xf = _stub(
        _cache_prefix="r2_21",
        _memory_manager=_DictCache(),
        fit_summary=_g2_fit_summary(1e-3),
        label="f",
    )
    out = xf.fit_tauq((0.0, 1.0), _TAUQ_BOUNDS, np.array([True, True]))
    assert not out["tauq_success"]
    assert out["tauq_fit_line"] is None
    assert out["tauq_fit_val"].shape == (2, 2)
    assert np.all(np.isnan(out["tauq_fit_val"]))
    text = xf.get_fitting_info(mode="tauq_fitting")
    assert "fail" in text.lower()
    assert "a = 0" not in text


def test_failed_tauq_fit_clears_a_previous_successful_fit():
    xf = _stub(
        _cache_prefix="r2_21b",
        _memory_manager=_DictCache(),
        fit_summary=_g2_fit_summary(1e-3),
        label="f",
    )
    flag = np.array([True, False])
    assert xf.fit_tauq((0.0, 1.0), _TAUQ_BOUNDS, flag)["tauq_success"]
    out = xf.fit_tauq((5.0, 6.0), _TAUQ_BOUNDS, flag)  # q-range with no data
    assert not out["tauq_success"]
    assert out["tauq_fit_line"] is None
    assert np.all(np.isnan(out["tauq_fit_val"]))


def test_successful_tauq_fit_info_text_unchanged():
    xf = _stub(
        _cache_prefix="r2_21c",
        _memory_manager=_DictCache(),
        fit_summary=_g2_fit_summary(1e-3),
        label="f",
    )
    xf.fit_tauq((0.0, 1.0), _TAUQ_BOUNDS, np.array([True, False]))
    text = xf.get_fitting_info(mode="tauq_fitting")
    assert text.startswith("a = ") and "; b = " in text


# --------------------------------------------------------------------- R2-30
class _BigArray(np.ndarray):
    """ndarray that reports a 600 MiB footprint (avoids allocating it)."""

    @property
    def nbytes(self):
        return 600 * 1024 * 1024


def _roi_stub(pressure: MemoryPressure, spmap=None) -> XpcsFile:
    saxs = np.arange(16, dtype=float).reshape(4, 4).view(_BigArray)
    return _stub(
        _saxs_2d_data=saxs,
        _memory_manager=SimpleNamespace(get_memory_pressure=lambda: pressure),
        mask=np.ones((4, 4)),
        bcx=0.0,
        bcy=0.0,
        sqmap=np.ones((4, 4)),
        spmap=np.zeros((4, 4)) if spmap is None else spmap,
        sqspan=np.array([0.0, 2.0]),
        sqlist=np.array([1.0]),
        label="roi",
    )


def test_large_saxs_roi_uses_initialised_memory_manager(monkeypatch):
    xf = _roi_stub(MemoryPressure.HIGH)
    sentinel = (np.array([1.0]), np.array([2.0]))
    monkeypatch.setattr(XpcsFile, "_get_roi_data_streaming", lambda *a, **k: sentinel)
    out = xf.get_roi_data({"sl_type": "Pie", "angle_range": (0, 1), "dist": 1})
    assert out is sentinel


def test_large_saxs_roi_without_pressure_takes_normal_path():
    xf = _roi_stub(MemoryPressure.LOW)
    x, y = xf.get_roi_data({"sl_type": "Ring", "radius": (0.5, 2.5)})
    assert np.all(np.isfinite(y))


def test_large_saxs_ring_roi_falls_back_when_streaming_has_no_ring_support():
    xf = _roi_stub(MemoryPressure.HIGH)
    out = xf.get_roi_data({"sl_type": "Ring", "radius": (0.5, 2.5)})
    assert out is not None


@pytest.mark.parametrize("dist", [1.0e6, 0.1])
def test_streaming_pie_roi_matches_eager_pie_roi(dist):
    rng = np.random.default_rng(0)
    xf = _stub(
        _saxs_2d_data=rng.uniform(1.0, 2.0, (4, 4)),
        mask=np.ones((4, 4)),
        bcx=0.0,
        bcy=0.0,
        sqmap=np.linspace(0.05, 0.95, 16).reshape(4, 4),
        spmap=np.zeros((4, 4)),
        sqspan=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        sqlist=np.array([0.125, 0.375, 0.625, 0.875]),
        X_energy=12.398,
        pixel_size=1.0,
        det_dist=1.0,
        label="pie",
    )
    roi = {"sl_type": "Pie", "angle_range": (-10.0, 10.0), "dist": dist}
    x_e, y_e = xf.get_roi_data(roi)
    x_s, y_s = xf._get_roi_data_streaming(roi)
    np.testing.assert_allclose(x_s, x_e)
    np.testing.assert_allclose(y_s, y_e, equal_nan=True)
    assert np.isnan(y_e).any() == (dist < 1)


# --------------------------------------------------------------------- R2-31
def _ring_stub(spmap) -> XpcsFile:
    return _stub(
        _saxs_2d_data=np.arange(1.0, 17.0).reshape(4, 4),
        mask=np.ones((4, 4)),
        bcx=0.0,
        bcy=0.0,
        sqmap=np.ones((4, 4)),
        spmap=spmap,
        label="ring",
    )


def test_empty_annulus_returns_empty_result_instead_of_raising():
    xf = _ring_stub(np.zeros((4, 4)))
    x, y = xf.get_roi_data({"sl_type": "Ring", "radius": (10, 20)})
    assert x.size == 0
    assert y.size == 0


def test_one_angle_annulus_has_no_zero_division():
    xf = _ring_stub(np.full((4, 4), 30.0))
    with np.errstate(all="raise"):
        x, y = xf.get_roi_data({"sl_type": "Ring", "radius": (0.5, 1.5)})
    # pixels with 0.5 <= r < 1.5: (0,1), (1,0), (1,1) -> values 2, 5, 6
    np.testing.assert_allclose(x, [30.0])
    np.testing.assert_allclose(y, [np.mean([2.0, 5.0, 6.0])])


def test_multi_angle_annulus_unchanged():
    yy, xx = np.mgrid[0:4, 0:4]
    phi = np.degrees(np.arctan2(yy, xx))
    xf = _ring_stub(phi)
    x, y = xf.get_roi_data({"sl_type": "Ring", "radius": (0.5, 5.0)}, phi_num=9)
    assert x.shape == (9,)
    assert y.shape == (9,)
    assert x[0] == pytest.approx(0.0)
    assert x[-1] == pytest.approx(90.0)
