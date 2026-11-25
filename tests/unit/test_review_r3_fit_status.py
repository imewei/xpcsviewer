"""Round 3 regressions: a failed G2 / tau-Q fit is never presented as a measurement.

R3-21, R3-22, R3-25, R3-26 (one invariant across every solver path and reader)
plus R3-15, R3-27 and R3-28 (same files). Tests marked "real file" write an HDF5
file with h5py and load it through the real ``XpcsFile``.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest

from xpcsviewer.fitting import legacy
from xpcsviewer.fitting.legacy import (
    fit_with_fixed,
    fit_with_fixed_parallel,
    fit_with_fixed_sequential,
    make_single_exp,
)
from xpcsviewer.module import g2mod, tauq
from xpcsviewer.xpcs_file import XpcsFile

pytestmark = pytest.mark.unit

TRUE_TAU = np.array([0.1, 0.2, 0.3])
# single_exp_all(x, a, b, c, d) = a * exp(-2 (x / b) ** c) + d
BOUNDS = np.array([[0.01, 1e-3, 0.5, 0.9], [1.0, 1e2, 1.5, 1.1]])
TAUQ_BOUNDS = np.array([[1e-3, -3.0], [1e3, 3.0]])


# --------------------------------------------------------------------------
# real-file helpers
# --------------------------------------------------------------------------
def _model(tau, taus=TRUE_TAU):
    return 1.0 + 0.5 * np.exp(-2 * (np.asarray(tau)[:, None] / taus[None, :]))


def _write_file(
    path, tau, g2=None, q=(0.1, 0.2, 0.3), saxs=(10.0, 5.0, 2.0), processed=None
):
    """Minimal APS-style Multitau (and optional Twotime) NeXus file."""
    tau = np.asarray(tau, dtype=float)
    nq, nt = len(q), len(tau)
    if g2 is None:
        g2 = _model(tau, np.resize(TRUE_TAU, nq))
    ns = len(saxs)
    with h5py.File(path, "w") as f:
        f["/xpcs/multitau/delay_list"] = tau
        f["/xpcs/multitau/normalized_g2"] = g2
        f["/xpcs/multitau/normalized_g2_err"] = np.full((nt, nq), 0.01)
        f["/xpcs/multitau/config/stride_frame"] = 1
        f["/xpcs/multitau/config/avg_frame"] = 1
        f["/entry/instrument/detector_1/frame_time"] = 1.0
        f["/entry/instrument/detector_1/count_time"] = 1.0
        f["/entry/instrument/detector_1/x_pixel_size"] = 0.075
        f["/entry/instrument/detector_1/y_pixel_size"] = 0.075
        f["/entry/instrument/incident_beam/incident_energy"] = 8.0
        f["/entry/start_time"] = "2024-01-01T00:00:00"
        f["/xpcs/qmap/dynamic_roi_map"] = np.tile(np.arange(1, nq + 1), (2, 1))
        f["/xpcs/qmap/static_roi_map"] = np.tile(np.arange(1, ns + 1), (2, 1))
        f["/xpcs/qmap/dynamic_v_list_dim0"] = np.asarray(q, dtype=float)
        f["/xpcs/qmap/static_v_list_dim0"] = np.linspace(0.1, 0.3, ns)
        f["/xpcs/qmap/dynamic_v_list_dim1"] = np.zeros(1)
        f["/xpcs/qmap/static_v_list_dim1"] = np.zeros(1)
        f["/xpcs/qmap/mask"] = np.ones((2, nq), dtype=int)
        f["/xpcs/qmap/beam_center_x"] = 1.0
        f["/xpcs/qmap/beam_center_y"] = 1.0
        f["/xpcs/qmap/detector_distance"] = 5000.0
        f["/xpcs/qmap/pixel_size"] = 0.075
        f["/xpcs/qmap/dynamic_index_mapping"] = np.arange(nq)
        f["/xpcs/qmap/static_index_mapping"] = np.arange(ns)
        f["/xpcs/qmap/dynamic_num_pts"] = np.array([nq, 1])
        f["/xpcs/qmap/static_num_pts"] = np.array([ns, 1])
        f["/xpcs/qmap/map_names"] = np.array([b"q", b"phi"])
        f["/xpcs/qmap/map_units"] = np.array([b"1/A", b"deg"])
        f["/xpcs/temporal_mean/scattering_1d"] = np.asarray(saxs, dtype=float)[None]
        f["/xpcs/temporal_mean/scattering_1d_segments"] = np.ones((2, 1, ns))
        f["/xpcs/temporal_mean/scattering_2d"] = np.ones((1, 2, nq))
        f["/xpcs/spatial_mean/intensity_vs_time"] = np.ones((1, 10))
        if processed is not None:
            f["/xpcs/twotime/processed_bins"] = processed
            f["/xpcs/twotime/normalized_g2"] = np.ones((nt, 1))
            f["/xpcs/twotime/normalized_g2_segments"] = np.ones((nt, 1, 1))
            f["/xpcs/twotime/delay_list"] = tau
            f["/xpcs/twotime/config/stride_frame"] = 1
            f["/xpcs/twotime/config/avg_frame"] = 1
            f["/xpcs/twotime/correlation_map/c2__1"] = np.ones((4, 4))


@pytest.fixture
def xf_factory(tmp_path):
    counter = iter(range(1000))

    def make(tau=(0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0), **kw):
        path = tmp_path / f"f{next(counter)}.h5"
        _write_file(path, tau, **kw)
        return XpcsFile(str(path))

    return make


def _fit(xf, **kw):
    kw.setdefault("bounds", BOUNDS)
    kw.setdefault("fit_flag", [True] * 4)
    return xf.fit_g2(**kw)


# --------------------------------------------------------------------------
# legacy solver paths (R3-21)
# --------------------------------------------------------------------------
def _exp_args(n_obs, n_cols=2):
    x = np.logspace(-2, 0, n_obs)
    y = np.stack([1.0 + 0.5 * np.exp(-2 * x / 0.1)] * n_cols, axis=1)
    return x, y, np.full_like(y, 0.01)


def _single_exp_all_args(n_obs, n_cols=2):
    x, y, sigma = _exp_args(n_obs, n_cols)
    return legacy.single_exp_all, x, y, sigma, BOUNDS, np.ones(4, bool), x


P0 = np.array([0.5, 0.1, 1.0, 1.0])


def test_parallel_worker_underdetermined_fit_is_flagged():
    out = fit_with_fixed_parallel(
        *_single_exp_all_args(2), p0=P0, use_threads=True, return_status=True
    )
    assert len(out) == 3
    assert not out[2].any()


def test_parallel_worker_evaluation_limit_is_flagged(monkeypatch):
    real = legacy.curve_fit
    monkeypatch.setattr(
        legacy, "curve_fit", lambda *a, **k: real(*a, **{**k, "max_nfev": 1})
    )
    args = _single_exp_all_args(12)
    serial = fit_with_fixed(*args, p0=P0, return_status=True)[2]
    parallel = fit_with_fixed_parallel(*args, p0=P0, return_status=True)[2]
    assert not serial.any()
    assert not parallel.any()


def test_parallel_converged_fit_reports_success():
    # legacy.single_exp_all is a * exp(..) + c + d: c and d are degenerate (R4-01
    # now rejects it), so a converged, identifiable fit needs the GUI's model.
    from xpcsviewer.xpcs_file.fitting import single_exp_all

    ok = fit_with_fixed_parallel(
        single_exp_all,
        *_single_exp_all_args(12)[1:],
        p0=P0,
        use_threads=True,
        return_status=True,
    )[2]
    assert ok.all()


def test_parallel_default_return_is_still_a_pair():
    assert len(fit_with_fixed_parallel(*_single_exp_all_args(12), p0=P0)) == 2


class _Result:
    def __init__(self, popt, pcov, success=True):
        self.popt, self.pcov, self.success = popt, pcov, success

    def __iter__(self):
        return iter((self.popt, self.pcov))


def test_sequential_unavailable_covariance_is_nan_not_invented(monkeypatch):
    monkeypatch.setattr(
        legacy,
        "curve_fit",
        lambda *a, **k: _Result(P0.copy(), np.full((4, 4), np.inf)),
    )
    _, fit_val, _, ok = fit_with_fixed_sequential(
        *_single_exp_all_args(12), p0=P0, return_status=True
    )
    assert ok.all()  # the solver converged; only the covariance is unavailable
    assert np.isnan(fit_val[:, 1, :]).all()
    assert not np.any(np.isin(fit_val[:, 1, :], [1000.0, 1e6, *(0.1 * P0)]))


def test_sequential_underdetermined_fit_is_flagged():
    _, _, _, ok = fit_with_fixed_sequential(
        *_single_exp_all_args(2), p0=P0, return_status=True
    )
    assert not ok.any()


def test_sequential_not_converged_is_flagged(monkeypatch):
    monkeypatch.setattr(
        legacy,
        "curve_fit",
        lambda *a, **k: _Result(P0.copy(), np.eye(4), success=False),
    )
    _, _, _, ok = fit_with_fixed_sequential(
        *_single_exp_all_args(12), p0=P0, return_status=True
    )
    assert not ok.any()


def test_sequential_solver_failure_is_flagged_without_invented_errors(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("solver failure")

    monkeypatch.setattr(legacy, "curve_fit", boom)
    _, fit_val, methods, ok = fit_with_fixed_sequential(
        *_single_exp_all_args(12), p0=P0, return_status=True
    )
    assert not ok.any()
    assert not np.any(fit_val[:, 1, :] == 1000.0)
    assert not np.any(fit_val[:, 1, :] == 1e6)
    assert all(m.startswith("fallback") for m in methods)


def test_sequential_default_return_is_still_a_triple():
    assert len(fit_with_fixed_sequential(*_single_exp_all_args(12), p0=P0)) == 3


def test_sequential_fitting_never_returns_invented_covariance(monkeypatch):
    monkeypatch.setattr(
        legacy, "curve_fit", lambda *a, **k: _Result(P0, np.inf * np.eye(4))
    )
    popt, pcov, method = legacy.sequential_fitting(
        lambda x, *p: x, np.arange(5.0), np.arange(5.0), p0=P0, bounds=(P0 * 0, P0 * 9)
    )
    assert method == "nlsq_partial"
    assert not np.any(np.isfinite(pcov))

    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(legacy, "curve_fit", boom)
    _, pcov, method = legacy.sequential_fitting(
        lambda x, *p: x, np.arange(5.0), np.arange(5.0), p0=P0, bounds=(P0 * 0, P0 * 9)
    )
    assert method == "fallback"
    assert not np.any(np.isfinite(pcov))


def test_robust_curve_fit_failure_is_nan_not_ones(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(legacy, "curve_fit", boom)
    popt, pcov = legacy.robust_curve_fit(
        lambda x, a, b: a * x + b, np.arange(4.0), np.arange(4.0)
    )
    assert np.isnan(popt).all()
    assert np.isnan(pcov).all()


# --------------------------------------------------------------------------
# XpcsFile.fit_g2 (R3-22), real file
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["serial", "robust"])
def test_fit_g2_underdetermined_marks_fit_failed(xf_factory, mode):
    xf = xf_factory(tau=(0.0, 0.01, 0.1, 1.0, 10.0))
    kw = {"robust_fitting": True} if mode == "robust" else {}
    fs = _fit(xf, t_range=(0.005, 0.5), **kw)  # delays 0.01 and 0.1: 2 points, 4 params
    ok = fs["fit_success"]
    assert ok.dtype == bool
    assert ok.shape == (fs["q_val"].size,)
    assert not ok.any()
    assert np.isnan(fs["fit_val"]).all()
    assert np.isnan(fs["fit_line"]).all()


def test_fit_g2_parallel_underdetermined_marks_fit_failed(xf_factory):
    xf = xf_factory(
        tau=(0.0, 0.01, 0.1, 1.0, 10.0),
        q=tuple(0.1 * np.arange(1, 7)),
    )
    fs = _fit(xf, t_range=(0.005, 0.5), use_parallel=True)
    assert not fs["fit_success"].any()
    assert np.isnan(fs["fit_val"]).all()


def test_fit_g2_good_fit_carries_success_and_finite_values(xf_factory):
    xf = xf_factory(tau=np.logspace(-2.5, 1, 14))
    fs = _fit(xf)
    assert fs["fit_success"].all()
    assert np.isfinite(fs["fit_val"][:, 0, :]).all()
    np.testing.assert_allclose(fs["fit_val"][:, 0, 1], TRUE_TAU, rtol=5e-2)


def test_fit_g2_one_failed_q_only_blanks_that_q(xf_factory, monkeypatch):
    xf = xf_factory(tau=np.logspace(-2.5, 1, 14))
    real = legacy.fit_with_fixed

    def fail_middle(*a, **k):
        line, val, ok = real(*a, **{**k, "return_status": True})
        ok = ok.copy()
        ok[1] = False
        return line, val, ok

    monkeypatch.setattr("xpcsviewer.fitting.fit_with_fixed", fail_middle)
    fs = _fit(xf)
    assert fs["fit_success"].tolist() == [True, False, True]
    assert np.isnan(fs["fit_val"][1]).all()
    assert np.isnan(fs["fit_line"][1]).all()
    assert np.isfinite(fs["fit_val"][[0, 2]]).all()
    assert np.isfinite(fs["fit_line"][[0, 2]]).all()


def test_fit_g2_robust_good_fit_has_success_mask(xf_factory):
    xf = xf_factory(tau=np.logspace(-2.5, 1, 14))
    fs = _fit(xf, robust_fitting=True)
    assert fs["fit_success"].shape == (3,)
    assert fs["fit_success"].dtype == bool
    # noise-free exponential data: the robust solver must succeed, not always fail
    assert fs["fit_success"].all()
    np.testing.assert_allclose(fs["fit_val"][:, 0, 1], TRUE_TAU, rtol=5e-2)


def test_fit_g2_high_performance_underdetermined_marks_fit_failed(xf_factory):
    xf = xf_factory(tau=(0.0, 0.01, 0.1, 1.0, 10.0))
    fs = xf.fit_g2_high_performance(
        t_range=(0.005, 0.5), bounds=BOUNDS, fit_flag=[True] * 4, fit_func="single"
    )
    assert not fs["fit_success"].any()
    assert np.isnan(fs["fit_val"]).all()
    assert np.isnan(fs["fit_line"]).all()


# --------------------------------------------------------------------------
# R3-26 zero delay
# --------------------------------------------------------------------------
def _assert_zero_delay_fit_ok(fs):
    assert fs["fit_success"].all()
    np.testing.assert_allclose(fs["fit_val"][:, 0, 1], TRUE_TAU, rtol=5e-2)
    fit_x = fs["fit_x"]
    assert np.isfinite(fit_x).all()
    assert (fit_x > 0).all()
    assert np.isfinite(fs["fit_line"]).all()
    assert (fs["t_el"] > 0).all()  # fitted delays exclude the zero delay


def test_fit_g2_zero_delay_uses_positive_delays_and_finite_grid(xf_factory, caplog):
    xf = xf_factory(tau=(0.0, *np.logspace(-2.5, 1, 14)))
    with caplog.at_level(logging.INFO):
        fs = _fit(xf)
    _assert_zero_delay_fit_ok(fs)
    assert any("non-positive" in r.getMessage() for r in caplog.records)


def test_fit_g2_high_performance_zero_delay(xf_factory):
    xf = xf_factory(tau=(0.0, *np.logspace(-2.5, 1, 14)))
    fs = xf.fit_g2_high_performance(
        bounds=BOUNDS, fit_flag=[True] * 4, fit_func="single"
    )
    _assert_zero_delay_fit_ok(fs)


def test_fit_g2_only_zero_delay_raises_clearly(xf_factory):
    xf = xf_factory(tau=(0.0, 0.01, 0.1, 1.0, 10.0))
    with pytest.raises(ValueError, match="positive"):
        _fit(xf, t_range=(-1.0, 0.0))


# --------------------------------------------------------------------------
# tau-Q derived fit (R3-25), real file
# --------------------------------------------------------------------------
def _tauq(xf, q_range):
    return xf.fit_tauq(q_range, TAUQ_BOUNDS, np.array([True, True]))


def test_fit_tauq_skips_failed_q_bins_explicitly(xf_factory):
    xf = xf_factory(tau=np.logspace(-2.5, 1, 14))
    fs = _fit(xf)
    # an old / foreign summary: finite junk in a failed bin, only the mask says no
    fs["fit_success"] = np.array([True, False, True])
    fs["fit_val"][1, 0, 1] = 123.0
    fs["fit_val"][1, 1, 1] = 1.0
    out = _tauq(xf, (0.0, 1.0))
    np.testing.assert_allclose(out["tauq_q"], [0.1, 0.3])
    assert 123.0 not in out["tauq_tau"]


def test_fit_tauq_failure_clears_previous_observations(xf_factory):
    xf = xf_factory(tau=np.logspace(-2.5, 1, 14))
    _fit(xf)
    ok = _tauq(xf, (0.05, 0.5))
    assert ok["tauq_success"]
    assert ok["tauq_q"].size == 3
    bad = _tauq(xf, (0.5, 0.6))
    assert not bad["tauq_success"]
    for key in ("tauq_q", "tauq_tau", "tauq_tau_err"):
        assert np.asarray(bad[key]).size == 0, key
    assert bad["tauq_fit_line"] is None
    assert np.isnan(bad["tauq_fit_val"]).all()


def test_fit_tauq_all_g2_fits_failed_has_no_observations(xf_factory):
    xf = xf_factory(tau=(0.0, 0.01, 0.1, 1.0, 10.0))
    _fit(xf, t_range=(0.005, 0.5))
    out = _tauq(xf, (0.0, 1.0))
    assert not out["tauq_success"]
    assert np.asarray(out["tauq_q"]).size == 0


class _Hdl:
    def __init__(self):
        self.ax = MagicMock()
        self.fig = MagicMock()

    def clear(self):
        pass

    def subplots(self, *a, **k):
        return self.ax

    def draw(self):
        pass


def test_tauq_plot_draws_no_stale_points_after_failure(xf_factory):
    xf = xf_factory(tau=np.logspace(-2.5, 1, 14))
    _fit(xf)
    _tauq(xf, (0.05, 0.5))
    _tauq(xf, (0.5, 0.6))
    hdl = _Hdl()
    tauq.plot([xf], hdl, (0.5, 0.6), 0)
    hdl.ax.errorbar.assert_not_called()
    hdl.ax.plot.assert_not_called()


def test_tauq_plot_without_tauq_keys_skips_failed_q(xf_factory):
    xf = xf_factory(tau=np.logspace(-2.5, 1, 14))
    fs = _fit(xf)
    fs["fit_success"] = np.array([True, False, True])
    fs["fit_val"][1, 0, 1] = 123.0
    fs["fit_val"][1, 1, 1] = 1.0
    hdl = _Hdl()
    tauq.plot([xf], hdl, (0.0, 1.0), 0)
    xs = hdl.ax.errorbar.call_args.args[0]
    np.testing.assert_allclose(xs, [0.1, 0.3])


# --------------------------------------------------------------------------
# G2 plot consumers
# --------------------------------------------------------------------------
T = np.logspace(-3, 0, 5)


class _PgHdl:
    def __init__(self):
        self.axes = []

    def adjust_canvas_size(self, **_):
        pass

    def clear(self):
        self.axes = []

    def addPlot(self, row=0, col=0):  # noqa: N802
        ax = MagicMock()
        self.axes.append(ax)
        return ax


@pytest.fixture
def drawn(monkeypatch):
    calls, notes = [], []
    monkeypatch.setattr(
        g2mod,
        "pg_plot_one_g2",
        lambda ax, x, y, *a, **k: calls.append(np.asarray(y, dtype=float)),
    )
    monkeypatch.setattr(
        g2mod, "_add_fit_param_annotation", lambda *a, **k: notes.append(a)
    )
    return calls, notes


def _failed_summary(with_mask=True, finite_junk=False):
    val = np.full((1, 2, 4), np.nan)
    line = np.full((1, 8), np.nan)
    if finite_junk:
        val[:] = [0.2, 0.003, 1.2, 1.03]
        line[:] = 1.03
    summary = {"fit_val": val, "fit_line": line, "fit_x": np.logspace(-3, 0, 8)}
    if with_mask:
        summary["fit_success"] = np.array([False])
    return summary


class _Xf:
    atype = ["Multitau"]
    label = "a"

    def __init__(self, summary):
        self.summary = summary

    def get_g2_data(self, qrange=None, trange=None):
        g2 = np.array([[1.2], [1.1], [1.05], [1.02], [1.0]])
        return [0.01], T, g2, np.zeros((5, 1)), ["q0.01"]

    def fit_g2(self, *a, **k):
        return self.summary


@pytest.mark.parametrize(
    "summary",
    [
        _failed_summary(),  # NaN placeholders + mask
        _failed_summary(finite_junk=True),  # finite placeholders, mask is authoritative
    ],
)
def test_pg_plot_failed_q_has_no_fit_line_annotation_or_baseline_shift(drawn, summary):
    calls, notes = drawn
    hdl = _PgHdl()
    g2mod.pg_plot(
        hdl,
        [_Xf(summary)],
        None,
        None,
        None,
        y_auto=True,
        t_auto=True,
        show_fit=True,
        bounds=[[0] * 4, [1] * 4],
    )
    hdl.axes[0].plot.assert_not_called()
    assert notes == []
    # data curve kept, default baseline 1.0 (no NaN, no fit-parameter shift)
    np.testing.assert_allclose(calls[0], [1.2, 1.1, 1.05, 1.02, 1.0])


def test_pg_plot_from_data_failed_q_has_no_fit_line_annotation_or_baseline_shift(
    drawn,
):
    calls, notes = drawn
    hdl = _PgHdl()
    g2 = [np.array([[1.2], [1.1], [1.05], [1.02], [1.0]])]
    g2mod.pg_plot_from_data(
        hdl,
        q=[[0.01]],
        tel=[T],
        g2=g2,
        g2_err=[np.zeros_like(g2[0])],
        labels=[["q0.01"]],
        num_figs=1,
        plot_type="multiple",
        show_fit=True,
        fit_results=[_failed_summary(finite_junk=True)],
    )
    hdl.axes[0].plot.assert_not_called()
    assert notes == []
    np.testing.assert_allclose(calls[0], [1.2, 1.1, 1.05, 1.02, 1.0])


def test_pg_plot_nan_baseline_without_mask_does_not_blank_curve(drawn):
    calls, _ = drawn
    summary = _failed_summary(with_mask=False)  # legacy summary, NaN values only
    g2mod.pg_plot(
        _PgHdl(),
        [_Xf(summary)],
        None,
        None,
        None,
        y_auto=True,
        t_auto=True,
        show_fit=True,
        bounds=[[0] * 4, [1] * 4],
    )
    assert np.isfinite(calls[0]).all()


def test_pg_plot_successful_q_still_draws_fit(drawn):
    calls, notes = drawn
    summary = _failed_summary(finite_junk=True)
    summary["fit_success"] = np.array([True])
    hdl = _PgHdl()
    g2mod.pg_plot(
        hdl,
        [_Xf(summary)],
        None,
        None,
        None,
        y_auto=True,
        t_auto=True,
        show_fit=True,
        bounds=[[0] * 4, [1] * 4],
    )
    hdl.axes[0].plot.assert_called_once()
    assert len(notes) == 1


# --------------------------------------------------------------------------
# export (viewer_kernel.export_g2)
# --------------------------------------------------------------------------
def test_export_g2_marks_failed_q_without_placeholder_numbers(
    xf_factory, tmp_path, monkeypatch
):
    from xpcsviewer.viewer_kernel import ViewerKernel

    xf = xf_factory(tau=np.logspace(-2.5, 1, 14))
    fs = _fit(xf)
    fs["fit_success"] = np.array([True, False, True])
    # finite placeholders must not leak into the file for the failed Q
    fs["fit_val"][1] = 777.0
    fs["fit_line"][1] = 777.0
    vk = ViewerKernel.__new__(ViewerKernel)
    monkeypatch.setattr(vk, "get_xf_list", lambda **k: [xf], raising=False)
    out = tmp_path / "out"
    out.mkdir()
    vk.export_g2(str(out))

    fit_file = next(out.glob("*_g2_fitting.txt")).read_text()
    assert "777" not in fit_file
    rows = [
        ln.split("\t")
        for ln in fit_file.splitlines()
        if ln[:1].isdigit() and "\t" in ln  # parameter rows (the bounds comment wraps)
    ]
    assert [r[-1] for r in rows] == ["ok", "failed", "ok"]
    assert "fit_status" in fit_file
    assert all(v == "nan" for v in rows[1][1:-1])
    curves = np.loadtxt(next(out.glob("*_g2_fitted_curves.txt")))
    assert np.isnan(curves[:, 2]).all()
    assert np.isfinite(curves[:, [1, 3]]).all()
    assert 777.0 not in curves


def test_get_fitting_info_reports_failed_q_instead_of_nan_parameters(xf_factory):
    xf = xf_factory(tau=(0.0, 0.01, 0.1, 1.0, 10.0))
    _fit(xf, t_range=(0.005, 0.5))
    info = xf.get_fitting_info("g2_fitting")
    assert all("failed" in m for m in info["fit_val"])


# --------------------------------------------------------------------------
# R3-15 scalar one-bin processed list (real file)
# --------------------------------------------------------------------------
def test_scalar_processed_bin_does_not_crash_labels_or_c2(xf_factory):
    xf = xf_factory(tau=(0.01, 0.1, 1.0), processed=np.array([[1]]))
    assert np.ndim(xf.c2_processed_bins) >= 1
    labels = xf.get_twotime_qbin_labels()
    assert len(labels) == 1
    res = xf.get_twotime_c2(selection=0, correct_diag=False)
    assert res is not None


# --------------------------------------------------------------------------
# R3-27 / R3-28 SAXS (real file)
# --------------------------------------------------------------------------
def test_saxs_i_over_i0_normalises_each_curve_by_its_own_first_point(xf_factory):
    xf = xf_factory(saxs=(10.0, 5.0, 2.0))
    _, iq, _, ylabel = xf.get_saxs1d_data(norm_method="I0")
    np.testing.assert_allclose(iq, [[1.0, 0.5, 0.2]])
    assert "I_0" in ylabel


def test_saxs_i0_uses_first_valid_nonzero_point_per_curve():
    xf = XpcsFile.__new__(XpcsFile)
    iq = np.array(
        [[0.0, np.nan, 4.0, 2.0], [2.0, 1.0, 0.5, 0.25], [0.0, 0.0, 0.0, 0.0]]
    )
    _, out, _, _ = xf.norm_saxs_data(np.arange(4.0), iq, norm_method="I0")
    np.testing.assert_allclose(out[0], [0.0, np.nan, 1.0, 0.5])
    np.testing.assert_allclose(out[1], [1.0, 0.5, 0.25, 0.125])
    assert np.isnan(out[2]).all()  # no reference point: unnormalisable, never inf


def test_saxs_background_with_other_q_count_keeps_sample_and_warns(xf_factory, caplog):
    sample = xf_factory(saxs=(10.0, 5.0, 2.0))
    bkg = xf_factory(saxs=(1.0, 1.0))
    with caplog.at_level(logging.WARNING):
        q, iq, _, _ = sample.get_saxs1d_data(bkg_xf=bkg, bkg_weight=1.0)
    np.testing.assert_allclose(iq, [[10.0, 5.0, 2.0]])
    assert q.shape == (3,)
    assert any("background" in r.getMessage() for r in caplog.records)


def test_saxs_background_with_same_q_is_still_subtracted(xf_factory):
    sample = xf_factory(saxs=(10.0, 5.0, 2.0))
    bkg = xf_factory(saxs=(1.0, 1.0, 1.0))
    _, iq, _, _ = sample.get_saxs1d_data(bkg_xf=bkg, bkg_weight=1.0)
    np.testing.assert_allclose(iq, [[9.0, 4.0, 1.0]])


def test_saxs_absolute_scale_does_not_mutate_stored_curve(xf_factory):
    xf = xf_factory(saxs=(10.0, 5.0, 2.0))
    xf.abs_cross_section_scale = 2.0
    first = xf.get_saxs1d_data(use_absolute_crosssection=True)[1]
    second = xf.get_saxs1d_data(use_absolute_crosssection=True)[1]
    np.testing.assert_allclose(first, [[20.0, 10.0, 4.0]])
    np.testing.assert_allclose(second, first)


# keep an import used above referenced for linters
_ = make_single_exp
