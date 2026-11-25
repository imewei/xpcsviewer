"""Regression tests for review round 1 findings R1-06, R1-13 and R1-20 (G2 rendering)."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from xpcsviewer.module import g2mod
from xpcsviewer.threading.plot_workers import G2PlotWorker

T = np.logspace(-3, 0, 5)


class _Hdl:
    """Minimal plot-handler stand-in; each panel is a MagicMock."""

    def __init__(self):
        self.axes = []

    def adjust_canvas_size(self, **_):
        pass

    def clear(self):
        self.axes = []

    def addPlot(self, row=0, col=0):  # noqa: N802
        ax = MagicMock(name=f"panel{len(self.axes)}")
        self.axes.append(ax)
        return ax


@pytest.fixture
def drawn(monkeypatch):
    """Record (panel index, y) for every curve drawn via pg_plot_one_g2."""
    calls = []
    hdl = _Hdl()

    def record(ax, x, y, dy, color, label, symbol, symbol_size=5):
        calls.append((hdl.axes.index(ax), np.asarray(y, dtype=float)))

    monkeypatch.setattr(g2mod, "pg_plot_one_g2", record)
    monkeypatch.setattr(g2mod, "_add_fit_param_annotation", lambda *a, **k: None)
    return hdl, calls


def _g2(n_q, fill=1.2):
    return np.full((T.size, n_q), fill)


def _render(hdl, q, g2, plot_type="multiple", num_figs=None, **kw):
    if num_figs is None:
        num_figs = {"multiple": g2[0].shape[1], "single": len(g2)}.get(plot_type, 1)
    g2mod.pg_plot_from_data(
        hdl,
        q=q,
        tel=[T] * len(g2),
        g2=g2,
        g2_err=[np.zeros_like(a) for a in g2],
        labels=[[f"q{v}" for v in qq] for qq in q],
        num_figs=num_figs,
        plot_type=plot_type,
        **kw,
    )


# ---------------------------------------------------------------- R1-06
def _fit_summary(n_q=1):
    # single_exp_all parameter order: (a=contrast, b=tau, c=stretch, d=baseline)
    val = np.zeros((n_q, 2, 4))
    val[:, 0, :] = [0.2, 0.003, 1.2, 1.03]
    return {
        "fit_val": val,
        "fit_line": np.full((n_q, 8), 1.03),
        "fit_x": np.logspace(-3, 0, 8),
    }


@pytest.mark.unit
def test_r1_06_from_data_baseline_is_param_3_only(drawn):
    hdl, calls = drawn
    g2 = [np.array([[1.2], [1.03], [1.03], [1.03], [1.03]])]
    _render(hdl, [[0.01]], g2, show_fit=True, fit_results=[_fit_summary()])
    y = calls[0][1]
    np.testing.assert_allclose(y[:2], [1.2 - 1.03 + 1.0, 1.0])
    # fitted line tends to the baseline, so it must be normalised to 1.0
    np.testing.assert_allclose(hdl.axes[0].plot.call_args.args[1], 1.0)


class _FakeXf:
    atype = ["Multitau"]

    def __init__(self, label, q, fit=None, fit_raises=False):
        self.label = label
        self._q = np.asarray(q, dtype=float)
        self._fit = fit
        self._raises = fit_raises

    def get_g2_data(self, qrange=None, trange=None):
        n = self._q.size
        return (
            list(self._q),
            T,
            _g2(n),
            np.zeros((T.size, n)),
            [f"q{v}" for v in self._q],
        )

    def fit_g2(self, *a, **k):
        if self._raises:
            raise RuntimeError("fit failed")
        return self._fit


def _pg_plot(hdl, xfs, **kw):
    g2mod.pg_plot(hdl, xfs, None, None, None, y_auto=True, t_auto=True, **kw)


@pytest.mark.unit
def test_r1_06_pg_plot_baseline_is_param_3_only(drawn):
    hdl, calls = drawn
    xf = _FakeXf("a", [0.01], fit=_fit_summary())
    xf.get_g2_data = lambda qrange=None, trange=None: (
        [0.01],
        T,
        np.array([[1.2], [1.03], [1.03], [1.03], [1.03]]),
        np.zeros((5, 1)),
        ["q0.01"],
    )
    _pg_plot(hdl, [xf], show_fit=True, bounds=[[0] * 4, [1] * 4])
    np.testing.assert_allclose(calls[0][1][:2], [1.17, 1.0])


# ---------------------------------------------------------------- R1-13
@pytest.mark.unit
def test_r1_13_unlike_q_not_overlaid_and_warned(drawn, caplog):
    hdl, calls = drawn
    with caplog.at_level(logging.WARNING):
        _render(hdl, [[0.01], [0.02]], [_g2(1), _g2(1)])
    assert [p for p, _ in calls] == [0]  # only the Q=0.01 curve is drawn
    assert any("0.02" in r.getMessage() for r in caplog.records)


@pytest.mark.unit
def test_r1_13_shorter_second_file_aligned_by_q(drawn):
    hdl, calls = drawn
    _render(hdl, [[0.01, 0.02], [0.02]], [_g2(2), _g2(1)])
    assert sorted(p for p, _ in calls) == [0, 1, 1]  # file 1's 0.02 -> panel 1


@pytest.mark.unit
def test_r1_13_longer_second_file_warns_instead_of_dropping(drawn, caplog):
    hdl, calls = drawn
    with caplog.at_level(logging.WARNING):
        _render(hdl, [[0.01], [0.01, 0.02]], [_g2(1), _g2(2)])
    assert sorted(p for p, _ in calls) == [0, 0]
    assert any("0.02" in r.getMessage() for r in caplog.records)


@pytest.mark.unit
def test_r1_13_single_mode_uses_each_files_own_q_count(drawn):
    hdl, calls = drawn
    _render(hdl, [[0.01], [0.01, 0.02]], [_g2(1), _g2(2)], plot_type="single")
    assert sorted(p for p, _ in calls) == [0, 1, 1]


@pytest.mark.unit
def test_r1_13_empty_selection_does_not_divide_by_zero(drawn):
    hdl, calls = drawn
    _render(hdl, [[], [0.01]], [_g2(0), _g2(1)], num_figs=0)
    assert calls == []


@pytest.mark.unit
def test_r1_13_pg_plot_unlike_q_and_count(drawn, caplog):
    hdl, calls = drawn
    with caplog.at_level(logging.WARNING):
        _pg_plot(hdl, [_FakeXf("a", [0.01, 0.02]), _FakeXf("b", [0.02])])
    assert sorted(p for p, _ in calls) == [0, 1, 1]
    calls.clear()
    with caplog.at_level(logging.WARNING):
        _pg_plot(hdl, [_FakeXf("a", [0.01]), _FakeXf("b", [0.02])])
    assert [p for p, _ in calls] == [0]
    assert any("0.02" in r.getMessage() for r in caplog.records)


@pytest.mark.unit
def test_r1_13_pg_plot_empty_selection(drawn):
    hdl, calls = drawn
    _pg_plot(hdl, [_FakeXf("a", [])])
    assert calls == []


# ---------------------------------------------------------------- R1-20
def _worker():
    return G2PlotWorker(
        viewer_kernel=None,
        plot_handler=None,
        plot_kwargs={
            "bounds": [[0] * 4, [1] * 4],
            "fit_flag": [True] * 4,
            "fit_func": "single",
        },
    )


@pytest.mark.unit
def test_r1_20_worker_returns_position_aligned_list(qapp):
    ok = _fit_summary()
    xfs = [
        _FakeXf("runA", [0.01], fit=ok),
        _FakeXf("runA", [0.01], fit_raises=True),  # same label, failed fit
        _FakeXf("runA", [0.01], fit=ok),
    ]
    res = _worker()._perform_g2_fitting(xfs, None, None)
    assert isinstance(res, list)
    assert [r is None for r in res] == [False, True, False]


@pytest.mark.unit
def test_r1_20_worker_result_renders_fit_line(drawn, qapp):
    hdl, calls = drawn
    xfs = [_FakeXf("runA", [0.01], fit=_fit_summary())]
    fit_results = _worker()._perform_g2_fitting(xfs, None, None)
    _render(hdl, [[0.01]], [_g2(1)], show_fit=True, fit_results=fit_results)
    assert len(calls) == 1
    hdl.axes[0].plot.assert_called_once()
