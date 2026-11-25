"""Regression tests for review round 2: R2-06 (worker), R2-11, R2-24, R2-36."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from xpcsviewer.module import g2mod, twotime
from xpcsviewer.threading.plot_workers import G2PlotWorker

pytestmark = pytest.mark.unit

T = np.logspace(-3, 0, 5)


class _Hdl:
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
    return hdl, calls


def _render(hdl, q, labels, fills=None, **kw):
    g2 = [np.full((T.size, len(qq)), 1.0 + 0.1 * i) for i, qq in enumerate(q)]
    g2mod.pg_plot_from_data(
        hdl,
        q=q,
        tel=[T] * len(g2),
        g2=g2,
        g2_err=[np.zeros_like(a) for a in g2],
        labels=labels,
        num_figs=len(q[0]),
        **kw,
    )


# ---------------------------------------------------------------- R2-24
Q_DUP = [0.1, 0.1]
LAB_DUP = ["q=0.1 phi=0", "q=0.1 phi=90"]


def test_r2_24_duplicate_radial_q_identical_grids_map_positionally(drawn):
    hdl, calls = drawn
    _render(hdl, [Q_DUP, Q_DUP], [LAB_DUP, LAB_DUP])
    assert sorted(p for p, _ in calls) == [0, 0, 1, 1]


def test_r2_24_duplicate_q_reordered_file_matches_by_label(drawn):
    hdl, calls = drawn
    _render(hdl, [Q_DUP, Q_DUP], [LAB_DUP, LAB_DUP[::-1]])
    # file 1 curve 0 is phi=90 -> panel 1; curve 1 is phi=0 -> panel 0
    assert [p for p, _ in calls] == [0, 1, 1, 0]


def test_r2_24_extra_curve_is_skipped_with_warning(drawn, caplog):
    hdl, calls = drawn
    with caplog.at_level(logging.WARNING):
        _render(hdl, [Q_DUP, Q_DUP + [0.1]], [LAB_DUP, LAB_DUP + ["q=0.1 phi=180"]])
    assert sorted(p for p, _ in calls) == [0, 0, 1, 1]
    assert any("skipped" in r.getMessage() for r in caplog.records)


def test_r2_24_panel_helper_never_reuses_a_panel_without_labels():
    out = g2mod._q_panel_indices([[0.1, 0.1], [0.1, 0.1, 0.1]], 1, 3)
    assert out[:2] == [0, 1] and out[2] is None


# ---------------------------------------------------------------- R2-11
def _annotation_text(monkeypatch, fit_val, fit_func):
    seen = {}

    def fake_text(text, **_):
        seen["text"] = text
        return MagicMock()

    monkeypatch.setattr(g2mod.pg, "TextItem", fake_text)
    ax = MagicMock()
    ax.getViewBox.return_value.viewRange.return_value = [[0, 1], [0, 1]]
    g2mod._add_fit_param_annotation(ax, fit_val, fit_func, (0, 0, 0))
    return dict(line.split(" = ", 1) for line in seen["text"].split("\n"))


def test_r2_11_single_exp_labels_follow_model_order(monkeypatch):
    # single_exp_all(x, a=contrast, b=tau, c=stretch, d=baseline)
    val = np.array([[0.2, 0.003, 1.2, 1.03], [0.01, 0.0001, 0.02, 0.03]])
    lines = _annotation_text(monkeypatch, val, "single")
    assert lines["τ"].startswith("3.000e-03")
    assert lines["baseline"].startswith("1.030")
    assert lines["contrast"].startswith("0.200")
    assert lines["stretch"].startswith("1.200")


def test_r2_11_double_exp_labels_every_param(monkeypatch):
    # double_exp_all(x, a, b1, c1, d, b2, c2, f)
    val = np.zeros((2, 7))
    val[0] = [0.2, 0.003, 1.1, 1.03, 0.03, 0.9, 0.4]
    lines = _annotation_text(monkeypatch, val, "double")
    assert len(lines) == 7
    assert lines["τ1"].startswith("3.000e-03")
    assert lines["τ2"].startswith("3.000e-02")
    assert lines["baseline"].startswith("1.030")
    assert lines["f"].startswith("0.400")
    assert lines["stretch1"].startswith("1.100")
    assert lines["stretch2"].startswith("0.900")
    assert lines["contrast"].startswith("0.200")


# ---------------------------------------------------------------- R2-36
def test_r2_36_more_partial_segments_than_palette_colours():
    n_seg = len(twotime.PG_COLORS) + 1
    plot = MagicMock()
    hdl = {"c2g2": plot}
    twotime.plot_twotime_g2(
        hdl,
        {
            "g2_full": np.ones(8),
            "g2_partial": np.ones((n_seg, 8)),
            "acquire_period": 1.0,
        },
    )
    assert plot.plot.call_count == n_seg + 1  # every segment plus the full curve


# ---------------------------------------------------------------- R2-06 (worker)
class _FakeXf:
    atype = ["Multitau"]
    label = "f"
    fname = "f.h5"

    def __init__(self):
        self.get_calls = []
        self.fit_calls = []

    def get_g2_data(self, qrange=None, trange=None):
        self.get_calls.append((qrange, trange))
        return [0.01], T, np.ones((T.size, 1)), np.zeros((T.size, 1)), ["q"]

    def fit_g2(self, q_range, t_range, *a, **k):
        self.fit_calls.append((q_range, t_range))
        return


def _run_worker(qapp, **flags):
    xf = _FakeXf()
    kernel = MagicMock()
    kernel.get_xf_list.return_value = [xf]
    kwargs = {
        "q_range": (0.5, 0.6),
        "t_range": (1.0, 2.0),
        "y_range": (0.9, 1.1),
        "show_fit": True,
        "bounds": [[0] * 4, [1] * 4],
        "fit_flag": [True] * 4,
        **flags,
    }
    G2PlotWorker(kernel, None, kwargs).do_work()
    return xf


def test_r2_06_worker_auto_flags_do_not_slice_by_manual_limits(qapp):
    xf = _run_worker(qapp, q_auto=True, t_auto=True)
    assert xf.get_calls == [(None, None)]
    assert xf.fit_calls == [(None, None)]


def test_r2_06_worker_manual_limits_still_applied(qapp):
    xf = _run_worker(qapp, q_auto=False, t_auto=False)
    assert xf.get_calls == [((0.5, 0.6), (1.0, 2.0))]
    assert xf.fit_calls == [((0.5, 0.6), (1.0, 2.0))]


def test_r2_06_worker_flags_are_independent(qapp):
    xf = _run_worker(qapp, q_auto=True, t_auto=False)
    assert xf.get_calls == [(None, (1.0, 2.0))]
