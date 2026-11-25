"""Regression tests for review round 3: R3-18 (bin identity) and R3-19 (empty selection)."""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from xpcsviewer.fileIO.qmap_utils import QMap
from xpcsviewer.module import g2mod
from xpcsviewer.xpcs_file import XpcsFile

pytestmark = pytest.mark.unit

T = np.logspace(-3, 0, 5)


class _Hdl:
    def __init__(self):
        self.axes = []
        self.clears = 0

    def adjust_canvas_size(self, **_):
        pass

    def clear(self):
        self.clears += 1
        self.axes = []

    def addPlot(self, row=0, col=0):  # noqa: N802
        ax = MagicMock(name=f"panel{len(self.axes)}")
        self.axes.append(ax)
        return ax


@pytest.fixture
def drawn(monkeypatch):
    """Record (panel index, colour, label, y) for every curve drawn."""
    calls = []
    hdl = _Hdl()

    def record(ax, x, y, dy, color, label, symbol, symbol_size=5):
        calls.append((hdl.axes.index(ax), tuple(color), label, np.asarray(y, float)))

    monkeypatch.setattr(g2mod, "pg_plot_one_g2", record)
    monkeypatch.setattr(g2mod, "_add_fit_param_annotation", lambda *a, **k: None)
    return hdl, calls


# ------------------------------------------------------------ real labels
def _real_file(dq, dp=(0.0,), label="f"):
    """File whose G2 data and labels come from the real QMap/get_g2_data code."""
    qm = QMap.__new__(QMap)
    qm._qbin_cache = {}
    qm.map_names = ["q", "phi"]
    qm.map_units = ["Å⁻¹", "°"]
    qm.dqlist = np.asarray(dq, dtype=float)
    qm.dplist = np.asarray(dp, dtype=float)
    qm.dynamic_num_pts = (len(dq), len(dp))
    qm.dynamic_index_mapping = np.arange(len(dq) * len(dp))
    qm.qbin_labels = qm.create_qbin_labels()
    n = qm.dynamic_index_mapping.size
    ns = types.SimpleNamespace(
        atype=["Multitau"],
        qmap=qm,
        label=label,
        g2=np.full((T.size, n), 1.2),
        g2_err=np.zeros((T.size, n)),
        t_el=T,
    )
    ns.get_g2_data = types.MethodType(XpcsFile.get_g2_data, ns)
    return ns


def _data(*files, q_range=None):
    out = [f.get_g2_data(qrange=q_range) for f in files]
    q = [np.asarray(o[0]) for o in out]
    g2 = [o[2] for o in out]
    return q, [o[1] for o in out], g2, [o[3] for o in out], [o[4] for o in out]


def _render(hdl, files, plot_type="multiple", **kw):
    q, tel, g2, g2_err, labels = _data(*files)
    num_figs, _ = g2mod.compute_geometry(g2, plot_type)
    g2mod.pg_plot_from_data(
        hdl,
        q=q,
        tel=tel,
        g2=g2,
        g2_err=g2_err,
        labels=labels,
        num_figs=num_figs,
        plot_type=plot_type,
        **kw,
    )
    return labels


# ---------------------------------------------------------------- R3-18
def test_r3_18_real_labels_collide_but_bins_are_not_overlaid(drawn):
    hdl, calls = drawn
    both = _real_file([0.100001, 0.100004])
    only_second = _real_file([0.100004])
    labels = _render(hdl, [both, only_second])
    # the genuine QMap labels of the two distinct bins are identical text ...
    assert labels[0][0] == labels[0][1] == labels[1][0]
    # ... yet file 1's Q=0.100004 curve belongs to panel 1, not panel 0
    assert sorted(p for p, *_ in calls) == [0, 1, 1]
    assert calls[-1][0] == 1


def test_r3_18_real_labels_other_bin_is_skipped_not_overlaid(drawn):
    hdl, calls = drawn
    _render(hdl, [_real_file([0.100004]), _real_file([0.100001])])
    assert [p for p, *_ in calls] == [0]  # file 1's different bin has no panel


def test_r3_18_reordered_file_with_identical_labels_matches_numerically():
    a = _real_file([0.100001, 0.100004])
    b = _real_file([0.100004, 0.100001])
    q, _, _, _, labels = _data(a, b)
    assert labels[0][0] == labels[0][1]
    assert g2mod._q_panel_indices(q, 1, 2, labels) == [1, 0]


def test_r3_18_identical_grids_stay_positional():
    q, _, _, _, labels = _data(_real_file([0.1, 0.2]), _real_file([0.1, 0.2]))
    assert g2mod._q_panel_indices(q, 1, 2, labels) == [0, 1]


def test_r3_18_same_q_distinct_azimuth_still_one_to_one():
    a = _real_file([0.1], dp=[0.0, 90.0])
    b = _real_file([0.1], dp=[90.0, 0.0])
    q, _, _, _, labels = _data(a, b)
    assert g2mod._q_panel_indices(q, 1, 2, labels) == [1, 0]


def test_r3_18_equal_q_but_different_label_is_not_matched():
    q = [np.array([0.1]), np.array([0.1])]
    assert g2mod._q_panel_indices(q, 1, 1, [["phi=0"], ["phi=90"]]) == [None]


# ---------------------------------------------------------------- R3-19
class _FakeXf:
    atype = ["Multitau"]

    def __init__(self, label, n_q):
        self.label = label
        self.n_q = n_q

    def get_g2_data(self, qrange=None, trange=None):
        n = self.n_q
        q = [0.01 * (i + 1) for i in range(n)]
        return (
            q,
            T,
            np.full((T.size, n), 1.2),
            np.zeros((T.size, n)),
            [f"q{v}" for v in q],
        )

    def fit_g2(self, *a, **k):
        raise AssertionError("a file with no Q bins must not be fitted")


def test_r3_19_pg_plot_empty_selection_clears_previous_curve(drawn):
    hdl, calls = drawn
    g2mod.pg_plot(hdl, [_FakeXf("a", 2)], None, None, None, y_auto=True, t_auto=True)
    clears = hdl.clears
    assert calls
    g2mod.pg_plot(hdl, [_FakeXf("a", 0)], None, None, None, y_auto=True, t_auto=True)
    assert hdl.clears == clears + 1
    assert hdl.axes == []


def test_r3_19_pg_plot_from_data_empty_selection_clears(drawn):
    hdl, _ = drawn
    g2mod.pg_plot_from_data(
        hdl,
        q=[[]],
        tel=[T],
        g2=[np.zeros((T.size, 0))],
        g2_err=[np.zeros((T.size, 0))],
        labels=[[]],
        num_figs=0,
    )
    assert hdl.clears == 1


def test_r3_19_pg_plot_stability_empty_selection_clears():
    hdl = _Hdl()
    xf = types.SimpleNamespace(
        atype=["Multitau"],
        get_g2_stability_data=lambda qrange=None, trange=None: (
            [],
            T,
            [np.zeros((T.size, 0))],
            [np.zeros((T.size, 0))],
            [],
            [0],
        ),
    )
    g2mod.pg_plot_stability(hdl, xf, None, None, None, y_auto=True, t_auto=True)
    assert hdl.clears == 1


def test_r3_19_geometry_uses_first_usable_file():
    empty, three = np.zeros((5, 0)), np.zeros((5, 3))
    assert g2mod.compute_geometry([empty, three], "multiple") == (3, 2)
    assert g2mod.compute_geometry([empty, three], "single") == (2, 3)
    assert g2mod.compute_geometry([empty, three], "single-combined") == (1, 6)
    assert g2mod.compute_geometry([empty, empty], "multiple")[0] == 0
    # first file non-empty: first-file semantics unchanged
    assert g2mod.compute_geometry([three, empty], "multiple") == (3, 2)


@pytest.mark.parametrize("plot_type", ["multiple", "single", "single-combined"])
def test_r3_19_empty_first_file_does_not_hide_later_file(drawn, plot_type):
    hdl, calls = drawn
    xfs = [_FakeXf("a", 0), _FakeXf("b", 2), _FakeXf("c", 2)]
    g2mod.pg_plot(
        hdl, xfs, None, None, None, y_auto=True, t_auto=True, plot_type=plot_type
    )
    assert len(calls) == 4  # files b and c, two curves each
    if plot_type == "multiple":
        # colour follows the file index m (files b, c are m=1, 2), not the position
        assert {c for _, c, *_ in calls} == {
            tuple(g2mod.colors[1]),
            tuple(g2mod.colors[2]),
        }
        assert sorted(p for p, *_ in calls) == [0, 0, 1, 1]
        assert {lab for _, _, lab, _ in calls} == {"b", "c"}
        # the panel titles come from the reference file (first usable file)
        assert hdl.axes[0].setTitle.call_args.args[0] == "q0.01"
    if plot_type == "single":
        assert sorted(p for p, *_ in calls) == [1, 1, 2, 2]


def test_r3_19_from_data_keeps_file_index_for_fit_results(drawn):
    hdl, calls = drawn
    val = np.zeros((1, 2, 4))
    val[:, 0, :] = [0.2, 0.003, 1.2, 1.03]
    summary = {
        "fit_val": val,
        "fit_line": np.full((1, 8), 1.03),
        "fit_x": np.logspace(-3, 0, 8),
        "fit_success": np.array([True]),
        "fit_status": np.array([1]),
    }
    g2 = [np.zeros((T.size, 0)), np.full((T.size, 1), 1.2)]
    g2mod.pg_plot_from_data(
        hdl,
        q=[[], [0.01]],
        tel=[T, T],
        g2=g2,
        g2_err=[np.zeros_like(a) for a in g2],
        labels=[[], ["q0.01"]],
        num_figs=1,
        show_fit=True,
        fit_results=[None, summary],
        rows=[3, 4],
    )
    assert len(calls) == 1
    _, colour, _, _ = calls[0]
    assert colour == tuple(g2mod.colors[4 % len(g2mod.colors)])  # row of file index 1
    assert hdl.clears == 1
