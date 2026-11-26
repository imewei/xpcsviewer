"""Regression tests for the four round-8 leftovers.

L1  ``QMap.get_qbin_in_qrange`` reads the range on the map's FIRST axis whatever it
    is called; a range on a Cartesian map no longer selects every bin
L2  the two-time CLI accepts the nearest bin only within half a bin spacing
L3  I*q^2 / I*q^4 are refused for a first axis that is not a radial ``q``
L4  a saved / remembered two-time bin that left the file is announced on the
    status bar instead of silently showing the first bin

Files are written with h5py and read back through the real ``XpcsFile``,
``ViewerKernel``, CLI and ``XpcsViewer``.
"""

from __future__ import annotations

import logging

import h5py
import numpy as np
import pyqtgraph as pg
import pytest
from qtpy import QtWidgets

from tests.unit import test_review_r3_maps_roi as r3
from tests.unit import test_review_r5_kernel as r5
from tests.unit import test_review_r6_kernel_units as ku
from tests.unit import test_review_r7_state as st
from tests.unit.test_review_r4_cli_average import _args, _write_twotime
from tests.unit.test_review_r4_viewer_gui import _tab
from tests.unit.test_review_r6_viewer_gui import _make_env, _select
from xpcsviewer.cli import twotime_batch as tb
from xpcsviewer.fileIO.hdf_reader import clear_connection_pool
from xpcsviewer.fileIO.qmap_utils import SaxsNormalizationRefused
from xpcsviewer.module import g2mod
from xpcsviewer.plothandler.pyqtgraph_handler import ImageViewDev
from xpcsviewer.viewer_kernel import ViewerKernel
from xpcsviewer.xpcs_file import XpcsFile

pytestmark = pytest.mark.unit

K = r3.K
qapp = st.qapp


@pytest.fixture(autouse=True)
def _fresh_pool():
    clear_connection_pool()
    yield
    clear_connection_pool()


def _rename_axes(path, names: tuple[bytes, bytes], units: tuple[bytes, bytes]):
    with h5py.File(path, "r+") as f:
        del f[K["map_names"]], f[K["map_units"]]
        f[K["map_names"]] = list(names)
        f[K["map_units"]] = list(units)
    return path


def _g2_axis_file(path, first, names=(b"q", b"phi"), units=(b"1/nm", b"deg")):
    """A G2 file with one bin per value of the first axis (no second axis)."""
    n = len(first)
    g2 = np.array([[1.9] * n, [1.2] * n, [1.0] * n]) + np.arange(n) * 0.01
    r5._write_g2_file(path, list(first), [0.0], list(range(n)), [n, 1], g2)
    return _rename_axes(path, names, units)


# ======================================================================== L1
@pytest.mark.parametrize(
    ("names", "units", "first"),
    [
        ((b"x", b"y"), (b"px", b"px"), [10.0, 20.0, 30.0]),
        ((b"qx", b"qy"), (b"1/nm", b"1/nm"), [-0.2, 0.1, 0.4]),
    ],
    ids=["x-pixels", "qx-signed"],
)
def test_l1_a_range_selects_on_the_first_axis_whatever_it_is_called(
    tmp_path, names, units, first
):
    xf = XpcsFile(str(_g2_axis_file(tmp_path / "c.h5", first, names, units)))
    qmap = xf.qmap
    mid = first[1]
    sel, values = qmap.get_qbin_in_qrange((mid - 1e-3, mid + 1e-3))
    assert sel.tolist() == [1] and values.tolist() == [mid]  # all three before
    sel, values = qmap.get_qbin_in_qrange((first[0], first[1]), zero_based=False)
    assert sel.tolist() == [1, 2] and values.tolist() == first[:2]
    # a range that matches nothing is the empty result, never "all bins"
    sel, values = qmap.get_qbin_in_qrange((1e6, 2e6))
    assert len(sel) == 0 and len(values) == 0
    # None still means every bin
    assert qmap.get_qbin_in_qrange(None)[0].tolist() == [0, 1, 2]


def test_l1_the_g2_data_of_a_cartesian_file_follows_the_typed_range(tmp_path):
    xf = XpcsFile(
        str(
            _g2_axis_file(
                tmp_path / "c.h5", [10.0, 20.0, 30.0], (b"x", b"y"), (b"px", b"px")
            )
        )
    )
    q, _tel, g2, _err, labels = g2mod.get_data([xf], q_range=(15.0, 25.0))
    assert [list(np.ravel(v)) for v in q] == [[20.0]]  # three values before
    assert np.shape(g2[0])[-1] == 1 and len(labels[0]) == 1
    q, *_ = g2mod.get_data([xf], q_range=(100.0, 200.0))
    assert len(q[0]) == 0  # the user never sees a bin outside the typed range


def test_l1_the_q_phi_case_is_unchanged(tmp_path):
    """The ordinary (q, phi) selection: exact arrays, and the same as any axis name."""
    q = [0.1, 0.2, 0.3]
    radial = XpcsFile(str(_g2_axis_file(tmp_path / "q.h5", q))).qmap
    other = XpcsFile(
        str(_g2_axis_file(tmp_path / "x.h5", q, (b"x", b"y"), (b"px", b"px")))
    ).qmap
    sel, values = radial.get_qbin_in_qrange((0.15, 0.25))
    assert (sel.tolist(), values.tolist()) == ([1], [0.2])
    sel, values = radial.get_qbin_in_qrange((0.1, 0.3), zero_based=False)
    assert (sel.tolist(), values.tolist()) == ([1, 2, 3], q)
    for rng, zero_based in (((0.15, 0.25), True), ((0.1, 0.3), False), (None, True)):
        sel, values = radial.get_qbin_in_qrange(rng, zero_based=zero_based)
        sel2, values2 = other.get_qbin_in_qrange(rng, zero_based=zero_based)
        assert sel.tobytes() == sel2.tobytes()
        assert values.tobytes() == values2.tobytes()


# ======================================================================== L2
def _cli(tmp_path, dq, dp=None, **selector):
    root, out = tmp_path / "in", tmp_path / "out"
    root.mkdir(parents=True)
    _write_twotime(root / "scan.h5", dq, dplist=dp)
    status = tb.run_twotime_batch(_args(root, out, **selector))
    images = sorted(p.name for p in out.rglob("*.png")) if out.exists() else []
    return status, images


def test_l2_a_q_far_outside_the_data_writes_nothing(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        status, images = _cli(tmp_path, [0.01, 0.02], q=0.5)
    assert status != 0 and images == []  # a bin-1 image named for q=0.02 before
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "scan.h5" in text and "q=0.5" in text and "q=0.02" in text
    assert "no matching bin" in text and "skipped" in text


@pytest.mark.parametrize(
    ("q", "written"),
    [
        (0.0249, True),  # within half a bin (0.005) above the last bin
        (0.0251, False),
        (0.0151, True),  # between bins: the nearest is always within half a gap
        (0.0051, True),  # half a bin below the first
        (0.0049, False),
    ],
)
def test_l2_q_tolerance_is_half_the_local_bin_spacing(tmp_path, q, written):
    status, images = _cli(tmp_path, [0.01, 0.02], q=q)
    assert (status == 0) is written
    assert bool(images) is written


def test_l2_a_single_bin_axis_uses_a_relative_five_percent(tmp_path):
    assert _cli(tmp_path / "a", [0.02], q=0.0209)[0] == 0  # 4.5 %
    assert _cli(tmp_path / "b", [0.02], q=0.0215)[0] != 0  # 7.5 %


def test_l2_an_angle_is_compared_on_the_circle_and_within_half_a_bin(tmp_path):
    dq, dp = [0.01, 0.02], [10.0, 100.0]
    assert _cli(tmp_path / "a", dq, dp, phi=355.0)[0] == 0  # 15 deg from 10: wraps
    assert _cli(tmp_path / "b", dq, dp, phi=140.0)[0] == 0  # 40 <= 45 above 100
    status, images = _cli(tmp_path / "c", dq, dp, phi=200.0)
    assert status != 0 and images == []  # 100 deg from the nearest sector, 45 allowed


def test_l2_q_phi_needs_both_coordinates_inside(tmp_path):
    dq, dp = [0.01, 0.02], [10.0, 100.0]
    ok = _cli(tmp_path / "a", dq, dp, q_phi="0.01,10")
    assert ok[0] == 0 and len(ok[1]) == 1
    assert _cli(tmp_path / "b", dq, dp, q_phi="0.5,10")[0] != 0  # q far
    assert _cli(tmp_path / "c", dq, dp, q_phi="0.01,200")[0] != 0  # phi far


def test_l2_a_batch_skips_only_the_far_file_and_summarises_it(tmp_path, caplog):
    root, out = tmp_path / "in", tmp_path / "out"
    root.mkdir()
    _write_twotime(root / "near.h5", [0.01, 0.02])
    _write_twotime(root / "far.h5", [1.0, 2.0])  # same request, other q range
    with caplog.at_level(logging.WARNING):
        status = tb.run_twotime_batch(_args(root, out, q=0.02))
    assert status == 0  # one file matched
    assert [p.name for p in out.rglob("*.png")] == ["near.h5_bin0001_q0.0200.png"]
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "far.h5" in text and "1 file(s) had no matching bin" in text


def test_l2_the_help_text_states_the_rule(monkeypatch, capsys):
    from xpcsviewer import cli_main

    monkeypatch.setattr("sys.argv", ["xpcsviewer", "twotime", "--help"])
    with pytest.raises(SystemExit):
        cli_main.main()
    assert "half a bin spacing" in " ".join(capsys.readouterr().out.split())


# ======================================================================== L3
def _xf_saxs(tmp_path, name: bytes = b"x", value=3.0):
    path = ku._saxs_file(tmp_path / "s.h5", [1.0, 2.0, 3.0], b"px", value=value)
    _rename_axes(path, (name, b"y"), (b"px", b"px"))
    return XpcsFile(str(path))


@pytest.mark.parametrize("norm", ["q2", "q4"])
def test_l3_a_non_radial_first_axis_refuses_the_q_weightings(tmp_path, norm):
    xf = _xf_saxs(tmp_path)
    with pytest.raises(SaxsNormalizationRefused, match="radial 'q'") as err:
        xf.get_saxs1d_data(norm_method=norm)
    assert isinstance(err.value, ValueError) and "'x'" in str(err.value)
    # the two normalisations that do not use the axis are untouched
    _q, iq, xlabel, ylabel = xf.get_saxs1d_data(norm_method=None)
    assert xlabel == "x (px)" and ylabel == "Intensity"
    _q, iq0, _x, ylabel = xf.get_saxs1d_data(norm_method="I0")
    np.testing.assert_allclose(iq0[0], 1.0)
    assert ylabel.endswith("/ I_0")


def test_l3_the_radial_q_weighting_is_unchanged(tmp_path):
    path = ku._saxs_file(tmp_path / "r.h5", [0.1, 0.2, 0.3], b"1/A", value=2.0)
    xf = XpcsFile(str(path))
    for norm, power in (("q2", 2), ("q4", 4)):
        q, iq, _x, ylabel = xf.get_saxs1d_data(norm_method=norm)
        np.testing.assert_allclose(iq[0], 2.0 * np.array([0.1, 0.2, 0.3]) ** power)
        assert ylabel == f"Intensity * q^{power}"


def test_l3_the_refusal_reaches_the_status_bar(tmp_path, qapp):
    _rename_axes(
        ku._saxs_file(tmp_path / "x.h5", [1.0, 2.0, 3.0], b"px", value=9.0),
        (b"x", b"y"),
        (b"px", b"px"),
    )
    bar = QtWidgets.QStatusBar()
    kernel = ViewerKernel(str(tmp_path), statusbar=bar)
    kernel.add_target(["x.h5"])
    widget = pg.PlotWidget()
    kernel.plot_saxs_1d(
        ImageViewDev(), widget, plot_type=0, plot_norm=1, qmin=0.0, qmax=100.0
    )
    message = bar.currentMessage()
    assert "x.h5" in message and "radial 'q'" in message and "I*q^2" in message
    assert widget.getPlotItem().listDataItems() == []  # no wrong curve is drawn
    # without the weighting the curve is drawn and nothing is announced
    bar.clearMessage()
    kernel.plot_saxs_1d(
        ImageViewDev(), widget, plot_type=0, plot_norm=0, qmin=0.0, qmax=100.0
    )
    assert "radial" not in bar.currentMessage()
    assert widget.getPlotItem().listDataItems()


# ======================================================================== L4
def test_l4_a_saved_bin_that_is_gone_is_announced(qtbot, monkeypatch, tmp_path):
    env = st._twotime_session(qtbot, monkeypatch, tmp_path)
    assert env.view.session_manager.save_session(env.view._collect_session_state())
    env.view.close()
    st._reprocess(env.data / "scan.h5", [1], value=7)  # bin 2 is no longer processed

    second = st._restored(qtbot, env)
    assert second.comboBox_twotime_selection.currentIndex() == 0  # choice unchanged
    assert (
        second.statusbar.currentMessage()
        == "saved two-time bin 2 is no longer in scan.h5; showing bin 1"
    )


def test_l4_a_saved_bin_that_is_still_there_says_nothing(qtbot, monkeypatch, tmp_path):
    env = st._twotime_session(qtbot, monkeypatch, tmp_path)
    assert env.view.session_manager.save_session(env.view._collect_session_state())
    env.view.close()
    second = st._restored(qtbot, env)
    assert "no longer" not in second.statusbar.currentMessage()


def test_l4_a_remembered_bin_that_vanishes_on_reload_is_announced(
    qtbot, monkeypatch, tmp_path
):
    env = _make_env(
        qtbot,
        monkeypatch,
        tmp_path,
        {"target.h5": lambda p: _write_twotime(p, [0.01, 0.02], value=5)},
    )
    view = env.view
    _select(view, 0)
    view.tabWidget.setCurrentIndex(_tab(env.xv, "twotime"))
    view.twotime_correct_diag.setChecked(False)
    view.plot_twotime()
    view.comboBox_twotime_selection.setCurrentIndex(1)  # processed bin 2
    view.plot_twotime()

    replacement = env.root / "replacement.h5"
    _write_twotime(replacement, [0.01], value=33)
    import os

    os.replace(replacement, env.data / "target.h5")
    view.reload_source()

    assert view.comboBox_twotime_selection.currentText().startswith("qbin=1")
    assert (
        view.statusbar.currentMessage()
        == "previously selected two-time bin 2 is no longer in target.h5; "
        "showing bin 1"
    )
