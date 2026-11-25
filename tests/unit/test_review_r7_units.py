"""Regression tests for review round 7: Q units and axis identity across files.

One class: every consumer of a Q coordinate that more than one file reaches must
(1) decide compatibility on the axis NAME and UNIT (``axis_mismatch``), (2) get
its range through the ONE conversion (``convert_q_range`` / ``q_range_in_file_unit``
/ ``target_q_in_file_unit``), (3) match bins on numbers in a shared unit, never on
a label, (4) state the file's own axis name and unit wherever a value is shown,
and (5) tell the user which file was left out.

R7-03  G2 panels match equivalent bins of an A^-1 and a nm^-1 file; an inclusive
       bound survives the unit conversion
R7-04  Plot Diffusion refits each file with the Q range in its own unit
R7-05  the two-time CLI selects one physical Q in every file (declared unit)
R7-08  SAXS-1D: x and y in the same unit are different axes
R7-09  the All-Q colour bar, the Bayesian status / title / exports name the unit

Files are written with h5py and read back through the real ``XpcsFile``,
``ViewerKernel`` and ``XpcsViewer``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pyqtgraph as pg
import pytest
from qtpy import QtCore, QtWidgets

from tests.unit import test_review_r3_fit_status as fit
from tests.unit import test_review_r3_maps_roi as r3
from tests.unit import test_review_r4_cli_average as r4cli
from tests.unit import test_review_r5_kernel as r5
from tests.unit import test_review_r6_kernel_units as ku
from tests.unit.test_review_r6_viewer_gui import _make_env
from xpcsviewer.cli import twotime_batch as tb
from xpcsviewer.fileIO.hdf_reader import clear_connection_pool
from xpcsviewer.fileIO.qmap_utils import (
    Q_UNIT_DISPLAY,
    axis_mismatch,
    convert_q_range,
    q_axis_label,
)
from xpcsviewer.module import g2mod
from xpcsviewer.plothandler.pyqtgraph_handler import ImageViewDev
from xpcsviewer.viewer_kernel import ViewerKernel
from xpcsviewer.xpcs_file import XpcsFile

pytestmark = pytest.mark.unit

K = r3.K
NM = ku.NM
PACKAGE = Path(__file__).resolve().parents[2] / "xpcsviewer"


@pytest.fixture(autouse=True)
def _fresh_pool():
    clear_connection_pool()
    yield
    clear_connection_pool()


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _Hdl(pg.GraphicsLayoutWidget):
    def adjust_canvas_size(self, **_kw):
        pass


def _curves_per_panel(hdl):
    return [len(i.listDataItems()) for i in hdl.ci.items if isinstance(i, pg.PlotItem)]


def _plot_both_renderers(xfs, q_range=None):
    """Curve counts per panel from ``pg_plot`` and from the async ``pg_plot_from_data``."""
    direct = _Hdl()
    g2mod.pg_plot(
        direct, xfs, q_range, (1e-3, 10.0), (0.9, 1.5), q_auto=q_range is None
    )
    q, tel, g2, g2_err, labels = g2mod.get_data(xfs, q_range=q_range)
    num_figs, _ = g2mod.compute_geometry(g2, "multiple")
    from_data = _Hdl()
    g2mod.pg_plot_from_data(
        from_data,
        q=q,
        tel=tel,
        g2=g2,
        g2_err=g2_err,
        labels=labels,
        num_figs=num_figs,
        t_range=(1e-3, 10.0),
        y_range=(0.9, 1.5),
    )
    return _curves_per_panel(direct), _curves_per_panel(from_data), labels


def _sector_file(path, q, phi, unit: bytes):
    """A (q, phi) G2 file: ``len(q) * len(phi)`` bins, axis 0 major, in ``unit``."""
    n = len(q) * len(phi)
    g2 = np.array([[1.9] * n, [1.2] * n, [1.0] * n]) + np.arange(n) * 0.01
    r5._write_g2_file(path, list(q), list(phi), list(range(n)), [len(q), len(phi)], g2)
    with h5py.File(path, "r+") as f:
        del f[K["map_units"]]
        f[K["map_units"]] = [unit, b"deg"]
    return path


# ===================================================================== helpers
def test_r7_axis_mismatch_decides_on_name_and_unit():
    assert axis_mismatch(("q", "1/nm"), ("q", Q_UNIT_DISPLAY)) is None
    assert axis_mismatch(("q", "px"), ("q", "px")) is None
    # the same unit does NOT make two coordinates one axis
    assert "'y'" in axis_mismatch(("y", "px"), ("x", "px"))
    assert axis_mismatch(("q", "px"), ("q", Q_UNIT_DISPLAY)) is not None
    assert axis_mismatch(("x", "deg"), ("x", NM)) is not None


@pytest.mark.parametrize("step", range(1, 100))
def test_r7_03_a_converted_inclusive_bound_keeps_the_bin_stored_at_it(step):
    v = step / 100  # e.g. 0.28: 0.28 * 10 == 2.8000000000000003 in float64
    stored_nm = float(f"{v * 10:.12g}")  # what an independent file producer writes
    lo, hi = convert_q_range((v, v + 1.0), Q_UNIT_DISPLAY, NM)
    assert lo <= stored_nm <= hi
    back_lo, _ = convert_q_range((stored_nm, 100.0), NM, Q_UNIT_DISPLAY)
    assert back_lo <= v
    # widening is a roundoff allowance, not a different bin: a neighbour 0.1 %
    # away is still outside
    assert lo > stored_nm * (1 - 1e-9)


def test_r7_03_a_range_in_the_same_unit_is_returned_untouched():
    rng = (0.28, 0.31)
    assert convert_q_range(rng, "1/A", Q_UNIT_DISPLAY) is rng
    assert convert_q_range(None, Q_UNIT_DISPLAY, NM) is None


def test_r7_q_axis_label_names_the_files_own_unit():
    assert q_axis_label("q", NM) == f"q ({NM})"
    assert q_axis_label("Q", Q_UNIT_DISPLAY, mathtext=True) == r"Q ($\AA^{-1}$)"
    assert "nm" in q_axis_label("q", "1/nm", mathtext=True)


# ===================================================================== R7-03
def _edge_pair(tmp_path, order):
    ku._g2_file(tmp_path / "a.h5", [0.1, 0.28, 0.3])
    ku._g2_file(tmp_path / "nm.h5", [1.0, 2.8, 3.0], unit=b"1/nm")
    return _mixed_list(tmp_path, order)


def _mixed_list(tmp_path, order):
    return ku._kernel(tmp_path, order).get_xf_list()


def test_r7_03_inclusive_lower_limit_keeps_the_stored_bin_in_both_directions(tmp_path):
    xfs = _edge_pair(tmp_path, ["a.h5", "nm.h5"])
    q, *_ = g2mod.get_data(xfs, q_range=(0.28, 0.31))
    assert [len(x) for x in q] == [2, 2]  # nm: 2.8000000000000003 used to drop 2.8
    np.testing.assert_allclose(q[1], [0.28, 0.3])
    xfs = _mixed_list(tmp_path, ["nm.h5", "a.h5"])
    q, *_ = g2mod.get_data(xfs, q_range=(2.8, 3.1))
    assert [len(x) for x in q] == [2, 2]


def test_r7_03_equivalent_bins_of_an_A_and_a_nm_file_share_panels(tmp_path, qapp):
    ku._g2_file(tmp_path / "a.h5", [0.1, 0.2, 0.3])
    ku._g2_file(tmp_path / "nm.h5", [1.0, 2.0, 3.0], unit=b"1/nm")
    xfs = ku._kernel(tmp_path, ["a.h5", "nm.h5"]).get_xf_list()
    direct, from_data, labels = _plot_both_renderers(xfs)
    # the labels really differ (native Q and unit); the bins are the same
    assert labels[0] != labels[1]
    assert direct == from_data == [2, 2, 2]


def test_r7_03_a_second_axis_is_matched_by_value_in_both_renderers(tmp_path, qapp):
    _sector_file(tmp_path / "a.h5", [0.1], [0.0, 90.0], b"1/A")
    _sector_file(tmp_path / "nm.h5", [1.0], [0.0, 90.0], b"1/nm")
    xfs = ku._kernel(tmp_path, ["a.h5", "nm.h5"]).get_xf_list()
    direct, from_data, _ = _plot_both_renderers(xfs)
    assert direct == from_data == [2, 2]


def test_r7_03_a_bin_on_another_azimuth_does_not_join_a_panel(tmp_path, qapp):
    _sector_file(tmp_path / "a.h5", [0.1], [0.0, 90.0], b"1/A")
    _sector_file(tmp_path / "nm.h5", [1.0], [0.0, 45.0], b"1/nm")  # 45 is not 90
    xfs = ku._kernel(tmp_path, ["a.h5", "nm.h5"]).get_xf_list()
    direct, from_data, _ = _plot_both_renderers(xfs)
    assert direct == from_data == [2, 1]  # the 45 degree curve is drawn nowhere


# ============================================================ visibility (G2)
def _px_pair(tmp_path):
    ku._g2_file(tmp_path / "a.h5", [0.1, 0.2, 0.3])
    ku._g2_file(tmp_path / "px.h5", [1.0, 2.0, 3.0], unit=b"px")
    return ["a.h5", "px.h5"]


def test_r7_06_g2mod_exposes_the_left_out_file_without_a_gui_call(tmp_path):
    xfs = ku._kernel(tmp_path, _px_pair(tmp_path)).get_xf_list()
    left = g2mod.left_out_files(xfs)
    assert list(left) == ["px.h5"] and "px" in left["px.h5"]
    assert g2mod.left_out_files([]) == {}


def test_r7_06_the_kernel_tells_the_user_which_g2_file_was_left_out(tmp_path, qapp):
    names = _px_pair(tmp_path)
    bar = QtWidgets.QStatusBar()
    kernel = ViewerKernel(str(tmp_path), statusbar=bar)
    kernel.add_target(names)
    kernel.plot_g2(_Hdl(), None, (1e-3, 10.0), (0.9, 1.5), rows=[0, 1], q_auto=True)
    assert "px.h5" in bar.currentMessage() and "left out" in bar.currentMessage()


def test_r7_06_the_async_g2_worker_carries_the_left_out_files(tmp_path, qapp):
    from xpcsviewer.threading.plot_workers import G2PlotWorker

    kernel = ku._kernel(tmp_path, _px_pair(tmp_path))
    result = G2PlotWorker(kernel, _Hdl(), {"rows": [0, 1], "q_auto": True}).do_work()
    assert list(result["left_out"]) == ["px.h5"]


# ===================================================================== R7-08
def _xy_saxs(tmp_path):
    for name, first, second, value in (
        ("x.h5", b"x", b"y", 3.0),
        ("y.h5", b"y", b"x", 11.0),
    ):
        ku._saxs_file(tmp_path / name, [1.0, 2.0, 3.0], b"px", value=value)
        with h5py.File(tmp_path / name, "r+") as f:
            del f[K["map_names"]], f[K["map_units"]]
            f[K["map_names"]] = [first, second]
            f[K["map_units"]] = [b"px", b"px"]
    return ["x.h5", "y.h5"]


def test_r7_08_saxs_does_not_put_a_y_profile_on_an_x_axis(tmp_path, qapp, caplog):
    bar = QtWidgets.QStatusBar()
    kernel = ViewerKernel(str(tmp_path), statusbar=bar)
    kernel.add_target(_xy_saxs(tmp_path))
    widget = pg.PlotWidget()
    kernel.plot_saxs_1d(ImageViewDev(), widget, plot_type=0, qmin=0.0, qmax=100.0)
    item = widget.getPlotItem()
    heights = {round(float(np.mean(d.getData()[1]))) for d in item.listDataItems()}
    assert heights == {3}  # the y file (11) is not on the x axis
    assert item.getAxis("bottom").labelText.startswith("x")
    assert "y.h5" in bar.currentMessage()  # the user sees which file is missing
    assert any("y.h5" in r.getMessage() for r in caplog.records)


def test_r7_08_the_first_file_decides_whichever_comes_first(tmp_path, qapp):
    kernel = ViewerKernel(str(tmp_path))
    kernel.add_target(list(reversed(_xy_saxs(tmp_path))))
    widget = pg.PlotWidget()
    kernel.plot_saxs_1d(ImageViewDev(), widget, plot_type=0, qmin=0.0, qmax=100.0)
    item = widget.getPlotItem()
    assert {round(float(np.mean(d.getData()[1]))) for d in item.listDataItems()} == {11}
    assert item.getAxis("bottom").labelText.startswith("y")


def test_r7_03_saxs_inclusive_range_keeps_the_stored_bin(tmp_path, qapp):
    ku._saxs_file(tmp_path / "a.h5", [0.1, 0.28, 0.3], b"1/A")
    ku._saxs_file(tmp_path / "b.h5", [1.0, 2.8, 3.0], b"1/nm")
    _label, curves = ku._draw(
        ku._kernel(tmp_path, ["a.h5", "b.h5"]), qmin=0.28, qmax=0.31
    )
    assert curves == [(0.28, 0.3)]  # one physical grid, both bins, both files


# ===================================================================== R7-04
def _diffusion_env(qtbot, monkeypatch, tmp_path, extra=()):
    writers = {
        "a.h5": lambda p: ku._g2_file(p, [0.1, 0.2, 0.3]),
        "nm.h5": lambda p: ku._g2_file(p, [1.0, 2.0, 3.0], unit=b"1/nm"),
    }
    for name, q in extra:
        writers[name] = lambda p, q=q: ku._g2_file(p, q)
    env = _make_env(qtbot, monkeypatch, tmp_path, writers)
    view = env.view
    env.xfs = view.vk.get_xf_list(list(range(len(writers))))
    for xf in env.xfs:
        assert all(xf.fit_g2(bounds=fit.BOUNDS, fit_flag=[True] * 4)["fit_success"])
    sel = view.list_view_target.selectionModel()
    sel.clearSelection()
    for row in range(len(writers)):
        sel.select(
            view.vk.target.index(row, 0),
            QtCore.QItemSelectionModel.SelectionFlag.Select,
        )
    view.g2_qauto.setChecked(False)
    view.g2_tauto.setChecked(False)
    view.g2_qmin.setValue(0.09)
    view.g2_qmax.setValue(0.31)
    view.g2_tmin.setText("0.002")
    view.g2_tmax.setText("2")
    view.tauq_qmin.setText("0.09")
    view.tauq_qmax.setText("0.31")
    return env


def test_r7_04_plot_diffusion_refits_each_file_in_its_own_unit(
    qtbot, monkeypatch, tmp_path
):
    env = _diffusion_env(qtbot, monkeypatch, tmp_path)
    a, nm = env.xfs
    env.view.plot_diffusion()  # raised "no Q bins in range (0.09, 0.31)" for nm.h5
    for xf in env.xfs:
        assert xf.fit_summary["tauq_success"], xf.label
    # the summary records the range each file was fitted with (as text)
    np.testing.assert_allclose(ast.literal_eval(a.fit_summary["q_range"]), (0.09, 0.31))
    np.testing.assert_allclose(
        ast.literal_eval(nm.fit_summary["q_range"]), (0.9, 3.1), rtol=1e-9
    )
    np.testing.assert_allclose(a.fit_summary["tauq_q"], nm.fit_summary["tauq_q"])
    np.testing.assert_allclose(
        a.fit_summary["tauq_fit_val"], nm.fit_summary["tauq_fit_val"], rtol=1e-5
    )


def test_r7_04_a_file_without_bins_is_set_aside_not_fatal(qtbot, monkeypatch, tmp_path):
    env = _diffusion_env(
        qtbot, monkeypatch, tmp_path, extra=[("far.h5", [5.0, 6.0, 7.0])]
    )
    a, nm, far = env.xfs
    before = far.fit_summary
    values = dict(before)
    env.view.plot_diffusion()
    # the other two files were refitted and plotted
    assert a.fit_summary["tauq_success"] and nm.fit_summary["tauq_success"]
    # the file with no bin in the range keeps its earlier fit and is named
    assert far.fit_summary is before
    assert all(np.array_equal(before[k], v) for k, v in values.items() if k in before)
    message = env.view.statusbar.currentMessage()
    assert "far.h5" in message and "no Q bins" in message


# ===================================================================== R7-05
def _axes(path, names, units):
    with h5py.File(path, "a") as f:
        del f[K["map_names"]], f[K["map_units"]]
        f[K["map_names"]] = names
        f[K["map_units"]] = units


def _images(out):
    return sorted(str(p.relative_to(out)) for p in out.rglob("*.png"))


def _mixed_cli(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    r4cli._write_twotime(source / "angstrom.h5", [0.01, 0.02], value=2)
    r4cli._write_twotime(source / "nanometer.h5", [0.1, 0.2], value=11)
    _axes(source / "nanometer.h5", [b"q", b"phi"], [b"1/nm", b"deg"])
    return source


def test_r7_05_one_q_request_selects_one_physical_q_in_every_file(tmp_path):
    source, out = _mixed_cli(tmp_path), tmp_path / "out"
    assert tb.run_twotime_batch(r4cli._args(source, out, q=0.02)) == 0
    # 0.02 A^-1 is 0.2 nm^-1: bin 2 (index 1) of BOTH files
    assert _images(out) == [
        "angstrom.h5_bin0001_q0.0200.png",
        "nanometer.h5_bin0001_q0.2000.png",
    ]


def test_r7_05_the_declared_unit_converts_the_other_way(tmp_path):
    source, out = _mixed_cli(tmp_path), tmp_path / "out"
    args = r4cli._args(source, out, q=0.2, q_unit="nm^-1")
    assert tb.run_twotime_batch(args) == 0
    assert _images(out) == [
        "angstrom.h5_bin0001_q0.0200.png",
        "nanometer.h5_bin0001_q0.2000.png",
    ]


def test_r7_05_q_phi_converts_its_q_per_file(tmp_path):
    source, out = tmp_path / "source", tmp_path / "out"
    source.mkdir()
    r4cli._write_twotime(source / "angstrom.h5", [0.02], dplist=[10.0, 20.0])
    r4cli._write_twotime(source / "nanometer.h5", [0.2], dplist=[10.0, 20.0])
    _axes(source / "nanometer.h5", [b"q", b"phi"], [b"1/nm", b"deg"])
    assert tb.run_twotime_batch(r4cli._args(source, out, q_phi="0.02,20")) == 0
    assert _images(out) == [
        "angstrom.h5_bin0001_q0.0200_phi20.0.png",
        "nanometer.h5_bin0001_q0.2000_phi20.0.png",
    ]


def _cartesian_cli(path, *, units=b"1/A"):
    r4cli._write_twotime(path, [-0.03, 0.01], dplist=[-0.02, 0.02])
    _axes(path, [b"qx", b"qy"], [units, units])
    return path


def test_r7_05_a_signed_cartesian_request_is_checked_against_the_files_axes(tmp_path):
    cart, out = _cartesian_cli(tmp_path / "cart.h5"), tmp_path / "signed"
    assert tb.run_twotime_batch(r4cli._args(cart, out, q=-0.03)) == 0
    # both coordinates AND their units are in the name
    assert _images(out) == [
        "cart.h5_bin0000_qx-0.0300invA_qy-0.0200invA.png",
        "cart.h5_bin0001_qx-0.0300invA_qy0.0200invA.png",
    ]


def test_r7_05_a_negative_radial_q_is_refused_per_file(tmp_path):
    source, out = _mixed_cli(tmp_path), tmp_path / "out"
    assert tb.run_twotime_batch(r4cli._args(source, out, q=-0.02)) == 1
    assert not list(out.rglob("*.png"))


def test_r7_05_a_pixel_axis_needs_the_declared_pixel_unit(tmp_path):
    cart = _cartesian_cli(tmp_path / "px.h5", units=b"px")
    out = tmp_path / "out"
    assert tb.run_twotime_batch(r4cli._args(cart, out, q=-0.03)) == 1  # not in A^-1
    assert not list(out.rglob("*.png"))
    assert tb.run_twotime_batch(r4cli._args(cart, out, q=-0.03, q_unit="px")) == 0
    assert _images(out) == [
        "px.h5_bin0000_qx-0.0300px_qy-0.0200px.png",
        "px.h5_bin0001_qx-0.0300px_qy0.0200px.png",
    ]


def test_r7_05_the_command_line_declares_the_unit(monkeypatch):
    from xpcsviewer import cli_main

    seen = {}
    monkeypatch.setattr(
        cli_main, "_run_twotime_batch", lambda a: seen.update(args=a) or 0
    )
    for name in ("initialize_logging", "setup_exception_logging", "safe_shutdown"):
        monkeypatch.setattr(cli_main, name, lambda *a, **k: None, raising=False)
    monkeypatch.setattr(cli_main.signal, "signal", lambda *a, **k: None)
    monkeypatch.setattr(cli_main.atexit, "register", lambda *a, **k: None)
    for extra, expected in (([], None), (["--q-unit", "nm^-1"], "nm^-1")):
        monkeypatch.setattr(
            sys,
            "argv",
            ["xpcsviewer-cli", "twotime", "-i", "x", "-o", "y", "--q", "1", *extra],
        )
        with pytest.raises(SystemExit):
            cli_main.main()
        assert seen["args"].q_unit == expected


# ===================================================================== R7-09
def _all_q_summary(bin_axes):
    t = np.geomspace(0.005, 1, 12)
    return {
        "fit_line": np.ones((2, 12)),
        "fit_x": t,
        "q_val": np.array([1.0, 2.0]),
        "t_el": t,
        "obs_g2": np.ones((12, 2)),
        "obs_t_el": t,
        "failed_mask": np.array([False, False]),
        "bin_axes": bin_axes,
    }


@pytest.mark.parametrize(("unit", "expected"), [(NM, "nm"), (Q_UNIT_DISPLAY, r"\AA")])
def test_r7_09_the_all_q_colour_bar_names_the_files_unit(unit, expected):
    import matplotlib.pyplot as plt

    from xpcsviewer.fitting.viz import plot_bayesian_all_q

    axes = {"names": ("q", "phi"), "units": (unit, "deg"), "axis1": np.zeros(2)}
    fig = plot_bayesian_all_q(_all_q_summary(axes))
    labels = [a.get_ylabel() for a in fig.axes]
    assert any(expected in text and "q" in text for text in labels), labels
    if expected == "nm":
        assert not any(r"\AA" in text for text in labels)
    plt.close(fig)


def test_r7_09_a_summary_without_axes_keeps_the_legacy_label():
    import matplotlib.pyplot as plt

    from xpcsviewer.fitting.viz import plot_bayesian_all_q

    fig = plot_bayesian_all_q(_all_q_summary(None))
    assert any(r"\AA" in a.get_ylabel() for a in fig.axes)
    plt.close(fig)


def _nm_bayes_env(qtbot, monkeypatch, tmp_path):
    env = _make_env(
        qtbot,
        monkeypatch,
        tmp_path,
        {"a.h5": lambda p: ku._g2_file(p, [1.0, 2.0, 3.0], unit=b"1/nm")},
    )
    view = env.view
    view.list_view_target.selectionModel().select(
        view.vk.target.index(0, 0),
        QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect,
    )
    view.g2_qmin.setValue(0.0)
    view.g2_qmax.setValue(5.0)
    view.g2_tmin.setText("0.001")
    view.g2_tmax.setText("2")
    return env


def test_r7_09_the_bin_status_names_one_unit(qtbot, monkeypatch, tmp_path):
    view = _nm_bayes_env(qtbot, monkeypatch, tmp_path).view
    view._on_g2_qbin_changed(0)
    message = view.statusbar.currentMessage()
    assert "nm⁻¹" in message and Q_UNIT_DISPLAY not in message, message


def test_r7_09_the_bayesian_bundle_and_title_carry_the_axis(
    qtbot, monkeypatch, tmp_path
):
    view = _nm_bayes_env(qtbot, monkeypatch, tmp_path).view
    bundle, _ = view._extract_g2_for_bayesian()
    assert bundle.q_axis == ("q", NM)
    assert bundle.q_text == f"q=1 {NM}"
    done = bundle.with_result(None)
    assert done.q_axis == ("q", NM)  # survives the commit of the result


def test_r7_09_diagnosis_exports_state_the_axis_of_q_value(
    qtbot, monkeypatch, tmp_path
):
    from tests.unit.test_review_r5_bayes_gui import _real_result
    from tests.unit.test_review_r6_viewer_gui import _FakeIdata
    from xpcsviewer.fitting.models import single_exp_func
    from xpcsviewer.gui.bayesian_diagnosis import (
        BayesianDiagnosisWindow,
        DiagnosisIdentity,
    )

    sink: list = []
    window = BayesianDiagnosisWindow()
    qtbot.addWidget(window)
    result = _real_result()
    result.arviz_data = _FakeIdata(sink)
    window.update_results(
        result=result,
        model_func=single_exp_func,
        x_data=[0.1, 0.2, 0.3],
        y_data=[1.3, 1.2, 1.1],
        yerr=[0.01] * 3,
        q_value=1.0,
        q_axis=("q", NM),
        identity=DiagnosisIdentity(str(tmp_path / "scan.h5"), None, 7, None),
    )
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *a, **k: (str(tmp_path / "t.nc"), ""),
    )
    window._export_traces()
    attrs = sink[0][1]
    assert attrs["q_value"] == 1.0
    assert (attrs["axis0_name"], attrs["axis0_unit"]) == ("q", NM)
    assert NM in window._q_text(1.0)
    window._export_fit_data = window._export_fit_data  # (writes via a dialog below)
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *a, **k: (str(tmp_path / "f.csv"), ""),
    )
    window._export_fit_data()
    assert f"# q = 1.0 {NM}" in (tmp_path / "f.csv").read_text().splitlines()[0]


# ===================================================================== guards
def _py_files():
    skip = ("_ui.py", "icons_rc.py")
    return [p for p in sorted(PACKAGE.rglob("*.py")) if not p.name.endswith(skip)]


def _functions(tree):
    """``(function_name, node)`` of every function, innermost name for its calls."""
    out = []

    class V(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            out.append(node)
            self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

    V().visit(tree)
    return out


def _call_name(node):
    f = node.func
    return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)


#: the single converters a Q range may come through
_CONVERTERS = {"q_range_in_file_unit", "convert_q_range", "target_q_in_file_unit"}

#: function -> position of its Q-range argument (and its keyword spellings)
_RANGE_ARG = {
    "fit_g2": (0, {"q_range", "qrange"}),
    "fit_g2_robust": (0, {"q_range", "qrange"}),
    "fit_g2_high_performance": (0, {"q_range", "qrange"}),
    "get_g2_data": (0, {"q_range", "qrange"}),
    "get_g2_stability_data": (0, {"q_range", "qrange"}),
    "get_saxs1d_data": (None, {"q_range", "qrange"}),
    "fit_tauq": (0, {"q_range", "qrange"}),
    "get_qbin_in_qrange": (0, {"q_range", "qrange"}),
    "resolve_g2_bins": (1, {"q_range", "qrange"}),
    "find_qbins_for_q": (1, {"target_q"}),
    "find_qbin_for_qphi": (1, {"target_q"}),
}

#: (file, function, callee) -> why the range is already in that file's own unit
_RANGE_ALLOWED = {
    (
        "xpcs_file.py",
        "*",
        "*",
    ): "XpcsFile methods: a q_range argument is the file's own unit by contract",
    (
        "module/g2mod.py",
        "_labelled",
        "get_qbin_in_qrange",
    ): "own_q_range is get_data's converted file_q_range",
    (
        "module/g2mod.py",
        "pg_plot_stability",
        "get_g2_stability_data",
    ): "one file: the one the Q boxes were built from",
    (
        "viewer_kernel.py",
        "plot_tauq",
        "fit_tauq",
    ): "tau-q is inverse angstrom: summaries are converted by _in_diffusion_unit",
    (
        "fitting/bayesian_assembly.py",
        "resolve_g2_bins",
        "get_qbin_in_qrange",
    ): "the caller passes the range in xf's own unit",
    (
        "xpcs_viewer.py",
        "_g2_bin_table",
        "resolve_g2_bins",
    ): "first selected file: the G2 boxes are in its unit",
    (
        "xpcs_viewer.py",
        "_g2_resolved_bins",
        "resolve_g2_bins",
    ): "first selected file: the G2 boxes are in its unit",
}


def _range_expr(call, name):
    pos, keywords = _RANGE_ARG[name]
    for kw in call.keywords:
        if kw.arg in keywords:
            return kw.value
    if pos is not None and len(call.args) > pos:
        return call.args[pos]
    return None


def _converted_names(fn):
    """Names assigned in ``fn`` from a call to one of the converters."""
    names = set()
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and _call_name(node.value) in _CONVERTERS
        ):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _unconverted_range_calls():
    found = []
    for path in _py_files():
        rel = path.relative_to(PACKAGE).as_posix()
        tree = ast.parse(path.read_text())
        for fn in _functions(tree):
            converted = _converted_names(fn)
            for call in (n for n in ast.walk(fn) if isinstance(n, ast.Call)):
                name = _call_name(call)
                if name not in _RANGE_ARG or name == fn.name:
                    continue
                arg = _range_expr(call, name)
                if arg is None or (isinstance(arg, ast.Constant) and arg.value is None):
                    continue  # no range: every bin
                if isinstance(arg, ast.Call) and _call_name(arg) in _CONVERTERS:
                    continue
                if isinstance(arg, ast.Name) and arg.id in converted:
                    continue
                if (rel, fn.name, name) in _RANGE_ALLOWED or (
                    rel,
                    "*",
                    "*",
                ) in _RANGE_ALLOWED:
                    continue
                found.append((rel, fn.name, name, ast.unparse(arg), call.lineno))
    return found


def test_r7_every_q_range_reaches_a_file_through_the_one_conversion():
    """A new caller of a range-taking function must convert (or be listed above)."""
    assert _unconverted_range_calls() == []


def test_r7_the_range_guard_would_catch_a_new_consumer(tmp_path, monkeypatch):
    bad = tmp_path / "newmod.py"
    bad.write_text(
        "def plot(xf, q_range):\n    return xf.get_g2_data(qrange=q_range)\n"
    )
    good = tmp_path / "goodmod.py"
    good.write_text(
        "def plot(xf, q_range, shared):\n"
        "    own = q_range_in_file_unit(xf, q_range, shared)\n"
        "    return xf.get_g2_data(qrange=own)\n"
    )
    monkeypatch.setattr(sys.modules[__name__], "PACKAGE", tmp_path)
    monkeypatch.setattr(sys.modules[__name__], "_py_files", lambda: [bad, good])
    found = _unconverted_range_calls()
    assert [(f[0], f[1], f[2]) for f in found] == [("newmod.py", "plot", "get_g2_data")]


#: tokens that spell an inverse-angstrom unit (or its mathtext) in a string
_LABEL_TOKENS = ("Å", "A^-1", "1/A", "\\AA", "Angstrom", "angstrom")

#: (file, function) -> why a literal / ``{Q_UNIT_DISPLAY}`` is not the label of a
#: file's own Q value; ``*`` allows the file
_LABEL_ALLOWED = {
    ("fileIO/qmap_utils.py", "*"): "the display helper and the unit tables",
    ("module/tauq.py", "plot"): "tau-q rows are converted to inverse angstrom first",
    (
        "module/tauq.py",
        "plot_pre",
    ): "tau-q rows are converted to inverse angstrom first",
    (
        "helper/utils.py",
        "norm_saxs_data",
    ): "legacy helper without a caller in the package",
    ("cli_main.py", "main"): "help text that documents the --q-unit default",
    ("schemas/validators.py", "*"): "schema of the units stored in a file",
    ("io/hdf5_facade.py", "read_qmap"): "typing Literal of the units stored in a file",
    ("xpcs_file.py", "_q_header_label"): "export header keyed by the file's own unit",
    (
        "xpcs_file.py",
        "_roi_profile",
    ): "refusal: the SAXS background must be in inverse angstrom",
    (
        "viewer_kernel.py",
        "_diffusion_axis_problem",
    ): "diffusion works in, and says, inverse angstrom",
    (
        "simplemask/simplemask_kernel.py",
        "show_location",
    ): "SimpleMask computes its Q map in inverse angstrom",
    (
        "simplemask/simplemask_window.py",
        "_on_generate_qmap",
    ): "SimpleMask computes its Q map in inverse angstrom",
    ("plothandler/matplot_qt.py", "plot_line"): "a length in angstrom, not a Q",
}


def _enclosing_functions(tree):
    owner = {}

    class V(ast.NodeVisitor):
        def __init__(self):
            self.stack = ["<module>"]

        def generic_visit(self, node):
            owner[id(node)] = self.stack[-1]
            super().generic_visit(node)

        def visit_FunctionDef(self, node):
            owner[id(node)] = self.stack[-1]
            self.stack.append(node.name)
            super().generic_visit(node)
            self.stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

    V().visit(tree)
    return owner


def _docstring_nodes(tree):
    ids = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
            ):
                ids.add(id(body[0].value))
    return ids


def _hardcoded_unit_labels():
    found = []
    for path in _py_files():
        rel = path.relative_to(PACKAGE).as_posix()
        tree = ast.parse(path.read_text())
        docs, owner = _docstring_nodes(tree), _enclosing_functions(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                hit = id(node) not in docs and any(
                    t in node.value for t in _LABEL_TOKENS
                )
            else:
                hit = (
                    isinstance(node, ast.FormattedValue)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "Q_UNIT_DISPLAY"
                )
            fn = owner.get(id(node))
            if (
                hit
                and (rel, fn) not in _LABEL_ALLOWED
                and (rel, "*") not in _LABEL_ALLOWED
            ):
                found.append((rel, fn, node.lineno))
    return found


def test_r7_no_q_unit_is_hard_coded_into_a_label_outside_the_display_helper():
    assert _hardcoded_unit_labels() == []


def test_r7_the_label_guard_would_catch_a_new_literal(tmp_path, monkeypatch):
    bad = tmp_path / "newlabel.py"
    bad.write_text('def f():\n    return "Q (Å⁻¹)"\n')
    monkeypatch.setattr(sys.modules[__name__], "_py_files", lambda: [bad])
    monkeypatch.setattr(sys.modules[__name__], "PACKAGE", tmp_path)
    assert [f[0] for f in _hardcoded_unit_labels()] == ["newlabel.py"]
