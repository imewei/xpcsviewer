"""Regression tests for review round 8: axis values compared or selected across files.

One class: a value on an axis that two files (or a file and a selector) share is
compared or selected by ONE rule, in a stated unit, on a stated axis.

R8-02  G2 panels: an azimuth is periodic (-15 deg is 345 deg)
R8-08  the two-time CLI: ``--phi`` / ``--q-phi`` are degrees, a file storing
       radians is converted before it is matched, named and labelled
R8-09  diffusion selects in the file's NATIVE unit (inclusive bounds survive),
       NLSQ and Bayesian through one row selection
R8-10  diffusion fits the radial |q|, never a Cartesian component read as |q|

Files are written with h5py and read back through the real ``XpcsFile``,
``ViewerKernel``, ``XpcsViewer`` and CLI.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import h5py
import numpy as np
import pytest

from tests.unit import test_review_r3_fit_status as fit
from tests.unit import test_review_r4_cli_average as r4cli
from tests.unit import test_review_r6_kernel_units as ku
from tests.unit import test_review_r7_units as r7
from tests.unit.test_review_r6_viewer_gui import _make_env
from xpcsviewer.cli import twotime_batch as tb
from xpcsviewer.fileIO.hdf_reader import clear_connection_pool
from xpcsviewer.fileIO.qmap_utils import (
    Q_UNIT_DISPLAY,
    same_bin_coordinate,
    to_degrees,
)
from xpcsviewer.module import g2mod
from xpcsviewer.xpcs_file import XpcsFile

pytestmark = pytest.mark.unit

K = r7.K
qapp = r7.qapp

PHI_DEG = ("phi", "deg")


@pytest.fixture(autouse=True)
def _fresh_pool():
    clear_connection_pool()
    yield
    clear_connection_pool()


def _phi_units(path, unit: bytes, first: bytes = b"1/A"):
    with h5py.File(path, "r+") as f:
        del f[K["map_units"]]
        f[K["map_units"]] = [first, unit]


def _images(out):
    return sorted(str(p.relative_to(out)) for p in out.rglob("*.png"))


# =============================================================== the helper
@pytest.mark.parametrize(
    ("a", "a_axis", "b", "b_axis", "same"),
    [
        (-15.0, PHI_DEG, 345.0, PHI_DEG, True),  # the wrap
        (359.9995, PHI_DEG, -0.0004, PHI_DEG, True),
        (180.0, PHI_DEG, -180.0, PHI_DEG, True),
        (10.0, PHI_DEG, np.deg2rad(10.0), ("phi", "rad"), True),
        (10.0, ("phi", "degree"), np.deg2rad(10.0), ("phi", "radian"), True),
        (-np.pi / 12, ("phi", "rad"), 345.0, PHI_DEG, True),  # both at once
        (10.0, PHI_DEG, 10.5, PHI_DEG, False),
        (10.0, PHI_DEG, 10.0, ("chi", "deg"), False),  # another axis
        (10.0, PHI_DEG, 10.0, ("phi", "px"), False),  # an angle is not a length
        (0.2, ("qy", "1/A"), 2.0, ("qy", "1/nm"), True),  # linear, converted
        (0.2, ("qy", "1/A"), 2.1, ("qy", "1/nm"), False),
        (5.0, ("y", "px"), 5.0, ("y", "px"), True),
        (5.0, ("y", "px"), 365.0, ("y", "px"), False),  # NOT periodic off an angle
    ],
)
def test_r8_02_one_second_axis_comparison(a, a_axis, b, b_axis, same):
    assert bool(same_bin_coordinate(a, a_axis, b, b_axis)) is same
    assert bool(same_bin_coordinate(b, b_axis, a, a_axis)) is same  # symmetric


def test_r8_02_angle_units_are_the_spellings_of_files_and_docs():
    for unit in ("deg", "degree", "degrees", "°", "rad", "radian", "radians", "RAD"):
        to_degrees(1.0, unit)
    assert float(to_degrees(np.pi, "rad")) == pytest.approx(180.0)
    with pytest.raises(ValueError, match="not an angle unit"):
        to_degrees(1.0, "px")


# ====================================================================== R8-02
def _sectors(path, phi, unit: bytes = b"deg", q=0.1):
    r7._sector_file(path, [q], phi, b"1/A")
    _phi_units(path, unit)
    return path


def test_r8_02_signed_and_wrapped_azimuths_share_their_panels(tmp_path, qapp):
    _sectors(tmp_path / "signed.h5", [-15.0, 15.0])
    _sectors(tmp_path / "wrapped.h5", [345.0, 15.0])
    xfs = ku._kernel(tmp_path, ["signed.h5", "wrapped.h5"]).get_xf_list()
    direct, from_data, _ = r7._plot_both_renderers(xfs)
    assert direct == from_data == [2, 2]  # [1, 2] before: -15 != 345 linearly


def test_r8_02_a_radian_file_shares_panels_with_a_degree_file(tmp_path, qapp):
    _sectors(tmp_path / "deg.h5", [10.0, 80.0])
    _sectors(tmp_path / "rad.h5", list(np.deg2rad([10.0, 80.0])), unit=b"rad")
    xfs = ku._kernel(tmp_path, ["deg.h5", "rad.h5"]).get_xf_list()
    direct, from_data, labels = r7._plot_both_renderers(xfs)
    assert labels[0] != labels[1]  # the labels really are in different units
    assert direct == from_data == [2, 2]


def test_r8_02_a_different_azimuth_still_does_not_join(tmp_path, qapp):
    _sectors(tmp_path / "a.h5", [-15.0, 15.0])
    _sectors(tmp_path / "b.h5", [345.0, 20.0])  # 20 is not 15
    xfs = ku._kernel(tmp_path, ["a.h5", "b.h5"]).get_xf_list()
    direct, from_data, _ = r7._plot_both_renderers(xfs)
    assert direct == from_data == [2, 1]


# ====================================================================== R8-08
def _cli_pair(tmp_path, unit=b"rad", angles=(10.0, 80.0), q=(0.02,)):
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    r4cli._write_twotime(source / "degree.h5", list(q), dplist=list(angles))
    r4cli._write_twotime(source / "radian.h5", list(q), dplist=list(np.deg2rad(angles)))
    r7._axes(source / "radian.h5", [b"q", b"phi"], [b"1/A", unit])
    return source


def test_r8_08_a_radian_file_is_matched_in_degrees(tmp_path):
    source = _cli_pair(tmp_path)
    for name in ("degree.h5", "radian.h5"):
        xf = XpcsFile(str(source / name))
        match = tb.find_qbin_for_qphi(xf, 0.02, 10.0)
        assert match[0] == 0, name  # radian.h5 matched bin 1 (1.4 "degrees") before
        assert match[3] == pytest.approx(10.0)  # the value is in degrees
        assert [m[0] for m in tb.find_qbins_for_phi(xf, 80.0)] == [1], name


def test_r8_08_the_cli_names_the_same_sector_the_same_way(tmp_path):
    source, out = _cli_pair(tmp_path), tmp_path / "out"
    assert tb.run_twotime_batch(r4cli._args(source, out, q_phi="0.02,10")) == 0
    assert _images(out) == [
        "degree.h5_bin0000_q0.0200_phi10.0.png",
        "radian.h5_bin0000_q0.0200_phi10.0.png",  # phi1.4 and bin 1 before
    ]
    out = tmp_path / "out_phi"
    assert tb.run_twotime_batch(r4cli._args(source, out, phi=80.0)) == 0
    assert _images(out) == [
        "degree.h5_bin0001_q0.0200_phi80.0.png",
        "radian.h5_bin0001_q0.0200_phi80.0.png",
    ]


def test_r8_08_the_selector_wraps_for_a_radian_file(tmp_path):
    source = _cli_pair(tmp_path, angles=(345.0, 20.0))
    xf = XpcsFile(str(source / "radian.h5"))
    assert [m[0] for m in tb.find_qbins_for_phi(xf, -15.0)] == [0]
    assert tb.find_qbin_for_qphi(xf, 0.02, -15.0)[0] == 0


def test_r8_08_a_phi_axis_that_is_not_an_angle_is_refused_only_by_angle_modes(
    tmp_path, caplog
):
    source = _cli_pair(tmp_path, unit=b"px")
    xf = XpcsFile(str(source / "radian.h5"))
    assert tb.find_qbin_for_qphi(xf, 0.02, 10.0) is None
    assert tb.find_qbins_for_phi(xf, 10.0) == []
    assert any("degrees or radians" in r.getMessage() for r in caplog.records)
    # --q never needed an angle: the file still yields its bins
    assert [m[0] for m in tb.find_qbins_for_q(xf, 0.02)] == [0, 1]


def test_r8_08_the_cli_difference_is_the_one_in_qmap_utils():
    from xpcsviewer.fileIO.qmap_utils import angular_difference

    assert tb._angular_diff(350.0, 10.0) == 20.0 == float(angular_difference(350, 10))
    assert tb._angular_diff(-15.0, 345.0) == 0.0


# ====================================================================== R8-09
def _edge_files(tmp_path):
    ku._g2_file(tmp_path / "a.h5", [0.1, 0.28, 0.32])
    ku._g2_file(tmp_path / "nm.h5", [1.0, 2.8, 3.2], unit=b"1/nm")
    return ["a.h5", "nm.h5"]


def test_r8_09_the_deterministic_fit_keeps_a_bin_stored_at_the_bound(tmp_path, qapp):
    kernel = ku._fitted_kernel(tmp_path, _edge_files(tmp_path))
    ku._tauq(kernel, [0, 1], q_range=(0.28, 0.35))
    for xf in kernel.get_xf_list():
        # 2.8 nm^-1 became 0.27999999999999997 and failed ">= 0.28" before
        assert xf.fit_summary["tauq_success"], xf.label
        np.testing.assert_allclose(xf.fit_summary["tauq_q"], [0.28, 0.32])


def test_r8_09_the_upper_bound_is_inclusive_too(tmp_path, qapp):
    kernel = ku._fitted_kernel(tmp_path, _edge_files(tmp_path))
    ku._tauq(kernel, [0, 1], q_range=(0.09, 0.28))
    for xf in kernel.get_xf_list():
        np.testing.assert_allclose(xf.fit_summary["tauq_q"], [0.1, 0.28])


def _bayes_env(qtbot, monkeypatch, tmp_path, writers):
    env = _make_env(qtbot, monkeypatch, tmp_path, writers)
    env.xfs = env.view.vk.get_xf_list(list(range(len(writers))))
    for xf in env.xfs:
        assert all(xf.fit_g2(bounds=fit.BOUNDS, fit_flag=[True] * 4)["fit_success"])
    for name, text in {
        "tauq_amin": "1e-5",
        "tauq_amax": "100",
        "tauq_bmin": "-5",
        "tauq_bmax": "1",
    }.items():
        getattr(env.view, name).setText(text)
    return env


def _select_only(view, row):
    sel = view.list_view_target.selectionModel()
    sel.clearSelection()
    sel.select(
        view.vk.target.index(row, 0),
        r7.QtCore.QItemSelectionModel.SelectionFlag.Select,
    )


def test_r8_09_the_bayesian_launch_keeps_the_same_rows_as_the_nlsq_fit(
    qtbot, monkeypatch, tmp_path
):
    env = _bayes_env(
        qtbot,
        monkeypatch,
        tmp_path,
        {
            "a.h5": lambda p: ku._g2_file(p, [0.1, 0.28, 0.32]),
            "nm.h5": lambda p: ku._g2_file(p, [1.0, 2.8, 3.2], unit=b"1/nm"),
        },
    )
    view = env.view
    view.tauq_qmin.setText("0.28")
    view.tauq_qmax.setText("0.35")
    for row in (0, 1):
        _select_only(view, row)
        bundle = view._extract_diffusion_for_bayesian()
        assert bundle is not None, view.statusbar.currentMessage()
        np.testing.assert_allclose(bundle.x, [0.28, 0.32])
        # the NLSQ fit (the kernel) selects the same rows
        view.vk.plot_tauq(
            hdl=ku.MplCanvas(),
            bounds=[[1e-5, -5], [100.0, 1]],
            rows=[row],
            q_range=[0.28, 0.35],
            offset=0,
            fit_flag=[True, True],
        )
        xf = env.xfs[row]
        np.testing.assert_array_equal(bundle.x, xf.fit_summary["tauq_q"])
        np.testing.assert_array_equal(bundle.y, xf.fit_summary["tauq_tau"])
        np.testing.assert_array_equal(bundle.yerr, xf.fit_summary["tauq_tau_err"])


def test_r8_09_the_two_selections_are_one_function():
    """``fit_tauq`` and the Bayesian launch select through ``tauq_rows``; the
    range is converted to the file's unit INSIDE it, never compared raw."""
    from xpcsviewer import xpcs_viewer

    for fn in (XpcsFile.fit_tauq, xpcs_viewer._tauq_rows):
        calls = {
            n.func.attr
            if isinstance(n.func, ast.Attribute)
            else getattr(n.func, "id", "")
            for n in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(fn))))
            if isinstance(n, ast.Call)
        }
        assert "tauq_rows" in calls, fn.__qualname__
    from xpcsviewer.module import tauq

    names = {
        n.func.id
        for n in ast.walk(ast.parse(inspect.getsource(tauq.tauq_rows)))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert {"convert_q_range", "convert_q"} <= names
    # no raw "x >= q_range[0]" survives in either entry point
    for fn in (XpcsFile.fit_tauq, xpcs_viewer._tauq_rows):
        assert "q_range[0]" not in inspect.getsource(fn)


# ====================================================================== R8-10
def _cartesian(path, q, qy=0.2, *, names=(b"qx", b"qy"), unit=b"1/A"):
    ku._g2_file(path, list(q))
    with h5py.File(path, "r+") as f:
        del f[K["map_names"]], f[K["map_units"]]
        f[K["map_names"]] = list(names)
        f[K["map_units"]] = [unit, unit]
        f[K["dplist"]][:] = [qy]
    return path


MAGNITUDE = np.hypot([0.1, 0.2, 0.3], 0.2)


def test_r8_10_a_cartesian_pair_is_fitted_on_the_radial_magnitude(tmp_path, qapp):
    _cartesian(tmp_path / "cart.h5", [0.1, 0.2, 0.3])
    ku._g2_file(tmp_path / "radial.h5", list(MAGNITUDE))  # the same physics, as |q|
    kernel = ku._fitted_kernel(tmp_path, ["cart.h5", "radial.h5"])
    canvas, _ = ku._tauq(kernel, [0, 1])
    cart, radial = kernel.get_xf_list()
    assert cart.fit_summary["tauq_success"]
    # qx = [.1 .2 .3] used to be fitted and drawn as |q|
    np.testing.assert_allclose(cart.fit_summary["tauq_q"], MAGNITUDE)
    np.testing.assert_allclose(
        cart.fit_summary["tauq_fit_val"], radial.fit_summary["tauq_fit_val"], rtol=1e-6
    )
    drawn = [np.asarray(line.get_xdata()) for line in canvas.fig.axes[0].lines]
    assert any(x.shape == MAGNITUDE.shape and np.allclose(x, MAGNITUDE) for x in drawn)
    # the file's own G2 fit is not rewritten
    np.testing.assert_allclose(cart.fit_summary["q_val"], [0.1, 0.2, 0.3])


def test_r8_10_the_range_selects_on_the_magnitude(tmp_path, qapp):
    _cartesian(tmp_path / "cart.h5", [0.1, 0.2, 0.3])
    kernel = ku._fitted_kernel(tmp_path, ["cart.h5"])
    # |q| = 0.224, 0.283, 0.361: [0.1, 0.3] holds qx of all three but |q| of one
    ku._tauq(kernel, [0], q_range=(0.1, 0.3))
    (xf,) = kernel.get_xf_list()
    np.testing.assert_allclose(xf.fit_summary["tauq_q"], MAGNITUDE[:2])


def test_r8_10_the_export_says_how_q_was_computed(tmp_path, qapp):
    _cartesian(tmp_path / "cart.h5", [0.1, 0.2, 0.3])
    kernel = ku._fitted_kernel(tmp_path, ["cart.h5"])
    ku._tauq(kernel, [0])
    out = tmp_path / "out"
    out.mkdir()
    kernel.export_diffusion(str(out), rows=[0])
    text = (out / "cart_diffusion_fitting.txt").read_text()
    assert "# Q source: |q| = sqrt(qx^2 + qy^2)" in text
    assert "qy = 0.2 Å⁻¹" in text
    assert "# Q source of cart.h5" in (out / "diffusion_summary.txt").read_text()


def test_r8_10_the_pre_plot_draws_the_magnitude_too(tmp_path, qapp):
    _cartesian(tmp_path / "cart.h5", [0.1, 0.2, 0.3])
    kernel = ku._fitted_kernel(tmp_path, ["cart.h5"])
    canvas = ku.MplCanvas()
    kernel.plot_tauq_pre(hdl=canvas, rows=[0])
    drawn = [
        np.asarray(line.get_xdata())
        for ax in canvas.fig.axes
        for line in ax.lines
        if len(line.get_xdata()) == 3
    ]
    assert drawn
    for x in drawn:
        np.testing.assert_allclose(x, MAGNITUDE)


def test_r8_10_plot_diffusion_applies_its_range_to_the_magnitude(
    qtbot, monkeypatch, tmp_path
):
    """The GUI entry: the G2 refit runs on the file's own axis, the tau-q range on |q|."""
    env = _bayes_env(
        qtbot,
        monkeypatch,
        tmp_path,
        {"cart.h5": lambda p: _cartesian(p, [0.1, 0.2, 0.3])},
    )
    view = env.view
    _select_only(view, 0)
    view.g2_qauto.setChecked(True)
    view.g2_tauto.setChecked(True)
    view.tauq_qmin.setText("0.25")  # qx = 0.2 is below it, |q| = 0.283 is not
    view.tauq_qmax.setText("0.4")
    view.plot_diffusion()
    (xf,) = env.xfs
    assert xf.fit_summary["tauq_success"], view.statusbar.currentMessage()
    np.testing.assert_allclose(xf.fit_summary["tauq_q"], MAGNITUDE[1:])
    np.testing.assert_allclose(xf.fit_summary["q_val"], [0.1, 0.2, 0.3])


def test_r8_10_the_bayesian_launch_gets_the_magnitude(qtbot, monkeypatch, tmp_path):
    env = _bayes_env(
        qtbot,
        monkeypatch,
        tmp_path,
        {"cart.h5": lambda p: _cartesian(p, [0.1, 0.2, 0.3])},
    )
    env.view.tauq_qmin.setText("0.01")
    env.view.tauq_qmax.setText("5")
    _select_only(env.view, 0)
    bundle = env.view._extract_diffusion_for_bayesian()
    assert bundle is not None, env.view.statusbar.currentMessage()
    np.testing.assert_allclose(bundle.x, MAGNITUDE)
    assert bundle.q_axis == ("|q|", Q_UNIT_DISPLAY)
    # the view's file keeps its own values
    np.testing.assert_allclose(env.xfs[0].fit_summary["q_val"], [0.1, 0.2, 0.3])


def test_r8_10_a_cartesian_pair_in_nm_is_converted_after_the_magnitude(tmp_path, qapp):
    _cartesian(tmp_path / "nm.h5", [1.0, 2.0, 3.0], qy=2.0, unit=b"1/nm")
    kernel = ku._fitted_kernel(tmp_path, ["nm.h5"])
    ku._tauq(kernel, [0])
    (xf,) = kernel.get_xf_list()
    np.testing.assert_allclose(xf.fit_summary["tauq_q"], MAGNITUDE)
    assert xf.fit_summary["tauq_q_unit"] == Q_UNIT_DISPLAY


@pytest.mark.parametrize(
    "names",
    [(b"qx", b"qz"), (b"qz", b"qy"), (b"x", b"y")],
)
def test_r8_10_an_axis_that_is_not_radial_or_a_known_pair_is_refused(
    tmp_path, qapp, names
):
    ku._g2_file(tmp_path / "a.h5", [0.1, 0.2, 0.3])
    _cartesian(tmp_path / "odd.h5", [0.1, 0.2, 0.3], names=names)
    kernel = ku._fitted_kernel(tmp_path, ["a.h5", "odd.h5"])
    _canvas, result = ku._tauq(kernel, [0, 1])
    refused = {k: v for k, v in result.items() if "refused" in str(v)}
    assert list(refused) == ["odd.h5"], result
    assert "radial q" in refused["odd.h5"]
    assert kernel.get_xf_list()[0].fit_summary["tauq_success"]
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="radial q"):
        kernel.export_diffusion(str(out), rows=[1])


def test_r8_10_a_qx_qy_grid_is_refused_not_fitted_as_qx(tmp_path, qapp):
    """Two qy per qx: the fitted rows do not record their qy, so |q| is unknown."""
    path = tmp_path / "grid.h5"
    t = np.geomspace(0.002, 2, 40)
    g2 = fit._model(t, 0.004 * np.array([0.1, 0.1, 0.3, 0.3]) ** -2)
    fit._write_file(path, t, g2=g2, q=[0.1, 0.1, 0.3, 0.3])
    with h5py.File(path, "r+") as f:
        for name in ("dqlist", "dplist", "dynamic_num_pts", "map_names", "map_units"):
            del f[K[name]]
        f[K["dqlist"]] = [0.1, 0.3]
        f[K["dplist"]] = [0.2, 0.4]
        f[K["dynamic_num_pts"]] = [2, 2]
        f[K["map_names"]] = [b"qx", b"qy"]
        f[K["map_units"]] = [b"1/A", b"1/A"]
    kernel = ku._kernel(tmp_path, ["grid.h5"])
    (xf,) = kernel.get_xf_list()
    assert xf.qmap.map_names[0] == "qx"
    xf.fit_g2(bounds=fit.BOUNDS, fit_flag=[True] * 4)
    _canvas, result = ku._tauq(kernel, [0])
    assert "refused" in str(result["grid.h5"]) and "qx x qy" in str(result["grid.h5"])
    assert "tauq_q" not in xf.fit_summary or xf.fit_summary["tauq_q"].size == 0


def test_r8_10_the_bayesian_launch_refuses_a_non_radial_axis(
    qtbot, monkeypatch, tmp_path
):
    env = _bayes_env(
        qtbot,
        monkeypatch,
        tmp_path,
        {"odd.h5": lambda p: _cartesian(p, [0.1, 0.2, 0.3], names=(b"qx", b"qz"))},
    )
    env.view.tauq_qmin.setText("0.01")
    env.view.tauq_qmax.setText("5")
    _select_only(env.view, 0)
    assert env.view._extract_diffusion_for_bayesian() is None
    assert "radial q" in env.view.statusbar.currentMessage()


# ================================================== the common case, unchanged
def test_r8_the_single_unit_case_behaves_as_before(tmp_path, qapp):
    """q in inverse angstrom, phi in degrees: G2 panels, CLI selection, diffusion."""
    # G2 panels: equal grids share panels, a different azimuth does not
    _sectors(tmp_path / "a.h5", [0.0, 90.0])
    _sectors(tmp_path / "b.h5", [0.0, 90.0])
    _sectors(tmp_path / "c.h5", [0.0, 45.0])
    kernel = ku._kernel(tmp_path, ["a.h5", "b.h5", "c.h5"])
    xfs = kernel.get_xf_list()
    direct, from_data, _ = r7._plot_both_renderers(xfs[:2])
    assert direct == from_data == [2, 2]
    direct, from_data, _ = r7._plot_both_renderers([xfs[0], xfs[2]])
    assert direct == from_data == [2, 1]
    # CLI: --q-phi picks the nearest sector and names it in degrees
    source, out = tmp_path / "cli", tmp_path / "out"
    source.mkdir()
    r4cli._write_twotime(source / "deg.h5", [0.02], dplist=[10.0, 80.0])
    assert tb.run_twotime_batch(r4cli._args(source, out, q_phi="0.02,10")) == 0
    assert _images(out) == ["deg.h5_bin0000_q0.0200_phi10.0.png"]
    # diffusion: rows and the range are exactly the stored values
    ku._g2_file(tmp_path / "d.h5", [0.1, 0.2, 0.3])
    dkernel = ku._fitted_kernel(tmp_path, ["d.h5"])
    ku._tauq(dkernel, [0], q_range=(0.15, 0.35))
    (xf,) = dkernel.get_xf_list()
    np.testing.assert_array_equal(xf.fit_summary["tauq_q"], [0.2, 0.3])
