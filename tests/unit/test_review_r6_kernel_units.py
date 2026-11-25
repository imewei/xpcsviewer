"""Regression tests for review round 6: shared Q axes, bin identity, averaging.

R6-10  everything drawn on one axis, fitted together or range-filtered together
       is in ONE unit and every export states it: SAXS 1D converts to the first
       plotted file's unit, tau-q works in inverse Angstrom, G2 panels refuse a
       file on another axis. ``convert_q`` is the single conversion.
R6-17  the G2 export names a bin by the file's ACTUAL axes and units.
R6-12  every path that empties or shrinks the target list goes through
       ``FileLocator`` (cache entries and revisions go with it) and a removed
       averaging job's progress record is retired.

Real NeXus-layout files are written with h5py and loaded through the real
``XpcsFile`` / ``QMap`` / ``ViewerKernel``.
"""

from __future__ import annotations

import ast
import gc
import logging
import re
import weakref
from pathlib import Path

import h5py
import numpy as np
import pyqtgraph as pg
import pytest
from matplotlib.figure import Figure
from qtpy import QtWidgets

from tests.unit import test_review_r3_average as r3avg
from tests.unit import test_review_r3_fit_status as fit
from tests.unit import test_review_r3_maps_roi as r3
from tests.unit import test_review_r5_kernel as r5
from xpcsviewer.fileIO.hdf_reader import clear_connection_pool
from xpcsviewer.fileIO.qmap_utils import Q_UNIT_DISPLAY, convert_q, map_axis
from xpcsviewer.plothandler.pyqtgraph_handler import ImageViewDev
from xpcsviewer.viewer_kernel import ViewerKernel
from xpcsviewer.xpcs_file import XpcsFile

pytestmark = pytest.mark.unit

K = r3.K
PACKAGE = Path(__file__).resolve().parents[2] / "xpcsviewer"
NM = "nm⁻¹"


class MplCanvas:
    """The slice of the GUI's matplotlib canvas the tau-q plots use.

    The real ``MplCanvas`` needs the Qt matplotlib backend, which other tests in
    the same worker switch away from; a plain ``Figure`` draws the same axes.
    """

    def __init__(self):
        self.fig = Figure(figsize=(6, 4))

    def clear(self):
        self.fig.clear()

    def subplots(self, *args, **kwargs):
        return self.fig.subplots(*args, **kwargs)

    def draw(self):
        pass


@pytest.fixture(autouse=True)
def _fresh_pool():
    clear_connection_pool()
    yield
    clear_connection_pool()


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# ---------------------------------------------------------------- the helper
def test_convert_q_between_reciprocal_lengths():
    # 1 nm = 10 A, so 1 A^-1 = 10 nm^-1
    assert convert_q(0.1, Q_UNIT_DISPLAY, NM) == pytest.approx(1.0)
    np.testing.assert_allclose(
        convert_q([1, 2, 3], NM, Q_UNIT_DISPLAY), [0.1, 0.2, 0.3]
    )
    # raw spellings found in files are understood; the same unit is unchanged
    assert convert_q(2.0, "1/nm", "1/A") == pytest.approx(0.2)
    assert convert_q(5.0, "px", "px") == 5.0


def test_convert_q_refuses_an_axis_that_is_not_a_reciprocal_length():
    with pytest.raises(ValueError, match="not both reciprocal lengths"):
        convert_q(1.0, "px", Q_UNIT_DISPLAY)
    with pytest.raises(ValueError):
        convert_q(1.0, NM, "deg")


def test_map_axis_reads_the_files_own_names_and_units(tmp_path):
    path = tmp_path / "nm.h5"
    r5._write_g2_file(path, [0.1, 0.2], [0.0, 90.0], [0, 1], [2, 2], np.ones((3, 2)))
    xf = XpcsFile(str(path))
    assert map_axis(xf.qmap, 0) == ("q", NM)
    assert map_axis(xf.qmap, 1)[0] == "phi"
    assert map_axis(None, 0) == ("q", Q_UNIT_DISPLAY)  # no map: the documented default


# ------------------------------------------------------- SAXS 1D (R6-10)
def _saxs_file(path, q, unit: bytes, value=1.0):
    r3._write_nexus(path)
    with h5py.File(path, "r+") as f:
        f[K["sqlist"]][:] = q
        f[K["saxs_1d"]][:] = value
        del f[K["map_units"]]
        f[K["map_units"]] = [unit, b"deg"]
    return path


def _kernel(tmp_path, names):
    kernel = ViewerKernel(str(tmp_path))
    kernel.add_target(list(names))
    return kernel


def _draw(kernel, **kw):
    widget = pg.PlotWidget()
    kw = {"plot_type": 0, "qmin": 0.0, "qmax": 1.0, **kw}
    kernel.plot_saxs_1d(ImageViewDev(), widget, **kw)
    item = widget.getPlotItem()
    # each curve is drawn as a line item plus a marker item: keep the distinct x grids
    curves = {tuple(np.round(d.getData()[0], 9)) for d in item.listDataItems()}
    return item.getAxis("bottom").labelText, sorted(curves)


@pytest.fixture
def mixed_saxs(tmp_path, qapp):
    a = _saxs_file(tmp_path / "a.h5", [0.01, 0.02, 0.03], b"1/A")
    b = _saxs_file(tmp_path / "b.h5", [0.1, 0.2, 0.3], b"1/nm")
    return tmp_path, a, b


def test_r6_10_saxs_overlay_is_on_the_first_files_axis(mixed_saxs):
    tmp, a, b = mixed_saxs
    label, curves = _draw(_kernel(tmp, [a.name, b.name]))
    assert Q_UNIT_DISPLAY in label and NM not in label
    # both files are the same physical q grid, drawn once, in A^-1
    assert len(curves) == 1
    np.testing.assert_allclose(curves[0], [0.01, 0.02, 0.03])


def test_r6_10_saxs_overlay_follows_the_first_file_when_order_swaps(mixed_saxs):
    tmp, a, b = mixed_saxs
    label, curves = _draw(_kernel(tmp, [b.name, a.name]))
    assert NM in label and len(curves) == 1
    np.testing.assert_allclose(curves[0], [0.1, 0.2, 0.3])


def test_r6_10_saxs_q_range_is_read_in_the_displayed_unit(mixed_saxs):
    tmp, a, b = mixed_saxs
    # 0.015 .. 0.025 A^-1 holds one point of each file (0.02 A^-1 = 0.2 nm^-1)
    _label, curves = _draw(_kernel(tmp, [a.name, b.name]), qmin=0.015, qmax=0.025)
    np.testing.assert_allclose(curves, [[0.02]])


def test_r6_10_saxs_q2_weighting_uses_the_displayed_unit(tmp_path, qapp):
    a = _saxs_file(tmp_path / "a.h5", [0.01, 0.02, 0.03], b"1/A", value=2.0)
    b = _saxs_file(tmp_path / "b.h5", [0.1, 0.2, 0.3], b"1/nm", value=2.0)
    kernel = _kernel(tmp_path, [a.name, b.name])
    widget = pg.PlotWidget()
    kernel.plot_saxs_1d(
        ImageViewDev(), widget, plot_type=0, plot_norm=1, qmin=0.0, qmax=1.0
    )
    # every curve is a line item plus a marker item: keep the distinct I*q^2 curves.
    # The files are one physical curve, so with ONE weighting unit there is one.
    ys = {
        tuple(np.round(d.getData()[1], 12))
        for d in widget.getPlotItem().listDataItems()
    }
    assert len(ys) == 1
    np.testing.assert_allclose(
        next(iter(ys)), 2.0 * np.array([0.01, 0.02, 0.03]) ** 2, rtol=1e-9
    )


def test_r6_10_saxs_overlay_skips_a_file_on_an_incompatible_axis(
    tmp_path, qapp, caplog
):
    a = _saxs_file(tmp_path / "a.h5", [0.01, 0.02, 0.03], b"1/A")
    px = _saxs_file(tmp_path / "px.h5", [1.0, 2.0, 3.0], b"px")
    kernel = _kernel(tmp_path, [a.name, px.name])
    with caplog.at_level(logging.WARNING):
        label, curves = _draw(kernel)
    assert Q_UNIT_DISPLAY in label
    np.testing.assert_allclose(curves, [[0.01, 0.02, 0.03]])  # no px curve
    assert any(
        "px.h5" in r.getMessage() and "skipping" in r.getMessage()
        for r in caplog.records
    )


def test_r6_10_saxs_export_still_states_each_files_own_unit(mixed_saxs, tmp_path):
    tmp, a, b = mixed_saxs
    out = tmp_path / "out"
    out.mkdir()
    kernel = _kernel(tmp, [a.name, b.name])
    kernel.export_saxs_1d(ImageViewDev(), str(out))
    heads = [(out / f"{p.stem}_saxs1d.txt").read_text().splitlines()[0] for p in (a, b)]
    assert "1/Angstrom" in heads[0] and "1/nm" in heads[1]


# ------------------------------------------------------ diffusion (R6-10)
def _g2_file(path, q, unit: bytes | None = None, scale=1.0):
    t = np.geomspace(0.002, 2, 40)
    rng = np.random.default_rng(42)
    base = np.array([0.1, 0.2, 0.3])
    g2 = fit._model(t, 0.004 * base**-2) + rng.normal(0, 0.001, (len(t), 3))
    fit._write_file(path, t, g2=g2, q=np.asarray(q, dtype=float))
    if unit is not None:
        with h5py.File(path, "r+") as f:
            del f["/xpcs/qmap/map_units"]
            f["/xpcs/qmap/map_units"] = np.array([unit, b"deg"])
    return path


def _fitted_kernel(tmp_path, names):
    kernel = _kernel(tmp_path, names)
    for xf in kernel.get_xf_list():
        summary = xf.fit_g2(bounds=fit.BOUNDS, fit_flag=[True] * 4)
        assert all(summary["fit_success"])
    return kernel


def _tauq(kernel, rows, q_range=(0.01, 5.0)):
    canvas = MplCanvas()
    result = kernel.plot_tauq(
        hdl=canvas,
        bounds=[[1e-5, -5], [100.0, 1]],
        rows=rows,
        q_range=list(q_range),
        offset=0,
        fit_flag=[True, True],
    )
    return canvas, result


@pytest.fixture
def mixed_diffusion(tmp_path, qapp):
    _g2_file(tmp_path / "a.h5", [0.1, 0.2, 0.3])
    _g2_file(tmp_path / "nm.h5", [1.0, 2.0, 3.0], unit=b"1/nm")
    return tmp_path, _fitted_kernel(tmp_path, ["a.h5", "nm.h5"])


def test_r6_10_diffusion_fits_every_file_in_inverse_angstrom(mixed_diffusion):
    _tmp, kernel = mixed_diffusion
    canvas, _result = _tauq(kernel, [0, 1])
    xa, xn = kernel.get_xf_list()
    assert canvas.fig.axes[0].get_xlabel() == "$q (\\AA^{-1})$"
    for xf in (xa, xn):
        assert xf.fit_summary["tauq_success"]
        np.testing.assert_allclose(xf.fit_summary["tauq_q"], [0.1, 0.2, 0.3])
        assert xf.fit_summary["tauq_q_unit"] == Q_UNIT_DISPLAY
    # the same physical data gives the same diffusion law, whatever the file's unit
    np.testing.assert_allclose(
        xn.fit_summary["tauq_fit_val"], xa.fit_summary["tauq_fit_val"], rtol=1e-6
    )
    # the file's own G2 fit is not rewritten in the other unit
    np.testing.assert_allclose(xn.fit_summary["q_val"], [1.0, 2.0, 3.0])


def test_r6_10_diffusion_q_range_is_read_in_inverse_angstrom(mixed_diffusion):
    _tmp, kernel = mixed_diffusion
    _tauq(kernel, [0, 1], q_range=(0.15, 0.35))
    for xf in kernel.get_xf_list():
        np.testing.assert_allclose(xf.fit_summary["tauq_q"], [0.2, 0.3])


def test_r6_10_diffusion_export_records_the_q_unit(mixed_diffusion, tmp_path):
    tmp, kernel = mixed_diffusion
    _tauq(kernel, [0, 1])
    out = tmp_path / "out"
    out.mkdir()
    kernel.export_diffusion(str(out), rows=[0, 1])
    nm_text = (out / "nm_diffusion_fitting.txt").read_text()
    assert f"# Q unit: {Q_UNIT_DISPLAY}" in nm_text
    assert f"axis is q in {NM}" in nm_text  # the file's own unit is stated too
    summary = (out / "diffusion_summary.txt").read_text().splitlines()
    table = [ln.split("\t") for ln in summary if "\t" in ln]
    assert table[0][-1] == "q_unit"
    assert [row[-1] for row in table[1:]] == [Q_UNIT_DISPLAY, Q_UNIT_DISPLAY]
    # same physical fit: the exported coefficients agree between the two files
    a_vals = [float(row[2]) for row in table[1:]]
    assert a_vals[0] == pytest.approx(a_vals[1], rel=1e-6)


def test_r6_10_diffusion_preplot_uses_one_unit(mixed_diffusion):
    _tmp, kernel = mixed_diffusion
    canvas = MplCanvas()
    kernel.plot_tauq_pre(hdl=canvas, rows=[0, 1])
    drawn = [
        np.asarray(line.get_xdata())
        for ax in canvas.fig.axes
        for line in ax.lines
        if len(line.get_xdata()) == 3
    ]
    assert drawn, "no curve drawn"
    for x in drawn:
        np.testing.assert_allclose(x, [0.1, 0.2, 0.3])


def test_r6_10_diffusion_refuses_a_file_that_is_not_on_a_reciprocal_length(
    tmp_path, qapp
):
    _g2_file(tmp_path / "a.h5", [0.1, 0.2, 0.3])
    _g2_file(tmp_path / "px.h5", [1.0, 2.0, 3.0], unit=b"px")
    kernel = _fitted_kernel(tmp_path, ["a.h5", "px.h5"])
    _canvas, result = _tauq(kernel, [0, 1])
    messages = {k: v for k, v in result.items() if "refused" in str(v)}
    assert list(messages) == ["px.h5"] and "reciprocal length" in messages["px.h5"]
    xa = kernel.get_xf_list()[0]
    assert xa.fit_summary["tauq_success"]
    out = tmp_path / "out"
    out.mkdir()
    kernel.export_diffusion(str(out), rows=[0, 1])  # the compatible file is exported
    assert (out / "a_diffusion_fitting.txt").exists()
    assert not (out / "px_diffusion_fitting.txt").exists()
    with pytest.raises(ValueError, match="reciprocal length"):
        kernel.export_diffusion(str(out), rows=[1])


# ------------------------------------------------------------ G2 (R6-10/17)
def test_r6_10_g2_plot_converts_a_file_on_another_unit(tmp_path, qapp, monkeypatch):
    _g2_file(tmp_path / "a.h5", [0.1, 0.2, 0.3])
    _g2_file(tmp_path / "nm.h5", [1.0, 2.0, 3.0], unit=b"1/nm")
    kernel = _kernel(tmp_path, ["a.h5", "nm.h5"])
    seen = []
    from xpcsviewer.module import g2mod

    monkeypatch.setattr(
        g2mod,
        "pg_plot",
        lambda handler, xfs, *a, **k: seen.append([x.label for x in xfs]),
    )
    kernel.plot_g2(None, None, None, None, rows=[0, 1])
    # a 1/nm file is CONVERTED by g2mod.get_data now, no longer left out here
    assert seen == [["a.h5", "nm.h5"]]


def _cartesian_file(path):
    r5._write_g2_file(
        path,
        [40.0, 50.0],
        [120.0, 130.0],
        [0, 3],
        [2, 2],
        np.array([[1.9, 1.5], [1.2, 1.1], [1.0, 1.0]]),
    )
    with h5py.File(path, "r+") as f:
        del f[K["map_names"]], f[K["map_units"]]
        f[K["map_names"]] = [b"x", b"y"]
        f[K["map_units"]] = [b"px", b"px"]
    return path


def test_r6_17_cartesian_bins_are_exported_with_their_own_axes(tmp_path):
    path = _cartesian_file(tmp_path / "cartesian.h5")
    kernel = _kernel(tmp_path, [path.name])
    (xf,) = kernel.get_xf_list()
    assert xf.get_g2_data()[-1] == ["x=40.0 px, y=120.0 px", "x=50.0 px, y=130.0 px"]
    out = tmp_path / "out"
    out.mkdir()
    kernel.export_g2(str(out))
    lines = (out / "cartesian_g2_data.txt").read_text().splitlines()
    assert "[bin=0,map=0,x=40.0px,y=120.0px]" in lines[0]
    assert "[bin=1,map=3,x=50.0px,y=130.0px]" in lines[0]
    assert "phi" not in lines[0] and "q=" not in lines[0].replace("G2_q", "")
    assert lines[1] == "# axes: x [px], y [px]"


def test_r6_17_cartesian_fit_and_curve_exports_carry_the_axes(tmp_path):
    path = _cartesian_file(tmp_path / "cartesian.h5")
    kernel = _kernel(tmp_path, [path.name])
    (xf,) = kernel.get_xf_list()
    xf.fit_g2(bounds=fit.BOUNDS, fit_flag=[True, True, False, False])
    out = tmp_path / "out"
    out.mkdir()
    kernel.export_g2(str(out))
    fit_lines = (out / "cartesian_g2_fitting.txt").read_text().splitlines()
    assert [ln for ln in fit_lines if ln.startswith("# row ")] == [
        "# row 0: dynamic_bin_id=0 map_index=0 x=40.0px y=120.0px",
        "# row 1: dynamic_bin_id=1 map_index=3 x=50.0px y=130.0px",
    ]
    curves = (out / "cartesian_g2_fitted_curves.txt").read_text().splitlines()[0]
    assert "[bin=1,map=3,x=50.0px,y=130.0px]" in curves


def test_r6_17_q_phi_files_keep_their_wording_and_state_the_unit(tmp_path):
    a, _b = r5._sector_pair(tmp_path)  # a real (q, phi) map in nm^-1 and degrees
    kernel = _kernel(tmp_path, [a.name])
    out = tmp_path / "out"
    out.mkdir()
    kernel.export_g2(str(out))
    lines = (out / f"{a.stem}_g2_data.txt").read_text().splitlines()
    assert r5._ids(lines[0]) == [(0, 0, 0.1, 0.0)] * 2 + [(1, 2, 0.2, 0.0)] * 2
    assert lines[1] == f"# axes: q [{NM}], phi [deg]"


# ------------------------------------------------- target list (R6-12)
def _average_kwargs(tmp_path):
    return {
        "save_path": str(tmp_path / "avg.h5"),
        "chunk_size": 256,
        "avg_blmin": 0.0,
        "avg_blmax": 2.0,
        "avg_qindex": 0,
        "avg_window": 3,
        "fields": ["g2", "g2_err"],
    }


def _averaging_kernel(tmp_path, n=2):
    for i in range(n):
        r3avg._write_nexus(tmp_path / f"scan{i}.h5", g2=1 + i / 100)
    kernel = ViewerKernel(str(tmp_path))
    kernel.add_target([f"scan{i}.h5" for i in range(n)])
    return kernel


def test_r6_12_submitting_a_job_releases_the_loaded_files(tmp_path, qapp):
    kernel = _averaging_kernel(tmp_path)
    refs = [weakref.ref(xf) for xf in kernel.get_xf_list()]
    assert len(kernel.cache) == 2 and len(kernel._cache_identity) == 2
    kernel.submit_job(**_average_kwargs(tmp_path))
    assert len(kernel.target) == 0
    assert kernel.cache == {} and kernel._cache_identity == {}
    gc.collect()
    assert all(ref() is None for ref in refs)
    # the worker kept its own copy of the names
    assert kernel.avg_worker.file_list == ["scan0.h5", "scan1.h5"]


def test_r6_12_removing_a_job_retires_its_progress_record(tmp_path, qapp):
    kernel = _averaging_kernel(tmp_path)
    kernel.submit_job(**_average_kwargs(tmp_path))
    jid = kernel.avg_worker.jid
    kernel.update_avg_values((jid, 1.0))
    assert jid in kernel.avg_worker_active
    kernel.remove_job()
    assert kernel.avg_worker is None and kernel.avg_worker_active == {}
    kernel.remove_job()  # nothing to remove: no error


def test_r6_12_repeated_jobs_do_not_accumulate(tmp_path, qapp):
    for i in range(4):
        r3avg._write_nexus(tmp_path / f"s{i}.h5", g2=1.0 + i)
    kernel = ViewerKernel(str(tmp_path))
    refs = []
    for batch in range(2):
        kernel.add_target([f"s{2 * batch}.h5", f"s{2 * batch + 1}.h5"])
        refs += [weakref.ref(xf) for xf in kernel.get_xf_list()]
        kernel.submit_job(**_average_kwargs(tmp_path))
        kernel.update_avg_values((kernel.avg_worker.jid, 1.0))
        kernel.remove_job()
        gc.collect()
        assert kernel.cache == {} and kernel.avg_worker_active == {}
    assert all(ref() is None for ref in refs)


def test_r6_12_removed_targets_leave_cache_and_revisions(tmp_path, qapp):
    kernel = _averaging_kernel(tmp_path, n=3)
    refs = [weakref.ref(xf) for xf in kernel.get_xf_list([0])]
    kernel.remove_target(["scan0.h5"])
    assert len(kernel.cache) == 2 and len(kernel._cache_identity) == 2
    gc.collect()
    assert refs[0]() is None
    kernel.reset_kernel()
    assert len(kernel.target) == 0 and kernel.cache == {}
    assert kernel._cache_identity == {}


_SHRINKING = {"clear", "remove", "pop", "replace", "__delitem__"}
# the ONLY functions that may change the target list: FileLocator's own methods
# (they drop the cache entries and their revisions with it). reorder_target pops
# and re-inserts the same row, so the list never shrinks.
_ALLOWED = {
    ("file_locator.py", "clear_target"),
    ("file_locator.py", "remove_target"),
    ("file_locator.py", "reorder_target"),
}


def _is_target(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "target"


def _target_mutations(tree: ast.AST):
    """``(function, line, what)`` for each call / assignment that can shrink or
    replace ``<obj>.target`` (the locator's target list model)."""
    found = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.stack = ["<module>"]

        def visit_FunctionDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            f = node.func
            if (
                isinstance(f, ast.Attribute)
                and f.attr in _SHRINKING
                and _is_target(f.value)
            ):
                found.append((self.stack[-1], node.lineno, f"target.{f.attr}()"))
            self.generic_visit(node)

        def visit_Assign(self, node):
            for t in node.targets:
                if _is_target(t) and self.stack[-1] != "__init__":
                    found.append((self.stack[-1], node.lineno, "target = ..."))
            self.generic_visit(node)

        def visit_Delete(self, node):
            for t in node.targets:
                if isinstance(t, ast.Subscript) and _is_target(t.value):
                    found.append((self.stack[-1], node.lineno, "del target[...]"))
            self.generic_visit(node)

    Visitor().visit(tree)
    return found


def test_r6_12_only_the_locator_changes_the_target_list():
    """A source scan: no module reaches into ``.target`` to empty or shrink it."""
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        for func, line, what in _target_mutations(ast.parse(path.read_text())):
            if (path.name, func) not in _ALLOWED:
                offenders.append(f"{path.relative_to(PACKAGE)}:{line} {func}: {what}")
    assert offenders == []


def test_r6_12_the_scan_sees_a_direct_clear():
    tree = ast.parse("class K:\n    def go(self):\n        self.target.clear()\n")
    assert _target_mutations(tree) == [("go", 3, "target.clear()")]


def test_r6_12_the_locator_methods_drop_what_they_loaded():
    """Each allowed shrinking method is the one that also forgets the cache."""
    tree = ast.parse((PACKAGE / "file_locator.py").read_text())
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        for fn in (n for n in cls.body if isinstance(n, ast.FunctionDef)):
            if fn.name in ("clear_target", "remove_target"):
                src = ast.unparse(fn)
                assert "_cache_identity" in src and "cache" in src, fn.name


_ID = re.compile(r"\[bin=(\d+),map=(\d+),q=([^,\]]+),phi=([^,\]]+)\]")


def test_r6_17_the_legacy_token_still_matches_the_existing_reader(tmp_path):
    a, _b = r5._sector_pair(tmp_path)
    kernel = _kernel(tmp_path, [a.name])
    out = tmp_path / "out"
    out.mkdir()
    kernel.export_g2(str(out))
    header = (out / f"{a.stem}_g2_data.txt").read_text().splitlines()[0]
    assert len(_ID.findall(header)) == 4
