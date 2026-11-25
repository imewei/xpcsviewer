"""Regression tests for review round 5, ViewerKernel part.

R5-01  every file ``ViewerKernel`` writes is claimed through ONE ``ExportGuard``
       per export call (inputs = the selected files), so an output symlink to an
       input, or two destinations that are one file, cannot overwrite anything.
       A source scan makes a future un-gated write fail the build.
R5-02  the SAXS background is re-read when its file is replaced on disk (or
       cleared, with a warning, when it is gone) and its revision is exposed.
R5-13  files are paired with the target ROW they came from; a missing target
       does not shift the other files' rows.
R5-05  G2 exports carry each bin's absolute identity (dynamic bin ID, map index,
       numeric q and azimuth), not only the radial Q.

Real NeXus-layout files are written with h5py and loaded through the real
``XpcsFile`` / ``QMap`` / ``ViewerKernel``.
"""

from __future__ import annotations

import ast
import logging
import os
import re
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pyqtgraph as pg
import pytest
from qtpy import QtWidgets

from tests.unit import test_review_r3_maps_roi as r3
from xpcsviewer.fileIO.hdf_reader import clear_connection_pool
from xpcsviewer.module import saxs1d
from xpcsviewer.plothandler.pyqtgraph_handler import ImageViewDev
from xpcsviewer.viewer_kernel import ViewerKernel
from xpcsviewer.xpcs_file import XpcsFile

pytestmark = pytest.mark.unit

K = r3.K
KERNEL_SRC = Path(__file__).resolve().parents[2] / "xpcsviewer" / "viewer_kernel.py"
MODULE_DIR = KERNEL_SRC.parent / "module"


@pytest.fixture(autouse=True)
def _fresh_pool():
    clear_connection_pool()
    yield
    clear_connection_pool()


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")


# ----------------------------------------------------------- SAXS 1D helpers
def _saxs_file(path: Path, value: float) -> Path:
    r3._write_nexus(path)
    with h5py.File(path, "a") as f:
        f[K["saxs_1d"]][:] = value
    return path


def _plot(kernel: ViewerKernel, rows, monkeypatch, **kw):
    """Run the real SAXS1D plot; return [(intensity, alpha)] per drawn curve."""
    drawn: list[tuple[list[float], float]] = []
    real = saxs1d.plot_line_with_marker

    def spy(plot_item, q, iq, plot_id, label, alpha, **k):
        drawn.append((np.asarray(iq).tolist(), round(float(alpha), 2)))
        return real(plot_item, q, iq, plot_id, label, alpha, **k)

    monkeypatch.setattr(saxs1d, "plot_line_with_marker", spy)
    kernel.plot_saxs_1d(
        ImageViewDev(),
        pg.PlotWidget(),
        plot_type=0,
        qmin=0.0,
        qmax=1.0,
        rows=rows,
        **kw,
    )
    return drawn


# --------------------------------------------------------------------- R5-02
def test_r5_02_replaced_background_is_used_after_reload_and_replot(
    tmp_path, qapp, monkeypatch
):
    sample = _saxs_file(tmp_path / "sample.h5", 10.0)
    bkg = _saxs_file(tmp_path / "bkg.h5", 1.0)
    kernel = ViewerKernel(str(tmp_path))
    kernel.add_target([sample.name])
    kernel.select_bkgfile(str(bkg))
    kw = {"subtract_background": True, "weight": 1.0}
    assert _plot(kernel, [0], monkeypatch, **kw)[0][0] == [9.0] * 3

    replacement = _saxs_file(tmp_path / "replacement.h5", 2.0)
    os.replace(replacement, bkg)
    kernel.build(str(tmp_path))  # Reload
    assert _plot(kernel, [0], monkeypatch, **kw)[0][0] == [8.0] * 3


def test_r5_02_background_revision_follows_the_file_on_disk(tmp_path):
    bkg = _saxs_file(tmp_path / "bkg.h5", 1.0)
    kernel = ViewerKernel(str(tmp_path))
    assert kernel.background_revision() is None  # nothing selected
    kernel.select_bkgfile(str(bkg))
    before = kernel.background_revision()
    assert before is not None and before[0] == str(bkg)

    os.replace(_saxs_file(tmp_path / "new.h5", 2.0), bkg)
    after = kernel.background_revision()
    assert after != before  # the SAXS-1D plot request sees the replacement
    assert kernel.refresh_background() is True  # Reload reloads it ...
    assert kernel.refresh_background() is False  # ... exactly once
    assert kernel.background_revision() == after
    np.testing.assert_allclose(kernel.get_bkg_xf().saxs_1d["Iq"], 2.0)


def test_r5_02_reload_flags_a_replaced_background_for_the_gui(tmp_path):
    bkg = _saxs_file(tmp_path / "bkg.h5", 1.0)
    kernel = ViewerKernel(str(tmp_path))
    kernel.select_bkgfile(str(bkg))
    kernel.build(str(tmp_path))
    assert kernel.background_replaced is False
    os.replace(_saxs_file(tmp_path / "new.h5", 2.0), bkg)
    kernel.build(str(tmp_path))
    assert kernel.background_replaced is True
    kernel.build(str(tmp_path))
    assert kernel.background_replaced is False


def test_r5_02_vanished_background_is_cleared_with_a_warning(
    tmp_path, qapp, monkeypatch, caplog
):
    sample = _saxs_file(tmp_path / "sample.h5", 10.0)
    bkg = _saxs_file(tmp_path / "bkg.h5", 1.0)
    kernel = ViewerKernel(str(tmp_path))
    kernel.add_target([sample.name])
    kernel.select_bkgfile(str(bkg))
    bkg.unlink()
    with caplog.at_level(logging.WARNING):
        drawn = _plot(kernel, [0], monkeypatch, subtract_background=True)
    assert drawn[0][0] == [10.0] * 3  # not the deleted file's data
    assert kernel.get_bkg_xf() is None and kernel.background_revision() is None
    assert any("background" in r.getMessage().lower() for r in caplog.records)


def test_r5_02_inventory_xpcsfile_objects_outside_the_locator_cache():
    """Every XpcsFile the kernel itself loads lives in a revision-checked path."""
    tree = ast.parse(KERNEL_SRC.read_text())
    loaders: dict[str, set[str]] = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for call in ast.walk(fn):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                if call.func.id in {"XpcsFile", "XF", "create_xpcs_dataset"}:
                    loaders.setdefault(call.func.id, set()).add(fn.name)
    # only the background loader (select_bkgfile / refresh_background) loads one
    assert loaders == {"XpcsFile": {"select_bkgfile", "refresh_background"}}


# --------------------------------------------------------------------- R5-13
def _three_files(tmp_path):
    names = []
    for name, value in (("a.h5", 1.0), ("b.h5", 2.0), ("c.h5", 3.0)):
        names.append(_saxs_file(tmp_path / name, value).name)
    kernel = ViewerKernel(str(tmp_path))
    kernel.add_target(names)
    return kernel, names


def test_r5_13_missing_target_does_not_break_the_plot_of_the_others(
    tmp_path, qapp, monkeypatch
):
    kernel, names = _three_files(tmp_path)
    (tmp_path / names[0]).unlink()
    kernel.build(str(tmp_path))  # Reload: the target list still has 3 rows
    assert list(kernel.target) == names

    drawn = _plot(kernel, [2], monkeypatch)  # row 2 = c.h5
    # b.h5 (row 1) is dimmed, c.h5 (row 2) is highlighted; a.h5 is skipped
    assert drawn == [([2.0] * 3, 0.35), ([3.0] * 3, 1.0)]


def test_r5_13_highlighting_a_missing_file_dims_nothing_and_does_not_raise(
    tmp_path, qapp, monkeypatch
):
    kernel, names = _three_files(tmp_path)
    (tmp_path / names[0]).unlink()
    drawn = _plot(kernel, [0], monkeypatch)
    # the selected row is gone: nothing is highlighted, no other file is mistaken
    # for it (the compact list's index 0 is b.h5, not row 0)
    assert drawn == [([2.0] * 3, 1.0), ([3.0] * 3, 1.0)]


def test_r5_13_get_xf_rows_pairs_each_file_with_its_target_row(tmp_path):
    kernel, names = _three_files(tmp_path)
    (tmp_path / names[1]).unlink()
    pairs = kernel.get_xf_rows()
    assert [row for row, _ in pairs] == [0, 2]
    assert [os.path.basename(xf.fname) for _, xf in pairs] == [names[0], names[2]]
    assert [r for r, _ in kernel.get_xf_rows(rows=[2, 1, 0])] == [2, 0]
    assert [r for r, _ in kernel.get_xf_rows(rows=2)] == [2]
    assert kernel.get_xf_rows(rows=[7, -1]) == []


def test_r5_13_pg_plot_ignores_highlight_rows_beyond_the_files(
    tmp_path, qapp, monkeypatch
):
    xf = XpcsFile(str(_saxs_file(tmp_path / "a.h5", 1.0)))
    drawn: list[float] = []
    monkeypatch.setattr(
        saxs1d, "plot_line_with_marker", lambda *a, **k: drawn.append(float(a[5]))
    )
    saxs1d.pg_plot([xf], pg.PlotWidget(), plot_type=0, qmin=0.0, qmax=1.0, rows=[5])
    assert drawn == [1.0]


def _subscripted_names(tree: ast.AST) -> list[tuple[str, int]]:
    """(name, line) of every ``x[rows]`` / ``rows[...]`` subscript."""
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        for part in (node.value, node.slice):
            if isinstance(part, ast.Name) and part.id == "rows":
                hits.append((part.id, node.lineno))
    return hits


def test_r5_13_no_module_indexes_a_compacted_list_with_target_rows():
    """``get_xf_list`` drops missing files, so ``rows`` (target rows) must never
    index its result. Subscripting by ``rows`` is allowed only where listed."""
    allowed = {
        # g2mod: ``rows`` is a colour index; ViewerKernel.plot_g2 does not forward
        # target rows and the function rebuilds it as range(len(xf_list)) when
        # missing, so it is always positional there.
        "g2mod.py": 4,
    }
    found = {}
    for path in [KERNEL_SRC, *sorted(MODULE_DIR.glob("*.py"))]:
        hits = _subscripted_names(ast.parse(path.read_text()))
        if hits:
            found[path.name] = len(hits)
    assert found == allowed


# --------------------------------------------------------------------- R5-05
def _write_g2_file(path, dq, dp, mapping, num_pts, g2):
    r3._write_nexus(path)
    n = len(mapping)
    with h5py.File(path, "a") as f:
        for k in (
            "dqlist",
            "dplist",
            "dynamic_index_mapping",
            "dynamic_num_pts",
            "g2",
            "g2_err",
            "map_units",
        ):
            del f[K[k]]
        f[K["dqlist"]] = dq
        f[K["dplist"]] = dp
        f[K["dynamic_index_mapping"]] = mapping
        f[K["dynamic_num_pts"]] = num_pts
        f[K["g2"]] = g2
        f[K["g2_err"]] = np.full((3, n), 0.1)
        f[K["map_units"]] = [b"1/nm", b"deg"]
        f[K["sqlist"]][:] = [0.1, 0.2, 0.3]
    return path


def _sector_pair(tmp_path):
    """Same q grid [0.1, 0.2] and phi grid [0, 90]; occupied bins differ."""
    files = []
    for name, pos in (("sector-zero.h5", 0), ("sector-right.h5", 1)):
        g2 = np.array([[1.9, 1.5], [1.2, 1.1], [1.0, 1.0]]) + pos
        files.append(
            _write_g2_file(
                tmp_path / name, [0.1, 0.2], [0.0, 90.0], [pos, 2 + pos], [2, 2], g2
            )
        )
    return files


_ID = re.compile(r"\[bin=(\d+),map=(\d+),q=([^,\]]+),phi=([^,\]]+)\]")


def _ids(header: str):
    return [(int(b), int(m), float(q), float(p)) for b, m, q, p in _ID.findall(header)]


def test_r5_05_g2_data_headers_carry_the_bin_identity(tmp_path):
    a, b = _sector_pair(tmp_path)
    kernel = ViewerKernel(str(tmp_path))
    kernel.add_target([a.name, b.name])
    out = tmp_path / "out"
    out.mkdir()
    kernel.export_g2(str(out))
    ha, hb = (
        (out / f"{p.stem}_g2_data.txt").read_text().splitlines()[0] for p in (a, b)
    )
    assert ha != hb
    # existing column names are untouched
    assert "G2_q000(q=0.100000)" in ha and "G2_err_q001" in hb
    # one entry per G2 column and per error column: absolute bin, map index, q, phi
    assert _ids(ha) == [(0, 0, 0.1, 0.0)] * 2 + [(1, 2, 0.2, 0.0)] * 2
    assert _ids(hb) == [(0, 1, 0.1, 90.0)] * 2 + [(1, 3, 0.2, 90.0)] * 2


def test_r5_05_identity_agrees_with_the_cli_bin_coordinates(tmp_path):
    from xpcsviewer import viewer_kernel
    from xpcsviewer.cli.twotime_batch import _bin_coords

    path = _write_g2_file(
        tmp_path / "grid.h5",
        [0.1, 0.2],
        [0.0, 90.0],
        [0, 1, 2, 3],
        [2, 2],
        np.ones((3, 4)),
    )
    xf = XpcsFile(str(path))
    columns = viewer_kernel._g2_bin_identities(xf, 4)
    assert columns is not None
    for _bin, absolute, q, phi in columns:
        assert (q, phi) == _bin_coords(xf.qmap, absolute)


def test_r5_05_radial_only_map_has_no_azimuth(tmp_path):
    path = r3._write_nexus(tmp_path / "radial.h5") or tmp_path / "radial.h5"
    kernel = ViewerKernel(str(tmp_path))
    kernel.add_target([path.name])
    out = tmp_path / "out"
    out.mkdir()
    kernel.export_g2(str(out))
    header = (out / "radial_g2_data.txt").read_text().splitlines()[0]
    assert "[bin=0,map=0,q=0.01,phi=NA]" in header
    assert "[bin=1,map=1,q=0.02,phi=NA]" in header


def test_r5_05_fit_and_curve_exports_carry_the_bin_identity(tmp_path):
    a, b = _sector_pair(tmp_path)
    kernel = ViewerKernel(str(tmp_path))
    kernel.add_target([a.name, b.name])
    bounds = np.array([[0.01, 1e-3, 0.5, 0.9], [1.0, 1e2, 1.5, 1.1]])
    for xf in kernel.get_xf_list():
        xf.fit_g2(bounds=bounds, fit_flag=[True, True, False, False])
    out = tmp_path / "out"
    out.mkdir()
    kernel.export_g2(str(out))

    for path, phi, maps in ((a, 0.0, (0, 2)), (b, 90.0, (1, 3))):
        fit = (out / f"{path.stem}_g2_fitting.txt").read_text().splitlines()
        rows = [ln for ln in fit if ln.startswith("# row ")]
        assert rows == [
            f"# row 0: dynamic_bin_id=0 map_index={maps[0]} q=0.1 phi={phi}",
            f"# row 1: dynamic_bin_id=1 map_index={maps[1]} q=0.2 phi={phi}",
        ]
        # the table itself keeps its columns
        assert fit[[ln.startswith("q_value") for ln in fit].index(True)].endswith(
            "fit_status"
        )
        curves = (out / f"{path.stem}_g2_fitted_curves.txt").read_text()
        ids = _ids(curves.splitlines()[0])
        assert ids == [(0, maps[0], 0.1, phi), (1, maps[1], 0.2, phi)]


def test_r5_05_cropped_fit_keeps_the_absolute_bin_of_each_sector(tmp_path):
    """Two sectors at one Q: the fit of Q=0.2 only is bins 2 and 3, not 0 and 1."""
    path = _write_g2_file(
        tmp_path / "grid.h5",
        [0.1, 0.2],
        [0.0, 90.0],
        [0, 1, 2, 3],
        [2, 2],
        np.array([[1.9, 1.8, 1.5, 1.4], [1.2, 1.2, 1.1, 1.1], [1.0] * 4]),
    )
    kernel = ViewerKernel(str(tmp_path))
    kernel.add_target([path.name])
    (xf,) = kernel.get_xf_list()
    bounds = np.array([[0.01, 1e-3, 0.5, 0.9], [1.0, 1e2, 1.5, 1.1]])
    xf.fit_g2(q_range=(0.2, 0.2), bounds=bounds, fit_flag=[True, True, False, False])
    out = tmp_path / "out"
    out.mkdir()
    kernel.export_g2(str(out))
    fit = (out / "grid_g2_fitting.txt").read_text().splitlines()
    assert [ln for ln in fit if ln.startswith("# row ")] == [
        "# row 0: dynamic_bin_id=2 map_index=2 q=0.2 phi=0.0",
        "# row 1: dynamic_bin_id=3 map_index=3 q=0.2 phi=90.0",
    ]


def test_r5_05_summary_bin_ids_are_used_when_the_fit_records_them(tmp_path):
    a, _b = _sector_pair(tmp_path)
    kernel = ViewerKernel(str(tmp_path))
    kernel.add_target([a.name])
    (xf,) = kernel.get_xf_list()
    bounds = np.array([[0.01, 1e-3, 0.5, 0.9], [1.0, 1e2, 1.5, 1.1]])
    fs = xf.fit_g2(bounds=bounds, fit_flag=[True, True, False, False])
    xf.fit_summary = {**fs, "bin_ids": np.array([1, 0])}  # Bayesian-style provenance
    out = tmp_path / "out"
    out.mkdir()
    kernel.export_g2(str(out))
    fit = (out / f"{a.stem}_g2_fitting.txt").read_text().splitlines()
    assert [ln for ln in fit if ln.startswith("# row ")] == [
        "# row 0: dynamic_bin_id=1 map_index=2 q=0.2 phi=0.0",
        "# row 1: dynamic_bin_id=0 map_index=0 q=0.1 phi=0.0",
    ]


# --------------------------------------------------------------------- R5-01
def _g2_files(tmp_path):
    a, b = _sector_pair(tmp_path)
    kernel = ViewerKernel(str(tmp_path))
    kernel.add_target([a.name, b.name])
    out = tmp_path / "out"
    out.mkdir()
    return kernel, a, b, out


def test_r5_01_g2_export_refuses_an_output_symlink_to_an_input(tmp_path, caplog):
    kernel, a, b, out = _g2_files(tmp_path)
    before = a.read_bytes()
    _symlink(out / f"{a.stem}_g2_data.txt", a)
    with caplog.at_level(logging.WARNING):
        kernel.export_g2(str(out))

    assert a.read_bytes() == before
    with h5py.File(a, "r"):  # still a real HDF5 file
        pass
    assert (out / f"{b.stem}_g2_data.txt").exists()  # the other file is exported
    text = caplog.text
    assert "refusing to write" in text and a.name in text
    assert "partially completed" in text


def test_r5_01_g2_export_refuses_two_destinations_that_are_one_file(tmp_path, caplog):
    kernel, a, b, out = _g2_files(tmp_path)
    shared = tmp_path / "shared.txt"
    shared.write_text("untouched\n")
    for p in (a, b):
        _symlink(out / f"{p.stem}_g2_data.txt", shared)
    with caplog.at_level(logging.WARNING):
        kernel.export_g2(str(out))
    first = shared.read_text()
    assert first != "untouched\n"  # the first file was written ...
    assert "G2_q000" in first
    assert "already written in this run" in caplog.text  # ... the second refused
    assert "partially completed" in caplog.text
    assert not (out / f"{b.stem}_g2_fitting.txt").exists()


def test_r5_01_g2_export_into_the_input_folder_cannot_hit_an_input(tmp_path):
    kernel, a, b, _out = _g2_files(tmp_path)
    kernel.export_g2(str(tmp_path))  # same folder as the inputs
    for p in (a, b):
        with h5py.File(p, "r"):
            pass


def _tauq_stub(fname: Path, value: float):
    return SimpleNamespace(
        label=fname.stem,
        fname=str(fname),
        atype="Multitau",
        fit_summary={
            "tauq_success": True,
            "tauq_fit_val": np.array([[value, -1.0], [0.1, 0.2]]),
        },
    )


def _stub_kernel(monkeypatch, xfs):
    vk = ViewerKernel.__new__(ViewerKernel)
    monkeypatch.setattr(vk, "get_xf_list", lambda **k: xfs, raising=False)
    return vk


def test_r5_01_diffusion_export_refuses_input_aliases(tmp_path, monkeypatch, caplog):
    a, b = tmp_path / "scan.h5", tmp_path / "other.h5"
    a.write_bytes(b"input A")
    b.write_bytes(b"input B")
    out = tmp_path / "out"
    out.mkdir()
    _symlink(out / "scan_diffusion_fitting.txt", a)
    _symlink(out / "diffusion_summary.txt", b)
    vk = _stub_kernel(monkeypatch, [_tauq_stub(a, 2.0), _tauq_stub(b, 3.0)])
    with caplog.at_level(logging.WARNING):
        vk.export_diffusion(str(out))
    assert a.read_bytes() == b"input A" and b.read_bytes() == b"input B"
    assert (out / "other_diffusion_fitting.txt").exists()
    assert "partially completed" in caplog.text
    assert caplog.text.count("refusing to write") == 2


def test_r5_01_diffusion_export_refuses_two_destinations_that_are_one_file(
    tmp_path, monkeypatch, caplog
):
    a, b = tmp_path / "scan.h5", tmp_path / "other.h5"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    out = tmp_path / "out"
    out.mkdir()
    shared = tmp_path / "shared.txt"
    for stem in ("scan", "other"):
        _symlink(out / f"{stem}_diffusion_fitting.txt", shared)
    vk = _stub_kernel(monkeypatch, [_tauq_stub(a, 2.0), _tauq_stub(b, 3.0)])
    with caplog.at_level(logging.WARNING):
        vk.export_diffusion(str(out))
    assert "2.000000e+00" in shared.read_text()  # first wins, second is refused
    assert "3.000000e+00" not in shared.read_text()
    assert "already written in this run" in caplog.text


def _saxs_export_files(tmp_path):
    a = _saxs_file(tmp_path / "scan.h5", 2.0)
    b = _saxs_file(tmp_path / "scan_results.h5", 9.0)
    kernel = ViewerKernel(str(tmp_path))
    kernel.add_target([a.name, b.name])
    out = tmp_path / "out"
    out.mkdir()
    return kernel, a, b, out


def test_r5_01_saxs_export_refuses_two_destinations_that_are_one_file(
    tmp_path, qapp, caplog
):
    """Reviewer's probe: two stable aliases to one file, one GUI SAXS export."""
    kernel, a, b, out = _saxs_export_files(tmp_path)
    shared = tmp_path / "shared.txt"
    for p in (a, b):
        _symlink(out / f"{p.stem}_saxs1d.txt", shared)
    with caplog.at_level(logging.WARNING):
        kernel.export_saxs_1d(ImageViewDev(), str(out))
    # the first file (intensity 2) is written; the second (9) must not overwrite it
    assert np.loadtxt(shared)[:, 1].tolist() == [2.0] * 3
    assert "already written in this run" in caplog.text
    assert "partially completed" in caplog.text


def test_r5_01_saxs_export_refuses_an_output_symlink_to_an_input(
    tmp_path, qapp, caplog
):
    kernel, a, b, out = _saxs_export_files(tmp_path)
    before = a.read_bytes()
    _symlink(out / f"{a.stem}_saxs1d.txt", a)
    with caplog.at_level(logging.WARNING):
        kernel.export_saxs_1d(ImageViewDev(), str(out))
    assert a.read_bytes() == before
    assert (out / f"{b.stem}_saxs1d.txt").exists()
    assert "refusing to write" in caplog.text


def test_r5_01_saxs_export_fails_loudly_when_nothing_could_be_written(tmp_path, qapp):
    kernel, a, b, out = _saxs_export_files(tmp_path)
    for p in (a, b):
        _symlink(out / f"{p.stem}_saxs1d.txt", p)
    with pytest.raises(ValueError, match="0/2"):
        kernel.export_saxs_1d(ImageViewDev(), str(out))


# ---- source scan: no file write in viewer_kernel.py bypasses the guard -------
_WRITE_MODES = set("wax+")
_ATTR_WRITERS = {
    "savetxt",
    "savefig",
    "savez",
    "savez_compressed",
    "to_csv",
    "write_text",
    "write_bytes",
    "tofile",
}
_QUALIFIED_WRITERS = {
    ("shutil", "copy"),
    ("shutil", "copy2"),
    ("shutil", "copyfile"),
    ("shutil", "copytree"),
    ("shutil", "move"),
    ("os", "replace"),
    ("os", "rename"),
    ("os", "symlink"),
    ("os", "link"),
    ("np", "save"),
    ("numpy", "save"),
    ("json", "dump"),
    ("pickle", "dump"),
}

# Writing calls that are deliberately not gated, with the reason. Empty: every
# file the kernel writes goes through an ExportGuard.
NOT_GATED: dict[tuple[str, str], str] = {}


def _mode_of(call: ast.Call) -> str:
    mode = ""
    if len(call.args) > 1 and isinstance(call.args[1], ast.Constant):
        mode = str(call.args[1].value)
    for kw in call.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            mode = str(kw.value.value)
    return mode


def _write_calls(fn: ast.AST) -> list[str]:
    found = []
    for call in ast.walk(fn):
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if isinstance(func, ast.Name) and func.id == "open":
            if _WRITE_MODES & set(_mode_of(call)):
                found.append("open")
        elif isinstance(func, ast.Attribute):
            owner = func.value.id if isinstance(func.value, ast.Name) else ""
            if func.attr in _ATTR_WRITERS or (owner, func.attr) in _QUALIFIED_WRITERS:
                found.append(func.attr)
            elif func.attr == "File" and owner == "h5py":
                if _mode_of(call) not in {"", "r"}:
                    found.append("h5py.File")
            elif func.attr.startswith("export_"):  # delegated writer (xf.export_*)
                found.append(func.attr)
    return found


def _is_gated(fn: ast.AST) -> bool:
    # claims itself, or hands its guard to the delegated writer (``guard=``)
    claims = any(
        isinstance(c, ast.Call)
        and (
            (isinstance(c.func, ast.Attribute) and c.func.attr == "claim")
            or any(kw.arg == "guard" for kw in c.keywords)
        )
        for c in ast.walk(fn)
    )
    creates = any(
        isinstance(c, ast.Call)
        and isinstance(c.func, ast.Name)
        and c.func.id == "ExportGuard"
        for c in ast.walk(fn)
    )
    return claims and creates


def test_r5_01_every_file_write_in_the_kernel_goes_through_an_export_guard():
    tree = ast.parse(KERNEL_SRC.read_text())
    writers = {}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            calls = _write_calls(fn)
            if calls:
                writers[fn.name] = (fn, calls)
    # the scan must see the known writers, or it proves nothing
    assert {"export_g2", "export_diffusion", "export_saxs_1d"} <= set(writers)

    ungated = [
        f"{name}: {call}"
        for name, (fn, calls) in writers.items()
        for call in calls
        if not _is_gated(fn) and (name, call) not in NOT_GATED
    ]
    assert not ungated, f"file writes outside an ExportGuard: {ungated}"
    for key, reason in NOT_GATED.items():  # every exemption is justified and live
        assert reason and key[0] in writers


def test_r5_01_every_kernel_export_method_creates_a_guard():
    tree = ast.parse(KERNEL_SRC.read_text())
    exports = [
        fn
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef) and fn.name.startswith("export")
    ]
    assert {fn.name for fn in exports} >= {
        "export_g2",
        "export_diffusion",
        "export_saxs_1d",
    }
    assert [fn.name for fn in exports if not _is_gated(fn)] == []


def test_r5_01_delegated_xpcsfile_exports_receive_the_runs_guard():
    tree = ast.parse(KERNEL_SRC.read_text())
    delegated = [
        call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr.startswith("export_")
        and not (isinstance(call.func.value, ast.Name) and call.func.value.id == "self")
    ]
    assert delegated  # export_saxs_1d delegates to XpcsFile.export_saxs1d
    for call in delegated:
        assert any(kw.arg == "guard" for kw in call.keywords), call.lineno
