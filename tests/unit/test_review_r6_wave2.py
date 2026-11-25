"""Regression tests for review round 6, wave 2: R6-10 / R6-17 leftovers.

A  R6-10  ``g2mod.get_data`` is the ONE place that puts G2 files on one Q axis:
          the unit is the first file's, a q range is converted into each file's
          own unit before ``get_g2_data`` and the q it returns is converted
          back; a file on an incompatible axis gets an empty entry (the lists
          stay aligned with ``xf_list``) and a warning that names it.
B  R6-17  the Bayesian CSV / netCDF writers and the two-time CLI carry the
          file's real two axis names, values and units for maps that are not
          (q, phi); (q, phi) maps keep every existing column, attribute, name.
C  R6    a one-element calibration array is the scalar it holds; a Q map that
          really is a substituted default says so (``is_default`` + a warning).

Every file is written with h5py and read back through the real ``XpcsFile``.
"""

from __future__ import annotations

import csv
import logging

import h5py
import numpy as np
import pyqtgraph as pg
import pytest
from qtpy import QtWidgets

from tests.unit import test_review_r3_maps_roi as r3
from tests.unit import test_review_r4_cli_average as r4cli
from tests.unit import test_review_r5_bayes_gui as bayes
from tests.unit import test_review_r5_kernel as r5
from tests.unit import test_review_r6_kernel_units as ku
from xpcsviewer.cli import twotime_batch as tb
from xpcsviewer.fileIO.hdf_reader import clear_connection_pool
from xpcsviewer.fileIO.qmap_utils import QMap
from xpcsviewer.module import g2mod
from xpcsviewer.xpcs_file import XpcsFile

pytestmark = pytest.mark.unit

K = r3.K
NM = ku.NM


@pytest.fixture(autouse=True)
def _fresh_pool():
    clear_connection_pool()
    yield
    clear_connection_pool()


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# ===================================================================== A
def _mixed(tmp_path, order=("a.h5", "nm.h5"), extra=None):
    """A (A^-1) and a (nm^-1) file holding the same physical q [0.1, 0.2, 0.3]."""
    ku._g2_file(tmp_path / "a.h5", [0.1, 0.2, 0.3])
    ku._g2_file(tmp_path / "nm.h5", [1.0, 2.0, 3.0], unit=b"1/nm")
    for name, q, unit in extra or ():
        ku._g2_file(tmp_path / name, q, unit=unit)
    kernel = ku._kernel(tmp_path, [*order, *(e[0] for e in extra or ())])
    return kernel.get_xf_list()


def test_r6_10_get_data_converts_a_nm_file_to_the_first_files_unit(tmp_path):
    xfs = _mixed(tmp_path)
    q, tel, g2, g2_err, labels = g2mod.get_data(xfs, q_range=(0.15, 0.35))
    # 0.15..0.35 A^-1 is 1.5..3.5 nm^-1: the nm file keeps bins 2 and 3 as well
    np.testing.assert_allclose(q[0], [0.2, 0.3])
    np.testing.assert_allclose(q[1], [0.2, 0.3])  # returned in the shared unit
    assert g2[0].shape == g2[1].shape == (40, 2)
    assert len(labels[1]) == 2
    # the file's own q is not rewritten
    np.testing.assert_allclose(xfs[1].get_g2_data()[0], [1.0, 2.0, 3.0])


def test_r6_10_get_data_follows_the_first_file_when_the_order_swaps(tmp_path):
    xfs = _mixed(tmp_path, order=("nm.h5", "a.h5"))
    q, _tel, g2, _err, _labels = g2mod.get_data(xfs, q_range=(1.5, 3.5))
    np.testing.assert_allclose(q[0], [2.0, 3.0])
    np.testing.assert_allclose(q[1], [2.0, 3.0])  # A^-1 file, shown in nm^-1
    assert g2[0].shape == g2[1].shape == (40, 2)


def test_r6_10_get_data_without_a_q_range_still_uses_one_unit(tmp_path):
    xfs = _mixed(tmp_path)
    q, *_rest = g2mod.get_data(xfs)
    np.testing.assert_allclose(q[0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(q[1], [0.1, 0.2, 0.3])


def test_r6_10_get_data_leaves_out_an_incompatible_file_and_names_it(tmp_path, caplog):
    xfs = _mixed(tmp_path, extra=[("px.h5", [1.0, 2.0, 3.0], b"px")])
    assert [x.label for x in xfs] == ["a.h5", "nm.h5", "px.h5"]
    with caplog.at_level(logging.WARNING):
        q, tel, g2, g2_err, labels = g2mod.get_data(xfs, q_range=(0.15, 0.35))
    # every list stays aligned with xf_list: the left-out file is an EMPTY entry
    assert all(len(v) == 3 for v in (q, tel, g2, g2_err, labels))
    assert np.size(q[2]) == 0 and np.size(g2[2]) == 0 and len(labels[2]) == 0
    np.testing.assert_allclose(q[1], [0.2, 0.3])
    assert any(
        "px.h5" in r.getMessage() and r.levelno >= logging.WARNING
        for r in caplog.records
    )
    # a renderer sees "a file without Q bins": it draws and fits nothing
    assert g2mod.compute_geometry(g2, "multiple") == (2, 3)


class _Hdl(pg.GraphicsLayoutWidget):
    def adjust_canvas_size(self, **_kw):
        pass


def test_r6_10_pg_plot_fits_each_file_in_its_own_unit(tmp_path, qapp):
    xfs = _mixed(tmp_path)
    seen = {}
    for xf in xfs:

        def spy(q_range, *a, _label=xf.label, **k):
            seen[_label] = q_range

        xf.fit_g2 = spy
    g2mod.pg_plot(
        _Hdl(),
        xfs,
        (0.15, 0.35),
        (1e-3, 10.0),
        (0.9, 1.5),
        show_fit=True,
        bounds=ku.fit.BOUNDS,
        fit_flag=[True] * 4,
    )
    np.testing.assert_allclose(seen["a.h5"], (0.15, 0.35))
    np.testing.assert_allclose(seen["nm.h5"], (1.5, 3.5))  # the file's own unit


# ===================================================================== B
def _cartesian(tmp_path):
    return XpcsFile(str(ku._cartesian_file(tmp_path / "cartesian.h5")))


def test_r6_17_resolve_bin_phi_follows_the_index_mapping(tmp_path):
    from xpcsviewer.fitting.bayesian_assembly import resolve_bin_phi, resolve_g2_bins

    zero, right = (XpcsFile(str(p)) for p in r5._sector_pair(tmp_path))
    for xf, phi in ((zero, 0.0), (right, 90.0)):
        ids = [b[0] for b in resolve_g2_bins(xf, None)]
        # both occupied bins of a sector file sit on ONE azimuth of the map
        assert resolve_bin_phi(xf, ids).tolist() == [phi, phi]


def test_r6_17_resolve_bin_axes_reads_the_files_own_axes(tmp_path):
    from xpcsviewer.fitting.bayesian_assembly import resolve_bin_axes, resolve_bin_phi

    xf = _cartesian(tmp_path)
    axes = resolve_bin_axes(xf, [0, 1])
    assert axes["names"] == ("x", "y") and axes["units"] == ("px", "px")
    np.testing.assert_allclose(axes["axis1"], [120.0, 130.0])
    assert np.isnan(resolve_bin_phi(xf, [0, 1])).all()  # the q/phi column stays empty
    sector, _ = r5._sector_pair(tmp_path)
    q_axes = resolve_bin_axes(XpcsFile(str(sector)), [0, 1])
    assert q_axes["names"] == ("q", "phi") and q_axes["units"] == (NM, "deg")


def _csv_rows(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        return header, list(reader)


def test_r6_17_csv_names_the_cartesian_axes_after_the_existing_columns(tmp_path):
    from xpcsviewer.fitting.bayesian_assembly import resolve_bin_axes
    from xpcsviewer.fitting.viz import export_bayesian_csv

    xf = _cartesian(tmp_path)
    axes = resolve_bin_axes(xf, [0, 1])
    kw = {
        "failed_mask": np.array([False, False]),
        "bin_ids": np.array([0, 1]),
        "bin_phi": np.array([np.nan, np.nan]),
    }
    fit_val = np.ones((2, 2, 4))
    q_val = np.array([40.0, 50.0])
    plain = tmp_path / "plain.csv"
    export_bayesian_csv(plain, fit_val, q_val, "single", **kw)
    out = tmp_path / "cart.csv"
    export_bayesian_csv(out, fit_val, q_val, "single", bin_axes=axes, **kw)
    h0, r0 = _csv_rows(plain)
    header, rows = _csv_rows(out)
    # existing columns: same names, same positions, same cells
    assert header[: len(h0)] == h0
    assert [r[: len(h0)] for r in rows] == r0
    extra = dict(
        zip(header[len(h0) :], zip(*(r[len(h0) :] for r in rows)), strict=True)
    )
    assert list(extra) == [
        "axis0_name",
        "axis0_unit",
        "axis1_name",
        "axis1_value",
        "axis1_unit",
    ]
    assert extra["axis0_name"] == ("x", "x") and extra["axis0_unit"] == ("px", "px")
    assert extra["axis1_name"] == ("y", "y") and extra["axis1_unit"] == ("px", "px")
    assert [float(v) for v in extra["axis1_value"]] == [120.0, 130.0]


def test_r6_17_csv_of_a_q_phi_map_keeps_its_columns_and_gains_the_units(tmp_path):
    from xpcsviewer.fitting.bayesian_assembly import resolve_bin_axes
    from xpcsviewer.fitting.viz import export_bayesian_csv

    sector, _ = r5._sector_pair(tmp_path)
    axes = resolve_bin_axes(XpcsFile(str(sector)), [0, 1])
    kw = {"bin_ids": np.array([0, 1]), "bin_phi": np.array([0.0, 0.0])}
    plain, out = tmp_path / "p.csv", tmp_path / "q.csv"
    export_bayesian_csv(plain, np.ones((2, 2, 4)), np.array([0.1, 0.2]), "single", **kw)
    export_bayesian_csv(
        out, np.ones((2, 2, 4)), np.array([0.1, 0.2]), "single", bin_axes=axes, **kw
    )
    h0, r0 = _csv_rows(plain)
    header, rows = _csv_rows(out)
    assert header[: len(h0)] == h0 and h0[0] == "q_value" and h0[-1] == "phi"
    assert [r[: len(h0)] for r in rows] == r0
    assert rows[0][len(h0) :] == ["q", NM, "phi", "0", "deg"]


def test_r6_17_netcdf_attrs_name_the_cartesian_axes(tmp_path, monkeypatch):
    from xpcsviewer.fitting.bayesian_assembly import resolve_bin_axes
    from xpcsviewer.fitting.viz import export_bayesian_diagnostics

    bayes._patch_backends(monkeypatch)
    xf = _cartesian(tmp_path)
    sink: dict = {}
    export_bayesian_diagnostics(
        tmp_path / "d.nc",
        bayes._fake_results(sink, 2),
        failed_mask=np.array([False, False]),
        bin_ids=np.array([0, 1]),
        q_val=np.array([40.0, 50.0]),
        bin_phi=np.array([np.nan, np.nan]),
        bin_axes=resolve_bin_axes(xf, [0, 1]),
        source="cartesian.h5",
    )
    attrs = sink["d_q001.nc"]
    assert attrs["axis0_name"] == "x" and attrs["axis0_unit"] == "px"
    assert attrs["axis1_name"] == "y" and attrs["axis1_unit"] == "px"
    assert attrs["axis1_value"] == 130.0
    assert "phi" not in attrs


def test_r6_17_netcdf_attrs_of_a_plain_map_are_unchanged_without_bin_axes(
    tmp_path, monkeypatch
):
    from xpcsviewer.fitting.viz import export_bayesian_diagnostics

    bayes._patch_backends(monkeypatch)
    sink: dict = {}
    export_bayesian_diagnostics(
        tmp_path / "d.nc",
        bayes._fake_results(sink, 1),
        bin_ids=np.array([7]),
        q_val=np.array([0.1]),
        bin_phi=np.array([90.0]),
    )
    assert set(sink["d_q000.nc"]) == {
        "bin_status",
        "q_index",
        "bin_id",
        "q_value",
        "phi",
    }


def test_r6_17_assembled_summary_carries_the_axes(tmp_path):
    from xpcsviewer.fitting.bayesian_assembly import (
        assemble_fit_summary,
        resolve_bin_axes,
    )
    from xpcsviewer.fitting.models import single_exp_func

    axes = resolve_bin_axes(_cartesian(tmp_path), [0, 1])
    args = (
        {},
        np.array([40.0, 50.0]),
        np.array([1.0, 2.0]),
        "single",
        single_exp_func,
    )
    assert assemble_fit_summary(*args, bin_axes=axes)["bin_axes"] == axes
    assert assemble_fit_summary(*args)["bin_axes"] is None  # callers that omit it
    with pytest.raises(ValueError, match="bin_axes"):
        assemble_fit_summary(*args, bin_axes={**axes, "axis1": np.zeros(3)})


def _cartesian_twotime(path):
    r4cli._write_twotime(path, [40.0], dplist=[120.0, 130.0])
    with h5py.File(path, "a") as f:
        del f[K["map_names"]], f[K["map_units"]]
        f[K["map_names"]] = [b"x", b"y"]
        f[K["map_units"]] = [b"px", b"px"]
    return path


def test_r6_17_cli_bins_of_a_cartesian_map_are_identified_by_both_axes(tmp_path):
    path = _cartesian_twotime(tmp_path / "cart.h5")
    xf = XpcsFile(str(path))
    bins = tb._collect_valid_qbins(xf)
    assert [(b[2], b[3]) for b in bins] == [(40.0, 120.0), (40.0, 130.0)]
    out = tmp_path / "out"
    # R7-05: the selector's unit is declared (pixels here), and the names carry units
    assert tb.run_twotime_batch(r4cli._args(path, out, q=40.0, q_unit="px")) == 0
    assert sorted(p.name for p in out.glob("*.png")) == [
        "cart.h5_bin0000_x40.0000px_y120.0000px.png",
        "cart.h5_bin0001_x40.0000px_y130.0000px.png",
    ]


def test_r6_17_cli_keeps_the_q_phi_names(tmp_path):
    path = tmp_path / "scan.h5"
    r4cli._write_twotime(path, [0.02], dplist=[10.0, 20.0])
    out = tmp_path / "out"
    assert tb.run_twotime_batch(r4cli._args(path, out, q=0.02)) == 0
    assert sorted(p.name for p in out.glob("*.png")) == [
        "scan.h5_bin0000_q0.0200_phi10.0.png",
        "scan.h5_bin0001_q0.0200_phi20.0.png",
    ]


def test_r6_17_cli_angle_selection_is_refused_when_the_second_axis_is_not_phi(
    tmp_path,
):
    xf = XpcsFile(str(_cartesian_twotime(tmp_path / "cart.h5")))
    # y is a pixel coordinate, not an angle: a wrap-around angular match is wrong
    assert tb.find_qbins_for_phi(xf, 120.0) == []
    assert tb.find_qbin_for_qphi(xf, 50.0, 120.0) is None


# ===================================================================== C
def _one_element_calibration(path):
    r3._write_nexus(path)
    values = {"bcx": 3.5, "bcy": 3.5, "pixel_size": r3.PIXEL, "det_dist": r3.DIST}
    with h5py.File(path, "r+") as f:
        for name, value in values.items():
            del f[K[name]]
            f[K[name]] = np.array([value])  # one element, rank 1
        del f[K["X_energy"]]
        f[K["X_energy"]] = np.array([[r3.ENERGY]])  # one element, rank 2
    return path


def test_r6_qmap_one_element_calibration_arrays_are_the_scalars_they_hold(tmp_path):
    path = _one_element_calibration(tmp_path / "one.h5")
    qmap = XpcsFile(str(path)).qmap
    assert qmap.is_default is False
    assert qmap.bcx == 3.5 and qmap.bcy == 3.5
    assert qmap.pixel_size == pytest.approx(r3.PIXEL)
    assert qmap.det_dist == pytest.approx(r3.DIST)
    assert qmap.X_energy == pytest.approx(r3.ENERGY)
    assert np.ndim(qmap.k0) == 0 and qmap.k0 == pytest.approx(r3.K0)
    assert qmap.dqmap.shape == (
        r3.H,
        r3.W,
    )  # the file's partition, not a 1024^2 default
    # the same numbers as the file with scalar calibration
    ref = XpcsFile(str(_scalar_twin(tmp_path))).qmap
    np.testing.assert_allclose(qmap.qmap["q"], ref.qmap["q"])


def _scalar_twin(tmp_path):
    path = tmp_path / "scalar.h5"
    r3._write_nexus(path)
    return path


def test_r6_qmap_that_cannot_be_loaded_says_it_is_a_default(tmp_path, caplog):
    path = tmp_path / "bad.h5"
    r3._write_nexus(path)
    with h5py.File(path, "r+") as f:
        del f[K["X_energy"]]
        f[K["X_energy"]] = np.array([8.0, 9.0])  # two energies: not a scalar
    with caplog.at_level(logging.WARNING):
        qmap = QMap(fname=str(path))
    assert qmap.is_default is True
    records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "bad.h5" in r.getMessage() and "default" in r.getMessage().lower()
        for r in records
    )


def test_r6_qmap_without_a_qmap_group_is_flagged_as_a_default(tmp_path, caplog):
    path = tmp_path / "empty.h5"
    with h5py.File(path, "w") as f:
        f["/unrelated"] = 1
    with caplog.at_level(logging.WARNING):
        qmap = QMap(fname=str(path))
    assert qmap.is_default is True
    assert any(
        "empty.h5" in r.getMessage() and r.levelno >= logging.WARNING
        for r in caplog.records
    )


def test_r6_qmap_is_not_default_for_a_normal_file(tmp_path):
    path = tmp_path / "ok.h5"
    r3._write_nexus(path)
    assert XpcsFile(str(path)).qmap.is_default is False
