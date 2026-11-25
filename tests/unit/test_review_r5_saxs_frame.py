"""Regression tests for review round 5: R5-06, R5-07, R5-12 and the xpcs_file
part of R5-01.

R5-12  the exported Pie / Ring ROI is exactly the drawn region, tested at the
       pixel CENTRES in the frame the ROI was drawn in. The ImageItem draws pixel
       (row, col) over [col, col+1] x [row, row+1] (centre at +0.5); the oracle
       here maps each pixel centre through the real ImageItem into the ROI's frame
       and asks Qt's own shape / ellipse geometry, so it does not depend on any
       frame assumption of the exporter.
R5-07  the two circles of a Phi-Ring are one removable unit; the ROI collector
       survives an incomplete pair.
R5-06  SAXS plot axis label and export header carry the file's own static-axis
       name and unit.
R5-01  ``XpcsFile.export_saxs1d`` writes through an ``ExportGuard``.

Real h5py-written NeXus files go through the real ``XpcsFile`` / ``QMap``; the
view, image item and ROIs are the real offscreen pyqtgraph objects.
"""

from __future__ import annotations

import logging
import os

import h5py
import numpy as np
import pytest
from qtpy import QtCore, QtWidgets

from tests.unit import test_review_r3_maps_roi as r3
from tests.unit.test_review_r4_saxs_roi import (
    PIE_CASES,
    _centres_in_roi_frame,
    _inside_drawn,
    _make_pie,
    _mask,
    _view_with,
    big_xf,  # noqa: F401  (fixture)
)
from xpcsviewer.fileIO.hdf_reader import clear_connection_pool
from xpcsviewer.utils.export_guard import ExportCollisionError, ExportGuard
from xpcsviewer.viewer_kernel import ViewerKernel
from xpcsviewer.xpcs_file import XpcsFile

pytestmark = pytest.mark.unit

K = r3.K


@pytest.fixture(autouse=True)
def _fresh_pool():
    clear_connection_pool()
    yield
    clear_connection_pool()


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# ------------------------------------------------------------------ R5-12
def _expected_pie(xf, inside):
    out = []
    for q in range(len(xf.sqlist)):
        sel = inside & (xf.qmap.sqmap == q + 1)
        out.append(xf.saxs_2d[sel].mean() if sel.any() else np.nan)
    out = np.array(out)
    out[~(out > 0)] = np.nan
    return out


def test_r5_12_reviewer_wedge_selects_the_painted_pixels(tmp_path, qapp):
    """The reviewer's 8x8 file and 3 pixel Q-Wedge: 5 pixels, mean 38.0."""
    path = tmp_path / "wedge_0001.h5"
    r3._write_nexus(path)
    xf = XpcsFile(str(path))
    view = _view_with(xf)
    item = view.getImageItem()
    assert item.transform().isIdentity()  # the frame the finding relied on
    roi = view.roi_record[
        view.add_roi(cen=(r3.BC, r3.BC), radius=3.0, sl_type="Q-Wedge")
    ]
    inside = _inside_drawn(view, roi, xf.saxs_2d.shape)
    # the integer-point test of round 4 selects a different set: the bug
    corner = np.array(
        [
            [
                roi.shape().contains(roi.mapFromItem(item, QtCore.QPointF(c, r)))
                for c in range(8)
            ]
            for r in range(8)
        ]
    )
    assert not np.array_equal(inside, corner)

    (param,) = view.get_roi_list()
    assert np.array_equal(_mask(param, xf.saxs_2d.shape), inside)
    qlist, got = xf.get_roi_data(param)
    assert got[0] == pytest.approx(38.0)
    np.testing.assert_allclose(got, _expected_pie(xf, inside), equal_nan=True)


def _transform_view(xf, pos, scale):
    """A view whose ImageItem is shifted and scaled (a non-identity frame)."""
    view = _view_with(xf)
    item = view.getImageItem()
    item.setScale(scale)
    item.setPos(*pos)
    return view


@pytest.mark.parametrize(
    "pos,scale", [((0.0, 0.0), 1.0), ((2.5, -1.25), 1.0), ((3.0, 4.0), 0.5)]
)
@pytest.mark.parametrize("case", PIE_CASES)
def test_r5_12_export_follows_the_image_item_transform(big_xf, qapp, case, pos, scale):  # noqa: F811
    """No frame constant in the exporter: the descriptor carries the item's frame."""
    view = _transform_view(big_xf, pos, scale)
    # draw the ROI over the (moved) image so a real, partial selection results
    cx, cy, size, angle, wh = case
    cx, cy = pos[0] + cx * scale, pos[1] + cy * scale
    roi = view.roi_record[
        view.add_roi(cen=(cx, cy), radius=size * scale, sl_type="Pie")
    ]
    if wh is not None:
        roi.setSize((wh[0] * scale, wh[1] * scale))
    roi.setAngle(angle)
    inside = _inside_drawn(view, roi, big_xf.saxs_2d.shape)
    assert 0 < inside.sum() < inside.size

    (param,) = view.get_roi_list()
    assert np.array_equal(_mask(param, big_xf.saxs_2d.shape), inside)
    qlist, got = big_xf.get_roi_data(param)
    np.testing.assert_allclose(got, _expected_pie(big_xf, inside), equal_nan=True)


def _ring_oracle(view, shape, cx, cy):
    """Annulus mask and azimuth map of the drawn circles, through the image item."""
    item = view.getImageItem()
    ring_a, ring_b = view.roi_record["RingA"], view.roi_record["RingB"]

    def disc(roi):
        rect = roi.boundingRect()
        c = rect.center()
        return np.array(
            [
                [
                    np.hypot(p.x() - c.x(), p.y() - c.y()) < rect.width() / 2.0
                    for p in row
                ]
                for row in _centres_in_roi_frame(view, roi, shape)
            ]
        )

    annulus = disc(ring_a) & ~disc(ring_b)
    parent = ring_a.parentItem()
    xy = np.array(
        [
            [
                (q.x(), q.y())
                for q in (
                    parent.mapFromItem(item, QtCore.QPointF(c + 0.5, r + 0.5))
                    for c in range(shape[1])
                )
            ]
            for r in range(shape[0])
        ]
    )
    phi = np.mod(-np.degrees(np.arctan2(xy[..., 1] - cy, xy[..., 0] - cx)), 360.0)
    return annulus, phi


@pytest.mark.parametrize(
    "pos,scale", [((0.0, 0.0), 1.0), ((2.5, -1.25), 1.0), ((3.0, 4.0), 0.5)]
)
def test_r5_12_ring_selects_the_painted_pixel_centres(big_xf, qapp, pos, scale):  # noqa: F811
    cx, cy = pos[0] + 10.3 * scale, pos[1] + 13.1 * scale
    view = _transform_view(big_xf, pos, scale)
    view.add_roi(cen=(cx, cy), radius=9.0 * scale, sl_type="Phi-Ring", label="RingA")
    view.add_roi(cen=(cx, cy), radius=4.0 * scale, sl_type="Phi-Ring", label="RingB")
    shape = big_xf.saxs_2d.shape
    annulus, phi = _ring_oracle(view, shape, cx, cy)
    assert annulus.any() and not annulus.all()

    (param,) = view.get_roi_list()
    lo, hi = phi[annulus].min(), phi[annulus].max()
    x, y = big_xf.get_roi_data(param, phi_num=6)
    idx = np.clip(((phi - lo) / ((hi - lo) / 6)).astype(int), 0, 5)
    expected = [big_xf.saxs_2d[annulus & (idx == i)].mean() for i in range(6)]
    np.testing.assert_allclose(x, np.linspace(lo, hi, 6))
    np.testing.assert_allclose(y, expected)


def test_r5_12_descriptor_without_a_frame_means_the_unmodified_image_item(big_xf, qapp):  # noqa: F811
    """Hand-built descriptors default to the display frame of a plain ImageItem."""
    view = _view_with(big_xf)
    roi = _make_pie(view, PIE_CASES[1])
    (param,) = view.get_roi_list()
    bare = {k: v for k, v in param.items() if k != "pixel_frame"}
    assert "pixel_frame" in param
    _, with_frame = big_xf.get_roi_data(param)
    _, without = big_xf.get_roi_data(bare)
    np.testing.assert_allclose(without, with_frame, equal_nan=True)
    assert np.array_equal(
        _mask(bare, big_xf.saxs_2d.shape),
        _inside_drawn(view, roi, big_xf.saxs_2d.shape),
    )


# ------------------------------------------------------------------ R5-07
def _remove_via_menu(roi):
    actions = [a for a in roi.getMenu().actions() if "remove" in a.text().lower()]
    assert len(actions) == 1
    actions[0].trigger()
    QtWidgets.QApplication.processEvents()  # pyqtgraph defers the removal signal


@pytest.mark.parametrize("sl_type", ["Phi-Ring", "Circle", "Pie", "Q-Wedge"])
@pytest.mark.parametrize("part", [0, 1])
def test_r5_07_removing_any_part_of_a_roi_leaves_no_orphan(
    tmp_path, qapp, sl_type, part
):
    path = tmp_path / "ring_0001.h5"
    r3._write_nexus(path)
    kernel = ViewerKernel(str(tmp_path))
    kernel.add_target([path.name], preload=False)
    view = _view_with(kernel.get_xf_list()[0])
    assert kernel.add_roi(view, sl_type=sl_type)
    parts = [k for k in view.roi_record if k != "Center"]
    assert parts
    _remove_via_menu(view.roi_record[parts[min(part, len(parts) - 1)]])

    left = set(view.roi_record) - {"Center"}
    assert left == set()  # a ring (or any multi-part ROI) leaves nothing behind
    assert view.get_roi_list() == []
    out = tmp_path / "out"
    out.mkdir()
    kernel.export_saxs_1d(view, str(out))  # KeyError('RingA') before the fix
    assert [p.name for p in out.iterdir()] == ["ring_0001_saxs1d.txt"]


def test_r5_07_ring_survives_a_second_add_and_exports(tmp_path, qapp):
    path = tmp_path / "ring_0001.h5"
    r3._write_nexus(path)
    kernel = ViewerKernel(str(tmp_path))
    kernel.add_target([path.name], preload=False)
    view = _view_with(kernel.get_xf_list()[0])
    assert kernel.add_roi(view, sl_type="Phi-Ring")
    assert kernel.add_roi(view, sl_type="Phi-Ring")  # replaces, not duplicates
    assert sorted(set(view.roi_record) - {"Center"}) == ["RingA", "RingB"]
    assert [p["sl_type"] for p in view.get_roi_list()] == ["Ring"]


@pytest.mark.parametrize("orphan", ["RingA", "RingB"])
def test_r5_07_collector_skips_an_incomplete_pair_with_a_warning(qapp, caplog, orphan):
    view = ImageViewDev_with_image()
    view.add_roi(cen=(5.0, 5.0), radius=3.0, sl_type="Circle", label=orphan)
    assert set(view.roi_record) == {orphan}
    with caplog.at_level(logging.WARNING):
        assert view.get_roi_list() == []
    assert orphan in caplog.text and "incomplete" in caplog.text.lower()


def ImageViewDev_with_image():
    from xpcsviewer.plothandler.pyqtgraph_handler import ImageViewDev

    view = ImageViewDev()
    view.setImage(np.ones((16, 16)))
    return view


# ------------------------------------------------------------------ R5-06
def _write_units(path, unit=b"1/nm", name=b"q"):
    r3._write_nexus(path)
    with h5py.File(path, "a") as f:
        for key, value in (("map_units", unit), ("map_names", name)):
            del f[K[key]]  # recreate: the written dataset's string width is fixed
            f[K[key]] = np.array([value, b"deg" if key == "map_units" else b"phi"])


def test_r5_06_reviewer_scenario_nm_file_is_not_labelled_angstrom(tmp_path):
    path = tmp_path / "nm_0001.h5"
    _write_units(path)
    xf = XpcsFile(str(path))
    assert xf.qmap.map_units[0] == "nm⁻¹"
    for norm in (None, "q2", "q4", "I0"):
        _, _, xlabel, _ = xf.get_saxs1d_data(norm_method=norm)
        assert xlabel == "q (nm⁻¹)"
    # a Pie ROI as the viewer draws it (the legacy no-polygon descriptor is
    # refused for a static axis that is not inverse angstrom)
    drawn_pie = {
        "sl_type": "Pie",
        "polygon": [(0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0)],
        "origin": (0.0, 0.0),
        "rotation": 0.0,
    }
    xf.export_saxs1d([drawn_pie], str(tmp_path))
    header = (tmp_path / "nm_0001_saxs1d.txt").read_text().splitlines()[0]
    assert header.startswith("# q(1/nm) Intensity")
    pie = (tmp_path / "nm_0001_Pie_000.txt").read_text().splitlines()[0]
    assert pie.startswith("# q(1/nm) Intensity")
    first = (tmp_path / "nm_0001_saxs1d.txt").read_text().splitlines()[1].split()[0]
    assert float(first) == pytest.approx(xf.saxs_1d["q"][0])  # numbers unchanged


def test_r5_06_default_angstrom_file_keeps_its_labels(tmp_path):
    path = tmp_path / "a_0001.h5"
    r3._write_nexus(path)
    xf = XpcsFile(str(path))
    assert xf.get_saxs1d_data()[2] == "q (Å⁻¹)"
    xf.export_saxs1d([], str(tmp_path))
    assert (
        (tmp_path / "a_0001_saxs1d.txt")
        .read_text()
        .startswith("# q(1/Angstrom) Intensity")
    )


def test_r5_06_axis_name_and_unit_come_from_the_file(tmp_path):
    path = tmp_path / "px_0001.h5"
    _write_units(path, unit=b"px", name=b"x")
    xf = XpcsFile(str(path))
    assert xf.get_saxs1d_data()[2] == "x (px)"
    xf.export_saxs1d([], str(tmp_path))
    assert (tmp_path / "px_0001_saxs1d.txt").read_text().startswith("# x(px) Intensity")


# ------------------------------------------------------------------ R5-01
def test_r5_01_export_refuses_a_destination_that_is_its_own_input(tmp_path):
    path = tmp_path / "scan_0001.h5"
    r3._write_nexus(path)
    before = path.read_bytes()
    xf = XpcsFile(str(path))
    out = tmp_path / "out"
    out.mkdir()
    os.symlink(path, out / "scan_0001_saxs1d.txt")
    with pytest.raises(ExportCollisionError):
        xf.export_saxs1d([], str(out))
    assert path.read_bytes() == before  # the input was never written through


def test_r5_01_shared_guard_refuses_an_alias_between_two_exports(tmp_path):
    a, b = tmp_path / "a_0001.h5", tmp_path / "b_0001.h5"
    r3._write_nexus(a, saxs=np.full((8, 8), 2.0))
    r3._write_nexus(b, saxs=np.full((8, 8), 9.0))
    xa, xb = XpcsFile(str(a)), XpcsFile(str(b))
    out = tmp_path / "out"
    out.mkdir()
    (out / "target.txt").write_text("keep")
    os.symlink(out / "target.txt", out / "a_0001_saxs1d.txt")
    os.symlink(out / "target.txt", out / "b_0001_saxs1d.txt")
    guard = ExportGuard([a, b])
    xa.export_saxs1d([], str(out), guard=guard)
    with pytest.raises(ExportCollisionError):
        xb.export_saxs1d([], str(out), guard=guard)


def test_r5_01_roi_files_are_claimed_before_they_are_written(tmp_path):
    path = tmp_path / "scan_0001.h5"
    r3._write_nexus(path)
    xf = XpcsFile(str(path))
    out = tmp_path / "out"
    out.mkdir()
    guard = ExportGuard([path])
    xf.export_saxs1d(
        [{"sl_type": "Pie", "angle_range": (0.0, 360.0), "dist": 50.0}],
        str(out),
        guard=guard,
    )
    assert sorted(p.name for p in out.iterdir()) == [
        "scan_0001_Pie_000.txt",
        "scan_0001_saxs1d.txt",
    ]
    with pytest.raises(ExportCollisionError):  # same destinations again, same run
        xf.export_saxs1d([], str(out), guard=guard)
