"""Regression tests for review round 4: R4-06, R4-08, R4-09.

R4-06  the SAXS 1D / ROI export names its files from the source file name, not
       the display label, so two selected files sharing a label (``scan.hdf`` /
       ``scan_results.hdf`` both display as ``scan``) do not overwrite each other.
R4-08  the exported Pie / Ring profile averages exactly the detector pixels
       inside the ROI that is DRAWN (apex, angular extent and radius of the Qt
       shape; centre and radii of the ring circles), binned by the file's Q.
R4-09  a SAXS background measured at other azimuth sectors (or on other map
       axes) is not subtracted row by row.

Every test writes real NeXus-layout HDF5 files with h5py and loads them through
the real ``XpcsFile`` / ``QMap`` (and ``ViewerKernel`` / ``ImageViewDev`` where the
finding concerns them). The R4-08 tests compare against Qt's own hit test.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from qtpy import QtCore, QtWidgets

from tests.unit import test_review_r3_maps_roi as r3
from xpcsviewer.fileIO.hdf_reader import clear_connection_pool
from xpcsviewer.plothandler.pyqtgraph_handler import ImageViewDev, PieROI
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


# ------------------------------------------------------------------ R4-06
def test_r4_06_same_label_files_do_not_overwrite_each_others_saxs_export(
    tmp_path, qapp
):
    a, b = tmp_path / "scan.hdf", tmp_path / "scan_results.hdf"
    r3._write_nexus(a, saxs=np.full((8, 8), 2.0))
    r3._write_nexus(b, saxs=np.full((8, 8), 9.0))
    view = ImageViewDev()
    kernel = ViewerKernel(str(tmp_path))
    kernel.add_target([a.name, b.name], preload=False)
    xfs = kernel.get_xf_list()
    assert len(xfs) == 2
    assert xfs[0].label == xfs[1].label  # the collision the finding relies on
    assert kernel.add_roi(view, sl_type="Pie")

    dest = tmp_path / "out"
    dest.mkdir()
    kernel.export_saxs_1d(view, str(dest))

    assert sorted(p.name for p in dest.iterdir()) == [
        "scan_Pie_000.txt",
        "scan_results_Pie_000.txt",
        "scan_results_saxs1d.txt",
        "scan_saxs1d.txt",
    ]
    for stem, value in (("scan", 2.0), ("scan_results", 9.0)):
        pie = np.loadtxt(dest / f"{stem}_Pie_000.txt")
        finite = pie[np.isfinite(pie[:, 1]), 1]
        assert finite.size and np.allclose(finite, value)
        saxs1d = np.loadtxt(dest / f"{stem}_saxs1d.txt")
        assert np.allclose(saxs1d[:, 1], 1.0)  # file's own saxs_1d, both written


def test_r4_06_same_stem_in_different_folders_gets_a_suffix(tmp_path, qapp):
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
        r3._write_nexus(tmp_path / sub / "scan.hdf", saxs=np.full((8, 8), 3.0))
    xfs = [XpcsFile(str(tmp_path / s / "scan.hdf")) for s in ("a", "b")]
    kernel = ViewerKernel.__new__(ViewerKernel)
    kernel.get_xf_list = lambda: xfs
    view = ImageViewDev()
    dest = tmp_path / "out"
    dest.mkdir()
    ViewerKernel.export_saxs_1d(kernel, view, str(dest))
    assert sorted(p.name for p in dest.iterdir()) == [
        "scan_2_saxs1d.txt",
        "scan_saxs1d.txt",
    ]


def test_r4_06_direct_export_defaults_to_the_source_file_name(tmp_path):
    path = tmp_path / "scan_results.hdf"
    r3._write_nexus(path)
    xf = XpcsFile(str(path))
    assert xf.label != "scan_results"
    xf.export_saxs1d([], str(tmp_path))
    assert (tmp_path / "scan_results_saxs1d.txt").exists()
    xf.export_saxs1d([], str(tmp_path), basename="custom")
    assert (tmp_path / "custom_saxs1d.txt").exists()


# ------------------------------------------------------------------ R4-08
def _inside_drawn(shape_owner, shape, path_of=lambda roi: roi.shape()):
    """Pixels whose (col, row) point lies inside the Qt shape of an ROI."""
    path = path_of(shape_owner)
    return np.array(
        [
            [
                path.contains(
                    shape_owner.mapFromParent(QtCore.QPointF(float(x), float(y)))
                )
                for x in range(shape[1])
            ]
            for y in range(shape[0])
        ]
    )


def _mask(param, shape):
    """The pixel mask the export builds from a ROI descriptor."""
    return XpcsFile._pixels_in_polygon(
        param["polygon"], shape, param["origin"], param["rotation"]
    )


def _expected_pie(xf, inside):
    sqmap, image = xf.qmap.sqmap, xf.saxs_2d
    out = []
    for q in range(len(xf.sqlist)):
        sel = inside & (sqmap == q + 1)
        out.append(image[sel].mean() if sel.any() else np.nan)
    return np.array(out)


def test_r4_08_reviewer_scenario_exports_the_pixels_of_the_drawn_wedge(tmp_path, qapp):
    path = tmp_path / "wedge_0001.h5"
    r3._write_nexus(path)
    xf = XpcsFile(str(path))
    roi = PieROI((r3.BC, r3.BC), 3.0)  # 3 pixel Pie centred on the beam centre
    apex = roi.mapToParent(QtCore.QPointF(0.0, roi.size().y() / 2))
    assert (apex.x(), apex.y()) == (2.0, 3.5)  # the drawn apex is NOT the centre
    inside = _inside_drawn(roi, (8, 8))
    assert inside.sum() == 6

    qlist, got = xf.get_roi_data(roi.get_parameter())
    np.testing.assert_allclose(got, _expected_pie(xf, inside), equal_nan=True)
    assert got[0] == pytest.approx(44.5)  # not the wedge about the beam centre


@pytest.fixture
def big_xf(tmp_path, monkeypatch):
    """A real 24 x 24 file: several q bins and room for rotated, off-centre ROIs."""
    n = 24
    monkeypatch.setattr(r3, "H", n)
    monkeypatch.setattr(r3, "W", n)
    monkeypatch.setattr(r3, "BC", 11.5)
    monkeypatch.setattr(r3, "R_EDGES", (4.0, 8.0))
    monkeypatch.setattr(r3, "R_CENTRES", (3.0, 6.0, 10.0))
    path = tmp_path / "big_0001.h5"
    r3._write_nexus(path, dqmap=np.tile(np.repeat([1, 2], n // 2), (n, 1)))
    return XpcsFile(str(path))


# (centre x, centre y, size, rotation [degree], optional (w, h) resize)
PIE_CASES = [
    (11.5, 11.5, 8.0, 0.0, None),
    (9.3, 12.7, 10.0, 35.0, None),
    (14.2, 8.6, 7.0, -70.0, (7.0, 11.0)),
    (6.1, 15.9, 12.0, 200.0, (12.0, 6.0)),
    (11.5, 11.5, 5.0, 90.0, None),
    (17.7, 5.3, 14.0, 310.0, (14.0, 9.0)),
]


def _make_pie(case):
    cx, cy, size, angle, wh = case
    roi = PieROI((cx, cy), size)
    if wh is not None:
        roi.setSize(wh)
    roi.setAngle(angle)
    return roi


@pytest.mark.parametrize("case", PIE_CASES)
def test_r4_08_export_mask_equals_the_drawn_shape_pixel_for_pixel(big_xf, qapp, case):
    roi = _make_pie(case)
    inside_qt = _inside_drawn(roi, big_xf.saxs_2d.shape)
    assert 0 < inside_qt.sum() < inside_qt.size  # a real, partial selection

    polygon = roi.get_parameter()["polygon"]
    mask = _mask(roi.get_parameter(), big_xf.saxs_2d.shape)
    assert np.array_equal(mask, inside_qt)  # equal, not merely overlapping

    qlist, got = big_xf.get_roi_data(roi.get_parameter())
    expected = _expected_pie(big_xf, inside_qt)
    expected[~(expected > 0)] = np.nan
    np.testing.assert_allclose(got, expected, equal_nan=True)


def test_r4_08_handler_roi_in_a_view_uses_the_parent_frame(big_xf, qapp):
    view = ImageViewDev()
    view.setImage(big_xf.saxs_2d)
    label = view.add_roi(cen=(10.3, 13.1), radius=9.0, sl_type="Pie")
    roi = view.roi_record[label]
    roi.setAngle(25.0)
    (param,) = view.get_roi_list()
    inside_qt = _inside_drawn(roi, big_xf.saxs_2d.shape)
    assert 0 < inside_qt.sum() < inside_qt.size
    mask = _mask(param, big_xf.saxs_2d.shape)
    assert np.array_equal(mask, inside_qt)


def test_r4_08_resize_without_repaint_does_not_use_a_stale_angle(big_xf, qapp):
    roi = PieROI((11.5, 11.5), 8.0)
    roi.shape()  # last hit test used the square
    roi.setSize((8.0, 16.0))  # wider-than-high ratio changes; no shape() call
    param = roi.get_parameter()
    half = np.degrees(np.arctan2(16.0 / 2.0, 8.0))  # drawn half-angle, w=8 h=16
    lo, hi = param["angle_range"]
    assert (hi - lo) % 360.0 == pytest.approx(2 * half)
    inside_qt = _inside_drawn(roi, big_xf.saxs_2d.shape)
    assert np.array_equal(_mask(param, big_xf.saxs_2d.shape), inside_qt)


@pytest.mark.parametrize("angle", [0.0, 90.0, 180.0, 270.0])
def test_r4_08_lattice_aligned_rois_match_qt_on_their_edges(big_xf, qapp, angle):
    # integer-aligned ROIs put detector pixels exactly on the polygon edges
    roi = PieROI((12.0, 12.0), 6.0)
    roi.setAngle(angle)
    inside_qt = _inside_drawn(roi, big_xf.saxs_2d.shape)
    assert np.array_equal(_mask(roi.get_parameter(), big_xf.saxs_2d.shape), inside_qt)


def test_r4_08_descriptor_without_polygon_keeps_the_wedge_about_the_beam_centre(
    big_xf,
):
    roi = {"sl_type": "Pie", "angle_range": (0.0, 90.0), "dist": 6.0}
    qlist, y = big_xf.get_roi_data(roi)
    assert np.isfinite(y[0]) and np.isnan(y[2])  # cut at dist, as before


def test_r4_08_ring_uses_the_drawn_centre_and_radii(big_xf, qapp):
    cx, cy, r_in, r_out = 10.3, 13.1, 4.0, 9.0
    view = ImageViewDev()
    view.setImage(big_xf.saxs_2d)
    view.add_roi(cen=(cx, cy), radius=r_out, sl_type="Circle", label="RingA")
    view.add_roi(cen=(cx, cy), radius=r_in, sl_type="Circle", label="RingB")
    (param,) = view.get_roi_list()
    assert param["sl_type"] == "Ring"
    assert param["center"] == pytest.approx((cx, cy))
    assert sorted(param["radius"]) == pytest.approx([r_in, r_out])

    # The circles are PAINTED as true ellipses (EllipseROI.paint -> drawEllipse of
    # the bounding rect), so the oracle is the geometry of the drawn items.
    shape = big_xf.saxs_2d.shape
    rows, cols = np.indices(shape)

    def disc(roi):
        rect = roi.boundingRect()
        c = roi.mapToParent(rect.center())
        return np.hypot(cols - c.x(), rows - c.y()) < rect.width() / 2.0

    annulus = disc(view.roi_record["RingA"]) & ~disc(view.roi_record["RingB"])
    assert annulus.any() and not annulus.all()

    phi = np.mod(-np.degrees(np.arctan2(rows - cy, cols - cx)), 360.0)
    lo, hi = phi[annulus].min(), phi[annulus].max()
    x, y = big_xf.get_roi_data(param, phi_num=6)
    idx = np.clip(((phi - lo) / ((hi - lo) / 6)).astype(int), 0, 5)
    expected = [big_xf.saxs_2d[annulus & (idx == i)].mean() for i in range(6)]
    np.testing.assert_allclose(x, np.linspace(lo, hi, 6))
    np.testing.assert_allclose(y, expected)


# ------------------------------------------------------------------ R4-09
def _saxs_pair(tmp_path, bkg_phi, bkg_values=(13.0, 15.0, 12.0, 25.0)):
    sample_path, bkg_path = tmp_path / "sample.h5", tmp_path / "bkg_scan.h5"
    r3._write_nexus(sample_path, azimuthal=True)
    r3._write_nexus(bkg_path, azimuthal=True)
    with h5py.File(sample_path, "a") as h:
        h[K["saxs_1d"]][:] = [[30.0, 25.0, 20.0, 40.0]]
    with h5py.File(bkg_path, "a") as h:
        h[K["saxs_1d"]][:] = [list(bkg_values)]
        h[K["splist"]][:] = bkg_phi
    return XpcsFile(str(sample_path)), XpcsFile(str(bkg_path))


def test_r4_09_background_at_other_azimuths_is_not_subtracted(tmp_path, caplog):
    sample, bkg = _saxs_pair(tmp_path, [90.0, 270.0])
    _, raw, *_ = sample.get_saxs1d_data()
    assert raw.shape == (3, 3)  # average + two sectors; same Q grid as bkg
    with caplog.at_level("WARNING"):
        _, got, *_ = sample.get_saxs1d_data(bkg_xf=bkg)
    np.testing.assert_allclose(got, raw, equal_nan=True)  # sample trace kept
    assert "azimuth" in caplog.text and "[90.0, 270.0]" in caplog.text
    assert bkg.label in caplog.text


def test_r4_09_background_at_the_same_azimuths_is_still_subtracted(tmp_path):
    sample, bkg = _saxs_pair(tmp_path, [0.0, 180.0])
    _, raw, *_ = sample.get_saxs1d_data()
    _, bkg_raw, *_ = bkg.get_saxs1d_data()
    _, got, *_ = sample.get_saxs1d_data(bkg_xf=bkg, bkg_weight=0.5)
    expected = raw - 0.5 * bkg_raw
    expected[expected < 0] = np.nan
    np.testing.assert_allclose(got, expected, equal_nan=True)
    assert np.isfinite(got).any()


def test_r4_09_single_curve_background_still_broadcasts(tmp_path):
    sample, _ = _saxs_pair(tmp_path, [0.0, 180.0])
    flat = tmp_path / "flat.h5"
    r3._write_nexus(flat)  # one azimuth bin: one background curve, 3 q bins
    bkg = XpcsFile(str(flat))
    _, raw, *_ = sample.get_saxs1d_data()
    _, bkg_raw, *_ = bkg.get_saxs1d_data()
    assert bkg_raw.shape[0] == 1
    _, got, *_ = sample.get_saxs1d_data(bkg_xf=bkg)
    expected = raw - bkg_raw
    expected[expected < 0] = np.nan
    np.testing.assert_allclose(got, expected, equal_nan=True)


def test_r4_09_background_on_other_map_axes_is_not_subtracted(tmp_path, caplog):
    sample, _ = _saxs_pair(tmp_path, [0.0, 180.0])
    other = tmp_path / "pixel_axes.h5"
    r3._write_nexus(other, azimuthal=True)
    with h5py.File(other, "a") as h:
        h[K["saxs_1d"]][:] = [[13.0, 15.0, 12.0, 25.0]]
        h[K["map_names"]][:] = [b"x", b"y"]
        h[K["map_units"]][:] = [b"px", b"px"]
    bkg = XpcsFile(str(other))
    _, raw, *_ = sample.get_saxs1d_data()
    with caplog.at_level("WARNING"):
        _, got, *_ = sample.get_saxs1d_data(bkg_xf=bkg)
    np.testing.assert_allclose(got, raw, equal_nan=True)
    assert "axis" in caplog.text
