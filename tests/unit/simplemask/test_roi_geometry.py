"""ROIs are rasterised through their own transform (rotation and scale).

Expected masks come from the upstream-ported vertex rasteriser; the tests drive
only ``SimpleMaskKernel.apply_drawing`` so a wrong geometry fails on pixels.
"""

import gc
import math

import numpy as np
import pyqtgraph as pg
import pytest

from xpcsviewer.simplemask.pyqtgraph_mod import ImageViewROI, LineROI
from xpcsviewer.simplemask.rasterize import (
    RoiPolygon,
    circle_vertices,
    ellipse_vertices,
    line_vertices,
    rasterize,
)
from xpcsviewer.simplemask.simplemask_kernel import SimpleMaskKernel
from xpcsviewer.simplemask.simplemask_window import SimpleMaskWindow

pg.setConfigOption("imageAxisOrder", "row-major")

SHAPE = (24, 28)


def excluded(mask):
    return {(int(r), int(c)) for r, c in zip(*np.nonzero(~mask), strict=True)}


def expected(vertices, mode="exclusive"):
    return rasterize(SHAPE, [RoiPolygon(np.asarray(vertices, float), mode)])


@pytest.fixture(scope="module")
def _kernel(qapp):
    # One view for the module: collecting a per-test ImageViewROI together with
    # PolyLineROI cycles segfaulted PySide6 inside pytest's teardown GC.
    image = np.ones(SHAPE)
    view = ImageViewROI()
    kernel = SimpleMaskKernel(pg_hdl=view)
    kernel.read_data(image, {"bcx": 14.0, "bcy": 12.0})
    view.setImage(image)
    yield kernel
    view.close()


@pytest.fixture
def tool(_kernel):
    _kernel.hdl.remove_rois()
    yield _kernel
    _kernel.hdl.remove_rois()
    gc.collect()  # collect removed ROIs while their view is still alive


def draw(tool, roi, mode="exclusive"):
    roi.sl_mode = mode
    tool.hdl.add_item(roi)
    return tool.apply_drawing()


def test_rectangle_roi(tool):
    mask = draw(tool, pg.RectROI((7, 5), (10, 8)))
    assert excluded(mask) == {(r, c) for r in range(5, 13) for c in range(7, 17)}
    assert len(excluded(mask)) == 80


def test_vertical_line_is_not_a_horizontal_bar(tool):
    mask = draw(tool, LineROI((10, 3), (10, 18), 4.0))
    assert excluded(mask) == {(r, c) for r in range(3, 18) for c in range(8, 12)}
    assert len(excluded(mask)) == 60


def test_angled_line_matches_upstream_geometry(tool):
    mask = draw(tool, LineROI((5, 4), (17, 14), 3.0))
    want = expected(line_vertices((3.5, 4.5), (13.5, 16.5), 3.0))
    assert excluded(mask) == excluded(want)
    assert (8, 10) in excluded(mask)
    assert (2, 10) not in excluded(mask)
    assert (20, 25) not in excluded(mask)


def test_circle_and_ellipse_match_upstream(tool):
    mask = draw(tool, pg.CircleROI((8, 6), radius=4.0))
    assert excluded(mask) == excluded(expected(circle_vertices((9.5, 11.5), 4.0)))
    mask = draw(tool, pg.EllipseROI((6, 7), (12, 6)))
    want = expected(ellipse_vertices((9.5, 11.5), (3.0, 6.0)))
    assert excluded(mask) == excluded(want)


def test_polygon_roi(tool):
    pts = [(5, 4), (14, 6), (10, 14)]
    mask = draw(tool, pg.PolyLineROI(pts, closed=True))
    want = expected([(y - 0.5, x - 0.5) for x, y in pts])
    assert excluded(mask) == excluded(want)


def test_rotated_rectangle_roi(tool):
    roi = pg.RectROI((7, 5), (10, 8))
    roi.setAngle(30)
    cos, sin = math.cos(math.radians(30)), math.sin(math.radians(30))
    corners = []
    for lx, ly in ((0, 0), (10, 0), (10, 8), (0, 8)):
        x = 7 + lx * cos - ly * sin
        y = 5 + lx * sin + ly * cos
        corners.append((y - 0.5, x - 0.5))
    mask = draw(tool, roi)
    assert excluded(mask) == excluded(expected(corners))


def test_rotated_ellipse_matches_upstream(tool):
    roi = pg.EllipseROI((6, 7), (12, 6))
    roi.setAngle(40, center=[0.5, 0.5])
    mask = draw(tool, roi)
    # pyqtgraph rotates (x, y); upstream rotates (row, col): the sign flips.
    want = expected(ellipse_vertices((9.5, 11.5), (3.0, 6.0), angle_deg=-40.0))
    assert excluded(mask) == excluded(want)


def test_inclusive_only_keeps_roi_region(tool):
    mask = draw(tool, pg.RectROI((7, 5), (10, 8)), "inclusive")
    assert mask.sum() == 80


def test_inclusive_plus_exclusive_overlap(tool):
    inc = pg.RectROI((7, 5), (10, 8))
    exc = pg.CircleROI((8, 6), radius=4.0)
    inc.sl_mode, exc.sl_mode = "inclusive", "exclusive"
    tool.hdl.add_item(inc)
    tool.hdl.add_item(exc)
    mask = tool.apply_drawing()
    want = rasterize(
        SHAPE,
        [
            RoiPolygon(
                np.array([(4.5, 6.5), (4.5, 16.5), (12.5, 16.5), (12.5, 6.5)]),
                "inclusive",
            ),
            RoiPolygon(circle_vertices((9.5, 11.5), 4.0), "exclusive"),
        ],
    )
    assert np.array_equal(mask, want)
    assert 0 < mask.sum() < 80


@pytest.mark.parametrize("mode", ["exclusive", "inclusive"])
def test_roi_outside_image_is_ignored_in_both_modes(tool, mode):
    mask = draw(tool, pg.CircleROI((-34, -34), radius=4.0), mode)
    assert mask.shape == SHAPE
    assert np.all(mask)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LineROI((10, 10), (10, 10), 4.0),
        lambda: pg.RectROI((7, 5), (0, 8)),
        lambda: pg.PolyLineROI([(5, 4)], closed=True),
    ],
    ids=["zero-length-line", "zero-width-rect", "one-point-polyline"],
)
def test_degenerate_shapes_are_ignored(tool, factory):
    assert np.all(draw(tool, factory()))


def test_partly_clipped_shape(tool):
    mask = draw(tool, pg.RectROI((-5, -4), (10, 8)))
    want = np.ones(SHAPE, dtype=bool)
    want[0:4, 0:5] = False
    assert np.array_equal(mask, want)


def test_self_crossing_polyline_is_drawn(tool):
    pts = [(4, 4), (14, 14), (14, 4), (4, 14)]
    mask = draw(tool, pg.PolyLineROI(pts, closed=True))
    want = expected([(y - 0.5, x - 0.5) for x, y in pts])
    assert excluded(want)
    assert excluded(mask) == excluded(want)


def test_roi_without_sl_mode_is_ignored(tool):
    roi = pg.RectROI((7, 5), (10, 8))
    tool.hdl.add_item(roi)
    assert np.all(tool.apply_drawing())


def test_apply_drawing_without_rois_is_all_true(tool):
    mask = tool.apply_drawing()
    assert mask.dtype == bool
    assert mask.shape == SHAPE
    assert np.all(mask)


def test_pending_roi_count_counts_only_drawable_shapes(tool):
    for roi in (
        pg.RectROI((7, 5), (10, 8)),
        pg.CircleROI((-34, -34), radius=4.0),
        LineROI((10, 10), (10, 10), 4.0),
    ):
        roi.sl_mode = "exclusive"
        tool.hdl.add_item(roi)
    assert tool.pending_roi_count() == 1
    assert len(tool.hdl.roi_items) == 3  # counting does not consume the ROIs


@pytest.fixture
def window(qtbot):
    win = SimpleMaskWindow()
    qtbot.addWidget(win)
    win.load_from_viewer(
        np.ones(SHAPE),
        {
            "bcx": 14.0,
            "bcy": 12.0,
            "det_dist": 5000.0,
            "pix_dim": 0.075,
            "energy": 10.0,
        },
    )
    return win


def test_apply_with_no_pending_drawings_keeps_previous_layer(window):
    roi = pg.RectROI((7, 5), (10, 8))
    roi.sl_mode = "exclusive"
    window.kernel.hdl.add_item(roi)
    window.action_apply.trigger()
    first = excluded(window.kernel.mask)
    assert len(first) == 80

    window.action_apply.trigger()  # (a) nothing pending
    assert excluded(window.kernel.mask) == first
    assert "No drawings to apply" in window.status_bar.currentMessage()

    window.status_bar.showMessage("")
    for roi in (pg.CircleROI((-34, -34), radius=4.0), LineROI((10, 10), (10, 10), 4)):
        roi.sl_mode = "exclusive"
        window.kernel.hdl.add_item(roi)
    window.action_apply.trigger()  # (b) only undrawable shapes pending
    assert excluded(window.kernel.mask) == first
    assert "No drawings to apply" in window.status_bar.currentMessage()
