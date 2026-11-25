"""Parity and behaviour tests for the ported pySimpleMask vertex rasteriser."""

import hashlib
import json

import numpy as np
import pytest

from xpcsviewer.simplemask.rasterize import (
    RoiPolygon,
    circle_vertices,
    ellipse_vertices,
    group_index_map,
    group_masks,
    line_vertices,
    rasterize,
    rectangle_vertices,
)

SHAPE = (24, 28)

# Goldens generated with upstream pySimpleMask core/rasterize.py (0.4.7 wheel);
# each is the sorted np.argwhere(~mask).
POLYGON_REMOVED = [
    [4, 5], [5, 6], [5, 7], [5, 8], [5, 9], [6, 6], [6, 7], [6, 8], [6, 9],
    [6, 10], [6, 11], [6, 12], [6, 13], [6, 14], [7, 7], [7, 8], [7, 9],
    [7, 10], [7, 11], [7, 12], [7, 13], [8, 7], [8, 8], [8, 9], [8, 10],
    [8, 11], [8, 12], [8, 13], [9, 8], [9, 9], [9, 10], [9, 11], [9, 12],
    [10, 8], [10, 9], [10, 10], [10, 11], [10, 12], [11, 9], [11, 10],
    [11, 11], [12, 9], [12, 10], [12, 11], [13, 10], [14, 10],
]  # fmt: skip

# (removed-pixel count, sha256 of json.dumps(argwhere(~mask).tolist()))
GOLDENS = {
    "circle": (
        49,
        "a67ec7f0c3f8b0b308938f4449b310eecfae08df0d012ec1e9a87a6e9833045f",
    ),
    "ellipse": (
        55,
        "e1bcf597988ce33ae7506dae850d2a630d7fd6a4c329076afb9f8f92d26f6e4e",
    ),
    "rectangle": (
        99,
        "0f774d3935e2671b4b620e1efd49f06f5442090a1bf9571bfc01303790b7e934",
    ),
    "line": (
        47,
        "b2d72fe13938daa46ac70313190c0b1351c24e2d9b159fc6c2cf2327c7c05ead",
    ),
    "inclusive_rect_exclusive_circle": (
        621,
        "5281c4b202cea6c1c51f31012f8587112e434c9855fae009c8c23fa5ab6b2c38",
    ),
}


def _removed(mask):
    return np.argwhere(~mask).tolist()


def _assert_golden(name, mask):
    removed = _removed(mask)
    count, digest = GOLDENS[name]
    assert len(removed) == count
    assert hashlib.sha256(json.dumps(removed).encode()).hexdigest() == digest


def test_matches_upstream_polygon():
    roi = RoiPolygon(np.array([(4, 5), (6, 14), (14, 10)], dtype=float))
    assert _removed(rasterize(SHAPE, [roi])) == POLYGON_REMOVED


def test_matches_upstream_circle():
    roi = RoiPolygon(circle_vertices((10, 12), 4.0))
    _assert_golden("circle", rasterize(SHAPE, [roi]))


def test_matches_upstream_ellipse():
    roi = RoiPolygon(ellipse_vertices((10, 12), (3.0, 6.0)))
    _assert_golden("ellipse", rasterize(SHAPE, [roi]))


def test_matches_upstream_rectangle():
    roi = RoiPolygon(rectangle_vertices((9, 12), (8, 10)))
    _assert_golden("rectangle", rasterize(SHAPE, [roi]))


def test_matches_upstream_line():
    # Endpoints off the pixel lattice: with integer endpoints two pixel centres
    # sit exactly on the end caps and flip with last-bit libm differences.
    roi = RoiPolygon(line_vertices((4.4, 5.3), (14.1, 17.4), 3.0))
    _assert_golden("line", rasterize(SHAPE, [roi]))


def test_matches_upstream_inclusive_rectangle_exclusive_circle():
    rois = [
        RoiPolygon(rectangle_vertices((9, 12), (8, 10)), "inclusive"),
        RoiPolygon(circle_vertices((10, 12), 4.0)),
    ]
    _assert_golden("inclusive_rect_exclusive_circle", rasterize(SHAPE, rois))


def test_no_inclusive_keeps_everything_else():
    roi = RoiPolygon(circle_vertices((10, 12), 4.0))
    mask = rasterize(SHAPE, [roi])
    assert mask.dtype == bool
    assert mask.sum() == mask.size - 49
    assert rasterize(SHAPE, []).all()


def test_inclusive_limits_keep_region():
    roi = RoiPolygon(rectangle_vertices((9, 12), (8, 10)), "inclusive")
    mask = rasterize(SHAPE, [roi])
    assert mask.sum() == 99
    assert not mask[0, 0]
    assert mask[9, 12]


def test_group_index_map_last_write_wins():
    a = RoiPolygon(rectangle_vertices((9, 12), (8, 10)), "inclusive", 1)
    b = RoiPolygon(rectangle_vertices((9, 14), (8, 10)), "inclusive", 2)
    ignored = [
        RoiPolygon(circle_vertices((20, 5), 2.0), "inclusive", 0),
        RoiPolygon(circle_vertices((20, 20), 2.0), "exclusive", 3),
    ]
    gmap = group_index_map(SHAPE, [a, b, *ignored])
    assert gmap.dtype == np.uint32
    assert gmap[9, 8] == 1  # only in a
    assert gmap[9, 12] == 2  # overlap: later ROI wins
    assert gmap[9, 18] == 2  # only in b
    assert set(np.unique(gmap)) == {0, 1, 2}


def test_group_masks_or_within_group():
    a1 = RoiPolygon(rectangle_vertices((5, 5), (4, 4)), "inclusive", 1)
    a2 = RoiPolygon(rectangle_vertices((18, 20), (4, 4)), "inclusive", 1)
    b = RoiPolygon(rectangle_vertices((9, 12), (8, 10)), "inclusive", 2)
    skipped = RoiPolygon(circle_vertices((10, 12), 4.0))
    masks = group_masks(SHAPE, [a1, a2, b, skipped])
    assert set(masks) == {1, 2}
    assert masks[1][5, 5] and masks[1][18, 20]
    assert not masks[1][9, 12]
    assert masks[2][9, 12]


@pytest.mark.parametrize("fn", [group_index_map, group_masks])
def test_group_helpers_ignore_exclusive(fn):
    roi = RoiPolygon(circle_vertices((10, 12), 4.0), "exclusive", 5)
    result = fn(SHAPE, [roi])
    assert (not result.any()) if fn is group_index_map else result == {}
