"""Convert between the pixel-INDEX frame and the frame ROIs are drawn in.

Calibrated coordinates (``bcx``/``bcy``, ``compute_qmap``, skimage rasterising)
put the centre of pixel ``i`` at coordinate ``i``; an ``ImageItem`` draws pixel
``i`` over the unit square ``[i, i + 1]``. ``PIXEL_CENTRE_OFFSET`` is the only
place that half-pixel appears; every conversion below goes through the image
item's own transform.
"""

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import QPointF

PIXEL_CENTRE_OFFSET = 0.5


def pixel_centre(
    image_item: pg.ImageItem, col: float, row: float
) -> tuple[float, float]:
    """Position of pixel-index point ``(col, row)`` in the frame ROIs live in.

    ROIs, markers and the view's coordinates are children of the image item's
    parent, so the item's real transform (shift, scale, rotation) is applied.
    """
    mapped = image_item.mapToParent(
        QPointF(col + PIXEL_CENTRE_OFFSET, row + PIXEL_CENTRE_OFFSET)
    )
    return mapped.x(), mapped.y()


def pixel_index_at(image_item: pg.ImageItem, x: float, y: float) -> tuple[int, int]:
    """``(col, row)`` of the pixel drawn under the ROI-frame position ``(x, y)``."""
    mapped = image_item.mapFromParent(QPointF(x, y))
    return int(np.floor(mapped.x())), int(np.floor(mapped.y()))


def roi_to_polygon(roi: pg.ROI, image_item: pg.ImageItem, n: int = 180) -> np.ndarray:
    """Return the ROI outline as (N, 2) ``(row, col)`` pixel-centre vertices.

    Local points are mapped with ``roi.mapToItem`` so rotation and scaling are
    honoured. Item x is the column and item y the row; pixel centres sit half a
    pixel inside the item coordinates (``PIXEL_CENTRE_OFFSET``).
    """
    if isinstance(roi, pg.PolyLineROI):
        local = [(p.x(), p.y()) for _, p in roi.getLocalHandlePositions()]
    else:
        w, h = roi.size()
        if isinstance(roi, pg.EllipseROI):
            t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
            local = list(
                zip(w / 2 * (1 + np.cos(t)), h / 2 * (1 + np.sin(t)), strict=True)
            )
        else:
            local = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    mapped = [roi.mapToItem(image_item, QPointF(float(x), float(y))) for x, y in local]
    rows_cols = [
        (p.y() - PIXEL_CENTRE_OFFSET, p.x() - PIXEL_CENTRE_OFFSET) for p in mapped
    ]
    return np.array(rows_cols, dtype=float).reshape(-1, 2)  # (0, 2) when empty
