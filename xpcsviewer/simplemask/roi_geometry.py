"""Convert pyqtgraph ROIs to image-space polygons through their transform."""

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import QPointF


def roi_to_polygon(roi: pg.ROI, image_item: pg.ImageItem, n: int = 180) -> np.ndarray:
    """Return the ROI outline as (N, 2) ``(row, col)`` pixel-centre vertices.

    Local points are mapped with ``roi.mapToItem`` so rotation and scaling are
    honoured. Item x is the column and item y the row; pixel centres sit half a
    pixel inside the item coordinates, hence the ``- 0.5``.
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
    return np.array([(p.y() - 0.5, p.x() - 0.5) for p in mapped], dtype=float)
