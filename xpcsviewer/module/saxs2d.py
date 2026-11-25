"""
2D SAXS scattering pattern visualization.

Provides visualization of integrated 2D small-angle X-ray scattering patterns
with support for log/linear scaling, colormap selection, and image rotation.

There is ONE function that turns ``(file, display options)`` into what is
displayed (:func:`render_frame`: the image, its orientation and the beam
centre, in one :class:`Saxs2dFrame`) and ONE that puts a frame on screen
(:func:`draw`). The synchronous :func:`plot` and the background
``SaxsPlotWorker`` + ``XpcsViewer.apply_saxs_2d_result`` both go through them,
so the two paths cannot disagree about orientation or beam-centre position.

Functions:
    render_frame: Image, orientation and beam centre of one file.
    draw: Show a frame in an ImageView with its beam-centre marker.
    plot: ``draw(render_frame(...))`` for the synchronous path.
"""

from dataclasses import dataclass

import numpy as np

from xpcsviewer.backends._conversions import ensure_numpy
from xpcsviewer.utils.logging_config import get_logger

logger = get_logger(__name__)

# (m11, m12, m21, m22, dx, dy) of a QTransform: x' = m11 x + m21 y + dx,
# y' = m12 x + m22 y + dy.
_IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


@dataclass(frozen=True)
class Saxs2dFrame:
    """What one SAXS-2D draw shows, all from ONE file and ONE set of options.

    Attributes:
        label: Name of the file the frame was made from.
        image: The detector image as stored (rows x columns), log scaled or
            linear. It is never rotated in memory: the orientation is the
            ``transform`` of the image item, so every pixel-index coordinate
            (beam centre, ROI export, mouse readout) stays in the file's frame.
        plot_type: ``"log"`` or ``"linear"``.
        rotate: Whether the display is turned by 90 degrees (``np.rot90``).
        transform: QTransform arguments mapping the image item's coordinates to
            the view's. Part of the frame so the marker and the image can only
            be placed with the same orientation.
        center: ``(bcx, bcy)``, the calibrated beam centre in the pixel-INDEX
            frame (the centre of pixel i is at coordinate i).
    """

    label: str
    image: np.ndarray
    plot_type: str
    rotate: bool
    transform: tuple[float, float, float, float, float, float]
    center: tuple[float, float]

    @property
    def display_shape(self) -> tuple[int, int]:
        """Shape of the image as displayed (rows, columns)."""
        rows, cols = self.image.shape[:2]
        return (cols, rows) if self.rotate else (rows, cols)


def _orientation(shape, rotate):
    """Item transform that shows an image of ``shape`` as ``np.rot90`` of it.

    ``np.rot90(A)[W - 1 - c, r] = A[r, c]`` for ``A`` of shape (H, W): item
    point (x, y) = (c, r) goes to view point (r, W - c), which is a pure
    rotation plus a shift of W.
    """
    if not rotate:
        return _IDENTITY
    return (0.0, -1.0, 1.0, 0.0, 0.0, float(shape[1]))


def render_frame(xfile, plot_type="log", rotate=False):
    """The single place that decides what SAXS-2D shows for ``xfile``.

    Args:
        xfile: XpcsFile with ``saxs_2d`` / ``saxs_2d_log`` and ``bcx`` / ``bcy``.
        plot_type: ``"log"`` uses the pre-computed log-scaled image; any other
            value uses the linear image.
        rotate: Turn the display by 90 degrees.

    Returns:
        Saxs2dFrame: image, orientation and beam centre, all of ``xfile``.
    """
    img = ensure_numpy(xfile.saxs_2d_log if plot_type == "log" else xfile.saxs_2d)
    rotate = bool(rotate)
    return Saxs2dFrame(
        label=str(getattr(xfile, "label", getattr(xfile, "fname", "unknown file"))),
        image=img,
        plot_type=str(plot_type),
        rotate=rotate,
        transform=_orientation(img.shape, rotate),
        center=(float(xfile.bcx), float(xfile.bcy)),
    )


def draw(
    pg_hdl,
    frame,
    cmap=None,
    autolevel=False,
    autorange=False,
    vmin=None,
    vmax=None,
):
    """Show ``frame`` in the ImageView ``pg_hdl`` and mark its beam centre.

    The marker is placed through the image item's own mapping, at the centre of
    the pixel whose index is the calibrated beam centre, in the displayed
    orientation.

    Args:
        pg_hdl: ImageViewDev handle for rendering.
        frame: The Saxs2dFrame from :func:`render_frame`.
        cmap: Colour-map name passed to ``pg_hdl.set_colormap``. ``None`` keeps
            the current colour map.
        autolevel: Automatically scale intensity levels to the data range.
        autorange: Force the view to fit the full image. When False the view
            range is kept unless the displayed image changed shape or
            orientation.
        vmin: Manual lower intensity clamp. Applied only when *autolevel* is
            False.
        vmax: Manual upper intensity clamp. Applied only when *autolevel* is
            False.
    """
    from pyqtgraph import QtGui

    logger.info(f"Drawing SAXS2D frame for {frame.label}")
    logger.debug(
        f"Frame: plot_type='{frame.plot_type}', rotate={frame.rotate}, "
        f"shape={frame.image.shape}, center={frame.center}, cmap='{cmap}', "
        f"autolevel={autolevel}, autorange={autorange}"
    )

    if cmap is not None:
        pg_hdl.set_colormap(cmap)

    prev_img = pg_hdl.image
    prev_tf = pg_hdl.getImageItem().transform()
    prev_transform = (
        prev_tf.m11(),
        prev_tf.m12(),
        prev_tf.m21(),
        prev_tf.m22(),
        prev_tf.dx(),
        prev_tf.dy(),
    )
    do_autorange = (
        autorange
        or prev_img is None
        or prev_img.shape != frame.image.shape
        or prev_transform != frame.transform
    )

    # Save view range if keeping it
    if not do_autorange:
        view_range = pg_hdl.view.viewRange()

    # Set new image - ensure NumPy at PyQtGraph boundary
    pg_hdl.setImage(
        ensure_numpy(frame.image),
        autoLevels=autolevel,
        autoRange=do_autorange,
        transform=QtGui.QTransform(*frame.transform),
    )

    # Restore view range if we skipped auto-ranging
    if not do_autorange:
        pg_hdl.view.setRange(xRange=view_range[0], yRange=view_range[1], padding=0)

    # Restore levels if needed
    if not autolevel and vmin is not None and vmax is not None:
        pg_hdl.setLevels(vmin, vmax)

    pg_hdl.add_roi(
        sl_type="Center", center=pg_hdl.pixel_centre(*frame.center), label="Center"
    )
    logger.info("SAXS2D frame drawn successfully")


def plot(
    xfile,
    pg_hdl=None,
    plot_type="log",
    cmap="jet",
    rotate=False,
    autolevel=False,
    autorange=False,
    vmin=None,
    vmax=None,
):
    """Display a 2-D SAXS detector image in a PyQtGraph ImageView.

    The synchronous path: ``draw(pg_hdl, render_frame(xfile, ...), ...)``. See
    :func:`render_frame` and :func:`draw` for the arguments.

    Returns:
        bool: The *rotate* flag, echoed back for caller convenience.

    Example:
        >>> rotate = plot(xfile, pg_hdl=viewer, plot_type="log",
        ...               cmap="viridis", vmin=0, vmax=100)
    """
    logger.info(f"Starting SAXS2D plot for {getattr(xfile, 'label', 'unknown file')}")
    frame = render_frame(xfile, plot_type=plot_type, rotate=rotate)
    draw(
        pg_hdl,
        frame,
        cmap=cmap,
        autolevel=autolevel,
        autorange=autorange,
        vmin=vmin,
        vmax=vmax,
    )
    return rotate
