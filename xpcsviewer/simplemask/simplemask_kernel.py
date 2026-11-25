"""Core kernel for SimpleMask operations.

This module provides the SimpleMaskKernel class that handles all mask operations,
Q-map computation, partition generation, and ROI management.

Adapted from pySimpleMask for integration with XPCS Viewer.
"""

import logging
from collections.abc import Iterable
from typing import Any, Literal

import h5py
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from skimage.draw import polygon2mask

from xpcsviewer.backends._conversions import ensure_numpy
from xpcsviewer.fileIO.qmap_utils import Q_UNIT_DISPLAY
from xpcsviewer.simplemask.area_mask import MaskAssemble
from xpcsviewer.simplemask.calibration import compute_center_from_ring
from xpcsviewer.simplemask.pyqtgraph_mod import ImageViewROI, LineROI
from xpcsviewer.simplemask.qmap import compute_qmap
from xpcsviewer.simplemask.roi_geometry import (
    pixel_centre,
    pixel_index_at,
    roi_to_polygon,
)
from xpcsviewer.simplemask.utils import (
    check_consistency,
    combine_partitions,
    generate_partition,
    hash_numpy_dict,
    optimize_integer_array,
)
from xpcsviewer.utils.export_guard import ExportGuard

pg.setConfigOptions(imageAxisOrder="row-major")

logger = logging.getLogger(__name__)

# Compression threshold for HDF5 datasets (elements)
HDF5_COMPRESSION_THRESHOLD = 1024

# Version for partition file metadata
__version__ = "1.0.0"


class SimpleMaskKernel:
    """Core kernel for SimpleMask operations.

    This class manages mask creation, editing, Q-map computation,
    and partition generation for XPCS analysis.

    Attributes:
        shape: Detector shape as (height, width)
        detector_image: 2D array of detector data
        mask: Current combined mask array
        qmap: Dictionary of Q-map arrays
        qmap_unit: Dictionary of Q-map units
        metadata: Geometry metadata dictionary
        seed_mask: The mask the editor started from (the file's applied mask),
            or None when it started with every pixel valid
    """

    def __init__(
        self,
        pg_hdl: ImageViewROI | None = None,
        infobar: Any | None = None,
    ):
        """Initialize SimpleMask kernel.

        Args:
            pg_hdl: PyQtGraph ImageView handle for display
            infobar: Status bar widget for messages
        """
        self.detector_image: np.ndarray | None = None
        self.shape: tuple[int, int] | None = None
        self.qmap: dict[str, np.ndarray] | None = None
        self.qmap_unit: dict[str, str] | None = None
        self.mask: np.ndarray | None = None
        self.seed_mask: np.ndarray | None = None
        self.mask_kernel: MaskAssemble | None = None
        self.new_partition: dict | None = None
        self._geometry_pending = False
        self.metadata: dict[str, Any] = {}

        self.hdl = pg_hdl
        self.infobar = infobar

        if self.hdl is not None:
            self.hdl.scene.sigMouseMoved.connect(self.show_location)

    def is_ready(self) -> bool:
        """Check if kernel has data loaded.

        Returns:
            True if detector image is loaded
        """
        return self.detector_image is not None

    def invalidate_partition(self) -> None:
        """Discard the partition derived from the previous inputs."""
        self.new_partition = None

    def set_geometry_pending(self, pending: bool) -> None:
        """Prevent partition computation while geometry edits are unapplied."""
        self._geometry_pending = pending
        if pending:
            self.invalidate_partition()

    def clear_data(self) -> None:
        """Discard the loaded detector data and derived results."""
        self.invalidate_partition()
        self.set_geometry_pending(False)
        self.detector_image = self.shape = self.mask = self.mask_kernel = None
        self.seed_mask = None
        self.qmap = self.qmap_unit = None
        self.metadata = {}

    def read_data(
        self,
        detector_image: np.ndarray,
        metadata: dict[str, Any],
        *,
        qmap_result: tuple[dict[str, np.ndarray], dict[str, str]] | None = None,
    ) -> bool:
        """Load detector data and initialize mask.

        Args:
            detector_image: 2D array of detector counts
            metadata: Dictionary with geometry parameters:
                - bcx: Beam center X (column)
                - bcy: Beam center Y (row)
                - det_dist: Detector distance (mm)
                - pix_dim: Pixel size (mm)
                - energy: X-ray energy (keV)
                - shape: Detector shape (height, width)
                - mask: optional 2-D array of ``shape`` (nonzero = valid pixel),
                  the file's own mask; it is the editor's base layer (kept as
                  ``seed_mask``). A missing key or a different shape starts with
                  every pixel accepted.
                - source_file: optional path of the analysis file the image
                  came from; no export may overwrite it (``_claim_output``).
            qmap_result: Precomputed Q-map and units for the supplied geometry.

        Returns:
            True if successful
        """
        image = detector_image.copy()
        shape = image.shape
        metadata = metadata.copy()
        metadata["shape"] = shape
        # An array in the metadata would break dict comparisons (update_parameters).
        seed = metadata.pop("mask", None)
        base_mask = None
        if seed is not None:
            seed = np.asarray(seed)
            if seed.shape == shape:
                base_mask = seed != 0
            else:
                logger.warning(
                    f"Ignoring the file's mask: its shape {seed.shape} differs "
                    f"from the detector image shape {shape}"
                )

        # Fill in defaults for missing geometry parameters so Q-map
        # computation can proceed even when HDF5 metadata is incomplete.
        # These match the spinbox defaults in simplemask_ui.py.
        _geometry_defaults = {
            "pix_dim": 0.075,
            "energy": 10.0,
            "det_dist": 5000.0,
            "bcx": shape[1] / 2.0,
            "bcy": shape[0] / 2.0,
        }
        for key, default in _geometry_defaults.items():
            if metadata.get(key) is None:
                logger.info(
                    f"Geometry parameter '{key}' missing from HDF5 metadata, "
                    f"using default: {default}"
                )
                metadata[key] = default

        mask_kernel = MaskAssemble(shape, image, initial_mask=base_mask)
        mask = mask_kernel.get_mask()
        self.clear_data()
        self.detector_image, self.shape, self.metadata = image, shape, metadata
        self.mask = mask
        self.seed_mask = None if base_mask is None else base_mask.copy()
        if qmap_result is None:
            self._apply_geometry(metadata)
        else:
            self.qmap, self.qmap_unit = qmap_result

        self.mask_kernel = mask_kernel
        self.mask_kernel.update_qmap(self.qmap)

        return True

    def compute_qmap(self) -> tuple[dict[str, np.ndarray], dict[str, str]]:
        """Recompute Q-map from current metadata.

        Returns:
            Tuple of (qmap_dict, units_dict)
        """
        self.invalidate_partition()
        return self._apply_geometry(self.metadata)

    def _apply_geometry(
        self, metadata: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], dict[str, str]]:
        stype = metadata.get("stype", "Transmission")
        logger.debug(f"compute_qmap: stype={stype}, shape={self.shape}")
        qmap, units = compute_qmap(stype, metadata)
        self.metadata = metadata
        self.qmap, self.qmap_unit = qmap, units
        if self.mask_kernel is not None:
            self.mask_kernel.update_qmap(qmap)
        if logger.isEnabledFor(logging.DEBUG) and qmap:
            sqmap = qmap.get("sqmap")
            if sqmap is not None:
                logger.debug(f"compute_qmap: result sqmap shape={sqmap.shape}")
        return qmap, units

    def mask_action(self, action: Literal["undo", "redo", "reset"] = "undo") -> None:
        """Execute undo/redo/reset on mask history.

        Args:
            action: One of "undo", "redo", or "reset"
        """
        if self.mask_kernel is None:
            return
        self.mask_kernel.redo_undo(action=action)
        self.mask_apply()

    def mask_apply(self, target: str | None = None) -> np.ndarray:
        """Apply a mask type and update current mask.

        Args:
            target: Mask type to apply, or None to get current mask

        Returns:
            Current combined mask array
        """
        if self.mask_kernel is None or self.mask is None:
            return self.mask if self.mask is not None else np.zeros((0, 0), dtype=bool)

        mask = self.mask_kernel.apply(target)
        if self.new_partition is not None and not np.array_equal(self.mask, mask):
            self.invalidate_partition()
        self.mask = mask
        return self.mask

    def _claim_output(
        self, path: str, what: str, input_paths: Iterable[str] = ()
    ) -> str:
        """The write gate: refuse a destination that is a file in use as input.

        The loaded detector's own source file is always protected, besides the
        ``input_paths`` the caller knows about (raw file, files of the viewer).

        Raises:
            ExportCollisionError: ``path`` resolves to one of those files.
        """
        source = self.metadata.get("source_file")
        inputs = [
            *input_paths,
            *([source] if isinstance(source, str) and source else []),
        ]
        return ExportGuard(inputs).claim(path, what)

    def save_mask(self, save_name: str, input_paths: Iterable[str] = ()) -> None:
        """Save mask to HDF5 file.

        Args:
            save_name: Output file path (.h5/.hdf5)
            input_paths: Files that must never be overwritten (see `_claim_output`)

        Raises:
            ExportCollisionError: ``save_name`` is one of the input files.
            OSError: the file cannot be written.
        """
        if self.mask is None:
            logger.warning("No mask to save")
            return

        save_name = self._claim_output(save_name, "mask", input_paths)
        mask = self.mask.astype(np.uint8)

        try:
            with h5py.File(save_name, "w") as hf:
                hf.create_dataset("mask", data=mask, compression="lzf")
                hf.attrs["shape"] = self.shape
                hf.attrs["version"] = __version__
        except OSError as e:
            logger.error(f"Failed to save mask to {save_name}: {e}")
            raise

    def load_mask(self, fname: str, key: str = "mask") -> bool:
        """Load mask from HDF5 file.

        Args:
            fname: Path to HDF5 file
            key: Dataset key for mask data

        Returns:
            True if a mask of the detector shape (or its transpose) was read
            and applied; False leaves the current mask and history unchanged.
        """
        if self.mask_kernel is None:
            return False

        # MaskAssemble.evaluate returns a description, not a success flag.
        if not self.mask_kernel.workers["mask_file"].evaluate(fname=fname, key=key):
            return False
        self.mask_apply("mask_file")
        return True

    def save_partition(
        self,
        save_fname: str,
        root: str = "/qmap",
        input_paths: Iterable[str] = (),
    ) -> None:
        """Save partition to HDF5 file.

        Args:
            save_fname: Output file path (.h5/.hdf5)
            root: HDF5 group path for partition data
            input_paths: Files that must never be overwritten (see `_claim_output`)

        Raises:
            ExportCollisionError: ``save_fname`` is one of the input files.
        """
        if self.new_partition is None:
            logger.warning("No partition to save")
            return

        save_fname = self._claim_output(save_fname, "partition", input_paths)

        # Optimize integer arrays
        for key, val in self.new_partition.items():
            self.new_partition[key] = optimize_integer_array(val)

        hash_val = hash_numpy_dict(self.new_partition)
        logger.info(f"Hash value of the partition: {hash_val}")

        def optimize_save(group_handle, key, val):
            compression = (
                "lzf"
                if isinstance(val, np.ndarray) and val.size > HDF5_COMPRESSION_THRESHOLD
                else None
            )
            return group_handle.create_dataset(key, data=val, compression=compression)

        with h5py.File(save_fname, "w") as hf:
            if root in hf:
                del hf[root]
            group_handle = hf.create_group(root)

            for key, val in self.new_partition.items():
                dset = optimize_save(group_handle, key, val)
                if "_v_list_dim" in key:
                    dim = int(key[-1])
                    dset.attrs["unit"] = self.new_partition["map_units"][dim]
                    dset.attrs["name"] = self.new_partition["map_names"][dim]
                    dset.attrs["size"] = val.size

            group_handle.attrs["hash"] = hash_val
            group_handle.attrs["version"] = __version__

    def show_location(self, pos) -> None:
        """Display pixel coordinates and Q values at mouse position.

        Args:
            pos: Mouse scene position
        """
        if self.hdl is None or self.shape is None:
            return
        if not self.hdl.scene.itemsBoundingRect().contains(pos):
            return

        mouse_point = self.hdl.getView().mapSceneToView(pos)
        col, row = pixel_index_at(self.hdl.imageItem, mouse_point.x(), mouse_point.y())

        if 0 <= row < self.shape[0] and 0 <= col < self.shape[1]:
            msg = f"x={col}, y={row}"
            q_val = (
                self.qmap.get("q", self.qmap.get("sqmap"))
                if self.qmap is not None
                else None
            )
            if self.qmap is not None and q_val is not None:
                msg += f", q={q_val[row, col]:.4f} {Q_UNIT_DISPLAY}"
            if self.detector_image is not None:
                intensity = self.detector_image[row, col]
                msg += f", I={intensity:.1f}"
            if self.infobar is not None:
                self.infobar.clearMessage()
                self.infobar.showMessage(msg)

    def show_saxs(
        self,
        cmap: str = "jet",
        log: bool = True,
        plot_center: bool = True,
    ) -> None:
        """Display detector image with optional beam center marker.

        Args:
            cmap: Matplotlib colormap name
            log: Whether to use log scale (unused, kept for API compatibility)
            plot_center: Whether to display beam center marker
        """
        if self.detector_image is None or self.hdl is None:
            return

        self.hdl.clear()
        center = self.get_center()

        # Create display data (original + mask overlay)
        # Convert to NumPy to guard against JAX arrays at PyQtGraph boundary (BUG-027)
        data = self.detector_image.copy()
        self.hdl.setImage(ensure_numpy(data))
        self.hdl.adjust_viewbox()
        self.hdl.set_colormap(cmap)

        if plot_center and center[0] is not None and center[1] is not None:
            t = pg.ScatterPlotItem()
            x, y = pixel_centre(self.hdl.imageItem, center[0], center[1])
            t.addPoints(x=[x], y=[y], symbol="+", size=15)
            self.hdl.add_item(t, label="center")

    def drawable_masks(self) -> list[tuple[str, np.ndarray]]:
        """``(sl_mode, filled)`` for each pending ``roi_`` item that would
        change the mask, rasterising each ROI exactly once.

        Skips ROIs without ``sl_mode``, with non-finite or collinear vertices
        (rank < 2, not signed area: a self-crossing polyline has zero area yet
        fills pixels), or that cover no image pixel.
        """
        if self.hdl is None or self.shape is None:
            return []
        drawn = []
        for key, roi in self.hdl.roi_items.items():
            if not key.startswith("roi_") or not hasattr(roi, "sl_mode"):
                continue
            v = roi_to_polygon(roi, self.hdl.imageItem)
            if (
                not np.isfinite(v).all()
                or len(v) < 3
                or np.linalg.matrix_rank(v - v.mean(axis=0), tol=1e-9) < 2
            ):
                continue
            filled = polygon2mask(self.shape, v)
            if filled.any():
                drawn.append((roi.sl_mode, filled))
        return drawn

    def apply_drawing(
        self, drawn: list[tuple[str, np.ndarray]] | None = None
    ) -> np.ndarray:
        """Combine the pending drawing ROIs into the draw layer and remove them.

        The draw layer is replaced on every Apply, so the new layer starts from
        the drawn exclusions already in the current history state: exclusive
        shapes add to them, "erase" shapes subtract from them (and from nothing
        else: other mask layers are separate), and "inclusive" shapes keep only
        their own region.

        Args:
            drawn: result of `drawable_masks` from the same Apply, so the
                caller's emptiness check does not rasterise a second time.

        Returns:
            Boolean keep-mask of ``self.shape`` (all True when nothing is drawn)
        """
        if self.detector_image is None or self.hdl is None or self.shape is None:
            return np.ones(self.shape, dtype=bool) if self.shape else np.array([])

        if drawn is None:
            drawn = self.drawable_masks()
        # Same rule as rasterize.rasterize, on the already filled polygons.
        exclusive = np.zeros(self.shape, dtype=bool)
        inclusive = np.zeros(self.shape, dtype=bool)
        erase = np.zeros(self.shape, dtype=bool)
        has_inclusive = False
        for mode, filled in drawn:
            if mode == "inclusive":
                has_inclusive = True
                inclusive |= filled
            elif mode == "erase":
                erase |= filled
            else:
                exclusive |= filled
        previous = (
            self.mask_kernel.current_layer("mask_draw") if self.mask_kernel else None
        )
        if previous is not None:
            exclusive |= ~previous
        exclusive &= ~erase
        keep_inclusive = inclusive if has_inclusive else np.ones(self.shape, bool)
        keep = np.logical_and(~exclusive, keep_inclusive)
        self.hdl.remove_rois(filter_str="roi_")
        return keep

    def add_drawing(
        self,
        sl_type: Literal[
            "Rectangle", "Circle", "Ellipse", "Polygon", "Line"
        ] = "Polygon",
        sl_mode: Literal["exclusive", "inclusive", "erase"] = "exclusive",
        num_edges: int | None = None,
        radius: float = 60,
        color: str = "r",
        width: int = 3,
        second_point: tuple[float, float] | None = None,
        label: str | None = None,
        movable: bool = True,
    ) -> pg.ROI | None:
        """Add a drawing ROI to the view.

        Args:
            sl_type: Shape type - Rectangle, Circle, Ellipse, Polygon, or Line
            sl_mode: "exclusive" masks pixels, "inclusive" keeps only its region,
                "erase" removes its region from the drawn exclusions
            num_edges: Number of edges for Polygon (random 6-10 if None)
            radius: Default radius for Polygon; default length for Line
            color: Pen color
            width: Pen width; also the width in pixels of a Line strip
            second_point: Circle edge point or Line end point, in the pixel-index
                frame of the beam centre (the Line default is horizontal)
            label: ROI label (auto-generated if None)
            movable: Whether ROI can be moved

        Returns:
            Created ROI object, or None if creation failed
        """
        if self.hdl is None or self.shape is None:
            return None

        if label is not None and label in self.hdl.roi_items:
            self.hdl.remove_item(label)

        bcx, bcy = self.get_center()
        if bcx is None or bcy is None or bcx < 0 or bcy < 0:
            cx, cy = float(self.shape[1] // 2), float(self.shape[0] // 2)
        elif bcx > self.shape[1] or bcy > self.shape[0]:
            logger.warning("Beam center out of range, using image center")
            cx, cy = float(self.shape[1] // 2), float(self.shape[0] // 2)
        else:
            cx, cy = float(bcx), float(bcy)

        # Pixel-index frame -> the frame ROIs are drawn in, through the image item.
        item = self.hdl.imageItem
        ox, oy = pixel_centre(item, cx, cy)

        if sl_mode in ("inclusive", "erase"):
            pen = pg.mkPen(color=color, width=width, style=QtCore.Qt.DotLine)
        else:
            pen = pg.mkPen(color=color, width=width)

        handle_pen = pg.mkPen(color=color, width=width)

        kwargs = {
            "pen": pen,
            "removable": True,
            "hoverPen": pen,
            "handlePen": handle_pen,
            "movable": movable,
        }

        new_roi: pg.ROI | None = None

        if sl_type == "Ellipse":
            new_roi = pg.EllipseROI((ox, oy), [60, 80], **kwargs)
            # Midpoint handles (4)
            new_roi.addScaleHandle([0.5, 0], [0.5, 1])
            new_roi.addScaleHandle([0.5, 1], [0.5, 0])
            new_roi.addScaleHandle([0, 0.5], [1, 0.5])
            new_roi.addScaleHandle([1, 0.5], [0, 0.5])
            # Corner handles (4) - positions adjusted for ellipse geometry
            new_roi.addScaleHandle([0.1464, 0.1464], [1, 1])
            new_roi.addScaleHandle([0.1464, 0.8536], [1, 0])
            new_roi.addScaleHandle([0.8536, 0.1464], [0, 1])
            new_roi.addScaleHandle([0.8536, 0.8536], [0, 0])

        elif sl_type == "Circle":
            radius = 10.0
            if second_point is not None:
                sx, sy = pixel_centre(item, second_point[0], second_point[1])
                radius = float(np.hypot(sx - ox, sy - oy))
            new_roi = pg.CircleROI(
                pos=[ox - radius, oy - radius], radius=radius, **kwargs
            )
            # Add 2 opposite-side handles for uniform scaling
            new_roi.addScaleHandle([0.5, 0], [0.5, 0.5])
            new_roi.addScaleHandle([0.5, 1], [0.5, 0.5])

        elif sl_type == "Polygon":
            if num_edges is None:
                num_edges = np.random.randint(6, 11)
            offset = np.random.randint(0, 360)
            theta = np.linspace(0, np.pi * 2, num_edges + 1) + np.deg2rad(offset)
            x = radius * np.cos(theta) + ox
            y = radius * np.sin(theta) + oy
            pts = np.vstack([x, y]).T
            new_roi = pg.PolyLineROI(pts, closed=True, **kwargs)

        elif sl_type == "Rectangle":
            new_roi = pg.RectROI((ox, oy), [200, 150], **kwargs)
            # Corner handles (4)
            new_roi.addScaleHandle([0, 0], [1, 1])  # bottom-left
            new_roi.addScaleHandle([0, 1], [1, 0])  # top-left
            new_roi.addScaleHandle([1, 0], [0, 1])  # bottom-right
            new_roi.addScaleHandle([1, 1], [0, 0])  # top-right
            # Midpoint handles (4)
            new_roi.addScaleHandle([0, 0.5], [1, 0.5])  # left-mid
            new_roi.addScaleHandle([1, 0.5], [0, 0.5])  # right-mid
            new_roi.addScaleHandle([0.5, 0], [0.5, 1])  # bottom-mid
            new_roi.addScaleHandle([0.5, 1], [0.5, 0])  # top-mid

        elif sl_type == "Line":
            if second_point is None:
                # A movable default: horizontal, towards the side with room.
                length = float(max(1, min(radius, min(self.shape) // 2)))
                reach = cx + length
                second_point = (
                    reach if reach <= self.shape[1] - 1 else cx - length,
                    cy,
                )
            end = pixel_centre(item, second_point[0], second_point[1])
            new_roi = LineROI((ox, oy), end, float(width), **kwargs)

        else:
            raise TypeError(f"ROI type not implemented: {sl_type}")

        if new_roi is not None:
            new_roi.sl_mode = sl_mode
            roi_key = self.hdl.add_item(new_roi, label)
            new_roi.sigRemoveRequested.connect(lambda: self.remove_roi(roi_key))

        return new_roi

    def remove_roi(self, roi_key: str) -> None:
        """Remove an ROI by key.

        Args:
            roi_key: Label of the ROI to remove
        """
        if self.hdl is not None:
            self.hdl.remove_item(roi_key)

    def compute_partition(
        self,
        mode: str = "q-phi",
        dq_num: int = 10,
        sq_num: int = 100,
        dp_num: int = 36,
        sp_num: int = 360,
        style: str = "linear",
        phi_offset: float = 0.0,
        symmetry_fold: int = 1,
    ) -> dict | None:
        """Compute Q-space partition for XPCS analysis.

        Args:
            mode: Partition mode (e.g., "q-phi", "x-y")
            dq_num: Number of dynamic Q bins
            sq_num: Number of static Q bins
            dp_num: Number of dynamic phi bins
            sp_num: Number of static phi bins
            style: Binning style - "linear" or "logarithmic"
            phi_offset: Phi angle offset
            symmetry_fold: Symmetry fold for phi

        Returns:
            Partition dictionary or None if data or applied geometry is missing
        """
        self.invalidate_partition()
        if (
            self._geometry_pending
            or self.detector_image is None
            or self.qmap is None
            or self.mask is None
        ):
            return None

        map_names = mode.split("-")
        logger.info(f"Computing partition with mode {mode}: map_names {map_names}")

        name0, name1 = map_names

        # Generate dynamic partition
        pack_dq = generate_partition(
            name0, self.mask, self.qmap[name0], dq_num, style=style
        )
        pack_dp = generate_partition(
            name1,
            self.mask,
            self.qmap[name1],
            dp_num,
            style=style,
            phi_offset=phi_offset,
            symmetry_fold=symmetry_fold,
        )
        dynamic_map = combine_partitions(pack_dq, pack_dp, prefix="dynamic")

        # Generate static partition
        pack_sq = generate_partition(
            name0, self.mask, self.qmap[name0], sq_num, style=style
        )
        pack_sp = generate_partition(
            name1,
            self.mask,
            self.qmap[name1],
            sp_num,
            style=style,
            phi_offset=phi_offset,
            symmetry_fold=symmetry_fold,
        )
        static_map = combine_partitions(pack_sq, pack_sp, prefix="static")

        # Check consistency
        d_map = ensure_numpy(dynamic_map["dynamic_roi_map"])
        s_map = ensure_numpy(static_map["static_roi_map"])
        flag_consistency = check_consistency(d_map, s_map, self.mask)
        logger.info(f"dqmap/sqmap consistency check: {flag_consistency}")

        center = self._display_center()

        # Warn when using default physical constants
        _defaults = {"pix_dim": 0.075, "energy": 10.0, "det_dist": 5000.0}
        for _key, _default in _defaults.items():
            if _key not in self.metadata or self.metadata[_key] is None:
                logger.warning(
                    f"Using default {_key}={_default} — no value found in metadata"
                )

        partition = {
            "beam_center_x": center[0],
            "beam_center_y": center[1],
            "pixel_size": self.metadata.get("pix_dim", 0.075),
            "mask": self.mask,
            "blemish": self.mask_kernel.blemish if self.mask_kernel else self.mask,
            "energy": self.metadata.get("energy", 10.0),
            "detector_distance": self.metadata.get("det_dist", 5000.0),
            "map_names": list(map_names),
            "map_units": [
                self.qmap_unit[name0] if self.qmap_unit else "",
                self.qmap_unit[name1] if self.qmap_unit else "",
            ],
            "source_file": self.metadata.get("source_file", ""),
        }
        partition.update(dynamic_map)
        partition.update(static_map)

        self.new_partition = partition
        return partition

    def update_parameters(self, new_metadata: dict[str, Any]) -> None:
        """Update geometry parameters and recompute Q-map.

        Args:
            new_metadata: Dictionary with updated geometry values
        """
        metadata = self.metadata | new_metadata
        if metadata != self.metadata:
            self.invalidate_partition()
        self._apply_geometry(metadata)
        self.set_geometry_pending(False)

    def get_center(
        self, mode: Literal["xy", "vh"] = "xy"
    ) -> tuple[float | None, float | None]:
        """Get beam center coordinates.

        Args:
            mode: "xy" for (x, y) = (column, row), "vh" for (vertical, horizontal)

        Returns:
            Beam center coordinates
        """
        if not self.metadata:
            return (None, None)

        bcx = self.metadata.get("bcx")
        bcy = self.metadata.get("bcy")

        if bcx is None or bcy is None:
            return (None, None)

        if mode == "vh":
            return (bcy, bcx)
        return (bcx, bcy)

    def _display_center(self) -> tuple[float | None, float | None]:
        """Beam position on the detector, as pySimpleMask saves it.

        The metadata centre is the direct-beam pixel of an unswung detector; a
        swung detector sees it shifted by ``D * tan(swing) / pixel`` along
        each axis. With zero swing the centre is returned untouched.
        """
        bcx, bcy = self.get_center()
        if bcx is None or bcy is None:
            return (bcx, bcy)
        swing_h = float(self.metadata.get("swing_angle_horizontal") or 0.0)
        swing_v = float(self.metadata.get("swing_angle_vertical") or 0.0)
        if swing_h == 0.0 and swing_v == 0.0:
            return (bcx, bcy)
        scale = (self.metadata.get("det_dist") or 5000.0) / (
            self.metadata.get("pix_dim") or 0.075
        )
        return (
            bcx + scale * float(np.tan(np.deg2rad(swing_h))),
            bcy + scale * float(np.tan(np.deg2rad(swing_v))),
        )

    def find_beam_center(
        self, intensity_threshold: float | None = None
    ) -> tuple[float, float, dict]:
        """Auto-detect beam center from the diffraction ring pattern.

        Detects ring pixels in the detector image and refines the center
        via gradient-based (JAX) optimization. Updates ``self.metadata``
        with the result on success.

        Args:
            intensity_threshold: Threshold for detecting ring pixels.
                If None, uses mean + 2*std of the (masked) image.

        Returns:
            Tuple of (bcx, bcy, diagnostics_dict).

        Raises:
            RuntimeError: If no detector image is loaded, or the JAX
                backend is unavailable.
            ValueError: If insufficient ring pixels are detected.
        """
        if self.detector_image is None:
            raise RuntimeError("No detector image loaded.")
        if self._geometry_pending or self.qmap is None:
            raise ValueError("Generate Q-map to apply geometry changes first.")

        bcx, bcy = self.get_center(mode="xy")
        initial_center = (bcx, bcy) if bcx is not None and bcy is not None else None

        cx, cy, diagnostics = compute_center_from_ring(
            self.detector_image,
            mask=self.mask,
            intensity_threshold=intensity_threshold,
            initial_center=initial_center,
        )

        self.update_parameters({"bcx": cx, "bcy": cy})
        return cx, cy, diagnostics
