"""SimpleMask window for mask editing.

This module provides the main QMainWindow for SimpleMask integration
with XPCS Viewer.
"""

import logging
import os
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from xpcsviewer.backends._conversions import ensure_numpy
from xpcsviewer.fileIO.qmap_utils import Q_UNIT_DISPLAY

# Qt imports via compatibility layer
from xpcsviewer.gui.qt_compat import (
    QAction,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QtCore,
    QThreadPool,
    QToolBar,
    QtWidgets,
    QVBoxLayout,
    QWidget,
    Signal,
    Slot,
)
from xpcsviewer.simplemask.drawing_tools import (
    DRAWING_TOOLS,
    ERASER_TOOL,
    get_tool_color,
)
from xpcsviewer.simplemask.pyqtgraph_mod import ImageViewROI
from xpcsviewer.simplemask.raw_file_worker import (
    RawFileLoadResult,
    RawFileLoadWorker,
    RawFramePreviewResult,
    RawFramePreviewWorker,
)
from xpcsviewer.simplemask.reader.formats import HDF5_SUFFIXES
from xpcsviewer.simplemask.simplemask_kernel import SimpleMaskKernel
from xpcsviewer.simplemask.ui.simplemask_ui import setup_ui
from xpcsviewer.utils.export_guard import ExportCollisionError, ExportGuard

logger = logging.getLogger(__name__)


def _is_raw_file(filename: str) -> bool:
    name = Path(filename).name.casefold()
    return name.endswith(HDF5_SUFFIXES) and not name.endswith("_metadata.hdf")


class _RawFileFilter(QtCore.QSortFilterProxyModel):
    def filterAcceptsRow(
        self, source_row: int, source_parent: QtCore.QModelIndex
    ) -> bool:
        model = self.sourceModel()
        if not isinstance(model, QtWidgets.QFileSystemModel):
            return False
        index = model.index(source_row, 0, source_parent)
        return model.isDir(index) or _is_raw_file(model.fileName(index))


class SimpleMaskWindow(QMainWindow):
    """Main window for SimpleMask mask editor.

    This window provides mask creation and editing tools integrated
    with XPCS Viewer.

    Signals:
        mask_exported: Emitted with mask array when user clicks "Apply to Viewer"
        mask_applied: The same click, with the context the viewer needs to tell
            the user's edits from the file's own exclusions: (mask, seed mask the
            editor started from or None, path of the file the editor displays or
            None). Emitted right after ``mask_exported``.
        qmap_exported: Emitted with partition dict when user exports Q-map
    """

    mask_exported = Signal(np.ndarray)
    mask_applied = Signal(np.ndarray, object, object)
    qmap_exported = Signal(dict)

    # UI Components (injected by setup_ui)
    status_bar: QStatusBar
    image_view: ImageViewROI
    info_label: QLabel
    tool_actions: dict[str, QAction]
    toolbar: QToolBar
    splitter: QSplitter
    left_panel: QWidget
    right_panel: QWidget
    right_layout: QVBoxLayout
    file_menu: QMenu
    raw_frame_display: QComboBox
    raw_frame_start: QSpinBox
    raw_frame_count: QSpinBox
    raw_frame_range_label: QLabel

    # Buttons (partition panel only - other controls are toolbar actions)
    btn_compute_partition: QPushButton
    btn_toggle_partition: QPushButton
    btn_export_partition: QPushButton

    btn_find_beam_center: QPushButton

    # SpinBoxes
    spin_bcx: QDoubleSpinBox
    spin_bcy: QDoubleSpinBox
    spin_det_dist: QDoubleSpinBox
    spin_pix_dim: QDoubleSpinBox
    spin_energy: QDoubleSpinBox
    spin_dq_num: QSpinBox
    spin_sq_num: QSpinBox
    spin_dp_num: QSpinBox
    spin_sp_num: QSpinBox

    # Actions
    action_eraser: QAction
    action_apply: QAction
    action_undo: QAction
    action_redo: QAction
    action_reset: QAction
    action_apply_viewer: QAction
    action_generate_qmap: QAction
    action_toggle_qmap: QAction
    action_toggle_mask: QAction
    action_save_mask: QAction
    action_load_mask: QAction
    action_open_raw_file: QAction
    action_apply_to_viewer: QAction
    action_close: QAction

    def __init__(self, parent_viewer: Any = None):
        """Initialize SimpleMask window.

        Args:
            parent_viewer: Reference to parent XPCS Viewer instance
        """
        super().__init__()
        self.parent_viewer = parent_viewer
        self.kernel: SimpleMaskKernel | None = None
        self._unsaved_changes = False
        self._detector_image: NDArray[np.float64] | None = None
        self._current_tool: str | None = None
        self._show_mask_overlay = False
        self._show_qmap_overlay = False
        self._show_partition_overlay = False
        self._last_save_path: str | None = None
        # Closing the window must not destroy a pool that is still reading frames.
        self.thread_pool = QThreadPool.globalInstance()
        self._raw_file_worker: RawFileLoadWorker | None = None
        self._raw_open_committing = False
        self._pending_viewer_load: (
            tuple[NDArray[np.float64] | None, dict[str, Any] | None] | None
        ) = None
        self._close_after_raw_open = False
        self._raw_source_path: str | None = None
        self._raw_source_frame_count: int | None = None
        self._raw_preview_worker: RawFramePreviewWorker | None = None
        self._raw_preview_pending: tuple[str, int, int, int] | None = None
        self._raw_preview_request_id = 0
        self._raw_preview_image: np.ndarray | None = None
        self._raw_preview_timer = QtCore.QTimer(self)
        self._raw_preview_timer.setSingleShot(True)
        self._raw_preview_timer.setInterval(100)

        self.setWindowTitle("Mask Editor")

        # Set up the UI
        setup_ui(self)
        self._geometry_fields = {
            "bcx": self.spin_bcx,
            "bcy": self.spin_bcy,
            "det_dist": self.spin_det_dist,
            "pix_dim": self.spin_pix_dim,
            "energy": self.spin_energy,
        }
        self._committed_geometry: dict[str, float] = {}

        # Connect signals
        self._connect_signals()
        self._refresh_partition_controls()

    def _connect_signals(self) -> None:
        """Connect UI signals to handlers."""
        # Drawing tool actions (toolbar)
        for tool_key, action in self.tool_actions.items():
            action.triggered.connect(
                lambda checked, k=tool_key: self._on_tool_selected(k)
            )

        # Eraser action (toolbar)
        self.action_eraser.triggered.connect(lambda: self._on_eraser_selected())

        # Apply drawing action (toolbar)
        self.action_apply.triggered.connect(self._on_apply_drawing)

        # Mask actions (toolbar)
        self.action_undo.triggered.connect(lambda: self._on_mask_action("undo"))
        self.action_redo.triggered.connect(lambda: self._on_mask_action("redo"))
        self.action_reset.triggered.connect(lambda: self._on_mask_action("reset"))

        # Toggle mask overlay (toolbar)
        self.action_toggle_mask.triggered.connect(self._on_toggle_mask)

        # File menu actions
        self.action_save_mask.triggered.connect(self._on_save_mask)
        self.action_load_mask.triggered.connect(self._on_load_mask)
        self.action_open_raw_file.triggered.connect(self._on_open_raw_file)
        self.action_apply_to_viewer.triggered.connect(self.export_mask_to_viewer)
        self.action_close.triggered.connect(self.close)
        self.raw_frame_display.currentIndexChanged.connect(self._on_raw_display_changed)
        self.raw_frame_start.valueChanged.connect(self._queue_raw_preview)
        self.raw_frame_count.valueChanged.connect(self._queue_raw_preview)
        self._raw_preview_timer.timeout.connect(self._start_raw_preview)

        # Toolbar Apply to Viewer button
        self.action_apply_viewer.triggered.connect(self.export_mask_to_viewer)

        # Q-Map controls (toolbar)
        self.action_generate_qmap.triggered.connect(self._on_generate_qmap)
        self.action_toggle_qmap.triggered.connect(self._on_toggle_qmap)

        for spinbox in self._geometry_fields.values():
            spinbox.valueChanged.connect(self._on_geometry_changed)

        # Auto beam-center finder
        self.btn_find_beam_center.clicked.connect(self._on_find_beam_center)

        # Partition controls
        self.btn_compute_partition.clicked.connect(self._on_compute_partition)
        self.btn_toggle_partition.clicked.connect(self._on_toggle_partition)
        self.btn_export_partition.clicked.connect(self.export_partition_to_viewer)
        for spinbox in (
            self.spin_dq_num,
            self.spin_sq_num,
            self.spin_dp_num,
            self.spin_sp_num,
        ):
            spinbox.valueChanged.connect(self._on_partition_bins_changed)

    def _on_tool_selected(self, tool_key: str) -> None:
        """Handle drawing tool selection.

        Args:
            tool_key: Key of selected tool from DRAWING_TOOLS
        """
        if self._raw_file_worker is not None:
            return
        if self.kernel is None or not self.kernel.is_ready():
            self.status_bar.showMessage("Load data first before drawing")
            return

        tool = DRAWING_TOOLS.get(tool_key)
        if tool is None:
            return

        self._current_tool = tool_key
        color = get_tool_color(tool.default_mode)

        # Add ROI to the view
        roi = self.kernel.add_drawing(
            sl_type=tool.tool_type,
            sl_mode=tool.default_mode,
            color=color,
        )

        if roi is not None:
            self.status_bar.showMessage(
                f"Added {tool.name} - drag to position, click Apply when done"
            )
            logger.debug(f"Added {tool.name} ROI")

        # Sync button states
        self._sync_tool_buttons(tool_key)

    def _on_eraser_selected(self) -> None:
        """Handle eraser tool selection."""
        if self._raw_file_worker is not None:
            return
        if self.kernel is None or not self.kernel.is_ready():
            self.status_bar.showMessage("Load data first before drawing")
            return

        self._current_tool = "eraser"
        color = get_tool_color(ERASER_TOOL.default_mode)

        # Add inclusive ROI (eraser)
        roi = self.kernel.add_drawing(
            sl_type=ERASER_TOOL.tool_type,
            sl_mode=ERASER_TOOL.default_mode,
            color=color,
        )

        if roi is not None:
            self.status_bar.showMessage(
                "Added Eraser region - drag to position, click Apply to remove from mask"
            )
            logger.debug("Added Eraser ROI")

        # Sync button states
        self._sync_tool_buttons("eraser")

    def _sync_tool_buttons(self, selected_tool: str | None) -> None:
        """Synchronize toolbar action checked states.

        Args:
            selected_tool: Currently selected tool key, or None
        """
        # Uncheck all tool actions
        for action in self.tool_actions.values():
            action.setChecked(False)

        self.action_eraser.setChecked(False)

        # Check the selected tool
        if selected_tool == "eraser":
            self.action_eraser.setChecked(True)
        elif selected_tool in self.tool_actions:
            self.tool_actions[selected_tool].setChecked(True)

    def _on_apply_drawing(self) -> None:
        """Apply all pending drawings to the mask."""
        if self._raw_file_worker is not None:
            return
        if self.kernel is None or not self.kernel.is_ready():
            self.status_bar.showMessage("No data loaded")
            return

        # An empty Apply must not replace (and so wipe) the previous draw layer.
        drawn = self.kernel.drawable_masks()
        if not drawn:
            self.status_bar.showMessage("No drawings to apply")
            return

        # Apply drawings from ROIs
        mask_from_drawing = self.kernel.apply_drawing(drawn)

        # Update mask via mask assembly
        if self.kernel.mask_kernel is not None:
            self.kernel.mask_kernel.evaluate("mask_draw", arr=~mask_from_drawing)
            self.kernel.mask_apply("mask_draw")

        self.mark_unsaved()
        self._ensure_mask_overlay_visible()
        self._refresh_mask_display()
        self.status_bar.showMessage("Applied drawings to mask")
        logger.info("Applied drawings to mask")

    def _on_mask_action(self, action: Literal["undo", "redo", "reset"]) -> None:
        """Handle undo/redo/reset mask action.

        Args:
            action: Action type
        """
        if self._raw_file_worker is not None:
            return
        if self.kernel is None:
            logger.debug(f"_on_mask_action({action}): No kernel loaded")
            return

        logger.debug(f"Mask action: {action}")
        previous_mask = self.kernel.mask
        self.kernel.mask_action(action)
        if not np.array_equal(previous_mask, self.kernel.mask):
            self.mark_unsaved()

        if action == "reset":
            if self.kernel.mask is not None and not self.kernel.mask.all():
                # The initial state is the file's own mask: keep showing it.
                self._ensure_mask_overlay_visible()
            else:
                # All-True initial state: nothing to visualize
                self._show_mask_overlay = False
                self.action_toggle_mask.setChecked(False)
                self.action_toggle_mask.setText("Show Mask")
            self._refresh_mask_display()
            self.status_bar.showMessage("Mask reset to initial state")
            logger.info("Mask reset to initial state")
        else:
            # Auto-enable overlay so undo/redo results are visible
            self._ensure_mask_overlay_visible()
            self._refresh_mask_display()
            # Check if mask is all-True (no masked pixels)
            if self.kernel.mask is not None and np.all(self.kernel.mask):
                self.status_bar.showMessage(
                    f"Mask {action} - no masked pixels (all pixels valid)"
                )
            else:
                self.status_bar.showMessage(f"Mask {action}")

    def _on_save_mask(self) -> None:
        """Handle save mask action with file dialog."""
        if self.kernel is None or self.kernel.mask is None:
            self.status_bar.showMessage("No mask to save")
            return

        # Suggest default filename
        default_name = "mask.h5"
        default_path = self._last_save_path or str(Path.home() / default_name)

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Mask",
            default_path,
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)",
        )

        if not file_path:
            return

        # Resolve the final name first: the checks below are about this file,
        # not about the name the dialog confirmed.
        confirmed = file_path
        if not file_path.endswith((".h5", ".hdf5")):
            file_path += ".h5"

        inputs = self._protected_paths()
        try:
            ExportGuard(inputs).claim(file_path, "mask")
        except ExportCollisionError as exc:
            self._report_failed_save("mask", file_path, exc)
            return

        if file_path != confirmed and Path(file_path).exists():
            # The file dialog only asked about the name without the extension.
            reply = QMessageBox.question(
                self,
                "Overwrite File",
                f"{Path(file_path).name} already exists.\n\nReplace it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                self.status_bar.showMessage("Mask not saved")
                return

        try:
            self.kernel.save_mask(file_path, input_paths=inputs)
        except (OSError, ExportCollisionError) as exc:
            self._report_failed_save("mask", file_path, exc)
            return

        self._last_save_path = file_path
        self.mark_saved()
        self.status_bar.showMessage(f"Mask saved to {Path(file_path).name}")
        logger.info(f"Saved mask to {file_path}")

    def _protected_paths(self) -> list[str]:
        """Files the editor or the viewer has open: no export may write to them.

        ``metadata["source_file"]`` (the analysis file the editor was opened on)
        stays protected for as long as the editor displays its data, whether or
        not the viewer still holds that file; only loading other data clears it.
        """
        paths = [self._raw_source_path] if self._raw_source_path else []
        source = self.kernel.metadata.get("source_file") if self.kernel else None
        if isinstance(source, str) and source:
            paths.append(source)
        vk = getattr(self.parent_viewer, "vk", None)
        if vk is not None:
            paths += [os.path.join(vk.path, name) for name in vk.target.copy()]
            paths += list(vk.cache)
        return paths

    def _report_failed_save(self, what: str, destination: str, exc: Exception) -> None:
        """Say which destination failed; the edits stay unsaved."""
        reason = getattr(exc, "strerror", None) or str(exc)
        logger.error(f"Failed to save {what} to {destination}: {exc}")
        self.status_bar.showMessage(f"Could not save {what} to {destination}: {reason}")

    def _on_load_mask(self) -> None:
        """Handle load mask action with file dialog and validation."""
        if self._raw_file_worker is not None:
            return
        if self.kernel is None or not self.kernel.is_ready():
            self.status_bar.showMessage("Load data first before loading a mask")
            return

        # Start from last save location or home
        start_path = self._last_save_path or str(Path.home())

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Mask",
            start_path,
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)",
        )

        if not file_path:
            return

        if self._raw_file_worker is not None:
            return  # type: ignore[unreachable]  # file dialog runs a nested event loop

        # Load the mask
        previous_mask = self.kernel.mask
        success = self.kernel.load_mask(file_path)
        if success:
            if not np.array_equal(previous_mask, self.kernel.mask):
                self.mark_unsaved()
            self._last_save_path = file_path
            self._refresh_mask_display()
            self.status_bar.showMessage(f"Loaded mask from {Path(file_path).name}")
            logger.info(f"Loaded mask from {file_path}")
        else:
            self.status_bar.showMessage(
                f"Failed to load mask from {Path(file_path).name}"
            )
            logger.warning(f"Failed to load mask from {file_path}")

    def _on_open_raw_file(self) -> None:
        """Choose an HDF5 detector file and its averaging range."""
        if not self._can_start_raw_open():
            return

        last_path = self._raw_source_path or self._last_save_path
        start_path = Path(last_path).parent if last_path else Path.home()
        dialog = QFileDialog(self, "Open Raw File", str(start_path))
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter(
            f"HDF5 Files ({' '.join(f'*{suffix}' for suffix in HDF5_SUFFIXES)})"
        )
        dialog.setProxyModel(_RawFileFilter(dialog))
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        selected = dialog.selectedFiles()
        dialog.deleteLater()
        if not accepted or not selected or not self._validate_raw_path(selected[0]):
            return

        frame_range = self._choose_raw_frame_range()
        if frame_range is not None:
            begin_idx, num_frames = frame_range
            self.load_from_raw_file(
                selected[0], "APS_8IDI", begin_idx=begin_idx, num_frames=num_frames
            )

    def _validate_raw_path(self, file_path: str) -> bool:
        if _is_raw_file(file_path):
            return True
        self.status_bar.showMessage(
            f"Failed to load raw file: select an HDF5 detector file ({Path(file_path).name})",
            8000,
        )
        return False

    def _choose_raw_frame_range(self) -> tuple[int, int] | None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Average Detector Frames")
        form = QFormLayout(dialog)
        mode = QComboBox(dialog)
        for label, count in (
            ("Representative sample", -1),
            ("All frames", 0),
            ("Custom", None),
        ):
            mode.addItem(label, count)
        start = QSpinBox(dialog)
        count = QSpinBox(dialog)
        for control in (start, count):
            control.setRange(0, 2_147_483_647)
            control.setEnabled(False)
        count.setValue(1)

        def set_custom_enabled() -> None:
            custom = mode.currentData() is None
            start.setEnabled(custom)
            count.setEnabled(custom)

        mode.currentIndexChanged.connect(set_custom_enabled)
        form.addRow("Frames", mode)
        form.addRow("Start frame (zero-based)", start)
        form.addRow("Frame count (0 for remaining)", count)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        preset_count = mode.currentData()
        frame_range = (
            (start.value(), count.value())
            if preset_count is None
            else (0, preset_count)
        )
        dialog.deleteLater()
        return frame_range if accepted else None

    def load_from_raw_file(
        self, file_path: str, beamline: str, *, begin_idx: int = 0, num_frames: int = -1
    ) -> None:
        """Read a candidate raw detector image on the thread pool."""
        if not self._can_start_raw_open() or not self._validate_raw_path(file_path):
            return

        if self.has_unsaved_changes():
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved mask changes. Replace this image and discard its mask?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if (
                reply != QMessageBox.StandardButton.Yes
                or not self._can_start_raw_open()
            ):
                return

        worker = RawFileLoadWorker(
            file_path, beamline, uuid4().hex, begin_idx=begin_idx, num_frames=num_frames
        )
        self._raw_file_worker = worker
        self._set_raw_open_busy(True)
        if not self._owns_raw_open(worker.worker_id):
            return
        self.status_bar.showMessage(f"Loading {Path(file_path).name}...")
        worker.signals.finished.connect(self._on_raw_file_loaded)
        worker.signals.error.connect(self._on_raw_file_error)
        worker.signals.cancelled.connect(self._on_raw_file_cancelled)
        if self._owns_raw_open(worker.worker_id):
            self.thread_pool.start(worker)

    def _can_start_raw_open(self) -> bool:
        return (
            not self._raw_open_committing
            and self._raw_file_worker is None
            and self.action_open_raw_file.isEnabled()
        )

    def _set_raw_open_busy(self, busy: bool) -> None:
        worker = self._raw_file_worker
        for control in (
            self.action_load_mask,
            self.action_undo,
            self.action_redo,
            self.action_reset,
            self.action_eraser,
            self.action_apply,
            *self.tool_actions.values(),
            self.image_view,
            # Enabling Open can synchronously start a new request.
            self.action_open_raw_file,
        ):
            if self._raw_file_worker is not worker:
                return
            control.setEnabled(not busy)

    def _owns_raw_open(self, request_id: str) -> bool:
        worker = self._raw_file_worker
        return worker is not None and worker.worker_id == request_id

    def _release_raw_open(self, request_id: str) -> bool:
        if not self._owns_raw_open(request_id):
            return False
        self._raw_file_worker = None
        self._set_raw_open_busy(False)
        return True

    def _cancel_raw_open(self) -> None:
        worker = self._raw_file_worker
        if worker is not None and self._release_raw_open(worker.worker_id):
            worker.cancel()

    @Slot(object)
    def _on_raw_file_loaded(self, result: RawFileLoadResult) -> None:
        """Handle a completed RawFileLoadWorker result on the GUI thread."""
        request_id = result["request_id"]
        if self._raw_open_committing or not self._owns_raw_open(request_id):
            return
        self._raw_open_committing = True
        try:
            try:
                self._install_detector_image(
                    result["scattering"],
                    result["metadata"],
                    qmap_result=result["qmap_result"],
                )
            finally:
                released = self._release_raw_open(request_id)
            if not released or self._pending_viewer_load is not None:
                return
            self._set_raw_source(result["source_path"], result["frame_count"])
            if result["metadata_is_placeholder"]:
                self.status_bar.showMessage(
                    "Loaded with placeholder geometry — the file's metadata "
                    "could not be read; edit the geometry fields by hand.",
                    8000,
                )
            else:
                self.status_bar.showMessage("Raw file loaded", 3000)
        finally:
            self._raw_open_committing = False
            pending = self._pending_viewer_load
            self._pending_viewer_load = None
            close_after_install = self._close_after_raw_open
            self._close_after_raw_open = False
            if pending is not None:
                self.load_from_viewer(*pending)
            if close_after_install:
                self.close()

    @Slot(str, str, str, int)
    def _on_raw_file_error(
        self, worker_id: str, error_msg: str, tb_str: str, retry_count: int
    ) -> None:
        """Handle a failed RawFileLoadWorker on the GUI thread."""
        if not self._owns_raw_open(worker_id):
            return
        logger.error(f"Raw file load failed: {error_msg}\n{tb_str}")
        self.status_bar.showMessage(f"Failed to load raw file: {error_msg}", 8000)
        self._release_raw_open(worker_id)

    @Slot(str, str)
    def _on_raw_file_cancelled(self, worker_id: str, message: str) -> None:
        if self._owns_raw_open(worker_id):
            self.status_bar.showMessage(f"Raw file load cancelled: {message}", 3000)
            self._release_raw_open(worker_id)

    def _set_raw_source(self, path: str | None, frame_count: int | None) -> None:
        self._invalidate_raw_preview()
        self._raw_source_path = path
        self._raw_source_frame_count = frame_count
        with (
            QtCore.QSignalBlocker(self.raw_frame_display),
            QtCore.QSignalBlocker(self.raw_frame_start),
            QtCore.QSignalBlocker(self.raw_frame_count),
        ):
            self.raw_frame_display.setCurrentIndex(0)
            self.raw_frame_display.setEnabled(frame_count is not None)
            self.raw_frame_start.setMaximum(min((frame_count or 1) - 1, 2_147_483_647))
            self.raw_frame_start.setValue(0)
            self.raw_frame_count.setValue(1)
            self.raw_frame_start.setEnabled(False)
            self.raw_frame_count.setEnabled(False)

    def _invalidate_raw_preview(self) -> None:
        self._raw_preview_request_id += 1
        self._raw_preview_timer.stop()
        self._raw_preview_pending = None
        self._raw_preview_image = None
        self.raw_frame_range_label.clear()
        if self._raw_preview_worker is not None:
            self._raw_preview_worker.cancel()

    @Slot()
    def _on_raw_display_changed(self) -> None:
        raw = self.raw_frame_display.currentData() is not None
        self.raw_frame_start.setEnabled(raw)
        self.raw_frame_count.setEnabled(raw)
        if not raw:
            self._invalidate_raw_preview()
        elif (
            self._raw_preview_image is None
            and self._raw_preview_pending is None
            and not self._is_current_raw_preview(self._raw_preview_worker)
        ):
            self._queue_raw_preview()
        self._refresh_display()

    @Slot()
    def _queue_raw_preview(self) -> None:
        path = self._raw_source_path
        if (
            path is None
            or self.raw_frame_display.currentData() is None
            or self._raw_open_committing
        ):
            return
        self._raw_preview_request_id += 1
        self._raw_preview_pending = (
            path,
            self.raw_frame_start.value(),
            self.raw_frame_count.value(),
            self._raw_preview_request_id,
        )
        self._raw_preview_timer.start()
        if self._raw_preview_worker is not None:
            self._raw_preview_worker.cancel()

    @Slot()
    def _start_raw_preview(self) -> None:
        if (
            self._raw_preview_worker is not None
            or self._raw_preview_pending is None
            or self._raw_preview_timer.isActive()
            or self._raw_open_committing
        ):
            return
        worker = RawFramePreviewWorker(*self._raw_preview_pending)
        self._raw_preview_pending = None
        self._raw_preview_worker = worker
        worker.signals.finished.connect(self._on_raw_preview_loaded)
        worker.signals.error.connect(self._on_raw_preview_error)
        worker.signals.cancelled.connect(self._on_raw_preview_cancelled)
        self.thread_pool.start(worker)

    def _is_current_raw_preview(self, worker: RawFramePreviewWorker | None) -> bool:
        return (
            worker is not None
            and worker.source_path == self._raw_source_path
            and worker.request_id == self._raw_preview_request_id
            and self.raw_frame_display.currentData() is not None
            and not self._raw_open_committing
        )

    @Slot(object)
    def _on_raw_preview_loaded(self, result: RawFramePreviewResult) -> None:
        worker = self._raw_preview_worker
        if (
            worker is None
            or result["source_path"] != worker.source_path
            or result["request_id"] != worker.request_id
        ):
            return
        self._raw_preview_worker = None
        if self._is_current_raw_preview(worker) and self._detector_image is not None:
            image = result["image"]
            if image.shape == self._detector_image.shape:
                begin, last = result["begin_idx"], result["end_idx"] - 1
                self._raw_preview_image = image
                self.raw_frame_range_label.setText(
                    f"Frame {begin}" if begin == last else f"Frames {begin} to {last}"
                )
                self._refresh_display()
            elif self._raw_file_worker is None:
                self.status_bar.showMessage(
                    "Failed to preview frames: detector dimensions changed", 8000
                )
        self._start_raw_preview()

    @Slot(str, str, str, int)
    def _on_raw_preview_error(
        self, worker_id: str, error_msg: str, tb_str: str, retry_count: int
    ) -> None:
        worker = self._raw_preview_worker
        if worker is None or worker.worker_id != worker_id:
            return
        self._raw_preview_worker = None
        if self._is_current_raw_preview(worker) and self._raw_file_worker is None:
            self.status_bar.showMessage(f"Failed to preview frames: {error_msg}", 8000)
        self._start_raw_preview()

    @Slot(str, str)
    def _on_raw_preview_cancelled(self, worker_id: str, message: str) -> None:
        worker = self._raw_preview_worker
        if worker is not None and worker.worker_id == worker_id:
            self._raw_preview_worker = None
            self._start_raw_preview()

    def _ensure_mask_overlay_visible(self) -> None:
        """Enable mask overlay and sync all toggle states.

        Called after mask-modifying operations (apply, undo, redo) so
        users can immediately see the result of their action.
        """
        if not self._show_mask_overlay:
            self._show_mask_overlay = True
            self._show_qmap_overlay = False
            self._show_partition_overlay = False
            self.action_toggle_mask.setChecked(True)
            self.action_toggle_mask.setText("Hide Mask")
            self.action_toggle_qmap.setChecked(False)
            self.action_toggle_qmap.setText("Show Q-Map")
            self.btn_toggle_partition.setChecked(False)
            self.btn_toggle_partition.setText("Show Partition")

    def _on_toggle_mask(self) -> None:
        """Toggle mask overlay visibility."""
        self._show_mask_overlay = not self._show_mask_overlay
        if self._show_mask_overlay:
            self._show_qmap_overlay = False  # Can only show one overlay at a time

        # Sync action states
        self.action_toggle_mask.setChecked(self._show_mask_overlay)
        self.action_toggle_qmap.setChecked(False)

        # Update action text
        text = "Hide Mask" if self._show_mask_overlay else "Show Mask"
        self.action_toggle_mask.setText(text)
        self.action_toggle_qmap.setText("Show Q-Map")

        self._refresh_display()

    def _on_toggle_qmap(self) -> None:
        """Toggle Q-map overlay visibility."""
        if self.kernel is None or self.kernel.qmap is None:
            self.status_bar.showMessage("Generate Q-map first")
            return

        self._show_qmap_overlay = not self._show_qmap_overlay
        if self._show_qmap_overlay:
            self._show_mask_overlay = False  # Can only show one overlay at a time

        # Sync action states
        self.action_toggle_qmap.setChecked(self._show_qmap_overlay)
        self.action_toggle_mask.setChecked(False)

        # Update action text
        text = "Hide Q-Map" if self._show_qmap_overlay else "Show Q-Map"
        self.action_toggle_qmap.setText(text)
        self.action_toggle_mask.setText("Show Mask")

        self._refresh_display()

    def _on_generate_qmap(self) -> None:
        """Generate Q-map from current geometry parameters."""
        if self.kernel is None or not self.kernel.is_ready():
            self.status_bar.showMessage("Load data first")
            return

        new_metadata = {
            key: field.value()
            for key, field in self._geometry_fields.items()
            if self.kernel.qmap is None
            or field.value() != self._committed_geometry[key]
        }

        try:
            self.kernel.update_parameters(new_metadata)
        except ValueError as e:
            # Handle missing/invalid parameters
            logger.warning(f"Q-map generation failed: {e}")
            self.status_bar.showMessage("Q-map failed: check geometry values")
            QMessageBox.warning(
                self,
                "Q-Map Generation Failed",
                f"Failed to generate Q-map:\n\n{e}",
            )
            return

        self._update_geometry_spinboxes()
        qmap = self.kernel.qmap
        # Initialize mask_kernel if this is the first successful Q-map computation
        # (happens when initial load failed due to missing geometry)
        if (
            self.kernel.mask_kernel is None
            and self.kernel.detector_image is not None
            and self.kernel.shape is not None
        ):
            from xpcsviewer.simplemask.area_mask import MaskAssemble

            self.kernel.mask_kernel = MaskAssemble(
                self.kernel.shape, self.kernel.detector_image
            )
            if self.kernel.qmap is not None:
                self.kernel.mask_kernel.update_qmap(self.kernel.qmap)
            logger.info("Initialized mask kernel after successful Q-map generation")

        if qmap is not None and "q" in qmap:
            q_min = np.nanmin(qmap["q"])
            q_max = np.nanmax(qmap["q"])
            self.status_bar.showMessage(
                f"Q-map generated: {q_min:.4f} - {q_max:.4f} {Q_UNIT_DISPLAY}"
            )
            logger.info(
                f"Generated Q-map: q range [{q_min:.4f}, {q_max:.4f}] {Q_UNIT_DISPLAY}"
            )

            # Update info label to show successful state
            if self._detector_image is not None:
                shape = self._detector_image.shape
                bcx, bcy = self.kernel.get_center()
                self.info_label.setText(
                    f"Image: {shape[1]} x {shape[0]} pixels\n"
                    f"Beam center: ({bcx:.1f}, {bcy:.1f})"
                )
        else:
            self.status_bar.showMessage("Q-map generation failed")

    def _on_geometry_changed(self) -> None:
        """Track unapplied edits against the last displayed geometry values."""
        if self.kernel is None or not self.kernel.is_ready():
            return

        pending = any(
            field.value() != self._committed_geometry[key]
            for key, field in self._geometry_fields.items()
        )
        self.kernel.set_geometry_pending(pending)
        self.btn_compute_partition.setEnabled(not pending)
        self.btn_find_beam_center.setEnabled(
            not pending and self.kernel.qmap is not None
        )
        self._refresh_partition_controls()

    def _on_find_beam_center(self) -> None:
        """Auto-detect beam center from the diffraction ring pattern."""
        if self.kernel is None or not self.kernel.is_ready():
            self.status_bar.showMessage("Load data first")
            return

        try:
            cx, cy, diagnostics = self.kernel.find_beam_center()
        except RuntimeError as e:
            logger.warning(f"Beam center detection unavailable: {e}")
            self.status_bar.showMessage("Beam center detection unavailable")
            QMessageBox.warning(self, "Beam Center Detection Unavailable", str(e))
            return
        except ValueError as e:
            logger.warning(f"Beam center detection failed: {e}")
            self._refresh_partition_controls()
            self.status_bar.showMessage("Beam center detection failed")
            QMessageBox.warning(self, "Beam Center Detection Failed", str(e))
            return

        self._update_geometry_spinboxes()

        converged = diagnostics.get("converged", False)
        iterations = diagnostics.get("iterations", 0)
        state = "converged" if converged else "stopped at max iterations"
        self.status_bar.showMessage(
            f"Beam center: ({cx:.2f}, {cy:.2f}) [{state} after {iterations} iterations]"
        )

    def _on_compute_partition(self) -> None:
        """Compute partition from current Q-map and binning parameters."""
        if self.kernel is None or not self.kernel.is_ready():
            self.status_bar.showMessage("Load data first")
            return

        if self.kernel.qmap is None:
            self.status_bar.showMessage("Generate Q-map first")
            return

        # Get binning parameters from spinboxes
        dq_num = self.spin_dq_num.value()
        sq_num = self.spin_sq_num.value()
        dp_num = self.spin_dp_num.value()
        sp_num = self.spin_sp_num.value()

        try:
            partition = self.kernel.compute_partition(
                mode="q-phi",
                dq_num=dq_num,
                sq_num=sq_num,
                dp_num=dp_num,
                sp_num=sp_num,
            )
        except ValueError as e:
            logger.warning(f"Partition computation failed: {e}")
            partition = None
        finally:
            self._refresh_partition_controls()

        if partition is not None:
            dq_count = (
                partition.get("dynamic_roi_map", np.array([])).max()
                if "dynamic_roi_map" in partition
                else 0
            )
            self.status_bar.showMessage(
                f"Partition computed: {dq_count} dynamic Q-bins"
            )
            logger.info(
                f"Computed partition: {dq_count} dynamic Q-bins, {sq_num} static Q-bins"
            )
        else:
            self.status_bar.showMessage("Partition computation failed")

    def _on_partition_bins_changed(self) -> None:
        """Discard the partition computed with the previous bin counts."""
        if self.kernel is not None:
            self.kernel.invalidate_partition()
            self._refresh_partition_controls()

    def _refresh_partition_controls(self) -> None:
        """Synchronize the overlay and export controls with the current result."""
        available = self.kernel is not None and self.kernel.new_partition is not None
        self.btn_export_partition.setEnabled(available)
        self.btn_toggle_partition.setEnabled(available)
        if self._show_partition_overlay:
            if not available:
                self._show_partition_overlay = False
                self.btn_toggle_partition.setChecked(False)
                self.btn_toggle_partition.setText("Show Partition")
            self._refresh_display()

    def _on_toggle_partition(self) -> None:
        """Toggle partition overlay visibility."""
        if self.kernel is None or self.kernel.new_partition is None:
            self.status_bar.showMessage("Compute partition first")
            self.btn_toggle_partition.setChecked(False)
            return

        self._show_partition_overlay = not self._show_partition_overlay
        if self._show_partition_overlay:
            self._show_mask_overlay = False
            self._show_qmap_overlay = False

        # Sync button/action states
        self.btn_toggle_partition.setChecked(self._show_partition_overlay)
        self.action_toggle_mask.setChecked(False)
        self.action_toggle_qmap.setChecked(False)

        # Update button/action text
        text = "Hide Partition" if self._show_partition_overlay else "Show Partition"
        self.btn_toggle_partition.setText(text)
        self.action_toggle_mask.setText("Show Mask")
        self.action_toggle_qmap.setText("Show Q-Map")

        self._refresh_display()

    def _update_geometry_spinboxes(self) -> None:
        """Display committed geometry and remember the rounded field values."""
        if self.kernel is None:
            return

        for key, field in self._geometry_fields.items():
            value = self.kernel.metadata.get(key)
            if value is not None:
                previous = field.blockSignals(True)
                try:
                    field.setValue(float(value))
                finally:
                    field.blockSignals(previous)
            self._committed_geometry[key] = field.value()
        self._on_geometry_changed()

    def _refresh_mask_display(self) -> None:
        """Refresh display with optional mask overlay (legacy method)."""
        self._refresh_partition_controls()
        self._refresh_display()

    def _refresh_display(self) -> None:
        """Refresh display with optional overlays (mask, Q-map, or partition)."""
        if self._detector_image is None:
            return

        channel = self.raw_frame_display.currentData()
        if channel is not None and self._raw_preview_image is not None:
            display = self._log_detector_image(self._raw_preview_image)
            if (
                channel == "masked"
                and self.kernel is not None
                and self.kernel.mask is not None
            ):
                display = self._create_masked_display(display, self.kernel.mask)
                self.image_view.setImage(
                    display, autoLevels=False, autoRange=False, levels=(0, 1)
                )
            else:
                self.image_view.setImage(display, autoLevels=True, autoRange=False)
            return

        # Base image (log scale)
        log_image = self._log_detector_image(self._detector_image)

        if (
            self._show_partition_overlay
            and self.kernel is not None
            and self.kernel.new_partition is not None
        ):
            # Show partition overlay
            display_image = self._create_partition_display(self.kernel.new_partition)
            # Ensure NumPy at PyQtGraph boundary (BUG-027)
            self.image_view.setImage(
                ensure_numpy(display_image), autoLevels=True, autoRange=False
            )
        elif (
            self._show_qmap_overlay
            and self.kernel is not None
            and self.kernel.qmap is not None
        ):
            # Show Q-map overlay
            display_image = self._create_qmap_display(log_image, self.kernel.qmap)
            # Ensure NumPy at PyQtGraph boundary (BUG-027)
            self.image_view.setImage(
                ensure_numpy(display_image), autoLevels=True, autoRange=False
            )
        elif (
            self._show_mask_overlay
            and self.kernel is not None
            and self.kernel.mask is not None
        ):
            # Create image with mask overlay
            # Masked pixels (False in mask) shown darker
            display_image = self._create_masked_display(log_image, self.kernel.mask)
            # Ensure NumPy at PyQtGraph boundary (BUG-027)
            self.image_view.setImage(
                ensure_numpy(display_image), autoLevels=False, autoRange=False
            )
        else:
            # Ensure NumPy at PyQtGraph boundary (BUG-027)
            self.image_view.setImage(
                ensure_numpy(log_image), autoLevels=True, autoRange=False
            )

    def _create_qmap_display(self, image: NDArray, qmap: dict[str, NDArray]) -> NDArray:
        """Create display image with Q-map overlay.

        Args:
            image: Base image (2D)
            qmap: Q-map dictionary with 'q' key

        Returns:
            Q-map visualization
        """
        if "q" not in qmap:
            return image

        q_array = qmap["q"]

        # Return Q-map values directly for visualization
        # Apply mask if available
        if self.kernel is not None and self.kernel.mask is not None:
            display = q_array.copy()
            display[~self.kernel.mask] = np.nan
            return display

        return q_array

    def _create_partition_display(self, partition: dict) -> NDArray:
        """Create display image with partition overlay.

        Args:
            partition: Partition dictionary from kernel

        Returns:
            Partition visualization (color-coded by Q-bin)
        """
        # Use dynamic ROI map for visualization
        if "dynamic_roi_map" in partition:
            roi_map = partition["dynamic_roi_map"].astype(np.float64)
        elif "static_roi_map" in partition:
            roi_map = partition["static_roi_map"].astype(np.float64)
        else:
            return (
                np.zeros(self._detector_image.shape)
                if self._detector_image is not None
                else np.array([])
            )

        # Apply mask if available
        if self.kernel is not None and self.kernel.mask is not None:
            display = roi_map.copy()
            display[~self.kernel.mask] = np.nan
            return display

        return roi_map

    @staticmethod
    def _log_detector_image(image: NDArray) -> NDArray:
        """Render nonfinite counts and counts at or below -1 at the zero-count level."""
        counts = np.where(np.isfinite(image) & (image > -1), image, 0.0)
        return np.log10(counts + 1)

    def _create_masked_display(self, image: NDArray, mask: NDArray) -> NDArray:
        """Create display image with mask overlay.

        Args:
            image: Base image (2D)
            mask: Boolean mask array (True = keep, False = masked)

        Returns:
            Image with mask overlay visualization
        """
        # Normalize image to 0-1 range, guarding against all-NaN or constant
        vmin, vmax = np.nanmin(image), np.nanmax(image)
        if np.isnan(vmin) or np.isnan(vmax) or vmax <= vmin:
            normalized = np.zeros_like(image)
        else:
            normalized = (image - vmin) / (vmax - vmin)

        # Create display - masked pixels shown darker/red-tinted
        display = normalized.copy()

        # Darken masked pixels
        masked_pixels = ~mask
        display[masked_pixels] = display[masked_pixels] * 0.3

        return display

    def load_from_viewer(
        self,
        detector_image: NDArray[np.float64] | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Load data from XPCS Viewer.

        Args:
            detector_image: 2D detector image array, or None for empty canvas
            metadata: Geometry metadata dict containing:
                - bcx, bcy: Beam center in pixels
                - det_dist: Sample-to-detector distance in mm
                - pix_dim: Pixel dimension in mm
                - energy: X-ray energy in keV
                - shape: Detector shape (height, width)
        """
        if self._raw_open_committing:
            self._pending_viewer_load = detector_image, metadata
            return
        self._cancel_raw_open()
        self._set_raw_source(None, None)
        if detector_image is None:
            if self.kernel is not None:
                self.kernel.clear_data()
            self._detector_image = None
            self._committed_geometry.clear()
            self._reset_image_controls()
            self.btn_compute_partition.setEnabled(False)
            self.btn_find_beam_center.setEnabled(False)
            self._refresh_partition_controls()
            self.image_view.clear()
            self.mark_saved()
            logger.info("No detector image provided, starting with empty canvas")
            self.info_label.setText(
                "No data loaded.\n\n"
                "Open an HDF5 file in XPCS Viewer first,\n"
                "then click the Mask Editor tab again."
            )
            self.status_bar.showMessage("No data - load file in XPCS Viewer first")
            return

        self._install_detector_image(detector_image, metadata or {})

    def _install_detector_image(
        self,
        detector_image: NDArray[np.float64],
        metadata: dict[str, Any],
        *,
        qmap_result: tuple[dict[str, np.ndarray], dict[str, str]] | None = None,
    ) -> None:
        # Initialize kernel with image view handle
        if self.kernel is None:
            self.kernel = SimpleMaskKernel(
                pg_hdl=self.image_view, infobar=self.status_bar
            )

        try:
            success = self.kernel.read_data(
                detector_image, metadata, qmap_result=qmap_result
            )
        except ValueError as e:
            if qmap_result is not None:
                raise
            # Handle missing geometry parameters gracefully
            logger.warning(f"Failed to initialize Q-map: {e}")

            # Display the image anyway
            self._detector_image = detector_image
            self._display_image(detector_image)

            # Show info about missing parameters
            shape = detector_image.shape
            self.info_label.setText(
                f"Image: {shape[1]} x {shape[0]} pixels\n\n"
                f"Warning: Missing geometry parameters.\n"
                f"Set values in Geometry panel,\n"
                f"then click 'Generate Q-Map'."
            )
            self.status_bar.showMessage(
                "Loaded image - set geometry parameters manually"
            )

            # Populate spinboxes with any available values
            self._update_geometry_spinboxes()

            logger.info(
                f"Loaded detector image {shape} without Q-map (missing geometry)"
            )
            return

        if success:
            logger.info(f"Loaded detector image with shape {detector_image.shape}")
            self._detector_image = detector_image
            self._reset_image_controls()
            self.image_view.remove_rois()

            # Display the image
            self._display_image(detector_image)
            if self.kernel.mask is not None and not self.kernel.mask.all():
                # Seeded with the file's own exclusions: show them from the start.
                self._ensure_mask_overlay_visible()
                self._refresh_mask_display()

            # Update info label
            shape = detector_image.shape
            bcx, bcy = self.kernel.get_center()
            self.info_label.setText(
                f"Image: {shape[1]} x {shape[0]} pixels\nBeam center: ({bcx}, {bcy})"
            )
            # Populate geometry spinboxes from metadata
            self._update_geometry_spinboxes()
            if qmap_result is None:
                self.status_bar.showMessage(
                    f"Loaded {shape[1]}x{shape[0]} detector image"
                )
            # Title-change listeners can synchronously load a different image.
            self.mark_saved()

    def _reset_image_controls(self) -> None:
        self._current_tool = None
        self._sync_tool_buttons(None)
        self._show_mask_overlay = False
        self._show_qmap_overlay = False
        self._show_partition_overlay = False
        for control, label in (
            (self.action_toggle_mask, "Show Mask"),
            (self.action_toggle_qmap, "Show Q-Map"),
            (self.btn_toggle_partition, "Show Partition"),
        ):
            control.setChecked(False)
            control.setText(label)

    def _display_image(self, image: NDArray[np.float64]) -> None:
        """Display detector image in the ImageView.

        Args:
            image: 2D detector image array
        """
        log_image = self._log_detector_image(image)

        # Ensure NumPy at PyQtGraph boundary (BUG-027)
        self.image_view.setImage(
            ensure_numpy(log_image),
            autoLevels=True,
            autoRange=True,
        )

        # Set aspect ratio to 1:1 (square pixels)
        self.image_view.getView().setAspectLocked(True)

    def refresh_display(self) -> None:
        """Refresh the image display with current mask overlay."""
        self._refresh_mask_display()

    def export_mask_to_viewer(self) -> None:
        """Emit current mask via mask_exported signal."""
        if self.kernel is not None and self.kernel.mask is not None:
            mask = self.kernel.mask
            if not isinstance(mask, np.ndarray) or mask.ndim != 2:
                logger.warning(
                    f"Invalid mask: expected 2D ndarray, got "
                    f"{'non-array' if not isinstance(mask, np.ndarray) else f'{mask.ndim}D'}"
                )
                return
            self.mask_exported.emit(mask)
            self.mask_applied.emit(
                mask, self.kernel.seed_mask, self.kernel.metadata.get("source_file")
            )
            logger.info("Mask exported to viewer")
            self.status_bar.showMessage("Mask exported to XPCS Viewer")

    def export_partition_to_viewer(self) -> None:
        """Emit current partition via qmap_exported signal."""
        if self.kernel is not None and self.kernel.new_partition is not None:
            partition = self.kernel.new_partition
            if not isinstance(partition, dict):
                logger.warning(  # type: ignore[unreachable]
                    f"Invalid partition: expected dict, got {type(partition).__name__}"
                )
                return
            required_keys = {"dynamic_roi_map", "static_roi_map"}
            missing = required_keys - set(partition.keys())
            if missing:
                logger.warning(f"Partition missing required keys: {missing}")
                return
            self.qmap_exported.emit(partition)
            logger.info("Partition exported to viewer")
            self.status_bar.showMessage("Partition exported to XPCS Viewer")

    def has_unsaved_changes(self) -> bool:
        """Check if there are unsaved mask modifications.

        Returns:
            True if there are unsaved changes
        """
        return self._unsaved_changes

    def mark_unsaved(self) -> None:
        """Mark that there are unsaved changes."""
        self._unsaved_changes = True
        if not self.windowTitle().endswith("*"):
            self.setWindowTitle(self.windowTitle() + " *")

    def mark_saved(self) -> None:
        """Mark that changes have been saved."""
        self._unsaved_changes = False
        title = self.windowTitle()
        if title.endswith(" *"):
            self.setWindowTitle(title[:-2])

    def closeEvent(self, event) -> None:
        """Handle window close with unsaved changes prompt.

        Args:
            event: Close event
        """
        if self._raw_open_committing:
            self._close_after_raw_open = True
            event.ignore()
            return

        # Skip confirmation dialog in test/headless mode to prevent blocking
        import os

        is_test_mode = (
            os.environ.get("XPCSVIEWER_TEST_MODE") == "1"
            or os.environ.get("QT_QPA_PLATFORM") == "offscreen"
        )

        if self._unsaved_changes and not is_test_mode:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved mask changes.\n\nClose without saving?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return

        self._cancel_raw_open()
        self.raw_frame_display.setCurrentIndex(0)

        # Clear parent reference to allow garbage collection
        if self.parent_viewer is not None:
            self.parent_viewer._simplemask_window = None

        event.accept()
