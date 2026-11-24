"""Desktop preview for regrouping multi-tau G2 with an external Q-map."""

from html import escape
from pathlib import Path
from uuid import uuid4

import numpy as np
import pyqtgraph as pg

from xpcsviewer.fileIO.g2_regroup import (
    G2RegroupResult,
    export_g2_regroup,
    read_g2_regroup,
)
from xpcsviewer.gui.qt_compat import QtCore, QtWidgets, Signal, Slot
from xpcsviewer.threading.async_workers import BaseAsyncWorker

_HDF5_FILTER = "HDF5 files (*.h5 *.hdf *.hdf5);;All files (*)"


class _DelayAxis(pg.AxisItem):
    def logTickValues(
        self,
        minimum: float,
        maximum: float,
        size: float,
        standard_ticks: list[tuple[float, list[float]]],
    ) -> list[tuple[float | None, list[float]]]:
        ticks = super().logTickValues(minimum, maximum, size, standard_ticks)
        if any(minimum <= value <= maximum for _, values in ticks for value in values):
            return ticks

        # Place interior ticks when the visible range falls between standard log ticks.
        step = (maximum - minimum) / 4
        if step <= 0:
            return ticks
        return [(step, [minimum + i * step for i in (1, 2, 3)])]

    def logTickStrings(
        self, values: list[float], scale: float, spacing: float | None
    ) -> list[str]:
        if spacing is None or spacing >= 1:
            return super().logTickStrings(values, scale, spacing)
        precision = max(2, int(np.ceil(-np.log10(spacing))) + 1)
        return [f"{value:.{precision}g}" for value in np.power(10.0, values) * scale]


class _PreviewWorker(BaseAsyncWorker):
    def __init__(self, source: Path, qmap: Path) -> None:
        super().__init__(f"g2-regroup-preview-{uuid4().hex}")
        self.source = source
        self.qmap = qmap

    def do_work(self) -> tuple[str, G2RegroupResult]:
        self.check_cancelled()
        result = read_g2_regroup(
            self.source, self.qmap, check_cancelled=self.check_cancelled
        )
        return self.worker_id, result


class G2RegroupExportWorker(BaseAsyncWorker):
    """Publish a result and destination already accepted by the Viewer."""

    def __init__(self, result: G2RegroupResult, destination: Path) -> None:
        super().__init__()
        self.result = result
        self.destination = destination

    def do_work(self) -> Path:
        try:
            export_g2_regroup(self.result, self.destination)
        except InterruptedError as exc:
            raise OSError(str(exc)) from exc
        return self.destination


class G2RegroupDialog(QtWidgets.QDialog):
    """Preview normalized G2 for one pinned source and request an export."""

    save_requested = Signal(object, object)

    def __init__(
        self,
        source_path: Path,
        thread_pool: QtCore.QThreadPool,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.source_path = source_path
        self._thread_pool = thread_pool
        self._qmap_path: Path | None = None
        self._preview_worker: _PreviewWorker | None = None
        self._displayed_result: G2RegroupResult | None = None
        self._save_result: G2RegroupResult | None = None
        self._export_active = False
        self._choosing_destination = False
        self._retired = False

        self.setObjectName("g2_regroup_dialog")
        self.setWindowTitle("Regroup G2")
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(740, 520)

        layout = QtWidgets.QVBoxLayout(self)
        inputs = QtWidgets.QFormLayout()
        layout.addLayout(inputs)
        source = QtWidgets.QLabel(str(source_path))
        source.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        source.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        source.setWordWrap(True)
        inputs.addRow("Source", source)

        qmap_row = QtWidgets.QHBoxLayout()
        choose_map = QtWidgets.QPushButton("Choose Q-map…")
        choose_map.clicked.connect(self._choose_map)
        qmap_row.addWidget(choose_map)
        self._qmap_label = QtWidgets.QLabel("No Q-map selected")
        self._qmap_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self._qmap_label.setWordWrap(True)
        qmap_row.addWidget(self._qmap_label, 1)
        inputs.addRow("Q-map", qmap_row)

        delay_axis = _DelayAxis("bottom")
        self._plot = pg.PlotWidget(axisItems={"bottom": delay_axis})
        # Plot Options can transform markers independently of their error bars.
        self._plot.getPlotItem().setMenuEnabled(False, enableViewBoxMenu=True)
        self._plot.setLabel("bottom", "Delay", units="s")
        self._plot.setLabel("left", "Normalized G2")
        # Format ticks on the axis. Both plot items receive the same transformed x.
        delay_axis.setLogMode(True)
        delay_axis.setTickSpacing(5, 1)
        delay_axis.setStyle(textFillLimits=[(1, 0.28)])
        self._curve = self._plot.plot(pen=None, symbol="o", symbolSize=7)
        self._error_bars = pg.ErrorBarItem(pen=self._curve.opts["symbolPen"])
        self._plot.addItem(self._error_bars)
        layout.addWidget(self._plot, 1)

        self._status = QtWidgets.QLabel()
        self._status.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        self._export_status = QtWidgets.QLabel()
        self._export_status.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self._export_status.setWordWrap(True)
        self._export_status.hide()
        layout.addWidget(self._export_status)

        controls = QtWidgets.QHBoxLayout()
        self._bin_box = QtWidgets.QComboBox()
        self._bin_box.currentIndexChanged.connect(self._plot_bin)
        bin_label = QtWidgets.QLabel("&Dynamic bin")
        bin_label.setBuddy(self._bin_box)
        controls.addWidget(bin_label)
        controls.addWidget(self._bin_box)
        controls.addStretch()
        self._preview_button = QtWidgets.QPushButton("Preview")
        self._preview_button.clicked.connect(self._preview)
        controls.addWidget(self._preview_button)
        self._save_button = QtWidgets.QPushButton("Save as…")
        self._save_button.clicked.connect(self._choose_destination)
        controls.addWidget(self._save_button)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.reject)
        controls.addWidget(close)
        layout.addLayout(controls)
        self._update_controls()

    def _update_controls(self) -> None:
        self._bin_box.setEnabled(self._displayed_result is not None)
        self._preview_button.setEnabled(
            self._qmap_path is not None and self._preview_worker is None
        )
        self._save_button.setEnabled(
            self._save_result is not None
            and not self._export_active
            and not self._choosing_destination
        )

    @Slot()
    def _choose_map(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose an external Q-map", str(self.source_path.parent), _HDF5_FILTER
        )
        if self._retired or not filename:
            return

        self._cancel_preview()
        self._qmap_path = None
        qmap_path = Path(filename)
        self._qmap_label.setText(str(qmap_path))
        self._status.clear()
        self._update_controls()
        answer = QtWidgets.QMessageBox.question(
            self,
            "Confirm detector registration",
            "Does this Q-map match the source detector's pixel coordinates and "
            "orientation? Matching array shapes alone cannot establish alignment.",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if self._retired:
            return
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            self._qmap_path = qmap_path
            self._status.setText("Q-map registered. Select Preview to calculate G2.")
        else:
            self._status.setText("Choose a Q-map registered to the source detector.")
        self._update_controls()

    @Slot()
    def _preview(self) -> None:
        if self._qmap_path is None or self._preview_worker is not None:
            return
        self._save_result = None
        worker = _PreviewWorker(self.source_path, self._qmap_path)
        self._preview_worker = worker
        queued = QtCore.Qt.ConnectionType.QueuedConnection
        worker.signals.finished.connect(self._preview_finished, queued)
        worker.signals.error.connect(self._preview_error, queued)
        worker.signals.cancelled.connect(self._preview_cancelled, queued)
        self._status.setText("Calculating regrouped G2…")
        self._update_controls()
        self._thread_pool.start(worker)

    @Slot(object)
    def _preview_finished(self, completed: tuple[str, G2RegroupResult]) -> None:
        worker_id, result = completed
        if self._preview_worker is None or worker_id != self._preview_worker.worker_id:
            return
        self._preview_worker = None
        self._displayed_result = self._save_result = result
        self._bin_box.blockSignals(True)
        self._bin_box.clear()
        self._bin_box.addItems([str(bin_id) for bin_id in result.dynamic_bin_ids])
        self._bin_box.blockSignals(False)
        self._plot.setTitle(f"Q-map: {escape(result.qmap_path.name)}")
        self._plot_bin(0)
        self._status.clear()
        self._update_controls()

    @Slot(str, str, str, int)
    def _preview_error(
        self, worker_id: str, error: str, _traceback: str, _retry: int
    ) -> None:
        reason = error.removeprefix(f"Error in worker {worker_id}: ")
        self._preview_failed(worker_id, reason)

    @Slot(str, str)
    def _preview_cancelled(self, worker_id: str, reason: str) -> None:
        self._preview_failed(worker_id, reason)

    def _preview_failed(self, worker_id: str, reason: str) -> None:
        if self._preview_worker is None or worker_id != self._preview_worker.worker_id:
            return
        self._preview_worker = None
        self._status.setText(f"Preview failed. {reason}")
        self._update_controls()

    @Slot(int)
    def _plot_bin(self, column: int) -> None:
        result = self._displayed_result
        if result is None or column < 0:
            return
        positive = result.delay_seconds > 0
        x = np.log10(result.delay_seconds[positive])
        y = result.normalized_g2[positive, column]
        error = result.normalized_g2_err[positive, column]
        self._curve.setData(x, y)
        self._error_bars.setData(x=x, y=y, top=error, bottom=error)

    def set_export_active(self, active: bool, message: str = "") -> None:
        """Show the Viewer's export status in the modal dialog."""
        self._export_active = active
        if active and not message:
            message = "Saving regrouped G2…"
        self._export_status.setText(message)
        self._export_status.setVisible(bool(message))
        self._update_controls()

    @Slot()
    def _choose_destination(self) -> None:
        result = self._save_result
        if result is None or self._export_active or self._choosing_destination:
            return
        self._choosing_destination = True
        self._update_controls()
        suggestion = self.source_path.with_name(
            f"{self.source_path.stem}_regrouped_g2.h5"
        )
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save regrouped G2 to a new file",
            str(suggestion),
            _HDF5_FILTER,
            "",
            QtWidgets.QFileDialog.Option.DontConfirmOverwrite,
        )
        if self._retired:
            return
        self._choosing_destination = False
        if filename and self._save_result is result and not self._export_active:
            self.save_requested.emit(result, Path(filename))
        self._update_controls()

    def _cancel_preview(self) -> None:
        self._save_result = None
        if self._preview_worker is not None:
            self._preview_worker.cancel()
            self._preview_worker = None

    def done(self, result: int) -> None:
        """Retire preview work when the dialog is dismissed."""
        self._retired = True
        self._cancel_preview()
        super().done(result)
