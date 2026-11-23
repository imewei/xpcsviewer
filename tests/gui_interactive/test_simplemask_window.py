"""GUI tests for SimpleMask window integration.

Tests the SimpleMask window launch, data loading, and basic interactions
from the XPCS Viewer.
"""

from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from xpcsviewer.simplemask import SimpleMaskWindow, simplemask_kernel
from xpcsviewer.simplemask.reader.beamlines.aps_8idi import METADATA_KEYMAPS


@pytest.fixture
def partition_window(qtbot):
    window = SimpleMaskWindow()
    qtbot.addWidget(window)
    window.load_from_viewer(
        np.full((4, 4), 10.0),
        {
            "bcx": 1.501,
            "bcy": 1.5,
            "det_dist": 5000.0,
            "pix_dim": 0.075,
            "energy": 10.0,
        },
    )
    window.spin_dq_num.setValue(2)
    window.spin_sq_num.setValue(2)
    window.spin_dp_num.setValue(1)
    window.spin_sp_num.setValue(1)
    assert window.kernel is not None
    window.btn_compute_partition.click()
    assert window.kernel.new_partition is not None
    return window


def _write_raw_scan(path, image, distance=5.0):
    with h5py.File(path, "w") as file:
        file["/entry/data/data"] = image
        for field, value in {
            "energy": 10.0,
            "detector_distance": distance,
            "x_pixel_size": 75e-6,
            "y_pixel_size": 75e-6,
            "ccdx": 0.0,
            "ccdy": 0.0,
            "ccdx0": 0.0,
            "ccdy0": 0.0,
            "bcx0": 1.5,
            "bcy0": 1.5,
        }.items():
            file[METADATA_KEYMAPS[field]] = value
    return str(path)


@pytest.fixture
def raw_editor(qtbot, tmp_path):
    window = SimpleMaskWindow()
    qtbot.addWidget(window)
    workers = []
    window.thread_pool = SimpleNamespace(start=workers.append)
    source = _write_raw_scan(tmp_path / "first.h5", np.full((2, 4, 4), 11))
    window.load_from_raw_file(source, "APS_8IDI")
    workers.pop().run()
    return window, workers, source


def _mask_one_pixel(window):
    invalid = np.zeros((4, 4), dtype=bool)
    invalid[0, 1] = True
    window.kernel.mask_kernel.evaluate("mask_draw", arr=invalid)
    window.kernel.mask_apply("mask_draw")
    assert not window.kernel.mask[0, 1]
    window.mark_unsaved()


@pytest.mark.parametrize(("mode", "expected"), [(0, (0, -1)), (1, (0, 0)), (2, (2, 0))])
def test_open_raw_range_picker_and_sidecar_filter(
    qtbot, monkeypatch, tmp_path, mode, expected
):
    window = SimpleMaskWindow()
    qtbot.addWidget(window)
    sidecar = str(tmp_path / "scan_metadata.hdf")
    unsupported = str(tmp_path / "scan.raw")
    supported = str(tmp_path / "scan.H5")
    for filename in (sidecar, unsupported, supported):
        with open(filename, "wb"):
            pass
    subdirectory = tmp_path / "subdirectory"
    subdirectory.mkdir()

    def choose_file(dialog):
        assert dialog.testOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)
        proxy = dialog.proxyModel()
        model = proxy.sourceModel()
        dialog.setDirectory(str(tmp_path))
        qtbot.waitUntil(lambda: model.index(supported).isValid())
        assert proxy.mapFromSource(model.index(supported)).isValid()
        assert proxy.mapFromSource(model.index(str(subdirectory))).isValid()
        assert not proxy.mapFromSource(model.index(sidecar)).isValid()
        assert not proxy.mapFromSource(model.index(unsupported)).isValid()
        dialog.selectFile(supported)
        return QtWidgets.QDialog.DialogCode.Accepted

    def choose_range(dialog):
        choices = dialog.findChild(QtWidgets.QComboBox)
        assert [choices.itemText(i) for i in range(choices.count())] == [
            "Representative sample",
            "All frames",
            "Custom",
        ]
        assert choices.currentIndex() == 0
        choices.setCurrentIndex(mode)
        start, count = dialog.findChildren(QtWidgets.QSpinBox)
        assert start.isEnabled() == count.isEnabled() == (mode == 2)
        assert start.minimum() == count.minimum() == 0
        if mode == 2:
            start.setValue(2)
            count.setValue(0)
        return QtWidgets.QDialog.DialogCode.Accepted

    workers = []
    window.thread_pool = SimpleNamespace(start=workers.append)
    for invalid in (sidecar, unsupported):
        window.load_from_raw_file(invalid, "APS_8IDI")
        assert not workers
        assert "Failed to load raw file:" in window.status_bar.currentMessage()

    monkeypatch.setattr(QtWidgets.QFileDialog, "exec", choose_file)
    monkeypatch.setattr(QtWidgets.QDialog, "exec", choose_range)
    with patch.object(window, "load_from_raw_file") as submit:
        window._on_open_raw_file()
    submit.assert_called_once_with(
        supported, "APS_8IDI", begin_idx=expected[0], num_frames=expected[1]
    )


def test_open_failure_or_decline_preserves_unsaved_mask(
    raw_editor, monkeypatch, tmp_path
):
    window, workers, source = raw_editor
    _mask_one_pixel(window)
    old_mask = window.kernel.mask.copy()
    candidate = _write_raw_scan(tmp_path / "second.h5", np.full((4, 4), 22))
    no = QtWidgets.QMessageBox.StandardButton.No
    yes = QtWidgets.QMessageBox.StandardButton.Yes
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args: no)
    window.load_from_raw_file(candidate, "APS_8IDI")
    assert not workers

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args: yes)
    window.load_from_raw_file(candidate, "APS_8IDI")
    worker = workers.pop()
    assert not window.action_load_mask.isEnabled()
    window._on_mask_action("reset")
    np.testing.assert_array_equal(window.kernel.mask, old_mask)
    worker.signals.error.emit(worker.worker_id, "simulated read error", "", 0)
    assert window.has_unsaved_changes()
    assert window._raw_source_path == source
    assert window._raw_source_frame_count == 2
    np.testing.assert_array_equal(window.kernel.detector_image, np.full((4, 4), 11))
    np.testing.assert_array_equal(window.kernel.mask, old_mask)
    assert window.action_load_mask.isEnabled()

    window.load_from_raw_file(candidate, "APS_8IDI")

    def unexpected_qmap(*args):
        pytest.fail("the window must install the worker's Q-map")

    monkeypatch.setattr(simplemask_kernel, "compute_qmap", unexpected_qmap)
    workers.pop().run()
    assert not window.has_unsaved_changes()
    assert not window.windowTitle().endswith("*")
    assert window._raw_source_path == candidate
    assert window._raw_source_frame_count is None
    np.testing.assert_array_equal(window.kernel.detector_image, np.full((4, 4), 22))
    assert np.all(window.kernel.mask)


def test_raw_open_invalid_geometry_leaves_previous_editor_state(
    raw_editor, monkeypatch, tmp_path
):
    window, workers, source = raw_editor
    _mask_one_pixel(window)
    window.btn_compute_partition.click()
    kernel = window.kernel
    image, mask, metadata, partition = (
        kernel.detector_image,
        kernel.mask,
        kernel.metadata,
        kernel.new_partition,
    )
    assert partition is not None
    candidate = _write_raw_scan(tmp_path / "bad.h5", np.full((4, 4), 22), distance=0.0)
    yes = QtWidgets.QMessageBox.StandardButton.Yes
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args: yes)

    window.load_from_raw_file(candidate, "APS_8IDI")
    workers.pop().run()

    assert kernel.detector_image is image
    assert kernel.mask is mask
    assert kernel.metadata is metadata
    assert kernel.new_partition is partition
    assert window._raw_source_path == source
    assert window.has_unsaved_changes()
    assert window.action_open_raw_file.isEnabled()
    assert window.action_load_mask.isEnabled()
    assert window.btn_export_partition.isEnabled()
    assert "Failed to load raw file:" in window.status_bar.currentMessage()
    assert "detector_distance" in window.status_bar.currentMessage()


@pytest.mark.parametrize("terminal", ["finished", "error", "cancelled"])
def test_raw_open_ignores_superseded_terminal_signals(raw_editor, tmp_path, terminal):
    window, workers, source = raw_editor
    candidate = _write_raw_scan(tmp_path / "second.h5", np.full((4, 4), 22))
    viewer_metadata = window.kernel.metadata.copy()
    viewer_image = np.full((4, 4), 33)
    window.load_from_raw_file(candidate, "APS_8IDI")
    stale = workers.pop()
    result = stale.do_work()
    window.load_from_viewer(viewer_image, viewer_metadata)
    assert stale.is_cancelled
    assert window._raw_source_path is None
    assert window.action_open_raw_file.isEnabled()
    window.load_from_raw_file(source, "APS_8IDI")
    current = workers.pop()
    status = window.status_bar.currentMessage()

    if terminal == "finished":
        stale.signals.finished.emit(result)
    elif terminal == "error":
        stale.signals.error.emit(stale.worker_id, "stale error", "", 0)
    else:
        stale.signals.cancelled.emit(stale.worker_id, "stale cancellation")

    np.testing.assert_array_equal(window.kernel.detector_image, viewer_image)
    assert window._raw_file_worker is current
    assert not window.action_open_raw_file.isEnabled()
    assert window.status_bar.currentMessage() == status
    current.signals.cancelled.emit(current.worker_id, "cancelled")
    assert window.action_open_raw_file.isEnabled()


def test_raw_open_close_retires_request(raw_editor, tmp_path):
    window, workers, source = raw_editor
    candidate = _write_raw_scan(tmp_path / "second.h5", np.full((4, 4), 22))
    window.load_from_raw_file(candidate, "APS_8IDI")
    pending = workers.pop()
    result = pending.do_work()

    window.close()
    pending.signals.finished.emit(result)

    assert pending.is_cancelled
    assert window._raw_file_worker is None
    assert window._raw_source_path == source
    np.testing.assert_array_equal(window.kernel.detector_image, np.full((4, 4), 11))


@pytest.mark.parametrize("replace_with_viewer", [False, True], ids=["raw", "viewer"])
def test_raw_open_overlay_close_leaves_coherent_editor_on_reshow(
    raw_editor, tmp_path, replace_with_viewer
):
    window, workers, _ = raw_editor
    window.show()
    window._ensure_mask_overlay_visible()
    image = np.full((3, 5), 22, dtype=np.float32)
    candidate = _write_raw_scan(tmp_path / "second.h5", image)
    viewer_image = np.full((2, 6), 33.0)
    metadata = window.kernel.metadata.copy()
    visible_during_install = []
    visible_on_viewer_status = []

    def record_viewer_status(message):
        if message == "Loaded 6x2 detector image":
            visible_on_viewer_status.append(window.isVisible())

    def close_on_overlay_change(checked):
        window.action_toggle_mask.toggled.disconnect(close_on_overlay_change)
        window.close()
        visible_during_install.append(window.isVisible())
        if replace_with_viewer:
            window.load_from_viewer(viewer_image, metadata)

    window.status_bar.messageChanged.connect(record_viewer_status)
    window.action_toggle_mask.toggled.connect(close_on_overlay_change)
    window.load_from_raw_file(candidate, "APS_8IDI")
    workers.pop().run()

    assert not window.isVisible()
    window.show()
    assert window._raw_source_path == (None if replace_with_viewer else candidate)
    assert window._raw_source_frame_count is None
    installed_image = viewer_image if replace_with_viewer else image
    np.testing.assert_array_equal(window.kernel.detector_image, installed_image)
    np.testing.assert_array_equal(
        window.image_view.image, np.log10(installed_image + 1)
    )
    expected_status = (
        "Loaded 6x2 detector image" if replace_with_viewer else "Raw file loaded"
    )
    assert window.status_bar.currentMessage() == expected_status
    assert window.action_open_raw_file.isEnabled()
    assert visible_during_install == [True]
    assert visible_on_viewer_status == ([True] if replace_with_viewer else [])


@pytest.mark.parametrize("notification", ["title", "overlay", "enabled"])
def test_raw_open_notifications_can_install_viewer_image(
    raw_editor, monkeypatch, tmp_path, notification
):
    window, workers, _ = raw_editor
    _mask_one_pixel(window)
    candidate = _write_raw_scan(tmp_path / "second.h5", np.full((4, 4), 22))
    viewer_image = np.full((3, 5), 33.0)
    viewer_metadata = window.kernel.metadata.copy()
    yes = QtWidgets.QMessageBox.StandardButton.Yes
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args: yes)
    window._ensure_mask_overlay_visible()
    window.load_from_raw_file(candidate, "APS_8IDI")
    signal = {
        "title": window.windowTitleChanged,
        "overlay": window.action_toggle_mask.toggled,
        "enabled": window.action_open_raw_file.changed,
    }[notification]
    messages = []
    window.status_bar.messageChanged.connect(messages.append)

    def load_viewer_image(*args):
        signal.disconnect(load_viewer_image)
        window.load_from_viewer(np.full((2, 2), 7.0), viewer_metadata)
        window.load_from_viewer(viewer_image, viewer_metadata)

    signal.connect(load_viewer_image)
    workers.pop().run()

    np.testing.assert_array_equal(window.kernel.detector_image, viewer_image)
    np.testing.assert_array_equal(window.image_view.image, np.log10(viewer_image + 1))
    assert window._raw_source_path is None
    assert window._raw_source_frame_count is None
    assert "Raw file loaded" not in messages
    assert "Loaded 2x2 detector image" not in messages
    assert window.status_bar.currentMessage() == "Loaded 5x3 detector image"
    assert window.info_label.text().startswith("Image: 5 x 3 pixels")
    assert window.action_open_raw_file.isEnabled()
    assert window.action_load_mask.isEnabled()


def test_raw_open_error_release_can_start_new_request(raw_editor, tmp_path):
    window, workers, source = raw_editor
    failed = str(tmp_path / "failed.h5")
    window.load_from_raw_file(failed, "APS_8IDI")
    stale = workers.pop()

    def open_new_raw_file():
        if window.action_open_raw_file.isEnabled():
            window.action_open_raw_file.changed.disconnect(open_new_raw_file)
            window.load_from_raw_file(source, "APS_8IDI")

    window.action_open_raw_file.changed.connect(open_new_raw_file)
    stale.signals.error.emit(stale.worker_id, "read error", "", 0)

    assert window._raw_file_worker is workers.pop()
    assert not window.action_open_raw_file.isEnabled()
    assert not window.action_load_mask.isEnabled()
    assert not window.action_apply.isEnabled()
    assert window.status_bar.currentMessage() == "Loading first.h5..."


def test_raw_open_recognizes_real_unsaved_mask_edits(raw_editor, monkeypatch, tmp_path):
    window, workers, source = raw_editor
    mask_path = str(tmp_path / "mask.h5")
    mask = np.ones((4, 4), dtype=np.uint8)
    mask[0, 1] = 0
    with h5py.File(mask_path, "w") as file:
        file["mask"] = mask
    questions = []
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getOpenFileName", lambda *args: (mask_path, "")
    )

    def decline(*args):
        questions.append(args)
        return QtWidgets.QMessageBox.StandardButton.No

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", decline)
    for edit in (
        window._on_load_mask,
        lambda: window._on_mask_action("undo"),
        lambda: window._on_mask_action("redo"),
        lambda: window._on_mask_action("reset"),
    ):
        window.mark_saved()
        edit()
        assert window.has_unsaved_changes()
        window.load_from_raw_file(source, "APS_8IDI")
        assert not workers
    assert len(questions) == 4
    window.mark_saved()
    window._on_mask_action("undo")
    assert not window.has_unsaved_changes()
    window.load_from_raw_file(source, "APS_8IDI")
    assert len(questions) == 4
    assert len(workers) == 1


def test_partition_controls_follow_freshness(partition_window):
    window = partition_window
    kernel = window.kernel
    received = []
    window.qmap_exported.connect(received.append)
    window.btn_toggle_partition.click()
    assert window.btn_toggle_partition.isChecked()

    window.spin_dq_num.setValue(3)
    assert kernel.new_partition is None
    assert not window.btn_toggle_partition.isChecked()
    assert not window.btn_export_partition.isEnabled()
    window.export_partition_to_viewer()
    assert received == []

    window.btn_compute_partition.click()
    assert window.btn_export_partition.isEnabled()
    window.export_partition_to_viewer()
    assert len(received) == 1
    assert list(received[0]["dynamic_num_pts"]) == [3, 1]


def test_partition_unavailable_after_viewer_clears_image(partition_window, tmp_path):
    window = partition_window
    kernel = window.kernel
    metadata = kernel.metadata.copy()
    received = []
    window.qmap_exported.connect(received.append)
    window.btn_toggle_partition.click()
    window.export_partition_to_viewer()
    assert window.btn_toggle_partition.isChecked()
    assert len(received) == 1
    destination = tmp_path / "partition.h5"
    destination.write_bytes(b"original")

    window.load_from_viewer(None, None)

    assert not kernel.is_ready()
    assert kernel.new_partition is None
    assert kernel.compute_partition() is None
    kernel.save_partition(str(destination))
    assert destination.read_bytes() == b"original"
    assert window.image_view.image is None
    assert not window.btn_toggle_partition.isChecked()
    assert not window.btn_toggle_partition.isEnabled()
    assert not window.btn_export_partition.isEnabled()
    window.export_partition_to_viewer()
    assert len(received) == 1

    window.load_from_viewer(np.full((4, 4), 20.0), metadata)
    assert window.kernel is kernel
    window.btn_compute_partition.click()
    window.export_partition_to_viewer()
    assert len(received) == 2


def test_pending_invalid_geometry_blocks_partition_export(
    partition_window, monkeypatch, tmp_path
):
    window = partition_window
    kernel = window.kernel
    original_image, original_qmap = kernel.detector_image, kernel.qmap
    received = []
    window.qmap_exported.connect(received.append)
    destination = tmp_path / "partition.h5"
    destination.write_bytes(b"original")

    def invalid_geometry(*args, **kwargs):
        raise ValueError("invalid detector distance")

    window.spin_det_dist.setValue(6000.0)
    with monkeypatch.context() as patch_geometry:
        patch_geometry.setattr(simplemask_kernel, "compute_qmap", invalid_geometry)
        patch_geometry.setattr(QtWidgets.QMessageBox, "warning", lambda *a: None)
        window.action_generate_qmap.trigger()

    assert kernel.detector_image is original_image
    assert kernel.qmap is original_qmap
    assert kernel.metadata["det_dist"] == 5000.0
    assert kernel.new_partition is None
    window.export_partition_to_viewer()
    assert received == []
    assert not window.btn_find_beam_center.isEnabled()
    window.btn_compute_partition.click()
    assert kernel.new_partition is None
    assert kernel.compute_partition() is None
    kernel.save_partition(str(destination))
    assert destination.read_bytes() == b"original"

    window.spin_det_dist.setValue(5000.0)
    assert window.btn_find_beam_center.isEnabled()
    window.btn_compute_partition.click()
    assert kernel.new_partition is not None
    assert kernel.metadata["bcx"] == 1.501

    window.spin_det_dist.setValue(5500.0)
    compute_qmap = simplemask_kernel.compute_qmap
    proposed_distances = []

    def record_geometry(scattering_type, metadata):
        proposed_distances.append(metadata["det_dist"])
        return compute_qmap(scattering_type, metadata)

    monkeypatch.setattr(simplemask_kernel, "compute_qmap", record_geometry)
    window.action_generate_qmap.trigger()
    assert proposed_distances == [5500.0]
    window.btn_compute_partition.click()
    assert kernel.new_partition["detector_distance"] == 5500.0
    assert kernel.new_partition["beam_center_x"] == 1.501


@pytest.mark.parametrize("edit_bins", [True, False], ids=["edited-bins", "same-bins"])
def test_failed_partition_compute_cannot_export_previous_result(
    partition_window, monkeypatch, edit_bins
):
    window = partition_window
    kernel = window.kernel
    partition = kernel.new_partition
    received = []
    window.qmap_exported.connect(received.append)

    kernel.mask_kernel.evaluate("mask_draw", arr=np.zeros((4, 4), dtype=bool))
    kernel.mask_apply("mask_draw")
    assert kernel.new_partition is partition

    def invalid_partition(*args, **kwargs):
        raise ValueError("invalid partition")

    if edit_bins:
        window.spin_dq_num.setValue(3)
    monkeypatch.setattr(simplemask_kernel, "generate_partition", invalid_partition)
    window._on_compute_partition()
    assert kernel.new_partition is None
    assert not window.btn_export_partition.isEnabled()
    window.export_partition_to_viewer()
    assert received == []


class TestSimpleMaskWindowCreation:
    """Tests for SimpleMask window creation and initialization."""

    def test_window_creation(self, qapp, qtbot):
        """SimpleMask window should be created successfully."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        assert window is not None
        assert window.windowTitle() == "Mask Editor"

    def test_window_minimum_size(self, qapp, qtbot):
        """SimpleMask window should have minimum size set."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)
        window.show()
        qtbot.wait(50)

        assert window.minimumWidth() >= 800
        assert window.minimumHeight() >= 600

    def test_window_has_image_view(self, qapp, qtbot):
        """SimpleMask window should have ImageView widget."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        assert hasattr(window, "image_view")
        assert window.image_view is not None

    def test_window_has_status_bar(self, qapp, qtbot):
        """SimpleMask window should have status bar."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        assert hasattr(window, "status_bar")
        assert window.status_bar is not None


class TestSimpleMaskDataLoading:
    """Tests for loading detector data into SimpleMask."""

    def test_load_none_data(self, qapp, qtbot):
        """Loading None should show empty canvas message."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)
        window.show()
        qtbot.wait(50)

        window.load_from_viewer(None, None)

        # Check info label contains message about no data
        assert (
            "No data" in window.info_label.text()
            or "no" in window.info_label.text().lower()
        )

    def test_load_valid_detector_image(self, qapp, qtbot):
        """Loading valid detector image should display it."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)
        window.show()
        qtbot.wait(50)

        # Create synthetic detector image
        detector_image = np.random.poisson(100, (512, 1024)).astype(np.float64)
        metadata = {
            "bcx": 512,
            "bcy": 256,
            "det_dist": 5000.0,
            "pix_dim": 0.075,
            "energy": 10.0,
            "shape": (512, 1024),
        }

        window.load_from_viewer(detector_image, metadata)

        # Check that kernel was initialized
        assert window.kernel is not None
        # Check that image was loaded
        assert window._detector_image is not None
        assert window._detector_image.shape == (512, 1024)

    def test_load_updates_info_label(self, qapp, qtbot):
        """Loading data should update info label with image info."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)
        window.show()
        qtbot.wait(50)

        detector_image = np.ones((256, 512), dtype=np.float64) * 100
        metadata = {
            "bcx": 256,
            "bcy": 128,
            "det_dist": 5000.0,
            "pix_dim": 0.075,
            "energy": 10.0,
            "shape": (256, 512),
        }

        window.load_from_viewer(detector_image, metadata)

        # Info label should contain dimensions
        info_text = window.info_label.text()
        assert "512" in info_text  # width
        assert "256" in info_text  # height


class TestSimpleMaskUnsavedChanges:
    """Tests for unsaved changes tracking."""

    def test_initial_no_unsaved_changes(self, qapp, qtbot):
        """New window should have no unsaved changes."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        assert not window.has_unsaved_changes()

    def test_mark_unsaved_changes(self, qapp, qtbot):
        """Marking unsaved should update state and title."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        window.mark_unsaved()

        assert window.has_unsaved_changes()
        assert window.windowTitle().endswith("*")

    def test_mark_saved_clears_flag(self, qapp, qtbot):
        """Marking saved should clear unsaved state."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        window.mark_unsaved()
        window.mark_saved()

        assert not window.has_unsaved_changes()
        assert not window.windowTitle().endswith("*")


class TestSimpleMaskSignals:
    """Tests for SimpleMask signal emission."""

    def test_mask_exported_signal_exists(self, qapp, qtbot):
        """mask_exported signal should exist."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        assert hasattr(window, "mask_exported")

    def test_qmap_exported_signal_exists(self, qapp, qtbot):
        """qmap_exported signal should exist."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        assert hasattr(window, "qmap_exported")


class TestSimpleMaskFromViewer:
    """Tests for launching SimpleMask from XPCS Viewer."""

    def test_mask_editor_action_exists(self, gui_main_window, qtbot):
        """Mask Editor should be accessible from XPCS Viewer toolbar."""
        # Mask Editor is a toolbar action, not a tab — verify the method exists
        assert hasattr(gui_main_window, "open_simplemask"), (
            "open_simplemask method not found on XpcsViewer"
        )
        assert callable(gui_main_window.open_simplemask)

    def test_open_simplemask_creates_window(self, gui_main_window, qtbot):
        """open_simplemask should create SimpleMask window."""
        # Ensure no window exists
        gui_main_window._simplemask_window = None

        # Patch _get_simplemask_data to avoid Mock geometry values hitting qmap
        with patch.object(
            gui_main_window, "_get_simplemask_data", return_value=(None, None)
        ):
            gui_main_window.open_simplemask()
            qtbot.wait(100)

        # Check window was created
        assert gui_main_window._simplemask_window is not None
        assert isinstance(gui_main_window._simplemask_window, SimpleMaskWindow)

    def test_open_simplemask_shows_window(self, gui_main_window, qtbot):
        """open_simplemask should show the window."""
        gui_main_window._simplemask_window = None

        with patch.object(
            gui_main_window, "_get_simplemask_data", return_value=(None, None)
        ):
            gui_main_window.open_simplemask()
            qtbot.wait(100)

        assert gui_main_window._simplemask_window.isVisible()

    def test_open_simplemask_reuses_window(self, gui_main_window, qtbot):
        """Calling open_simplemask twice should reuse the window."""
        gui_main_window._simplemask_window = None

        with patch.object(
            gui_main_window, "_get_simplemask_data", return_value=(None, None)
        ):
            gui_main_window.open_simplemask()
            qtbot.wait(50)
            first_window = gui_main_window._simplemask_window

            gui_main_window.open_simplemask()
            qtbot.wait(50)
            second_window = gui_main_window._simplemask_window

        assert first_window is second_window

    def test_simplemask_window_parent_reference(self, gui_main_window, qtbot):
        """SimpleMask window should have reference to parent viewer."""
        gui_main_window._simplemask_window = None

        with patch.object(
            gui_main_window, "_get_simplemask_data", return_value=(None, None)
        ):
            gui_main_window.open_simplemask()
            qtbot.wait(100)

        assert gui_main_window._simplemask_window.parent_viewer is gui_main_window


class TestSimpleMaskCloseEvent:
    """Tests for SimpleMask window close behavior."""

    def test_close_without_unsaved_changes(self, qapp, qtbot):
        """Window should close without prompt when no unsaved changes."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)
        window.show()
        qtbot.wait(50)

        # Close should succeed without prompt
        window.close()
        qtbot.wait(50)

        assert not window.isVisible()


class TestSimpleMaskOpenRawFile:
    """Tests for the standalone raw-file open flow (Open Raw File action)."""

    def test_action_open_raw_file_exists(self, qapp, qtbot):
        window = SimpleMaskWindow()
        qtbot.addWidget(window)
        assert hasattr(window, "action_open_raw_file")
        assert window.action_open_raw_file.isEnabled()

    def test_successful_load_populates_kernel_and_reenables_action(
        self, qapp, qtbot, tmp_path
    ):
        fname = _write_raw_scan(
            tmp_path / "scan.h5", np.full((2, 3, 3), 5, dtype=np.uint16)
        )

        window = SimpleMaskWindow()
        qtbot.addWidget(window)
        window.show()

        window.load_from_raw_file(fname, "APS_8IDI")

        # Poll until the worker finishes (real QThreadPool background thread).
        for _ in range(200):
            if window.kernel is not None and window.kernel.is_ready():
                break
            qtbot.wait(25)

        assert window.kernel is not None
        assert window.kernel.is_ready()
        assert window.action_open_raw_file.isEnabled()

    def test_failed_load_shows_error_and_reenables_action(self, qapp, qtbot, tmp_path):
        fname = str(tmp_path / "not_hdf5.txt")
        with open(fname, "w") as f:
            f.write("not an hdf5 file")

        window = SimpleMaskWindow()
        qtbot.addWidget(window)
        window.show()

        window.load_from_raw_file(fname, "APS_8IDI")

        for _ in range(200):
            if window.action_open_raw_file.isEnabled():
                break
            qtbot.wait(25)

        assert window.action_open_raw_file.isEnabled()
        assert window.kernel is None
        assert "Failed to load raw file:" in window.status_bar.currentMessage()

    def test_placeholder_metadata_shows_warning_status_message(
        self, qapp, qtbot, tmp_path
    ):
        # No NeXus metadata fields present -- FileReader falls back to
        # placeholder geometry, and _on_raw_file_loaded must surface that
        # via a distinct status-bar warning rather than the plain
        # "Raw file loaded" message.
        fname = str(tmp_path / "no_metadata.h5")
        with h5py.File(fname, "w") as f:
            f.create_dataset(
                "/entry/data/data", data=np.full((2, 3, 3), 5, dtype=np.uint16)
            )

        window = SimpleMaskWindow()
        qtbot.addWidget(window)
        window.show()

        window.load_from_raw_file(fname, "APS_8IDI")

        for _ in range(200):
            if window.kernel is not None and window.kernel.is_ready():
                break
            qtbot.wait(25)

        assert window.kernel is not None
        assert "placeholder geometry" in window.status_bar.currentMessage()

    def test_repeated_click_ignores_stale_completion(self, qapp, qtbot, tmp_path):
        # Two rapid loads: the second call must be a no-op while the first
        # is still in flight -- not queue a second worker whose completion
        # could race the first and silently win regardless of click order.
        fname = str(tmp_path / "scan.h5")
        with h5py.File(fname, "w") as f:
            f.create_dataset(
                "/entry/data/data", data=np.full((3, 3), 4, dtype=np.uint16)
            )

        window = SimpleMaskWindow()
        qtbot.addWidget(window)
        window.show()

        window.load_from_raw_file(fname, "APS_8IDI")
        assert window.action_open_raw_file.isEnabled() is False
        first_worker = window._raw_file_worker

        # Second call while the first is still in flight must be a no-op:
        # no new worker created, no crash, no double-connected signals.
        window.load_from_raw_file(fname, "APS_8IDI")
        assert window._raw_file_worker is first_worker

        for _ in range(200):
            if window.action_open_raw_file.isEnabled():
                break
            qtbot.wait(25)

        assert window.action_open_raw_file.isEnabled()
