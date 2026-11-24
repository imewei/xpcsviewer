"""Exercise the desktop G2 regroup workflow with temporary APS inputs."""

from shutil import copyfile
from threading import Event

import h5py
import numpy as np
import pytest
from pyqtgraph import ErrorBarItem, PlotWidget

from xpcsviewer.fileIO import g2_regroup
from xpcsviewer.gui.qt_compat import QtCore, QtGui, QtWidgets
from xpcsviewer.viewer_kernel import ViewerKernel
from xpcsviewer.xpcs_viewer import XpcsViewer

pytestmark = pytest.mark.gui


@pytest.fixture
def input_paths(tmp_path):
    source = tmp_path / "nested" / "source.h5"
    source.parent.mkdir()
    channels = np.array([[[4, 12, 24, 10]], [[1, 2, 2, 1]], [[2, 3, 2, 5]]])
    with h5py.File(source, "w") as handle:
        handle["/xpcs/multitau/unnormalized_G2"] = [channels] * 3
        handle["/xpcs/multitau/delay_list"] = [0, 2, 4]
        handle["/entry/instrument/detector_1/frame_time"] = 0.25
        handle["/xpcs/multitau/config/stride_frame"] = 2
        handle["/xpcs/multitau/config/avg_frame"] = 3

    qmap = tmp_path / "qmap.h5"
    with h5py.File(qmap, "w") as handle:
        handle["/qmap/static_roi_map"] = [[1, 1, 2, 3]]
        handle["/qmap/dynamic_roi_map"] = [[1, 1, 1, 2]]
    return source, qmap


@pytest.fixture
def viewer(qtbot, monkeypatch, tmp_path, input_paths):
    monkeypatch.setattr(XpcsViewer, "_restore_session", lambda _window: None)
    window = XpcsViewer()
    qtbot.addWidget(window)
    monkeypatch.setattr(window.session_manager, "save_session", lambda _session: None)
    monkeypatch.setattr(window.recent_paths_manager, "add_path", lambda _path: None)
    window.vk = ViewerKernel(str(tmp_path))
    window.vk.target.append(str(input_paths[0].relative_to(tmp_path)))
    window.update_box(window.vk.target, mode="target")
    window.show()
    return window


def _regroup_action(viewer):
    file_menu = viewer.menuBar().actions()[0].menu()
    actions = [a for a in file_menu.actions() if a.text() == "Regroup G2…"]
    assert len(actions) == 1
    return actions[0]


def _open_regroup(viewer):
    viewer.list_view_target.setCurrentIndex(viewer.vk.target.index(0, 0))
    _regroup_action(viewer).trigger()
    dialog = viewer.findChild(QtWidgets.QDialog, "g2_regroup_dialog")
    assert dialog is not None and dialog.isVisible()
    return dialog


def _button(dialog, text):
    return next(
        button
        for button in dialog.findChildren(QtWidgets.QPushButton)
        if button.text() == text
    )


def _choose_map(dialog, monkeypatch, path, *, answer=QtWidgets.QMessageBox.Yes):
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getOpenFileName", lambda *_args: (str(path), "")
    )
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *_args: answer)
    _button(dialog, "Choose Q-map…").click()


def _curve(dialog):
    return dialog.findChild(PlotWidget).getPlotItem().listDataItems()[0]


def _preview(dialog, monkeypatch, qmap, qtbot):
    _choose_map(dialog, monkeypatch, qmap)
    _button(dialog, "Preview").click()
    qtbot.waitUntil(_button(dialog, "Save as…").isEnabled)


def test_regroup_action_requires_one_explicit_target(viewer, input_paths):
    action = _regroup_action(viewer)
    action.trigger()
    assert viewer.findChild(QtWidgets.QDialog, "g2_regroup_dialog") is None

    viewer.vk.target.append("other.h5")
    viewer.list_view_target.selectAll()
    action.trigger()
    assert viewer.findChild(QtWidgets.QDialog, "g2_regroup_dialog") is None

    dialog = _open_regroup(viewer)
    assert dialog.windowModality() == QtCore.Qt.WindowModality.WindowModal
    assert dialog.source_path == input_paths[0].resolve()


def test_regroup_requires_registration_and_plots_positive_delays(
    viewer, input_paths, monkeypatch, qtbot
):
    jobs = []
    monkeypatch.setattr(viewer.thread_pool, "start", jobs.append)
    dialog = _open_regroup(viewer)
    preview = _button(dialog, "Preview")
    assert not preview.isEnabled()
    _choose_map(dialog, monkeypatch, "")
    preview.click()
    _choose_map(dialog, monkeypatch, input_paths[1], answer=QtWidgets.QMessageBox.No)
    preview.click()
    assert not jobs

    _choose_map(dialog, monkeypatch, input_paths[1])
    preview.click()
    assert len(jobs) == 1
    jobs[0].run()
    qtbot.waitUntil(_button(dialog, "Save as…").isEnabled)

    combo = dialog.findChild(QtWidgets.QComboBox)
    assert [combo.itemText(i) for i in range(combo.count())] == ["1", "2"]
    plot = dialog.findChild(PlotWidget).getPlotItem()
    errors = next(item for item in plot.items if isinstance(item, ErrorBarItem))
    x, y = _curve(dialog).getData()
    np.testing.assert_allclose(10**x, [3, 6])
    np.testing.assert_array_equal(x, np.log10([3.0, 6.0]))
    np.testing.assert_array_equal(errors.opts["x"], x)
    np.testing.assert_allclose(y, [61 / 15, 61 / 15])
    np.testing.assert_allclose(errors.opts["y"], y)
    np.testing.assert_allclose(errors.opts["top"], [4 * np.sqrt(6) / 9] * 2)
    np.testing.assert_array_equal(errors.opts["bottom"], errors.opts["top"])
    assert plot.getAxis("bottom").logMode
    assert not plot.getAxis("left").logMode

    combo.setCurrentIndex(1)
    np.testing.assert_allclose(_curve(dialog).getData()[1], [2, 2])
    np.testing.assert_array_equal(errors.opts["top"], [0, 0])
    assert len(jobs) == 1


def test_regroup_plot_menu_keeps_markers_and_errors_aligned(
    viewer, input_paths, monkeypatch, qtbot
):
    with h5py.File(input_paths[0], "r+") as source:
        group = source["/xpcs/multitau"]
        del group["delay_list"]
        group["delay_list"] = np.array([0.1, 1, 10]) / 1.5

    dialog = _open_regroup(viewer)
    _preview(dialog, monkeypatch, input_paths[1], qtbot)
    widget = dialog.findChild(PlotWidget)
    plot = widget.getPlotItem()
    view = plot.getViewBox()
    menu = view.getMenu(None)
    errors = next(item for item in plot.items if isinstance(item, ErrorBarItem))
    qtbot.mouseClick(
        widget.viewport(),
        QtCore.Qt.MouseButton.RightButton,
        pos=widget.mapFromScene(view.sceneBoundingRect().center()),
    )
    qtbot.waitUntil(menu.isVisible)
    try:
        actions = {action.text(): action for action in menu.actions()}
        assert "View All" in actions
        options = actions.get("Plot Options")
        if options is not None:
            options.menu().findChild(QtWidgets.QCheckBox, "logXCheck").click()
        actions["View All"].trigger()
        np.testing.assert_allclose(_curve(dialog).getData()[0], [-1, 0, 1])
        np.testing.assert_array_equal(_curve(dialog).getData()[0], errors.opts["x"])
        assert options is None
    finally:
        menu.close()


@pytest.mark.parametrize("delays", [[0, 3.2, 3.3], [0, 0, 3.25]])
def test_regroup_labels_narrow_delay_ranges(
    viewer, input_paths, monkeypatch, qtbot, delays
):
    with h5py.File(input_paths[0], "r+") as source:
        group = source["/xpcs/multitau"]
        del group["delay_list"]
        group["delay_list"] = np.array(delays) / 1.5

    dialog = _open_regroup(viewer)
    _preview(dialog, monkeypatch, input_paths[1], qtbot)
    widget = dialog.findChild(PlotWidget)
    widget.grab()
    axis = widget.getAxis("bottom")
    picture = QtGui.QPicture()
    painter = QtGui.QPainter(picture)
    try:
        _, _, specs = axis.generateDrawSpecs(painter)
    finally:
        painter.end()

    labels = {float(label) for _, _, label in specs}
    assert len(labels) >= 2
    if delays[1] > 0:
        assert all(3.2 <= label <= 3.3 for label in labels)


def test_regroup_discards_stale_preview_after_map_change(
    viewer, input_paths, monkeypatch, qtbot, tmp_path
):
    jobs = []
    monkeypatch.setattr(viewer.thread_pool, "start", jobs.append)
    dialog = _open_regroup(viewer)
    preview = _button(dialog, "Preview")
    save = _button(dialog, "Save as…")
    _choose_map(dialog, monkeypatch, input_paths[1])
    preview.click()
    jobs[-1].run()
    qtbot.waitUntil(save.isEnabled)
    _, previous = _curve(dialog).getData()

    other_map = tmp_path / "other-map.h5"
    with h5py.File(other_map, "w") as handle:
        handle["/qmap/static_roi_map"] = [[1, 1, 2, 3]]
        handle["/qmap/dynamic_roi_map"] = [[4, 4, 4, 4]]
    _choose_map(dialog, monkeypatch, other_map)
    preview.click()
    jobs[-1].run()  # Queue a valid but soon-to-be-stale result for the other map.

    bad_map = tmp_path / "empty-map.h5"
    with h5py.File(bad_map, "w"):
        pass
    _choose_map(dialog, monkeypatch, bad_map)
    qtbot.wait(1)
    assert not save.isEnabled()
    np.testing.assert_array_equal(_curve(dialog).getData()[1], previous)

    preview.click()
    jobs[-1].run()
    qtbot.waitUntil(
        lambda: any(
            "Expected both ROI maps" in label.text()
            for label in dialog.findChildren(QtWidgets.QLabel)
        )
    )
    assert not save.isEnabled()
    np.testing.assert_array_equal(_curve(dialog).getData()[1], previous)

    _choose_map(dialog, monkeypatch, other_map)
    preview.click()
    jobs[-1].run()
    qtbot.waitUntil(save.isEnabled)
    assert dialog.findChild(QtWidgets.QComboBox).currentText() == "4"


def test_regroup_cancelled_save_never_submits(
    viewer, input_paths, monkeypatch, qtbot, tmp_path
):
    jobs = []
    errors = []
    monkeypatch.setattr(viewer.thread_pool, "start", jobs.append)
    monkeypatch.setattr(viewer.toast_manager, "show_error", errors.append)
    dialog = _open_regroup(viewer)
    _choose_map(dialog, monkeypatch, input_paths[1])
    _button(dialog, "Preview").click()
    jobs.pop().run()
    save = _button(dialog, "Save as…")
    qtbot.waitUntil(save.isEnabled)
    before = [path.read_bytes() for path in input_paths]
    directory_entries = set(tmp_path.iterdir())
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", lambda *_args: ("", "")
    )
    save.click()
    assert not jobs
    assert set(tmp_path.iterdir()) == directory_entries
    assert [path.read_bytes() for path in input_paths] == before

    destination = tmp_path / "export.h5"
    destination.write_bytes(b"keep me")
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", lambda *_args: (str(destination), "")
    )
    save.click()
    assert len(jobs) == 1
    jobs.pop().run()
    qtbot.waitUntil(save.isEnabled)
    assert "already exists" in errors[0]
    assert "choose" in errors[0].lower()
    assert any(
        "already exists" in label.text() and label.isVisible()
        for label in dialog.findChildren(QtWidgets.QLabel)
    )
    assert destination.read_bytes() == b"keep me"

    destination.unlink()
    targets = list(viewer.vk.target)
    save.click()
    assert len(jobs) == 1
    jobs.pop().run()
    qtbot.waitUntil(save.isEnabled)
    assert destination.is_file()
    assert list(viewer.vk.target) == targets
    assert [path.read_bytes() for path in input_paths] == before


@pytest.mark.parametrize("retire", ["close", "parent", "replace", "clear", "average"])
def test_regroup_discards_previews_after_retirement(
    viewer, input_paths, monkeypatch, qtbot, tmp_path, retire
):
    jobs = []
    monkeypatch.setattr(viewer.thread_pool, "start", jobs.append)
    dialog = _open_regroup(viewer)
    _choose_map(dialog, monkeypatch, input_paths[1])
    _button(dialog, "Preview").click()
    jobs[0].run()  # Queue completion before retiring the session or dialog.
    if retire == "close":
        dialog.close()
    elif retire == "parent":
        viewer.close()
    elif retire == "replace":
        replacement = tmp_path / "replacement"
        replacement.mkdir()
        viewer.work_dir.setText(str(replacement))
        viewer.reload_source()
        assert viewer.vk.path == str(replacement)
    elif retire == "average":
        viewer.submit_job()
        assert dialog.isVisible()
        viewer.vk.target.append("other.h5")
        viewer.avg_save_path.setText(str(tmp_path))
        viewer.avg_save_name.setText("average.h5")
        viewer.bx_avg_G2IPIF.setChecked(True)
        viewer.avg_blmin.setValue(0)
        viewer.avg_blmax.setValue(2)
        viewer.submit_job()
        assert viewer.vk.avg_worker is not None
        assert not viewer.vk.target
        assert not dialog.isVisible()
    else:
        viewer.reset_gui()
        assert not viewer.vk.target

    qtbot.waitUntil(
        lambda: not viewer.findChildren(QtWidgets.QDialog, "g2_regroup_dialog")
    )


@pytest.mark.parametrize("retire", ["map", "session"])
def test_regroup_rechecks_preview_after_save_picker(
    viewer, input_paths, monkeypatch, qtbot, tmp_path, retire
):
    dialog = _open_regroup(viewer)
    _preview(dialog, monkeypatch, input_paths[1], qtbot)
    destination = tmp_path / "export.h5"

    def choose_destination(*_args):
        if retire == "map":
            _choose_map(dialog, monkeypatch, input_paths[1])
        else:
            viewer.reset_gui()
        return str(destination), ""

    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", choose_destination)
    _button(dialog, "Save as…").click()
    assert viewer.thread_pool.waitForDone(2000)
    assert not destination.exists()


@pytest.mark.parametrize("retire", ["dialog", "session"])
def test_regroup_accepted_save_finishes_after_dialog_closes(
    viewer, input_paths, monkeypatch, qtbot, tmp_path, retire
):
    dialog = _open_regroup(viewer)
    _preview(dialog, monkeypatch, input_paths[1], qtbot)
    destination = tmp_path / "export.h5"
    first_publish = Event()
    release = Event()
    save_calls = []
    successes = []
    errors = []
    link = g2_regroup.os.link

    def pause_before_publish(temporary, target):
        if target == destination:
            save_calls.append(target)
            first_publish.set()
            assert release.wait(5)
        link(temporary, target)

    def record_success(message):
        assert destination.is_file()
        assert QtCore.QThread.currentThread() == viewer.thread()
        successes.append(message)

    monkeypatch.setattr(g2_regroup.os, "link", pause_before_publish)
    monkeypatch.setattr(viewer.toast_manager, "show_success", record_success)
    monkeypatch.setattr(viewer.toast_manager, "show_error", errors.append)
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", lambda *_args: (str(destination), "")
    )
    save = _button(dialog, "Save as…")
    save.click()
    save.click()
    try:
        qtbot.waitUntil(first_publish.is_set)
        if retire == "session":
            replacement = tmp_path / "replacement"
            replacement.mkdir()
            new_source = replacement / "other-source.h5"
            copyfile(input_paths[0], new_source)
            viewer.load_path(str(replacement))
            viewer.vk.target.append(new_source.name)
            viewer.update_box(viewer.vk.target, mode="target")
        else:
            dialog.close()

        qtbot.waitUntil(
            lambda: not viewer.findChildren(QtWidgets.QDialog, "g2_regroup_dialog")
        )
        other_map = tmp_path / "other-map.h5"
        with h5py.File(other_map, "w") as handle:
            handle["/qmap/static_roi_map"] = [[1, 1, 2, 3]]
            handle["/qmap/dynamic_roi_map"] = [[4, 4, 4, 4]]
        new_dialog = _open_regroup(viewer)
        _choose_map(new_dialog, monkeypatch, other_map)
        _button(new_dialog, "Preview").click()
        qtbot.waitUntil(
            lambda: new_dialog.findChild(QtWidgets.QComboBox).currentText() == "4"
        )
        second_save = _button(new_dialog, "Save as…")
        assert not second_save.isEnabled()
        second_save.click()
        assert save_calls == [destination]
        assert not destination.exists()
        assert not successes and not errors
    finally:
        release.set()

    qtbot.waitUntil(lambda: len(successes) == 1)
    assert second_save.isEnabled()
    assert save_calls == [destination]
    assert not errors
    with h5py.File(destination) as artifact:
        assert artifact.attrs["xpcsviewer_artifact"] == "g2_regroup"
        group = artifact["/xpcs/multitau"]
        assert group.attrs["source_filename"] == input_paths[0].name
        assert group.attrs["qmap_filename"] == input_paths[1].name
        np.testing.assert_array_equal(group["delay_seconds"], [0, 3, 6])
        np.testing.assert_array_equal(group["dynamic_bin_ids"], [1, 2])
        np.testing.assert_allclose(group["normalized_g2"], [[61 / 15, 2]] * 3)
