"""Regression tests for the lifetime of the G2 fit bounds."""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from xpcsviewer import file_locator
from xpcsviewer.gui.qt_compat import QtCore, QtWidgets
from xpcsviewer.gui.widgets.toast_notification import ToastWidget
from xpcsviewer.module import g2mod
from xpcsviewer.threading.async_kernel import AsyncViewerKernel
from xpcsviewer.threading.plot_workers import G2PlotWorker
from xpcsviewer.viewer_kernel import ViewerKernel
from xpcsviewer.xpcs_file import XpcsFile
from xpcsviewer.xpcs_viewer import XpcsViewer


class StubKernel(file_locator.FileLocator):
    reset_kernel = file_locator.FileLocator.clear_target
    submit_job = ViewerKernel.submit_job

    def plot_g2(self, **kwargs):
        return [[0.01]], self.delays

    def plot_saxs_2d(self, **kwargs):
        pass


@pytest.fixture
def viewer(qtbot, monkeypatch, tmp_path):
    monkeypatch.setattr(XpcsViewer, "_restore_session", lambda self: None)
    monkeypatch.setattr(XpcsViewer, "closeEvent", lambda self, event: event.accept())
    window = XpcsViewer()
    qtbot.addWidget(window)
    window.vk = StubKernel(str(tmp_path))
    monkeypatch.setattr(window, "init_async_kernel", lambda: None)
    monkeypatch.setattr(window, "update_tab_availability", lambda: None)
    monkeypatch.setattr(window.recent_paths_manager, "add_path", lambda path: None)
    window.g2_tauto.setChecked(True)
    window.tabWidget.setCurrentIndex(4)
    return window


@pytest.fixture
def deferred_plot_workers(viewer):
    workers = []
    viewer.async_vk = AsyncViewerKernel(
        viewer.vk, SimpleNamespace(start=workers.append)
    )
    queued = QtCore.Qt.ConnectionType.QueuedConnection
    viewer.async_vk.plot_ready.connect(viewer.on_async_plot_ready, queued)
    viewer.async_vk.operation_error.connect(viewer.on_async_operation_error, queued)
    return workers


def _bounds(viewer):
    return viewer.g2_bmin.value(), viewer.g2_bmax.value()


def _edit_bounds(viewer):
    viewer.g2_bmin.setValue(0.03)
    viewer.g2_bmax.setValue(2.0)


def _add_targets(viewer, *files):
    viewer.vk.add_target(list(files), preload=False)
    viewer.update_box(viewer.vk.target, mode="target")


def test_g2_fit_bounds_survive_replots_and_nonempty_target_changes(viewer):
    _add_targets(viewer, "first.h5", "second.h5")
    viewer.vk.delays = [[0.0, 0.2, 0.8]]
    viewer.plot_g2()
    assert _bounds(viewer) == (0.01, 8.0)
    _edit_bounds(viewer)

    viewer.vk.delays = [[0.4, 1.6]]
    first = viewer.vk.target.index(0, 0)
    viewer.list_view_target.setCurrentIndex(first)
    viewer.list_view_target.clicked.emit(first)
    assert _bounds(viewer) == (0.03, 2.0)
    assert viewer.g2_tmin.text() == "3.64e-01"
    assert viewer.g2_tmax.text() == "1.76e+00"

    viewer.tabWidget.setCurrentIndex(5)
    viewer.tabWidget.setCurrentIndex(4)
    viewer.plot_g2()
    assert _bounds(viewer) == (0.03, 2.0)

    viewer.remove_target()
    assert list(viewer.vk.target) == ["second.h5"]
    _add_targets(viewer, "third.h5")
    viewer.plot_g2()
    assert _bounds(viewer) == (0.03, 2.0)


def test_g2_fit_bounds_wait_for_usable_delays_and_reset_with_session(viewer, tmp_path):
    _add_targets(viewer, "scan.h5")
    defaults = _bounds(viewer)
    old_range = viewer.g2_tmin.text(), viewer.g2_tmax.text()
    viewer.vk.delays = [[], [0.0, np.nan, np.inf]]
    viewer.plot_g2()
    assert _bounds(viewer) == defaults
    assert (viewer.g2_tmin.text(), viewer.g2_tmax.text()) == old_range
    assert not viewer._g2_fit_bounds_initialized

    viewer.vk.delays = [[0.4, 1.6]]
    viewer.plot_g2()
    assert _bounds(viewer) == (0.02, 16.0)
    assert viewer._g2_fit_bounds_initialized

    targets_at_cancel = []
    viewer.async_vk = SimpleNamespace(
        cancel_operation=lambda operation_id: targets_at_cancel.append(
            list(viewer.vk.target)
        )
    )
    viewer.active_plot_operations["removed"] = "g2"
    viewer.list_view_target.setCurrentIndex(viewer.vk.target.index(0, 0))
    viewer.remove_target()
    assert targets_at_cancel == [["scan.h5"]]
    viewer.async_vk = None
    assert not viewer.vk.target
    assert not viewer._g2_fit_bounds_initialized

    _add_targets(viewer, "scan.h5")
    viewer.vk.delays = [[0.6, 1.2]]
    viewer.plot_g2()
    assert _bounds(viewer) == (0.03, 12.0)
    _edit_bounds(viewer)

    replacement = tmp_path / "replacement"
    replacement.mkdir()
    viewer.load_path(str(replacement))
    assert not viewer._g2_fit_bounds_initialized
    _add_targets(viewer, "scan.h5")
    viewer.plot_g2()
    assert _bounds(viewer) == (0.03, 12.0)


def test_g2_fit_bounds_reset_only_when_averaging_is_submitted(viewer, tmp_path):
    files = ["first.h5", "second.h5"]
    _add_targets(viewer, *files)
    viewer.vk.delays = [[0.2, 0.8]]
    viewer.plot_g2()
    _edit_bounds(viewer)
    viewer.avg_save_path.setText(str(tmp_path))
    viewer.avg_save_name.setText("average.h5")
    viewer.bx_avg_saxs.setChecked(True)

    targets_at_cancel = []
    viewer.async_vk = SimpleNamespace(
        cancel_operation=lambda operation_id: targets_at_cancel.append(
            (operation_id, list(viewer.vk.target))
        )
    )
    viewer.active_plot_operations["before_submission"] = "g2"
    viewer.avg_blmin.setValue(1.0)
    viewer.avg_blmax.setValue(1.0)
    viewer.btn_submit_job.click()
    assert viewer.active_plot_operations == {"before_submission": "g2"}
    assert list(viewer.vk.target) == files
    assert _bounds(viewer) == (0.03, 2.0)
    assert viewer._g2_fit_bounds_initialized
    assert targets_at_cancel == []

    viewer.avg_blmax.setValue(1.1)
    viewer.btn_submit_job.click()
    assert viewer.vk.avg_worker.file_list == files
    assert not viewer.vk.target
    assert targets_at_cancel == [("before_submission", files)]
    assert not viewer.active_plot_operations
    assert not viewer._g2_fit_bounds_initialized

    _add_targets(viewer, "replacement.h5")
    viewer.vk.delays = [[0.4, 1.6]]
    viewer.plot_g2()
    assert _bounds(viewer) == (0.02, 16.0)


def test_g2_fit_bounds_reset_only_when_reload_changes_directory(viewer, tmp_path):
    _add_targets(viewer, "original.h5")
    viewer.vk.delays = [[0.2, 0.8]]
    viewer.plot_g2()
    _edit_bounds(viewer)
    viewer.work_dir.setText(str(tmp_path))

    paths_at_cancel = []
    viewer.async_vk = SimpleNamespace(
        cancel_operation=lambda operation_id: paths_at_cancel.append(
            (operation_id, viewer.vk.path)
        )
    )
    viewer.active_plot_operations["before_reload"] = "g2"
    viewer.reload_source()
    assert list(viewer.vk.target) == ["original.h5"]
    assert _bounds(viewer) == (0.03, 2.0)
    assert viewer._g2_fit_bounds_initialized
    assert viewer.active_plot_operations == {"before_reload": "g2"}
    assert paths_at_cancel == []

    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / "new.h5").touch()
    viewer.work_dir.setText(str(replacement))
    viewer.reload_source()
    assert viewer.vk.path == str(replacement)
    assert list(viewer.vk.source) == ["new.h5"]
    assert not viewer.vk.target
    assert paths_at_cancel == [("before_reload", str(tmp_path))]
    assert not viewer.active_plot_operations
    assert not viewer._g2_fit_bounds_initialized

    _add_targets(viewer, "new.h5")
    viewer.vk.delays = [[0.4, 1.6]]
    viewer.plot_g2()
    assert _bounds(viewer) == (0.02, 16.0)


def _scan(path, tau):
    xf = SimpleNamespace(
        fname=str(path),
        atype=["Multitau"],
        fit_summary=None,
        tau=np.array(tau),
        g2_t0=1.0,
        g2=np.ones((len(tau), 1)),
        g2_err=np.zeros((len(tau), 1)),
        qmap=SimpleNamespace(
            get_qbin_in_qrange=lambda *args, **kwargs: ([0], [0.01]),
            get_qbin_label=lambda index: "q1",
        ),
    )
    xf.get_g2_data = Mock(wraps=partial(XpcsFile.get_g2_data, xf))
    return xf


def test_g2_async_bounds_use_native_delays_and_ignore_reset_completion(
    viewer, monkeypatch
):
    scan = _scan("scan.h5", [0.0, 0.2, 0.4, 0.8])
    monkeypatch.setattr(file_locator, "create_xpcs_dataset", lambda *a, **kw: scan)
    render = Mock()
    monkeypatch.setattr(g2mod, "pg_plot_from_data", render)
    _add_targets(viewer, "scan.h5")

    result = G2PlotWorker(
        viewer.vk, viewer.mp_g2, {"rows": [0], "t_range": (0.3, 0.5)}
    ).do_work()
    assert result["tel"][0].tolist() == [0.4]
    assert result["fit_delay_extent"] == (0.2, 0.8)
    scan.get_g2_data.assert_called_once()

    viewer.active_plot_operations["first"] = "g2"
    viewer.on_async_plot_ready("first", result)
    assert _bounds(viewer) == (0.01, 8.0)
    _edit_bounds(viewer)
    viewer.active_plot_operations["second"] = "g2"
    viewer.on_async_plot_ready("second", result)
    assert _bounds(viewer) == (0.03, 2.0)
    assert render.call_count == 2

    viewer.active_plot_operations.update(retired="g2", unrelated="qmap")
    viewer.progress_manager.start_operation("retired", "Plotting G2")
    viewer.reset_gui()
    _add_targets(viewer, "replacement.h5")
    viewer.on_async_plot_ready("retired", result)
    assert viewer.active_plot_operations == {"unrelated": "qmap"}
    assert not viewer.progress_manager.is_operation_active("retired")
    assert not viewer._g2_fit_bounds_initialized
    assert render.call_count == 2

    replacement = _scan("replacement.h5", [0.6, 1.2])
    monkeypatch.setattr(
        file_locator, "create_xpcs_dataset", lambda *a, **kw: replacement
    )
    current_result = G2PlotWorker(viewer.vk, viewer.mp_g2, {"rows": [0]}).do_work()
    viewer.active_plot_operations["current"] = "g2"
    viewer.on_async_plot_ready("current", current_result)
    assert _bounds(viewer) == (0.03, 12.0)
    assert render.call_count == 3


def test_g2_reset_discards_queued_plot_error(viewer, deferred_plot_workers):
    _add_targets(viewer, "old.h5")
    viewer.update_plot_async("g2")
    worker = deferred_plot_workers.pop()
    worker.signals.error.emit(worker.worker_id, "retired plot failed", "", 0)
    viewer.reset_gui()

    _add_targets(viewer, "current.h5")
    viewer.update_plot_async("g2")
    worker = deferred_plot_workers.pop()
    worker.signals.error.emit(worker.worker_id, "current plot failed", "", 0)
    viewer.progress_manager.start_operation("load_files_test", "Loading files")
    viewer.async_vk.operation_error.emit("load_files_test", "load failed", "")

    QtCore.QCoreApplication.sendPostedEvents(viewer, QtCore.QEvent.Type.MetaCall)
    errors = [
        toast.findChild(QtWidgets.QLabel).text()
        for toast in viewer.findChildren(ToastWidget, "toast_error")
    ]
    assert errors == [
        "Operation failed: current plot failed",
        "Operation failed: load failed",
    ]
    assert not viewer.active_plot_operations
    assert not viewer.progress_manager.get_active_operations()


def test_g2_plots_started_in_one_clock_tick_have_distinct_lifetimes(
    viewer, deferred_plot_workers, monkeypatch
):
    monkeypatch.setattr("time.time", lambda: 1000.0)
    scans = {
        "old.h5": _scan("old.h5", [0.2, 0.8]),
        "current.h5": _scan("current.h5", [0.6, 1.2]),
    }
    monkeypatch.setattr(
        file_locator, "create_xpcs_dataset", lambda path, **kw: scans[Path(path).name]
    )
    _add_targets(viewer, "old.h5")
    viewer.update_plot_async("g2")
    (retired,) = viewer.active_plot_operations
    deferred_plot_workers.pop().run()
    viewer.reset_gui()

    _add_targets(viewer, "current.h5")
    viewer.update_plot_async("g2")
    (current,) = viewer.active_plot_operations
    QtCore.QCoreApplication.sendPostedEvents(viewer, QtCore.QEvent.Type.MetaCall)

    assert not viewer._g2_fit_bounds_initialized
    assert retired != current
    assert viewer.progress_manager.is_operation_active(current)

    deferred_plot_workers.pop().run()
    QtCore.QCoreApplication.sendPostedEvents(viewer, QtCore.QEvent.Type.MetaCall)
    assert _bounds(viewer) == (0.03, 12.0)
    assert not viewer.active_plot_operations
    assert not viewer.progress_manager.is_operation_active(current)


def test_g2_reset_does_not_cache_an_inflight_file_load(viewer, monkeypatch, tmp_path):
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    viewer.vk.set_path(str(old))
    _add_targets(viewer, "scan.h5", "other.h5")
    read_started = Event()
    release_read = Event()
    opened = []

    def load(path, **kwargs):
        opened.append(path)
        if path == str(old / "scan.h5"):
            read_started.set()
            assert release_read.wait(5)
        return _scan(path, [0.2, 0.8])

    monkeypatch.setattr(file_locator, "create_xpcs_dataset", load)
    render = Mock()
    monkeypatch.setattr(g2mod, "pg_plot_from_data", render)
    worker = G2PlotWorker(viewer.vk, viewer.mp_g2, {"rows": [0, 1]})
    viewer.active_plot_operations["old"] = "g2"

    def cancel(operation_id):
        assert operation_id == "old"
        assert viewer.vk.path == str(old)
        worker.cancel()

    viewer.async_vk = SimpleNamespace(cancel_operation=cancel)

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(worker.do_work)
        try:
            assert read_started.wait(5)
            viewer.load_path(str(new))
            _add_targets(viewer, "scan.h5")
        finally:
            release_read.set()
        with pytest.raises(InterruptedError):
            pending.result(timeout=5)

    assert opened == [str(old / "scan.h5")]
    assert not viewer.vk.cache
    assert "old" not in viewer.active_plot_operations
    render.assert_not_called()

    assert viewer.vk.get_xf_list(rows=[0])[0].fname == str(new / "scan.h5")
    assert opened == [str(old / "scan.h5"), str(new / "scan.h5")]
