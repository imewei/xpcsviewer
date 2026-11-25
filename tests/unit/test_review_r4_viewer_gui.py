"""Regression tests for review round 4 in the viewer GUI layer.

One invariant: what the GUI shows and stores always belongs to the CURRENT
revision of the CURRENT selection.

R4-10  row 0 / the clicked(bool) argument are not "no row"
R4-11  reload of a replaced file / partition import redraws the plot
R4-12  a Bayesian bundle is only offered for the file revision it was fitted to
R4-13  a modeless All-Q dialog exports the snapshot it was created from
R4-14  bundles of files that left the target list are evicted
R4-02  a non-converged single-Q result is announced as not a measurement

The tests marked "real" build the real ``XpcsViewer`` offscreen over HDF5
files written with h5py.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import h5py
import numpy as np
import pandas as pd
import pytest
from qtpy import QtCore, QtWidgets

from tests.unit.test_review_r3_fit_status import _write_file
from xpcsviewer.fileIO.hdf_reader import _file_identity

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _tau():
    return np.logspace(-3, 0, 14)


def _write(path, intensity, q=(0.1, 0.2, 0.3)):
    """Real NeXus-style file whose 2-D SAXS image is ``intensity`` everywhere."""
    _write_file(path, _tau(), q=q)
    with h5py.File(path, "r+") as f:
        f["/xpcs/temporal_mean/scattering_2d"][:] *= intensity


def _replace(root, data, name, intensity, q):
    """Atomically replace ``data/name`` by a separately written file."""
    tmp = root / f"replacement_{name}"
    _write(tmp, intensity, q)
    os.replace(tmp, data / name)


@pytest.fixture
def viewer_env(qtbot, monkeypatch, tmp_path):
    """A real offscreen viewer over ``tmp/data`` holding a.h5 (10) and b.h5 (100)."""
    from xpcsviewer.gui.state.recent_paths import RecentPathsManager
    from xpcsviewer.gui.state.session_manager import (
        AnalysisParameters,
        SessionManager,
    )

    xv = sys.modules["xpcsviewer.xpcs_viewer"]
    data = tmp_path / "data"
    data.mkdir()
    _write(data / "a.h5", 10)
    _write(data / "b.h5", 100)
    session = SimpleNamespace(
        data_path=str(data),
        target_files=[],
        active_tab=-1,
        window_geometry=SimpleNamespace(maximized=True),
        analysis_params=AnalysisParameters(),
    )
    monkeypatch.setattr(SessionManager, "load_session", lambda self: session)
    monkeypatch.setattr(SessionManager, "get_warnings", lambda self: [])
    monkeypatch.setattr(xv.XpcsViewer, "closeEvent", lambda self, ev: ev.accept())
    monkeypatch.setattr(xv.XpcsViewer, "_maybe_prompt_start_path", lambda self: None)
    # keep pytest tmp dirs out of the user's real recent-directories list
    monkeypatch.setattr(RecentPathsManager, "add_path", lambda self, path: None)
    view = xv.XpcsViewer(path=str(data))
    qtbot.addWidget(view)
    view.vk.add_target(["a.h5", "b.h5"])
    view.update_box(view.vk.target, mode="target")
    view.cb_saxs2D_type.setCurrentText("linear")
    return SimpleNamespace(view=view, root=tmp_path, data=data, xv=xv)


def _idle(view, seconds=15.0):
    app = QtWidgets.QApplication.instance()
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        if (
            not view.active_plot_operations
            and view.thread_pool.activeThreadCount() == 0
            and not view._saxs2d_selection_timer.isActive()
        ):
            app.processEvents()
            return
        time.sleep(0.01)
    raise AssertionError(
        f"plot operations did not finish: {view.active_plot_operations}"
    )


def _select(view, row):
    view.list_view_target.selectionModel().select(
        view.vk.target.index(row, 0),
        QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect,
    )
    view.update_plot()
    _idle(view)


def _image_values(widget):
    return np.unique(np.asarray(widget.image))


def _tab(xv, name):
    return next(i for i, n in xv.tab_mapping.items() if n == name)


# --------------------------------------------------------------------------
# R4-10
# --------------------------------------------------------------------------
def test_r4_10_real_selector_row_zero_wins_over_highlighted_row(viewer_env):
    view = viewer_env.view
    view.tabWidget.setCurrentIndex(0)
    _select(view, 1)
    view.spinBox_saxs2d_selection.setValue(1)
    _idle(view)
    np.testing.assert_array_equal(_image_values(view.pg_saxs), [100])
    view.spinBox_saxs2d_selection.setValue(0)
    _idle(view)
    np.testing.assert_array_equal(_image_values(view.pg_saxs), [10])


def test_r4_10_selection_none_means_no_row_zero_means_row_zero(viewer_env):
    view = viewer_env.view
    assert view.plot_saxs_2d(selection=0, dryrun=True)["rows"] == [0]
    assert view.plot_saxs_2d(selection=None, dryrun=True)["rows"] == [0, 1]


def test_r4_10_real_button_checked_flag_is_not_an_index(viewer_env):
    view = viewer_env.view
    view.tabWidget.setCurrentIndex(0)
    spy = MagicMock()
    view.vk.plot_saxs_2d = spy
    view.pushButton_plot_saxs2d.setCheckable(True)
    view.pushButton_plot_saxs2d.click()  # clicked(True)
    assert spy.call_args.kwargs["rows"] == view.get_selected_rows() == [0, 1]


@pytest.mark.parametrize(
    "name", ["cb_g2map_cmap", "cb_g2map_scale", "cb_g2map_qmap_cmap"]
)
def test_r4_10_real_g2map_combo_index_is_not_the_dryrun_flag(viewer_env, name):
    # currentIndexChanged(int) used to arrive as plot_g2_map(dryrun=<int>): any
    # index > 0 made the call a dry run and nothing was drawn.
    view = viewer_env.view
    _select(view, 0)
    spy = MagicMock()
    view.vk.plot_g2map = spy
    combo = getattr(view, name)
    combo.setCurrentIndex(1)
    assert spy.call_count == 1


# --------------------------------------------------------------------------
# R4-11
# --------------------------------------------------------------------------
def test_r4_11_real_reload_of_replaced_file_redraws_saxs_image(viewer_env):
    env = viewer_env
    view = env.view
    view.tabWidget.setCurrentIndex(0)
    _select(view, 0)
    np.testing.assert_array_equal(_image_values(view.pg_saxs), [10])
    _replace(env.root, env.data, "a.h5", 777, (0.12, 0.22, 0.32))
    view.reload_source()
    view.update_plot()
    _idle(view)
    assert view.vk.get_xf_list([0])[0].saxs_2d.max() == 777  # the locator is fresh
    np.testing.assert_array_equal(_image_values(view.pg_saxs), [777])


def test_r4_11_real_partition_import_redraws_qmap(viewer_env):
    env = viewer_env
    view = env.view
    view.tabWidget.setCurrentIndex(_tab(env.xv, "qmap"))
    view.comboBox_qmap_target.setCurrentText("dqmap")
    _select(view, 0)
    old = np.asarray(view.pg_qmap.image).copy()
    swapped = np.tile([3, 2, 1], (2, 1))
    view.import_partition({"dynamic_roi_map": swapped, "static_roi_map": swapped})
    _idle(view)
    shown = np.asarray(view.pg_qmap.image)
    assert not np.array_equal(shown, old)
    assert shown.shape == old.shape
    assert set(np.unique(shown)) == {1, 2, 3}


def test_r4_11_real_sync_tab_replots_after_replacement(viewer_env):
    # saxs_1d goes through update_plot_sync: same signature rule.
    env = viewer_env
    view = env.view
    _select(view, 0)
    calls = []
    real = view.plot_saxs_1d

    def spy(dryrun=False):
        out = real(dryrun=dryrun)
        if not dryrun:
            calls.append(1)
        return out

    view.plot_saxs_1d = spy
    view.update_plot_sync("saxs_1d")
    view.update_plot_sync("saxs_1d")
    assert len(calls) == 1  # unchanged request is skipped
    _replace(env.root, env.data, "a.h5", 777, (0.12, 0.22, 0.32))
    view.update_plot_sync("saxs_1d")
    assert len(calls) == 2


def test_r4_11_real_reload_invalidates_records_and_retires_operations(viewer_env):
    env = viewer_env
    view = env.view
    _select(view, 0)
    assert view.plot_kwargs_record["saxs_2d"]
    retired = MagicMock()
    view._retire_plot_operations = retired
    _replace(env.root, env.data, "a.h5", 777, (0.12, 0.22, 0.32))
    view.reload_source()
    retired.assert_called()
    assert view.plot_kwargs_record["g2"] == {}


# --------------------------------------------------------------------------
# R4-12 / R4-14 helpers
# --------------------------------------------------------------------------
class _FakeSignal:
    def connect(self, *_a, **_k):
        pass


class _FakeWorker:
    """Stands in for BayesianFitWorker: records the launch, never samples."""

    instances: list = []

    def __init__(self, *_a, launch=None, **_k):
        self.launch = launch
        self.worker_id = f"fake{len(self.instances)}"
        self.cancelled = False
        self.signals = SimpleNamespace(
            finished=_FakeSignal(), error=_FakeSignal(), cancelled=_FakeSignal()
        )
        self.instances.append(self)

    def cancel(self):
        self.cancelled = True


def _fit_result(converged=True, chains=2):
    return SimpleNamespace(
        diagnostics=SimpleNamespace(converged=converged),
        config=SimpleNamespace(num_chains=chains),
        summary=pd.DataFrame({"mean": [0.5, 2.0]}, index=["alpha", "tau0"]),
    )


@pytest.fixture
def g2_env(viewer_env, monkeypatch):
    from xpcsviewer.threading import bayesian_worker

    _FakeWorker.instances = []
    monkeypatch.setattr(bayesian_worker, "BayesianFitWorker", _FakeWorker)
    view = viewer_env.view
    view.thread_pool = MagicMock()
    view.tabWidget.setCurrentIndex(_tab(viewer_env.xv, "g2_fitting"))
    for name, val in {
        "g2_qmin": 0.0,
        "g2_qmax": 0.5,
        "g2_tmin": 0.001,
        "g2_tmax": 1.0,
    }.items():
        w = getattr(view, name)
        w.setValue(val) if hasattr(w, "setValue") else w.setText(str(val))
    view.list_view_target.selectionModel().select(
        view.vk.target.index(0, 0),
        QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect,
    )
    view.sb_g2_bayesian_qidx.setValue(0)
    return viewer_env


def _launch_fit(view):
    view._fit_g2_bayesian()
    worker = _FakeWorker.instances[-1]
    return worker.launch


def _finish_fit(view, launch, **kw):
    view._on_g2_bayesian_finished({"launch": launch, "fit_result": _fit_result(**kw)})


def test_r4_12_real_bundle_not_offered_for_replaced_file(g2_env):
    env = g2_env
    view = env.view
    launch = _launch_fit(view)
    _finish_fit(view, launch)
    assert view._g2_current_bundle() is not None
    assert view.btn_g2_diagnosis.isEnabled()

    _replace(env.root, env.data, "a.h5", 999, (0.17, 0.27, 0.37))
    # even before Reload the old posterior must not pair with the new file
    assert view._g2_current_bundle() is None
    view.reload_source()
    assert view._g2_current_bundle() is None
    assert view._g2_bayesian_store == {}  # evicted, not just hidden
    assert not view.btn_g2_diagnosis.isEnabled()
    view._show_g2_diagnosis()
    assert view._g2_diagnosis_window is None


def test_r4_12_real_result_for_replaced_revision_is_dropped_at_completion(g2_env):
    env = g2_env
    view = env.view
    launch = _launch_fit(view)
    _replace(env.root, env.data, "a.h5", 999, (0.17, 0.27, 0.37))
    _finish_fit(view, launch)
    assert view._g2_bayesian_store == {}
    assert not view._g2_bayesian_worker_active  # the UI is released
    assert view.btn_g2_bayesian.isEnabled()


def test_r4_12_real_open_diagnosis_window_closed_for_replaced_file(g2_env):
    env = g2_env
    view = env.view
    launch = _launch_fit(view)
    _finish_fit(view, launch)
    window = MagicMock()
    view._g2_diagnosis_window = window
    view._g2_diagnosis_bundle = next(iter(view._g2_bayesian_store.values()))
    _replace(env.root, env.data, "a.h5", 999, (0.17, 0.27, 0.37))
    view.reload_source()
    window.close.assert_called()


def test_r4_12_real_diffusion_bundle_follows_file_revision(g2_env):
    from xpcsviewer.fitting.bayesian_assembly import BayesianBundle

    env = g2_env
    view = env.view
    xf = view.vk.get_xf_list([0])[0]
    bundle = BayesianBundle.create(
        xf.fname,
        "tau-q",
        "tau(q)",
        0.0,
        "power_law",
        [1.0, 2.0],
        [1.0, 2.0],
        [0.1, 0.1],
    )
    view._diff_bayesian_launch = bundle
    view._diff_bayesian_launch_revision = _file_identity(xf.fname)
    view._diff_bayesian_worker_active = True
    view._on_diff_bayesian_finished({"launch": bundle, "fit_result": _fit_result()})
    assert len(view._diff_bayesian_store) == 1
    _replace(env.root, env.data, "a.h5", 999, (0.17, 0.27, 0.37))
    view.reload_source()
    assert view._diff_bayesian_store == {}


# --------------------------------------------------------------------------
# R4-14
# --------------------------------------------------------------------------
def test_r4_14_real_remove_target_evicts_its_bundles_and_ignores_late_results(g2_env):
    env = g2_env
    view = env.view
    launch = _launch_fit(view)
    _finish_fit(view, launch)
    assert len(view._g2_bayesian_store) == 1

    # second fit in flight for the same file when the file leaves the targets
    launch2 = _launch_fit(view)
    worker = _FakeWorker.instances[-1]
    view.list_view_target.selectionModel().select(
        view.vk.target.index(0, 0),
        QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect,
    )
    view.remove_target()
    assert list(view.vk.target) == ["b.h5"]
    assert view._g2_bayesian_store == {}
    assert worker.cancelled
    assert not view._g2_bayesian_worker_active
    _finish_fit(view, launch2)  # late completion of the removed file
    assert view._g2_bayesian_store == {}


def test_r4_14_real_batch_results_of_removed_target_are_ignored(g2_env):
    env = g2_env
    view = env.view
    launch = _launch_fit(view)
    view._g2_batch_launch = {
        "fname": launch.fname,
        "revision": _file_identity(launch.fname),
    }
    view.list_view_target.selectionModel().select(
        view.vk.target.index(0, 0),
        QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect,
    )
    view.remove_target()
    assert view._g2_batch_launch is None
    view._on_g2_batch_single_q_done(0, {"launch": launch, "fit_result": _fit_result()})
    assert view._g2_bayesian_store == {}


# --------------------------------------------------------------------------
# R4-13
# --------------------------------------------------------------------------
def test_r4_13_real_dialog_exports_its_own_snapshot(viewer_env, monkeypatch, tmp_path):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from xpcsviewer.fitting import viz

    view = viewer_env.view
    xf = SimpleNamespace(
        fname="A.h5",
        bayesian_fit_summary={
            "q_val": np.array([0.17]),
            "fit_val": np.zeros((1, 2, 4)),
            "fit_func": "single",
        },
        bayesian_results={0: "old-result"},
    )
    view.vk.get_xf_list = lambda rows=None, **k: [xf]
    view.get_selected_rows = lambda: [0]
    fig = plt.figure()
    monkeypatch.setattr(viz, "plot_bayesian_all_q", lambda summary: fig)
    view._plot_g2_bayesian_all()
    dialog = view._bayesian_plot_dialog
    # a newer batch replaces the viewer's current summary while the dialog is open
    xf.bayesian_fit_summary = {
        "q_val": np.array([0.27]),
        "fit_val": np.zeros((1, 2, 4)),
        "fit_func": "single",
    }
    xf.bayesian_results = {0: "new-result"}

    seen = {}
    monkeypatch.setattr(
        viz,
        "export_bayesian_csv",
        lambda path, fit_val, q_val, fit_func, **k: seen.update(csv_q=list(q_val)),
    )
    monkeypatch.setattr(
        viz,
        "export_bayesian_diagnostics",
        lambda path, results: seen.update(nc=results),
    )
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path)
    )
    button = next(
        b
        for b in dialog.findChildren(QtWidgets.QPushButton)
        if b.text() == "Export Results"
    )
    button.click()
    assert seen["csv_q"] == [0.17]
    assert seen["nc"] == {0: "old-result"}
    assert (tmp_path / "bayesian_all_q.pdf").is_file()
    dialog.close()


# --------------------------------------------------------------------------
# R4-02
# --------------------------------------------------------------------------
def test_r4_02_nonconverged_single_chain_fit_is_announced(g2_env):
    view = g2_env.view
    view.toast_manager = MagicMock()
    launch = _launch_fit(view)
    _finish_fit(view, launch, converged=False, chains=1)
    text = view.statusbar.currentMessage().lower()
    assert "not converge" in text
    assert "single-chain" in text
    assert "not a measurement" in text
    view.toast_manager.show_warning.assert_called_once()


def test_r4_02_converged_fit_has_plain_status(g2_env):
    view = g2_env.view
    view.toast_manager = MagicMock()
    launch = _launch_fit(view)
    _finish_fit(view, launch, converged=True, chains=4)
    assert "converged" in view.statusbar.currentMessage()
    assert "not" not in view.statusbar.currentMessage().lower()
    view.toast_manager.show_warning.assert_not_called()


def test_r4_10_real_twotime_cmap_index_is_not_the_dryrun_flag(viewer_env):
    # viewer_ui.py wires currentIndexChanged(int) straight to plot_twotime(dryrun):
    # any index > 0 made it a dry run that drew nothing.
    view = viewer_env.view
    spy = MagicMock()
    view.plot_twotime = spy
    assert view.cb_twotime_cmap.count() > 1
    view.cb_twotime_cmap.setCurrentIndex(1)
    spy.assert_called_once_with()


# --------------------------------------------------------------------------
# R4-13 / R4-12 window bookkeeping
# --------------------------------------------------------------------------
def test_r4_13_snapshot_is_immutable(viewer_env):
    from xpcsviewer.xpcs_viewer import _bayesian_export_snapshot

    xf = SimpleNamespace(
        fname="A.h5", bayesian_fit_summary={"q_val": [0.1]}, bayesian_results={0: "r"}
    )
    snap = _bayesian_export_snapshot(xf)
    xf.bayesian_fit_summary["q_val"] = [
        0.9
    ]  # a later in-place edit of the summary dict
    xf.bayesian_fit_summary["new"] = 1
    assert snap.summary["q_val"] == [0.1] and "new" not in snap.summary
    with pytest.raises(TypeError):
        snap.summary["q_val"] = 1  # type: ignore[index]
    with pytest.raises(AttributeError):
        snap.fname = "B.h5"  # type: ignore[misc]


def test_r4_12_destroyed_old_dialog_does_not_forget_the_new_one(viewer_env):
    view = viewer_env.view
    old, new = object(), object()
    view._bayesian_plot_dialog = new
    view._forget_window("_bayesian_plot_dialog", old)  # late destroyed() of the old
    assert view._bayesian_plot_dialog is new
    view._forget_window("_bayesian_plot_dialog", new)
    assert view._bayesian_plot_dialog is None and view._bayesian_plot_snapshot is None


def test_r4_14_evicted_diagnosis_window_is_closed_and_released(g2_env):
    view = g2_env.view
    launch = _launch_fit(view)
    _finish_fit(view, launch)
    window = MagicMock()
    view._g2_diagnosis_window = window
    view._g2_diagnosis_bundle = next(iter(view._g2_bayesian_store.values()))
    view.list_view_target.selectionModel().select(
        view.vk.target.index(0, 0),
        QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect,
    )
    view.remove_target()
    window.close.assert_called()
    window.deleteLater.assert_called()
    assert view._g2_diagnosis_window is None and view._g2_diagnosis_bundle is None


def test_r4_02_diagnosis_csv_export_marks_a_nonconverged_fit(
    qtbot, monkeypatch, tmp_path
):
    from xpcsviewer.gui.bayesian_diagnosis import BayesianDiagnosisWindow

    window = BayesianDiagnosisWindow()
    qtbot.addWidget(window)
    window._result = SimpleNamespace(
        diagnostics=SimpleNamespace(converged=False),
        param_names=["a"],
        samples={"a": np.array([1.0, 2.0, 3.0])},
    )
    window._x_data = np.array([1.0, 2.0])
    window._y_data = np.array([1.0, 2.0])
    window._yerr = np.array([0.1, 0.1])
    window._model_func = lambda x, a: a * x
    out = tmp_path / "fit.csv"
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), "")
    )
    window._export_fit_data()
    assert "# NOT CONVERGED" in out.read_text()
