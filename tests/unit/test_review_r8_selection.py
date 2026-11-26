"""Regression tests for review round 8 (selections that match nothing, jobs).

Class 1, "a selection that matches nothing is replaced by something else":
R8-03  the averaging baseline check on a Q bin a file does not have rejects the
       file (named, with the reason) BEFORE it can set the reference grid; the
       GUI worker and the standalone ``do_average`` share one validation
R8-06  a SAXS-1D Q range that matches no bin of a file leaves that file out of
       the plot (named on the status bar) instead of drawing the full curve

Class 2, averaging jobs and progress records:
R8-07  a retired (removed or replaced) averaging worker never publishes, and a
       removed job leaves no partial / final output and no temp file
R8-07b a progress message queued before ``complete_operation`` does not recreate
       the finished operation

R8-11  the diagnosis window's own CSV / netCDF exports and its title carry the
       placeholder-Q-map warning

Real h5py-written files, the real ``XpcsFile``, kernel, worker and an offscreen
``XpcsViewer``; Events (never sleeps) order the worker threads.
"""

from __future__ import annotations

import csv
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pyqtgraph as pg
import pytest
from qtpy import QtCore, QtWidgets

from tests.unit import test_review_r3_average as r3avg
from tests.unit.test_review_r3_fit_status import _write_file
from tests.unit.test_review_r5_bayes_gui import _real_result
from tests.unit.test_review_r6_viewer_gui import _FakeIdata, _make_env
from xpcsviewer.fileIO.hdf_reader import clear_connection_pool
from xpcsviewer.module import average_toolbox as at
from xpcsviewer.module import saxs1d
from xpcsviewer.module.average_toolbox import AverageToolbox
from xpcsviewer.utils.qmap_notice import PLACEHOLDER_QMAP_WARNING
from xpcsviewer.xpcs_file import XpcsFile

pytestmark = pytest.mark.unit

K = r3avg.K
FIELDS = ["g2", "g2_err"]
PKG = Path(__file__).resolve().parents[2] / "xpcsviewer"


@pytest.fixture(autouse=True)
def _fresh_pool(monkeypatch):
    monkeypatch.setattr(
        "xpcsviewer.module.average_toolbox.MemoryMonitor.is_memory_pressure_high",
        staticmethod(lambda *a, **k: False),
    )
    clear_connection_pool()
    yield
    clear_connection_pool()


# ----------------------------------------------------------------- helpers
def _write_bins(path, bins, g2=1.0, dq_shift=0.0):
    """Real NeXus file with ``bins`` Q bins (2 = the stock file, 6 = a wider one)."""
    r3avg._write_nexus(path, g2=1.0)
    if bins == 2:
        return path
    vals = np.full((3, bins), g2)
    with h5py.File(path, "a") as f:
        for key, value in (
            ("g2", vals),
            ("g2_err", np.full_like(vals, 0.01)),
            ("G2", np.repeat(vals[:, :, None], 3, axis=2)),
            ("dqlist", np.arange(1, bins + 1) / 100.0 + dq_shift),
            ("dynamic_index_mapping", np.arange(bins)),
            ("dynamic_num_pts", np.array([bins, 1])),
            ("dqmap", (np.arange(16).reshape(4, 4) % bins) + 1),
        ):
            del f[K[key]]
            f[K[key]] = value
    return path


def _grid(tmp_path, spec):
    """``spec``: name -> (bins, g2[, dq_shift]); returns the file names in order."""
    for name, args in spec.items():
        _write_bins(tmp_path / name, *args)
    return list(spec)


SPEC_2_THEN_6 = {
    "a.h5": (2, 1.0),
    "b.h5": (2, 1.0),
    "c.h5": (6, 1.04),
    "d.h5": (6, 1.06),
}
BASELINE = {"avg_qindex": 5, "avg_blmin": 0.9, "avg_blmax": 1.1, "fields": FIELDS}


def _run_worker(tmp_path, names, out, **kw):
    worker = AverageToolbox(work_dir=str(tmp_path), flist=names, jid=1)
    worker.do_average(save_path=str(out), **{**BASELINE, **kw})
    return worker


def _read(path, name="g2"):
    with h5py.File(path, "r") as f:
        return f[K[name]][()]


# ------------------------------------------------------------------- R8-03
@pytest.mark.parametrize("kind", ["worker", "standalone"])
def test_r8_03_files_lacking_the_index_cannot_set_the_reference_grid(tmp_path, kind):
    names = _grid(tmp_path, SPEC_2_THEN_6)
    out = tmp_path / "avg.h5"
    if kind == "worker":
        rejected = _run_worker(tmp_path, names, out).rejected
    else:
        rejected = {}
        at.do_average(
            names,
            work_dir=str(tmp_path),
            save_path=str(out),
            n_jobs=1,
            rejected=rejected,
            **BASELINE,
        )
    data = _read(out)
    assert data.shape == (3, 6)  # the grid of the files that HAVE bin 5
    np.testing.assert_allclose(data, 1.05)
    assert set(rejected) == {"a.h5", "b.h5"}
    for why in rejected.values():
        assert "Q bin 5" in why and "2 Q bins" in why


def test_r8_03_the_parallel_worker_path_applies_the_same_rule(tmp_path):
    spec = {f"f{i}.h5": (2, 1.0) for i in range(4)}
    spec.update({f"g{i}.h5": (6, 1.04) for i in range(4)})
    names = _grid(tmp_path, spec)
    out = tmp_path / "avg.h5"
    worker = _run_worker(tmp_path, names, out)
    assert worker.n_accepted == 4
    assert _read(out).shape == (3, 6)
    assert set(worker.rejected) == {f"f{i}.h5" for i in range(4)}


def test_r8_03_the_standalone_process_pool_reports_its_rejections(tmp_path):
    names = _grid(tmp_path, SPEC_2_THEN_6)
    rejected = {}
    baseline = at.do_average(
        names,
        work_dir=str(tmp_path),
        save_path=str(tmp_path / "avg.h5"),
        n_jobs=2,  # four files and two jobs: the ProcessPoolExecutor branch
        rejected=rejected,
        **BASELINE,
    )
    assert _read(tmp_path / "avg.h5").shape == (3, 6)
    assert set(rejected) == {"a.h5", "b.h5"}
    assert np.isnan(baseline[:2]).all() and np.allclose(baseline[2:], [1.04, 1.06])


def test_r8_03_no_file_has_the_index_means_no_output(tmp_path):
    names = _grid(tmp_path, {"a.h5": (2, 1.0), "b.h5": (2, 1.0)})
    out = tmp_path / "avg.h5"
    worker = _run_worker(tmp_path, names, out)
    assert worker.n_accepted == 0 and not out.exists()
    assert set(worker.rejected) == {"a.h5", "b.h5"}


def test_r8_03_same_ordinal_on_a_different_physical_bin_is_refused(tmp_path):
    # the index is an ordinal into each file's own bin list: bin 5 of e.h5 sits at
    # another q than bin 5 of c.h5, and the reference-grid comparison refuses it
    names = _grid(
        tmp_path,
        {"c.h5": (6, 1.04, 0.0), "e.h5": (6, 1.30, 0.005)},
    )
    out = tmp_path / "avg.h5"
    worker = _run_worker(tmp_path, names, out, avg_blmax=1.5)
    np.testing.assert_allclose(_read(out), 1.04)
    assert list(worker.rejected) == ["e.h5"]
    assert "coordinates" in worker.rejected["e.h5"]


def test_r8_03_one_validation_function_serves_both_paths():
    import inspect

    sources = {
        name: inspect.getsource(obj)
        for name, obj in (
            ("worker", AverageToolbox.do_average),
            ("standalone", at._process_file_for_average),
            ("batch", AverageToolbox._process_batch),
            ("sequential", AverageToolbox._process_files_sequential),
        )
    }
    assert "using 0 instead" not in inspect.getsource(at)
    for name, text in sources.items():
        assert "idx = 0" not in text, name


def test_r8_03_the_gui_names_the_rejected_files(qtbot, monkeypatch, tmp_path):
    env = _make_env(
        qtbot,
        monkeypatch,
        tmp_path,
        {
            name: (lambda p, a=args: _write_bins(p, *a))
            for name, args in SPEC_2_THEN_6.items()
        },
    )
    view = env.view
    view.avg_save_path.setText(str(tmp_path / "out"))
    view.avg_save_name.setText("average.h5")
    for box, on in (
        ("bx_avg_G2IPIF", False),
        ("bx_avg_saxs", False),
        ("bx_avg_g2g2err", True),
    ):
        getattr(view, box).setChecked(on)
    view.avg_blmin.setValue(0.9)
    view.avg_blmax.setValue(1.1)
    assert view.avg_qindex.value() == 5  # the default the reviewer used
    # Audit A2 lowers a default index the SMALLEST selected file lacks (here the
    # 2-bin files) to that file's last bin at submit time. This test is about the
    # worker's per-file rejection that remains the backstop (a file that still
    # lacks the index, e.g. changed since it was cached), so the lowering is off.
    monkeypatch.setattr(view, "_baseline_qindex_for_job", lambda: 5)
    view.submit_job()
    worker = view.vk.avg_worker
    view.start_avg_job()
    assert view.thread_pool.waitForDone(5000)
    QtWidgets.QApplication.processEvents()
    view.timer.stop()
    view._poll_avg_job()
    assert _read(tmp_path / "out" / "average.h5").shape == (3, 6)
    shown = view.statusbar.currentMessage()
    assert "a.h5" in shown and "b.h5" in shown and "Q bin 5" in shown
    assert "c.h5" not in shown
    assert set(worker.rejected) == {"a.h5", "b.h5"}


# ------------------------------------------------------------------- R8-06
LOW = {"q": [0.08, 0.10, 0.12], "saxs": [11.0, 12.0, 13.0]}
HIGH = {"q": [0.20, 0.22, 0.24], "saxs": [21.0, 22.0, 23.0]}


def _saxs_file(path, spec):
    _write_file(path, np.geomspace(0.005, 1, 25), q=spec["q"], saxs=spec["saxs"])
    with h5py.File(path, "r+") as f:
        f["/xpcs/qmap/static_v_list_dim0"][:] = spec["q"]
    return path


@pytest.mark.parametrize("norm", [None, "q2", "q4", "I0"])
def test_r8_06_a_range_matching_no_bin_selects_nothing(tmp_path, norm):
    xf = XpcsFile(str(_saxs_file(tmp_path / "high.h5", HIGH)))
    q, Iq, xlabel, ylabel = xf.get_saxs1d_data(qrange=(0.075, 0.125), norm_method=norm)
    assert q.size == 0 and Iq.shape[-1] == 0
    full = xf.get_saxs1d_data(qrange=(0.15, 0.3), norm_method=None)[0]
    assert full.size == 3  # a matching range is untouched


def test_r8_06_the_renderer_leaves_the_file_out_and_names_it(tmp_path):
    files = [
        XpcsFile(str(_saxs_file(tmp_path / "low.h5", LOW))),
        XpcsFile(str(_saxs_file(tmp_path / "high.h5", HIGH))),
    ]
    widget = pg.PlotWidget()
    left_out = saxs1d.pg_plot(files, widget, qmin=0.075, qmax=0.125, plot_norm=3)
    assert list(left_out) == [files[1].label]
    assert "0.075" in left_out[files[1].label]
    drawn = [item.getData()[0] for item in widget.getPlotItem().listDataItems()]
    assert drawn and all(np.max(x) <= 0.125 for x in drawn)
    # nothing is left to draw when every file is out of range
    widget2 = pg.PlotWidget()
    both = saxs1d.pg_plot(files, widget2, qmin=0.5, qmax=0.9)
    assert set(both) == {f.label for f in files}
    assert widget2.getPlotItem().listDataItems() == []


def test_r8_06_the_gui_reports_the_left_out_file(qtbot, monkeypatch, tmp_path):
    env = _make_env(
        qtbot,
        monkeypatch,
        tmp_path,
        {
            "low.h5": lambda p: _saxs_file(p, LOW),
            "high.h5": lambda p: _saxs_file(p, HIGH),
        },
    )
    view = env.view
    view.cb_saxs_type.setCurrentIndex(0)
    view.saxs1d_qmin.setValue(0.075)
    view.saxs1d_qmax.setValue(0.125)
    view.plot_saxs_1d()
    curves = view.mp_saxs.getPlotItem().listDataItems()
    assert curves
    assert all(np.max(c.getData()[0]) <= 0.125 for c in curves)
    shown = view.statusbar.currentMessage()
    assert "high.h5" in shown and "low.h5" not in shown


def test_r8_06_the_export_writes_the_whole_curve_by_design(tmp_path):
    # export_saxs1d takes no Q range: there is no selection to fall back from
    xf = XpcsFile(str(_saxs_file(tmp_path / "high.h5", HIGH)))
    folder = tmp_path / "out"
    folder.mkdir()
    xf.export_saxs1d([], str(folder))
    table = np.loadtxt(folder / "high_saxs1d.txt")
    np.testing.assert_allclose(table[:, 0], HIGH["q"])


# ------------------------------------------------------------------- R8-07
@pytest.fixture
def job_env(qtbot, monkeypatch, tmp_path):
    names = {
        "oldA.h5": 0.98,
        "oldB.h5": 1.0,
        "newA.h5": 1.02,
        "newB.h5": 1.04,
    }
    env = _make_env(
        qtbot,
        monkeypatch,
        tmp_path,
        {n: (lambda p, v=v: r3avg._write_nexus(p, g2=v)) for n, v in names.items()},
    )
    view = env.view
    view.vk.clear_target()  # each job below targets only its own two files
    out = tmp_path / "out"
    view.avg_save_path.setText(str(out))
    view.avg_save_name.setText("average.h5")
    view.bx_avg_G2IPIF.setChecked(False)
    view.bx_avg_saxs.setChecked(False)
    view.bx_avg_g2g2err.setChecked(True)
    view.avg_blmin.setValue(0.9)
    view.avg_blmax.setValue(1.1)
    view.avg_qindex.setValue(0)
    view.max_thread_count.setValue(2)
    entered, release = threading.Event(), threading.Event()
    real = AverageToolbox._process_files_sequential

    def delayed(self, *a, **kw):
        if self.model[0] == "oldA.h5":
            entered.set()
            assert release.wait(2)
        return real(self, *a, **kw)

    monkeypatch.setattr(AverageToolbox, "_process_files_sequential", delayed)
    env.entered, env.release, env.out = entered, release, out
    env.dest = out / "average.h5"

    def start(*files):
        view.vk.add_target(list(files))
        view.update_box(view.vk.target, mode="target")
        view.submit_job()
        view.start_avg_job()
        return view.vk.avg_worker

    env.start = start
    yield env
    release.set()
    view.timer.stop()
    view.thread_pool.waitForDone(5000)


def _temp_files(folder):
    return [p.name for p in folder.iterdir() if p.name.startswith(".avg-")]


def test_r8_07_a_retired_worker_never_overwrites_the_newer_output(job_env, qtbot):
    env, view = job_env, job_env.view
    old = env.start("oldA.h5", "oldB.h5")
    assert env.entered.wait(2)
    new = env.start("newA.h5", "newB.h5")  # replaces the visible job, same destination
    qtbot.waitUntil(lambda: new.status == "finished", timeout=2000)
    np.testing.assert_allclose(_read(env.dest), 1.03)
    env.release.set()
    assert view.thread_pool.waitForDone(5000)
    QtWidgets.QApplication.processEvents()
    np.testing.assert_allclose(_read(env.dest), 1.03)  # the old job did not commit
    assert old.status == "killed"
    assert old.jid in view.vk._retired_avg_jobs and view.vk.avg_worker is new
    assert _temp_files(env.out) == []


@pytest.mark.parametrize("existing", [False, True])
def test_r8_07_removing_a_running_job_leaves_no_output_and_no_temp(job_env, existing):
    env, view = job_env, job_env.view
    env.out.mkdir()
    if existing:  # an earlier average at the destination stays exactly as it was
        r3avg._write_nexus(env.dest, g2=0.5)
    old = env.start("oldA.h5", "oldB.h5")
    assert env.entered.wait(2)
    view.avg_job_pop.click()
    env.release.set()
    assert view.thread_pool.waitForDone(5000)
    QtWidgets.QApplication.processEvents()
    assert old.status == "killed"
    assert _temp_files(env.out) == []
    if existing:
        np.testing.assert_allclose(_read(env.dest), 0.5)
    else:
        assert not env.dest.exists()


def test_r8_07_removal_during_the_publish_step_discards_the_staged_file(
    job_env, monkeypatch
):
    env, view = job_env, job_env.view
    staged, go = threading.Event(), threading.Event()
    real_put = at.put

    def slow_put(*a, **kw):
        real_put(*a, **kw)  # the whole staging file is written ...
        staged.set()
        assert go.wait(2)  # ... and the job is removed before it is renamed

    monkeypatch.setattr(at, "put", slow_put)
    env.release.set()  # no delay in the processing loop for this test
    env.start("oldA.h5", "oldB.h5")
    assert staged.wait(2)
    assert _temp_files(env.out)  # the staged file exists right now
    view.avg_job_pop.click()
    go.set()
    assert view.thread_pool.waitForDone(5000)
    assert not env.dest.exists()
    assert _temp_files(env.out) == []


def test_r8_07_a_finished_job_keeps_its_output_when_it_is_removed(job_env, qtbot):
    env, view = job_env, job_env.view
    env.release.set()
    job = env.start("oldA.h5", "oldB.h5")
    qtbot.waitUntil(lambda: job.status == "finished", timeout=2000)
    view.avg_job_pop.click()
    np.testing.assert_allclose(_read(env.dest), 0.99)


def test_r8_07_kill_waits_for_a_commit_in_progress(tmp_path):
    names = _grid(tmp_path, {"a.h5": (2, 1.0), "b.h5": (2, 1.0)})
    worker = AverageToolbox(work_dir=str(tmp_path), flist=names, jid=1)
    published = []
    assert worker._commit(lambda: published.append(1)) is True
    worker.kill()
    assert worker._commit(lambda: published.append(2)) is False
    assert published == [1]


# progress records ------------------------------------------------------
class _Emitter(QtCore.QObject):
    progress = QtCore.Signal(str, int, int, str)


def test_r8_07_a_queued_progress_message_does_not_recreate_a_finished_operation(
    qtbot,
):
    from xpcsviewer.threading.progress_manager import ProgressManager

    manager = ProgressManager()
    started = []
    manager.operation_started.connect(lambda op, text: started.append(op))
    emitter = _Emitter()
    emitter.progress.connect(
        manager.update_progress, QtCore.Qt.ConnectionType.QueuedConnection
    )
    manager.start_operation("batch", "Batch fit", total=10, show_after_delay=-1)
    emitter.progress.emit("batch", 3, 10, "working")  # queued ...
    manager.complete_operation("batch", True)  # ... the operation ends first
    QtWidgets.QApplication.processEvents()  # the late message is delivered now
    assert "batch" not in manager.active_operations
    assert started == ["batch"]


def test_r8_07_a_cancelled_operation_is_not_recreated_either(qtbot):
    from xpcsviewer.threading.progress_manager import ProgressManager

    manager = ProgressManager()
    manager.start_operation("fit", "Fit", total=10, show_after_delay=-1)
    manager.cancel_operation_ui("fit")
    manager.update_progress("fit", 4, 10, "late")
    assert not manager.is_operation_active("fit")


def test_r8_07_an_operation_id_can_be_started_again(qtbot):
    from xpcsviewer.threading.progress_manager import ProgressManager

    manager = ProgressManager()
    manager.start_operation("fit", "Fit", total=10, show_after_delay=-1)
    manager.complete_operation("fit", True)
    manager.start_operation("fit", "Fit again", total=10, show_after_delay=-1)
    manager.update_progress("fit", 5, 10, "half")
    assert manager.active_operations["fit"].current == 5


def test_r8_07_an_unknown_operation_is_still_registered_by_its_first_update(qtbot):
    from xpcsviewer.threading.progress_manager import ProgressManager

    manager = ProgressManager()
    manager.update_progress("never-started", 1, 4, "x")
    assert manager.is_operation_active("never-started")


# ------------------------------------------------------------------ R8-11
def _window(qtbot, tmp_path, converged=True, **identity):
    from xpcsviewer.fitting.models import single_exp_func
    from xpcsviewer.gui.bayesian_diagnosis import (
        BayesianDiagnosisWindow,
        DiagnosisIdentity,
    )

    window = BayesianDiagnosisWindow()
    qtbot.addWidget(window)
    sink: list = []
    result = _real_result(converged=converged)
    result.arviz_data = _FakeIdata(sink)
    window.update_results(
        result=result,
        model_func=single_exp_func,
        x_data=[0.1, 0.2, 0.3],
        y_data=[1.3, 1.2, 1.1],
        yerr=[0.01] * 3,
        q_value=0.1,
        title="G2 Bayesian Diagnosis (a.h5, bin 0)",
        identity=DiagnosisIdentity(str(tmp_path / "a.h5"), None, 0, None, **identity),
    )
    window.sink = sink
    return window


def _export(window, monkeypatch, path, method):
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), "")
    )
    getattr(window, method)()


def test_r8_11_the_identity_flag_defaults_to_off(tmp_path):
    from xpcsviewer.gui.bayesian_diagnosis import DiagnosisIdentity

    assert DiagnosisIdentity("a.h5", None, 0, None).placeholder_qmap is False


def test_r8_11_csv_title_and_traces_carry_the_warning(qtbot, monkeypatch, tmp_path):
    window = _window(qtbot, tmp_path, placeholder_qmap=True)
    assert PLACEHOLDER_QMAP_WARNING in window.windowTitle()
    out = tmp_path / "fit.csv"
    _export(window, monkeypatch, out, "_export_fit_data")
    lines = out.read_text().splitlines()
    assert lines[0] == f"# {PLACEHOLDER_QMAP_WARNING}"
    header = next(r for r in csv.reader(lines) if r and not r[0].startswith("#"))
    assert "yerr" in header and len(lines) == 3 + 3  # 2 comment lines, header, 3 rows
    traces = tmp_path / "traces.nc"
    _export(window, monkeypatch, traces, "_export_traces")
    attrs = window.sink[0][1]
    assert attrs["q_map_is_placeholder"] == 1
    assert attrs["q_map_warning"] == PLACEHOLDER_QMAP_WARNING


def test_r8_11_a_calibrated_file_is_not_marked(qtbot, monkeypatch, tmp_path):
    window = _window(qtbot, tmp_path)
    assert PLACEHOLDER_QMAP_WARNING not in window.windowTitle()
    out = tmp_path / "fit.csv"
    _export(window, monkeypatch, out, "_export_fit_data")
    assert "placeholder" not in out.read_text().lower()
    _export(window, monkeypatch, tmp_path / "t.nc", "_export_traces")
    assert "q_map_is_placeholder" not in window.sink[0][1]


def test_r8_11_the_title_mark_follows_the_identity_not_the_history(qtbot, tmp_path):
    from xpcsviewer.gui.bayesian_diagnosis import DiagnosisIdentity

    window = _window(qtbot, tmp_path, placeholder_qmap=True)
    marked = window.windowTitle()
    window.update_results(  # same window, now a calibrated file, no new title
        result=window._result,
        model_func=window._model_func,
        x_data=window._x_data,
        y_data=window._y_data,
        identity=DiagnosisIdentity(str(tmp_path / "b.h5"), None, 0, None),
    )
    assert PLACEHOLDER_QMAP_WARNING in marked
    assert PLACEHOLDER_QMAP_WARNING not in window.windowTitle()
    assert window.windowTitle().count("Bayesian Diagnosis") == 1


def test_r8_11_real_viewer_launch_marks_the_exports(qtbot, monkeypatch, tmp_path):
    """With the one-line change at the creation site (applied here by a shim)."""
    from tests.unit.test_review_r4_viewer_gui import (
        _FakeWorker,
        _launch_fit,
        _tab,
    )
    from tests.unit.test_review_r5_bayes_gui import _finish_real, _open_g2_window
    from xpcsviewer.threading import bayesian_worker

    def bare(path):
        _write_file(path, np.geomspace(0.005, 1, 25))
        with h5py.File(path, "a") as f:
            del f["/xpcs/qmap"]

    env = _make_env(qtbot, monkeypatch, tmp_path, {"a.h5": bare})
    view = env.view
    _FakeWorker.instances = []
    monkeypatch.setattr(bayesian_worker, "BayesianFitWorker", _FakeWorker)
    view.thread_pool = SimpleNamespace(start=lambda *a, **k: None)
    view.tabWidget.setCurrentIndex(_tab(env.xv, "g2_fitting"))
    for name, val in {
        "g2_qmin": 0.0,
        "g2_qmax": 0.5,
        "g2_tmin": 0.001,
        "g2_tmax": 1.0,
    }.items():
        widget = getattr(view, name)
        widget.setValue(val) if hasattr(widget, "setValue") else widget.setText(
            str(val)
        )
    view.list_view_target.selectionModel().select(
        view.vk.target.index(0, 0),
        QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect,
    )
    view.sb_g2_bayesian_qidx.setValue(0)
    assert view.vk.get_xf_list([0])[0].qmap.is_default
    _finish_real(view, _launch_fit(view))  # the viewer itself passes the flag
    window = _open_g2_window(view)
    assert PLACEHOLDER_QMAP_WARNING in window.windowTitle()
    out = tmp_path / "diagnosis.csv"
    _export(window, monkeypatch, out, "_export_fit_data")
    assert PLACEHOLDER_QMAP_WARNING in out.read_text()
