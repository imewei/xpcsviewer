"""Regression tests for review round 7 (state and records).

R7-06  a removed or replaced averaging job leaves no progress record behind, even
       when worker messages are still queued for it
R7-07  the session restores the two-time Q bin by its absolute processed-bin ID in
       the file it was chosen on, never by its position in the combo
R7-10  a file running on the placeholder Q map (``QMap.is_default``) says so in
       every export that carries Q values, and on the status bar
Sweep  (a) the Intensity time axis and its FFT come from ONE interval, also with
       stride/avg > 1; (b) the async two-time path is not reachable from the GUI;
       (c) a SAXS background that is refused is reported on the status bar;
       (e) the session fields without a widget really have no control

Real offscreen ``XpcsViewer`` / ``ViewerKernel`` over real h5py-written files.
"""

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pyqtgraph as pg
import pytest
from qtpy import QtWidgets

from tests.unit import test_review_r3_average as r3avg
from tests.unit import test_review_r3_maps_roi as r3
from tests.unit import test_review_r5_kernel as r5
from tests.unit import test_review_r6_kernel_units as ku
from tests.unit.test_review_r4_cli_average import _write_twotime
from tests.unit.test_review_r4_viewer_gui import _tab
from tests.unit.test_review_r6_file_intt import _sinusoid_file
from tests.unit.test_review_r6_viewer_gui import _make_env, _select
from xpcsviewer.fileIO.hdf_reader import clear_connection_pool
from xpcsviewer.gui.state.session_manager import SessionManager
from xpcsviewer.module import intt
from xpcsviewer.utils.qmap_notice import (
    PLACEHOLDER_QMAP_WARNING,
    has_placeholder_qmap,
)
from xpcsviewer.viewer_kernel import ViewerKernel
from xpcsviewer.xpcs_file import XpcsFile

pytestmark = pytest.mark.unit

K = r3.K
PKG = Path(__file__).resolve().parents[2] / "xpcsviewer"


@pytest.fixture(autouse=True)
def _fresh_pool():
    clear_connection_pool()
    yield
    clear_connection_pool()


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# --------------------------------------------------------------------- R7-06
@pytest.fixture
def average_env(qtbot, monkeypatch, tmp_path):
    env = _make_env(
        qtbot,
        monkeypatch,
        tmp_path,
        {
            "scan0.h5": lambda p: r3avg._write_nexus(p, g2=1.0),
            "scan1.h5": lambda p: r3avg._write_nexus(p, g2=1.02),
        },
    )
    view = env.view
    view.avg_save_path.setText(str(tmp_path / "out"))
    view.avg_save_name.setText("average.h5")
    view.bx_avg_G2IPIF.setChecked(False)
    view.bx_avg_saxs.setChecked(False)
    view.bx_avg_g2g2err.setChecked(True)
    view.avg_blmin.setValue(0)
    view.avg_blmax.setValue(2)
    yield env
    view.timer.stop()
    view.thread_pool.waitForDone(10000)


def test_r7_06_removing_a_running_job_leaves_no_progress_record(average_env):
    view = average_env.view
    view.submit_job()
    jid = view.vk.avg_worker.jid
    view.start_avg_job()
    view.avg_job_pop.click()  # the job is still queued / running
    assert view.vk.avg_worker is None
    assert view.thread_pool.waitForDone(10000)
    QtWidgets.QApplication.processEvents()  # deliver the worker's queued messages
    assert view.vk.avg_worker_active == {}
    assert jid in view.vk._retired_avg_jobs


def test_r7_06_a_late_message_for_a_retired_job_is_ignored(average_env):
    view = average_env.view
    kernel = view.vk
    view.submit_job()
    jid = kernel.avg_worker.jid
    kernel.update_avg_values((jid, 1.0))
    assert kernel.avg_worker_active[jid][0] == 1  # a live job keeps its record
    view.remove_avg_job()
    assert jid not in kernel.avg_worker_active
    kernel.update_avg_values((jid, 2.0))  # queued before the removal, delivered after
    assert jid not in kernel.avg_worker_active


def test_r7_06_replacing_a_job_retires_the_old_one(average_env):
    view = average_env.view
    kernel = view.vk
    view.submit_job()
    first = kernel.avg_worker.jid
    kernel.update_avg_values((first, 1.0))
    kernel.add_target(["scan0.h5", "scan1.h5"])
    view.submit_job()  # replaces the first job in the list
    second = kernel.avg_worker.jid
    assert second != first
    assert first not in kernel.avg_worker_active
    kernel.update_avg_values((first, 3.0))
    assert first not in kernel.avg_worker_active
    kernel.update_avg_values((second, 1.0))
    assert kernel.avg_worker_active[second][0] == 1


# --------------------------------------------------------------------- R7-07
def _reprocess(path, order, value):
    """Replace ``path`` by a file whose processed bins come in another order."""
    replacement = path.with_name("replacement.h5")
    _write_twotime(replacement, [0.01, 0.02], value=value)
    with h5py.File(replacement, "a") as f:
        del f[K["c2_processed_bins"]]
        f[K["c2_processed_bins"]] = order
    os.replace(replacement, path)


def _twotime_session(qtbot, monkeypatch, tmp_path):
    """A real viewer that chose processed bin 2 (position 1) and saved the session."""
    env = _make_env(
        qtbot,
        monkeypatch,
        tmp_path,
        {"scan.h5": lambda p: _write_twotime(p, [0.01, 0.02], value=5)},
        real_session=True,
    )
    first = env.view
    _select(first, 0)
    first.tabWidget.setCurrentIndex(_tab(env.xv, "twotime"))
    first.twotime_correct_diag.setChecked(False)
    first.plot_twotime()
    first.comboBox_twotime_selection.setCurrentIndex(1)
    first.plot_twotime()
    assert first.comboBox_twotime_selection.currentText().startswith("qbin=2")
    return env


def _restored(qtbot, env):
    second = env.xv.XpcsViewer(path=str(env.data))
    qtbot.addWidget(second)
    second.twotime_correct_diag.setChecked(False)
    second.plot_twotime()
    return second


def _c2_values(view):
    return np.unique(view.mp_2t_hdls["tt"].image)


def test_r7_07_session_stores_the_bin_identity(qtbot, monkeypatch, tmp_path):
    env = _twotime_session(qtbot, monkeypatch, tmp_path)
    state = env.view._collect_session_state()
    params = state.analysis_params
    assert params.twotime_selected_q == 1  # the position stays, as a legacy value
    assert params.twotime_selected_bin == 2
    assert params.twotime_source_file == os.path.abspath(env.data / "scan.h5")
    assert env.view.session_manager.save_session(state)
    saved = json.loads((env.root / "session.json").read_text())["analysis_params"]
    assert saved["twotime_selected_bin"] == 2


def test_r7_07_restore_follows_the_bin_when_the_file_is_reprocessed(
    qtbot, monkeypatch, tmp_path
):
    env = _twotime_session(qtbot, monkeypatch, tmp_path)
    assert env.view.session_manager.save_session(env.view._collect_session_state())
    env.view.close()
    _reprocess(env.data / "scan.h5", [2, 1], value=7)  # bin 2 is now position 0

    second = _restored(qtbot, env)
    assert second.comboBox_twotime_selection.currentText().startswith("qbin=2")
    np.testing.assert_allclose(_c2_values(second), 14.0)  # bin 2 holds 7 * 2


def test_r7_07_restore_still_selects_the_same_bin_in_an_unchanged_file(
    qtbot, monkeypatch, tmp_path
):
    env = _twotime_session(qtbot, monkeypatch, tmp_path)
    assert env.view.session_manager.save_session(env.view._collect_session_state())
    env.view.close()
    second = _restored(qtbot, env)
    assert second.comboBox_twotime_selection.currentIndex() == 1
    np.testing.assert_allclose(_c2_values(second), 10.0)


def test_r7_07_an_older_session_without_identity_falls_back_to_the_position(
    qtbot, monkeypatch, tmp_path
):
    env = _twotime_session(qtbot, monkeypatch, tmp_path)
    assert env.view.session_manager.save_session(env.view._collect_session_state())
    env.view.close()
    path = env.root / "session.json"
    data = json.loads(path.read_text())
    del data["analysis_params"]["twotime_source_file"]
    del data["analysis_params"]["twotime_selected_bin"]
    path.write_text(json.dumps(data))
    _reprocess(env.data / "scan.h5", [2, 1], value=7)

    second = _restored(qtbot, env)
    # the documented legacy behaviour: position 1 of the file as it is now
    assert second.comboBox_twotime_selection.currentIndex() == 1
    assert second.comboBox_twotime_selection.currentText().startswith("qbin=1")


def test_r7_07_a_bin_that_is_gone_selects_the_first_bin(qtbot, monkeypatch, tmp_path):
    env = _twotime_session(qtbot, monkeypatch, tmp_path)
    assert env.view.session_manager.save_session(env.view._collect_session_state())
    env.view.close()
    _reprocess(env.data / "scan.h5", [1], value=7)  # bin 2 is no longer processed

    second = _restored(qtbot, env)
    assert second.comboBox_twotime_selection.currentIndex() == 0
    assert second.comboBox_twotime_selection.currentText().startswith("qbin=1")
    np.testing.assert_allclose(_c2_values(second), 7.0)


def test_r7_07_the_saved_bin_belongs_to_its_file_not_to_the_selected_one(
    qtbot, monkeypatch, tmp_path
):
    env = _make_env(
        qtbot,
        monkeypatch,
        tmp_path,
        {
            "a.h5": lambda p: _write_twotime(p, [0.01, 0.02], value=5),
            "b.h5": lambda p: _write_twotime(p, [0.01, 0.02], value=3),
        },
        real_session=True,
    )
    first = env.view
    _select(first, 1)  # b.h5 is shown; its bin 2 (position 1) is chosen
    first.tabWidget.setCurrentIndex(_tab(env.xv, "twotime"))
    first.twotime_correct_diag.setChecked(False)
    first.plot_twotime()
    first.comboBox_twotime_selection.setCurrentIndex(1)
    first.plot_twotime()
    state = first._collect_session_state()
    assert state.analysis_params.twotime_source_file == os.path.abspath(
        env.data / "b.h5"
    )
    assert first.session_manager.save_session(state)
    first.close()
    second = env.xv.XpcsViewer(path=str(env.data))
    qtbot.addWidget(second)
    memory = second._twotime_qbin_memory
    # the bin is remembered for b.h5 (by ID, with its bins) and not for a.h5
    assert memory[str(env.data / "b.h5")] == 1
    assert str(env.data / "a.h5") not in memory


def test_r7_07_malformed_identity_fields_load_as_unknown(monkeypatch, tmp_path):
    from xpcsviewer.gui.state import session_manager

    path = tmp_path / "session.json"
    monkeypatch.setattr(session_manager, "get_session_path", lambda: path)
    path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "analysis_params": {
                    "twotime_source_file": 7,
                    "twotime_selected_bin": True,
                },
            }
        )
    )
    params = SessionManager().load_session().analysis_params
    assert params.twotime_source_file == ""
    assert params.twotime_selected_bin == -1


# --------------------------------------------------------------------- R7-10
def _placeholder_file(path, **kw):
    """A two-bin G2 file whose Q-map group is missing (fallback calibration)."""
    r3._write_nexus(path, **kw)
    with h5py.File(path, "r+") as f:
        del f["/xpcs/qmap"]
    return path


def _kernel(tmp_path, *names):
    kernel = ViewerKernel(str(tmp_path))
    kernel.add_target(list(names))
    return kernel


def test_r7_10_the_flag_reaches_the_file(tmp_path):
    r3._write_nexus(tmp_path / "ok.h5")
    xf = XpcsFile(str(_placeholder_file(tmp_path / "bare.h5")))
    ok = XpcsFile(str(tmp_path / "ok.h5"))
    assert has_placeholder_qmap(xf) and not has_placeholder_qmap(ok)
    assert not has_placeholder_qmap(object())  # a stand-in is no placeholder


def test_r7_10_g2_export_warns_and_keeps_its_numbers(tmp_path):
    _placeholder_file(tmp_path / "bare.h5")
    r3._write_nexus(tmp_path / "ok.h5")
    out = tmp_path / "out"
    out.mkdir()
    _kernel(tmp_path, "bare.h5", "ok.h5").export_g2(str(out))
    bare = (out / "bare_g2_data.txt").read_text().splitlines()
    ok = (out / "ok_g2_data.txt").read_text().splitlines()
    assert bare[0] == f"# {PLACEHOLDER_QMAP_WARNING}"
    assert not any("WARNING" in line for line in ok)
    # the numeric table is the one a calibrated file of the same data has
    np.testing.assert_array_equal(
        np.loadtxt(out / "bare_g2_data.txt"), np.loadtxt(out / "ok_g2_data.txt")
    )


def test_r7_10_g2_fit_exports_warn_too(tmp_path):
    ku._g2_file(tmp_path / "bare.h5", [0.1, 0.2, 0.3])
    with h5py.File(tmp_path / "bare.h5", "r+") as f:
        del f["/xpcs/qmap"]
    kernel = ku._fitted_kernel(tmp_path, ["bare.h5"])
    out = tmp_path / "out"
    out.mkdir()
    kernel.export_g2(str(out))
    for suffix in ("g2_data", "g2_fitting", "g2_fitted_curves"):
        text = (out / f"bare_{suffix}.txt").read_text()
        assert f"# {PLACEHOLDER_QMAP_WARNING}" in text, suffix


def test_r7_10_saxs1d_export_warns(tmp_path):
    _placeholder_file(tmp_path / "bare.h5")
    r3._write_nexus(tmp_path / "ok.h5")
    out = tmp_path / "out"
    out.mkdir()
    kernel = _kernel(tmp_path, "bare.h5", "ok.h5")
    kernel.export_saxs_1d(SimpleNamespace(get_roi_list=list), str(out))
    assert PLACEHOLDER_QMAP_WARNING in (out / "bare_saxs1d.txt").read_text()
    assert PLACEHOLDER_QMAP_WARNING not in (out / "ok_saxs1d.txt").read_text()


def test_r7_10_diffusion_export_warns(tmp_path, qapp):
    ku._g2_file(tmp_path / "bare.h5", [0.1, 0.2, 0.3])
    ku._g2_file(tmp_path / "ok.h5", [0.1, 0.2, 0.3])
    with h5py.File(tmp_path / "bare.h5", "r+") as f:
        del f["/xpcs/qmap"]
    kernel = ku._fitted_kernel(tmp_path, ["bare.h5", "ok.h5"])
    ku._tauq(kernel, [0, 1])
    out = tmp_path / "out"
    out.mkdir()
    kernel.export_diffusion(str(out), rows=[0, 1])
    assert PLACEHOLDER_QMAP_WARNING in (out / "bare_diffusion_fitting.txt").read_text()
    assert (
        PLACEHOLDER_QMAP_WARNING not in (out / "ok_diffusion_fitting.txt").read_text()
    )
    summary = (out / "diffusion_summary.txt").read_text()
    assert f"{PLACEHOLDER_QMAP_WARNING}: bare.h5" in summary
    assert "ok.h5:" not in summary.split("file_label")[0]


def test_r7_10_every_module_that_claims_an_output_is_accounted_for():
    """The modules that write files through ``ExportGuard.claim`` and what each
    does about a placeholder Q map. ``open_gaps`` are NOT handled yet (their
    code is owned elsewhere); a new writer fails this test until it is listed."""
    handled = {"viewer_kernel.py", "xpcs_file.py", "xpcs_viewer.py"}
    no_q_values = {"gui/g2_regroup_dialog.py"}  # an external Q map, never a default
    open_gaps = {"fitting/viz.py", "gui/bayesian_diagnosis.py", "cli/twotime_batch.py"}
    claiming = {
        p.relative_to(PKG).as_posix()
        for p in PKG.rglob("*.py")
        if ".claim(" in p.read_text() and "simplemask" not in p.parts  # masks only
    }
    assert claiming == handled | no_q_values | open_gaps
    for name in handled:
        assert "has_placeholder_qmap" in (PKG / name).read_text(), name


def test_r7_10_bayesian_export_marks_the_figure_and_the_status_bar(
    qtbot, monkeypatch, tmp_path
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from xpcsviewer.fitting import viz

    env = _make_env(
        qtbot,
        monkeypatch,
        tmp_path,
        {
            "bare.h5": lambda p: _placeholder_file(p),
            "ok.h5": lambda p: r3._write_nexus(p),
        },
    )
    view = env.view
    view.vk.get_xf_list()  # load both files
    out = tmp_path / "bayes"
    out.mkdir()
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getExistingDirectory", lambda *a, **k: str(out)
    )
    summary = {
        "fit_val": np.zeros((2, 2, len(viz._SINGLE_COLS))),
        "q_val": np.array([0.01, 0.02]),
        "fit_func": "single",
    }
    for name, expect in (("bare.h5", True), ("ok.h5", False)):
        snapshot = SimpleNamespace(
            fname=str(env.data / name), summary=summary, results=None
        )
        fig = plt.figure()
        fig.add_subplot().plot([0, 1])
        for stale in out.iterdir():
            stale.unlink()
        view._export_bayesian(snapshot, fig)
        assert (out / "bayesian_all_q.png").exists()
        assert (PLACEHOLDER_QMAP_WARNING in view.statusbar.currentMessage()) is expect
        # the dialog's own figure is left as it was
        assert not [t for t in fig.texts if PLACEHOLDER_QMAP_WARNING in t.get_text()]
        plt.close(fig)


def _status_env(qtbot, monkeypatch, tmp_path):
    return _make_env(
        qtbot,
        monkeypatch,
        tmp_path,
        {
            "bare.h5": lambda p: _placeholder_file(p),
            "ok.h5": lambda p: r3._write_nexus(p),
        },
    )


def test_r7_10_status_bar_warns_while_a_placeholder_file_is_selected(
    qtbot, monkeypatch, tmp_path
):
    env = _status_env(qtbot, monkeypatch, tmp_path)
    view = env.view
    view.tabWidget.setCurrentIndex(_tab(env.xv, "saxs_1d"))  # a synchronous tab
    _select(view, 1)
    view.update_plot()
    label = getattr(view, "_placeholder_qmap_label", None)
    assert label is None or label.isHidden()
    _select(view, 0)
    view.update_plot()  # the plot loads the file, then the notice looks at it
    label = view._placeholder_qmap_label
    assert label is not None and not label.isHidden()
    assert PLACEHOLDER_QMAP_WARNING in label.text() and "bare.h5" in label.text()
    assert "ok.h5" not in label.text()
    _select(view, 1)
    view.update_plot()
    assert label.isHidden()


def test_r7_10_the_notice_loads_no_file(qtbot, monkeypatch, tmp_path):
    env = _status_env(qtbot, monkeypatch, tmp_path)
    view = env.view
    view.vk.cache.clear()
    view.tabWidget.setCurrentIndex(_tab(env.xv, "average"))
    view.update_plot()
    view._update_placeholder_qmap_notice()
    assert view.vk.cache == {}  # every tab switch would otherwise load every target


# --------------------------------------------------------------- sweep (a)
@pytest.mark.parametrize(("stride", "avg"), [(1, 1), (3, 2)])
def test_sweep_a_intensity_time_axis_and_fft_share_one_interval(
    tmp_path, qapp, stride, avg
):
    path = _sinusoid_file(tmp_path / "s.h5", 0.25)
    with h5py.File(path, "r+") as f:
        for key, value in (("stride_frame", stride), ("avg_frame", avg)):
            del f[K[key]]
            f[K[key]] = value
    xf = XpcsFile(str(path))
    dt = 0.25 * stride * avg
    assert float(xf.t0) == pytest.approx(dt)
    widget = pg.GraphicsLayoutWidget()
    intt.plot([xf], widget, xlabel="Time (second)")
    t, _ = widget.getItem(0, 0).listDataItems()[0].getData()
    freq, _amp = widget.getItem(1, 0).listDataItems()[0].getData()
    assert float(t[1] - t[0]) == pytest.approx(dt)
    assert float(freq[1] - freq[0]) == pytest.approx(1.0 / (len(t) * dt))


# --------------------------------------------------------------- sweep (b)
def test_sweep_b_the_async_two_time_path_is_not_reachable_from_the_gui(
    qtbot, monkeypatch, tmp_path
):
    env = _make_env(
        qtbot,
        monkeypatch,
        tmp_path,
        {"scan.h5": lambda p: _write_twotime(p, [0.01, 0.02], value=5)},
    )
    view = env.view
    started = []
    monkeypatch.setattr(
        env.xv.XpcsViewer, "update_plot_async", lambda self, tab: started.append(tab)
    )
    _select(view, 0)
    view.tabWidget.setCurrentIndex(_tab(env.xv, "twotime"))
    view.twotime_correct_diag.setChecked(False)
    view.update_plot()
    view.comboBox_twotime_selection.setCurrentIndex(1)
    assert "twotime" not in started
    # and no code path names the async two-time entry except its own dispatch
    source = (PKG / "xpcs_viewer.py").read_text()
    assert source.count("plot_twotime_async") == 1
    tree = ast.parse(source)
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "update_plot_async"
    ]
    assert all(
        not (
            call.args
            and isinstance(call.args[0], ast.Constant)
            and call.args[0].value == "twotime"
        )
        for call in calls
    )


# --------------------------------------------------------------- sweep (c)
def _saxs_unit_file(path, unit: bytes, value):
    r3._write_nexus(path)
    with h5py.File(path, "r+") as f:
        f[K["saxs_1d"]][:] = value
        del f[K["map_units"]]
        f[K["map_units"]] = [unit, b"deg"]
    return path


def test_sweep_c_a_refused_background_is_reported_on_the_status_bar(
    tmp_path, qapp, monkeypatch
):
    _saxs_unit_file(tmp_path / "sample.h5", b"1/A", 10.0)
    _saxs_unit_file(tmp_path / "bkg.h5", b"1/nm", 1.0)
    status = QtWidgets.QStatusBar()
    kernel = ViewerKernel(str(tmp_path), status)
    kernel.add_target(["sample.h5"])
    kernel.select_bkgfile(str(tmp_path / "bkg.h5"))
    kw = {"subtract_background": True, "weight": 1.0}
    drawn = r5._plot(kernel, [0], monkeypatch, **kw)
    assert drawn[0][0] == [10.0] * 3  # the sample trace is unchanged ...
    message = status.currentMessage()  # ... and the user is told why
    assert "Background NOT subtracted" in message and "sample.h5" in message
    assert "axis units" in message


def test_sweep_c_a_compatible_background_shows_no_warning(tmp_path, qapp, monkeypatch):
    _saxs_unit_file(tmp_path / "sample.h5", b"1/A", 10.0)
    _saxs_unit_file(tmp_path / "bkg.h5", b"1/A", 1.0)
    status = QtWidgets.QStatusBar()
    kernel = ViewerKernel(str(tmp_path), status)
    kernel.add_target(["sample.h5"])
    kernel.select_bkgfile(str(tmp_path / "bkg.h5"))
    drawn = r5._plot(kernel, [0], monkeypatch, subtract_background=True, weight=1.0)
    assert drawn[0][0] == [9.0] * 3
    assert status.currentMessage() == ""


# --------------------------------------------------------------- sweep (e)
def test_sweep_e_session_fields_without_a_widget_have_no_control(
    qtbot, monkeypatch, tmp_path
):
    view = _make_env(
        qtbot, monkeypatch, tmp_path, {"a.h5": lambda p: r3._write_nexus(p)}
    ).view
    controls = {
        w.objectName()
        for kind in (
            QtWidgets.QAbstractButton,
            QtWidgets.QComboBox,
            QtWidgets.QAbstractSpinBox,
            QtWidgets.QSlider,
        )
        for w in view.findChildren(kind)
    }
    patterns = {
        "intensity_t_normalize": r"intt.*norm|norm.*intt|int.*normali[sz]",
        "diffusion_model": r"diff.*model|model.*diff|tauq.*model",
        "twotime_symmetric": r"symmetr",
        "qmap_show_rings": r"ring",
        "saxs2d_log_scale": r"saxs2d.*log|log.*saxs2d",
        "saxs1d_log_x_y": r"saxs1d.*log|log.*saxs1d",
        "g2_q_index": r"g2_q(idx|index)$",
    }
    for field, pattern in patterns.items():
        hits = [n for n in controls if re.search(pattern, n, re.I)]
        assert hits == [], (field, hits)
