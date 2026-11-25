"""Regression tests for review round 6 in the viewer GUI layer.

R6-01  the Mask Editor starts from the file's EFFECTIVE mask and an applied mask
       never re-enables a pixel the destination's own canonical mask excludes
R6-05  a fresh Mask Editor opens the highlighted target (other sites that take
       "the current file" are listed in ``test_r6_05_*scan``)
R6-11  after a replaced two-time file is reloaded, every panel is drawn from ONE
       bin resolved against the loaded revision
R6-13  a Bayesian G2 fit is refused (single-Q) / not sampled (all-Q) when the
       selection cannot identify the model; one helper for both paths
R6-14  the diagnosis window's Export Traces writes a non-converged posterior only
       marked ``bin_status=not_converged``
R6-15  no key sequence is registered twice in one context
R6-16  the session saves and restores analysis widgets through ONE table

The viewer is the real offscreen ``XpcsViewer`` over real h5py-written files.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtTest import QTest

from tests.unit import test_review_r3_maps_roi as r3
from tests.unit.test_review_r3_fit_status import _write_file
from tests.unit.test_review_r4_cli_average import _write_twotime
from tests.unit.test_review_r4_viewer_gui import (  # noqa: F401  (fixtures)
    _FakeWorker,
    _idle,
    _tab,
    viewer_env,
)
from tests.unit.test_review_r5_bayes_gui import _real_result
from xpcsviewer.fileIO.hdf_reader import clear_connection_pool

pytestmark = pytest.mark.unit

PKG = Path(__file__).resolve().parents[2] / "xpcsviewer"
K = r3.K


@pytest.fixture(autouse=True)
def _fresh_pool():
    clear_connection_pool()
    yield
    clear_connection_pool()


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _make_env(qtbot, monkeypatch, tmp_path, writers, *, real_session=False):
    """A real offscreen viewer over ``tmp/data`` holding ``writers`` (name -> fn)."""
    from xpcsviewer.gui.state import preferences, recent_paths, session_manager
    from xpcsviewer.gui.state.session_manager import (
        AnalysisParameters,
        SessionManager,
    )

    xv = sys.modules["xpcsviewer.xpcs_viewer"]
    data = tmp_path / "data"
    data.mkdir()
    for name, write in writers.items():
        write(data / name)
    monkeypatch.setattr(
        session_manager, "get_session_path", lambda: tmp_path / "session.json"
    )
    monkeypatch.setattr(
        preferences, "get_preferences_path", lambda: tmp_path / "preferences.json"
    )
    monkeypatch.setattr(xv, "home_dir", str(tmp_path))
    monkeypatch.setattr(recent_paths.RecentPathsManager, "add_path", lambda s, p: None)
    monkeypatch.setattr(xv.XpcsViewer, "closeEvent", lambda self, ev: ev.accept())
    monkeypatch.setattr(xv.XpcsViewer, "_maybe_prompt_start_path", lambda self: None)
    if not real_session:
        session = SimpleNamespace(
            data_path=str(data),
            target_files=[],
            active_tab=-1,
            window_geometry=SimpleNamespace(maximized=True),
            analysis_params=AnalysisParameters(),
        )
        monkeypatch.setattr(SessionManager, "load_session", lambda self: session)
        monkeypatch.setattr(SessionManager, "get_warnings", lambda self: [])
    view = xv.XpcsViewer(path=str(data))
    qtbot.addWidget(view)
    view.vk.add_target(list(writers))
    view.update_box(view.vk.target, mode="target")
    return SimpleNamespace(view=view, root=tmp_path, data=data, xv=xv)


def _select(view, *rows):
    sel = view.list_view_target.selectionModel()
    sel.clearSelection()
    for row in rows:
        sel.select(
            view.vk.target.index(row, 0),
            QtCore.QItemSelectionModel.SelectionFlag.Select,
        )


def _write_detector(path, shape, masked, centre=(2.0, 2.0)):
    """Real NeXus file with a ``shape`` detector whose canonical mask rejects
    ``masked`` pixels; the rejected pixels hold a huge intensity (5000)."""
    r3._write_nexus(path)
    image = np.full(shape, 20.0)
    mask = np.ones(shape, dtype=np.uint8)
    for r, c in masked:
        image[r, c] = 5000.0
        mask[r, c] = 0
    with h5py.File(path, "a") as f:
        for name, value in {
            "saxs_2d": image[None],
            "mask": mask,
            "dqmap": np.ones(shape, dtype=np.int32),
            "sqmap": np.ones(shape, dtype=np.int32),
            "bcx": centre[0],
            "bcy": centre[1],
        }.items():
            del f[K[name]]
            f[K[name]] = value


@pytest.fixture
def mask_env(qtbot, monkeypatch, tmp_path):
    env = _make_env(
        qtbot,
        monkeypatch,
        tmp_path,
        {
            "big.h5": lambda p: _write_detector(p, (16, 20), [(0, 0)]),
            "small.h5": lambda p: _write_detector(p, (6, 9), [(1, 1)]),
        },
    )
    yield env
    window = env.view._simplemask_window
    if window is not None:
        window.close()


# --------------------------------------------------------------------------
# R6-01
# --------------------------------------------------------------------------
def test_r6_01_editor_is_seeded_with_the_effective_mask(mask_env):
    view = mask_env.view
    _select(view, 0)
    image, meta = view._get_simplemask_data()
    xf = view.vk.get_xf_list([0])[0]
    assert image.shape == (16, 20)
    assert meta["mask"].shape == image.shape and meta["mask"].dtype == bool
    assert not meta["mask"][0, 0] and meta["mask"].sum() == image.size - 1
    # the effective mask, not the canonical one: an imported mask shows too
    imported = np.ones(image.shape, dtype=bool)
    imported[3, 4] = False
    xf.mask = imported
    _, meta = view._get_simplemask_data()
    assert not meta["mask"][3, 4]
    # a private copy: the editor cannot edit the viewer's array in place
    meta["mask"][5, 5] = False
    assert xf.mask[5, 5]


def test_r6_01_untouched_apply_does_not_reenable_masked_pixels(mask_env):
    view = mask_env.view
    _select(view, 0)
    xf = view.vk.get_xf_list([0])[0]
    wedge = {"sl_type": "Pie", "polygon": [(0, 0), (0, 2), (2, 2), (2, 0)]}
    before = float(xf.get_roi_data(wedge)[1][0])
    view.open_simplemask()
    editor = view._simplemask_window
    assert not xf.mask[0, 0]
    editor.action_apply_viewer.trigger()  # no edits at all
    assert not xf.mask[0, 0] and not xf.qmap.mask[0, 0]
    assert float(xf.get_roi_data(wedge)[1][0]) == before == 20.0


def test_r6_01_import_never_reenables_a_pixel_the_targets_own_mask_excludes(mask_env):
    view = mask_env.view
    xf_big, xf_small = view.vk.get_xf_list([0, 1])
    # two targets of ONE detector shape with different canonical masks
    other = np.ones((16, 20), dtype=np.int32)
    other[7, 7] = 0
    xf_small_like = SimpleNamespace(
        fname="same-shape.h5",
        saxs_2d=np.zeros((16, 20)),
        qmap=SimpleNamespace(mask=other),
        mask=other,
    )
    view.vk.get_xf_list = lambda *a, **k: [xf_big, xf_small_like, xf_small]
    editor_mask = np.ones((16, 20), dtype=bool)
    editor_mask[2, 3] = False  # an edit
    view.import_mask(editor_mask)
    assert not xf_big.mask[0, 0] and not xf_big.mask[2, 3] and xf_big.mask[7, 7]
    assert not xf_small_like.mask[7, 7] and not xf_small_like.mask[2, 3]
    assert xf_small_like.mask[0, 0]  # not excluded by ITS canonical mask
    assert xf_small.mask.shape == (6, 9) and not xf_small.mask[1, 1]  # untouched


# --------------------------------------------------------------------------
# R6-05
# --------------------------------------------------------------------------
def test_r6_05_fresh_editor_opens_the_highlighted_target(mask_env):
    view = mask_env.view
    _select(view, 1)
    view.open_simplemask()
    editor = view._simplemask_window
    assert editor.kernel.detector_image.shape == (6, 9)
    # round-1 rule: a visible editor is focused, never reloaded
    _select(view, 0)
    view.open_simplemask()
    assert editor.kernel.detector_image.shape == (6, 9)


def test_r6_05_row_zero_is_a_selection_and_nothing_selected_means_first(mask_env):
    view = mask_env.view
    _select(view, 0)
    assert view._get_simplemask_data()[0].shape == (16, 20)
    _select(view)  # nothing highlighted: the first valid target
    assert view.get_selected_rows() == [0, 1]
    assert view._get_simplemask_data()[0].shape == (16, 20)
    _select(view, 1)
    assert view._get_simplemask_data()[0].shape == (6, 9)


def test_r6_05_no_site_takes_the_first_file_without_consulting_the_selection():
    """Every ``get_xf_list`` result that is indexed ``[0]`` was asked for the
    selected rows (or a named subset), never for the bare target list."""
    tree = ast.parse((PKG / "xpcs_viewer.py").read_text())
    offenders = []
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef):
            continue
        bare = set()
        for node in ast.walk(func):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "attr", "") == "get_xf_list"
                and not node.value.args
                and not any(k.arg in ("rows", None) for k in node.value.keywords)
            ):
                bare |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        for node in ast.walk(func):
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id in bare
                and isinstance(node.slice, ast.Constant)
                and node.slice.value == 0
            ):
                offenders.append(f"{func.name}:{node.lineno}")
    assert offenders == []


# --------------------------------------------------------------------------
# R6-11
# --------------------------------------------------------------------------
@pytest.fixture
def tt_env(qtbot, monkeypatch, tmp_path):
    env = _make_env(
        qtbot,
        monkeypatch,
        tmp_path,
        {"target.h5": lambda p: _write_twotime(p, [0.01, 0.02], value=5)},
    )
    view = env.view
    _select(view, 0)
    view.tabWidget.setCurrentIndex(_tab(env.xv, "twotime"))
    view.twotime_correct_diag.setChecked(False)
    return env


_TT_NAME = "target.h5"


def _replace_with(env, writer):
    tmp = env.root / "replacement.h5"
    writer(tmp)
    os.replace(tmp, env.data / "target.h5")


def _c2_values(view):
    return np.unique(np.asarray(view.mp_2t_hdls["tt"].image)).tolist()


def test_r6_11_reload_resolves_the_remembered_bin_before_any_panel_draws(tt_env):
    view = tt_env.view
    view.plot_twotime()
    view.comboBox_twotime_selection.setCurrentIndex(1)
    view.plot_twotime()
    assert _c2_values(view) == [10.0]
    assert view.comboBox_twotime_selection.currentText().startswith("qbin=2")

    _replace_with(tt_env, lambda p: _write_twotime(p, [0.01], value=33))
    view.reload_source()

    assert list(view.vk.get_xf_list([0])[0].c2_processed_bins) == [1]
    assert view.comboBox_twotime_selection.count() == 1
    assert view.comboBox_twotime_selection.currentText().startswith("qbin=1")
    assert view.horizontalSlider_twotime_selection.value() == 0
    assert _c2_values(view) == [33.0]  # the new C2, not the old one
    assert view._twotime_qbin_memory[str(tt_env.data / _TT_NAME)] == 0


def test_r6_11_remembered_bin_follows_its_id_into_the_new_revision(tt_env):
    view = tt_env.view
    view.plot_twotime()
    view.comboBox_twotime_selection.setCurrentIndex(1)  # bin 2
    view.plot_twotime()

    def two_bins_reordered(path):
        _write_twotime(path, [0.01, 0.02], value=7)
        with h5py.File(path, "a") as f:
            del f[K["c2_processed_bins"]]
            f[K["c2_processed_bins"]] = np.array([2, 1])

    _replace_with(tt_env, two_bins_reordered)
    view.reload_source()
    # bin 2 is now the FIRST processed bin; the old ordinal 1 would be bin 1
    assert view.comboBox_twotime_selection.currentIndex() == 0
    assert view.comboBox_twotime_selection.currentText().startswith("qbin=2")
    assert view._twotime_qbin_memory[str(tt_env.data / _TT_NAME)] == 0
    assert _c2_values(view) == [14.0]  # c2_00002 = 7 * 2


def test_r6_11_file_without_processed_bins_clears_the_c2_panels(tt_env):
    view = tt_env.view
    view.plot_twotime()
    assert _c2_values(view) == [5.0]

    def no_bins(path):
        _write_twotime(path, [0.01], value=1)
        with h5py.File(path, "a") as f:
            del f[K["c2_processed_bins"]]
            f[K["c2_processed_bins"]] = np.zeros(0, dtype=int)

    _replace_with(tt_env, no_bins)
    view.reload_source()
    assert "no processed Q bins" in view.statusbar.currentMessage()
    assert view.mp_2t_hdls["tt"].image is None
    assert view.comboBox_twotime_selection.count() == 0


def test_r6_11_resolution_helper_edge_cases():
    from xpcsviewer.xpcs_viewer import _resolve_twotime_selection as resolve

    xf = SimpleNamespace(c2_processed_bins=np.array([4, 5, 6]))
    assert resolve(xf, 2, None) == (2, (4, 5, 6))
    assert resolve(xf, 3, None) == (0, (4, 5, 6))  # out of range
    assert resolve(xf, 1, (6, 5, 4)) == (1, (4, 5, 6))  # bin 5 stays bin 5
    assert resolve(xf, 0, (6, 5, 4)) == (2, (4, 5, 6))  # bin 6 moved to the end
    assert resolve(xf, 0, (9,)) == (0, (4, 5, 6))  # bin 9 is gone
    assert resolve(SimpleNamespace(c2_processed_bins=np.zeros(0)), 0, None) == (
        None,
        (),
    )
    assert resolve(SimpleNamespace(), 5, None) == (5, None)  # nothing to check


# --------------------------------------------------------------------------
# R6-13
# --------------------------------------------------------------------------
def _write_zero_delay(path):
    _write_file(path, np.r_[0.0, np.geomspace(0.005, 1, 25)])


@pytest.fixture
def bayes_env(qtbot, monkeypatch, tmp_path):
    from xpcsviewer.threading import bayesian_worker

    _FakeWorker.instances = []
    monkeypatch.setattr(bayesian_worker, "BayesianFitWorker", _FakeWorker)
    env = _make_env(qtbot, monkeypatch, tmp_path, {"a.h5": _write_zero_delay})
    view = env.view
    view.thread_pool = MagicMock()
    view.tabWidget.setCurrentIndex(_tab(env.xv, "g2_fitting"))
    _select(view, 0)
    view.sb_g2_bayesian_qidx.setValue(0)
    return env


def _set_g2_range(view, tmin, tmax, qmin=0.0, qmax=0.5):
    for name, val in {
        "g2_qmin": qmin,
        "g2_qmax": qmax,
        "g2_tmin": tmin,
        "g2_tmax": tmax,
    }.items():
        w = getattr(view, name)
        w.setValue(val) if hasattr(w, "setValue") else w.setText(str(val))


def test_r6_13_observation_rule_and_parameter_counts():
    from xpcsviewer.fitting import models
    from xpcsviewer.xpcs_viewer import (
        _G2_BAYES_FREE_PARAMS,
        _g2_bayesian_observations,
    )

    x = np.array([0.0, 0.1, -1.0, np.nan, 0.2, 0.3, 0.4])
    y = np.array([1.2, 1.2, 1.2, 1.2, np.nan, 1.1, 1.1])
    err = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1])
    valid, refusal = _g2_bayesian_observations(x, y, err, "single")
    assert valid.tolist() == [False, True, False, False, False, False, True]
    assert "only 2 observation" in refusal and "3 free parameters" in refusal
    valid, refusal = _g2_bayesian_observations(
        np.geomspace(0.01, 1, 5), np.full(5, 1.2), np.full(5, 0.1), "double"
    )
    assert valid.all() and refusal is None  # 5 observations for 5 parameters
    _, refusal = _g2_bayesian_observations(
        np.geomspace(0.01, 1, 4), np.full(4, 1.2), np.full(4, 0.1), "double"
    )
    assert refusal is not None
    # the table is the real models' count of free (non-deterministic) sites
    pytest.importorskip("numpyro")
    from numpyro import handlers

    for name, model in (
        ("single", models.single_exp_model),
        ("double", models.double_exp_model),
    ):
        trace = handlers.trace(handlers.seed(model, 0)).get_trace(
            np.ones(3), np.ones(3), np.ones(3)
        )
        free = [
            s
            for s in trace.values()
            if s["type"] == "sample" and not s.get("is_observed")
        ]
        assert len(free) == _G2_BAYES_FREE_PARAMS[name]


def test_r6_13_single_q_launch_on_the_zero_delay_only_is_refused(bayes_env):
    view = bayes_env.view
    _set_g2_range(view, 0.0, 0.001)  # exactly one observation: delay 0
    view._fit_g2_bayesian()
    assert _FakeWorker.instances == []
    assert not view._g2_bayesian_worker_active
    message = view.statusbar.currentMessage()
    assert "not started" in message and "only 0 observation" in message


def test_r6_13_single_q_launch_with_enough_positive_delays_still_runs(bayes_env):
    view = bayes_env.view
    _set_g2_range(view, 0.0, 0.02)  # 0, .005, ... : zero dropped, enough remain
    view._fit_g2_bayesian()
    worker = _FakeWorker.instances[-1]
    assert worker.launch.x.min() > 0 and len(worker.launch.x) >= 3
    # the double exponential needs five
    view._g2_bayesian_worker_active = False
    view.g2_fitting_function.setCurrentIndex(1)
    _set_g2_range(view, 0.0, 0.006)
    n = len(_FakeWorker.instances)
    view._fit_g2_bayesian()
    assert len(_FakeWorker.instances) == n
    assert "5 free parameters" in view.statusbar.currentMessage()


def test_r6_13_batch_launch_does_not_sample_unidentifiable_bins(bayes_env, monkeypatch):
    view = bayes_env.view
    _set_g2_range(view, 0.0, 0.001)
    view._fit_g2_bayesian_all()
    assert view._g2_batch_coordinator is None
    assert "No Q-bin can be fitted" in view.statusbar.currentMessage()

    # one bin keeps only 2 usable observations (g2 NaN elsewhere): that bin is
    # not sampled, the others are
    replacement = bayes_env.root / "replacement.h5"
    _write_zero_delay(replacement)
    with h5py.File(replacement, "r+") as f:
        column = f["/xpcs/multitau/normalized_g2"][:, 1]
        column[:] = np.nan  # unusable observations except for two delays
        column[[1, 2]] = 1.4
        f["/xpcs/multitau/normalized_g2"][:, 1] = column
    os.replace(replacement, bayes_env.data / "a.h5")
    view.reload_source()
    _select(view, 0)
    _set_g2_range(view, 0.0, 0.03)
    extracted = view._extract_g2_all_for_bayesian()
    assert sorted(s["q_idx"] for s in extracted["specs"]) == [0, 2]
    assert set(extracted["unidentifiable"]) == {1}
    assert "2 observation" in extracted["unidentifiable"][1]
    assert all(np.all(s["x"] > 0) for s in extracted["specs"])
    assert np.isnan(extracted["g2"][:, 1]).all()  # never in the fitted set

    # the batch finishes with results for the sampled bins only: the unsampled
    # one is a failed bin in the stored summary and the message says so
    from xpcsviewer.fitting.models import single_exp_func

    view._g2_batch_launch = {**extracted, "model_func": single_exp_func}
    finals: list[str] = []
    monkeypatch.setattr(
        view.progress_manager,
        "complete_operation",
        lambda op, success=True, final_message="": finals.append(final_message),
    )
    view._on_g2_batch_finished(
        {0: {"fit_result": _real_result(1)}, 2: {"fit_result": _real_result(2)}}
    )
    summary = extracted["xf"].bayesian_fit_summary
    assert summary["failed_mask"].tolist() == [False, True, False]
    assert "1 Q-bin(s) not sampled" in finals[-1]


# --------------------------------------------------------------------------
# R6-14
# --------------------------------------------------------------------------
class _FakeIdata:
    def __init__(self, sink, attrs=None):
        self.sink = sink
        self.attrs = dict(attrs or {})

    def copy(self):
        return _FakeIdata(self.sink, self.attrs)

    def to_netcdf(self, path):
        self.sink.append((Path(path).name, dict(self.attrs)))
        Path(path).write_bytes(b"netcdf")


def _diagnosis_window(qtbot, source, converged, sink):
    from xpcsviewer.fitting.models import single_exp_func
    from xpcsviewer.gui.bayesian_diagnosis import (
        BayesianDiagnosisWindow,
        DiagnosisIdentity,
    )

    window = BayesianDiagnosisWindow()
    qtbot.addWidget(window)
    result = _real_result(converged=converged)
    result.arviz_data = _FakeIdata(sink)
    window.update_results(
        result=result,
        model_func=single_exp_func,
        x_data=[0.1, 0.2, 0.3],
        y_data=[1.3, 1.2, 1.1],
        yerr=[0.01] * 3,
        q_value=0.1,
        identity=DiagnosisIdentity(str(source), None, 7, None),
    )
    return window


@pytest.mark.parametrize(
    ("converged", "status"), [(False, "not_converged"), (True, "ok")]
)
def test_r6_14_export_traces_is_marked_with_the_convergence_status(
    qtbot, monkeypatch, tmp_path, converged, status
):
    sink: list = []
    window = _diagnosis_window(qtbot, tmp_path / "scan.h5", converged, sink)
    destination = tmp_path / "traces.nc"
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (str(destination), "")
    )
    window._export_traces()
    assert [name for name, _ in sink] == ["traces.nc"]
    attrs = sink[0][1]
    assert attrs["bin_status"] == status and attrs["bin_id"] == 7
    assert attrs["q_value"] == pytest.approx(0.1)
    # the caller's FitResult is not modified by the tagging
    assert "bin_status" not in window._result.arviz_data.attrs
    shown = window.statusBar().currentMessage()
    assert ("not_converged" in shown) is (not converged)


def test_r6_14_a_posterior_that_cannot_be_tagged_is_not_written(
    qtbot, monkeypatch, tmp_path
):
    written: list = []
    window = _diagnosis_window(qtbot, tmp_path / "scan.h5", False, [])
    window._result.arviz_data = SimpleNamespace(
        to_netcdf=lambda path: written.append(path)  # no copy()/attrs: untaggable
    )
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *a, **k: (str(tmp_path / "t.nc"), ""),
    )
    window._export_traces()
    assert written == []
    assert "refused" in window.statusBar().currentMessage()


@pytest.mark.skipif(
    not any(importlib.util.find_spec(m) for m in ("h5netcdf", "netCDF4")),
    reason="no netCDF backend installed (not a dependency)",
)
def test_r6_14_real_netcdf_trace_file_carries_the_marker(qtbot, monkeypatch, tmp_path):
    from xpcsviewer.fitting import fit_single_exp
    from xpcsviewer.fitting.models import single_exp_func
    from xpcsviewer.fitting.viz import export_bayesian_diagnostics
    from xpcsviewer.gui.bayesian_diagnosis import (
        BayesianDiagnosisWindow,
        DiagnosisIdentity,
    )

    t = np.geomspace(0.005, 1, 25)
    y = 1.0 + 0.5 * np.exp(-2 * t / 0.1)
    result = fit_single_exp(
        t,
        y,
        np.full_like(y, 0.01),
        num_chains=1,  # no R-hat: never converged
        num_warmup=40,
        num_samples=40,
        random_seed=0,
        progress_bar=False,
    )
    assert not result.diagnostics.converged
    assert export_bayesian_diagnostics(tmp_path / "all.nc", {0: result}) == []
    window = BayesianDiagnosisWindow()
    qtbot.addWidget(window)
    window.update_results(
        result=result,
        model_func=single_exp_func,
        x_data=t,
        y_data=y,
        yerr=np.full_like(y, 0.01),
        identity=DiagnosisIdentity(str(tmp_path / "scan.h5"), None, 3, None),
    )
    destination = tmp_path / "traces.nc"
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (str(destination), "")
    )
    window._export_traces()
    with h5py.File(destination, "r") as f:
        assert f.attrs["bin_status"] == "not_converged"
        assert int(np.ravel(f.attrs["bin_id"])[0]) == 3
        assert "posterior" in f


# --------------------------------------------------------------------------
# R6-15
# --------------------------------------------------------------------------
@pytest.fixture
def plain_env(qtbot, monkeypatch, tmp_path):
    return _make_env(qtbot, monkeypatch, tmp_path, {"a.h5": _write_zero_delay})


def _registered_shortcuts(view):
    """``(key, context, scope, owner)`` of every shortcut the window registers:
    QAction shortcuts, QShortcut objects (including the ShortcutManager's)."""
    rows = []
    for action in view.findChildren(QtGui.QAction):
        scope = (
            action.parent()
            if action.shortcutContext() == (QtCore.Qt.ShortcutContext.WidgetShortcut)
            else None
        )
        for seq in action.shortcuts():
            if not seq.isEmpty():
                rows.append(
                    (seq.toString(), action.shortcutContext(), scope, action.text())
                )
    for shortcut in view.findChildren(QtGui.QShortcut):
        scope = (
            shortcut.parent()
            if shortcut.context() == (QtCore.Qt.ShortcutContext.WidgetShortcut)
            else None
        )
        rows.append((shortcut.key().toString(), shortcut.context(), scope, "QShortcut"))
    return rows


def test_r6_15_no_key_sequence_is_registered_twice_in_one_context(plain_env):
    view = plain_env.view
    rows = _registered_shortcuts(view)
    keys = [r[0] for r in rows]
    assert "Ctrl+Shift+P" in keys and "Ctrl+P" in keys and "Ctrl+O" in keys
    seen: dict = {}
    duplicates = []
    for key, context, scope, owner in rows:
        slot = (key, context, id(scope))
        if slot in seen:
            duplicates.append((key, seen[slot], owner))
        seen[slot] = owner
    assert duplicates == []
    # what the ShortcutManager knows is registered once as well
    manager_keys = list(view.shortcut_manager._key_to_id)
    assert len(manager_keys) == len(set(manager_keys))


def test_r6_15_ctrl_shift_p_opens_the_progress_dialog(plain_env):
    view = plain_env.view
    view.show()
    op = view.progress_manager.start_operation(
        "r6-probe", "Reading", show_after_delay=-1
    )
    view.progress_manager.progress_dialog.hide()
    assert not view.progress_manager.progress_dialog.isVisible()
    messages: list[str] = []
    QtCore.qInstallMessageHandler(
        lambda _k, _c, text: (
            messages.append(text) if "shortcut" in text.lower() else None
        )
    )
    try:
        view.activateWindow()
        QTest.qWait(100)
        QTest.keyClick(
            view,
            QtCore.Qt.Key.Key_P,
            QtCore.Qt.KeyboardModifier.ControlModifier
            | QtCore.Qt.KeyboardModifier.ShiftModifier,
        )
        end = time.monotonic() + 3
        while time.monotonic() < end and not (
            view.progress_manager.progress_dialog.isVisible()
        ):
            QTest.qWait(20)
    finally:
        QtCore.qInstallMessageHandler(None)
    assert not [m for m in messages if "ambiguous" in m.lower()]
    assert view.progress_manager.progress_dialog.isVisible()
    view.progress_manager.complete_operation(op)


def test_r6_15_source_registers_each_shortcut_once():
    """Every literal key sequence given to setShortcut / QShortcut / the manager
    appears once across xpcs_viewer.py (a second registration of the same key
    would be ambiguous for the window)."""
    tree = ast.parse((PKG / "xpcs_viewer.py").read_text())
    keys = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", getattr(node.func, "id", ""))
        if name in {"setShortcut", "QShortcut", "register_shortcut"}:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    if "+" in sub.value or sub.value in {"Esc", "F1"}:
                        keys.append(sub.value)
    assert len(keys) == len(set(keys)), keys


# --------------------------------------------------------------------------
# R6-16
# --------------------------------------------------------------------------
def _perturb(widget):
    """Set a control to a value different from its current one."""
    if hasattr(widget, "isChecked"):
        widget.setChecked(not widget.isChecked())
    else:
        count = widget.count()
        widget.setCurrentIndex(count - 1 if widget.currentIndex() != count - 1 else 0)


def test_r6_16_collect_and_restore_cover_the_same_fields():
    """Every session field is either in the one control table or is declared as
    having no widget in the window: nothing can be saved and silently not
    restored."""
    from dataclasses import fields

    from xpcsviewer.gui.state.session_manager import AnalysisParameters
    from xpcsviewer.xpcs_viewer import _SESSION_CONTROLS

    no_widget = {
        "saxs2d_log_scale",
        "saxs1d_log_x",
        "saxs1d_log_y",
        "intensity_t_normalize",
        "g2_q_index",
        "diffusion_model",
        "twotime_symmetric",
        "qmap_show_rings",
    }
    # R7-07: the identity of the selected two-time bin (file path + absolute bin
    # ID) is collected beside the table row of its combo position and applied by
    # that row's setter (see test_review_r7_state)
    identity = {"twotime_source_file", "twotime_selected_bin"}
    mapped = {row[0] for row in _SESSION_CONTROLS}
    assert mapped | no_widget | identity == {f.name for f in fields(AnalysisParameters)}
    assert not mapped & no_widget
    assert not identity & (mapped | no_widget)
    src = (PKG / "xpcs_viewer.py").read_text()
    assert (
        "_SESSION_CONTROLS"
        in src.split("def _collect_session_state")[1].split("def _restore_session")[0]
    )
    assert (
        "_SESSION_CONTROLS"
        in src.split("def _restore_session")[1].split("def _open_docs")[0]
    )


@pytest.fixture
def twotime_session_env(qtbot, monkeypatch, tmp_path):
    return _make_env(
        qtbot,
        monkeypatch,
        tmp_path,
        {"tt.h5": lambda p: _write_twotime(p, [0.01, 0.02], value=2)},
        real_session=True,
    )


def test_r6_16_every_saved_analysis_widget_round_trips_through_the_session(
    qtbot, twotime_session_env, monkeypatch
):
    from xpcsviewer.xpcs_viewer import _SESSION_CONTROLS

    env = twotime_session_env
    first = env.view
    _select(first, 0)
    first.tabWidget.setCurrentIndex(_tab(env.xv, "twotime"))
    first.twotime_correct_diag.setChecked(False)
    first.plot_twotime()  # populates the Q-bin combo
    assert first.comboBox_twotime_selection.count() == 2
    defaults = {}
    for fld, attr, getter, _setter, _sync in _SESSION_CONTROLS:
        widget = getattr(first, attr)
        defaults[fld] = getter(widget)
        _perturb(widget)
    changed = {
        f: getter(getattr(first, a)) for f, a, getter, _s, _y in _SESSION_CONTROLS
    }
    assert all(changed[f] != defaults[f] for f in defaults)

    state = first._collect_session_state()
    for fld, value in changed.items():
        assert getattr(state.analysis_params, fld) == value
    assert first.session_manager.save_session(state)
    first.close()

    second = env.xv.XpcsViewer(path=str(env.data))
    qtbot.addWidget(second)
    assert list(second.vk.target) == ["tt.h5"]
    for fld, attr, getter, _setter, _sync in _SESSION_CONTROLS:
        if fld == "twotime_selected_q":
            continue  # the combo is empty until a file is plotted (see below)
        assert getter(getattr(second, attr)) == changed[fld], fld
    # the remembered Q bin of the two-time file is applied when it is drawn
    assert (
        second._twotime_qbin_memory[str(env.data / "tt.h5")]
        == changed["twotime_selected_q"]
    )
    second.tabWidget.setCurrentIndex(_tab(env.xv, "twotime"))
    second.twotime_correct_diag.setChecked(False)
    second.plot_twotime()
    assert (
        second.comboBox_twotime_selection.currentIndex()
        == (changed["twotime_selected_q"])
    )
    second.close()


def test_r6_16_older_session_files_missing_keys_leave_those_widgets_alone(
    qtbot, twotime_session_env
):
    env = twotime_session_env
    first = env.view
    default_qmap = first.cb_qmap_cmap.currentText()  # the window's own default
    first.cb_qmap_cmap.setCurrentIndex(first.cb_qmap_cmap.count() - 1)
    first.cb_twotime_cmap.setCurrentIndex(first.cb_twotime_cmap.count() - 1)
    chosen_twotime = first.cb_twotime_cmap.currentText()
    assert first.cb_qmap_cmap.currentText() != default_qmap
    assert first.session_manager.save_session(first._collect_session_state())
    first.close()
    path = env.root / "session.json"
    data = json.loads(path.read_text())
    del data["analysis_params"]["qmap_colormap"]  # an older file never had it
    path.write_text(json.dumps(data))

    second = env.xv.XpcsViewer(path=str(env.data))
    qtbot.addWidget(second)
    assert second.cb_twotime_cmap.currentText() == chosen_twotime  # present: applied
    # missing: the window's own default, not a dataclass default applied blindly
    assert second.cb_qmap_cmap.currentText() == default_qmap


def test_r6_16_restore_blocks_widget_signals_and_redraws_once(
    qtbot, twotime_session_env, monkeypatch
):
    env = twotime_session_env
    first = env.view
    first.cb_saxs2D_cmap.setCurrentIndex(first.cb_saxs2D_cmap.count() - 1)
    first.cb_stab_type.setCurrentIndex(2)
    first.g2_fitting_function.setCurrentIndex(1)
    assert first.session_manager.save_session(first._collect_session_state())
    first.close()

    second = env.xv.XpcsViewer(path=str(env.data))
    qtbot.addWidget(second)
    fired, redraws = [], []
    for combo in (
        second.cb_saxs2D_cmap,
        second.cb_stab_type,
        second.g2_fitting_function,
    ):
        combo.setCurrentIndex(0)
        combo.currentIndexChanged.connect(fired.append)
    real = env.xv.XpcsViewer.update_plot
    monkeypatch.setattr(
        env.xv.XpcsViewer, "update_plot", lambda self: redraws.append(1) or real(self)
    )
    # only the analysis widgets: no workspace reload, so every redraw counted is
    # the restore's own
    session = second.session_manager.load_session()
    session.data_path, session.target_files = None, []
    monkeypatch.setattr(second.session_manager, "load_session", lambda: session)
    fired.clear()
    second._restore_session()
    assert second.cb_stab_type.currentIndex() == 2
    assert second.g2_fitting_function.currentIndex() == 1
    assert fired == []  # signals were blocked while the table set the widgets
    assert len(redraws) == 1
    # the handler that the blocked signal would have run was run explicitly
    assert "Double Exp" in second.groupBox_2.title()
    second.close()
