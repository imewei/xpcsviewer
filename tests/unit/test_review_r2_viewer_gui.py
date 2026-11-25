"""Regression tests for review round 2: R2-06 (GUI), R2-07, R2-08, R2-20,
R2-22, R2-23, R2-28 in xpcs_viewer.py and the batch Bayesian coordinator."""

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _viewer_cls():
    from xpcsviewer.xpcs_viewer import XpcsViewer

    return XpcsViewer


# ---------------------------------------------------------------------------
# R2-06 (GUI part): async G2 result must forward the effective Y/time ranges
# ---------------------------------------------------------------------------


def test_r2_06_async_g2_result_forwards_y_and_time_ranges():
    plot_params = {
        "q_range": (0.0, 1.0),
        "t_range": (1e-3, 1.0),
        "y_range": (0.9, 1.5),
        "y_auto": False,
        "t_auto": False,
        "q_auto": False,
        "rows": [0],
        "target_timestamp": "t",
        "num_col": 2,
    }
    result = {
        "plot_params": plot_params,
        "q": [np.array([0.1])],
        "tel": [np.array([1.0])],
        "g2": [np.ones((1, 1))],
        "g2_err": [np.ones((1, 1))],
        "labels": [["q"]],
        "num_figs": 1,
        "fit_results": None,
    }
    stub = SimpleNamespace(mp_g2=object(), _initialize_g2_fit_bounds=MagicMock())
    with patch("xpcsviewer.module.g2mod.pg_plot_from_data") as render:
        _viewer_cls().apply_g2_result(stub, result)
    kwargs = render.call_args.kwargs
    assert kwargs["y_range"] == (0.9, 1.5)
    assert kwargs["t_range"] == (1e-3, 1.0)
    assert kwargs["y_auto"] is False
    assert "q_range" not in kwargs
    assert "target_timestamp" not in kwargs


# ---------------------------------------------------------------------------
# R2-07: remove-from-target without a workspace / without a selection
# ---------------------------------------------------------------------------


def test_r2_07_remove_target_without_kernel_is_noop():
    stub = SimpleNamespace(
        vk=None, list_view_target=MagicMock(selectedIndexes=lambda: [])
    )
    _viewer_cls().remove_target(stub)  # must not raise


def test_r2_07_remove_target_without_selection_changes_nothing():
    vk = MagicMock()
    vk.target = ["a.h5", "b.h5"]
    stub = SimpleNamespace(
        vk=vk,
        list_view_target=MagicMock(selectedIndexes=lambda: []),
        clear_target_selection=MagicMock(),
        update_box=MagicMock(),
    )
    _viewer_cls().remove_target(stub)
    vk.remove_target.assert_not_called()
    stub.update_box.assert_not_called()


# ---------------------------------------------------------------------------
# R2-08: an explicit startup path must win over the saved session workspace
# ---------------------------------------------------------------------------


def _restore_stub(explicit, work_dir_text, session_dir):
    session = SimpleNamespace(
        data_path=str(session_dir),
        target_files=[],
        active_tab=-1,
        window_geometry=SimpleNamespace(maximized=True),
        analysis_params=SimpleNamespace(),
    )
    return SimpleNamespace(
        session_manager=SimpleNamespace(
            load_session=lambda: session, get_warnings=lambda: []
        ),
        toast_manager=MagicMock(),
        _path_explicit=explicit,
        work_dir=SimpleNamespace(text=lambda: work_dir_text),
        load_path=MagicMock(),
        vk=None,
        tabWidget=MagicMock(count=lambda: 0),
    )


def test_r2_08_explicit_path_is_not_replaced_by_session_workspace(tmp_path):
    a, b = tmp_path / "A", tmp_path / "B"
    a.mkdir()
    b.mkdir()
    stub = _restore_stub(True, str(b), a)
    _viewer_cls()._restore_session(stub)
    stub.load_path.assert_not_called()


def test_r2_08_session_workspace_restored_without_explicit_path(tmp_path):
    a = tmp_path / "A"
    a.mkdir()
    stub = _restore_stub(False, str(tmp_path), a)
    _viewer_cls()._restore_session(stub)
    stub.load_path.assert_called_once_with(str(a))


def test_r2_08_explicit_path_equal_to_session_workspace_still_restores(tmp_path):
    a = tmp_path / "A"
    a.mkdir()
    stub = _restore_stub(True, str(a), a)
    _viewer_cls()._restore_session(stub)
    stub.load_path.assert_called_once_with(str(a))


# ---------------------------------------------------------------------------
# R2-20: a detector click that resolves to another bin updates the controls
# ---------------------------------------------------------------------------


def _twotime_stub(qapp, resolved):
    from qtpy.QtWidgets import QComboBox, QSlider

    combo = QComboBox()
    combo.addItems(["bin0", "bin1", "bin2"])
    slider = QSlider()
    slider.setMaximum(2)
    xf = SimpleNamespace(
        fname="A.h5",
        get_twotime_maps=MagicMock(return_value=(None, None, resolved)),
    )
    stub = SimpleNamespace(
        comboBox_twotime_selection=combo,
        horizontalSlider_twotime_selection=slider,
        twotime_autocrop=MagicMock(),
        cb_twotime_cmap=MagicMock(),
        c2_min=MagicMock(),
        c2_max=MagicMock(),
        twotime_correct_diag=MagicMock(),
        checkBox_twotime_autolevel=MagicMock(),
        mp_2t_hdls={"tt": object()},
        _twotime_qbin_memory=OrderedDict(),
        _twotime_qbin_bins=OrderedDict(),
        _twotime_last_fname="A.h5",
        _current_twotime_fname=lambda: "A.h5",
        get_selected_rows=lambda: [0],
        _guard_no_data=lambda _t: True,
        vk=SimpleNamespace(
            plot_twotime=lambda hdl, **kw: ["bin0", "bin1", "bin2"],
            get_xf_list=lambda rows=None, filter_atype=None: [xf],
        ),
    )
    return stub, xf


def test_r2_20_click_on_other_bin_updates_controls_and_memory(qapp):
    stub, xf = _twotime_stub(qapp, resolved=np.int64(2))
    combo, slider = (
        stub.comboBox_twotime_selection,
        stub.horizontalSlider_twotime_selection,
    )
    combo.setCurrentIndex(0)
    changed = []
    combo.currentIndexChanged.connect(changed.append)
    _viewer_cls().plot_twotime(stub, highlight_xy=(3, 4))
    assert combo.currentIndex() == 2
    assert slider.value() == 2
    assert stub._twotime_qbin_memory["A.h5"] == 2
    assert changed == []  # signals blocked: no second plot is triggered
    assert xf.get_twotime_maps.call_args.kwargs["highlight_xy"] == (3, 4)


def test_r2_20_click_outside_processed_bins_keeps_selection(qapp):
    stub, _ = _twotime_stub(qapp, resolved=None)
    combo = stub.comboBox_twotime_selection
    combo.setCurrentIndex(1)
    _viewer_cls().plot_twotime(stub, highlight_xy=(3, 4))
    assert combo.currentIndex() == 1
    assert stub.horizontalSlider_twotime_selection.value() == 1


# ---------------------------------------------------------------------------
# R2-22: the cached fit must also match the observations it was fitted to
# ---------------------------------------------------------------------------


def _fit_bundle(x, y=None, bin_id=1, q=0.02, result="posterior"):
    from xpcsviewer.fitting.bayesian_assembly import BayesianBundle

    y = np.ones_like(x) if y is None else y
    return BayesianBundle.create(
        "A.h5", bin_id, f"q={q}", q, "single", x, y, np.full_like(x, 0.1)
    ).with_result(result)


def test_r2_22_diagnosis_shows_stored_delay_window_not_current_one():
    """The delay window may change after the fit: the stored observations (not
    the current window) are what the stored posterior is shown with."""
    fit_x = np.array([0.001, 0.01])
    stored = _fit_bundle(fit_x)
    stub = SimpleNamespace(
        sb_g2_bayesian_qidx=MagicMock(value=lambda: 0),
        statusbar=MagicMock(),
        _g2_current_bundle=lambda: stored,
        _update_g2_diagnosis=MagicMock(),
    )
    _viewer_cls()._show_g2_diagnosis(stub)
    shown = stub._update_g2_diagnosis.call_args[0][0]
    np.testing.assert_array_equal(shown.x, fit_x)
    assert shown.result == "posterior"


def test_r2_22_single_fit_records_fit_time_delays():
    x = np.array([0.001, 0.01])
    fit_result = SimpleNamespace(diagnostics=SimpleNamespace(converged=True))
    launch = _fit_bundle(x, bin_id=3, q=0.07, result=None)
    stub = SimpleNamespace(
        _g2_bayesian_worker_active=True,
        _g2_bayesian_worker=None,
        _g2_bayesian_launch=launch,
        btn_g2_bayesian=MagicMock(),
        btn_g2_diagnosis=MagicMock(),
        sb_g2_bayesian_qidx=MagicMock(),
        statusbar=MagicMock(),
        _g2_bayesian_store={},
        _g2_current_bundle=lambda: None,
        _g2_bayesian_launch_revision=None,  # "A.h5" does not exist: no revision
        _evict_stale_bayesian=lambda: 0,
        _announce_bayesian_fit=MagicMock(),
    )
    stub.sb_g2_bayesian_qidx.value.return_value = 3
    _viewer_cls()._on_g2_bayesian_finished(
        stub,
        {"fit_result": fit_result, "q_index": 3, "q_value": 0.07, "launch": launch},
    )
    done = stub._g2_bayesian_store[("A.h5", None, 3)]
    assert done.q_value == 0.07
    np.testing.assert_array_equal(done.x, x)


def test_r2_22_batch_fit_records_fit_time_delays():
    x = np.array([0.001, 0.01])
    launch = _fit_bundle(x, bin_id=4, q=0.02, result=None)
    stub = SimpleNamespace(
        _g2_bayesian_store={}, _g2_batch_launch={"specs": [], "revision": None}
    )
    _viewer_cls()._on_g2_batch_single_q_done(
        stub, 1, {"fit_result": "fit", "launch": launch}
    )
    done = stub._g2_bayesian_store[("A.h5", None, 4)]
    assert done.q_value == 0.02
    np.testing.assert_array_equal(done.x, x)


# ---------------------------------------------------------------------------
# R2-23: a direct SAXS spinbox draw retires the in-flight async SAXS plot
# ---------------------------------------------------------------------------


def test_r2_23_spinbox_draw_drops_older_async_saxs_result():
    cls = _viewer_cls()
    request = {"rows": [0, 1], "target_timestamp": "t1"}
    stub = SimpleNamespace(
        active_plot_operations={"plot_saxs_2d_x": "saxs_2d"},
        _plot_op_requests={"plot_saxs_2d_x": request},
        vk=SimpleNamespace(timestamp="t1"),
        # the selected rows are unchanged, so the row-based staleness check
        # alone would still accept the old result
        plot_saxs_2d=MagicMock(return_value={"rows": [0, 1]}),
        apply_saxs_2d_result=MagicMock(),
        update_plot_async=MagicMock(),
        progress_manager=MagicMock(),
        toast_manager=MagicMock(),
        spinBox_saxs2d_selection=MagicMock(value=lambda: 1),
        async_vk=None,
        _plot_pending=set(),
        _retiring_plots=False,
        plot_kwargs_record={},
    )
    stub.progress_manager.is_operation_active.return_value = False
    stub._plot_request_is_stale = lambda *a: cls._plot_request_is_stale(stub, *a)
    stub._settle_plot_operation = lambda *a, **k: cls._settle_plot_operation(
        stub, *a, **k
    )
    stub.cancel_async_operation = lambda op: cls.cancel_async_operation(stub, op)
    stub._retire_plot_operations = lambda *a: cls._retire_plot_operations(stub, *a)

    cls.plot_saxs_2d_selection(stub)  # draws B synchronously
    stub.plot_saxs_2d.assert_called_once_with(selection=1)

    cls.on_async_plot_ready(stub, "plot_saxs_2d_x", "old-image-A")
    stub.apply_saxs_2d_result.assert_not_called()


# ---------------------------------------------------------------------------
# R2-28: a cancelled batch must not emit into cleared GUI state
# ---------------------------------------------------------------------------


def _coordinator(n=2):
    from xpcsviewer.threading.batch_bayesian_coordinator import (
        BatchBayesianCoordinator,
    )

    pool = MagicMock()
    specs = [
        {
            "q_idx": i,
            "x": np.linspace(0.001, 100, 20),
            "y": np.ones(20),
            "yerr": np.full(20, 0.01),
            "q_value": 0.001 * (i + 1),
            "fit_func": MagicMock(),
        }
        for i in range(n)
    ]
    return BatchBayesianCoordinator(specs, pool, max_concurrent=n)


def test_r2_28_cancelled_coordinator_emits_nothing_more(qapp):
    coord = _coordinator()
    got = []
    coord.all_finished.connect(lambda r: got.append("all"))
    coord.single_q_finished.connect(lambda q, r: got.append("single"))
    coord.start()
    coord.cancel()
    coord._on_worker_finished(0, {"fit_result": MagicMock()})
    coord._on_worker_cancelled(1)
    assert got == []


def test_r2_28_batch_finished_after_state_reset_is_ignored():
    stub = SimpleNamespace(
        _g2_batch_launch=None,  # cleared by the workspace reset
        progress_manager=MagicMock(),
    )
    _viewer_cls()._on_g2_batch_finished(stub, {0: None, 1: None})  # must not raise
    stub.progress_manager.complete_operation.assert_not_called()
