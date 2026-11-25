"""Regression tests for review round 1 findings R1-11, R1-12, R1-14, R1-15, R1-17."""

from __future__ import annotations

import pytest
from qtpy.QtCore import QObject

from xpcsviewer.threading.batch_bayesian_coordinator import BatchBayesianCoordinator


@pytest.mark.unit
def test_r1_11_all_finished_delivers_int_keyed_mapping(qapp):
    """all_finished must hand a connected slot the original int-keyed dict."""
    coord = BatchBayesianCoordinator([], thread_pool=None)
    received: list = []

    class Sink(QObject):
        def slot(self, results):
            received.append(results)

    sink = Sink()
    coord.all_finished.connect(sink.slot)
    payload = {2: {"fit_result": "valid"}, 5: None}
    coord.all_finished.emit(payload)
    assert received == [payload]


# ---------------------------------------------------------------------------
# R1-12: diagnosis must refuse a cached fit from another file / Q value
# ---------------------------------------------------------------------------


FIT_X = [1.0]  # delay array every stubbed fit/extraction in this section shares


def _bundle(fname="A.h5", bin_id=0, q=0.01, result=None):
    from xpcsviewer.fitting.bayesian_assembly import BayesianBundle

    return BayesianBundle.create(
        fname, bin_id, f"q={q}", q, "single", FIT_X, [1.0], [0.1]
    ).with_result(result)


def _diagnosis_stub(fit_fname, fit_bin, cur_fname, cur_bin):
    """Minimal stand-in for XpcsViewer exercising _show_g2_diagnosis.

    The diagnosis looks the stored bundle up by the physical bin the spinbox
    currently addresses: (current file, current absolute bin ID).
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from xpcsviewer.xpcs_viewer import XpcsViewer

    stub = SimpleNamespace()
    stub.sb_g2_bayesian_qidx = MagicMock()
    stub.sb_g2_bayesian_qidx.value.return_value = 0
    stored = _bundle(fit_fname, fit_bin, result=object())
    stub._g2_bayesian_store = {(stored.fname, None, stored.bin_id): stored}
    stub.statusbar = MagicMock()
    stub._g2_bin_table = lambda: (cur_fname, [(cur_bin, 0.01, "q")])
    stub._g2_current_bundle = lambda: XpcsViewer._g2_current_bundle(stub)
    stub._update_g2_diagnosis = MagicMock()
    return stub


@pytest.mark.unit
@pytest.mark.parametrize(
    ("cur_fname", "cur_bin"),
    [("B.h5", 0), ("A.h5", 1)],
    ids=["other-file", "other-bin"],
)
def test_r1_12_diagnosis_refuses_mismatched_fit(cur_fname, cur_bin):
    from xpcsviewer.xpcs_viewer import XpcsViewer

    stub = _diagnosis_stub("A.h5", 0, cur_fname, cur_bin)
    XpcsViewer._show_g2_diagnosis(stub)
    stub._update_g2_diagnosis.assert_not_called()
    assert "run Fit first" in stub.statusbar.showMessage.call_args[0][0]


@pytest.mark.unit
def test_r1_12_diagnosis_accepts_matching_fit():
    from xpcsviewer.xpcs_viewer import XpcsViewer

    stub = _diagnosis_stub("A.h5", 0, "A.h5", 0)
    XpcsViewer._show_g2_diagnosis(stub)
    stub._update_g2_diagnosis.assert_called_once()


@pytest.mark.unit
def test_r1_12_single_fit_records_file_and_bin():
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from xpcsviewer.xpcs_viewer import XpcsViewer

    fit_result = SimpleNamespace(diagnostics=SimpleNamespace(converged=True))
    launch = _bundle("A.h5", 5, 0.07)
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
    XpcsViewer._on_g2_bayesian_finished(
        stub,
        {"fit_result": fit_result, "q_index": 3, "q_value": 0.07, "launch": launch},
    )
    # keyed by (file, absolute bin), not by the relative index 3
    assert list(stub._g2_bayesian_store) == [("A.h5", None, 5)]
    assert stub._g2_bayesian_store[("A.h5", None, 5)].result is fit_result


@pytest.mark.unit
def test_r1_12_batch_fit_records_file_and_bin():
    from types import SimpleNamespace

    from xpcsviewer.xpcs_viewer import XpcsViewer

    launch = _bundle("A.h5", 7, 0.03)
    stub = SimpleNamespace(
        _g2_bayesian_store={},
        _g2_batch_launch={"specs": [], "fname": "A.h5", "revision": None},
    )
    XpcsViewer._on_g2_batch_single_q_done(
        stub, 2, {"fit_result": "fit", "launch": launch}
    )
    assert list(stub._g2_bayesian_store) == [("A.h5", None, 7)]
    assert stub._g2_bayesian_store[("A.h5", None, 7)].result == "fit"


# ---------------------------------------------------------------------------
# R1-14: re-opening a visible Mask Editor must not reload (and wipe) it
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_r1_14_open_simplemask_focuses_visible_window_without_reload():
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from xpcsviewer.xpcs_viewer import XpcsViewer

    win = MagicMock()
    win.isVisible.return_value = True
    stub = SimpleNamespace(
        _simplemask_window=win,
        _get_simplemask_data=MagicMock(return_value=(None, None)),
    )
    XpcsViewer.open_simplemask(stub)
    win.load_from_viewer.assert_not_called()
    stub._get_simplemask_data.assert_not_called()
    win.raise_.assert_called_once()
    win.activateWindow.assert_called_once()


@pytest.mark.unit
def test_r1_14_open_simplemask_loads_hidden_window():
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from xpcsviewer.xpcs_viewer import XpcsViewer

    win = MagicMock()
    win.isVisible.return_value = False
    stub = SimpleNamespace(
        _simplemask_window=win,
        _get_simplemask_data=MagicMock(return_value=("img", {"k": 1})),
    )
    XpcsViewer.open_simplemask(stub)
    win.load_from_viewer.assert_called_once_with("img", {"k": 1})
    win.show.assert_called_once()


# ---------------------------------------------------------------------------
# R1-15: a stale async plot result must not repaint the current workspace
# ---------------------------------------------------------------------------


def _plot_stub(request, current):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from xpcsviewer.xpcs_viewer import XpcsViewer

    stub = SimpleNamespace(
        active_plot_operations={"plot_saxs_2d_x": "saxs_2d"},
        _plot_op_requests={"plot_saxs_2d_x": request},
        vk=SimpleNamespace(timestamp="t1"),
        plot_saxs_2d=lambda dryrun=False: current,
        apply_saxs_2d_result=MagicMock(),
        update_plot_async=MagicMock(),
        progress_manager=MagicMock(),
        toast_manager=MagicMock(),
        async_vk=None,
        _plot_pending=set(),
        _retiring_plots=False,
        plot_kwargs_record={},
        _plot_request=lambda kwargs: kwargs,  # no revision part in these stubs
        _update_placeholder_qmap_notice=lambda: None,
    )
    stub._plot_request_is_stale = lambda *a: XpcsViewer._plot_request_is_stale(stub, *a)
    stub._settle_plot_operation = lambda *a, **k: XpcsViewer._settle_plot_operation(
        stub, *a, **k
    )
    return stub


@pytest.mark.unit
def test_r1_15_stale_plot_result_is_ignored_and_refresh_queued():
    from xpcsviewer.xpcs_viewer import XpcsViewer

    stub = _plot_stub(
        request={"rows": [0], "target_timestamp": "t1"},
        current={"rows": [1]},  # selection changed while the worker ran
    )
    XpcsViewer.on_async_plot_ready(stub, "plot_saxs_2d_x", "old-image")
    stub.apply_saxs_2d_result.assert_not_called()
    stub.update_plot_async.assert_called_once_with("saxs_2d")
    assert stub.active_plot_operations == {}


@pytest.mark.unit
def test_r1_15_current_plot_result_is_applied():
    from xpcsviewer.xpcs_viewer import XpcsViewer

    stub = _plot_stub(
        request={"rows": [0], "target_timestamp": "t1"},
        current={"rows": [0]},
    )
    XpcsViewer.on_async_plot_ready(stub, "plot_saxs_2d_x", "image")
    stub.apply_saxs_2d_result.assert_called_once_with("image")
    stub.update_plot_async.assert_not_called()


@pytest.mark.unit
def test_r1_15_workspace_reset_retires_all_plot_operations():
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from xpcsviewer.xpcs_viewer import XpcsViewer

    stub = SimpleNamespace(
        active_plot_operations={"op1": "saxs_2d", "op2": "g2"},
        _plot_op_requests={"op1": {}, "op2": {}},
        progress_manager=MagicMock(),
        async_vk=None,
        _plot_pending=set(),
        _retiring_plots=False,
        plot_kwargs_record={},
        update_plot_async=MagicMock(),
    )
    stub.progress_manager.is_operation_active.return_value = False
    stub.cancel_async_operation = lambda op: XpcsViewer.cancel_async_operation(stub, op)
    stub._settle_plot_operation = lambda *a, **k: XpcsViewer._settle_plot_operation(
        stub, *a, **k
    )
    XpcsViewer._retire_plot_operations(stub, "g2")
    assert stub.active_plot_operations == {"op1": "saxs_2d"}
    XpcsViewer._retire_plot_operations(stub)
    assert stub.active_plot_operations == {}
    stub.update_plot_async.assert_not_called()  # retirement never replays a plot


# ---------------------------------------------------------------------------
# R1-17: viewer QMap stores metres; SimpleMask kernel expects millimetres
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_r1_17_simplemask_metadata_is_in_millimetres():
    from types import SimpleNamespace

    import numpy as np

    from xpcsviewer.xpcs_viewer import XpcsViewer

    xf = SimpleNamespace(
        saxs_2d=np.ones((2, 2)),
        bcx=0.5,
        bcy=0.5,
        det_dist=5.0,  # QMap: metres
        pixel_size=75e-6,  # QMap: metres
        X_energy=10.0,
        fname="a.h5",  # sent to the editor as its protected source file
    )
    stub = SimpleNamespace(
        vk=SimpleNamespace(target=[0], get_xf_list=lambda rows=None: [xf]),
        get_selected_rows=lambda: [0],  # the editor opens the highlighted target
    )
    _, meta = XpcsViewer._get_simplemask_data(stub)
    assert meta["det_dist"] == pytest.approx(5000.0)
    assert meta["pix_dim"] == pytest.approx(0.075)
