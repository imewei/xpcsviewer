"""Regression tests for the audit of review round 8's own fixes.

A1  replacing an averaging job kills the old worker only for the same destination
A2  the default baseline Q index is lowered, visibly, to the selected files' bins
A3  the Bayesian G2 bin table uses the Q range the G2 plot uses (None when auto)
A4  the two-time CLI tolerance uses the map's full spacing, not the processed one
A5  Bayesian diffusion reads the same G2 summary as the diffusion plot
A6  a Cartesian partner with a non-finite value is refused with a reason
A7  SaxsNormalizationRefused is handled by the Stability tab
A8  the SAXS-1D display axis is the first selected file's, in range or not
A9  SAXS-2D clears when the target list empties; a changed file is explained
A10 an auto-registered fixed operation id is not closed forever

Real h5py-written files, the real ``XpcsFile``, kernel, workers, an offscreen
``XpcsViewer``; Events (never sleeps) order the worker threads.
"""

from __future__ import annotations

import threading
from pathlib import Path

import h5py
import numpy as np
import pytest
from qtpy import QtWidgets

from tests.unit import test_review_r3_average as r3avg
from tests.unit.test_review_r6_viewer_gui import _make_env
from tests.unit.test_review_r8_selection import (  # noqa: F401
    _fresh_pool,
    _read,
    _write_bins,
    job_env,
)

pytestmark = pytest.mark.unit

K = r3avg.K
PKG = Path(__file__).resolve().parents[2] / "xpcsviewer"


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# --------------------------------------------------------------------- A1
@pytest.mark.parametrize("spelling", ["other-name", "same-file-other-spelling"])
def test_a1_replacing_a_job_kills_only_a_job_for_the_same_file(
    job_env,  # noqa: F811
    qtbot,
    spelling,
):
    env, view = job_env, job_env.view
    old = env.start("oldA.h5", "oldB.h5")
    assert env.entered.wait(2)
    if spelling == "other-name":
        view.avg_save_name.setText("second.h5")
    else:  # the same file reached through another spelling of the folder
        view.avg_save_path.setText(str(env.out) + "/../out")
    new = env.start("newA.h5", "newB.h5")
    qtbot.waitUntil(lambda: new.status == "finished", timeout=2000)
    env.release.set()
    assert view.thread_pool.waitForDone(5000)
    QtWidgets.QApplication.processEvents()
    kernel = view.vk
    assert kernel.avg_worker is new
    assert old.jid in kernel._retired_avg_jobs
    assert old.jid not in kernel.avg_worker_active
    # a late message of the detached job cannot recreate its record
    kernel.update_avg_values((old.jid, 1.0))
    assert old.jid not in kernel.avg_worker_active
    if spelling == "other-name":
        assert old.status == "finished"
        np.testing.assert_allclose(_read(env.out / "average.h5"), 0.99)
        np.testing.assert_allclose(_read(env.out / "second.h5"), 1.03)
    else:
        assert old.status == "killed"
        np.testing.assert_allclose(_read(env.dest), 1.03)


# --------------------------------------------------------------------- A2
def test_a2_default_baseline_index_is_lowered_for_a_small_dataset(
    qtbot,
    monkeypatch,
    tmp_path,
    _fresh_pool,  # noqa: F811
):
    names = {
        n: (lambda p, v=v: _write_bins(p, 3, g2=v))
        for n, v in {"a.h5": 0.99, "b.h5": 1.01}.items()
    }
    env = _make_env(qtbot, monkeypatch, tmp_path, names)
    view = env.view
    out = tmp_path / "out"
    view.avg_save_path.setText(str(out))
    view.avg_save_name.setText("average.h5")
    view.bx_avg_G2IPIF.setChecked(False)
    view.bx_avg_saxs.setChecked(False)
    view.bx_avg_g2g2err.setChecked(True)
    view.avg_blmin.setValue(0.9)
    view.avg_blmax.setValue(1.1)
    assert view.avg_qindex.value() == 5  # the shipped default, never touched
    view.submit_job()
    assert view.avg_qindex.value() == 2
    shown = view.statusbar.currentMessage()
    assert "lowered from 5 to 2" in shown and "3 Q bins" in shown
    view.start_avg_job()
    worker = view.vk.avg_worker
    qtbot.waitUntil(lambda: worker.status == "finished", timeout=5000)
    view.timer.stop()
    view.thread_pool.waitForDone(5000)
    assert worker.n_accepted == 2 and not worker.rejected
    np.testing.assert_allclose(_read(out / "average.h5"), 1.0)


def test_a2_a_mixed_set_is_lowered_to_the_smallest_and_names_what_it_leaves_out(
    qtbot,
    monkeypatch,
    tmp_path,
    _fresh_pool,  # noqa: F811
):
    spec = {"a.h5": 2, "b.h5": 2, "c.h5": 6, "d.h5": 6}
    env = _make_env(
        qtbot,
        monkeypatch,
        tmp_path,
        {n: (lambda p, b=b: _write_bins(p, b)) for n, b in spec.items()},
    )
    view = env.view
    out = tmp_path / "out"
    view.avg_save_path.setText(str(out))
    view.avg_save_name.setText("average.h5")
    view.bx_avg_G2IPIF.setChecked(False)
    view.bx_avg_saxs.setChecked(False)
    view.bx_avg_g2g2err.setChecked(True)
    view.avg_blmin.setValue(0.9)
    view.avg_blmax.setValue(1.1)
    view.submit_job()
    assert view.avg_qindex.value() == 1  # the smallest file has 2 bins
    worker = view.vk.avg_worker
    view.start_avg_job()
    qtbot.waitUntil(lambda: worker.status == "finished", timeout=5000)
    view.thread_pool.waitForDone(5000)
    view.timer.stop()
    view._poll_avg_job()
    # the backstop: the 6-bin files cannot join the 2-bin reference, and are named
    assert set(worker.rejected) == {"c.h5", "d.h5"}
    assert "c.h5" in view.statusbar.currentMessage()
    assert _read(out / "average.h5").shape == (3, 2)


def test_a2_an_index_inside_every_file_is_left_alone(
    qtbot,
    monkeypatch,
    tmp_path,
    _fresh_pool,  # noqa: F811
):
    names = {n: (lambda p: _write_bins(p, 3)) for n in ("a.h5", "b.h5")}
    env = _make_env(qtbot, monkeypatch, tmp_path, names)
    view = env.view
    view.avg_qindex.setValue(1)
    assert view._baseline_qindex_for_job() == 1
    assert view.avg_qindex.value() == 1
    assert "lowered" not in view.statusbar.currentMessage()


# --------------------------------------------------------------------- A3
def _cartesian_g2(path):
    """4 Cartesian bins: qx in {-0.03, 0.01} (negative!), qy in {-0.02, 0.02}."""
    from tests.unit import test_review_r5_kernel as r5
    from tests.unit.test_review_r7_units import _axes

    g2 = np.array([[1.9] * 4, [1.4] * 4, [1.0] * 4]) + np.arange(4) * 0.01
    r5._write_g2_file(path, [-0.03, 0.01], [-0.02, 0.02], [0, 1, 2, 3], [2, 2], g2)
    _axes(path, [b"qx", b"qy"], [b"1/A", b"1/A"])
    return path


def test_a3_the_bayesian_bin_table_is_the_plotted_bins_when_q_is_auto(
    qtbot,
    monkeypatch,
    tmp_path,
    _fresh_pool,  # noqa: F811
):
    from tests.unit.test_review_r6_viewer_gui import _select
    from xpcsviewer.module import g2mod

    env = _make_env(qtbot, monkeypatch, tmp_path, {"cart.h5": _cartesian_g2})
    view = env.view
    _select(view, 0)
    view.g2_qauto.setChecked(True)
    view.g2_qmin.setValue(0.0)  # the boxes hold a range that excludes qx < 0
    view.g2_qmax.setValue(0.5)
    plotted = {}
    real_plot = view.vk.plot_g2

    def spy(*a, **k):
        plotted["q"], tel = real_plot(*a, **k)
        return plotted["q"], tel

    monkeypatch.setattr(view.vk, "plot_g2", spy)
    view.plot_g2()
    shown = np.asarray(plotted["q"][0])
    assert shown.size == 4 and shown.min() < 0
    fname, bins = view._g2_bin_table()
    assert [b[0] for b in bins] == [0, 1, 2, 3]
    np.testing.assert_allclose([b[1] for b in bins], shown)
    # the Bayesian extraction addresses the same bins as the panels
    view.sb_g2_bayesian_qidx.setValue(2)
    bundle, q_idx = view._extract_g2_for_bayesian()
    assert q_idx == 2 and bundle.bin_id == bins[2][0]
    xf = view.vk.get_xf_list([0])[0]
    q, *_ = g2mod.get_data([xf], q_range=None)
    np.testing.assert_allclose(bundle.q_value, q[0][2])
    # with auto OFF the boxes are the selection, in plot and table alike
    view.g2_qauto.setChecked(False)
    view.g2_qmin.setValue(0.0)  # (the auto plot above rewrote the boxes)
    view.g2_qmax.setValue(0.5)
    assert view._g2_q_range() == (0.0, 0.5)
    assert [b[0] for b in view._g2_bin_table()[1]] == [2, 3]


def test_a3_only_check_g2_number_reads_the_q_boxes():
    import re

    allowed = re.compile(r"^\s*self\.g2_q(min|max),?\s*$|\.(setValue|setMaximum)\(")
    offenders = []
    for path in PKG.rglob("*.py"):
        if path.name == "viewer_ui.py":
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"\bg2_q(min|max)\b", line) and not allowed.search(line):
                offenders.append(f"{path.name}:{n}: {line.strip()}")
    assert offenders == []


# --------------------------------------------------------------------- A4
def _sparse_twotime(path):
    """Map q = 0.01, 0.02, 0.03; only bins 1 and 3 were processed (0.02 was not)."""
    from tests.unit import test_review_r4_cli_average as r4cli

    r4cli._write_twotime(path, [0.01, 0.02, 0.03])
    with h5py.File(path, "a") as f:
        for name in ("c2_processed_bins", "c2_g2", "c2_g2_segments"):
            del f[K[name]]
        f[K["c2_processed_bins"]] = np.array([1, 3])
        f[K["c2_g2"]] = np.ones((5, 2))
        f[K["c2_g2_segments"]] = np.ones((5, 2, 2))
        del f[f"{K['c2_prefix']}/c2_00002"]
    return path


def _cli(tmp_path, **kw):
    from tests.unit import test_review_r4_cli_average as r4cli
    from xpcsviewer.cli import twotime_batch as tb

    source, out = tmp_path / "src", tmp_path / "out"
    source.mkdir(exist_ok=True)
    unmatched: list[str] = []
    n = tb.process_single_file(
        str(source / "f.h5"), r4cli._args(source, out, **kw), unmatched=unmatched
    )
    return n, unmatched


def test_a4_an_unprocessed_map_bin_is_not_taken_for_its_processed_neighbour(
    tmp_path, caplog
):
    (tmp_path / "src").mkdir()
    _sparse_twotime(tmp_path / "src" / "f.h5")
    with caplog.at_level("WARNING"):
        n, unmatched = _cli(tmp_path, q=0.02)
    assert n == 0 and unmatched == [str(tmp_path / "src" / "f.h5")]
    assert "exists in the map but was not processed" in caplog.text


@pytest.mark.parametrize("q", [0.01, 0.0104, 0.0296])
def test_a4_a_request_at_or_near_a_processed_bin_is_unchanged(tmp_path, q):
    (tmp_path / "src").mkdir()
    _sparse_twotime(tmp_path / "src" / "f.h5")
    n, unmatched = _cli(tmp_path, q=q)
    assert n == 1 and unmatched == []


def test_a4_a_single_value_axis_at_zero_has_an_absolute_tolerance(tmp_path):
    from tests.unit import test_review_r4_cli_average as r4cli
    from tests.unit.test_review_r7_units import _axes

    (tmp_path / "src").mkdir()
    path = tmp_path / "src" / "f.h5"
    r4cli._write_twotime(path, [0.0])
    _axes(path, [b"qx", b"qy"], [b"1/A", b"1/A"])
    assert _cli(tmp_path, q=1e-9) == (1, [])
    n, unmatched = _cli(tmp_path, q=0.01)
    assert n == 0 and unmatched == [str(path)]


def test_a4_a_file_refused_for_its_unit_is_listed_as_skipped(tmp_path):
    from tests.unit.test_review_r7_units import _cartesian_cli

    (tmp_path / "src").mkdir()
    path = _cartesian_cli(tmp_path / "src" / "f.h5", units=b"px")
    n, unmatched = _cli(tmp_path, q=-0.03)  # pixels, selector in 1/A (no --q-unit)
    assert n == 0 and unmatched == [str(path)]


# --------------------------------------------------------------------- A5
def _diffusion_env(qtbot, monkeypatch, tmp_path):
    from tests.unit import test_review_r6_kernel_units as ku
    from tests.unit.test_review_r8_axes import _bayes_env, _select_only

    env = _bayes_env(
        qtbot,
        monkeypatch,
        tmp_path,
        {"a.h5": lambda p: ku._g2_file(p, [0.1, 0.2, 0.3])},
    )
    env.view.tauq_qmin.setText("0.01")
    env.view.tauq_qmax.setText("5")
    env.view.g2_qauto.setChecked(True)
    env.view.g2_tauto.setChecked(True)
    _select_only(env.view, 0)
    return env


def _bayesian_copy(xf, factor=2.0):
    import copy

    summary = copy.deepcopy(xf.fit_summary)
    summary["fit_val"][:, 0, 1] *= factor  # a Bayesian tau that differs from NLSQ's
    return summary


@pytest.mark.parametrize("nlsq_too", [True, False])
def test_a5_the_bayesian_launch_reads_the_summary_the_diffusion_plot_shows(
    qtbot,
    monkeypatch,
    tmp_path,
    nlsq_too,
    _fresh_pool,  # noqa: F811
):
    env = _diffusion_env(qtbot, monkeypatch, tmp_path)
    view, xf = env.view, env.xfs[0]
    nlsq_tau = xf.fit_summary["fit_val"][:, 0, 1].copy()
    xf.bayesian_fit_summary = bayesian = _bayesian_copy(xf)
    if not nlsq_too:
        xf.fit_summary = None  # a file with only a Bayesian G2 fit
    bundle = view._extract_diffusion_for_bayesian()
    assert bundle is not None, view.statusbar.currentMessage()
    np.testing.assert_allclose(bundle.y, 2.0 * nlsq_tau)
    view.plot_diffusion()  # what the diffusion tab shows and fits
    np.testing.assert_allclose(bundle.x, bayesian["tauq_q"])
    np.testing.assert_allclose(bundle.y, bayesian["tauq_tau"])


def test_a5_without_any_g2_fit_the_launch_still_asks_for_one(
    qtbot,
    monkeypatch,
    tmp_path,
    _fresh_pool,  # noqa: F811
):
    env = _diffusion_env(qtbot, monkeypatch, tmp_path)
    env.xfs[0].fit_summary = None
    assert env.view._extract_diffusion_for_bayesian() is None
    assert "Run G2 fitting first" in env.view.statusbar.currentMessage()


# --------------------------------------------------------------------- A6
def test_a6_a_non_finite_partner_component_is_refused_with_its_name(
    qtbot,
    monkeypatch,
    tmp_path,
    _fresh_pool,  # noqa: F811
):
    from tests.unit.test_review_r8_axes import _bayes_env, _cartesian, _select_only
    from xpcsviewer.viewer_kernel import _diffusion_axis_problem

    env = _bayes_env(
        qtbot,
        monkeypatch,
        tmp_path,
        {"nan.h5": lambda p: _cartesian(p, [0.1, 0.2, 0.3], qy=float("nan"))},
    )
    problem = _diffusion_axis_problem(env.xfs[0])
    assert problem is not None and "qy" in problem and "not a finite" in problem
    env.view.tauq_qmin.setText("0.01")
    env.view.tauq_qmax.setText("5")
    _select_only(env.view, 0)
    assert env.view._extract_diffusion_for_bayesian() is None
    assert "qy" in env.view.statusbar.currentMessage()


# --------------------------------------------------------------------- A7
class _Status:
    def __init__(self):
        self.messages = []

    def showMessage(self, text, *_):
        self.messages.append(text)


def _cartesian_saxs(path):
    from tests.unit import test_review_r6_kernel_units as ku

    ku._saxs_file(path, [0.1, 0.2, 0.3], b"1/A")
    with h5py.File(path, "r+") as f:
        del f[K["map_names"]]
        f[K["map_names"]] = [b"qx", b"qy"]
    return path


@pytest.mark.parametrize("cartesian", [True, False])
def test_a7_the_stability_tab_reports_a_refused_normalisation(
    tmp_path,
    qapp,
    cartesian,  # noqa: F811
):
    import pyqtgraph as pg

    from tests.unit import test_review_r6_kernel_units as ku
    from xpcsviewer.viewer_kernel import ViewerKernel

    make = (
        _cartesian_saxs
        if cartesian
        else lambda p: ku._saxs_file(p, [0.1, 0.2, 0.3], b"1/A")
    )
    make(tmp_path / "a.h5")
    status = _Status()
    kernel = ViewerKernel(str(tmp_path), statusbar=status)
    kernel.add_target(["a.h5"])
    widget = pg.PlotWidget()
    kernel.plot_stability(widget, rows=[0], plot_type=2, plot_norm=1)  # I * q^2
    drawn = len(widget.getPlotItem().listDataItems())
    if cartesian:
        assert drawn == 0
        assert status.messages and "radial 'q'" in status.messages[-1]
        assert "Stability" in status.messages[-1]
    else:
        assert drawn > 0 and not status.messages


def test_a7_every_caller_of_the_saxs_normalisation_handles_the_refusal():
    # get_saxs1d_data (which raises it) is called from exactly these two modules
    for module in ("saxs1d.py", "stability.py"):
        assert "SaxsNormalizationRefused" in (PKG / "module" / module).read_text()
    callers = [
        p.name
        for p in PKG.rglob("*.py")
        if ".get_saxs1d_data(" in p.read_text() and p.name != "xpcs_file.py"
    ]
    assert sorted(callers) == ["saxs1d.py", "stability.py"]


# --------------------------------------------------------------------- A8
def test_a8_the_display_axis_is_the_first_files_even_without_a_bin_in_range(
    tmp_path,
    qapp,  # noqa: F811
):
    from tests.unit import test_review_r6_kernel_units as ku

    first = ku._saxs_file(tmp_path / "a.h5", [0.1, 0.2, 0.3], b"1/A")
    second = ku._saxs_file(tmp_path / "b.h5", [3.0, 4.0, 5.0], b"1/nm")  # 0.3..0.5 A^-1
    # typed in the first file's unit; only the SECOND file has a bin (0.4 A^-1)
    label, curves = ku._draw(
        ku._kernel(tmp_path, [first.name, second.name]), qmin=0.35, qmax=0.45
    )
    assert ku.Q_UNIT_DISPLAY in label and ku.NM not in label
    assert len(curves) == 1
    np.testing.assert_allclose(curves[0], [0.4])


# --------------------------------------------------------------------- A9
from tests.unit.test_review_r8_identity import (  # noqa: E402
    _click,
    _hover,
    _idle,
    _image,
    _replace_with,
    _spin,
    _write_frame,
    saxs,  # noqa: F401
)


def test_a9_removing_the_last_file_clears_the_saxs2d_view(saxs):  # noqa: F811
    view = saxs.view
    _click(view, 0)
    assert view.pg_saxs.rendered_frame is not None and view.pg_saxs.image is not None
    view.list_view_target.selectAll()
    view.remove_target()
    _idle(view)
    assert len(view.vk.target) == 0
    assert view.pg_saxs.rendered_frame is None
    assert view.pg_saxs.image is None  # the picture is gone, not just its record
    # the view is usable again: a file added back is drawn and read at the cursor
    view.vk.add_target(["B.h5"])
    view.update_box(view.vk.target, mode="target")
    _click(view, 0)
    np.testing.assert_array_equal(_image(view), [81])
    assert _hover(view).startswith("I=8.1000e+01")


def test_a9_a_file_changed_on_disk_is_named_by_the_readout_and_the_roi_button(
    saxs,  # noqa: F811
):
    view = saxs.view
    _click(view, 0)
    _spin(view, 1)
    assert _hover(view).startswith("I=8.1000e+01")
    _replace_with(saxs.b, lambda p: _write_frame(p, 7, (6.0, 5.0)))
    assert _hover(view) == "file changed on disk"
    view.statusbar.clearMessage()
    view.saxs2d_roi_add()
    shown = view.statusbar.currentMessage()
    assert "changed on disk" in shown and "re-plot" in shown
    assert "No SAXS 2D image is displayed" not in shown
    assert view.pg_saxs.get_roi_list() == []


def test_a9_with_nothing_drawn_the_roi_button_still_says_so(saxs):  # noqa: F811
    view = saxs.view
    view.pg_saxs.clear()
    view.saxs2d_roi_add()
    assert "No SAXS 2D image is displayed" in view.statusbar.currentMessage()


# -------------------------------------------------------------------- A10
@pytest.mark.parametrize("end", ["complete", "cancel"])
def test_a10_an_auto_registered_fixed_id_reports_again_after_it_ends(qtbot, end):
    from xpcsviewer.threading.progress_manager import ProgressManager

    manager = ProgressManager()
    seen = []
    manager.operation_progress.connect(lambda op, cur, tot, st: seen.append((op, cur)))
    manager.update_progress("preload_data", 1, 10, "first pass")  # auto-registered
    assert manager.is_operation_active("preload_data")
    if end == "complete":
        manager.complete_operation("preload_data", True)
    else:
        manager.cancel_operation_ui("preload_data")
    assert not manager.is_operation_active("preload_data")
    manager.update_progress("preload_data", 2, 10, "second pass")  # the next run
    assert manager.is_operation_active("preload_data")
    assert manager.active_operations["preload_data"].current == 2
    assert seen == [("preload_data", 1), ("preload_data", 2)]


def test_a10_an_explicitly_started_id_stays_closed_to_late_messages(qtbot):
    from xpcsviewer.threading.progress_manager import ProgressManager

    manager = ProgressManager()
    manager.start_operation("batch", "Batch", total=10, show_after_delay=-1)
    manager.complete_operation("batch", True)
    manager.update_progress("batch", 3, 10, "late")
    assert not manager.is_operation_active("batch")


_ = threading, h5py, K
