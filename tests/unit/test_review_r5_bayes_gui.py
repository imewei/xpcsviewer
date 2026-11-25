"""Round 5 regressions for the Bayesian GUI layer.

R5-11, R5-16  every diagnosis window carries what it shows (file, revision,
              absolute bin, bundle); navigation and invalidation ask the window
R5-04, R5-14  Bayesian diffusion uses the deterministic fit's row selection with
              the CURRENT GUI Q range and refuses an unidentifiable exponent
R5-10         the All-Q netCDF writer skips failed bins and tags the rest
R5-05         the All-Q CSV names the absolute bin, numeric Q and azimuth
R5-01         every file written by these modules is claimed through a guard

Tests marked "real" build the real ``XpcsViewer`` offscreen over HDF5 files
written with h5py (see ``test_review_r4_viewer_gui`` for the fixtures).
"""

from __future__ import annotations

import ast
import csv
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from qtpy import QtCore, QtWidgets

from tests.unit.test_review_r3_fit_status import (
    BOUNDS,
    TAUQ_BOUNDS,
    _model,
    _write_file,
)
from tests.unit.test_review_r4_viewer_gui import (  # noqa: F401  (fixtures)
    _FakeWorker,
    _launch_fit,
    _replace,
    g2_env,
    viewer_env,
)
from xpcsviewer.fileIO.hdf_reader import _file_identity

pytestmark = pytest.mark.unit

PKG = Path(__file__).resolve().parents[2] / "xpcsviewer"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _real_result(seed=0, names=("tau", "baseline", "contrast"), converged=True):
    """A small real FitResult (converged unless ``converged`` is False)."""
    from xpcsviewer.fitting.results import FitDiagnostics, FitResult

    rng = np.random.default_rng(seed)
    centre = {"tau": 0.1, "baseline": 1.0, "contrast": 0.5, "tau0": 0.3, "alpha": 1.5}
    samples = {n: centre[n] + 0.01 * rng.standard_normal(100) for n in names}
    diagnostics = FitDiagnostics(bfmi=1.0, divergences=0 if converged else 50)
    return FitResult(samples=samples, diagnostics=diagnostics)


def _finish_real(view, launch):
    view._on_g2_bayesian_finished({"launch": launch, "fit_result": _real_result()})


def _open_g2_window(view):
    view.btn_g2_diagnosis.click()
    window = view._g2_diagnosis_window
    assert window is not None and window._result is not None
    return window


def _alive(window):
    try:
        return window.isVisible()
    except RuntimeError:  # C++ object already deleted
        return False


def _rev(path):
    return _file_identity(os.path.normpath(path))


def _select_row(view, row=0):
    view.list_view_target.selectionModel().select(
        view.vk.target.index(row, 0),
        QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect,
    )


# --------------------------------------------------------------------------
# R5-11: moving to an unfitted bin
# --------------------------------------------------------------------------
def test_r5_11_real_unfitted_bin_clears_the_open_diagnosis_window(g2_env):
    view = g2_env.view
    view.sb_g2_bayesian_qidx.setMaximum(2)
    launch = _launch_fit(view)
    _finish_real(view, launch)
    window = _open_g2_window(view)
    assert window._btn_export_fit_data.isEnabled()

    view.sb_g2_bayesian_qidx.setValue(1)  # bin 1 has no fit

    assert "no fit yet" in window.windowTitle()
    # nothing of bin 0 survives under the new title
    assert window._result is None and window._x_data is None
    assert window._pp_canvas.figure.axes == []
    for button in (
        window._btn_export_fit_plot,
        window._btn_export_fit_data,
        window._btn_export_diag_plots,
        window._btn_export_traces,
    ):
        assert not button.isEnabled()
    assert not window.shows(launch.fname, _rev(launch.fname), launch.bin_id)

    view.sb_g2_bayesian_qidx.setValue(0)  # back to the fitted bin: shown again
    assert window.shows(launch.fname, _rev(launch.fname), launch.bin_id)
    assert window._result is not None and window._btn_export_fit_data.isEnabled()


def test_r5_11_window_export_after_clear_writes_nothing(g2_env, monkeypatch, tmp_path):
    view = g2_env.view
    view.sb_g2_bayesian_qidx.setMaximum(2)
    _finish_real(view, _launch_fit(view))
    window = _open_g2_window(view)
    view.sb_g2_bayesian_qidx.setValue(1)
    out = tmp_path / "stale.csv"
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), "")
    )
    window._export_fit_data()
    window._export_fit_plot()
    assert not out.exists()


# --------------------------------------------------------------------------
# R5-16: windows are invalidated by their own identity
# --------------------------------------------------------------------------
def test_r5_16_real_window_of_a_superseded_fit_is_closed_on_reload(g2_env):
    env = g2_env
    view = env.view
    _finish_real(view, _launch_fit(view))
    window = _open_g2_window(view)
    # refit of the same bin replaces the stored bundle the window was built from
    first = view._g2_current_bundle()
    _finish_real(view, _launch_fit(view))
    assert view._g2_current_bundle() is not first

    _replace(env.root, env.data, "a.h5", 999, (0.17, 0.27, 0.37))
    view.reload_source()

    assert view._g2_diagnosis_window is None
    assert not _alive(window)


def test_r5_16_real_diffusion_window_of_a_superseded_fit_is_closed_on_reload(g2_env):
    from xpcsviewer.fitting.bayesian_assembly import BayesianBundle

    env = g2_env
    view = env.view
    xf = view.vk.get_xf_list([0])[0]

    def commit():
        bundle = BayesianBundle.create(
            xf.fname, "tau-q", "tau(q)", 0.0, "power_law", [1, 2], [1, 2], [0.1, 0.1]
        )
        view._diff_bayesian_launch = bundle
        view._diff_bayesian_launch_revision = _rev(xf.fname)
        view._diff_bayesian_worker_active = True
        result = _real_result(names=("tau0", "alpha"))
        view._on_diff_bayesian_finished({"launch": bundle, "fit_result": result})

    commit()
    view._show_diff_diagnosis()
    window = view._diff_diagnosis_window
    assert window is not None and window._result is not None
    commit()  # same file refit: the store now holds another bundle

    _replace(env.root, env.data, "a.h5", 999, (0.17, 0.27, 0.37))
    view.reload_source()

    assert view._diff_diagnosis_window is None
    assert not _alive(window)


def test_r5_16_real_window_of_another_file_survives_the_reload(g2_env):
    from xpcsviewer.fitting.bayesian_assembly import BayesianBundle

    env = g2_env
    view = env.view
    path_b = str(env.data / "b.h5")
    bundle = BayesianBundle.create(
        path_b, 0, "q", 0.1, "single", [0.1, 0.2, 0.3], [1.3, 1.2, 1.1], [0.01] * 3
    ).with_result(_real_result())
    view._update_g2_diagnosis(bundle)
    window = view._g2_diagnosis_window
    assert window.shows(path_b, _rev(path_b), 0)

    _replace(env.root, env.data, "a.h5", 999, (0.17, 0.27, 0.37))
    view.reload_source()

    assert view._g2_diagnosis_window is window and _alive(window)


def test_r5_16_destroyed_window_clears_the_viewers_reference(g2_env):
    import shiboken6

    view = g2_env.view
    _finish_real(view, _launch_fit(view))
    window = _open_g2_window(view)
    shiboken6.delete(window)
    assert view._g2_diagnosis_window is None

    # a late destroyed() of an old window never clears the newer one
    old, new = object(), object()
    view._g2_diagnosis_window = new
    view._forget_window("_g2_diagnosis_window", old)
    assert view._g2_diagnosis_window is new
    view._g2_diagnosis_window = None


def test_r5_16_window_api_identity(qtbot):
    from xpcsviewer.fitting.models import single_exp_func
    from xpcsviewer.gui.bayesian_diagnosis import (
        BayesianDiagnosisWindow,
        DiagnosisIdentity,
    )

    window = BayesianDiagnosisWindow()
    qtbot.addWidget(window)
    assert window.identity is None and not window.shows("f.h5", (1,), 0)
    ident = DiagnosisIdentity("f.h5", (1, 2), 3, bundle="the bundle")
    window.update_results(
        result=_real_result(),
        model_func=single_exp_func,
        x_data=[0.1, 0.2, 0.3],
        y_data=[1.3, 1.2, 1.1],
        yerr=[0.01] * 3,
        identity=ident,
    )
    assert window.identity is ident and window.identity.bundle == "the bundle"
    assert window.shows("f.h5", (1, 2), 3)
    assert window.shows("f.h5", (1, 2))  # any bin
    assert not window.shows("f.h5", (1, 2), 4)
    assert not window.shows("f.h5", (9, 9), 3)
    assert not window.shows("g.h5", (1, 2), 3)
    assert window.matches(lambda path, rev: (path, rev) == ("f.h5", (1, 2)))
    assert not window.matches(lambda path, rev: path == "g.h5")
    window.clear("nothing here")
    assert window.identity is None and window.windowTitle() == "nothing here"
    assert not window.shows("f.h5", (1, 2), 3)
    assert not window.matches(lambda path, rev: True)


def _calls(tree, attr=None, name=None):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if attr is not None and isinstance(func, ast.Attribute) and func.attr == attr:
            yield node
        if name is not None and isinstance(func, ast.Name) and func.id == name:
            yield node


def _enclosing(tree):
    owner = {}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(fn):
                owner[id(node)] = fn.name  # innermost function wins (walk order)
    return owner


def test_r5_every_diagnosis_window_creation_and_update_passes_its_identity():
    tree = ast.parse((PKG / "xpcs_viewer.py").read_text())
    owner = _enclosing(tree)

    updates = list(_calls(tree, attr="update_results"))
    assert {owner[id(c)] for c in updates} == {
        "_update_g2_diagnosis",
        "_update_diff_diagnosis",
    }
    for call in updates:
        keywords = {k.arg for k in call.keywords}
        assert "identity" in keywords, ast.unparse(call)[:80]

    creations = list(_calls(tree, name="BayesianDiagnosisWindow"))
    assert {owner[id(c)] for c in creations} == {
        "_update_g2_diagnosis",
        "_update_diff_diagnosis",
    }
    # the identity is built from the bundle shown, never from the store
    src = (PKG / "xpcs_viewer.py").read_text()
    assert "_g2_diagnosis_bundle" not in src and "_diff_diagnosis_bundle" not in src
    for fn in ("_update_g2_diagnosis", "_update_diff_diagnosis"):
        body = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fn
        )
        assert "DiagnosisIdentity" in ast.unparse(body)

    # the window API is explicit: update_results requires the identity
    diag = ast.parse((PKG / "gui" / "bayesian_diagnosis.py").read_text())
    update = next(
        n
        for n in ast.walk(diag)
        if isinstance(n, ast.FunctionDef) and n.name == "update_results"
    )
    required_kwonly = [
        a.arg
        for a, d in zip(update.args.kwonlyargs, update.args.kw_defaults, strict=True)
        if d is None
    ]
    assert "identity" in required_kwonly


# --------------------------------------------------------------------------
# R5-04 / R5-14: Bayesian diffusion rows
# --------------------------------------------------------------------------
@pytest.fixture
def diff_env(g2_env):
    """Real viewer over a.h5 with Q = (0.2, 0.3, 1.0) and a deterministic G2 fit."""
    env = g2_env
    view = env.view
    # 28 noisy delays (as the reviewer's probe): every bin gets a covariance
    delays = np.geomspace(0.002, 1, 28)
    g2 = _model(delays)
    g2 += np.random.default_rng(42).normal(0, 0.012, g2.shape)
    tmp = env.root / "replacement_a.h5"
    _write_file(tmp, delays, g2=g2, q=(0.2, 0.3, 1.0))
    os.replace(tmp, env.data / "a.h5")
    view.reload_source()
    _select_row(view, 0)
    xf = view.vk.get_xf_list([0])[0]
    summary = xf.fit_g2(bounds=BOUNDS, fit_flag=[True] * 4, force_refit=True)
    assert summary["fit_success"].all()
    for name, text in {"tauq_amin": "0.001", "tauq_amax": "1000"}.items():
        getattr(view, name).setText(text)
    view.tauq_bmin.setText("-3")
    view.tauq_bmax.setText("3")
    env.xf = xf
    return env


def _set_q(view, lo, hi):
    view.tauq_qmin.setText(str(lo))
    view.tauq_qmax.setText(str(hi))


def test_r5_14_real_bayesian_rows_follow_the_current_gui_q_range(diff_env):
    view, xf = diff_env.view, diff_env.xf
    xf.fit_tauq((0.1, 1.1), TAUQ_BOUNDS, [True, True], force_refit=True)
    assert xf.fit_summary["tauq_success"]
    assert xf.fit_summary["tauq_q"].tolist() == [0.2, 0.3, 1.0]  # earlier fit

    _set_q(view, 0.25, 1.1)
    bundle = view._extract_diffusion_for_bayesian()

    assert bundle.x.tolist() == [0.3, 1.0]
    assert bundle.q_range == (0.25, 1.1)  # the bundle describes its own range
    assert np.array_equal(bundle.y, xf.fit_summary["fit_val"][1:, 0, 1])


def test_r5_04_real_single_q_selection_is_refused_not_fitted(diff_env):
    view, xf = diff_env.view, diff_env.xf
    xf.fit_tauq((0.1, 1.1), TAUQ_BOUNDS, [True, True], force_refit=True)
    _set_q(view, 0.99, 1.01)  # one distinct Q: the exponent is not identifiable

    view._fit_diff_bayesian()

    assert _FakeWorker.instances == []
    assert not view._diff_bayesian_worker_active
    assert view.btn_diff_bayesian.isEnabled()
    message = view.statusbar.currentMessage()
    assert "2 distinct" in message and "positive" in message


def test_r5_04_real_two_q_selection_is_launched_with_its_rows(diff_env):
    view = diff_env.view
    _set_q(view, 0.0, 0.35)
    view._fit_diff_bayesian()
    (worker,) = _FakeWorker.instances
    assert worker.launch.x.tolist() == [0.2, 0.3]
    assert worker.launch.q_range == (0.0, 0.35)
    assert view._diff_bayesian_worker_active


def test_r5_04_real_failed_g2_bins_never_enter_the_rows(diff_env):
    view, xf = diff_env.view, diff_env.xf
    xf.fit_summary["fit_success"][1] = False  # the Q = 0.3 bin failed
    _set_q(view, 0.0, 2.0)
    bundle = view._extract_diffusion_for_bayesian()
    assert bundle.x.tolist() == [0.2, 1.0]


def test_r5_04_real_unreadable_q_range_is_refused(diff_env):
    view = diff_env.view
    view.tauq_qmin.setText("abc")
    assert view._extract_diffusion_for_bayesian() is None
    view._fit_diff_bayesian()
    assert _FakeWorker.instances == []


def test_r5_04_real_row_selection_agrees_with_fit_tauq(diff_env):
    """``XpcsFile.fit_tauq`` is the source of truth for the (q, tau, err) rows."""
    from xpcsviewer.xpcs_viewer import _tauq_rows

    xf = diff_env.xf
    ranges = [(0.1, 1.1), (0.25, 1.1), (0.0, 0.35), (0.99, 1.01), (0.2, 0.3), (5, 6)]
    for fail_bin in (None, 1):
        xf.fit_summary["fit_success"][:] = True
        if fail_bin is not None:
            xf.fit_summary["fit_success"][fail_bin] = False
        for q_range in ranges:
            xf.fit_tauq(q_range, TAUQ_BOUNDS, [True, True], force_refit=True)
            summary = xf.fit_summary
            q, tau, err = _tauq_rows(summary, q_range)
            if np.unique(q).size < 2:  # fit_tauq refuses: it empties its rows
                assert summary["tauq_q"].size == 0, (fail_bin, q_range)
                continue
            assert np.array_equal(q, summary["tauq_q"]), (fail_bin, q_range)
            assert np.array_equal(tau, summary["tauq_tau"]), (fail_bin, q_range)
            assert np.array_equal(err, summary["tauq_tau_err"]), (fail_bin, q_range)


# --------------------------------------------------------------------------
# R5-10: failed bins are not exported to netCDF, the others are tagged
# --------------------------------------------------------------------------
class _FakeIdata:
    """Stands in for an ArviZ DataTree: records what ``to_netcdf`` receives."""

    def __init__(self, sink, attrs=None):
        self.sink = sink
        self.attrs = dict(attrs or {})

    def copy(self):
        return _FakeIdata(self.sink, self.attrs)

    def to_netcdf(self, path):
        self.sink[Path(path).name] = dict(self.attrs)
        Path(path).write_bytes(b"netcdf")


def _patch_backends(monkeypatch, available=("h5netcdf",)):
    real = importlib.util.find_spec

    def find_spec(name, *a, **k):
        if name in ("netCDF4", "h5netcdf"):
            return object() if name in available else None
        return real(name, *a, **k)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


def _fake_results(sink, n):
    return {
        i: SimpleNamespace(arviz_data=_FakeIdata(sink), diagnostics=None)
        for i in range(n)
    }


def test_r5_10_writer_skips_failed_bins_and_tags_the_written_ones(
    tmp_path, monkeypatch
):
    from xpcsviewer.fitting.viz import export_bayesian_diagnostics

    _patch_backends(monkeypatch)
    sink: dict = {}
    results = _fake_results(sink, 3)
    results[3] = None
    written = export_bayesian_diagnostics(
        tmp_path / "d.nc",
        results,
        failed_mask=np.array([False, True, False, True]),
        bin_ids=np.array([10, 11, 12, 13]),
        q_val=np.array([0.1, 0.1, 0.2, 0.3]),
        bin_phi=np.array([0.0, 90.0, np.nan, np.nan]),
        source="scan.h5",
    )
    assert [p.name for p in written] == ["d_q000.nc", "d_q002.nc"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["d_q000.nc", "d_q002.nc"]
    assert sink["d_q000.nc"] == {
        "bin_status": "ok",
        "bin_id": 10,
        "q_index": 0,
        "q_value": 0.1,
        "phi": 0.0,
        "source_file": "scan.h5",
    }
    assert sink["d_q002.nc"]["bin_id"] == 12 and "phi" not in sink["d_q002.nc"]


def test_r5_10_writer_without_a_mask_writes_all_and_says_unassessed(
    tmp_path, monkeypatch
):
    from xpcsviewer.fitting.viz import export_bayesian_diagnostics

    _patch_backends(monkeypatch)
    sink: dict = {}
    written = export_bayesian_diagnostics(tmp_path / "d.nc", _fake_results(sink, 2))
    assert len(written) == 2
    assert {a["bin_status"] for a in sink.values()} == {"unassessed"}


def test_r5_10_writer_without_a_mask_skips_results_that_did_not_converge(
    tmp_path, monkeypatch
):
    from xpcsviewer.fitting.viz import export_bayesian_diagnostics

    _patch_backends(monkeypatch)
    sink: dict = {}
    results = _fake_results(sink, 2)
    results[0].diagnostics = SimpleNamespace(converged=True)
    results[1].diagnostics = SimpleNamespace(converged=False)
    written = export_bayesian_diagnostics(tmp_path / "d.nc", results)
    assert [p.name for p in written] == ["d_q000.nc"]
    assert sink["d_q000.nc"]["bin_status"] == "ok"


@pytest.mark.skipif(
    not any(importlib.util.find_spec(m) for m in ("h5netcdf", "netCDF4")),
    reason="no netCDF backend installed (not a dependency)",
)
def test_r5_10_real_netcdf_files_carry_status_and_identity(tmp_path):
    from xpcsviewer.fitting import fit_single_exp
    from xpcsviewer.fitting.viz import export_bayesian_diagnostics

    t = np.geomspace(0.005, 1, 25)
    y = _model(t)[:, 0]
    results = {
        j: fit_single_exp(
            t,
            y,
            np.full_like(y, 0.01),
            num_chains=2,
            num_warmup=40,
            num_samples=40,
            random_seed=j,
            progress_bar=False,
        )
        for j in range(2)
    }
    written = export_bayesian_diagnostics(
        tmp_path / "d.nc",
        results,
        failed_mask=np.array([False, True]),
        bin_ids=np.array([7, 8]),
        q_val=np.array([0.1, 0.2]),
        source="scan.h5",
    )
    assert [p.name for p in written] == ["d_q000.nc"]
    with h5py.File(written[0], "r") as f:
        assert f.attrs["bin_status"] == "ok"
        assert int(np.ravel(f.attrs["bin_id"])[0]) == 7
        assert "posterior" in f
    # the caller's FitResult is not modified by tagging
    assert "bin_id" not in results[0].arviz_data.attrs


# --------------------------------------------------------------------------
# R5-05: the All-Q CSV names the absolute bin, numeric Q and azimuth
# --------------------------------------------------------------------------
def test_r5_05_csv_rows_carry_bin_id_q_and_phi(tmp_path):
    from xpcsviewer.fitting.viz import export_bayesian_csv

    fit_val = np.ones((2, 2, 4))
    out = tmp_path / "p.csv"
    export_bayesian_csv(
        out,
        fit_val,
        np.array([0.1, 0.1]),
        "single",
        failed_mask=np.array([False, False]),
        bin_ids=np.array([0, 1]),
        bin_phi=np.array([0.0, 90.0]),
    )
    rows = list(csv.DictReader(out.open()))
    assert [r["q_value"] for r in rows] == ["0.100000", "0.100000"]  # kept
    assert [r["bin_id"] for r in rows] == ["0", "1"]
    assert [float(r["phi"]) for r in rows] == [0.0, 90.0]
    assert rows[0]["status"] == "ok" and rows[0]["tau_mean"]
    header = out.read_text().splitlines()[0]
    assert header.startswith("q_value,contrast_mean")  # existing columns first


def test_r5_05_csv_without_bin_info_keeps_the_old_layout(tmp_path):
    from xpcsviewer.fitting.viz import export_bayesian_csv

    out = tmp_path / "p.csv"
    export_bayesian_csv(
        out, np.ones((1, 2, 4)), np.array([0.1]), "single", failed_mask=np.array([True])
    )
    header = out.read_text().splitlines()[0]
    assert "bin_id" not in header and header.endswith("status")


def test_r5_05_real_file_azimuth_of_each_absolute_bin(tmp_path):
    from xpcsviewer.fitting.bayesian_assembly import resolve_bin_phi, resolve_g2_bins
    from xpcsviewer.xpcs_file import XpcsFile

    path = tmp_path / "aniso.h5"
    _write_file(path, np.geomspace(0.005, 1, 12), q=(0.1, 0.1), saxs=(10, 5))
    with h5py.File(path, "r+") as f:
        qmap = f["/xpcs/qmap"]
        for name, values in {
            "dynamic_v_list_dim0": [0.1],
            "dynamic_v_list_dim1": [0.0, 90.0],
        }.items():
            del qmap[name]
            qmap[name] = values
        qmap["dynamic_num_pts"][:] = [1, 2]
    xf = XpcsFile(str(path))
    bins = resolve_g2_bins(xf, None)
    assert [b[1] for b in bins] == [0.1, 0.1]
    assert resolve_bin_phi(xf, [b[0] for b in bins]).tolist() == [0.0, 90.0]


def test_r5_05_summary_carries_the_azimuths():
    from xpcsviewer.fitting.bayesian_assembly import assemble_fit_summary
    from xpcsviewer.fitting.models import single_exp_func

    t = np.geomspace(0.001, 1, 14)
    summary = assemble_fit_summary(
        {0: _real_result(), 1: _real_result(1)},
        np.array([0.1, 0.1]),
        t,
        "single",
        single_exp_func,
        bin_ids=[0, 1],
        bin_phi=[0.0, 90.0],
    )
    assert summary["bin_phi"].tolist() == [0.0, 90.0]


def _all_q_snapshot(path):
    from xpcsviewer.fitting.bayesian_assembly import assemble_fit_summary
    from xpcsviewer.fitting.models import single_exp_func
    from xpcsviewer.xpcs_viewer import _bayesian_export_snapshot

    t = np.geomspace(0.001, 1, 14)
    results = {
        0: _real_result(0),
        1: _real_result(1),
        2: _real_result(2, converged=False),
    }
    summary = assemble_fit_summary(
        results,
        np.array([0.1, 0.1, 0.2]),
        t,
        "single",
        single_exp_func,
        fname=str(path),
        bin_ids=[0, 1, 2],
        bin_phi=[0.0, 90.0, 0.0],
    )
    xf = SimpleNamespace(
        fname=str(path), bayesian_fit_summary=summary, bayesian_results=results
    )
    return _bayesian_export_snapshot(xf)


@pytest.fixture
def export_env(viewer_env, monkeypatch, tmp_path):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    _patch_backends(monkeypatch)
    sink: dict = {}
    snapshot = _all_q_snapshot(viewer_env.data / "a.h5")
    results = {
        i: SimpleNamespace(arviz_data=_FakeIdata(sink), diagnostics=None)
        for i in snapshot.results
    }
    snapshot = type(snapshot)(
        summary=snapshot.summary,
        results=results,
        fname=snapshot.fname,
        revision=snapshot.revision,
    )
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getExistingDirectory", lambda *a, **k: str(out)
    )
    fig = plt.figure()
    yield SimpleNamespace(
        view=viewer_env.view,
        env=viewer_env,
        snapshot=snapshot,
        fig=fig,
        out=out,
        sink=sink,
    )
    plt.close(fig)


def test_r5_05_r5_10_real_all_q_export_names_bins_and_skips_failed(export_env):
    e = export_env
    e.view._export_bayesian(e.snapshot, e.fig)
    rows = list(csv.DictReader((e.out / "bayesian_params.csv").open()))
    assert [r["bin_id"] for r in rows] == ["0", "1", "2"]
    assert [float(r["phi"]) for r in rows] == [0.0, 90.0, 0.0]
    assert [r["status"] for r in rows] == ["ok", "ok", "not_converged"]
    # the netCDF set holds exactly the converged bins, each tagged
    assert sorted(p.name for p in e.out.glob("*.nc")) == [
        "bayesian_diagnostics_q000.nc",
        "bayesian_diagnostics_q001.nc",
    ]
    assert e.sink["bayesian_diagnostics_q001.nc"]["bin_id"] == 1
    assert e.sink["bayesian_diagnostics_q001.nc"]["phi"] == 90.0


# --------------------------------------------------------------------------
# R5-01: one guard per export action
# --------------------------------------------------------------------------
def test_r5_01_real_all_q_export_never_writes_through_an_alias_of_the_input(
    export_env,
):
    e = export_env
    source = e.env.data / "a.h5"
    before = source.read_bytes()
    os.symlink(source, e.out / "bayesian_params.csv")
    os.symlink(source, e.out / "bayesian_all_q.png")

    e.view._export_bayesian(e.snapshot, e.fig)

    assert source.read_bytes() == before  # the input file is untouched
    assert (e.out / "bayesian_all_q.pdf").is_file()  # unaffected output
    assert (e.out / "bayesian_diagnostics_q000.nc").is_file()
    message = e.view.statusbar.currentMessage()
    assert "refus" in message.lower()


def test_r5_01_real_all_q_export_refuses_an_alias_between_its_own_outputs(export_env):
    e = export_env
    os.symlink(e.out / "bayesian_params.csv", e.out / "bayesian_all_q.pdf")
    e.view._export_bayesian(e.snapshot, e.fig)
    # the CSV was written first; the pdf alias of it was refused, not written
    text = (e.out / "bayesian_params.csv").read_text()
    assert text.startswith("q_value,")  # not overwritten by the figure
    assert "refus" in e.view.statusbar.currentMessage().lower()


def _window_with_source(qtbot, source, converged=True):
    from xpcsviewer.fitting.models import single_exp_func
    from xpcsviewer.gui.bayesian_diagnosis import (
        BayesianDiagnosisWindow,
        DiagnosisIdentity,
    )

    window = BayesianDiagnosisWindow()
    qtbot.addWidget(window)
    result = _real_result(converged=converged)
    window.update_results(
        result=result,
        model_func=single_exp_func,
        x_data=[0.1, 0.2, 0.3],
        y_data=[1.3, 1.2, 1.1],
        yerr=[0.01] * 3,
        identity=DiagnosisIdentity(str(source), _rev(source), 0, None),
    )
    return window


@pytest.mark.parametrize("export", ["_export_fit_plot", "_export_fit_data"])
def test_r5_01_diagnosis_window_single_file_export_refuses_the_input(
    qtbot, monkeypatch, tmp_path, export
):
    source = tmp_path / "scan.h5"
    source.write_bytes(b"HDF-the-original-bytes")
    alias = tmp_path / "alias.png"
    os.symlink(source, alias)
    window = _window_with_source(qtbot, source)
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (str(alias), "")
    )
    getattr(window, export)()
    assert source.read_bytes() == b"HDF-the-original-bytes"
    assert "refus" in window.statusBar().currentMessage().lower()


def test_r5_01_diagnosis_window_plot_folder_export_refuses_aliased_files(
    qtbot, monkeypatch, tmp_path
):
    source = tmp_path / "scan.h5"
    source.write_bytes(b"HDF-the-original-bytes")
    out = tmp_path / "plots"
    out.mkdir()
    os.symlink(source, out / "mcmc_energy.png")
    window = _window_with_source(qtbot, source)
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getExistingDirectory", lambda *a, **k: str(out)
    )
    window._export_diag_plots()
    assert source.read_bytes() == b"HDF-the-original-bytes"
    assert (out / "mcmc_pair.png").is_file()  # the other plots are still written
    assert "refus" in window.statusBar().currentMessage().lower()


def test_r5_01_diagnosis_window_traces_export_refuses_the_input(
    qtbot, monkeypatch, tmp_path
):
    source = tmp_path / "scan.h5"
    source.write_bytes(b"HDF-the-original-bytes")
    window = _window_with_source(qtbot, source)
    written = []
    window._result.arviz_data = SimpleNamespace(to_netcdf=lambda p: written.append(p))
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (str(source), "")
    )
    window._export_traces()
    assert written == []
    assert source.read_bytes() == b"HDF-the-original-bytes"


def test_r5_01_viz_writers_claim_through_a_given_guard(tmp_path):
    from xpcsviewer.fitting.viz import export_bayesian_csv
    from xpcsviewer.utils.export_guard import ExportCollisionError, ExportGuard

    source = tmp_path / "scan.h5"
    source.write_bytes(b"data")
    guard = ExportGuard([source])
    with pytest.raises(ExportCollisionError):
        export_bayesian_csv(
            source, np.ones((1, 2, 4)), np.array([0.1]), "single", guard=guard
        )
    assert source.read_bytes() == b"data"
    # a second claim of the same destination within one action is refused too
    ok = tmp_path / "p.csv"
    export_bayesian_csv(ok, np.ones((1, 2, 4)), np.array([0.1]), "single", guard=guard)
    with pytest.raises(ExportCollisionError):
        export_bayesian_csv(
            ok, np.ones((1, 2, 4)), np.array([0.1]), "single", guard=guard
        )


# Every file-writing call in these modules sits in a function that claims through
# a guard, or is listed here with the reason it is not a user export.
_WRITE_ATTRS = {
    "savefig",
    "savetxt",
    "to_netcdf",
    "to_csv",
    "write_text",
    "write_bytes",
    "save",
    "savez",
    "savez_compressed",
}
_EXEMPT = {
    ("xpcs_viewer.py", "load_default_setting"): (
        "creates the application's own default_setting.json in its home folder; "
        "not a user export"
    ),
}


def _writes_in(function):
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in _WRITE_ATTRS:
            yield node, func.attr
        elif (isinstance(func, ast.Name) and func.id == "open") or (
            isinstance(func, ast.Attribute) and func.attr == "open"
        ):
            mode = None
            args = node.args[1:2] if isinstance(func, ast.Name) else node.args[:1]
            if args and isinstance(args[0], ast.Constant):
                mode = args[0].value
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = kw.value.value
            if isinstance(mode, str) and any(c in mode for c in "wax+"):
                yield node, "open(w)"
        elif isinstance(func, ast.Attribute) and func.attr == "File":
            mode = None
            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                mode = node.args[1].value
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = kw.value.value
            if isinstance(mode, str) and mode != "r":
                yield node, "h5py.File(w)"


def _claims(function):
    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in ("claim", "_claim")
        for n in ast.walk(function)
    )


@pytest.mark.parametrize(
    "relative", ["xpcs_viewer.py", "gui/bayesian_diagnosis.py", "fitting/viz.py"]
)
def test_r5_01_every_file_write_is_claimed_through_a_guard(relative):
    tree = ast.parse((PKG / relative).read_text())
    name = relative.rsplit("/", 1)[-1]
    seen_writes = 0
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # only the innermost function owns a write (skip enclosing ones)
        nested = {
            id(n)
            for inner in ast.walk(function)
            if inner is not function
            and isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef))
            for n in ast.walk(inner)
        }
        for node, kind in _writes_in(function):
            if id(node) in nested:
                continue
            seen_writes += 1
            if (name, function.name) in _EXEMPT:
                continue
            assert _claims(function), (
                f"{relative}:{node.lineno} {kind} in {function.name}() "
                "is not claimed through an ExportGuard"
            )
    assert relative != "fitting/viz.py" or seen_writes >= 2


def test_r5_01_regroup_export_claims_its_destination(tmp_path):
    src = (PKG / "xpcs_viewer.py").read_text()
    tree = ast.parse(src)
    start = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_start_g2_regroup_export"
    )
    assert _claims(start)
