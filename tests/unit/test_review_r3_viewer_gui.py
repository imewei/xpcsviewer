"""Regression tests for review round 3 in the viewer GUI layer.

Invariant 1 (R3-14, R3-29, R3-30): a Bayesian posterior is only ever shown,
diagnosed or plotted together with the exact observations and physical bin it
was fitted to.  Invariant 2 (R3-16, R3-17): the newest plot request for a tab
is always the one that ends up drawn, and retired batch coordinators are
released.  R3-12: an explicitly supplied startup path wins everywhere.
"""

from __future__ import annotations

import sys
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

pytestmark = pytest.mark.unit

# four delays: a single-exponential Bayesian fit needs >= 3 observations (R6-13)
# and a test masks two of them in one bin
T_EL = np.array([1e-3, 1e-2, 1e-1, 1.0])


class _Stub(SimpleNamespace):
    """SimpleNamespace that supports weak references (Qt connects bound methods)."""


def _qobject_stub(**attrs):
    """A QObject carrying attributes: usable as the coordinator's parent."""
    from qtpy.QtCore import QObject

    obj = QObject()
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


def _cls():
    from xpcsviewer.xpcs_viewer import XpcsViewer

    return XpcsViewer


def _bind(stub, *names):
    cls = _cls()
    for name in names:
        setattr(stub, name, MethodType(getattr(cls, name), stub))
    return stub


# ---------------------------------------------------------------------------
# A file with three bins: Q=.01 @ 0 deg, Q=.02 @ 0 deg, Q=.02 @ 90 deg
# ---------------------------------------------------------------------------


class FakeQmap:
    def __init__(self, q, phi):
        self.q = np.asarray(q, float)
        self.phi = np.asarray(phi, float)

    def get_qbin_in_qrange(self, qrange, zero_based=True):
        sel = np.flatnonzero((self.q >= qrange[0]) & (self.q <= qrange[1]))
        return sel, self.q[sel]

    def get_qbin_label(self, n):
        return f"q={self.q[n - 1]:.4f}, phi={self.phi[n - 1]:g}"


class FakeXf:
    atype = ["Multitau"]

    def __init__(self, fname="A.h5", q=(0.01, 0.02, 0.02), phi=(0, 0, 90), g2=None):
        self.fname = fname
        self.qmap = FakeQmap(q, phi)
        self.t_el = T_EL
        self.g2 = (
            np.array(
                [[1.0, 1.4, 1.8], [0.9, 1.1, 1.6], [0.8, 1.05, 1.4], [0.7, 1.0, 1.3]]
            )
            if g2 is None
            else g2
        )
        self.g2_err = np.full_like(self.g2, 0.1)

    def get_g2_data(self, qrange=None, trange=None):
        sel, qv = self.qmap.get_qbin_in_qrange(qrange)
        labels = [self.qmap.get_qbin_label(int(i) + 1) for i in sel]
        return qv, self.t_el, self.g2[:, sel], self.g2_err[:, sel], labels


def _viewer(xf, *, q_range=(0.0, 1.0), q_idx=1):
    """Stand-in viewer: real Bayesian methods bound to a simple namespace."""
    st = _Stub(
        vk=SimpleNamespace(get_xf_list=lambda rows=None, filter_atype=None: [xf]),
        statusbar=MagicMock(),
        sb_g2_bayesian_qidx=MagicMock(),
        btn_g2_bayesian=MagicMock(),
        btn_g2_diagnosis=MagicMock(),
        sb_g2_bayesian_warmup=MagicMock(value=lambda: 10),
        sb_g2_bayesian_samples=MagicMock(value=lambda: 10),
        sb_g2_bayesian_chains=MagicMock(value=lambda: 1),
        thread_pool=MagicMock(),
        get_selected_rows=lambda: [0],
        _guard_no_data=lambda _t: True,
        check_g2_fitting_number=lambda: (None, None, "single"),
        _g2_bayesian_store={},
        _g2_bayesian_launch=None,
        _g2_bayesian_launch_revision=None,  # "A.h5" does not exist: no revision
        _g2_bayesian_worker=None,
        _g2_bayesian_worker_active=False,
        _g2_diagnosis_window=None,
        _update_g2_diagnosis=MagicMock(),
        _evict_stale_bayesian=lambda: 0,
        toast_manager=MagicMock(),
        q_range=list(q_range),
    )
    st.sb_g2_bayesian_qidx.value.return_value = q_idx
    st.check_g2_number = lambda: (st.q_range[0], st.q_range[1], 1e-4, 1.0)
    return _bind(
        st,
        "_g2_bin_table",
        "_g2_first_axis",
        "_g2_current_bundle",
        "_extract_g2_for_bayesian",
        "_g2_resolved_bins",
        "_fit_g2_bayesian",
        "_is_current_g2_worker",
        "_on_g2_bayesian_finished",
        "_on_g2_bayesian_error",
        "_on_g2_bayesian_cancelled",
        "_on_g2_qbin_changed",
        "_show_g2_diagnosis",
        "_announce_bayesian_fit",
    )


def _fit_current_bin(st, result="POSTERIOR"):
    """Launch a single-Q fit for the selected bin and let it complete."""
    st._fit_g2_bayesian()
    worker = st._g2_bayesian_worker
    bundle = st._g2_bayesian_launch
    fit_result = SimpleNamespace(
        diagnostics=SimpleNamespace(converged=True), tag=result
    )
    st._on_g2_bayesian_finished(
        {
            "fit_result": fit_result,
            "q_index": worker.q_index,
            "q_value": worker.q_value,
            "launch": worker.launch,
        }
    )
    assert bundle is not None
    return fit_result


# ---------------------------------------------------------------------------
# R3-14: a stored fit is addressed by its physical bin, never a relative index
# ---------------------------------------------------------------------------


def test_r3_14_narrowed_q_range_does_not_alias_other_azimuth():
    xf = FakeXf()
    st = _viewer(xf, q_idx=1)  # relative 1 == absolute bin 1 (Q=.02, 0 deg)
    fit_result = _fit_current_bin(st)

    # Narrow the Q range to exclude Q=.01: relative 1 now addresses bin 2 (90 deg)
    st.q_range[:] = [0.015, 0.03]
    st._show_g2_diagnosis()
    st._update_g2_diagnosis.assert_not_called()
    assert "run Fit first" in st.statusbar.showMessage.call_args[0][0]

    # The fitted bin is now relative 0 and diagnoses with ITS observations
    st.sb_g2_bayesian_qidx.value.return_value = 0
    st._show_g2_diagnosis()
    bundle = st._update_g2_diagnosis.call_args[0][0]
    assert bundle.result is fit_result
    assert bundle.bin_id == 1
    np.testing.assert_array_equal(bundle.y, [1.4, 1.1, 1.05, 1.0])


def test_r3_14_store_is_keyed_by_file_and_absolute_bin():
    st = _viewer(FakeXf(), q_idx=2)
    _fit_current_bin(st)
    assert list(st._g2_bayesian_store) == [("A.h5", None, 2)]
    done = st._g2_bayesian_store[("A.h5", None, 2)]
    assert "phi=90" in done.label and done.model_name == "single"


def test_r3_14_diagnosis_shows_stored_observations_not_current_data():
    xf = FakeXf()
    st = _viewer(xf, q_idx=1)
    _fit_current_bin(st)
    xf.g2 = xf.g2 + 5.0  # data of the same file/bin changed after the fit
    st._show_g2_diagnosis()
    bundle = st._update_g2_diagnosis.call_args[0][0]
    np.testing.assert_array_equal(bundle.y, [1.4, 1.1, 1.05, 1.0])
    np.testing.assert_array_equal(bundle.x, T_EL)


def test_r3_14_bundle_observations_are_private_read_only_copies():
    xf = FakeXf()
    st = _viewer(xf, q_idx=1)
    _fit_current_bin(st)
    bundle = st._g2_bayesian_store[("A.h5", None, 1)]
    xf.g2[:, 1] = 0.0
    np.testing.assert_array_equal(bundle.y, [1.4, 1.1, 1.05, 1.0])
    assert not bundle.y.flags.writeable


def test_r3_14_other_file_with_same_bins_has_no_stored_fit():
    st = _viewer(FakeXf("A.h5"), q_idx=1)
    _fit_current_bin(st)
    st.vk = SimpleNamespace(
        get_xf_list=lambda rows=None, filter_atype=None: [FakeXf("B.h5")]
    )
    assert st._g2_current_bundle() is None


def test_r3_14_unidentifiable_bins_refuse_the_fit():
    xf = FakeXf()
    st = _viewer(xf, q_idx=1)
    # g2 has one column fewer than the bin table says: identity is ambiguous
    xf.get_g2_data = lambda qrange=None, trange=None: (
        np.array([0.01, 0.02]),
        T_EL,
        xf.g2[:, :2],
        xf.g2_err[:, :2],
        ["a", "b"],
    )
    assert st._extract_g2_for_bayesian() is None
    assert "identify" in st.statusbar.showMessage.call_args[0][0]


def test_r3_14_open_diagnosis_follows_physical_bin_without_reextracting():
    xf = FakeXf()
    st = _viewer(xf, q_idx=1)
    _fit_current_bin(st)
    win = MagicMock()
    win.isVisible.return_value = True
    st._g2_diagnosis_window = win
    st._highlight_g2_subplot = lambda i: False

    st._on_g2_qbin_changed(1)  # fitted bin: window shows stored bundle
    assert st._update_g2_diagnosis.call_args[0][0].bin_id == 1
    st._update_g2_diagnosis.reset_mock()
    st.sb_g2_bayesian_qidx.value.return_value = 2
    st._on_g2_qbin_changed(2)  # another bin: no fit, nothing from bin 1
    st._update_g2_diagnosis.assert_not_called()
    # round 5: the window is cleared (not just retitled)
    assert "no fit yet" in win.clear.call_args[0][0]


# ---------------------------------------------------------------------------
# R3-14 / R3-30: superseded or reset single-Q fits never commit
# ---------------------------------------------------------------------------


def test_r3_14_result_of_superseded_launch_is_ignored():
    st = _viewer(FakeXf(), q_idx=1)
    st._fit_g2_bayesian()
    old = st._g2_bayesian_worker
    st._g2_bayesian_launch = None  # workspace reset dropped the launch
    st._g2_bayesian_worker = None
    st._on_g2_bayesian_finished(
        {"fit_result": object(), "q_index": 1, "q_value": 0.02, "launch": old.launch}
    )
    assert st._g2_bayesian_store == {}


def test_r3_14_error_of_superseded_worker_keeps_current_fit_running():
    st = _viewer(FakeXf(), q_idx=1)
    st._fit_g2_bayesian()
    current = st._g2_bayesian_worker
    st._on_g2_bayesian_error("bayesian_g2_1_0", "boom", "", False)  # an old worker
    assert st._g2_bayesian_worker_active and st._g2_bayesian_worker is current
    st._on_g2_bayesian_error(current.worker_id, "boom", "", False)
    assert not st._g2_bayesian_worker_active


def test_r3_14_status_bar_names_the_physical_bin_of_a_narrowed_range():
    xf = FakeXf()
    st = _viewer(xf, q_range=(0.015, 1.0), q_idx=0)  # position 0 is bin 1
    st._highlight_g2_subplot = lambda i: False
    st._on_g2_qbin_changed(0)
    message = st.statusbar.showMessage.call_args[0][0]
    assert "q=0.0200, phi=0" in message  # bin 1, not bin 0 (Q=.01)


def test_r3_14_reset_keeps_a_start_up_disabled_fit_button_disabled():
    st = _Stub(
        _g2_batch_coordinator=None,
        _g2_batch_launch=None,
        _g2_bayesian_worker=None,
        _diff_bayesian_worker=None,
        _g2_bayesian_worker_active=False,
        _diff_bayesian_worker_active=False,
        btn_g2_bayesian=MagicMock(),
        btn_diff_bayesian=MagicMock(),
        btn_g2_diagnosis=MagicMock(),
        btn_diff_diagnosis=MagicMock(),
        btn_g2_plot_all_q=MagicMock(),
        sb_g2_bayesian_qidx=MagicMock(value=lambda: 0),
    )
    _bind(st, "_reset_bayesian_state", "_close_bayesian_windows")
    st._reset_bayesian_state()  # no fit was running: NumPyro-missing state stays
    st.btn_g2_bayesian.setEnabled.assert_not_called()
    st.btn_diff_bayesian.setEnabled.assert_not_called()
    st._g2_bayesian_worker_active = True  # a running fit gets its button back
    st._reset_bayesian_state()
    st.btn_g2_bayesian.setEnabled.assert_called_once_with(True)


# ---------------------------------------------------------------------------
# The bin table must agree with what the real XpcsFile/QMap code selects
# ---------------------------------------------------------------------------


def _real_qmap():
    """A real QMap: 3 q x 2 phi grid, occupied flat positions 1, 2 and 5."""
    from xpcsviewer.fileIO.qmap_utils import QMap

    qmap = QMap.__new__(QMap)
    qmap.fname = "synthetic.hdf"
    qmap.map_names = ["q", "phi"]
    qmap.dqlist = np.array([0.1, 0.2, 0.3])
    qmap.dynamic_num_pts = np.array([3, 2])
    qmap.dynamic_index_mapping = np.array([1, 2, 5])
    qmap.qbin_labels = [f"label-{i}" for i in range(6)]
    qmap._qbin_cache = {}
    return qmap


def _real_file(kind):
    from xpcsviewer.xpcs_file import XpcsFile

    g2 = np.arange(6.0).reshape(2, 3) + 1.0
    base = {"qmap": _real_qmap(), "label": "synthetic", "fname": "S.h5"}
    if kind == "multitau":
        stub = SimpleNamespace(
            **base, atype=["Multitau"], g2=g2, g2_err=g2 * 0.1, t_el=T_EL
        )
    else:  # twotime: only dynamic bins 1 and 3 (ids 0 and 2) were processed
        stub = SimpleNamespace(
            **base,
            atype=["Twotime"],
            c2_g2=g2[:, :2],
            c2_g2_err=g2[:, :2] * 0.1,
            c2_processed_bins=np.array([1, 3]),
            c2_delay=T_EL,
        )
    stub.get_g2_data = lambda qrange=None, trange=None: XpcsFile.get_g2_data(
        stub, qrange, trange
    )
    return stub


@pytest.mark.parametrize("kind", ["multitau", "twotime"])
@pytest.mark.parametrize("q_range", [None, (0.0, 1.0), (0.15, 0.35), (0.25, 0.35)])
def test_resolve_g2_bins_matches_real_get_g2_data(kind, q_range):
    from xpcsviewer.fitting.bayesian_assembly import resolve_g2_bins

    xf = _real_file(kind)
    q, _, g2, _, labels = xf.get_g2_data(q_range, None)
    bins = resolve_g2_bins(xf, q_range)
    assert len(bins) == g2.shape[1] == len(q)
    np.testing.assert_allclose([b[1] for b in bins], q)
    assert [b[2] for b in bins] == list(labels)
    # absolute ids do not depend on the Q range used to select them
    full = {b[2]: b[0] for b in resolve_g2_bins(xf, None)}
    assert all(full[b[2]] == b[0] for b in bins)


# ---------------------------------------------------------------------------
# R3-29: diffusion posterior and observations are committed together
# ---------------------------------------------------------------------------


def _diffusion_viewer():
    from xpcsviewer.fitting.bayesian_assembly import BayesianBundle

    cur = {"tau": np.array([1.0, 2.0])}
    q = np.array([0.01, 0.02])
    st = _Stub(
        _diff_bayesian_worker_active=False,
        _diff_bayesian_worker=None,
        _diff_bayesian_launch=None,
        _diff_bayesian_launch_revision=None,  # "A.h5" does not exist
        _diff_bayesian_store={},
        statusbar=MagicMock(),
        sb_g2_bayesian_warmup=MagicMock(value=lambda: 10),
        sb_g2_bayesian_samples=MagicMock(value=lambda: 10),
        sb_g2_bayesian_chains=MagicMock(value=lambda: 1),
        tauq_amin=MagicMock(text=lambda: "bad"),
        btn_diff_bayesian=MagicMock(),
        btn_diff_diagnosis=MagicMock(),
        thread_pool=MagicMock(),
        vk=SimpleNamespace(
            get_xf_list=lambda rows=None: [SimpleNamespace(fname="A.h5")]
        ),
        get_selected_rows=lambda: [0],
        _update_diff_diagnosis=MagicMock(),
        cur=cur,
    )
    st._extract_diffusion_for_bayesian = lambda: BayesianBundle.create(
        "A.h5", "tau-q", "tau(q)", 0.0, "power_law", q, cur["tau"], np.full(2, 0.1)
    )
    return _bind(
        st,
        "_fit_diff_bayesian",
        "_on_diff_bayesian_finished",
        "_on_diff_bayesian_error",
        "_on_diff_bayesian_cancelled",
        "_is_current_diff_worker",
        "_show_diff_diagnosis",
    )


def _diff_result(worker, tag):
    import pandas as pd

    fit = SimpleNamespace(
        diagnostics=SimpleNamespace(converged=True), summary=pd.DataFrame(), tag=tag
    )
    return fit, {"fit_result": fit, "launch": worker.launch}


def test_r3_29_second_fit_does_not_replace_displayed_observations():
    st = _diffusion_viewer()
    st._fit_diff_bayesian()
    first, res = _diff_result(st._diff_bayesian_worker, "first")
    st._on_diff_bayesian_finished(res)

    st.cur["tau"] = np.array([3.0, 4.0])  # changed data; second fit starts
    st._fit_diff_bayesian()
    assert st._diff_bayesian_worker_active

    st._show_diff_diagnosis()  # opened while the second fit is running
    bundle = st._update_diff_diagnosis.call_args[0][0]
    assert bundle.result is first
    np.testing.assert_array_equal(bundle.y, [1.0, 2.0])


def test_r3_29_completion_of_superseded_fit_is_ignored_and_new_one_commits():
    st = _diffusion_viewer()
    st._fit_diff_bayesian()
    old_worker = st._diff_bayesian_worker
    first, res_old = _diff_result(old_worker, "first")
    st._diff_bayesian_launch = None  # reset: the first fit is superseded
    st._diff_bayesian_worker = None
    st._diff_bayesian_worker_active = False

    st.cur["tau"] = np.array([3.0, 4.0])
    st._fit_diff_bayesian()
    st._on_diff_bayesian_finished(res_old)  # late callback of the old fit
    assert st._diff_bayesian_store == {}
    assert st._diff_bayesian_worker_active  # new fit unaffected

    second, res_new = _diff_result(st._diff_bayesian_worker, "second")
    st._on_diff_bayesian_finished(res_new)
    bundle = st._diff_bayesian_store[("A.h5", None)]
    assert bundle.result is second
    np.testing.assert_array_equal(bundle.y, [3.0, 4.0])


def test_r3_29_stale_cancel_does_not_release_running_fit():
    st = _diffusion_viewer()
    st._fit_diff_bayesian()
    st._on_diff_bayesian_cancelled("bayesian_diffusion_-1_0", "old")
    assert st._diff_bayesian_worker_active
    st._on_diff_bayesian_cancelled(st._diff_bayesian_worker.worker_id, "now")
    assert not st._diff_bayesian_worker_active


def test_r3_29_diagnosis_without_fit_for_selected_file_says_so():
    st = _diffusion_viewer()
    st._show_diff_diagnosis()
    st._update_diff_diagnosis.assert_not_called()
    assert "run Fit first" in st.statusbar.showMessage.call_args[0][0]


# ---------------------------------------------------------------------------
# R3-30: each file's all-Q summary carries its own bins, ranges and observations
# ---------------------------------------------------------------------------


def _fit_result(tau=0.01):
    values = {"tau": tau, "baseline": 1.0, "contrast": 0.2}
    return SimpleNamespace(
        get_mean=lambda n: values[n],
        get_std=lambda n: 0.01,
        diagnostics=SimpleNamespace(converged=True),
    )


def _batch_viewer(xf, q_range):
    st = _qobject_stub(
        vk=SimpleNamespace(get_xf_list=lambda rows=None, filter_atype=None: [xf]),
        statusbar=MagicMock(),
        sb_g2_bayesian_qidx=MagicMock(value=lambda: 0),
        sb_g2_bayesian_workers=MagicMock(value=lambda: 1),
        sb_g2_bayesian_warmup=MagicMock(value=lambda: 10),
        sb_g2_bayesian_samples=MagicMock(value=lambda: 10),
        sb_g2_bayesian_chains=MagicMock(value=lambda: 1),
        btn_g2_bayesian=MagicMock(),
        btn_g2_bayesian_all=MagicMock(),
        btn_g2_diagnosis=MagicMock(),
        btn_g2_plot_all_q=MagicMock(),
        thread_pool=MagicMock(),
        progress_manager=MagicMock(),
        get_selected_rows=lambda: [0],
        _guard_no_data=lambda _t: True,
        check_g2_fitting_number=lambda: (None, None, "single"),
        check_g2_number=lambda: (q_range[0], q_range[1], 1e-4, 1.0),
        _g2_bayesian_store={},
        _g2_batch_coordinator=None,
        _g2_batch_launch=None,
        init_diffusion=MagicMock(),
        plot_g2_fitting=MagicMock(),
        _show_bayesian_plot_dialog=MagicMock(),
        _on_g2_batch_progress=lambda *a: None,
        _on_g2_batch_single_q_error=lambda *a: None,
    )
    return _bind(
        st,
        "_g2_bin_table",
        "_g2_current_bundle",
        "_g2_resolved_bins",
        "_extract_g2_all_for_bayesian",
        "_fit_g2_bayesian_all",
        "_on_g2_batch_single_q_done",
        "_on_g2_batch_finished",
        "_cancel_g2_bayesian_batch",
        "_plot_g2_bayesian_all",
    )


def _run_batch(st, xf, qapp):
    """Launch the all-Q batch and let every worker 'finish' successfully."""
    st._fit_g2_bayesian_all()
    coord = st._g2_batch_coordinator
    results = {}
    for spec in coord._q_specs:
        res = {
            "fit_result": _fit_result(0.01 * (spec["q_idx"] + 1)),
            "launch": spec["launch"],
            "elapsed_s": 0.1,
        }
        results[spec["q_idx"]] = res
        st._on_g2_batch_single_q_done(spec["q_idx"], res)
    st._on_g2_batch_finished(results)
    return coord


def test_r3_30_plot_uses_each_files_own_summary_not_last_global_range(
    qapp, monkeypatch
):
    from xpcsviewer.module import g2mod

    xf_a = FakeXf("A.h5")  # bins Q=.01, .02, .02
    xf_b = FakeXf("B.h5", q=(0.01, 0.02, 0.03), phi=(0, 0, 0))
    st_a = _batch_viewer(xf_a, (0.0, 1.0))
    _run_batch(st_a, xf_a, qapp)
    st_b = _batch_viewer(xf_b, (0.015, 0.035))  # later fit of B: other range
    _run_batch(st_b, xf_b, qapp)

    summary = xf_a.bayesian_fit_summary
    assert summary["fname"] == "A.h5"
    np.testing.assert_array_equal(summary["bin_ids"], [0, 1, 2])
    assert summary["fit_q_range"] == (0.0, 1.0)
    np.testing.assert_array_equal(summary["obs_g2"], xf_a.g2)
    assert xf_b.bayesian_fit_summary["fit_q_range"] == (0.015, 0.035)
    np.testing.assert_array_equal(xf_b.bayesian_fit_summary["bin_ids"], [1, 2])

    # Return to plotting A while B's range is the viewer's/last fit's range:
    # nothing may be re-extracted, the figure shows A's stored observations.
    monkeypatch.setattr(
        g2mod, "get_data", lambda *a, **k: pytest.fail("plot re-extracted data")
    )
    st_a.check_g2_number = lambda: (0.015, 0.035, 1e-4, 1.0)
    st_a._plot_g2_bayesian_all()
    fig = st_a._show_bayesian_plot_dialog.call_args[0][0]
    try:
        offsets = fig.axes[0].collections[0].get_offsets()
        np.testing.assert_allclose(sorted(offsets[:, 1]), sorted(xf_a.g2.ravel()))
        np.testing.assert_allclose(sorted(set(offsets[:, 0])), sorted(T_EL))
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_r3_30_batch_results_commit_to_store_by_absolute_bin(qapp):
    xf = FakeXf()
    st = _batch_viewer(xf, (0.015, 0.03))  # selects bins 1 and 2 only
    _run_batch(st, xf, qapp)
    assert sorted(st._g2_bayesian_store) == [("A.h5", None, 1), ("A.h5", None, 2)]
    assert "phi=90" in st._g2_bayesian_store[("A.h5", None, 2)].label
    np.testing.assert_array_equal(
        st._g2_bayesian_store[("A.h5", None, 1)].y, [1.4, 1.1, 1.05, 1.0]
    )


def test_r3_30_summary_without_stored_observations_draws_fit_lines_only(qapp):
    from xpcsviewer.fitting.bayesian_assembly import assemble_fit_summary
    from xpcsviewer.fitting.models import single_exp_func
    from xpcsviewer.fitting.viz import plot_bayesian_all_q

    summary = assemble_fit_summary(
        {0: _fit_result()}, np.array([0.01]), T_EL, "single", single_exp_func
    )
    fig = plot_bayesian_all_q(summary)
    try:
        assert fig is not None
        assert len(fig.axes[0].collections) == 1  # line collection, no scatter
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_r3_30_assembly_rejects_mismatched_bin_ids():
    from xpcsviewer.fitting.bayesian_assembly import assemble_fit_summary

    with pytest.raises(ValueError, match="bin_ids"):
        assemble_fit_summary(
            {}, np.array([0.01, 0.02]), T_EL, "single", None, bin_ids=[0]
        )


def test_r3_30_stored_observations_are_exactly_the_fitted_points(qapp):
    xf = FakeXf()
    xf.g2_err[1, 0] = 0.0  # excluded from the fit: yerr must be > 0
    xf.g2[0, 2] = np.nan
    st = _batch_viewer(xf, (0.0, 1.0))
    _run_batch(st, xf, qapp)
    obs = xf.bayesian_fit_summary["obs_g2"]
    assert np.isnan(obs[1, 0]) and np.isnan(obs[0, 2])
    assert obs[0, 0] == 1.0 and obs[1, 1] == 1.1
    # the masked observations still draw: a figure with finite limits
    st._plot_g2_bayesian_all()
    fig = st._show_bayesian_plot_dialog.call_args[0][0]
    try:
        assert np.all(np.isfinite(fig.axes[0].get_xlim()))
        assert np.all(np.isfinite(fig.axes[0].get_ylim()))
        drawn = fig.axes[0].collections[0].get_offsets()
        assert np.isfinite(drawn[:, 1]).sum() == 10  # 12 points, 2 not fitted
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_r3_30_batch_finished_ignored_after_reset_and_consumed_once(qapp):
    xf = FakeXf()
    st = _batch_viewer(xf, (0.0, 1.0))
    _run_batch(st, xf, qapp)
    first = xf.bayesian_fit_summary
    st.progress_manager.reset_mock()
    st._on_g2_batch_finished({0: None})  # a repeated/late signal
    assert xf.bayesian_fit_summary is first
    st.progress_manager.complete_operation.assert_not_called()


# ---------------------------------------------------------------------------
# R3-17: finished and cancelled coordinators are retired, not kept as children
# ---------------------------------------------------------------------------


def _coordinator(parent, n=2):
    from xpcsviewer.threading.batch_bayesian_coordinator import (
        BatchBayesianCoordinator,
    )

    specs = [
        {
            "q_idx": i,
            "x": np.linspace(0.001, 1, 5),
            "y": np.ones(5),
            "yerr": np.full(5, 0.1),
            "q_value": 0.01 * (i + 1),
            "fit_func": MagicMock(),
        }
        for i in range(n)
    ]
    return BatchBayesianCoordinator(specs, MagicMock(), max_concurrent=n, parent=parent)


def _flush_deferred_deletes():
    from qtpy.QtCore import QCoreApplication, QEvent

    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


def test_r3_17_cancelled_coordinator_is_deleted_once_workers_reported(qapp):
    from qtpy.QtCore import QObject

    parent = QObject()
    coord = _coordinator(parent)
    coord.start()
    coord.cancel()
    _flush_deferred_deletes()
    # workers may still run NUTS: the coordinator must outlive them
    assert coord in parent.children()
    coord._on_worker_cancelled(0)
    assert coord in parent.children()
    coord._on_worker_finished(1, {"fit_result": object()})  # late result, muted
    _flush_deferred_deletes()
    assert parent.children() == []


def test_r3_17_finished_coordinator_releases_results_and_is_deleted(qapp):
    from qtpy.QtCore import QObject

    parent = QObject()
    coord = _coordinator(parent, n=1)
    got = []
    coord.all_finished.connect(lambda r: (got.append(r), coord.dispose()))
    coord.start()
    coord._on_worker_finished(0, {"fit_result": object()})
    assert len(got) == 1 and 0 in got[0]
    assert coord._results == {} and coord._q_specs == []
    _flush_deferred_deletes()
    assert parent.children() == []


def test_r3_17_cancel_before_any_worker_started_deletes_immediately(qapp):
    from qtpy.QtCore import QObject

    parent = QObject()
    coord = _coordinator(parent)
    coord.cancel()
    _flush_deferred_deletes()
    assert parent.children() == []


def test_r3_17_viewer_disposes_finished_and_cancelled_coordinators():
    coord = MagicMock()
    st = SimpleNamespace(
        _g2_batch_coordinator=coord,
        _g2_batch_launch={"x": 1},
        progress_manager=MagicMock(),
        btn_g2_bayesian_all=MagicMock(),
        btn_g2_bayesian=MagicMock(),
        sb_g2_bayesian_workers=MagicMock(),
        sb_g2_bayesian_warmup=MagicMock(),
        sb_g2_bayesian_samples=MagicMock(),
        sb_g2_bayesian_chains=MagicMock(),
        btn_g2_plot_all_q=MagicMock(),
        sb_g2_bayesian_qidx=MagicMock(value=lambda: 0),
    )
    _cls()._cancel_g2_bayesian_batch(st)
    coord.cancel.assert_called_once()  # cancel() retires itself
    assert st._g2_batch_coordinator is None and st._g2_batch_launch is None


def test_r3_17_viewer_batch_finished_disposes_coordinator(qapp):
    xf = FakeXf()
    st = _batch_viewer(xf, (0.0, 1.0))
    coord = _run_batch(st, xf, qapp)
    assert st._g2_batch_coordinator is None
    assert coord._retired and coord._results == {}
    assert st._g2_batch_launch is None


# ---------------------------------------------------------------------------
# R3-16: the newest plot request is drawn however the older operation ends
# ---------------------------------------------------------------------------


def _async_viewer(*, request, current, pending=False):
    cls = _cls()
    st = SimpleNamespace(
        active_plot_operations={"plot_g2_x": "g2"},
        _plot_op_requests={"plot_g2_x": request},
        _plot_pending={"g2"} if pending else set(),
        _retiring_plots=False,
        plot_kwargs_record={"g2": dict(request)},
        _plot_request=lambda kwargs: kwargs,  # no revision part in this stub
        vk=SimpleNamespace(timestamp="t1"),
        plot_g2=lambda dryrun=False: dict(current),
        update_plot_async=MagicMock(),
        progress_manager=MagicMock(),
        toast_manager=MagicMock(),
        async_vk=None,
    )
    return _bind(
        st,
        "_plot_request_is_stale",
        "_settle_plot_operation",
        "on_async_operation_error",
        "on_async_operation_cancelled",
        "cancel_async_operation",
    ), cls


def test_r3_16_error_of_older_operation_draws_suppressed_newer_request():
    st, _ = _async_viewer(
        request={"rows": [0], "target_timestamp": "t1"},
        current={"rows": [1]},  # B selected while A ran
        pending=True,
    )
    st.on_async_operation_error("plot_g2_x", "worker blew up", "tb")
    st.update_plot_async.assert_called_once_with("g2")
    assert st.active_plot_operations == {} and st._plot_op_requests == {}


def test_r3_16_error_with_stale_request_reschedules_even_without_pending_mark():
    st, _ = _async_viewer(
        request={"rows": [0], "target_timestamp": "t1"}, current={"rows": [1]}
    )
    st.on_async_operation_error("plot_g2_x", "boom", "tb")
    st.update_plot_async.assert_called_once_with("g2")


def test_r3_16_failed_request_can_be_issued_again():
    request = {"rows": [0], "target_timestamp": "t1"}
    st, _ = _async_viewer(request=request, current={"rows": [0]})
    st.on_async_operation_error("plot_g2_x", "boom", "tb")
    st.update_plot_async.assert_not_called()  # nothing newer
    # the failed draw is not remembered as "already plotted"
    assert st.plot_kwargs_record["g2"] == {}


def test_r3_16_cancel_draws_suppressed_newer_request():
    st, _ = _async_viewer(
        request={"rows": [0], "target_timestamp": "t1"},
        current={"rows": [1]},
        pending=True,
    )
    st.cancel_async_operation("plot_g2_x")
    st.update_plot_async.assert_called_once_with("g2")


def test_r3_16_deliberate_retirement_does_not_replay_a_plot():
    st, _ = _async_viewer(
        request={"rows": [0], "target_timestamp": "t1"},
        current={"rows": [1]},
        pending=True,
    )
    st._retiring_plots = True
    st.cancel_async_operation("plot_g2_x")
    st.update_plot_async.assert_not_called()


def test_r3_16_worker_cancelled_without_viewer_is_settled():
    st, _ = _async_viewer(
        request={"rows": [0], "target_timestamp": "t1"},
        current={"rows": [1]},
        pending=True,
    )
    st.on_async_operation_cancelled("plot_g2_x")
    assert st.active_plot_operations == {}
    st.update_plot_async.assert_called_once_with("g2")
    st.on_async_operation_cancelled("unknown")  # ignored


def test_r3_16_request_suppressed_while_running_is_remembered():
    cls = _cls()
    st = SimpleNamespace(
        active_plot_operations={"plot_g2_x": "g2"}, _plot_pending=set()
    )
    cls.update_plot_async(st, "g2")
    assert st._plot_pending == {"g2"}


# ---------------------------------------------------------------------------
# R3-12: an explicitly supplied startup path wins in every entry point
# ---------------------------------------------------------------------------


def _run_cli_gui(monkeypatch, argv):
    from xpcsviewer import cli_main

    seen = {}
    monkeypatch.setattr(sys, "argv", ["xpcsviewer-gui", *argv])
    monkeypatch.setattr(
        cli_main,
        "_start_gui",
        lambda path, style, explicit=False: (
            seen.update(path=path, explicit=explicit) or 0
        ),
    )
    for name in ("initialize_logging", "setup_exception_logging", "safe_shutdown"):
        monkeypatch.setattr(cli_main, name, lambda *a, **k: None, raising=False)
    monkeypatch.setattr(cli_main.signal, "signal", lambda *a, **k: None)
    monkeypatch.setattr(cli_main.atexit, "register", lambda *a, **k: None)
    with pytest.raises(SystemExit):
        cli_main.main_gui()
    return seen


def test_r3_12_cli_dot_slash_typed_by_user_is_explicit(monkeypatch):
    seen = _run_cli_gui(monkeypatch, ["--path", "./"])
    assert seen == {"path": "./", "explicit": True}


def test_r3_12_cli_positional_path_is_explicit(monkeypatch):
    assert _run_cli_gui(monkeypatch, ["/data/B"]) == {
        "path": "/data/B",
        "explicit": True,
    }


def test_r3_12_cli_plain_launch_opens_cwd_implicitly(monkeypatch):
    assert _run_cli_gui(monkeypatch, []) == {"path": "./", "explicit": False}


@pytest.fixture
def _offscreen_viewer(qtbot, monkeypatch, tmp_path):
    """Build real viewers whose saved session points at workspace A."""
    from xpcsviewer.gui.state.recent_paths import RecentPathsManager
    from xpcsviewer.gui.state.session_manager import (
        AnalysisParameters,
        SessionManager,
    )

    xv = sys.modules["xpcsviewer.xpcs_viewer"]
    a = tmp_path / "A"
    a.mkdir()
    session = SimpleNamespace(
        data_path=str(a),
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
    created = []

    def make(**kw):
        window = xv.XpcsViewer(**kw)
        qtbot.addWidget(window)
        created.append(window)
        return window

    return a, make


def test_r3_12_public_constructor_path_wins_over_saved_workspace(
    _offscreen_viewer, tmp_path
):
    a, make = _offscreen_viewer
    b = tmp_path / "B"
    b.mkdir()
    window = make(path=str(b))  # no path_explicit flag: a supplied path is explicit
    assert window._path_explicit is True
    assert window.work_dir.text() == str(b)


def test_r3_12_automatic_default_path_still_restores_saved_workspace(
    _offscreen_viewer, tmp_path
):
    a, make = _offscreen_viewer
    b = tmp_path / "B"
    b.mkdir()
    window = make(path=str(b), path_explicit=False)  # caller marks it automatic
    assert window._path_explicit is False
    assert window.work_dir.text() == str(a)
