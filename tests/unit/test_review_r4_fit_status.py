"""Round 4 fit-status regressions (R4-01, R4-02, R4-15).

Invariant: a fit that failed, did not converge, or is unidentifiable is never
presented, exported, or fed into a derived fit as a measurement.
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsviewer.fitting.legacy import (
    fit_success_mask,
    fit_with_fixed,
    fit_with_fixed_parallel,
    fit_with_fixed_sequential,
)
from xpcsviewer.xpcs_file import XpcsFile
from xpcsviewer.xpcs_file.fitting import double_exp_all, single_exp_all

from .test_review_r3_fit_status import P0, _write_file

pytestmark = pytest.mark.unit

T = np.geomspace(0.005, 1, 25)
CURVE = 1 + 0.5 * np.exp(-2 * T / 0.1)

# double_exp_all(x, a, b1, c1, d, b2, c2, f): both decay constants (b1, b2) and
# both stretch exponents (c1, c2) are fixed to the same values, so the mixing
# fraction f is arbitrary: only (a, d, f) are free and the Jacobian has rank 2.
DOUBLE_FLAG = np.array([True, False, False, True, False, False, True])
DOUBLE_P0 = np.array([0.5, 0.1, 1.0, 1.0, 0.1, 1.0, 0.5])


def _double_bounds(lo=0.0, hi=1.0):
    return np.array(
        [[0.0, 0.1, 0.9, 0.9, 0.1, 0.9, lo], [1.0, 0.2, 1.1, 1.1, 0.2, 1.1, hi]]
    )


def _double_args(n_cols=3):
    y = np.stack([CURVE] * n_cols, axis=1)
    return (
        double_exp_all,
        T,
        y,
        np.full_like(y, 0.01),
        _double_bounds(),
        DOUBLE_FLAG,
        T,
    )


# --------------------------------------------------------------------------
# R4-01: converged but rank-deficient fit
# --------------------------------------------------------------------------
@pytest.fixture
def equal_time_file(tmp_path):
    path = tmp_path / "equal_times.h5"
    _write_file(path, T, g2=np.column_stack([CURVE] * 3))
    return XpcsFile(str(path))


@pytest.mark.parametrize("robust", [False, True])
def test_unidentifiable_mixing_fraction_is_not_a_measurement(equal_time_file, robust):
    for lo, hi in [(0.0, 1.0), (0.1, 0.4)]:
        fit = equal_time_file.fit_g2(
            bounds=_double_bounds(lo, hi).tolist(),
            fit_flag=DOUBLE_FLAG.tolist(),
            fit_func="double",
            robust_fitting=robust,
            force_refit=True,
        )
        assert not np.asarray(fit["fit_success"]).any()
        assert not fit_success_mask(fit).any()


@pytest.mark.parametrize(
    "solver", [fit_with_fixed, fit_with_fixed_parallel, fit_with_fixed_sequential]
)
def test_every_solver_path_flags_rank_deficiency(solver):
    out = solver(*_double_args(), p0=DOUBLE_P0, return_status=True)
    assert not out[-1].any()


@pytest.mark.parametrize("solver", [fit_with_fixed, fit_with_fixed_parallel])
def test_identifiable_double_exponential_stays_successful(solver):
    # Free: contrast, baseline, fraction and the first decay constant; the model
    # now distinguishes the two components, so the Jacobian has full column rank.
    truth = DOUBLE_P0.copy()
    truth[[1, 4]] = [0.05, 0.5]
    y = np.stack([double_exp_all(T, *truth)] * 3, axis=1)
    flag = np.array([True, True, False, True, False, False, True])
    bounds = _double_bounds()
    bounds[:, 1] = [0.01, 0.2]
    out = solver(
        double_exp_all,
        T,
        y,
        np.full_like(y, 0.01),
        bounds,
        flag,
        T,
        p0=np.array([0.5, 0.06, 1.0, 1.0, 0.5, 1.0, 0.5]),
        return_status=True,
    )
    assert out[-1].all()


@pytest.mark.parametrize("solver", [fit_with_fixed, fit_with_fixed_parallel])
def test_identifiable_fit_on_a_bound_stays_successful(solver):
    # The true baseline 1.05 lies above its bound [0.9, 1.0]: the solution sits on
    # the bound but the Jacobian still has full column rank, so it is a measurement.
    y = np.stack([1.05 + 0.5 * np.exp(-2 * T / 0.1)] * 2, axis=1)
    bounds = np.array([[0.01, 1e-3, 0.5, 0.9], [1.0, 1e2, 1.5, 1.0]])
    out = solver(
        single_exp_all,
        T,
        y,
        np.full_like(y, 0.01),
        bounds,
        np.ones(4, bool),
        T,
        p0=P0,
        return_status=True,
    )
    assert out[-1].all()
    assert np.allclose(out[1][:, 0, 3], 1.0)  # baseline pinned at its bound


# --------------------------------------------------------------------------
# R4-02: a non-converged Bayesian fit is kept for diagnosis, not a measurement
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def one_chain_run(tmp_path_factory):
    """Real short single-chain NumPyro run (R-hat is unavailable for one chain)."""
    from xpcsviewer.fitting import fit_single_exp

    path = tmp_path_factory.mktemp("r4fit") / "sample.h5"
    _write_file(path, T)
    xf = XpcsFile(str(path))
    q, t_el, g2, err, _ = xf.get_g2_data()
    result = fit_single_exp(
        t_el,
        g2[:, 0],
        err[:, 0],
        num_warmup=100,
        num_samples=200,
        num_chains=1,
        random_seed=7,
        progress_bar=False,
    )
    return q, t_el, g2, result


def _summary(one_chain_run):
    from xpcsviewer.fitting.bayesian_assembly import assemble_fit_summary
    from xpcsviewer.fitting.models import single_exp_func

    q, t_el, g2, result = one_chain_run
    return assemble_fit_summary(
        {0: result},
        np.array(q[:1]),
        t_el,
        "single",
        single_exp_func,
        obs_g2=g2[:, :1],
        obs_t_el=t_el,
    )


def test_one_chain_run_is_not_converged(one_chain_run):
    assert one_chain_run[3].diagnostics.converged is False


def test_nonconverged_bayesian_fit_is_flagged_not_a_measurement(one_chain_run):
    s = _summary(one_chain_run)
    assert s["failed_mask"].tolist() == [True]
    assert s["nonconverged_mask"].tolist() == [True]
    assert not fit_success_mask(s).any()
    assert s["n_converged"] == 0
    assert not np.isfinite(s["fit_val"][0, 0]).any()  # no unmasked reader sees it
    assert not np.isfinite(s["fit_line"][0]).any()


def test_nonconverged_bayesian_posterior_stays_available_for_diagnosis(one_chain_run):
    result = one_chain_run[3]
    assert result.arviz_data is not None
    assert result.diagnostics.r_hat  # the diagnostics that explain the verdict


def test_nonconverged_bayesian_fit_is_not_exported_as_ok(one_chain_run, tmp_path):
    from xpcsviewer.fitting.viz import export_bayesian_csv

    s = _summary(one_chain_run)
    out = tmp_path / "params.csv"
    export_bayesian_csv(
        out,
        s["fit_val"],
        s["q_val"],
        s["fit_func"],
        failed_mask=s["failed_mask"],
        nonconverged_mask=s["nonconverged_mask"],
    )
    assert out.read_text().splitlines()[1].endswith(",not_converged")
    # a caller that only passes failed_mask still never writes "ok"
    export_bayesian_csv(
        out, s["fit_val"], s["q_val"], s["fit_func"], failed_mask=s["failed_mask"]
    )
    assert not out.read_text().splitlines()[1].endswith(",ok")


def test_nonconverged_bayesian_plot_has_no_fit_line_and_says_converged(one_chain_run):
    import matplotlib.pyplot as plt

    from xpcsviewer.fitting.viz import plot_bayesian_all_q

    fig = plot_bayesian_all_q(_summary(one_chain_run))
    try:
        ax = fig.axes[0]
        assert "0 converged of 1" in ax.get_title()
        assert "succeeded" not in ax.get_title()
        assert not any(len(c.get_segments()) for c in ax.collections[1:])
    finally:
        plt.close(fig)


def test_convergence_message_counts_converged_and_names_single_chain(
    one_chain_run, caplog
):
    from xpcsviewer.fitting.bayesian_assembly import convergence_message

    with caplog.at_level("INFO", logger="xpcsviewer.fitting.bayesian_assembly"):
        s = _summary(one_chain_run)
    msg = convergence_message(s)
    assert "0 converged of 1" in msg
    assert "single" in msg and "R-hat" in msg
    assert "succeeded" not in msg
    assert "succeeded" not in caplog.text
    assert "0 converged of 1" in caplog.text


def test_converged_bayesian_fit_remains_a_measurement():
    from xpcsviewer.fitting.bayesian_assembly import (
        assemble_fit_summary,
        convergence_message,
    )
    from xpcsviewer.fitting.models import single_exp_func
    from xpcsviewer.fitting.results import FitDiagnostics, FitResult

    rng = np.random.default_rng(0)
    samples = {
        "contrast": 0.5 + 0.01 * rng.standard_normal(100),
        "tau": 0.1 + 0.001 * rng.standard_normal(100),
        "baseline": 1.0 + 0.01 * rng.standard_normal(100),
    }
    good = FitResult(samples=samples, diagnostics=FitDiagnostics(bfmi=0.9))
    assert good.diagnostics.converged
    s = assemble_fit_summary(
        {0: good, 1: None}, np.array([0.1, 0.2]), T, "single", single_exp_func
    )
    assert s["failed_mask"].tolist() == [False, True]
    assert s["nonconverged_mask"].tolist() == [False, False]
    assert fit_success_mask(s).tolist() == [True, False]
    assert s["n_converged"] == 1
    assert "1 converged of 2" in convergence_message(s)


# --------------------------------------------------------------------------
# R4-15: netCDF export without an installed backend
# --------------------------------------------------------------------------
def _patch_backends(monkeypatch, available=()):
    import importlib.util

    real = importlib.util.find_spec

    def find_spec(name, *a, **k):
        if name in ("netCDF4", "h5netcdf"):
            return object() if name in available else None
        return real(name, *a, **k)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


def test_netcdf_export_without_backend_skips_with_a_named_warning(
    one_chain_run, tmp_path, monkeypatch, caplog
):
    from xpcsviewer.fitting.viz import export_bayesian_diagnostics

    _patch_backends(monkeypatch)
    with caplog.at_level("WARNING", logger="xpcsviewer.fitting.viz"):
        written = export_bayesian_diagnostics(
            tmp_path / "bayesian_diagnostics.nc", {0: one_chain_run[3]}
        )
    assert written == []
    assert "netCDF4" in caplog.text and "h5netcdf" in caplog.text
    assert list(tmp_path.iterdir()) == []


def test_netcdf_export_with_backend_returns_the_files_written(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from xpcsviewer.fitting.viz import export_bayesian_diagnostics

    _patch_backends(monkeypatch, available=("h5netcdf",))
    calls = []

    def to_netcdf(p):
        calls.append(p)
        open(p, "w").close()

    fr = SimpleNamespace(arviz_data=SimpleNamespace(to_netcdf=to_netcdf))
    written = export_bayesian_diagnostics(tmp_path / "d.nc", {0: fr, 2: fr, 3: None})
    assert [p.name for p in written] == ["d_q000.nc", "d_q002.nc"]
    assert [str(p) for p in written] == calls
