"""Round-2 review regressions: R2-12, R2-14, R2-21 (solver status), R2-37."""

from __future__ import annotations

import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

from xpcsviewer.fitting import legacy
from xpcsviewer.fitting.legacy import (
    fit_with_fixed,
    fit_with_fixed_parallel,
    single_exp_all,
)
from xpcsviewer.xpcs_file.fitting import power_law

pytestmark = pytest.mark.unit

# Data from the review report (R2-21).
TAUQ_Q = np.array([0.05, 0.1, 0.2, 0.4, 0.6])
TAUQ_TAU = np.array([5.0, 11.0, 1.0, 3.0, 0.07])
POWER_LAW_BOUNDS = np.array([[1e-3, -3.0], [1e3, 3.0]])


def _tauq_args(n_cols: int = 1):
    y = np.tile(TAUQ_TAU[:, None], (1, n_cols))
    sigma = np.full_like(y, 0.1)
    return (
        power_law,
        TAUQ_Q,
        y,
        sigma,
        POWER_LAW_BOUNDS,
        np.ones(2, dtype=bool),
        TAUQ_Q,
    )


class _EvalLimited:
    """Force a tiny evaluation limit on the real nlsq solver."""

    def __init__(self, monkeypatch, max_nfev: int) -> None:
        real = legacy.curve_fit
        monkeypatch.setattr(
            legacy,
            "curve_fit",
            lambda *a, **k: real(*a, **{**k, "max_nfev": max_nfev}),
        )


# --------------------------------------------------------------------------
# R2-21: solver termination status
# --------------------------------------------------------------------------
def test_fit_with_fixed_flags_evaluation_limit_as_failure(monkeypatch):
    _EvalLimited(monkeypatch, 1)
    out = fit_with_fixed(*_tauq_args(2), p0=np.array([1.0, -1.0]), return_status=True)
    assert len(out) == 3
    assert out[2].shape == (2,)
    assert not out[2].any()


def test_fit_with_fixed_default_return_is_still_a_pair(monkeypatch):
    _EvalLimited(monkeypatch, 1)
    out = fit_with_fixed(*_tauq_args(), p0=np.array([1.0, -1.0]))
    assert len(out) == 2


def test_fit_with_fixed_converged_fit_reports_success():
    x = np.array([0.05, 0.1, 0.2, 0.4, 0.6])
    y = (2.0 * x**-1.5)[:, None]
    _, fit_val, ok = fit_with_fixed(
        power_law,
        x,
        y,
        np.full_like(y, 0.01),
        POWER_LAW_BOUNDS,
        np.ones(2, dtype=bool),
        x,
        p0=np.array([1.0, -1.0]),
        return_status=True,
    )
    assert ok.all()
    np.testing.assert_allclose(fit_val[0, 0], [2.0, -1.5], rtol=1e-3)


# --------------------------------------------------------------------------
# R2-37: invented uncertainties
# --------------------------------------------------------------------------
def test_underdetermined_column_is_flagged_not_given_invented_errors():
    x = np.array([0.03, 0.04])
    y = np.array([[1.1], [1.2]])
    bounds = np.array([[0.0, 1e-3, 0.0, 0.5], [1.0, 1e3, 1.0, 1.5]])
    _, fit_val, ok = fit_with_fixed(
        single_exp_all,
        x,
        y,
        np.full_like(y, 0.01),
        bounds,
        np.ones(4, dtype=bool),
        x,
        p0=np.array([0.1, 1.0, 0.1, 1.0]),
        return_status=True,
    )
    assert not ok[0]
    # Not the old "10% of the estimate" invention.
    assert not np.allclose(fit_val[0, 1], 0.1 * np.abs(fit_val[0, 0]))


def test_unavailable_covariance_is_reported_as_nan():
    # single_exp_all has c + d degenerate and 4 observations for 4 parameters:
    # nlsq cannot estimate the covariance.
    rng = np.random.default_rng(0)
    x = np.logspace(-3, 0, 4)
    y = (0.2 * np.exp(-2 * x / 0.05) + 1.0 + rng.normal(0, 0.005, 4))[:, None]
    bounds = np.array([[0.0, 1e-3, 0.0, 0.0], [1.0, 1e3, 2.0, 2.0]])
    _, fit_val, _ = fit_with_fixed(
        single_exp_all,
        x,
        y,
        np.full_like(y, 0.01),
        bounds,
        np.ones(4, dtype=bool),
        x,
        p0=np.array([0.1, 0.1, 0.5, 0.5]),
        return_status=True,
    )
    assert np.isnan(fit_val[0, 1]).all()
    assert np.isfinite(fit_val[0, 0]).all()


def test_identifiable_fit_with_unavailable_covariance_has_nan_errors(monkeypatch):
    # Since R4-01 the test above is failed by the rank check before the
    # covariance is read. This one keeps _param_errors honest end to end: an
    # identifiable, converged fit whose covariance could not be estimated is a
    # success with NaN errors, never invented ones.
    popt = np.array([2.0, -1.0])

    class _Result:
        success = True
        message = ""
        jac = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # full column rank

        def __iter__(self):
            return iter((popt, np.inf * np.eye(2)))

    monkeypatch.setattr(legacy, "curve_fit", lambda *a, **k: _Result())
    x = np.array([0.1, 0.2, 0.4])
    y = np.array([[4.0], [2.0], [1.0]])
    _, fit_val, ok = fit_with_fixed(
        lambda q, a, b: a * q**b,
        x,
        y,
        np.full_like(y, 0.1),
        np.array([[0.0, -3.0], [5.0, 0.0]]),
        np.ones(2, dtype=bool),
        x,
        p0=np.array([1.0, -1.0]),
        return_status=True,
    )
    assert ok.all()
    np.testing.assert_allclose(fit_val[0, 0], popt)
    assert np.isnan(fit_val[0, 1]).all()


def test_param_errors_nan_for_nonfinite_or_negative_diagonal():
    pcov = np.array([[4.0, 0.0, 0.0], [0.0, np.inf, 0.0], [0.0, 0.0, -1.0]])
    errors = legacy._param_errors(pcov, 3)
    assert errors[0] == 2.0
    assert np.isnan(errors[1:]).all()
    assert np.isnan(legacy._param_errors(np.zeros((2, 2)), 3)).all()


# --------------------------------------------------------------------------
# R2-14: process-parallel fitting must pickle its work
# --------------------------------------------------------------------------
class _PicklingInlineExecutor:
    """ProcessPoolExecutor stand-in: pickles callables/args, then runs inline."""

    def __init__(self, max_workers=None, **_):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args):
        from concurrent.futures import Future

        fn, args = pickle.loads(pickle.dumps((fn, args)))
        fut: Future = Future()
        fut.set_result(fn(*args))
        return fut


def _parallel_args():
    x = TAUQ_Q
    y = np.stack([2.0 * x**-1.5, 3.0 * x**-1.0], axis=1)
    return power_law, x, y, np.full_like(y, 0.01), POWER_LAW_BOUNDS, np.ones(2, bool), x


def test_parallel_process_path_submits_picklable_work(monkeypatch):
    monkeypatch.setattr(legacy, "ProcessPoolExecutor", _PicklingInlineExecutor)
    _, fit_val = fit_with_fixed_parallel(
        *_parallel_args(), p0=np.array([1.0, -1.0]), use_threads=False
    )
    np.testing.assert_allclose(fit_val[:, 0, :], [[2.0, -1.5], [3.0, -1.0]], rtol=1e-3)


def test_parallel_process_path_rejects_unpicklable_model_clearly():
    args = list(_parallel_args())
    args[0] = lambda x, a, b: a * x**b  # noqa: E731 - local, unpicklable
    with pytest.raises(ValueError, match="picklable"):
        fit_with_fixed_parallel(*args, p0=np.array([1.0, -1.0]), use_threads=False)


def test_parallel_thread_path_matches_sequential_fit():
    _, par = fit_with_fixed_parallel(
        *_parallel_args(), p0=np.array([1.0, -1.0]), use_threads=True
    )
    _, seq = fit_with_fixed(*_parallel_args(), p0=np.array([1.0, -1.0]))
    np.testing.assert_allclose(par[:, 0, :], seq[:, 0, :], rtol=1e-6)


def test_parallel_real_spawned_processes_fit(monkeypatch):
    ctx = multiprocessing.get_context("spawn")
    monkeypatch.setattr(
        legacy,
        "ProcessPoolExecutor",
        lambda max_workers=None: ProcessPoolExecutor(
            max_workers=max_workers, mp_context=ctx
        ),
    )
    try:
        _, fit_val = fit_with_fixed_parallel(
            *_parallel_args(),
            p0=np.array([1.0, -1.0]),
            max_workers=2,
            use_threads=False,
        )
    except (PermissionError, OSError) as exc:  # sandbox without process pipes
        pytest.skip(f"process pool unavailable here: {exc}")
    np.testing.assert_allclose(fit_val[:, 0, :], [[2.0, -1.5], [3.0, -1.0]], rtol=1e-3)


# --------------------------------------------------------------------------
# R2-12: unavailable ArviZ diagnostics on very short chains
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def short_chain_result():
    pytest.importorskip("numpyro")
    pytest.importorskip("arviz")
    from xpcsviewer.fitting.sampler import run_single_exp_fit

    x = np.logspace(-3, 0, 30)
    y = 1.0 + 0.2 * np.exp(-2 * x / 0.05)
    return run_single_exp_fit(
        x,
        y,
        yerr=np.full(30, 0.01),
        num_warmup=10,
        num_samples=3,
        num_chains=1,
        random_seed=42,
    )


def test_short_chain_does_not_crash_and_carries_nan(short_chain_result):
    diag = short_chain_result.diagnostics
    assert set(diag.ess_bulk) == set(short_chain_result.param_names)
    unavailable = [
        v
        for d in (diag.r_hat, diag.ess_bulk, diag.ess_tail)
        for v in d.values()
        if not np.isfinite(v)
    ]
    assert unavailable, "expected ArviZ to report NaN diagnostics for 3 draws"
    # Finite ESS values (if any) stay integers; NaN is never cast to int.
    for d in (diag.ess_bulk, diag.ess_tail):
        for v in d.values():
            assert np.isnan(v) or isinstance(v, int)


def test_short_chain_is_never_reported_converged(short_chain_result):
    assert short_chain_result.diagnostics.converged is False
