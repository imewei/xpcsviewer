"""Fixed fit parameters must carry explicit values; no silent midpoint fallback."""

from __future__ import annotations

import numpy as np
import pytest

from xpcsviewer.fitting import legacy
from xpcsviewer.fitting.legacy import (
    fit_with_fixed,
    fit_with_fixed_parallel,
    fit_with_fixed_sequential,
    make_single_exp,
)

# make_single_exp signature: (x, tau, bkg, cts)
TRUE = (0.5, 1.0, 0.2)
BOUNDS = np.array([[0.1, 0.5, 0.05], [1.0, 1.5, 0.35]])
FIT_X = np.logspace(-2, 1, 40)

SOLVERS = [fit_with_fixed, fit_with_fixed_sequential, fit_with_fixed_parallel]


def _data(n_columns: int = 2):
    x = np.logspace(-2, 1, 30)
    model = make_single_exp()
    y = np.stack([np.asarray(model(x, *TRUE)) for _ in range(n_columns)], axis=1)
    return x, y, np.full_like(y, 1e-3)


def _call(solver, fit_flag, p0, n_columns: int = 2):
    x, y, sigma = _data(n_columns)
    return solver(
        make_single_exp(),
        x,
        y,
        sigma,
        BOUNDS,
        np.array(fit_flag),
        FIT_X,
        p0=p0,
    )[:2]


@pytest.mark.parametrize("solver", SOLVERS)
def test_fixed_parameter_holds_explicit_p0(solver):
    p0 = np.array([0.4, 0.9, 1.1])
    _, fit_val = _call(solver, [True, True, False], p0)
    assert fit_val[0, 0, 2] == 1.1


@pytest.mark.parametrize(
    ("solver", "target"),
    [
        (fit_with_fixed, "curve_fit"),
        (fit_with_fixed_parallel, "curve_fit"),
        (fit_with_fixed_sequential, "sequential_fitting"),
    ],
)
def test_failed_fit_keeps_fixed_slots(solver, target, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("solver failure")

    monkeypatch.setattr(legacy, target, boom)
    p0 = np.array([0.4, 0.9, 1.1])
    _, fit_val = _call(solver, [True, True, False], p0)
    assert fit_val[0, 0, 2] == 1.1
    np.testing.assert_allclose(fit_val[0, 0, :2], BOUNDS.mean(axis=0)[:2])
    assert np.all(fit_val[0, 1, :] == 0)


@pytest.mark.parametrize("solver", SOLVERS)
def test_missing_p0_with_fixed_parameter_raises(solver):
    with pytest.raises(ValueError, match="explicit p0"):
        _call(solver, [True, True, False], None)


@pytest.mark.parametrize("solver", SOLVERS)
def test_nonfinite_fixed_p0_raises(solver):
    with pytest.raises(ValueError, match="finite"):
        _call(solver, [True, True, False], np.array([0.4, 0.9, np.nan]))


@pytest.mark.parametrize("solver", SOLVERS)
def test_all_free_without_p0_still_fits(solver):
    fit_line, _ = _call(solver, [True, True, True], None)
    truth = np.asarray(make_single_exp()(FIT_X, *TRUE))
    np.testing.assert_allclose(fit_line[0], truth, rtol=1e-6)


def test_initial_fit_guess_midpoint_and_geometric_tau():
    from xpcsviewer.fitting import initial_fit_guess

    bounds = np.array([[0.1, 1e-3, 0.5], [1.0, 1e1, 1.5]])
    np.testing.assert_allclose(initial_fit_guess(bounds, [1]), [0.55, 0.1, 1.0])
    # A non-positive tau bound falls back to the arithmetic midpoint.
    bounds[0, 1] = 0.0
    np.testing.assert_allclose(initial_fit_guess(bounds, [1]), [0.55, 5.0, 1.0])


def test_fit_g2_fixed_baseline_is_held_at_bounds_midpoint():
    from xpcsviewer.utils.memory_manager import get_memory_manager
    from xpcsviewer.xpcs_file import XpcsFile
    from xpcsviewer.xpcs_file.fitting import single_exp_all

    t_el = np.logspace(-2, 1, 30)
    g2 = np.stack(
        [np.asarray(single_exp_all(t_el, 0.2, 0.5, 1.0, 1.0)) for _ in range(2)],
        axis=1,
    )
    source = XpcsFile.__new__(XpcsFile)
    source._memory_manager = get_memory_manager()
    source._cache_prefix = "task2-fit-g2-fixed-baseline"
    source.get_g2_data = lambda qrange=None, trange=None: (
        np.array([0.01, 0.02]),
        t_el,
        g2,
        np.full_like(g2, 1e-3),
        ["q0", "q1"],
    )
    bounds = np.array([[0.05, 0.1, 0.5, 0.9], [0.35, 2.0, 1.5, 1.3]])
    summary = source.fit_g2(
        bounds=bounds,
        fit_flag=[True, True, True, False],
        fit_func="single",
        force_refit=True,
    )
    np.testing.assert_allclose(summary["fit_val"][:, 0, 3], 1.1)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize(
    "flags",
    [[1, 1, 0], np.array([1, 1, 0], dtype=np.int64), [1, 1, 1]],
    ids=["int", "np-int64", "int-all-free"],
)
def test_integer_fit_flags_match_bool_flags(solver, flags):
    p0 = np.array([0.4, 0.9, 0.2])
    want = _call(solver, np.asarray(flags, dtype=bool), p0)
    got = _call(solver, flags, p0)
    np.testing.assert_array_equal(got[1], want[1])
    np.testing.assert_array_equal(got[0], want[0])
