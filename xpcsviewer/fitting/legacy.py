"""Legacy fitting utilities for backward compatibility.

Provides the same API as the old xpcsviewer.helper.fitting module,
migrated to use the new JAX-accelerated backend where possible.

These functions maintain the same interface for xpcs_file.py compatibility
while leveraging the new fitting infrastructure internally.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, cast

import numpy as np
from nlsq import curve_fit
from numpy.typing import NDArray

from xpcsviewer.backends._conversions import ensure_numpy
from xpcsviewer.utils.log_utils import log_timing
from xpcsviewer.utils.logging_config import get_logger

logger = get_logger(__name__)


# Model factory functions for nlsq.curve_fit.
# Each factory imports jax.numpy directly because NLSQ JIT-traces model
# functions with JAX regardless of the xpcsviewer backend setting.


def make_single_exp() -> Callable[..., NDArray[np.floating[Any]]]:
    """Factory: create a single exponential closure.

    Uses ``jax.numpy`` directly because ``nlsq.curve_fit`` JIT-traces
    model functions with JAX regardless of the xpcsviewer backend setting.
    """
    import jax.numpy as jnp

    def _single_exp(
        x: NDArray[np.floating[Any]], tau: float, bkg: float, cts: float
    ) -> NDArray[np.floating[Any]]:
        return cts * jnp.exp(-2 * jnp.asarray(x) / tau) + bkg  # type: ignore[return-value]

    return _single_exp


def make_double_exp() -> Callable[..., NDArray[np.floating[Any]]]:
    """Factory: create a double exponential closure.

    Uses ``jax.numpy`` directly because ``nlsq.curve_fit`` JIT-traces
    model functions with JAX regardless of the xpcsviewer backend setting.
    """
    import jax.numpy as jnp

    def _double_exp(
        x: NDArray[np.floating[Any]],
        tau1: float,
        bkg: float,
        cts1: float,
        tau2: float,
        cts2: float,
    ) -> Any:
        xa: Any = jnp.asarray(x)
        return cts1 * jnp.exp(-2 * xa / tau1) + cts2 * jnp.exp(-2 * xa / tau2) + bkg

    return _double_exp


def make_single_exp_all() -> Callable[..., NDArray[np.floating[Any]]]:
    """Factory: create a single_exp_all closure.

    Uses ``jax.numpy`` directly because ``nlsq.curve_fit`` JIT-traces
    model functions with JAX regardless of the xpcsviewer backend setting.
    """
    import jax.numpy as jnp

    def _single_exp_all(
        x: NDArray[np.floating[Any]], a: float, b_: float, c: float, d: float
    ) -> NDArray[np.floating[Any]]:
        return a * jnp.exp(-2 * jnp.asarray(x) / b_) + c + d  # type: ignore[return-value]

    return _single_exp_all


def make_double_exp_all() -> Callable[..., NDArray[np.floating[Any]]]:
    """Factory: create a double_exp_all closure.

    Uses ``jax.numpy`` directly because ``nlsq.curve_fit`` JIT-traces
    model functions with JAX regardless of the xpcsviewer backend setting.
    """
    import jax.numpy as jnp

    def _double_exp_all(
        x: NDArray[np.floating[Any]],
        a: float,
        b_: float,
        c: float,
        d: float,
        e: float,
        f: float,
    ) -> Any:
        xa: Any = jnp.asarray(x)
        return a * jnp.exp(-2 * xa / b_) + c * jnp.exp(-2 * xa / d) + e + f

    return _double_exp_all


# ---------------------------------------------------------------------------
# Backward-compatible module-level functions (kept for callers that import them
# directly by name). These delegate to per-call factory closures.
# ---------------------------------------------------------------------------
_single_exp_fn: Callable[..., NDArray[np.floating[Any]]] | None = None
_double_exp_fn: Callable[..., NDArray[np.floating[Any]]] | None = None
_single_exp_all_fn: Callable[..., NDArray[np.floating[Any]]] | None = None
_double_exp_all_fn: Callable[..., NDArray[np.floating[Any]]] | None = None


def reset_legacy_closures() -> None:
    """Reset cached model function closures (call after reset_backend()).

    The module-level singletons capture the backend at first use. If the
    backend is reset (e.g., switching from NumPy to JAX), these stale
    closures must be invalidated so the next call re-creates them with
    the new backend.
    """
    global _single_exp_fn, _double_exp_fn, _single_exp_all_fn, _double_exp_all_fn
    _single_exp_fn = None
    _double_exp_fn = None
    _single_exp_all_fn = None
    _double_exp_all_fn = None


def single_exp(
    x: NDArray[np.floating[Any]], tau: float, bkg: float, cts: float
) -> NDArray[np.floating[Any]]:
    """Single exponential model for G2 correlation function.

    Delegates to a module-level closure created once via make_single_exp().
    Uses jax.numpy directly for NLSQ JIT compatibility.
    """
    global _single_exp_fn
    if _single_exp_fn is None:
        _single_exp_fn = make_single_exp()
    return _single_exp_fn(x, tau, bkg, cts)


def double_exp(
    x: NDArray[np.floating[Any]],
    tau1: float,
    bkg: float,
    cts1: float,
    tau2: float,
    cts2: float,
) -> NDArray[np.floating[Any]]:
    """Double exponential model for G2 correlation function.

    Delegates to a module-level closure created once via make_double_exp().
    Uses jax.numpy directly for NLSQ JIT compatibility.
    """
    global _double_exp_fn
    if _double_exp_fn is None:
        _double_exp_fn = make_double_exp()
    return _double_exp_fn(x, tau1, bkg, cts1, tau2, cts2)


def single_exp_all(
    x: NDArray[np.floating[Any]], a: float, b_: float, c: float, d: float
) -> NDArray[np.floating[Any]]:
    """Single exponential with all parameters.

    Delegates to a module-level closure created once via make_single_exp_all().
    Uses jax.numpy directly for NLSQ JIT compatibility.
    """
    global _single_exp_all_fn
    if _single_exp_all_fn is None:
        _single_exp_all_fn = make_single_exp_all()
    return _single_exp_all_fn(x, a, b_, c, d)


def double_exp_all(
    x: NDArray[np.floating[Any]],
    a: float,
    b_: float,
    c: float,
    d: float,
    e: float,
    f: float,
) -> NDArray[np.floating[Any]]:
    """Double exponential with all parameters.

    Delegates to a module-level closure created once via make_double_exp_all().
    Uses jax.numpy directly for NLSQ JIT compatibility.
    """
    global _double_exp_all_fn
    if _double_exp_all_fn is None:
        _double_exp_all_fn = make_double_exp_all()
    return _double_exp_all_fn(x, a, b_, c, d, e, f)


def _resolve_full_p0(
    p0: NDArray[np.floating[Any]] | None,
    bounds: NDArray[np.floating[Any]],
    fit_flag: NDArray[np.bool_],
) -> NDArray[np.floating[Any]]:
    """Return the full starting vector, refusing to invent values for fixed slots.

    Fixed parameters are held at their ``p0`` value, so ``p0`` is mandatory
    whenever any ``fit_flag`` entry is False. With every parameter free, the
    bounds midpoint is only an optimiser starting point and is acceptable.
    """
    fit_flag = np.asarray(fit_flag, dtype=bool)
    if p0 is None:
        if not fit_flag.all():
            raise ValueError("fixed parameters require explicit p0 values")
        return np.mean(bounds, axis=0)
    full_p0 = np.array(p0, dtype=float)
    if full_p0.shape != fit_flag.shape:
        raise ValueError(
            f"p0 has shape {full_p0.shape} but fit_flag has shape {fit_flag.shape}"
        )
    if not np.all(np.isfinite(full_p0[~fit_flag])):
        raise ValueError("fixed parameters require finite explicit p0 values")
    return full_p0


def initial_fit_guess(
    bounds: NDArray[np.floating[Any]], tau_indices: Sequence[int]
) -> NDArray[np.floating[Any]]:
    """Bounds midpoint, using the geometric mean for positive-bounded taus."""
    bounds = np.asarray(bounds, dtype=float)
    guess = np.mean(bounds, axis=0)
    for idx in tau_indices:
        if bounds[0, idx] > 0 and bounds[1, idx] > 0:
            guess[idx] = np.sqrt(bounds[0, idx] * bounds[1, idx])
    return guess


def _param_errors(pcov: Any, n_free: int) -> NDArray[np.floating[Any]]:
    """Standard errors from the covariance diagonal; NaN where unavailable.

    An uncertainty the solver could not estimate is reported as NaN, never
    replaced by an invented value.
    """
    pcov = np.asarray(pcov, dtype=float)
    if pcov.shape != (n_free, n_free):
        return np.full(n_free, np.nan)
    diag = np.diag(pcov)
    return np.sqrt(np.where(np.isfinite(diag) & (diag >= 0), diag, np.nan))


def fit_success_mask(fit_summary: dict[str, Any]) -> NDArray[np.bool_]:
    """Per-Q boolean mask: True where a G2 fit result is a usable measurement.

    The solver's stored ``fit_success`` (or a Bayesian summary's ``failed_mask``)
    is authoritative; a row also needs finite fitted parameters. Summaries that
    store neither fall back to the finiteness test alone. Every consumer of
    ``fit_val`` / ``fit_line`` (plots, exports, tau-Q) uses this one rule.
    """
    val = np.asarray(fit_summary["fit_val"], dtype=float)
    if val.ndim != 3:
        return np.zeros(len(val), dtype=bool)
    ok = np.isfinite(val[:, 0, :]).all(axis=1)
    stored = fit_summary.get("fit_success")
    if stored is None and fit_summary.get("failed_mask") is not None:
        stored = ~np.asarray(fit_summary["failed_mask"], dtype=bool)
    if stored is None:
        return ok
    stored = np.asarray(stored, dtype=bool).ravel()
    aligned = np.zeros(len(ok), dtype=bool)  # rows without a status are not trusted
    n = min(len(stored), len(ok))
    aligned[:n] = stored[:n]
    return ok & aligned


def _solve_column(
    wrapper_func: Callable[..., Any],
    x: NDArray[np.floating[Any]],
    y_col: NDArray[np.floating[Any]],
    sigma_col: NDArray[np.floating[Any]],
    p0: NDArray[np.floating[Any]],
    bounds_fit: NDArray[np.floating[Any]],
    **curve_fit_kwargs: Any,
) -> tuple[
    NDArray[np.floating[Any]] | None, NDArray[np.floating[Any]] | None, bool, str
]:
    """Fit one column under the single set of rules every solver path shares.

    Returns ``(popt, errors, ok, reason)``. ``ok`` is False when the column has
    fewer finite observations than free parameters, when the solver raised, or
    when it stopped without converging (``result.success`` False, e.g. at the
    evaluation limit). ``popt`` is None when no iterate exists; a non-converged
    column keeps the solver's last iterate (still ``ok`` False). ``errors`` come
    from ``_param_errors`` (NaN where the covariance is unavailable).
    """
    n_obs = int(np.count_nonzero(np.isfinite(y_col) & np.isfinite(x)))
    if n_obs < len(p0):
        return (
            None,
            None,
            False,
            f"{n_obs} observations cannot determine {len(p0)} free parameters",
        )
    try:
        result = curve_fit(
            wrapper_func,
            x,
            y_col,
            sigma=sigma_col,
            p0=p0,
            bounds=(bounds_fit[0], bounds_fit[1]),
            **curve_fit_kwargs,
        )
        popt, pcov = result
    except Exception as e:
        return None, None, False, str(e)
    popt = np.asarray(popt, dtype=float)
    errors = _param_errors(pcov, len(p0))
    if not getattr(result, "success", True):
        return (
            popt,
            errors,
            False,
            f"did not converge: {getattr(result, 'message', '')}",
        )
    if not np.all(np.isfinite(popt)):
        return popt, errors, False, "non-finite parameters"
    return popt, errors, True, ""


@log_timing(threshold_ms=100)
def fit_with_fixed(
    base_func: Callable[..., NDArray[np.floating[Any]]],
    x: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    sigma: NDArray[np.floating[Any]],
    bounds: NDArray[np.floating[Any]],
    fit_flag: NDArray[np.bool_],
    fit_x: NDArray[np.floating[Any]],
    p0: NDArray[np.floating[Any]] | None = None,
    return_status: bool = False,
    **kwargs: Any,
) -> tuple[NDArray[Any], ...]:
    """Fitting with fixed parameters using nlsq.curve_fit.

    Parameters
    ----------
    base_func : callable
        Function to fit
    x : array
        Input data
    y : array
        Output data
    sigma : array
        Error bars
    bounds : tuple
        (lower_bounds, upper_bounds)
    fit_flag : array
        Boolean array indicating which parameters to fit
    fit_x : array
        X values for output curve
    p0 : array, optional
        Starting values, one per parameter. Required, and finite, for every
        parameter with ``fit_flag`` False: such a parameter is held at its
        ``p0`` value. With every parameter free it may be omitted and the
        bounds midpoint is used.
    return_status : bool, optional
        If True, also return a per-column boolean array that is False where the
        solver raised, where the column has fewer observations than free
        parameters, or where the solver stopped without converging (for
        example at the evaluation limit). Columns where the solver raised or
        the problem is underdetermined hold bounds-midpoint placeholders with
        zero errors, which are not a fit result; a non-converged column keeps
        the solver's last iterate.

    Returns
    -------
    tuple
        (fit_line, fit_params), or (fit_line, fit_params, success) when
        ``return_status`` is True. Parameter errors are NaN where the solver
        could not estimate the covariance.

    Raises
    ------
    ValueError
        If ``p0`` is missing or non-finite for a fixed parameter, or its shape
        differs from ``fit_flag``.
    """
    # Ensure numpy arrays at nlsq boundary
    x = ensure_numpy(x)
    y = ensure_numpy(y)
    sigma = ensure_numpy(sigma)
    fit_x = ensure_numpy(fit_x)

    fit_flag = np.asarray(fit_flag, dtype=bool)

    fix_flag = np.logical_not(fit_flag)

    bounds = np.asarray(bounds)

    num_args = len(fit_flag)

    # Process boundaries for fitting parameters only
    bounds_fit = bounds[:, fit_flag]

    # Preserve full initial guess before slicing to free params.
    # Fixed parameters are held at their explicit p0 value.
    full_p0 = _resolve_full_p0(p0, bounds, fit_flag)

    # Slice to free parameters for optimizer
    p0 = full_p0[fit_flag]

    fit_val = np.zeros((y.shape[1], 2, num_args))
    success = np.ones(y.shape[1], dtype=bool)

    # Fixed parameter values: use initial guess (semantically correct)
    fixed_values = full_p0.tolist()
    fit_indices = [i for i in range(num_args) if fit_flag[i]]

    # Create wrapper function for fixed parameters
    # Uses a Python list (not np.array) to avoid TracerArrayConversionError
    # when nlsq JIT-traces this function with JAX tracers as fit_params.
    def wrapper_func(x_data, *fit_params):
        full_params = list(fixed_values)
        for idx, val in zip(fit_indices, fit_params):
            full_params[idx] = val
        return base_func(x_data, *full_params)

    # Fit each column
    for n in range(y.shape[1]):
        sigma_col = sigma[:, n] if sigma.ndim > 1 else sigma
        popt, errors, ok, reason = _solve_column(
            wrapper_func, x, y[:, n], sigma_col, p0, bounds_fit, max_nfev=5000
        )
        if not ok:
            logger.warning(f"Fitting failed for column {n}: {reason}")
            success[n] = False
        if popt is not None:
            # nlsq reports non-convergence (e.g. evaluation limit) in `success`
            # instead of raising; the last iterate is kept but flagged.
            fit_val[n, 0, fit_flag] = popt
            fit_val[n, 1, fit_flag] = errors
            fit_val[n, 0, fix_flag] = full_p0[fix_flag]
            fit_val[n, 1, fix_flag] = 0
        else:
            fit_val[n, 0, :] = np.mean(bounds, axis=0)
            fit_val[n, 0, fix_flag] = full_p0[fix_flag]
            fit_val[n, 1, :] = 0

    # Generate fit lines
    fit_line = np.zeros((y.shape[1], len(fit_x)))
    for n in range(y.shape[1]):
        fit_line[n] = ensure_numpy(base_func(fit_x, *fit_val[n, 0, :]))

    if return_status:
        return fit_line, fit_val, success
    return fit_line, fit_val


class _FixedParamModel:
    """Model with some parameters held fixed, as a picklable callable.

    Module level (not a closure) so a process pool can pickle it. Uses a plain
    list, not np.array, to avoid TracerArrayConversionError when nlsq
    JIT-traces it with JAX tracers as ``fit_params``.
    """

    def __init__(
        self,
        base_func: Callable[..., Any],
        fixed_values: list[float],
        fit_indices: list[int],
    ) -> None:
        self.base_func = base_func
        self.fixed_values = fixed_values
        self.fit_indices = fit_indices

    def __call__(self, x_data: Any, *fit_params: Any) -> Any:
        full_params = list(self.fixed_values)
        for idx, val in zip(self.fit_indices, fit_params):
            full_params[idx] = val
        return self.base_func(x_data, *full_params)


def _fit_single_qvalue(
    args: tuple[Any, ...],
) -> tuple[
    int, NDArray[np.floating[Any]] | None, NDArray[np.floating[Any]] | None, bool
]:
    """Worker function for parallel fitting of a single q-value.

    Applies the same rules as ``fit_with_fixed`` (``_solve_column``): the bool is
    False for an underdetermined column, a solver error or a non-converged
    solve; a non-converged column still returns its last iterate.
    """
    col_idx, x, y_col, sigma_col, wrapper_func, p0, bounds_fit = args
    popt, errors, ok, reason = _solve_column(
        wrapper_func,
        x,
        y_col,
        sigma_col,
        p0,
        bounds_fit,
        method="trf",
        max_nfev=5000,
    )
    if not ok:
        logger.warning(f"Fitting failed for q-value {col_idx}: {reason}")
    return col_idx, popt, errors, ok


@log_timing(threshold_ms=500)
def fit_with_fixed_parallel(
    base_func: Callable[..., NDArray[np.floating[Any]]],
    x: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    sigma: NDArray[np.floating[Any]],
    bounds: NDArray[np.floating[Any]],
    fit_flag: NDArray[np.bool_],
    fit_x: NDArray[np.floating[Any]],
    p0: NDArray[np.floating[Any]] | None = None,
    max_workers: int | None = None,
    use_threads: bool = True,
    return_status: bool = False,
    **kwargs: Any,
) -> tuple[NDArray[Any], ...]:
    """Parallel version of fit_with_fixed for processing multiple q-values simultaneously.

    Same contract as ``fit_with_fixed``, including ``return_status`` (a per-column
    boolean success mask as a third return value).
    """
    # Ensure numpy arrays at nlsq boundary
    x = ensure_numpy(x)
    y = ensure_numpy(y)
    sigma = ensure_numpy(sigma)
    fit_x = ensure_numpy(fit_x)

    fit_flag = np.asarray(fit_flag, dtype=bool)
    fix_flag = np.logical_not(fit_flag)
    bounds = np.asarray(bounds)

    num_args = len(fit_flag)
    num_qvals = y.shape[1]

    bounds_fit = bounds[:, fit_flag]

    # Preserve full initial guess for fixed parameter values
    full_p0 = _resolve_full_p0(p0, bounds, fit_flag)
    p0 = full_p0[fit_flag]

    fit_val = np.zeros((num_qvals, 2, num_args))
    status = np.ones(num_qvals, dtype=bool)

    # Fixed parameter values: use initial guess (not upper bound)
    fixed_values_par = full_p0.tolist()
    fit_indices_par = [i for i in range(num_args) if fit_flag[i]]

    wrapper_func = _FixedParamModel(base_func, fixed_values_par, fit_indices_par)
    if not use_threads:
        try:
            pickle.dumps(wrapper_func)
        except Exception as e:
            raise ValueError(
                "use_threads=False needs a picklable, module-level base_func; "
                f"got {base_func!r} ({e}). Use use_threads=True instead."
            ) from e

    fit_args = []
    for n in range(num_qvals):
        sigma_col = sigma[:, n] if sigma.ndim > 1 else sigma
        fit_args.append((n, x, y[:, n], sigma_col, wrapper_func, p0, bounds_fit))

    if max_workers is None:
        import os

        max_workers = min(num_qvals, os.cpu_count() or 1)

    logger.info(
        f"Starting parallel G2 fitting for {num_qvals} q-values using {max_workers} workers"
    )

    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    with executor_class(max_workers=max_workers) as executor:
        future_to_col = {
            executor.submit(_fit_single_qvalue, args): args[0] for args in fit_args
        }

        for completed_fits, future in enumerate(as_completed(future_to_col), start=1):
            col_idx, popt, errors, success = future.result()
            status[col_idx] = success

            if popt is not None:
                fit_val[col_idx, 0, fit_flag] = popt
                fit_val[col_idx, 1, fit_flag] = errors
                fit_val[col_idx, 0, fix_flag] = full_p0[fix_flag]
                fit_val[col_idx, 1, fix_flag] = 0
            else:
                fit_val[col_idx, 0, :] = np.mean(bounds, axis=0)
                fit_val[col_idx, 0, fix_flag] = full_p0[fix_flag]
                fit_val[col_idx, 1, :] = 0

            if completed_fits % max(1, num_qvals // 10) == 0:
                progress = (completed_fits / num_qvals) * 100
                logger.debug(
                    f"Parallel fitting progress: {progress:.1f}% ({completed_fits}/{num_qvals})"
                )

    # Evaluating the model on fit_x is cheap; do it in-process (a local closure
    # could not be sent to a process pool).
    fit_line = np.zeros((num_qvals, len(fit_x)))
    for n in range(num_qvals):
        fit_line[n] = ensure_numpy(base_func(fit_x, *fit_val[n, 0, :]))

    logger.info(f"Parallel G2 fitting completed for {num_qvals} q-values")
    if return_status:
        return fit_line, fit_val, status
    return fit_line, fit_val


def sequential_fitting(
    func: Callable[..., NDArray[np.floating[Any]]],
    x: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    sigma: NDArray[np.floating[Any]] | None = None,
    p0: NDArray[np.floating[Any]] | None = None,
    bounds: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]] | None = None,
    **kwargs: Any,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], str]:
    """Robust fitting using NLSQ 0.6.0 with multistart and fallback strategies.

    Replaces the legacy TRF → LM → DE chain with a single NLSQ call that handles
    fallback automatically.
    """
    x = ensure_numpy(x)
    y = ensure_numpy(y)
    sigma = ensure_numpy(sigma) if sigma is not None else None

    # Determine n_params from bounds or p0 (more reliable than inspecting wrapper func)
    if bounds is not None:
        n_params = len(bounds[0])
    elif p0 is not None:
        n_params = len(p0)
    else:
        # Last resort: try to infer from function signature
        # Note: This may fail for wrapper functions with *args
        try:
            n_params = func.__code__.co_argcount - 1
            if n_params <= 0:
                n_params = 1  # Minimum 1 parameter
        except AttributeError:
            n_params = 1

    # Filter kwargs for nlsq compatibility
    safe_kwargs = {k: v for k, v in kwargs.items() if k not in ["max_nfev", "maxfev"]}

    popt: NDArray[np.floating[Any]] | None = None
    pcov: NDArray[np.floating[Any]] | None = None

    try:
        from typing import Any as TypingAny

        result: TypingAny = curve_fit(
            func,
            x,
            y,
            sigma=sigma,
            p0=p0,
            bounds=bounds,
            multistart=True,
            fallback=True,
            stability="auto",
            **safe_kwargs,
        )
        if hasattr(result, "popt"):
            popt, pcov = np.asarray(result.popt), np.asarray(result.pcov)
        else:
            popt, pcov = np.asarray(result[0]), np.asarray(result[1])

        # Validate that popt has the expected shape
        if popt.shape == (n_params,) and np.all(np.isfinite(popt)):
            # An unavailable covariance is reported as NaN, never a made-up one.
            unavailable_pcov = np.full((n_params, n_params), np.nan)
            if not getattr(result, "success", True):
                logger.debug("NLSQ fitting stopped without converging")
                return popt, unavailable_pcov, "nlsq_not_converged"
            if pcov.shape == (n_params, n_params) and np.all(np.isfinite(pcov)):
                logger.debug("NLSQ multistart fitting succeeded")
                return popt, pcov, "nlsq_multistart"
            logger.debug("NLSQ fitting: popt valid but pcov unavailable")
            # non-finite entries stay as they are; _param_errors turns them to NaN
            return (
                popt,
                pcov if pcov.shape == (n_params, n_params) else unavailable_pcov,
                "nlsq_partial",
            )

    except Exception as e:
        logger.debug(f"NLSQ fitting failed: {e}")

    # Fallback: construct valid parameters from bounds or p0
    logger.warning("All fitting methods failed, using fallback parameters")

    if p0 is not None:
        fallback_popt = np.asarray(p0).flatten()
        if fallback_popt.shape[0] != n_params:
            # p0 shape mismatch, use bounds mean
            if bounds is not None:
                fallback_popt = np.mean(bounds, axis=0)
            else:
                fallback_popt = np.ones(n_params)
    elif bounds is not None:
        fallback_popt = np.mean(bounds, axis=0)
    else:
        fallback_popt = np.ones(n_params)

    # Final shape validation
    fallback_popt = np.atleast_1d(fallback_popt)
    if fallback_popt.shape[0] != n_params:
        logger.warning(
            f"Fallback popt shape mismatch: got {fallback_popt.shape}, expected ({n_params},)"
        )
        fallback_popt = np.ones(n_params)

    # "fallback" is not a fit: callers must treat it as failed (no covariance).
    return fallback_popt, np.full((n_params, n_params), np.nan), "fallback"


# sequential_fitting methods whose popt came from a converged solve.
_CONVERGED_METHODS = frozenset({"nlsq_multistart", "nlsq_partial"})


@log_timing(threshold_ms=500)
def fit_with_fixed_sequential(
    base_func: Callable[..., NDArray[np.floating[Any]]],
    x: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    sigma: NDArray[np.floating[Any]],
    bounds: NDArray[np.floating[Any]],
    fit_flag: NDArray[np.bool_],
    fit_x: NDArray[np.floating[Any]],
    p0: NDArray[np.floating[Any]] | None = None,
    return_status: bool = False,
    **kwargs: Any,
) -> tuple[Any, ...]:
    """Enhanced fitting with sequential method approach.

    Returns ``(fit_line, fit_val, fit_methods)``, or ``(fit_line, fit_val,
    fit_methods, success)`` with ``return_status``. ``success`` follows the same
    rules as ``fit_with_fixed``: False for too few observations, a solver failure
    or a non-converged solve. Unavailable covariance gives NaN errors; failed
    columns hold bounds-midpoint placeholders with zero errors (not a result).
    """
    # Ensure numpy arrays
    x = ensure_numpy(x)
    y = ensure_numpy(y)
    sigma = ensure_numpy(sigma)
    fit_x = ensure_numpy(fit_x)

    fit_flag = np.asarray(fit_flag, dtype=bool)
    fix_flag = np.logical_not(fit_flag)
    bounds = np.asarray(bounds)

    num_args = len(fit_flag)
    bounds_fit = bounds[:, fit_flag]

    # Preserve full initial guess for fixed parameter values
    full_p0 = _resolve_full_p0(p0, bounds, fit_flag)
    p0 = full_p0[fit_flag]

    fit_val = np.zeros((y.shape[1], 2, num_args))
    fit_methods = []

    # Fixed parameter values: use initial guess (not upper bound)
    fixed_values_seq = full_p0.tolist()
    fit_indices_seq = [i for i in range(num_args) if fit_flag[i]]

    def wrapper_func(x_data, *fit_params):
        full_params = list(fixed_values_seq)
        for idx, val in zip(fit_indices_seq, fit_params):
            full_params[idx] = val
        return base_func(x_data, *full_params)

    # Compute expected number of fit parameters (True values in fit_flag)
    n_fit_params = int(np.sum(fit_flag))

    success = np.zeros(y.shape[1], dtype=bool)

    for n in range(y.shape[1]):
        sigma_col = sigma[:, n] if sigma.ndim > 1 else sigma
        # (popt, errors) hold a solver iterate; None -> bounds-midpoint placeholder
        popt: NDArray[np.floating[Any]] | None = None
        errors: NDArray[np.floating[Any]] | None = None
        try:
            n_obs = int(np.count_nonzero(np.isfinite(y[:, n]) & np.isfinite(x)))
            if n_obs < n_fit_params:
                method_used = "fallback_underdetermined"
                logger.warning(
                    f"Column {n}: {n_obs} observations cannot determine "
                    f"{n_fit_params} free parameters"
                )
            else:
                popt, pcov, method_used = sequential_fitting(
                    wrapper_func,
                    x,
                    y[:, n],
                    sigma=sigma_col,
                    p0=p0,
                    bounds=(bounds_fit[0], bounds_fit[1]),
                    max_nfev=5000,
                )
                popt = np.atleast_1d(popt)
                if popt.shape[0] != n_fit_params:
                    logger.warning(
                        f"Column {n}: popt shape mismatch - got {popt.shape}, "
                        f"expected ({n_fit_params},)."
                    )
                    method_used = "fallback_shape_mismatch"
                elif method_used in _CONVERGED_METHODS | {"nlsq_not_converged"}:
                    errors = _param_errors(pcov, n_fit_params)
                    success[n] = method_used in _CONVERGED_METHODS
                if errors is None:
                    popt = None
        except Exception as e:
            logger.warning(f"Sequential fitting failed for column {n}: {e}")
            method_used = "fallback_error"
            popt = errors = None

        fit_methods.append(method_used)
        logger.debug(f"Column {n}: fitted using {method_used}")

        if popt is not None and errors is not None:
            fit_val[n, 0, fit_flag] = popt
            fit_val[n, 1, fit_flag] = errors
            fit_val[n, 0, fix_flag] = full_p0[fix_flag]
            fit_val[n, 1, fix_flag] = 0
        else:
            fit_val[n, 0, :] = np.mean(bounds, axis=0)
            fit_val[n, 0, fix_flag] = full_p0[fix_flag]
            fit_val[n, 1, :] = 0

    fit_line = np.zeros((y.shape[1], len(fit_x)))
    for n in range(y.shape[1]):
        fit_line[n] = ensure_numpy(base_func(fit_x, *fit_val[n, 0, :]))

    method_counts: dict[str, int] = {}
    for method in fit_methods:
        method_counts[method] = method_counts.get(method, 0) + 1
    logger.info(f"Fitting methods used: {method_counts}")

    if return_status:
        return fit_line, fit_val, fit_methods, success
    return fit_line, fit_val, fit_methods


def robust_curve_fit(
    func: Callable[..., NDArray[np.floating[Any]]],
    x: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    **kwargs: Any,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Simple wrapper around nlsq.curve_fit with error handling.

    On failure (solver error or non-convergence) returns all-NaN ``popt`` and
    ``pcov`` of the right size, never made-up parameters or covariance.
    """
    x = ensure_numpy(x)
    y = ensure_numpy(y)
    try:
        result: Any = curve_fit(func, x, y, **kwargs)
        if not getattr(result, "success", True):
            raise RuntimeError(f"did not converge: {getattr(result, 'message', '')}")
        if hasattr(result, "popt"):
            return np.asarray(result.popt), np.asarray(result.pcov)
        return np.asarray(result[0]), np.asarray(result[1])
    except Exception as e:
        logger.warning(f"Curve fitting failed: {e}")
        p0 = kwargs.get("p0")
        try:
            n_params = len(p0) if p0 is not None else func.__code__.co_argcount - 1
        except (AttributeError, TypeError):
            n_params = 1
        return np.full(n_params, np.nan), np.full((n_params, n_params), np.nan)


def vectorized_parameter_estimation(
    x: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    model_type: str = "exponential",
) -> tuple | None:
    """Vectorized parameter estimation."""
    x, y = ensure_numpy(x), ensure_numpy(y)
    if model_type != "exponential":
        return None

    try:
        y_min, y_max = np.min(y), np.max(y)
        amp = y_max - y_min
        idx = np.argmin(np.abs(y - (y_min + amp / np.e)))
        tau = x[idx] if idx > 0 else x[len(x) // 2]

        # NLSQ JIT-traces model functions with JAX internally, so the
        # model must use jax.numpy ops regardless of the xpcsviewer backend.
        import jax.numpy as jnp

        def _model_func(x_val, tau_val, bkg_val, cts_val):
            return cts_val * jnp.exp(-2 * jnp.asarray(x_val) / tau_val) + bkg_val

        result = curve_fit(
            _model_func,
            x,
            y,
            p0=[tau, y_min, amp],
            bounds=(
                [x[1] * 0.1, -np.abs(y_max), amp * 0.1],
                [x[-1] * 10, y_max * 1.1, amp * 10],
            ),
            method="trf",
            maxfev=5000,
        )
        if not getattr(result, "success", True):
            return None
        if hasattr(result, "popt"):
            popt = cast(Any, result).popt
        else:
            popt = result[0]
        return tuple(popt)
    except Exception:
        return None


def vectorized_residual_analysis(
    x: NDArray[np.floating[Any]],
    y_true: NDArray[np.floating[Any]],
    y_pred: NDArray[np.floating[Any]],
) -> dict[str, float | NDArray[np.floating[Any]]]:
    """Vectorized residual analysis."""
    y_true = ensure_numpy(y_true)
    y_pred = ensure_numpy(y_pred)
    residuals = y_true - y_pred

    return {
        "mean_residual": np.mean(residuals),
        "std_residual": np.std(residuals),
        "rmse": np.sqrt(np.mean(residuals**2)),
        "mae": np.mean(np.abs(residuals)),
    }
