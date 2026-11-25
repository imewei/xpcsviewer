"""Assemble Bayesian fitting results into NLSQ-compatible fit_summary format.

Converts a dict of per-Q FitResult objects into the same ``fit_summary``
dict that :meth:`XpcsFile.fit_g2` produces, enabling seamless integration
with the tau-q diffusion pipeline and GUI plotting.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .results import FitResult

logger = logging.getLogger(__name__)


def _frozen(a: Any) -> NDArray:
    """Private read-only float copy, so later edits cannot alter a bundle."""
    out = np.array(a, dtype=float, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True, eq=False)  # identity semantics: fields hold arrays
class BayesianBundle:
    """A posterior together with the exact data and bin it was fitted to.

    Created when a fit is launched (``result is None``) and completed with
    :meth:`with_result` when it finishes, so diagnosis and plotting always
    show the stored observations with the stored posterior and never
    re-extract current data.  ``bin_id`` is the absolute (physical) bin
    identity, not a position within the currently selected Q range.
    """

    fname: str | None
    bin_id: Hashable
    label: str
    q_value: float
    model_name: str
    x: NDArray
    y: NDArray
    yerr: NDArray | None
    result: Any = None
    # Q range the observations were selected with (diffusion fits); None for G2.
    q_range: tuple[float, float] | None = None
    # (name, unit) of the file axis ``q_value`` is on (``map_axis``); None: unknown
    q_axis: tuple[str, str] | None = None

    @property
    def q_text(self) -> str:
        """``q_value`` with the file's own axis name and unit, for messages and titles."""
        if self.q_axis is None:
            return f"Q={self.q_value:.4g}"
        return f"{self.q_axis[0]}={self.q_value:.4g} {self.q_axis[1]}"

    @classmethod
    def create(
        cls,
        fname: str | None,
        bin_id: Hashable,
        label: str,
        q_value: float,
        model_name: str,
        x: Any,
        y: Any,
        yerr: Any = None,
        q_range: tuple[float, float] | None = None,
        q_axis: tuple[str, str] | None = None,
    ) -> BayesianBundle:
        return cls(
            fname,
            bin_id,
            str(label),
            float(q_value),
            model_name,
            _frozen(x),
            _frozen(y),
            None if yerr is None else _frozen(yerr),
            q_range=None if q_range is None else (float(q_range[0]), float(q_range[1])),
            q_axis=None if q_axis is None else (str(q_axis[0]), str(q_axis[1])),
        )

    @property
    def key(self) -> tuple[str | None, Hashable]:
        return (self.fname, self.bin_id)

    def with_result(self, result: Any) -> BayesianBundle:
        return replace(self, result=result)


def resolve_g2_bins(xf: Any, q_range: Any) -> list[tuple[int, float, str]]:
    """Absolute ``(bin_id, q, label)`` of each G2 column for *q_range*.

    Mirrors the column selection of ``XpcsFile.get_g2_data`` so that column
    ``j`` of the returned G2 matrix is the physical bin ``result[j]``.
    """
    sel, qvals = xf.qmap.get_qbin_in_qrange(q_range, zero_based=True)
    sel = [int(i) for i in sel]
    atype = xf.atype
    if "Multitau" in atype and hasattr(xf, "g2") and hasattr(xf, "g2_err"):
        hi = min(xf.g2.shape[1], xf.g2_err.shape[1]) - 1
        keep = [0 <= i <= hi for i in sel]
    elif "Twotime" in atype and hasattr(xf, "c2_g2"):
        processed = getattr(xf, "c2_processed_bins", None)
        if processed is None:
            cols = set(range(xf.c2_g2.shape[1]))
        else:
            n = xf.c2_g2.shape[1]
            cols = {int(b) - 1 for c, b in enumerate(np.ravel(processed)) if c < n}
        keep = [i in cols for i in sel]
    else:
        keep = [True] * len(sel)
    return [
        (i, float(qvals[k]), xf.qmap.get_qbin_label(i + 1))
        for k, (i, ok) in enumerate(zip(sel, keep, strict=True))
        if ok
    ]


def _second_axis_values(xf: Any, bin_ids: Any) -> NDArray:
    """Value on the map's second axis of each dynamic bin (NaN where unknown).

    Dynamic bin ``b`` is the file's G2 column; its absolute position on the map
    is ``dynamic_index_mapping[b]`` and, for ``dynamic_num_pts = (n0, n1)``, it
    sits at ``(a // n1, a % n1)`` with the value ``dplist[a % n1]``. A file
    without a mapping is read as the identity.
    """
    out = np.full(len(bin_ids), np.nan)
    try:
        qmap = xf.qmap
        n1 = int(qmap.dynamic_num_pts[1])
        values = np.asarray(qmap.dplist, dtype=float).ravel()
        mapping = getattr(qmap, "dynamic_index_mapping", None)
        mapping = None if mapping is None else np.asarray(mapping).ravel()
    except (AttributeError, IndexError, TypeError, ValueError):
        return out
    for k, bin_id in enumerate(bin_ids):
        absolute = int(bin_id)
        if mapping is not None:
            if not 0 <= absolute < mapping.size:
                continue
            absolute = int(mapping[absolute])
        j = absolute % n1 if n1 > 0 else -1
        if 0 <= j < values.size:
            out[k] = values[j]
    return out


def resolve_bin_phi(xf: Any, bin_ids: Any) -> NDArray:
    """Numeric azimuth of each dynamic bin (NaN where the map has none).

    Only a ``(q, phi)`` map has an azimuth; for any other map (e.g. Cartesian
    ``x`` / ``y``) the result is NaN and :func:`resolve_bin_axes` carries the
    second axis. The value is ``dplist[a % nphi]`` for the bin's absolute map
    position ``a`` (``dynamic_index_mapping``), in the map's own angle unit
    (degrees for APS files).
    """
    try:
        if list(xf.qmap.map_names)[:2] != ["q", "phi"]:
            return np.full(len(bin_ids), np.nan)
    except (AttributeError, TypeError):
        return np.full(len(bin_ids), np.nan)
    return _second_axis_values(xf, bin_ids)


def resolve_bin_axes(xf: Any, bin_ids: Any) -> dict[str, Any] | None:
    """Names, units and second-axis values of the dynamic bins of ``xf``.

    ``{"names": (n0[, n1]), "units": (u0[, u1]), "axis1": values}``: the file's own
    axis names and units (``map_axis``) and, per bin, the value on the second
    axis (the first axis is the ``q_value`` the writers already carry). A map
    with one axis (one point on axis 1) has one name and unit and NaN values.
    ``None`` for a stand-in without a Q map.
    """
    from xpcsviewer.fileIO.qmap_utils import map_axis

    qmap = getattr(xf, "qmap", None)
    if qmap is None:
        return None
    try:
        two_axes = int(qmap.dynamic_num_pts[1]) != 1
    except (AttributeError, IndexError, TypeError, ValueError):
        two_axes = False
    axes = [map_axis(qmap, i) for i in range(2 if two_axes else 1)]
    values = (
        _second_axis_values(xf, bin_ids) if two_axes else np.full(len(bin_ids), np.nan)
    )
    return {
        "names": tuple(a[0] for a in axes),
        "units": tuple(a[1] for a in axes),
        "axis1": values,
    }


# Legacy parameter order for single_exp_all(x, a, b, c, d):
#   idx 0 = a = contrast
#   idx 1 = b = tau
#   idx 2 = c = stretching (fixed 1.0)
#   idx 3 = d = baseline
_SINGLE_EXP_NPARAMS = 4

# Legacy parameter order for double_exp_all(x, a, b, c, d, e, f, g):
#   idx 0 = contrast1
#   idx 1 = tau1
#   idx 2 = stretching1 (fixed 1.0)
#   idx 3 = baseline
#   idx 4 = tau2
#   idx 5 = contrast2
#   idx 6 = stretching2 (fixed 1.0)
_DOUBLE_EXP_NPARAMS = 7


def _extract_single_exp_params(
    fit_result: FitResult,
) -> tuple[NDArray, NDArray]:
    """Extract single-exp params as (values, errors) in legacy order."""
    vals = np.zeros(_SINGLE_EXP_NPARAMS)
    errs = np.zeros(_SINGLE_EXP_NPARAMS)

    vals[0] = fit_result.get_mean("contrast")
    vals[1] = fit_result.get_mean("tau")
    vals[2] = 1.0  # stretching exponent fixed
    vals[3] = fit_result.get_mean("baseline")

    errs[0] = fit_result.get_std("contrast")
    errs[1] = fit_result.get_std("tau")
    errs[2] = 0.0
    errs[3] = fit_result.get_std("baseline")

    return vals, errs


def _extract_double_exp_params(
    fit_result: FitResult,
) -> tuple[NDArray, NDArray]:
    """Extract double-exp params as (values, errors) in legacy order."""
    vals = np.zeros(_DOUBLE_EXP_NPARAMS)
    errs = np.zeros(_DOUBLE_EXP_NPARAMS)

    vals[0] = fit_result.get_mean("contrast1")
    vals[1] = fit_result.get_mean("tau1")
    vals[2] = 1.0  # stretching exponent fixed
    vals[3] = fit_result.get_mean("baseline")
    vals[4] = fit_result.get_mean("tau2")
    vals[5] = fit_result.get_mean("contrast2")
    vals[6] = 1.0  # stretching exponent fixed

    errs[0] = fit_result.get_std("contrast1")
    errs[1] = fit_result.get_std("tau1")
    errs[2] = 0.0
    errs[3] = fit_result.get_std("baseline")
    errs[4] = fit_result.get_std("tau2")
    errs[5] = fit_result.get_std("contrast2")
    errs[6] = 0.0

    return vals, errs


def _compute_fit_line(
    model_func: Any,
    fit_x: NDArray,
    fit_result: FitResult,
    fit_func_name: str,
) -> NDArray:
    """Compute fitted curve from Bayesian posterior means."""
    if fit_func_name == "single":
        return model_func(
            fit_x,
            fit_result.get_mean("tau"),
            fit_result.get_mean("baseline"),
            fit_result.get_mean("contrast"),
        )
    return model_func(
        fit_x,
        fit_result.get_mean("tau1"),
        fit_result.get_mean("tau2"),
        fit_result.get_mean("baseline"),
        fit_result.get_mean("contrast1"),
        fit_result.get_mean("contrast2"),
    )


def assemble_fit_summary(
    results: dict[int, FitResult | None],
    q_arr: NDArray,
    t_el: NDArray,
    fit_func_name: str,
    model_func: Any,
    *,
    q_range: str = "",
    t_range: str = "",
    bounds: Any = None,
    fit_flag: str = "",
    label: str = "",
    fname: str | None = None,
    bin_ids: Any = None,
    fit_q_range: tuple[float, float] | None = None,
    fit_t_range: tuple[float, float] | None = None,
    obs_g2: Any = None,
    obs_t_el: Any = None,
    bin_phi: Any = None,
    bin_axes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert per-Q Bayesian results into NLSQ-compatible fit_summary.

    Parameters
    ----------
    results : dict[int, FitResult | None]
        Mapping of Q-index to FitResult (None for failed Q-bins).
    q_arr : ndarray
        Array of Q values (1D).
    t_el : ndarray
        Array of delay times (1D).
    fit_func_name : str
        ``"single"`` or ``"double"``.
    model_func : callable
        Bayesian model function (``single_exp_func`` or ``double_exp_func``).
    q_range, t_range, bounds, fit_flag, label : optional
        Metadata fields matching the NLSQ fit_summary format.
    bin_axes : dict or None
        ``resolve_bin_axes`` of the file: its two axis names, units and the
        second-axis value of every column. Stored as ``bin_axes`` so the
        writers can name the coordinates of a map that is not ``(q, phi)``.
    fname, bin_ids, bin_phi, fit_q_range, fit_t_range, obs_g2, obs_t_el : optional
        Provenance that makes the summary self-contained: the file, the
        absolute bin ID and numeric azimuth of every column, the Q/delay
        ranges of the fit and
        the observations (G2 matrix and delays) that were fitted.  Plotting
        uses these instead of re-extracting current data.

    Returns
    -------
    dict
        fit_summary dict with keys: ``fit_func``, ``fit_val``, ``t_el``,
        ``q_val``, ``q_range``, ``t_range``, ``bounds``, ``fit_flag``,
        ``fit_line``, ``fit_x``, ``label``, ``failed_mask``,
        ``nonconverged_mask``, ``n_converged``.
        Only Q-bins whose ``FitDiagnostics.converged`` is True are
        measurements. Failed and non-converged Q-bins have NaN in
        ``fit_val``/``fit_line`` and ``failed_mask[q_idx] == True`` (the rule
        ``fit_success_mask`` applies); ``nonconverged_mask`` marks those that
        sampled but did not pass the diagnostics, whose posterior stays on the
        ``FitResult`` for diagnosis. Single-chain runs have no R-hat, so they
        never converge.
    """
    num_q = len(q_arr)
    if bin_ids is not None and len(bin_ids) != num_q:
        raise ValueError(f"bin_ids has {len(bin_ids)} entries for {num_q} Q-bins")
    if bin_phi is not None and len(bin_phi) != num_q:
        raise ValueError(f"bin_phi has {len(bin_phi)} entries for {num_q} Q-bins")
    if bin_axes is not None and len(bin_axes["axis1"]) != num_q:
        raise ValueError(
            f"bin_axes has {len(bin_axes['axis1'])} values for {num_q} Q-bins"
        )
    if obs_g2 is not None and np.shape(obs_g2)[1:2] != (num_q,):
        raise ValueError("obs_g2 columns do not match the Q-bins of the summary")
    nparams = _SINGLE_EXP_NPARAMS if fit_func_name == "single" else _DOUBLE_EXP_NPARAMS
    extract_fn = (
        _extract_single_exp_params
        if fit_func_name == "single"
        else _extract_double_exp_params
    )

    # fit_val shape: [num_q, 2, nparams] — dim 1 is [value, error]
    # NaN default: failed Q-bins stay NaN so downstream isfinite() filters skip them
    fit_val = np.full((num_q, 2, nparams), np.nan)

    # Generate fit_x from t_el range (matching NLSQ behavior with logspace)
    # Cap at 500 points for visualization — keeps main-thread overhead low
    # even for datasets with thousands of time points and many Q-bins
    _FIT_LINE_POINTS = 500
    if len(t_el) == 0:
        fit_x = np.array([])
    elif t_el.min() > 0:
        fit_x = np.logspace(
            np.log10(t_el.min()),
            np.log10(t_el.max()),
            min(_FIT_LINE_POINTS, max(200, len(t_el) * 2)),
        )
    else:
        fit_x = np.linspace(
            t_el.min(), t_el.max(), min(_FIT_LINE_POINTS, max(200, len(t_el) * 2))
        )
    fit_line = np.full((num_q, len(fit_x)), np.nan)

    # failed_mask is True for every bin that is NOT a measurement: no result, a
    # failed extraction, or a posterior whose diagnostics did not pass
    # (nonconverged_mask). A non-converged posterior stays on its FitResult for
    # the diagnosis views; the summary never carries its values as a measurement.
    failed_mask = np.ones(num_q, dtype=bool)
    nonconverged_mask = np.zeros(num_q, dtype=bool)

    converged = 0
    failed = 0
    rhat_unavailable = 0

    for q_idx in range(num_q):
        fr = results.get(q_idx)
        if fr is None:
            failed += 1
            continue

        diag = getattr(fr, "diagnostics", None)
        if diag is None or not diag.converged:
            nonconverged_mask[q_idx] = True
            if diag is not None and not all(
                np.isfinite(v) for v in diag.r_hat.values()
            ):
                rhat_unavailable += 1
            continue

        try:
            vals, errs = extract_fn(fr)
            fit_val[q_idx, 0, :] = vals
            fit_val[q_idx, 1, :] = errs
            fit_line[q_idx, :] = np.asarray(
                _compute_fit_line(model_func, fit_x, fr, fit_func_name)
            )
            failed_mask[q_idx] = False
            converged += 1
        except (KeyError, ValueError) as exc:
            logger.warning("Failed to extract params for Q-index %d: %s", q_idx, exc)
            failed += 1

    nonconverged = int(nonconverged_mask.sum())
    logger.info(
        "Bayesian assembly: %d converged of %d Q-bins, %d not converged "
        "(kept for diagnosis only), %d failed",
        converged,
        num_q,
        nonconverged,
        failed,
    )

    return {
        "source": "bayesian",
        "fit_func": fit_func_name,
        "fit_val": fit_val,
        "t_el": t_el,
        "q_val": np.asarray(q_arr),
        "q_range": q_range,
        "t_range": t_range,
        "bounds": bounds,
        "fit_flag": fit_flag,
        "fit_line": fit_line,
        "fit_x": fit_x,
        "label": label,
        "failed_mask": failed_mask,
        "nonconverged_mask": nonconverged_mask,
        "n_converged": converged,
        "n_rhat_unavailable": rhat_unavailable,
        "fname": fname,
        "bin_ids": None if bin_ids is None else np.asarray(bin_ids),
        "bin_phi": None if bin_phi is None else np.asarray(bin_phi, dtype=float),
        "bin_axes": bin_axes,
        "fit_q_range": fit_q_range,
        "fit_t_range": fit_t_range,
        "obs_g2": None if obs_g2 is None else _frozen(obs_g2),
        "obs_t_el": None if obs_t_el is None else _frozen(obs_t_el),
    }


def convergence_message(fit_summary: dict[str, Any]) -> str:
    """User-facing outcome of a Bayesian batch: how many Q-bins are measurements.

    Counts converged bins ("N converged of M"), never "succeeded", and says why
    single-chain runs are not measurements.
    """
    total = len(fit_summary["failed_mask"])
    n_conv = int(np.sum(~np.asarray(fit_summary["failed_mask"], dtype=bool)))
    msg = f"Bayesian fit: {n_conv} converged of {total} Q-bins"
    n_non = int(np.sum(fit_summary.get("nonconverged_mask", [])))
    if n_non:
        msg += (
            f"; {n_non} not converged (excluded from plots, exports and tau-Q; "
            "kept for diagnosis)"
        )
    if fit_summary.get("n_rhat_unavailable"):
        msg += (
            f"; R-hat unavailable for {fit_summary['n_rhat_unavailable']} "
            "(single-chain runs have no R-hat and cannot count as converged: "
            "use 2 or more chains)"
        )
    return msg
