"""Bayesian all-Q visualization and export utilities.

Provides matplotlib figures for batch Bayesian fitting results
and export to CSV, PDF, and netCDF formats.
"""

from __future__ import annotations

import csv
import importlib.util
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Packages xarray can write grouped NETCDF4 files with; neither is a declared dependency.
_NETCDF_BACKENDS = ("netCDF4", "h5netcdf")

# Column names for single and double exponential legacy fit_val format
_SINGLE_COLS = ("contrast", "tau", "stretching", "baseline")
_DOUBLE_COLS = (
    "contrast1",
    "tau1",
    "stretch1",
    "baseline",
    "tau2",
    "contrast2",
    "stretch2",
)


def plot_bayesian_all_q(
    bayesian_summary: dict[str, Any] | None,
    g2_data: NDArray | None = None,
    *,
    data_t_el: NDArray | None = None,
) -> Figure | None:
    """Generate all-Q overlay figure with Bayesian fit lines.

    Parameters
    ----------
    bayesian_summary : dict or None
        Output of ``assemble_fit_summary`` with ``source='bayesian'``.
    g2_data : ndarray, shape (num_t, num_q) or None
        Raw G2 correlation data.  Ignored when the summary carries the
        observations it was fitted to (``obs_g2``/``obs_t_el``): those are
        always plotted with the summary's own bins and ranges, never other
        data.  Legacy summaries without them use this argument; with neither,
        only the fit lines are drawn.
    data_t_el : ndarray or None
        Time axis matching ``g2_data`` rows.  When the caller applies a
        ``t_range`` filter the resulting array may be shorter than the
        summary's ``t_el``.  Pass the filtered time array here so that
        data points are plotted at the correct times.  Falls back to
        ``bayesian_summary["t_el"]`` when *None*.

    Returns
    -------
    Figure or None
        Matplotlib figure, or None if no summary.
    """
    if bayesian_summary is None:
        return None
    if bayesian_summary.get("obs_g2") is not None:
        g2_data = bayesian_summary["obs_g2"]
        data_t_el = bayesian_summary.get("obs_t_el")

    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    fit_line = bayesian_summary["fit_line"]
    fit_x = bayesian_summary["fit_x"]
    q_val = bayesian_summary["q_val"]
    t_el = data_t_el if data_t_el is not None else bayesian_summary["t_el"]
    failed_mask = bayesian_summary.get("failed_mask")
    num_q = len(q_val)

    fig, ax = plt.subplots(figsize=(10, 7))
    q_min, q_max = q_val.min(), q_val.max()
    if q_min == q_max:
        q_max = q_min + 1.0
    norm = Normalize(vmin=q_min, vmax=q_max)
    cmap = plt.get_cmap("viridis")

    # --- Batch data points with a single scatter call ---
    all_t: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_c: list[np.ndarray] = []
    # no stored observations: draw the fit lines only
    obs = g2_data if g2_data is not None else np.empty((0, 0))
    for qi in range(min(num_q, obs.shape[1])):
        valid = np.isfinite(obs[:, qi])
        n_valid = int(valid.sum())
        if n_valid > 0:
            all_t.append(t_el[valid])
            all_y.append(obs[valid, qi])
            all_c.append(np.full(n_valid, q_val[qi]))

    if all_t:
        t_cat = np.concatenate(all_t)
        y_cat = np.concatenate(all_y)
        c_cat = np.concatenate(all_c)
        ax.scatter(
            t_cat,
            y_cat,
            c=c_cat,
            cmap=cmap,
            norm=norm,
            s=9,
            alpha=0.5,
            edgecolors="none",
            rasterized=True,
        )

    # --- Batch fit lines with a single LineCollection ---
    segments = []
    seg_colors = []
    for qi in range(num_q):
        if failed_mask is not None and failed_mask[qi]:
            continue
        if np.any(np.isfinite(fit_line[qi])):
            pts = np.column_stack([fit_x, fit_line[qi]])
            segments.append(pts)
            seg_colors.append(cmap(norm(q_val[qi])))

    if segments:
        lc = LineCollection(segments, colors=seg_colors, linewidths=1.5)
        ax.add_collection(lc)
        ax.autoscale_view()

    ax.set(xscale="log", xlabel="Delay time (s)", ylabel=r"$g_2(\tau)$")
    title = "All-Q Bayesian Fit Overview"
    if failed_mask is not None:
        title += f"\n{int((~failed_mask).sum())} converged of {num_q} Q-bins"
    ax.set_title(title)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.02).set_label(r"Q ($\AA^{-1}$)")

    fig.tight_layout()
    return fig


def export_bayesian_csv(
    path: Path | str,
    fit_val: NDArray,
    q_val: NDArray,
    fit_func_name: str,
    *,
    failed_mask: NDArray | None = None,
    nonconverged_mask: NDArray | None = None,
) -> None:
    """Export Bayesian fit parameters to CSV.

    Parameters
    ----------
    path : Path
        Output CSV file path.
    fit_val : ndarray, shape (num_q, 2, nparams)
        Parameter values (dim1=0) and std errors (dim1=1).
    q_val : ndarray, shape (num_q,)
        Q values.
    fit_func_name : str
        'single' or 'double'.
    failed_mask : ndarray or None
        Boolean array (True = not a measurement). Adds a ``status`` column
        when provided; only rows with False are written as ``ok``.
    nonconverged_mask : ndarray or None
        Subset of ``failed_mask`` that sampled but did not pass the
        convergence diagnostics; written as ``not_converged`` instead of
        ``failed``.
    """
    cols = _SINGLE_COLS if fit_func_name == "single" else _DOUBLE_COLS
    path = Path(path)

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["q_value"]
        for col in cols:
            header.extend([f"{col}_mean", f"{col}_std"])
        if failed_mask is not None:
            header.append("status")
        writer.writerow(header)

        for qi in range(len(q_val)):
            row = [f"{q_val[qi]:.6f}"]
            for ci in range(len(cols)):
                row.append(f"{fit_val[qi, 0, ci]:.6g}")
                row.append(f"{fit_val[qi, 1, ci]:.6g}")
            if failed_mask is not None:
                if not failed_mask[qi]:
                    row.append("ok")
                elif nonconverged_mask is not None and nonconverged_mask[qi]:
                    row.append("not_converged")
                else:
                    row.append("failed")
            writer.writerow(row)

    logger.info("Exported Bayesian parameters to %s", path)


def export_bayesian_diagnostics(
    path: Path | str,
    bayesian_results: dict[int, Any],
) -> list[Path]:
    """Export ArviZ DataTree to netCDF for all Q-bins.

    The grouped netCDF4 write needs the optional ``netCDF4`` or ``h5netcdf``
    package (neither is a declared dependency). Without one the format is
    skipped with a warning instead of raising, so the caller's other formats
    are unaffected.

    Parameters
    ----------
    path : Path
        Output netCDF file path.
    bayesian_results : dict[int, FitResult]
        Per-Q FitResult objects with arviz_data attribute.

    Returns
    -------
    list[Path]
        The netCDF files written; empty when the format was skipped (no
        backend, ArviZ unavailable, or no ArviZ data), so the caller can tell
        the user.
    """
    path = Path(path)

    try:
        import arviz as az  # noqa: F401
    except ImportError:
        logger.warning("ArviZ not available, skipping netCDF export")
        return []

    if not any(importlib.util.find_spec(m) for m in _NETCDF_BACKENDS):
        logger.warning(
            "Skipping netCDF export: no NETCDF4 backend is installed; "
            "install netCDF4 or h5netcdf to enable it"
        )
        return []

    datasets = {}
    for q_idx, fr in sorted(bayesian_results.items()):
        if fr is not None and hasattr(fr, "arviz_data") and fr.arviz_data is not None:
            datasets[q_idx] = fr.arviz_data

    if not datasets:
        logger.warning("No ArviZ data available for export")
        return []

    # Write one netCDF file per Q-bin to preserve all diagnostics
    written: list[Path] = []
    for q_idx, idata in sorted(datasets.items()):
        q_path = path.with_stem(f"{path.stem}_q{q_idx:03d}")
        idata.to_netcdf(str(q_path))
        written.append(q_path)

    logger.info(
        "Exported ArviZ diagnostics to %s (%d Q-bin files)", path.parent, len(written)
    )
    return written
