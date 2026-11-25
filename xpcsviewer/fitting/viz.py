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

from xpcsviewer.fileIO.qmap_utils import Q_UNIT_DISPLAY, q_axis_label
from xpcsviewer.utils.export_guard import ExportGuard

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
    # the plotted values are the file's own: name its axis and unit, never assume
    bin_axes = bayesian_summary.get("bin_axes")
    name, unit = (
        (bin_axes["names"][0], bin_axes["units"][0])
        if bin_axes is not None
        else ("Q", Q_UNIT_DISPLAY)
    )
    fig.colorbar(sm, ax=ax, pad=0.02).set_label(q_axis_label(name, unit, mathtext=True))

    fig.tight_layout()
    return fig


_AXIS_COLUMNS = ("axis0_name", "axis0_unit", "axis1_name", "axis1_value", "axis1_unit")


def export_bayesian_csv(
    path: Path | str,
    fit_val: NDArray,
    q_val: NDArray,
    fit_func_name: str,
    *,
    failed_mask: NDArray | None = None,
    nonconverged_mask: NDArray | None = None,
    bin_ids: Any = None,
    bin_phi: Any = None,
    bin_axes: dict[str, Any] | None = None,
    placeholder_q: bool = False,
    guard: ExportGuard | None = None,
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
    bin_ids, bin_phi : sequence or None
        Absolute bin ID and numeric azimuth (the map's angle unit) of each
        row. Two bins can share a radial Q, so they are appended as
        ``bin_id`` / ``phi`` columns after the existing ones (an empty
        ``phi`` cell: the map has no azimuth for that bin).
    bin_axes : dict or None
        ``bayesian_assembly.resolve_bin_axes`` of the file. Appends, after every
        other column, ``axis0_name, axis0_unit, axis1_name, axis1_value,
        axis1_unit``: the file's own axis names and units and the value on its
        second axis. ``q_value`` is the value on the FIRST axis, whatever that
        axis is (``x`` in px for a Cartesian map); ``phi`` stays the azimuth of
        a (q, phi) map only.
    guard : ExportGuard or None
        The export action's write gate; ``path`` is claimed through it before
        anything is written (raises ``ExportCollisionError``).
    """
    cols = _SINGLE_COLS if fit_func_name == "single" else _DOUBLE_COLS
    path = Path(path)
    (guard if guard is not None else ExportGuard()).claim(path, "Bayesian CSV")

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["q_value"]
        for col in cols:
            header.extend([f"{col}_mean", f"{col}_std"])
        if failed_mask is not None:
            header.append("status")
        if bin_ids is not None:
            header.append("bin_id")
        if bin_phi is not None:
            header.append("phi")
        if bin_axes is not None:
            header.extend(_AXIS_COLUMNS)
        if placeholder_q:  # Q values come from the substituted default Q map
            header.append("q_map_placeholder")
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
            if bin_ids is not None:
                row.append(str(int(bin_ids[qi])))
            if bin_phi is not None:
                phi = float(bin_phi[qi])
                row.append(f"{phi:.6g}" if np.isfinite(phi) else "")
            if bin_axes is not None:
                names, units = bin_axes["names"], bin_axes["units"]
                v1 = float(bin_axes["axis1"][qi])
                two = len(names) > 1
                row.extend(
                    [
                        names[0],
                        units[0],
                        names[1] if two else "",
                        f"{v1:.6g}" if two and np.isfinite(v1) else "",
                        units[1] if two else "",
                    ]
                )
            if placeholder_q:
                row.append("1")
            writer.writerow(row)

    logger.info("Exported Bayesian parameters to %s", path)


def _tagged(idata: Any, attrs: dict[str, Any]) -> Any:
    """A copy of ``idata`` whose root attributes carry ``attrs`` (the caller's
    FitResult is not modified)."""
    try:
        tagged = idata.copy()
        tagged.attrs.update(attrs)
    except AttributeError:  # shortcut: object without copy()/attrs (test doubles)
        return idata
    return tagged


def _is_unmeasured(fr: Any, q_idx: int, mask: NDArray | None) -> bool:
    """True if bin ``q_idx`` is not a measurement: the summary's mask says so
    or, without a mask, the result's own diagnostics did not converge."""
    if mask is not None:
        return q_idx >= mask.size or bool(mask[q_idx])
    diag = getattr(fr, "diagnostics", None)
    return diag is not None and not diag.converged


def _is_assessed_ok(fr: Any, mask: NDArray | None) -> bool:
    diag = getattr(fr, "diagnostics", None)
    return mask is not None or (diag is not None and diag.converged)


def export_bayesian_diagnostics(
    path: Path | str,
    bayesian_results: dict[int, Any],
    *,
    failed_mask: NDArray | None = None,
    bin_ids: Any = None,
    q_val: Any = None,
    bin_phi: Any = None,
    bin_axes: dict[str, Any] | None = None,
    source: str | None = None,
    placeholder_q: bool = False,
    guard: ExportGuard | None = None,
) -> list[Path]:
    """Export ArviZ DataTree to netCDF for the measured Q-bins.

    The grouped netCDF4 write needs the optional ``netCDF4`` or ``h5netcdf``
    package (neither is a declared dependency). Without one the format is
    skipped with a warning instead of raising, so the caller's other formats
    are unaffected.

    Parameters
    ----------
    path : Path
        Output netCDF file path.
    bayesian_results : dict[int, FitResult]
        Per-Q FitResult objects with arviz_data attribute, keyed by the
        column index of the summary they belong to.
    failed_mask : ndarray or None
        The summary's ``failed_mask`` (True = not a measurement). Bins it marks
        (or that lie beyond it) are not written: failed and non-converged
        posteriors stay available in the diagnosis views only. Without a mask a
        result whose own diagnostics did not converge is skipped the same way.
    bin_ids, q_val, bin_phi, bin_axes, source : optional
        Identity of each written bin, recorded as root attributes of its file
        together with ``bin_status`` (``ok``; ``unassessed`` without a mask)
        and ``q_index``. ``bin_axes`` (``resolve_bin_axes``) adds the file's own
        ``axis0_name`` / ``axis0_unit`` and, for a two-axis map, ``axis1_name`` /
        ``axis1_unit`` / ``axis1_value`` (``q_value`` is the first-axis value).
    guard : ExportGuard or None
        The export action's write gate; every file is claimed before the
        first is written.

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

    mask = None if failed_mask is None else np.asarray(failed_mask, dtype=bool)
    datasets = {}
    assessed: dict[int, bool] = {}
    skipped = 0
    for q_idx, fr in sorted(bayesian_results.items()):
        if fr is None or getattr(fr, "arviz_data", None) is None:
            continue
        if _is_unmeasured(fr, q_idx, mask):
            skipped += 1
            continue
        datasets[q_idx] = fr.arviz_data
        assessed[q_idx] = _is_assessed_ok(fr, mask)
    if skipped:
        logger.info("netCDF export: skipped %d failed/non-converged Q-bins", skipped)

    if not datasets:
        logger.warning("No ArviZ data available for export")
        return []

    # One netCDF file per Q-bin to preserve all diagnostics; claim them all
    # before the first write so a refusal leaves no partial set.
    guard = guard if guard is not None else ExportGuard()
    plan = []
    for q_idx, idata in sorted(datasets.items()):
        q_path = path.with_stem(f"{path.stem}_q{q_idx:03d}")
        guard.claim(q_path, "Bayesian diagnostics netCDF")
        attrs: dict[str, Any] = {
            "bin_status": "ok" if assessed[q_idx] else "unassessed",
            "q_index": int(q_idx),
        }
        if bin_ids is not None and q_idx < len(bin_ids):
            attrs["bin_id"] = int(bin_ids[q_idx])
        if q_val is not None and q_idx < len(q_val):
            attrs["q_value"] = float(q_val[q_idx])
        if bin_phi is not None and q_idx < len(bin_phi) and np.isfinite(bin_phi[q_idx]):
            attrs["phi"] = float(bin_phi[q_idx])
        if placeholder_q:
            attrs["q_map_is_placeholder"] = 1
        if bin_axes is not None and q_idx < len(bin_axes["axis1"]):
            names, units = bin_axes["names"], bin_axes["units"]
            attrs["axis0_name"], attrs["axis0_unit"] = names[0], units[0]
            if len(names) > 1:
                attrs["axis1_name"], attrs["axis1_unit"] = names[1], units[1]
                v1 = float(bin_axes["axis1"][q_idx])
                if np.isfinite(v1):
                    attrs["axis1_value"] = v1
        if source:
            attrs["source_file"] = str(source)
        plan.append((q_path, _tagged(idata, attrs)))

    written: list[Path] = []
    for q_path, tagged in plan:
        tagged.to_netcdf(str(q_path))
        written.append(q_path)

    logger.info(
        "Exported ArviZ diagnostics to %s (%d Q-bin files)", path.parent, len(written)
    )
    return written
