"""G2 correlation analysis module.

Provides multi-tau correlation analysis with single and double exponential
fitting for XPCS time correlation functions.

Functions:
    get_data: Extract G2 correlation data from XpcsFile objects
    fit_g2: Fit G2 data with exponential models
"""

# Standard library imports
import logging
from collections.abc import Sequence

# Third-party imports
import numpy as np
import pyqtgraph as pg
from numpy.typing import ArrayLike

from xpcsviewer.backends._conversions import ensure_numpy
from xpcsviewer.fileIO.qmap_utils import (
    axis_mismatch,
    convert_q,
    convert_q_range,
    map_axis,
)
from xpcsviewer.plothandler.plot_constants import MATPLOTLIB_COLORS_RGB as colors

# Local imports
from xpcsviewer.utils.logging_config import get_logger

pg.setConfigOption("foreground", pg.mkColor(80, 80, 80))
# pg.setConfigOption("background", 'w')
logger = get_logger(__name__)


def _log_array_dims(prefix: str, arr) -> None:
    """Log array shape/dtype for data flow tracing."""
    if logger.isEnabledFor(logging.DEBUG):
        if arr is None:
            logger.debug(f"{prefix}: None")
        elif hasattr(arr, "shape"):
            logger.debug(f"{prefix}: shape={arr.shape}, dtype={arr.dtype}")
        elif isinstance(arr, (list, tuple)):
            logger.debug(f"{prefix}: list of {len(arr)} items")


# https://www.geeksforgeeks.org/pyqtgraph-symbols/
symbols = ["o", "t", "t1", "t2", "t3", "s", "p", "h", "star", "+", "d", "x"]


def get_positive_delay_extent(
    delays: Sequence[ArrayLike],
) -> tuple[float, float] | None:
    """Return the finite positive delay extrema across per-file arrays."""
    lower, upper = np.inf, -np.inf
    for delay in delays:
        values = np.asarray(delay)
        positive = values[np.isfinite(values) & (values > 0)]
        if positive.size:
            lower = min(lower, float(positive.min()))
            upper = max(upper, float(positive.max()))
    return (lower, upper) if lower <= upper else None


def _shared_q_axis(xf_list) -> tuple[str, str]:
    """``(name, unit)`` of the Q axis every G2 file is shown on: the first file's."""
    return map_axis(getattr(xf_list[0], "qmap", None), 0)


def q_range_in_file_unit(xf, q_range, shared):
    """``q_range`` (given in the ``shared`` axis unit) in ``xf``'s own Q unit.

    ``None`` stays ``None`` and a file already on the shared unit gets the very
    same range back. Raises ``ValueError`` for a file whose dynamic axis has
    another name or is not a reciprocal length convertible to ``shared``.
    """
    axis = map_axis(getattr(xf, "qmap", None), 0)
    why = axis_mismatch(axis, shared)
    if why is not None:
        raise ValueError(f"its dynamic axis: {why}")
    return convert_q_range(q_range, shared[1], axis[1])


def left_out_files(xf_list) -> dict[str, str]:
    """``{label: reason}`` of the files ``get_data`` leaves out of the shared axis.

    The same predicate as ``q_range_in_file_unit``; GUI-free, so the GUI
    boundary (kernel / viewer) can tell the user which files are missing.
    """
    if not xf_list:
        return {}
    shared = _shared_q_axis(xf_list)
    out = {}
    for xf in xf_list:
        why = axis_mismatch(map_axis(getattr(xf, "qmap", None), 0), shared)
        if why is not None:
            out[str(getattr(xf, "label", xf))] = why
    return out


class _BinLabels(list):
    """Display labels of one file's G2 bins plus the numeric key of each bin.

    ``second`` / ``second_axis``: the value of each bin on the map's second axis
    and that axis' ``(name, unit)``, both ``None`` for a one-axis map. Panels are
    matched on these numbers (and the Q array), never on the label text, which
    holds the file's own formatted values and unit.
    """

    second = None
    second_axis = None


def _labelled(labels, fc, own_q_range, shared2, n_cols):
    """``labels`` as ``_BinLabels`` keyed by the bins' second-axis coordinate."""
    try:
        qmap = fc.qmap
        axis = map_axis(qmap, 1)
        out = _BinLabels(labels)
        if int(qmap.dynamic_num_pts[1]) == 1:
            return out
        sel, _ = qmap.get_qbin_in_qrange(own_q_range, zero_based=True)
        dplist = np.asarray(qmap.dplist, dtype=float)
        values = dplist[np.asarray(qmap.dynamic_index_mapping)[sel] % len(dplist)]
        if len(values) != n_cols:
            return labels
        if shared2 is not None and axis[1] != shared2[1] and axis[0] == shared2[0]:
            try:  # e.g. a Cartesian pair in inverse nm next to one in inverse A
                values, axis = convert_q(values, axis[1], shared2[1]), shared2
            except ValueError:
                pass
        out.second, out.second_axis = values, axis
        return out
    except (AttributeError, IndexError, TypeError, ValueError):
        return labels  # not a loaded file: the label text is all there is


def get_data(xf_list, q_range=None, t_range=None):
    """Extract G2 correlation data from a list of XpcsFile objects.

    Reads G2 correlation values, errors, delay times, and Q-values from
    each file, optionally filtered by Q-range and time-range.

    All files are shown on ONE Q axis, the first file's (name and unit, e.g.
    ``q`` in inverse Angstrom). ``q_range`` is read in that unit: it is
    converted into each file's own unit before the file is queried, and the Q
    values returned for the file are converted back. A file whose axis cannot
    be converted (another axis name, or not a reciprocal length) is left out
    with a warning that names it; to keep every list aligned with ``xf_list``
    its entry is empty (no Q bins), which every renderer already draws as
    "nothing".

    Args:
        xf_list: List of XpcsFile objects with correlation analysis data.
        q_range: Optional (min, max) Q-value filter in the first file's Q unit
            (inverse Angstroms for APS files).
        t_range: Optional (min, max) delay time filter in seconds.

    Returns:
        If all files contain correlation data, returns a 5-tuple:
        ``(q, tel, g2, g2_err, labels)`` where each element is a list
        of per-file arrays. Returns ``(False, None, None, None, None)``
        if any file lacks Multitau or Twotime analysis.

    Example:
        >>> q, tel, g2, g2_err, labels = get_data(xf_list, q_range=(0.01, 0.1))
        >>> if q is not False:
        ...     print(f"Loaded {len(q)} files, {g2[0].shape[1]} Q-bins")
    """
    logger.debug(
        f"get_data: entry with {len(xf_list)} files, q_range={q_range}, t_range={t_range}"
    )
    # Early validation - check all files have correlation analysis (Multitau or Twotime)
    analysis_types = [xf.atype for xf in xf_list]
    if not all(
        any(atype_part in ["Multitau", "Twotime"] for atype_part in atype)
        for atype in analysis_types
    ):
        logger.debug("get_data: exit early - no correlation analysis")
        return False, None, None, None, None

    # Pre-allocate lists with known size for better memory efficiency
    num_files = len(xf_list)
    q = [None] * num_files
    tel = [None] * num_files
    g2 = [None] * num_files
    g2_err = [None] * num_files
    labels = [None] * num_files

    shared = _shared_q_axis(xf_list) if num_files else None

    # Process all files - can potentially be parallelized
    for i, fc in enumerate(xf_list):
        try:
            file_q_range = q_range_in_file_unit(fc, q_range, shared)
        except ValueError as exc:
            logger.warning(
                "G2: left out %s: %s (the plot is on %s [%s] of %s)",
                getattr(fc, "label", fc),
                exc,
                shared[0],
                shared[1],
                getattr(xf_list[0], "label", xf_list[0]),
            )
            q[i], tel[i], g2[i], g2_err[i], labels[i] = (
                np.empty(0),
                np.empty(0),
                np.empty((0, 0)),
                np.empty((0, 0)),
                [],
            )
            continue
        _q, _tel, _g2, _g2_err, _labels = fc.get_g2_data(
            qrange=file_q_range, trange=t_range
        )
        own_unit = map_axis(getattr(fc, "qmap", None), 0)[1]
        if own_unit != shared[1]:  # the file's own unit: back to the shared one
            _q = convert_q(_q, own_unit, shared[1])
        q[i] = _q
        tel[i] = _tel
        g2[i] = _g2
        g2_err[i] = _g2_err
        labels[i] = _labelled(
            _labels,
            fc,
            file_q_range,
            map_axis(getattr(xf_list[0], "qmap", None), 1),
            _n_q(_g2),
        )

    _log_array_dims("get_data g2", g2[0] if g2 else None)
    logger.debug(f"get_data: exit with {num_files} datasets")
    return q, tel, g2, g2_err, labels


def get_g2_stability_data(xf_obj, q_range=None, t_range=None):
    """
    Extract G2 stability data from a single XpcsFile object.

    Parameters
    ----------
    xf_obj : XpcsFile
        Single XpcsFile object with g2_partial data
    q_range : tuple or None
        Q-range filter (qmin, qmax)
    t_range : tuple or None
        Time range filter (tmin, tmax)

    Returns
    -------
    tuple
        (q, tel, g2, g2_err, qbin_labels, labels)
    """
    if "Multitau" not in xf_obj.atype:
        return False, None, None, None, None, None

    q, tel, g2, g2_err, qbin_labels, labels = xf_obj.get_g2_stability_data(
        qrange=q_range, trange=t_range
    )
    return q, tel, g2, g2_err, qbin_labels, labels


def pg_plot_stability(
    hdl,
    xf_obj,
    q_range,
    t_range,
    y_range,
    y_auto=False,
    q_auto=False,
    t_auto=False,
    num_col=4,
    offset=0,
    show_label=False,
    plot_type="multiple",
    marker_size=5,
    **kwargs,
):
    """
    Plot G2 stability data showing frame-by-frame correlation analysis.

    Parameters
    ----------
    hdl : GraphicsLayoutWidget
        PyQtGraph layout widget for plotting
    xf_obj : XpcsFile
        XpcsFile object with g2_partial data
    q_range : tuple
        Q-range for filtering
    t_range : tuple
        Time range for filtering
    y_range : tuple
        Y-axis range
    y_auto, q_auto, t_auto : bool
        Auto-range flags
    num_col : int
        Number of columns in layout
    offset : float
        Y-offset between curves
    show_label : bool
        Show legend labels
    plot_type : str
        Plot type: 'multiple', 'single', 'single-combined'
    marker_size : int
        Symbol size
    **kwargs : dict
        Additional plotting parameters
    """
    if q_auto:
        q_range = None
    if t_auto:
        t_range = None
    if y_auto:
        y_range = None

    q, tel, g2, g2_err, qbin_labels, labels = get_g2_stability_data(
        xf_obj, q_range=q_range, t_range=t_range
    )

    # Handle case where data is not available
    if g2 is False or g2 is None:
        logger.warning("G2 stability data not available for this file")
        hdl.clear()  # never leave the previous selection's curves on screen
        return

    num_figs, num_lines = compute_geometry(g2, plot_type)
    if num_figs < 1 or num_lines < 1:
        logger.warning("G2 stability plot skipped: the Q selection has no Q bins")
        hdl.clear()
        return

    num_data, num_qval = len(g2), g2[0].shape[1]
    # col and row for the 2d layout
    col = min(num_figs, num_col)
    row = (num_figs + col - 1) // col

    # Frame indices for color/symbol cycling
    frame_indices = np.arange(num_data)

    hdl.adjust_canvas_size(num_col=col, num_row=row)
    hdl.clear()

    # Handle log scale for time range
    t0_range = None
    if t_range:
        with np.errstate(divide="ignore", invalid="ignore"):
            t_range_positive = np.asarray(t_range)
            t_range_positive = np.where(
                t_range_positive > 0, t_range_positive, np.finfo(float).eps
            )
            t0_range = np.log10(t_range_positive)

    axes = []
    for n in range(num_figs):
        i_col = n % col
        i_row = n // col
        t = hdl.addPlot(row=i_row, col=i_col)
        axes.append(t)
        if show_label:
            legend = t.addLegend(labelTextSize="6pt")
            legend.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(0, 0))

        t.setMouseEnabled(x=False, y=y_auto)
        # Set axis labels once during setup (hoisted out of the m×n inner loop)
        t.setLabel("bottom", "tau (s)")
        t.setLabel("left", "g2")

    color_len = len(colors)
    symbol_len = len(symbols)
    for m in range(num_data):
        # default base line to be 1.0; used for non-fitting or fit error cases
        baseline_offset = np.ones(num_qval)

        color = colors[frame_indices[m] % color_len]
        symbol = symbols[frame_indices[m] % symbol_len]

        for n in range(num_qval):
            label = None
            if plot_type == "multiple":
                ax = axes[n]
                label = f"frame={int(labels[m])}"
                if m == 0:
                    ax.setTitle(qbin_labels[n])
            elif plot_type == "single":
                ax = axes[m]
                # overwrite color; use the same color for the same set;
                color = colors[n % color_len]
                if m == 0 or n == 0:
                    ax.setTitle(str(labels[m]))
            elif plot_type == "single-combined":
                ax = axes[0]
                label = str(labels[m]) + qbin_labels[n]

            x = tel
            # normalize baseline
            y = g2[m][:, n] - baseline_offset[n] + 1.0 + m * offset
            y_err = g2_err[m][:, n]

            pg_plot_one_g2(
                ax,
                x,
                y,
                y_err,
                color,
                label=label,
                symbol=symbol,
                symbol_size=marker_size,
            )
            if not y_auto and y_range is not None:
                ax.setRange(yRange=y_range)
            if not t_auto and t_range is not None:
                ax.setRange(xRange=t0_range)

    return


def compute_geometry(g2, plot_type):
    """Compute subplot grid dimensions for G2 plots.

    Determines how many figures (subplots) and how many overlay
    lines per figure are needed based on the layout mode.  The Q-bin count
    is that of the first file that has Q bins (file 0 whenever it has any),
    so an empty first file does not suppress later valid files.

    Args:
        g2: List of 2-D G2 arrays, each shaped
            ``(n_delay, n_q)``.
        plot_type: Layout mode. ``"multiple"`` creates one subplot
            per Q-bin, ``"single"`` one per file, and
            ``"single-combined"`` puts everything on one axes.

    Returns:
        tuple[int, int]: ``(num_figs, num_lines)`` where *num_figs*
        is the number of subplot panels and *num_lines* is the
        number of curves per panel.

    Raises:
        ValueError: If *plot_type* is not a recognised layout mode.
    """
    ref = _reference_index(g2)
    n_ref = 0 if ref is None else _n_q(g2[ref])
    if plot_type == "multiple":
        num_figs = n_ref
        num_lines = len(g2)
    elif plot_type == "single":
        # one panel per file keeps panel index == file index for the renderers
        num_figs = len(g2)
        num_lines = n_ref
    elif plot_type == "single-combined":
        num_figs = 1
        num_lines = n_ref * len(g2)
    else:
        raise ValueError("plot_type not support.")
    return num_figs, num_lines


def _n_q(arr) -> int:
    """Number of Q-bin columns of one file's G2 array (0 when it has no data)."""
    arr = np.asarray(arr)
    # a non-empty 1-D array is malformed: shape[1] raises IndexError as before
    return 0 if arr.size == 0 else arr.shape[1]


def _reference_index(g2) -> int | None:
    """Index of the first file with Q bins; it defines the reference panels."""
    return next((m for m, arr in enumerate(g2) if _n_q(arr) > 0), None)


# Q identity tolerance: float32-storage noise (~6e-8) passes, while bins that
# merely share a 5-decimal label (0.100001 vs 0.100004) stay apart.
_Q_RTOL = 1e-6


def _q_panel_indices(q, m, n_curves, labels=None, ref=0):
    """Map each curve of file ``m`` to the Q-bin panel defined by file ``ref``.

    A curve belongs to a panel only if it is the same physical bin: its numeric
    Q (in the shared unit; ``get_data`` converts) equals the panel's within
    ``_Q_RTOL`` (relative) AND its value on the map's second axis equals the
    panel's (``_BinLabels.second``, same axis name and unit, relative
    ``_Q_RTOL`` / absolute 1e-9).  Labels are display text in each file's own
    unit and never decide a match, except for plain label lists that carry no
    numeric key (compared for equality as before).  Neither the rounded label
    nor Q alone can alias two bins.  Identical ordered grids map positionally;
    otherwise curves are matched one-to-one in order.  Returns one panel index
    per curve, or ``None`` (after logging a warning) where no free panel
    matches, so a curve is never drawn under another bin.
    """
    if m == ref:
        return list(range(n_curves))
    ref_q = np.asarray(q[ref], dtype=float)
    own = np.asarray(q[m], dtype=float)
    ref_lab = labels[ref] if labels is not None and len(labels) > ref else None
    own_lab = labels[m] if labels is not None and m < len(labels) else None
    use_labels = (
        ref_lab is not None
        and own_lab is not None
        and len(ref_lab) == ref_q.size
        and len(own_lab) == own.size
    )
    numeric = (
        use_labels
        and isinstance(ref_lab, _BinLabels)
        and isinstance(own_lab, _BinLabels)
    )

    def same_second(i, j):
        if not numeric:
            return not use_labels or ref_lab[i] == own_lab[j]
        if ref_lab.second_axis != own_lab.second_axis:
            return False
        if ref_lab.second is None:  # both maps have one axis
            return True
        return bool(
            np.isclose(ref_lab.second[i], own_lab.second[j], rtol=_Q_RTOL, atol=1e-9)
        )

    if (
        own.size == ref_q.size == n_curves
        and np.allclose(ref_q, own, rtol=_Q_RTOL, atol=1e-12)
        and all(same_second(k, k) for k in range(n_curves))
    ):
        return list(range(n_curves))
    taken: set[int] = set()
    out = []
    for j in range(n_curves):
        free = (
            np.flatnonzero(np.isclose(ref_q, own[j], rtol=_Q_RTOL, atol=1e-12))
            if j < own.size
            else []
        )
        free = [i for i in free if same_second(int(i), j)]
        free = [int(i) for i in free if i not in taken]
        if not free:
            logger.warning(
                "G2 curve %d of file %d (Q=%s) has no free panel with matching "
                "Q-bin; skipped",
                j,
                m,
                own[j] if j < own.size else "unknown",
            )
            out.append(None)
        else:
            taken.add(free[0])
            out.append(int(free[0]))
    return out


def pg_plot(
    hdl,
    xf_list,
    q_range,
    t_range,
    y_range,
    y_auto=False,
    q_auto=False,
    t_auto=False,
    num_col=4,
    rows=None,
    offset=0,
    show_fit=False,
    show_label=False,
    bounds=None,
    fit_flag=None,
    plot_type="multiple",
    subtract_baseline=True,
    marker_size=5,
    label_size=4,
    fit_func="single",
    robust_fitting=False,
    enable_diagnostics=False,
    **kwargs,
):
    """Plot G2 correlation data using PyQtGraph with optional curve fitting.

    Renders multi-panel G2 plots with configurable layout, fitting overlays,
    and baseline subtraction. Supports single, double, and stretched
    exponential fitting models.

    Args:
        hdl: PyQtGraph plot handler (GraphicsLayoutWidget).
        xf_list: List of XpcsFile objects containing G2 data.
        q_range: (min, max) Q-value range in inverse Angstroms, or None.
        t_range: (min, max) delay time range in seconds, or None.
        y_range: (min, max) y-axis range for G2 values, or None.
        y_auto: If True, auto-scale y-axis.
        q_auto: If True, ignore q_range and use all Q-bins.
        t_auto: If True, ignore t_range and use all delay times.
        num_col: Number of plot columns in the grid layout.
        rows: List of file indices to plot, or None for all.
        offset: Vertical offset between datasets for visibility.
        show_fit: If True, overlay fitted curves on the data.
        show_label: If True, display legend labels.
        bounds: Fitting bounds array, shape ``(2, n_params)``.
        fit_flag: Boolean array indicating which parameters to fit.
        plot_type: Layout mode: ``'multiple'`` (one panel per Q-bin),
            ``'single'`` (one panel per file), or ``'single-combined'``.
        subtract_baseline: If True, subtract fitted baseline from data.
        marker_size: Size of data point markers in pixels.
        label_size: Font size for legend labels in points.
        fit_func: Fitting model: ``'single'`` or ``'double'`` exponential.
        robust_fitting: If True, use NLSQ multistart robust fitting.
        enable_diagnostics: If True, compute model health diagnostics.
        **kwargs: Additional keyword arguments passed to fitting routines.
            Includes ``force_refit`` (bool) to force re-fitting.

    Example:
        >>> pg_plot(hdl, xf_list, q_range=(0.01, 0.1),
        ...         t_range=(1e-4, 10), y_range=(0.9, 1.5),
        ...         show_fit=True, fit_func='single')
    """
    if q_auto:
        q_range = None
    if t_auto:
        t_range = None
    if y_auto:
        y_range = None

    q, tel, g2, g2_err, labels = get_data(xf_list, q_range=q_range, t_range=t_range)
    num_figs, _num_lines = compute_geometry(g2 or [], plot_type)
    # single/single-combined keep one figure, so also check the curve count
    if num_figs < 1 or _num_lines < 1:
        logger.warning("G2 plot skipped: the Q selection contains no Q bins")
        hdl.clear()  # never leave the previous selection's curves on screen
        return

    num_data = len(g2)
    ref = _reference_index(g2)
    # col and rows for the 2d layout
    col = min(num_figs, num_col)
    row = (num_figs + col - 1) // col

    if rows is None or len(rows) == 0:
        rows = list(range(len(xf_list)))

    hdl.adjust_canvas_size(num_col=col, num_row=row)
    hdl.clear()
    # a bug in pyqtgraph; the log scale in x-axis doesn't apply
    if t_range:
        # Handle log10 of zero or negative values
        with np.errstate(divide="ignore", invalid="ignore"):
            # Only take log10 of positive values
            t_range_positive = np.asarray(t_range)
            t_range_positive = np.where(
                t_range_positive > 0, t_range_positive, np.finfo(float).eps
            )
            t0_range = np.log10(t_range_positive)
    else:
        t_auto = True  # no range to apply
    axes = []
    for n in range(num_figs):
        i_col = n % col
        i_row = n // col
        t = hdl.addPlot(row=i_row, col=i_col)
        axes.append(t)
        if show_label:
            t.addLegend(offset=(-1, 1), labelTextSize="9pt", verSpacing=-10)

        t.setMouseEnabled(x=False, y=y_auto)
        # Set axis labels once during setup (hoisted out of the m×n inner loop)
        t.setLabel("bottom", "tau (s)")
        t.setLabel("left", "g2")

    color_len = len(colors)
    symbol_len = len(symbols)
    for m in range(num_data):
        # each file has its own number of Q bins
        num_qval = _n_q(g2[m])
        if num_qval == 0:
            continue  # a file without Q bins draws and fits nothing
        panels = (
            _q_panel_indices(q, m, num_qval, labels, ref)
            if plot_type == "multiple"
            else None
        )
        # default base line to be 1.0; used for non-fitting or fit error cases
        baseline_offset = np.ones(num_qval)
        fit_summary = None
        fit_ok = np.zeros(num_qval, dtype=bool)
        if show_fit:
            # Extract force_refit parameter from kwargs
            force_refit = kwargs.get("force_refit", False)
            # the range is in the first file's unit: each file fits in its own
            file_q_range = q_range_in_file_unit(
                xf_list[m], q_range, _shared_q_axis(xf_list)
            )

            if robust_fitting:
                # Use robust fitting with diagnostics
                fit_summary = xf_list[m].fit_g2_robust(
                    file_q_range,
                    t_range,
                    bounds,
                    fit_flag,
                    fit_func,
                    enable_diagnostics=enable_diagnostics,
                    force_refit=force_refit,
                    **{k: v for k, v in kwargs.items() if k != "force_refit"},
                )
            else:
                # Use traditional fitting
                fit_summary = xf_list[m].fit_g2(
                    file_q_range,
                    t_range,
                    bounds,
                    fit_flag,
                    fit_func,
                    force_refit=force_refit,
                )

            fit_ok = _fit_ok_mask(fit_summary, num_qval)
            if fit_summary is not None and subtract_baseline:
                baseline_offset = _fit_baseline(fit_summary, fit_ok, num_qval)

        color = colors[rows[m] % color_len]
        symbol = symbols[rows[m] % symbol_len]

        for n in range(num_qval):
            label = None
            if plot_type == "multiple":
                if panels[n] is None:
                    continue
                ax = axes[panels[n]]
                label = xf_list[m].label
                if m == ref:
                    ax.setTitle(labels[m][n])
            elif plot_type == "single":
                ax = axes[m]
                # overwrite color; use the same color for the same set;
                color = colors[n % color_len]
                if n == 0:
                    ax.setTitle(xf_list[m].label)
            elif plot_type == "single-combined":
                ax = axes[0]
                label = xf_list[m].label + labels[m][n]

            x = tel[m]
            # normalize baseline
            y = g2[m][:, n] - baseline_offset[n] + 1.0 + m * offset
            y_err = g2_err[m][:, n]

            pg_plot_one_g2(
                ax,
                x,
                y,
                y_err,
                color,
                label=label,
                symbol=symbol,
                symbol_size=marker_size,
            )
            # if t_range is not None:
            if not y_auto:
                ax.setRange(yRange=y_range)
            if not t_auto:
                ax.setRange(xRange=t0_range)

            if show_fit and fit_summary is not None and fit_ok[n]:
                # Only a successful fit of this Q bin is drawn or annotated
                if (
                    fit_summary["fit_line"] is not None
                    and n < fit_summary["fit_line"].shape[0]
                    and fit_summary.get("fit_x") is not None
                ):
                    # Get fitted y-values from the numpy array
                    y_fit = fit_summary["fit_line"][n] + m * offset
                    # normalize baseline
                    y_fit = y_fit - baseline_offset[n] + 1.0
                    # Use the correct x-values that were used for fitting
                    fit_x = fit_summary["fit_x"]
                    ax.plot(
                        fit_x,
                        y_fit,
                        pen=pg.mkPen(color, width=2.5),
                    )

                    # Add fitted parameter text annotation
                    if (
                        fit_summary.get("fit_val") is not None
                        and n < fit_summary["fit_val"].shape[0]
                    ):
                        _add_fit_param_annotation(
                            ax, fit_summary["fit_val"][n], fit_func, color
                        )


_TAU_NAMES = ("τ", "τ1", "τ2")


def _fit_ok_mask(fit_summary, num_qval):
    """Per-Q bool array (length ``num_qval``): True where the fit may be shown.

    A failed, non-converged or unidentifiable fit (see ``fit_success_mask``) is
    never drawn, annotated or used to shift the data.
    """
    from xpcsviewer.fitting.legacy import fit_success_mask  # lazy: pulls in nlsq/JAX

    ok = np.zeros(num_qval, dtype=bool)
    if fit_summary is None:
        return ok
    try:
        mask = fit_success_mask(fit_summary)
    except (KeyError, TypeError, ValueError, IndexError):
        return ok
    n = min(num_qval, len(mask))
    ok[:n] = mask[:n]
    return ok


def _fit_baseline(fit_summary, fit_ok, num_qval):
    """Baseline offset per Q: fitted baseline (param 3) where the fit succeeded.

    Model is a*exp(-2(x/b)^c) + d, so the baseline is param 3 only (param 2 is
    the stretch exponent c). Failed Q bins use the default baseline 1.0, so
    their data curve is neither shifted by a placeholder nor blanked by a NaN.
    """
    baseline = np.ones(num_qval)
    try:
        fitted = np.asarray(fit_summary["fit_val"], dtype=float)[:, 0, 3]
    except (IndexError, TypeError, KeyError, ValueError):
        return baseline
    n = min(num_qval, len(fitted))
    use = fit_ok[:n] & np.isfinite(fitted[:n])
    baseline[:n][use] = fitted[:n][use]
    return baseline


def _add_fit_param_annotation(ax, fit_val, fit_func, color):
    """Add fitted parameter text annotation to plot.

    Args:
        ax: PyQtGraph plot axis
        fit_val: Fitted values array [2, n_params] - values and errors
        fit_func: Fitting function type ('single' or 'double')
        color: Color for the text
    """
    # Names follow the model signatures in xpcs_file/fitting.py:
    # single_exp_all(x, a, b, c, d); double_exp_all(x, a, b1, c1, d, b2, c2, f)
    if fit_func == "single":
        param_names = ["contrast", "τ", "stretch", "baseline"]
    else:  # double exponential
        param_names = [
            "contrast",
            "τ1",
            "stretch1",
            "baseline",
            "τ2",
            "stretch2",
            "f",
        ]

    # Create parameter text
    param_text_lines = []
    for p_idx in range(min(len(param_names), fit_val.shape[1])):
        value = fit_val[0, p_idx]  # fitted value
        error = fit_val[1, p_idx]  # error estimate

        # Format error gracefully
        if np.isfinite(error) and error > 0:
            if param_names[p_idx] in _TAU_NAMES:
                param_text_lines.append(
                    f"{param_names[p_idx]} = {value:.3e} ± {error:.2e}"
                )
            else:
                param_text_lines.append(
                    f"{param_names[p_idx]} = {value:.3f} ± {error:.3f}"
                )
        elif param_names[p_idx] in _TAU_NAMES:
            param_text_lines.append(f"{param_names[p_idx]} = {value:.3e} ± --")
        else:
            param_text_lines.append(f"{param_names[p_idx]} = {value:.3f} ± --")

    param_text = "\n".join(param_text_lines)

    # Add text item to plot
    text_item = pg.TextItem(param_text, anchor=(0, 1), color=color)

    # Position text in data coordinates (top-left of visible area)
    viewbox = ax.getViewBox()
    if viewbox is not None:
        # Get current view range
        [[xmin, xmax], [ymin, ymax]] = viewbox.viewRange()
        # Position at 5% from left edge, 95% from bottom (top area)
        text_x = xmin + 0.05 * (xmax - xmin)
        text_y = ymin + 0.95 * (ymax - ymin)
        text_item.setPos(text_x, text_y)
    else:
        # Fallback position
        text_item.setPos(-5, 1.5)

    ax.addItem(text_item)


def pg_plot_from_data(
    hdl,
    *,
    q,
    tel,
    g2,
    g2_err,
    labels,
    num_figs,
    fit_results=None,
    y_auto=False,
    q_auto=False,
    t_auto=False,
    num_col=4,
    rows=None,
    offset=0,
    show_fit=False,
    show_label=False,
    y_range=None,
    t_range=None,
    plot_type="multiple",
    subtract_baseline=True,
    marker_size=5,
    fit_func="single",
    **_ignored_kwargs,
):
    """Render pre-fetched G2 data without re-fetching from XpcsFile objects.

    This is the rendering-only counterpart of ``pg_plot``.  It accepts the
    data structures already extracted by the async worker (``q``, ``tel``,
    ``g2``, ``g2_err``, ``labels``) and renders them directly, avoiding the
    redundant ``get_data()`` and ``get_xf_list()`` calls that ``vk.plot_g2``
    would otherwise trigger on the main thread (BUG-014).

    Parameters mirror the ``pg_plot`` signature where applicable.
    """
    if not q or g2 is None or len(g2) == 0:
        hdl.clear()  # nothing selected: never leave the previous curves on screen
        return

    if num_figs < 1:
        logger.warning("G2 plot skipped: the Q selection contains no Q bins")
        hdl.clear()
        return

    num_data = len(g2)
    ref = _reference_index(g2)
    col = min(num_figs, num_col)
    row = (num_figs + col - 1) // col

    if rows is None or len(rows) == 0:
        rows = list(range(num_data))

    hdl.adjust_canvas_size(num_col=col, num_row=row)
    hdl.clear()

    t0_range = None
    if t_range and not t_auto:
        with np.errstate(divide="ignore", invalid="ignore"):
            t_range_arr = np.asarray(t_range)
            t_range_arr = np.where(t_range_arr > 0, t_range_arr, np.finfo(float).eps)
            t0_range = np.log10(t_range_arr)

    axes = []
    for n in range(num_figs):
        i_col = n % col
        i_row = n // col
        t = hdl.addPlot(row=i_row, col=i_col)
        axes.append(t)
        if show_label:
            t.addLegend(offset=(-1, 1), labelTextSize="9pt", verSpacing=-10)
        t.setMouseEnabled(x=False, y=y_auto)
        # Set axis labels once during setup (hoisted out of the m×n inner loop)
        t.setLabel("bottom", "tau (s)")
        t.setLabel("left", "g2")

    color_len = len(colors)
    symbol_len = len(symbols)
    for m in range(num_data):
        # each file has its own number of Q bins
        num_qval = _n_q(g2[m])
        if num_qval == 0:
            continue
        panels = (
            _q_panel_indices(q, m, num_qval, labels, ref)
            if plot_type == "multiple"
            else None
        )
        baseline_offset = np.ones(num_qval)

        # Use pre-computed fit_results from worker if available
        fit_summary = None
        fit_ok = np.zeros(num_qval, dtype=bool)
        if show_fit and fit_results is not None and m < len(fit_results):
            fit_summary = fit_results[m]
            fit_ok = _fit_ok_mask(fit_summary, num_qval)
            if fit_summary is not None and subtract_baseline:
                baseline_offset = _fit_baseline(fit_summary, fit_ok, num_qval)

        color = colors[rows[m] % color_len]
        symbol = symbols[rows[m] % symbol_len]

        for n in range(num_qval):
            label = None
            if plot_type == "multiple":
                if panels[n] is None:
                    continue
                ax = axes[panels[n]]
                if m == ref:
                    ax.setTitle(labels[m][n] if labels and m < len(labels) else "")
            elif plot_type == "single":
                ax = axes[m]
                color = colors[n % color_len]
                if n == 0:
                    ax.setTitle(labels[m][0] if labels and m < len(labels) else "")
            elif plot_type == "single-combined":
                ax = axes[0]
                label = labels[m][n] if labels and m < len(labels) else ""
            else:
                ax = axes[n % len(axes)]

            x = tel[m]
            y = g2[m][:, n] - baseline_offset[n] + 1.0 + m * offset
            y_err = g2_err[m][:, n]

            pg_plot_one_g2(
                ax,
                x,
                y,
                y_err,
                color,
                label=label,
                symbol=symbol,
                symbol_size=marker_size,
            )

            if not y_auto and y_range is not None:
                ax.setRange(yRange=y_range)
            if not t_auto and t0_range is not None:
                ax.setRange(xRange=t0_range)

            if show_fit and fit_summary is not None and fit_ok[n]:
                try:
                    if (
                        fit_summary.get("fit_line") is not None
                        and n < fit_summary["fit_line"].shape[0]
                        and fit_summary.get("fit_x") is not None
                    ):
                        y_fit = fit_summary["fit_line"][n] + m * offset
                        y_fit = y_fit - baseline_offset[n] + 1.0
                        fit_x = fit_summary["fit_x"]
                        ax.plot(fit_x, y_fit, pen=pg.mkPen(color, width=2.5))
                        if (
                            fit_summary.get("fit_val") is not None
                            and n < fit_summary["fit_val"].shape[0]
                        ):
                            _add_fit_param_annotation(
                                ax, fit_summary["fit_val"][n], fit_func, color
                            )
                except (IndexError, KeyError, TypeError):
                    pass


def pg_plot_one_g2(ax, x, y, dy, color, label, symbol, symbol_size=5):
    """Plot a single G2 correlation curve with error bars on a log-x axis.

    Filters NaN/inf values, applies log-scale x-axis, and downsamples
    dense data for rendering performance.

    Args:
        ax: PyQtGraph PlotItem to draw on.
        x: Delay times array (must be positive for log scale).
        y: G2 correlation values.
        dy: G2 error bars (standard deviation).
        color: RGB tuple for plot color, e.g. ``(255, 0, 0)``.
        label: Legend label string, or None.
        symbol: PyQtGraph symbol character (e.g. ``'o'``, ``'s'``).
        symbol_size: Marker size in pixels (default 5).
    """
    # Ensure NumPy arrays at PyQtGraph I/O boundary (JAX arrays not supported)
    from xpcsviewer.backends._conversions import ensure_numpy

    x = ensure_numpy(x)
    y = ensure_numpy(y)
    dy = ensure_numpy(dy)

    # Validate input data
    if len(x) == 0 or len(y) == 0:
        return

    # Filter out invalid data points (NaN, inf, non-positive x for log scale)
    valid_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(dy) & (x > 0)
    if not np.any(valid_mask):
        return  # Skip if no valid data

    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    dy_clean = dy[valid_mask]

    # Optimize pen creation
    pen_line = pg.mkPen(color=color, width=2)
    pen_symbol = pg.mkPen(color=color, width=1)

    # Create error bars more efficiently
    try:
        line = pg.ErrorBarItem(
            x=x_clean, y=y_clean, top=dy_clean, bottom=dy_clean, pen=pen_line
        )
    except (ValueError, RuntimeWarning):
        # Handle edge cases in logarithm calculation
        return

    # Downsample data if too many points for better performance
    if len(x_clean) > 500:
        step = len(x_clean) // 250
        x_plot = x_clean[::step]
        y_plot = y_clean[::step]
    else:
        x_plot = x_clean
        y_plot = y_clean

    # Plot symbols with optimized parameters
    ax.plot(
        x_plot,
        y_plot,
        pen=None,
        symbol=symbol,
        name=label,
        symbolSize=symbol_size,
        symbolPen=pen_symbol,
        symbolBrush=pg.mkBrush(color=(*color, 0)),
    )

    ax.setLogMode(x=True, y=None)
    ax.addItem(line)
    return


def vectorized_g2_baseline_correction(g2_data, baseline_values):
    """
    Vectorized baseline correction for G2 data.

    Args:
        g2_data: G2 data array [time, q_values]
        baseline_values: Baseline values [q_values]

    Returns:
        Baseline-corrected G2 data
    """
    # Ensure host-resident NumPy arrays before raw np.* operations.
    g2_data = ensure_numpy(g2_data)
    baseline_values = ensure_numpy(baseline_values)
    # Broadcast baseline subtraction across all time points
    return g2_data - baseline_values[np.newaxis, :] + 1.0


def batch_g2_normalization(g2_data_list, method="max"):
    """
    Batch normalization of multiple G2 datasets using vectorized operations.

    Args:
        g2_data_list: List of G2 data arrays
        method: Normalization method ('max', 'mean', 'std')

    Returns:
        List of normalized G2 data arrays
    """
    if not g2_data_list:
        return []

    # Ensure host-resident arrays; np.max/mean/std don't accept JAX arrays.
    arrays = [ensure_numpy(a) for a in g2_data_list]

    # Vectorized fast path for 'max' normalization with small-to-medium batch sizes
    # (B ≤ 100). For very large B with small T×Q the stack allocation itself becomes
    # the bottleneck, so fall back to the per-item loop in that case.
    # For 'std' and 'mean', the per-item loop is retained as it avoids materializing
    # a potentially large 3-D stack when the savings are smaller.
    n = len(arrays)
    use_vectorized = n <= 100 and method == "max"

    if use_vectorized:
        # Stack → single kernel for max + divide. The stack is unavoidable but
        # eliminates 2*n separate NumPy kernel dispatches (one max + one divide per item).
        g2_stack = np.stack(arrays, axis=0)  # [B, T, Q]
        scale = np.max(g2_stack, axis=1, keepdims=True)  # [B, 1, Q]
        scale = np.where(scale == 0, 1.0, scale)
        result = g2_stack / scale  # [B, T, Q]
        # Return list of views (no extra copy) for backward compatibility.
        return list(result)

    # Per-item path: handles 'mean', 'std', very large B, or non-uniform shapes.
    normalized_data = []
    for g2_data in arrays:
        if method == "max":
            max_vals = np.max(g2_data, axis=0, keepdims=True)
            max_vals = np.where(max_vals == 0, 1.0, max_vals)
            normalized = g2_data / max_vals
        elif method == "mean":
            mean_vals = np.mean(g2_data, axis=0, keepdims=True)
            mean_vals = np.where(mean_vals == 0, 1.0, mean_vals)
            normalized = g2_data / mean_vals
        elif method == "std":
            mean_vals = np.mean(g2_data, axis=0, keepdims=True)
            std_vals = np.std(g2_data, axis=0, keepdims=True)
            std_vals = np.where(std_vals == 0, 1.0, std_vals)
            normalized = (g2_data - mean_vals) / std_vals
        else:
            normalized = g2_data
        normalized_data.append(normalized)

    return normalized_data


def compute_g2_ensemble_statistics(g2_data_list, include_median: bool = False):
    """
    Compute ensemble statistics for multiple G2 datasets using vectorized operations.

    Args:
        g2_data_list: List of G2 data arrays [time, q_values]
        include_median: If True, compute and include `ensemble_median` in the result.
            This adds an O(B·T·Q·log B) partial-sort cost on top of the O(B·T·Q) mean/std
            computation. Defaults to False for performance. Set True only when the caller
            genuinely needs the median statistic.

    Returns:
        Dictionary with ensemble statistics. Always contains:
            ensemble_mean, ensemble_std, ensemble_min, ensemble_max,
            ensemble_var, q_mean_values, temporal_correlation.
        Optionally contains:
            ensemble_median (when include_median=True).
    """
    # Ensure host-resident NumPy arrays before stacking / raw np.* statistics.
    g2_data_list = [ensure_numpy(arr) for arr in g2_data_list]
    # Stack all data for vectorized operations
    g2_stack = np.stack(g2_data_list, axis=0)  # [batch, time, q_values]
    # Vectorized statistical computations — O(B·T·Q) operations only.
    # np.median is intentionally excluded from the default path: it requires a
    # full O(B·T·Q·log B) partial sort and accounts for 85% of this function's
    # runtime (see tests/reports/baseline_profile.md).
    stats = {
        "ensemble_mean": np.mean(g2_stack, axis=0),
        "ensemble_std": np.std(g2_stack, axis=0),
        "ensemble_min": np.min(g2_stack, axis=0),
        "ensemble_max": np.max(g2_stack, axis=0),
        "ensemble_var": np.var(g2_stack, axis=0),
        "q_mean_values": np.mean(g2_stack, axis=(0, 1)),  # Mean across time and batch
    }

    if include_median:
        stats["ensemble_median"] = np.median(g2_stack, axis=0)

    # Batched temporal correlation: transpose to (q, batch, time) and compute
    # correlation matrices for all q-values without a Python loop.
    # Each q-slice is (batch, time); corrcoef produces (batch, batch).
    q_transposed = np.transpose(g2_stack, (2, 0, 1))  # [q, batch, time]

    # Vectorized batched corrcoef: center, normalize, matmul
    # mean across time axis (last)
    q_mean = np.mean(q_transposed, axis=2, keepdims=True)  # [q, batch, 1]
    q_centered = q_transposed - q_mean  # [q, batch, time]
    # Standard deviation per (q, batch)
    q_std = np.sqrt(np.sum(q_centered**2, axis=2, keepdims=True))  # [q, batch, 1]
    # Avoid division by zero
    q_std = np.where(q_std == 0, 1.0, q_std)
    q_normed = q_centered / q_std  # [q, batch, time]
    # Batched correlation: (q, batch, time) @ (q, time, batch) -> (q, batch, batch)
    n_time = q_transposed.shape[2]
    corr_batch = np.matmul(q_normed, np.transpose(q_normed, (0, 2, 1))) / n_time

    # Return as 3-D ndarray for efficiency; callers that need per-q slices
    # can index directly (corr_batch[q]) without rebuilding a Python list.
    stats["temporal_correlation"] = corr_batch  # shape [num_q, batch, batch]

    return stats


def optimize_g2_error_propagation(g2_data, g2_errors, operations):
    """
    Vectorized error propagation for G2 data operations.

    Args:
        g2_data: G2 data array [time, q_values]
        g2_errors: G2 error array [time, q_values]
        operations: List of operations applied to data

    Returns:
        Propagated errors
    """
    # Ensure host-resident NumPy arrays; np.abs/power don't accept JAX arrays
    # without triggering an untraced host-device transfer.
    g2_data = ensure_numpy(g2_data)
    g2_errors = ensure_numpy(g2_errors)
    propagated_errors = g2_errors.copy()

    for op in operations:
        if op["type"] == "scale":
            # Error propagation for scaling: sigma_new = |scale| * sigma_old
            scale_factor = op["factor"]
            propagated_errors *= np.abs(scale_factor)

        elif op["type"] == "offset":
            # Error propagation for offset: sigma_new = sigma_old (additive operations don't change uncertainty)
            pass

        elif op["type"] == "power":
            # Error propagation for power: sigma_new = |n * x^(n-1)| * sigma_old
            power = op["power"]
            propagated_errors = (
                np.abs(power * np.power(g2_data, power - 1)) * propagated_errors
            )

        elif op["type"] == "log":
            # Error propagation for logarithm: sigma_new = sigma_old / |x|
            propagated_errors = propagated_errors / np.abs(g2_data)

    return propagated_errors


def vectorized_g2_interpolation(tel, g2_data, target_tel):
    """
    Vectorized interpolation of G2 data to new time points.

    Uses JAX vmap + interpax for batch interpolation when available,
    falling back to a single scipy interp1d call with 2D y otherwise.

    Args:
        tel: Original time points
        g2_data: G2 data [time, q_values]
        target_tel: Target time points for interpolation

    Returns:
        Interpolated G2 data
    """
    try:
        import interpax
        import jax
        import jax.numpy as jnp

        # Convert to JAX arrays once
        tel_jax = jnp.asarray(tel)
        target_jax = jnp.asarray(target_tel)
        g2_jax = jnp.asarray(g2_data)  # [time, q_values]

        # Define single-column interpolation function
        def _interp_single_q(y_col):
            return interpax.interp1d(
                target_jax, tel_jax, y_col, method="cubic", extrap=True
            )

        # vmap over columns (q-values): in_axes=1, out_axes=1
        interpolated_jax = jax.vmap(_interp_single_q, in_axes=1, out_axes=1)(g2_jax)

        from xpcsviewer.backends._conversions import ensure_numpy

        return ensure_numpy(interpolated_jax)

    except ImportError:
        # NumPy fallback: use project's Interp1d wrapper (avoids direct scipy import)
        from xpcsviewer.backends.scipy_replacements.interpolate import Interp1d

        # Interpolate each q-column using the wrapper
        result = np.empty((len(target_tel), g2_data.shape[1]))
        for q_idx in range(g2_data.shape[1]):
            interp_func = Interp1d(
                tel,
                g2_data[:, q_idx],
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )
            result[:, q_idx] = interp_func(target_tel)
        return result
