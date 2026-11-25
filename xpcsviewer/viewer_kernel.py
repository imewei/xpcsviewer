# Standard library imports
import contextlib
import os
import threading
import weakref
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .module.average_toolbox import AverageToolbox

# Third-party imports
import numpy as np
import pyqtgraph as pg

# Local imports
from .backends import ensure_numpy
from .constants import MEMORY_CLEANUP_TIMEOUT_S
from .file_locator import FileLocator
from .fileIO.hdf_reader import _file_identity
from .utils.export_guard import ExportCollisionError, ExportGuard
from .utils.log_utils import log_timing
from .utils.logging_config import get_logger
from .xpcs_file import MemoryMonitor, XpcsFile

# Lazy-loaded modules cache
_module_cache: dict[str, ModuleType | type] = {}
# Analysis modules loaded lazily as whole modules (name == submodule name)
_LAZY_MODULES = frozenset(
    {"g2mod", "intt", "saxs1d", "saxs2d", "stability", "tauq", "twotime"}
)


def _export_stems(xf_list) -> list[str]:
    """Collision-free export basenames, one per file, from the source file names.

    Display labels are not unique (``scan.hdf`` and ``scan_results.hdf`` both
    become ``scan``). Names that would still coincide get ``_2``, ``_3``, ...
    in selection order.
    """
    used: set[str] = set()
    stems = []
    for xf in xf_list:
        stem = os.path.splitext(os.path.basename(str(xf.fname)))[0]
        candidate, n = stem, 1
        while candidate.casefold() in used:
            n += 1
            candidate = f"{stem}_{n}"
        used.add(candidate.casefold())
        stems.append(candidate)
    return stems


def _diffusion_fit(xf) -> tuple[dict | None, str | None]:
    """``(G2 fit summary, method)`` the diffusion plot AND export both use.

    One rule: the Bayesian fit when the file has one, else the deterministic
    (NLSQ) fit; ``(None, None)`` when neither exists.
    """
    bayesian = getattr(xf, "bayesian_fit_summary", None)
    if bayesian is not None:
        return bayesian, "bayesian"
    nlsq = getattr(xf, "fit_summary", None)
    return (nlsq, "nlsq") if nlsq is not None else (None, None)


# What the exported tau-q parameters and error columns mean, per fit method.
_DIFFUSION_PROVENANCE = {
    "nlsq": (
        "tau(q) per Q bin: deterministic NLSQ fit of the G2 curve; its 1-sigma "
        "standard errors weight the power-law fit.",
        "a, b: weighted least-squares power-law fit; error columns are the "
        "1-sigma standard errors from that fit's covariance.",
    ),
    "bayesian": (
        "tau(q) per Q bin: posterior mean of the Bayesian (NUTS) G2 fit; the "
        "posterior standard deviation of tau weights the power-law fit.",
        "a, b: weighted least-squares power-law fit to those posterior means; "
        "error columns are the 1-sigma standard errors from that fit's "
        "covariance, NOT credible intervals (no posterior of a, b is sampled).",
    ),
}


def _g2_bin_identities(xf, n_columns: int) -> list[tuple] | None:
    """``(dynamic_bin_id, map_index, q, phi)`` of each G2 column of ``xf``.

    Column ``j`` of ``xf.get_g2_data()`` is bin ``result[j]``: the dynamic bin ID
    is the bin's own G2 column in the file, ``map_index`` its absolute position
    on the file's q x phi map (``dynamic_index_mapping``), and ``q`` / ``phi`` the
    numbers in the file's own arrays at full precision (``phi`` is None when the
    map has no azimuth axis). Mirrors ``cli.twotime_batch._bin_coords``, which
    cannot be imported here (it selects the matplotlib backend on import).
    ``None`` when the file's map cannot be resolved, so the export falls back to
    the plain Q labels.
    """
    from .fitting.bayesian_assembly import resolve_g2_bins

    try:
        qmap = xf.qmap
        mapping = np.asarray(qmap.dynamic_index_mapping)
        names = list(qmap.map_names)
        has_phi = int(qmap.dynamic_num_pts[1]) != 1 and names[1] == "phi"
        dp = np.asarray(qmap.dplist, dtype=np.float64)
        rows = []
        for bin_id, q, _label in resolve_g2_bins(xf, None):
            absolute = int(mapping[bin_id])
            phi = float(dp[absolute % len(dp)]) if has_phi else None
            rows.append((bin_id, absolute, float(q), phi))
    except AttributeError:  # not a loaded file (no map): plain Q labels
        logger.debug("G2 export: no q-map on %s, bin identity omitted", xf)
        return None
    except (IndexError, ValueError, TypeError) as exc:
        logger.warning(f"G2 export: bin identity unavailable for {xf.label}: {exc}")
        return None
    if len(rows) != n_columns:
        logger.warning(
            f"G2 export: {len(rows)} map bins for {n_columns} G2 columns of "
            f"{xf.label}; bin identity omitted"
        )
        return None
    return rows


def _fit_bin_identities(fit_summary: dict, fit_q, columns: list[tuple] | None):
    """Identity of each fitted row, from the fit's own ``bin_ids`` (Bayesian) or
    by matching the fitted Q values in order against the file's bins.

    A fit selects whole Q groups (every sector at a fitted Q), so the fitted Qs
    are an ordered subsequence of the file's Qs and the match is exact.
    """
    if columns is None:
        return None
    ids = fit_summary.get("bin_ids")
    if ids is not None:
        by_id = {c[0]: c for c in columns}
        found = [by_id.get(int(i)) for i in np.ravel(ids)[: len(fit_q)]]
        return found if all(f is not None for f in found) else None
    out, pos = [], 0
    for q in fit_q:
        while pos < len(columns) and columns[pos][2] != float(q):
            pos += 1
        if pos == len(columns):
            return None
        out.append(columns[pos])
        pos += 1
    return out


def _phi_text(phi) -> str:
    return "NA" if phi is None else repr(float(phi))


def _bin_tag(ident: tuple | None) -> str:
    """Suffix of a column name: the bin's absolute identity (empty if unknown)."""
    if ident is None:
        return ""
    bin_id, absolute, q, phi = ident
    return f"[bin={bin_id},map={absolute},q={float(q)!r},phi={_phi_text(phi)}]"


def _bin_row_comment(row: int, ident: tuple | None) -> str:
    if ident is None:
        return f"# row {row}: bin identity unavailable\n"
    bin_id, absolute, q, phi = ident
    return (
        f"# row {row}: dynamic_bin_id={bin_id} map_index={absolute} "
        f"q={float(q)!r} phi={_phi_text(phi)}\n"
    )


@contextlib.contextmanager
def _diffusion_fits_shown(xf_list):
    """Show each file's ``_diffusion_fit`` as ``xf.fit_summary`` (restored on exit).

    ``fit_tauq`` and the tau-q plot read ``xf.fit_summary``; the tau-q results
    are written into the shown summary, which is the one the export reads.
    """
    original = {id(xf): xf.fit_summary for xf in xf_list}
    try:
        for xf in xf_list:
            xf.fit_summary = _diffusion_fit(xf)[0]
        yield
    finally:
        for xf in xf_list:
            xf.fit_summary = original[id(xf)]


def _get_module(module_name: str) -> ModuleType | type:
    """Lazy load analysis modules to improve startup time"""
    if module_name not in _module_cache:
        if module_name in _LAZY_MODULES:
            import importlib

            _module_cache[module_name] = importlib.import_module(
                f".module.{module_name}", __package__
            )
        elif module_name == "average_toolbox":
            # Special case: callers expect the class, not the module
            from .module.average_toolbox import AverageToolbox

            _module_cache[module_name] = AverageToolbox
        else:
            raise ImportError(f"Unknown module: {module_name}")
    return _module_cache[module_name]


logger = get_logger(__name__)


class ViewerKernel(FileLocator):
    """
    Backend kernel for XPCS data processing and analysis coordination.

    ViewerKernel serves as the central coordinator between the GUI and various
    analysis modules. It manages file collections, handles data caching,
    coordinates background processing, and provides a unified interface
    for all analysis operations with lazy-loaded modules for improved startup performance.

    The kernel inherits from FileLocator to provide file management
    capabilities and extends it with advanced caching, memory management,
    and analysis coordination features.

    Parameters
    ----------
    path : str
        Base directory path for XPCS data files
    statusbar : QStatusBar, optional
        GUI status bar for displaying operation messages

    Attributes
    ----------
    meta : dict
        Metadata storage for analysis parameters and cached results
    avg_worker : AverageToolbox or None
        Current averaging worker, or None if no job is active

    Notes
    -----
    The kernel implements memory-aware caching using weak references
    and automatic cleanup mechanisms to handle large XPCS datasets
    efficiently. It supports both synchronous and asynchronous
    operation modes for optimal GUI responsiveness.

    Examples
    --------
    >>> kernel = ViewerKernel("/path/to/data")
    >>> kernel.plot_saxs_2d(plot_handler, rows=[0, 1, 2])
    """

    def get_module(self, module_name: str) -> ModuleType | type:
        """Public interface to lazy load analysis modules"""
        return _get_module(module_name)

    def __init__(self, path: str, statusbar: Any | None = None) -> None:
        super().__init__(path)
        self.statusbar = statusbar
        self.meta: dict[str, Any] = {}
        # True when the last build() (Reload) re-read or dropped the SAXS background
        self.background_replaced = False
        self.reset_meta()
        self.path = path
        # Plotting and analysis state
        self.avg_worker: AverageToolbox | None = None
        self.avg_worker_active: dict[int, Any] = {}

        # Memory-aware caching for current_dset with weak references.
        # Lock guards all reads/writes: WeakValueDictionary is not thread-safe
        # because its internal finalizer can delete entries during a concurrent
        # .get() on another thread (P1-03).
        self._current_dset_cache: weakref.WeakValueDictionary[str, XpcsFile] = (
            weakref.WeakValueDictionary()
        )
        self._dset_cache_lock = threading.Lock()
        self._plot_kwargs_record: dict[str, Any] = {}
        self._memory_cleanup_threshold = 0.8  # Trigger cleanup at 80% memory usage

    def reset_meta(self) -> None:
        """
        Reset metadata dictionary to default values.

        Initializes the metadata storage with default values for
        various analysis parameters and cached results.

        Returns
        -------
        None

        Notes
        -----
        The metadata dictionary stores analysis parameters including:
        - SAXS 1D background file information
        - Averaging file lists and parameters
        - G2 analysis results
        - Other analysis-specific cached data
        """
        self.meta = {
            # saxs 1d:
            "saxs1d_bkg_fname": None,
            "saxs1d_bkg_xf": None,
            # full path and the on-disk revision (``_file_identity``) the loaded
            # background was read from
            "saxs1d_bkg_path": None,
            "saxs1d_bkg_revision": None,
            # avg
            "avg_file_list": None,
            "avg_intt_minmax": None,
            "avg_g2_avg": None,
            # g2
        }

    def reset_kernel(self) -> None:
        """
        Reset kernel state and clear all cached data.

        Performs complete kernel reset including target file list clearing,
        metadata reset, and comprehensive memory cleanup operations.

        Notes
        -----
        - Clears target file selection
        - Resets all metadata to default values
        - Triggers comprehensive memory cleanup
        - Suitable for starting fresh analysis or recovering from errors

        This method should be called when switching to a new dataset
        or when memory cleanup is required.
        """
        self.clear_target()
        self.reset_meta()
        self._cleanup_memory()

    @log_timing(threshold_ms=100)
    def _cleanup_memory(self) -> None:
        """
        Clean up cached data and release memory resources.

        Performs comprehensive memory cleanup including dataset cache clearing,
        plot parameter cache management, and XpcsFile cache cleanup.
        Uses optimized cleanup strategies when available.

        Notes
        -----
        - Clears current dataset cache using weak references
        - Conditionally clears plot kwargs based on memory pressure
        - Schedules background cleanup for XpcsFile objects
        - Falls back to direct cleanup methods if optimized system unavailable
        - Implements timeout protection to prevent excessive blocking
        - Logs memory usage before and after cleanup operations

        The method is automatically called when memory pressure exceeds
        the configured threshold or when explicitly requested during
        kernel reset operations.
        """
        # Clear dataset cache
        with self._dset_cache_lock:
            self._current_dset_cache.clear()

        # Clear plot kwargs if memory pressure is high
        if MemoryMonitor.is_memory_pressure_high(self._memory_cleanup_threshold):
            self._plot_kwargs_record.clear()
            logger.info("Memory pressure detected, cleared plot kwargs cache")

        # Schedule background cleanup for XpcsFile objects (non-blocking)
        try:
            from .threading.cleanup_optimized import (
                CleanupPriority,
                schedule_type_cleanup,
            )

            schedule_type_cleanup("XpcsFile", CleanupPriority.HIGH)
            logger.debug("Scheduled background cache clearing for XpcsFile objects")
        except ImportError:
            # Fallback to direct cleanup if optimized system not available
            try:
                from .threading.cleanup_optimized import get_object_registry

                registry = get_object_registry()
                xpcs_files = registry.get_objects_by_type("XpcsFile")
                for xf in xpcs_files:
                    xf.clear_cache()
                logger.debug(
                    f"Directly cleared cache for {len(xpcs_files)} XpcsFile objects"
                )
            except Exception as e:
                logger.debug(f"Direct cache clearing failed, using fallback: {e}")
                # Final fallback: limited traversal with timeout
                import gc
                import time

                start_time = time.time()
                cleared_count = 0
                for obj in gc.get_objects():
                    if isinstance(obj, XpcsFile):
                        obj.clear_cache()
                        cleared_count += 1
                    # Prevent excessive blocking
                    if time.time() - start_time > MEMORY_CLEANUP_TIMEOUT_S:
                        logger.warning("Memory cleanup timeout, stopped early")
                        break
                logger.debug(
                    f"Fallback: cleared cache for {cleared_count} XpcsFile objects"
                )

        used_mb, _available_mb = MemoryMonitor.get_memory_usage()
        logger.debug(f"Memory cleanup completed, current usage: {used_mb:.1f}MB")

    def select_bkgfile(self, fname: str) -> None:
        """
        Select background file for SAXS 1D background subtraction.

        Sets up background file for subtraction in SAXS 1D analysis
        by creating an XpcsFile object and storing metadata.

        Parameters
        ----------
        fname : str
            Full path to the background file

        Notes
        -----
        The background file should have compatible Q-range and data
        structure with the analysis files for proper subtraction.
        """
        path = os.path.abspath(fname)
        revision = _file_identity(path)  # before the read, like the locator
        xf = XpcsFile(path, qmap_manager=self.qmap_manager)
        self.meta["saxs1d_bkg_fname"] = os.path.basename(fname)
        self.meta["saxs1d_bkg_path"] = path
        self.meta["saxs1d_bkg_xf"] = xf
        self.meta["saxs1d_bkg_revision"] = revision

    def background_revision(self) -> tuple[str, tuple[int, ...] | None] | None:
        """``(path, on-disk revision)`` of the selected SAXS background.

        ``None`` when none is selected. Part of the SAXS-1D plot request, so a
        replaced background makes the request differ from the drawn one.
        """
        path = self.meta.get("saxs1d_bkg_path")
        return None if path is None else (path, _file_identity(path))

    def refresh_background(self) -> bool:
        """Bring the background up to date with its file; True if it changed.

        The background is held outside the locator's cache, so it gets the
        locator's rule here: a file replaced on disk is read again, a file that
        is gone is dropped with a warning (subtraction is then skipped).
        """
        path = self.meta.get("saxs1d_bkg_path")
        if path is None:
            return False
        current = _file_identity(path)
        if current == self.meta.get("saxs1d_bkg_revision"):
            return False
        xf = None
        if current is None:
            logger.warning(f"SAXS background file {path} is gone; it is cleared")
        else:
            try:
                xf = XpcsFile(path, qmap_manager=self.qmap_manager)
            except Exception as exc:
                logger.warning(f"SAXS background file {path} cannot be read: {exc}")
        if xf is None:
            self.meta.update(
                saxs1d_bkg_fname=None,
                saxs1d_bkg_path=None,
                saxs1d_bkg_xf=None,
                saxs1d_bkg_revision=None,
            )
        else:
            logger.info(f"SAXS background file {path} was replaced; reloaded")
            self.meta["saxs1d_bkg_xf"] = xf
            self.meta["saxs1d_bkg_revision"] = current
        return True

    def get_bkg_xf(self) -> XpcsFile | None:
        """The SAXS background, current with its file (None if none/gone)."""
        self.refresh_background()
        return self.meta["saxs1d_bkg_xf"]

    def build(self, path=None, filter_list=(".hdf", ".h5"), sort_method="Filename"):
        """Rebuild the file list (Reload); also refreshes the SAXS background."""
        built = super().build(
            path=path, filter_list=filter_list, sort_method=sort_method
        )
        self.background_replaced = bool(built) and self.refresh_background()
        return built

    def get_pg_tree(self, rows: Any) -> Any | None:
        xf_list = self.get_xf_list(rows)
        if xf_list:
            return xf_list[0].get_pg_tree()
        return None

    def get_fitting_tree(self, rows: Any) -> Any | None:
        xf_list = self.get_xf_list(rows, filter_atype="Multitau")
        result = {}
        for x in xf_list:
            result[x.label] = x.get_fitting_info(mode="g2_fitting")
        tree = pg.DataTreeWidget(data=result)
        tree.setWindowTitle("fitting summary")
        tree.resize(1024, 800)
        return tree

    @log_timing(threshold_ms=500)
    def plot_g2(self, handler, q_range, t_range, y_range, rows=None, **kwargs):
        """
        Generate G2 correlation function plots for multi-tau analysis.

        Creates correlation function plots for selected files with specified
        Q-range and time-range parameters. Supports fitting with single or
        double exponential functions.

        Parameters
        ----------
        handler : PlotHandler
            Plot handler object for rendering the correlation functions
        q_range : tuple
            Q-value range as (q_min, q_max) in inverse Angstroms
        t_range : tuple
            Time delay range as (t_min, t_max) in seconds
        y_range : tuple
            Y-axis range for correlation function values
        rows : list, optional
            List of file indices to include in analysis
        **kwargs
            Additional plotting parameters (fitting options, display settings)

        Returns
        -------
        tuple
            (q_values, time_delays) arrays for successful plots, (None, None) if no data

        Notes
        -----
        Only processes files with "Multitau" analysis type. The method
        delegates actual plotting to the g2mod module while providing
        data preprocessing and validation.
        """
        xf_list = self.get_xf_list(rows=rows, filter_atype="Multitau")
        if xf_list:
            g2_module = self.get_module("g2mod")
            g2_module.pg_plot(handler, xf_list, q_range, t_range, y_range, **kwargs)
            q, tel, *_unused = g2_module.get_data(xf_list)
            return q, tel
        return None, None

    @log_timing(threshold_ms=500)
    def plot_g2map(
        self,
        g2map_hdl,
        qmap_hdl,
        g2_hdl,
        rows=None,
        qbin=0,
        normalization=False,
        g2map_cmap="jet",
        g2map_scale="linear",
        qmap_cmap="tab20b",
    ):
        """
        Generate G2 2D map visualization.

        Creates a 2D representation of G2 correlation data with associated
        Q-map and profile views for comprehensive analysis.

        Parameters
        ----------
        g2map_hdl : ImageView
            Plot handler for the G2 2D map display
        qmap_hdl : ImageView
            Plot handler for the cropped Q-map display
        g2_hdl : PlotWidget
            Plot handler for the G2 profile at selected Q-bin
        rows : list, optional
            List of file indices (uses first file only)
        qbin : int
            Index of Q-bin for profile display
        normalization : bool
            If True, apply baseline normalization to G2 data
        g2map_cmap : str
            Matplotlib colormap name for the G2 map (default: "jet")
        g2map_scale : str
            Scale type for the G2 map: "linear" or "log" (default: "linear")
        qmap_cmap : str
            Matplotlib colormap name for the Q-map (default: "tab20b")

        Notes
        -----
        This visualization shows the full G2 correlation map with the ability
        to select individual Q-bins for detailed profile analysis.
        """
        xf_list = self.get_xf_list(rows=rows)
        if not xf_list:
            return

        xf_obj = xf_list[0]

        # G2 map image with optional log scale
        g2_data = ensure_numpy(xf_obj.get_offseted_g2(normalization).T)
        if g2map_scale == "log":
            g2_data = np.log10(np.clip(g2_data, 1e-10, None))
        g2map_hdl.setImage(g2_data)
        g2map_hdl.setColorMap(pg.colormap.getFromMatplotlib(g2map_cmap))

        # Cropped Q-map image: the file's own detector map, because it sits next
        # to G2 columns that are fixed to that partition (an imported partition
        # is display-only and is not drawn here)
        qmap_hdl.setImage(ensure_numpy(xf_obj.get_cropped_qmap("dqmap")))
        qmap_hdl.setColorMap(pg.colormap.getFromMatplotlib(qmap_cmap))

        # Plot G2 profile for selected Q-bin
        g2_hdl.clear()
        color = (0, 128, 255)
        pen = pg.mkPen(color=color, width=2)

        x = xf_obj.t_el
        y = xf_obj.g2[:, qbin]
        dy = xf_obj.g2_err[:, qbin]

        # Filter valid data points (positive x for log scale)
        valid_mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
        if not np.any(valid_mask):
            logger.warning("No valid data points for G2 map profile")
            return

        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        dy_valid = dy[valid_mask] if dy is not None else np.zeros_like(y_valid)

        line = pg.ErrorBarItem(
            x=np.log10(x_valid), y=y_valid, top=dy_valid, bottom=dy_valid, pen=pen
        )
        pen_symbol = pg.mkPen(color=color, width=1)
        g2_hdl.plot(
            x_valid,
            y_valid,
            pen=None,
            symbol="o",
            name=f"{qbin=}",
            symbolSize=3,
            symbolPen=pen_symbol,
            symbolBrush=pg.mkBrush(color=(*color, 0)),
        )

        g2_hdl.setLogMode(x=True, y=None)
        g2_hdl.addItem(line)

    @log_timing(threshold_ms=500)
    def plot_g2_stability(
        self, handler, q_range, t_range, y_range, rows=None, **kwargs
    ):
        """
        Generate G2 stability plots showing frame-by-frame correlation analysis.

        Creates plots showing how G2 correlation varies across different frames
        for analyzing temporal stability of the measurement.

        Parameters
        ----------
        handler : GraphicsLayoutWidget
            PyQtGraph layout widget for plotting
        q_range : tuple
            Q-value range as (q_min, q_max)
        t_range : tuple
            Time delay range as (t_min, t_max)
        y_range : tuple
            Y-axis range for correlation function values
        rows : list, optional
            List of file indices (uses first file only)
        **kwargs
            Additional plotting parameters

        Notes
        -----
        Only processes files with "Multitau" analysis type and requires
        g2_partial data to be available in the HDF5 file.
        """
        xf_list = self.get_xf_list(rows=rows, filter_atype="Multitau")
        if not xf_list:
            logger.warning("No Multitau files available for G2 stability analysis")
            return

        xf_obj = xf_list[0]

        # Safety check: verify g2_partial data is available
        if xf_obj.g2_partial is None:
            logger.info("g2_partial data not available for %s", xf_obj.fname)
            if self.statusbar is not None:
                self.statusbar.showMessage("g2_partial data not available", 3000)
            return

        g2_module = self.get_module("g2mod")
        g2_module.pg_plot_stability(
            handler, xf_obj, q_range, t_range, y_range, **kwargs
        )

    def plot_qmap(self, hdl, rows=None, target=None, cmap="tab20b"):
        xf_list = self.get_xf_list(rows=rows)
        if xf_list:
            if target == "scattering":
                value = np.log10(xf_list[0].saxs_2d + 1)
                vmin, vmax = np.percentile(value, (2, 98))
                hdl.setImage(ensure_numpy(value), levels=(vmin, vmax))
            elif target == "dynamic_roi_map":
                # designated map display: shows an imported partition
                hdl.setImage(ensure_numpy(xf_list[0].get_display_qmap("dqmap")))
            elif target == "static_roi_map":
                hdl.setImage(ensure_numpy(xf_list[0].get_display_qmap("sqmap")))
            # Apply colormap
            hdl.setColorMap(pg.colormap.getFromMatplotlib(cmap))

    def plot_tauq_pre(self, hdl=None, rows=None):
        # Support both Multitau and Twotime files for diffusion analysis
        logger.debug(f"plot_tauq_pre called with hdl={hdl is not None}, rows={rows}")

        xf_list = self.get_xf_list(rows=rows)
        logger.debug(f"Got {len(xf_list)} files from get_xf_list")

        # Filter for files that support G2 analysis (same as G2 plotting)
        g2_compatible_list = []
        for xf in xf_list:
            if any(atype in xf.atype for atype in ["Multitau", "Twotime"]):
                g2_compatible_list.append(xf)
        xf_list = g2_compatible_list
        logger.debug(f"Filtered to {len(xf_list)} G2-compatible files")

        # For tau-q, prefer Bayesian results when available (_diffusion_fit)
        with _diffusion_fits_shown(xf_list):
            short_list = [xf for xf in xf_list if xf.fit_summary is not None]
            logger.debug(f"Found {len(short_list)} files with fit_summary")

            if len(short_list) == 0:
                logger.warning(
                    f"No files with G2 fitting results found for diffusion analysis. "
                    f"Found {len(xf_list)} G2-compatible files but none have been fitted."
                )
                # Clear the plot and show a message
                if hdl is not None:
                    logger.debug("Clearing plot and showing empty message")
                    hdl.clear()
                    ax = hdl.subplots(1, 1)
                    ax.text(
                        0.5,
                        0.5,
                        'No G2 fitting results available.\n\nPlease:\n1. Go to "g2" tab\n2. Select files\n3. Click "update plot" to perform G2 fitting\n4. Return to "Diffusion" tab',
                        ha="center",
                        va="center",
                        fontsize=12,
                        bbox={
                            "boxstyle": "round,pad=0.3",
                            "facecolor": "lightblue",
                            "alpha": 0.7,
                        },
                    )
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis("off")
                    hdl.fig.tight_layout()
                    hdl.draw()  # Add missing draw call
                    logger.debug("Empty message plot drawn")
                else:
                    logger.warning("hdl is None, cannot display empty message")
            else:
                logger.info(
                    f"Found {len(short_list)} files with G2 fitting results for diffusion analysis"
                )
                if hdl is not None:
                    logger.debug("Calling tauq.plot_pre with files")
                    _get_module("tauq").plot_pre(short_list, hdl)
                    logger.debug("tauq.plot_pre completed")
                else:
                    logger.warning("hdl is None, cannot call tauq.plot_pre")

    def plot_tauq(
        self,
        hdl=None,
        bounds=None,
        rows=None,
        plot_type=3,
        fit_flag=None,
        offset=None,
        q_range=None,
        force_refit=False,
    ):
        logger.debug(
            f"plot_tauq called with hdl={hdl is not None}, bounds={bounds}, rows={rows}, plot_type={plot_type}, fit_flag={fit_flag}, offset={offset}, q_range={q_range}"
        )
        if rows is None:
            rows = []
        # a file counts as fitted when it has any G2 fit (NLSQ or Bayesian)
        xf_list = [
            xf
            for xf in self.get_xf_list(rows=rows, filter_atype="Multitau")
            if _diffusion_fit(xf)[0] is not None
        ]

        # For tau-q, prefer Bayesian results when available (_diffusion_fit)
        result = {}
        with _diffusion_fits_shown(xf_list):
            for x in xf_list:
                x.fit_tauq(q_range, bounds, fit_flag, force_refit=force_refit)
                if (x.fit_summary or {}).get("tauq_success", False):
                    result[x.label] = x.get_fitting_info(mode="tauq_fitting")
                else:
                    logger.error("tau-q fit failed for %s", x.label)
                    result[x.label] = "tauq fitting failed"

            if len(result) > 0:
                _get_module("tauq").plot(
                    xf_list,
                    hdl=hdl,
                    q_range=q_range,
                    offset=offset,
                    plot_type=plot_type,
                )

        return result

    def get_info_at_mouse(self, rows, x, y):
        xf = self.get_xf_list(rows)
        if xf:
            info = xf[0].get_info_at_position(x, y)
            return info
        return None

    @log_timing(threshold_ms=500)
    def plot_saxs_2d(self, *args, rows=None, **kwargs):
        """
        Generate SAXS 2D scattering pattern visualization.

        Creates 2D scattering pattern plots for the first selected file
        with configurable display parameters including colormaps,
        rotation, and intensity scaling.

        Parameters
        ----------
        *args
            Variable positional arguments passed to saxs2d.plot()
        rows : list, optional
            List of file indices (uses first file only)
        **kwargs
            Plotting parameters including plot_type, cmap, rotate, etc.

        Notes
        -----
        Only plots the first file from the selection. For multiple
        file analysis, use other visualization methods or averaging tools.
        """
        xf_list = self.get_xf_list(rows)[0:1]
        if xf_list:
            _get_module("saxs2d").plot(xf_list[0], *args, **kwargs)

    def add_roi(self, hdl, rows=None, **kwargs):
        """Add a default ROI at the beam centre of the file shown in ``hdl``.

        ``rows`` are the target rows the SAXS-2D view shows (its first file is
        the one drawn); the beam centre is placed at the centre of its pixel in
        the displayed frame (``hdl.pixel_centre``), not at a raw pixel index.
        """
        logger.debug(f"ViewerKernel: add_roi called with kwargs: {kwargs}")

        xf_list = self.get_xf_list(rows)[0:1]
        logger.debug(
            f"ViewerKernel: get_xf_list returned {len(xf_list) if xf_list else 0} files"
        )

        if not xf_list:
            logger.warning(
                "Cannot add ROI: No XPCS files loaded or selected. Please add files to the target list first."
            )
            return None

        try:
            bcx, bcy = xf_list[0].bcx, xf_list[0].bcy  # pixel-index frame
            cen = hdl.pixel_centre(bcx, bcy)  # where that pixel is drawn
            logger.debug(f"ViewerKernel: beam center {(bcx, bcy)} drawn at {cen}")
        except (AttributeError, IndexError) as e:
            logger.error(f"Cannot get beam center coordinates: {e}")
            return None

        sl_type = kwargs["sl_type"]
        logger.debug(f"ViewerKernel: creating ROI of type '{sl_type}'")

        if sl_type == "Pie":
            result = hdl.add_roi(cen=cen, radius=100, **kwargs)
            logger.debug(f"ViewerKernel: Pie ROI creation result: {result}")
            return result
        if sl_type == "Q-Wedge":
            # Q-Wedge ROI for Q-space analysis (wedge-shaped)
            logger.debug(
                f"ViewerKernel: creating Q-Wedge ROI at center {cen} with radius 100"
            )
            result = hdl.add_roi(cen=cen, radius=100, **kwargs)
            logger.debug(f"ViewerKernel: Q-Wedge ROI creation result: {result}")
            return result
        if sl_type == "Circle":
            try:
                radius_v = min(xf_list[0].mask.shape[0] - bcy, bcy)
                radius_h = min(xf_list[0].mask.shape[1] - bcx, bcx)
                radius = min(radius_h, radius_v) * 0.8

                result1 = hdl.add_roi(cen=cen, radius=radius, label="RingA", **kwargs)
                result2 = hdl.add_roi(
                    cen=cen, radius=0.8 * radius, label="RingB", **kwargs
                )
                logger.debug(
                    f"ViewerKernel: Circle ROI creation results: {result1}, {result2}"
                )
                return result1  # Return first ring result
            except (AttributeError, IndexError) as e:
                logger.error(f"Cannot create Circle ROI: mask data not available - {e}")
                return None
        elif sl_type == "Phi-Ring":
            # Phi-Ring ROI for angular analysis (ring-shaped)
            try:
                radius_v = min(xf_list[0].mask.shape[0] - bcy, bcy)
                radius_h = min(xf_list[0].mask.shape[1] - bcx, bcx)
                radius = min(radius_h, radius_v) * 0.8

                result1 = hdl.add_roi(cen=cen, radius=radius, label="RingA", **kwargs)
                result2 = hdl.add_roi(
                    cen=cen, radius=0.8 * radius, label="RingB", **kwargs
                )
                logger.debug(
                    f"ViewerKernel: Phi-Ring ROI creation results: {result1}, {result2}"
                )
                return result1  # Return first ring result
            except (AttributeError, IndexError) as e:
                logger.error(
                    f"Cannot create Phi-Ring ROI: mask data not available - {e}"
                )
                return None
        else:
            logger.error(f"ViewerKernel: Unknown ROI type '{sl_type}'")
            return None

    def get_xf_rows(self, rows=None, filter_atype=None) -> list[tuple[int, XpcsFile]]:
        """``(target row, file)`` for each requested row whose file loads.

        ``get_xf_list`` drops rows whose file is missing or unreadable, so its
        positions no longer match target rows. Anything that needs to know WHICH
        row a file came from (highlighting, colours, labels) must use this.
        """
        selected: Any
        if rows is None:
            selected = range(len(self.target))
        elif isinstance(rows, int):
            selected = [rows]
        else:
            selected = rows
        return [
            (r, xf)
            for r in selected
            for xf in self.get_xf_list(rows=[r], filter_atype=filter_atype)
        ]

    def plot_saxs_1d(self, pg_hdl, mp_hdl, **kwargs):
        rows = kwargs.pop("rows", None)  # target rows to highlight
        pairs = self.get_xf_rows()
        if pairs or self.target:
            xf_list = [xf for _, xf in pairs]
            # the plot indexes the files it was given: translate target rows into
            # positions among them (a row whose file is gone highlights nothing)
            position = {r: k for k, (r, _) in enumerate(pairs)}
        else:  # no target: the locator's cached files, which have no rows
            xf_list, position = self.get_xf_list(), {}
        if xf_list:
            roi_list = pg_hdl.get_roi_list()
            self.get_module("saxs1d").pg_plot(
                xf_list,
                mp_hdl,
                rows=None
                if rows is None
                else [position[r] for r in rows if r in position],
                bkg_file=self.get_bkg_xf(),
                roi_list=roi_list,
                **kwargs,
            )

    def export_saxs_1d(self, pg_hdl, folder):
        xf_list = self.get_xf_list()
        roi_list = pg_hdl.get_roi_list()
        guard = ExportGuard(xf.fname for xf in xf_list)
        exported = 0
        # Stems are allocated for the whole export so two files sharing a display
        # label (scan.hdf / scan_results.hdf) cannot overwrite each other.
        for xf, stem in zip(xf_list, _export_stems(xf_list), strict=True):
            try:
                xf.export_saxs1d(roi_list, folder, basename=stem, guard=guard)
            except ExportCollisionError as exc:
                logger.error(f"SAXS 1D export of {xf.label} refused: {exc}")
                continue
            exported += 1
        total = len(xf_list)
        if total and exported == total:
            logger.info(f"SAXS 1D export completed: {exported}/{total} files")
        elif exported:
            logger.warning(
                f"SAXS 1D export partially completed: {exported}/{total} files "
                "exported successfully"
            )
        elif total:
            raise ValueError(
                f"SAXS 1D export failed: 0/{total} files exported successfully"
            )

    def switch_saxs1d_line(self, mp_hdl, lb_type):
        """Switch SAXS 1D line builder type (stub for future implementation)."""

    @log_timing(threshold_ms=1000)
    def plot_twotime(self, hdl, rows=None, **kwargs):
        """
        Generate two-time correlation plots for dynamic analysis.

        Creates two-time correlation maps and associated visualizations
        for analyzing sample dynamics over extended time periods.
        Uses memory-aware caching to handle large correlation datasets.

        Parameters
        ----------
        hdl : dict
            Dictionary of plot handlers for different visualization components
        rows : list, optional
            List of file indices to process (uses current selection if None)
        **kwargs
            Plotting parameters including colormap, cropping, and level settings

        Returns
        -------
        list or None
            List of Q-bin labels for successful processing, None if no data

        Notes
        -----
        - Only processes files with "Twotime" analysis type
        - Implements memory pressure monitoring and cleanup
        - Uses cached datasets to improve performance for repeated access
        - Automatically triggers memory cleanup when pressure exceeds threshold
        """
        xf_list = self.get_xf_list(rows, filter_atype="Twotime")
        if len(xf_list) == 0:
            return None
        xfile = xf_list[0]
        new_qbin_labels = None

        # Use memory-aware caching for current dataset
        current_dset = self._get_cached_dataset(xfile.fname, xfile)
        logger.debug(
            f"ViewerKernel.plot_twotime: current_dset cached={current_dset is not None}, fname={xfile.fname}"
        )

        # identity, not fname: the cache is keyed by fname and can hold an older
        # XpcsFile of the same path (the file was reloaded or replaced)
        if current_dset is None or current_dset is not xfile:
            logger.debug(f"Dataset changed, loading new qbin labels for {xfile.fname}")
            current_dset = xfile
            with self._dset_cache_lock:
                self._current_dset_cache[xfile.fname] = current_dset
            new_qbin_labels = xfile.get_twotime_qbin_labels()

            # Check memory pressure and cleanup if needed
            if MemoryMonitor.is_memory_pressure_high(self._memory_cleanup_threshold):
                self._cleanup_memory()
        else:
            logger.debug(
                "Using cached dataset, getting qbin labels for ComboBox population"
            )
            # Always get qbin labels for ComboBox population, even for cached datasets
            new_qbin_labels = current_dset.get_twotime_qbin_labels()

        _get_module("twotime").plot_twotime(current_dset, hdl, **kwargs)
        logger.debug(f"Returning new_qbin_labels: {new_qbin_labels}")
        return new_qbin_labels

    def _get_cache_count(self) -> int:
        """Return current _current_dset_cache size under lock (BUG-C)."""
        with self._dset_cache_lock:
            return len(self._current_dset_cache)

    def _get_cached_dataset(self, fname: str, fallback_xfile=None):
        """
        Retrieve cached dataset with memory management.

        Provides memory-efficient access to cached XpcsFile datasets
        using weak references to prevent memory leaks.

        Parameters
        ----------
        fname : str
            Filename identifier for the cached dataset
        fallback_xfile : XpcsFile, optional
            Fallback XpcsFile object if cache miss occurs

        Returns
        -------
        XpcsFile or None
            Cached dataset object or fallback object if available

        Notes
        -----
        Uses WeakValueDictionary for automatic garbage collection
        of unused datasets, helping to manage memory usage for
        large XPCS data collections.
        """
        # The caller's object comes from the locator, which checks the file
        # revision; this cache is keyed by name only, so it must never win.
        if fallback_xfile is not None:
            return fallback_xfile
        with self._dset_cache_lock:
            return self._current_dset_cache.get(fname)

    def plot_intt(self, pg_hdl, rows=None, **kwargs):
        """
        Generate intensity vs. time plots for temporal analysis.

        Creates plots showing intensity variations over time
        for monitoring sample behavior and beam stability.

        Parameters
        ----------
        pg_hdl : PyQtGraph PlotWidget
            PyQtGraph plot widget for real-time visualization
        rows : list, optional
            List of file indices to include in analysis
        **kwargs
            Plotting parameters including sampling, window size, etc.

        Notes
        -----
        Uses the intt module for generating time-series plots
        with configurable sampling rates and analysis windows.
        """
        xf_list = self.get_xf_list(rows=rows)
        _get_module("intt").plot(xf_list, pg_hdl, **kwargs)

    def plot_stability(self, mp_hdl, rows=None, **kwargs):
        """
        Generate sample stability analysis plots.

        Creates plots showing sample stability over time using
        intensity measurements and statistical analysis.

        Parameters
        ----------
        mp_hdl : PlotHandler
            Matplotlib-based plot handler for stability visualization
        rows : list, optional
            List of file indices (uses first file only)
        **kwargs
            Plotting parameters for stability analysis

        Notes
        -----
        Uses the first selected file for stability analysis.
        The stability module provides various metrics for assessing
        sample behavior during data collection.
        """
        xf_list = self.get_xf_list(rows)
        if not xf_list:
            logger.warning("No files available for stability analysis")
            return
        xf_obj = xf_list[0]
        _get_module("stability").plot(xf_obj, mp_hdl, **kwargs)

    def submit_job(self, *args, **kwargs):
        if len(self.target) <= 0:
            logger.error("no average target is selected")
            return
        worker = _get_module("average_toolbox")(self.path, flist=self.target)
        worker.setup(*args, **kwargs)
        self.avg_worker = worker
        logger.info("create average job, ID = %s", worker.jid)
        self.target.clear()

    def remove_job(self):
        self.avg_worker = None

    def update_avg_info(self):
        if self.avg_worker is not None:
            self.avg_worker.update_plot()

    def update_avg_values(self, data):
        """
        Update averaging worker values with memory management.

        Handles dynamic allocation of averaging result arrays with
        memory pressure monitoring and automatic cleanup when needed.

        Parameters
        ----------
        data : tuple
            (worker_key, value) pair for updating averaging results

        Notes
        -----
        - Dynamically resizes result arrays when capacity is exceeded
        - Monitors memory pressure before allocating additional memory
        - Triggers cleanup operations when memory pressure is high
        - Uses efficient numpy arrays for result storage

        The method implements automatic array doubling strategy
        for efficient memory allocation while monitoring system
        memory pressure to prevent out-of-memory conditions.
        """
        key, val = data[0], data[1]
        # .get() so a worker update for a key the GUI never pre-seeded (e.g. when
        # driven headlessly) initializes the record instead of raising KeyError.
        if self.avg_worker_active.get(key) is None:
            self.avg_worker_active[key] = [0, np.zeros(128, dtype=np.float32)]
        record = self.avg_worker_active[key]
        if record[0] == record[1].size:
            # Check memory pressure before allocating more memory
            if MemoryMonitor.is_memory_pressure_high(self._memory_cleanup_threshold):
                logger.warning(
                    "Memory pressure high during averaging, triggering cleanup"
                )
                self._cleanup_memory()

            new_g2 = np.zeros(record[1].size * 2, dtype=np.float32)
            new_g2[0 : record[0]] = record[1]
            record[1] = new_g2
        record[1][record[0]] = val
        record[0] += 1

    def get_memory_stats(self):
        """
        Get comprehensive memory usage statistics for the kernel.

        Returns detailed information about memory usage across different
        components including caches, active workers, and system memory status.

        Returns
        -------
        dict
            Memory statistics including:
            - kernel_cache_items: Number of cached datasets
            - plot_kwargs_items: Number of cached plot parameters
            - avg_worker_active_items: Number of active averaging workers
            - system_memory_used_mb: System memory usage in MB
            - system_memory_available_mb: Available system memory in MB
            - memory_pressure: Current memory pressure ratio (0-1)

        Notes
        -----
        This method is useful for monitoring memory usage during analysis
        and identifying potential memory pressure situations that may
        require cache cleanup or optimization.
        """
        """Get comprehensive memory statistics."""
        used_mb, available_mb = MemoryMonitor.get_memory_usage()
        return {
            "kernel_cache_items": self._get_cache_count(),
            "plot_kwargs_items": len(self._plot_kwargs_record),
            "avg_worker_active_items": len(self.avg_worker_active),
            "system_memory_used_mb": used_mb,
            "system_memory_available_mb": available_mb,
            "memory_pressure": MemoryMonitor.get_memory_pressure(),
        }

    @log_timing(threshold_ms=1000)
    def export_g2(self, folder, rows=None):
        """
        Export G2 correlation data and fitting results to text files.

        Parameters
        ----------
        folder : str
            Directory path where files will be saved
        rows : list, optional
            Selected row indices. If None, uses all files.
        """
        import os

        import numpy as np

        logger.info(
            f"G2 export kernel: Starting export to folder {folder} with rows {rows}"
        )
        xf_list = self.get_xf_list(rows=rows, filter_atype="Multitau")
        logger.info(f"G2 export kernel: Found {len(xf_list)} Multitau files for export")

        if not xf_list:
            logger.error("G2 export kernel: No files with G2 data found")
            raise ValueError("No files with G2 data found")

        successful_exports = 0
        total_files = len(xf_list)
        # One guard for the whole export: no destination may be (a link to) one of
        # the selected input files or to a file this export already wrote.
        guard = ExportGuard(xf.fname for xf in xf_list)

        for xf, stem in zip(xf_list, _export_stems(xf_list), strict=True):
            # Export raw G2 data
            logger.info(
                f"G2 export kernel: Processing file {xf.label} (atype: {getattr(xf, 'atype', 'unknown')})"
            )
            try:
                q_val, t_el, g2, g2_err, _labels = xf.get_g2_data()

                g2_filename = os.path.join(folder, f"{stem}_g2_data.txt")
                fit_filename = os.path.join(folder, f"{stem}_g2_fitting.txt")
                curve_filename = os.path.join(folder, f"{stem}_g2_fitted_curves.txt")
                # Claim every destination of this file before writing any of them
                # (a refused file is skipped whole and counted as failed).
                guard.claim(g2_filename, str(xf.fname))
                if xf.fit_summary is not None:
                    guard.claim(fit_filename, str(xf.fname))
                    if "fit_line" in xf.fit_summary and "fit_x" in xf.fit_summary:
                        guard.claim(curve_filename, str(xf.fname))

                # Absolute identity of each G2 column (None for stand-in files)
                columns = _g2_bin_identities(xf, len(q_val))

                # Create header: the plain Q labels stay, the bin's full identity
                # (dynamic bin, map position, numeric q and phi) is appended
                header = "tau(s)"
                for i, q in enumerate(q_val):
                    tag = _bin_tag(columns[i] if columns else None)
                    header += f"\tG2_q{i:03d}(q={q:.6f}){tag}\tG2_err_q{i:03d}{tag}"

                # Prepare data array
                data_array = [t_el]
                for i in range(len(q_val)):
                    data_array.append(g2[:, i])
                    data_array.append(g2_err[:, i])

                data_matrix = np.column_stack(data_array)
                np.savetxt(g2_filename, data_matrix, header=header, delimiter="\t")
                logger.info(f"G2 export kernel: Saved G2 data file: {g2_filename}")

                # Export fitting results if available
                if xf.fit_summary is not None:
                    fit_q = np.asarray(xf.fit_summary["q_val"])
                    fit_ids = _fit_bin_identities(xf.fit_summary, fit_q, columns)

                    with open(fit_filename, "w") as f:
                        f.write(f"# G2 Fitting Results for {xf.label}\n")
                        f.write(f"# Fitting function: {xf.fit_summary['fit_func']}\n")
                        f.write(f"# Q-range: {xf.fit_summary['q_range']}\n")
                        f.write(f"# Time-range: {xf.fit_summary['t_range']}\n")
                        f.write(f"# Bounds: {xf.fit_summary['bounds']}\n")
                        f.write(f"# Fit flags: {xf.fit_summary['fit_flag']}\n")
                        # The table's columns are unchanged; each row's absolute
                        # bin identity (a radial Q alone is ambiguous) follows here.
                        if fit_ids is not None:
                            for i, ident in enumerate(fit_ids):
                                f.write(_bin_row_comment(i, ident))
                        f.write("\n")

                        # Parameter names
                        if xf.fit_summary["fit_func"] == "single":
                            param_names = ["a", "b", "c", "d"]
                        else:
                            param_names = ["a", "b", "c", "d", "b2", "c2", "f"]

                        # Write header for fitting parameters; fit_status is
                        # "ok" or "failed" (a failed Q bin has nan values, never
                        # solver placeholders)
                        f.write("q_value\t")
                        for param in param_names:
                            f.write(f"{param}\t{param}_error\t")
                        f.write("fit_status\n")

                        # Write fitting parameters
                        # row i of the fit belongs to the fit's own q_val, which
                        # is a subset of the exported q_val when the fit was cropped
                        fit_val = xf.fit_summary["fit_val"]
                        from .fitting.legacy import fit_success_mask

                        fit_ok = fit_success_mask(xf.fit_summary)
                        for i, q in enumerate(fit_q):
                            if i < fit_val.shape[0]:
                                f.write(f"{q:.6f}\t")
                                for j in range(len(param_names)):
                                    if j >= fit_val.shape[2]:
                                        f.write("N/A\tN/A\t")
                                    elif fit_ok[i]:
                                        f.write(
                                            f"{fit_val[i, 0, j]:.6e}\t{fit_val[i, 1, j]:.6e}\t"
                                        )
                                    else:
                                        f.write("nan\tnan\t")
                                f.write("ok\n" if fit_ok[i] else "failed\n")
                        n_failed = int(np.count_nonzero(~fit_ok[: len(fit_q)]))
                        if n_failed:
                            logger.warning(
                                f"G2 export kernel: {n_failed} failed Q fit(s) for "
                                f"{xf.label} written as nan with fit_status=failed"
                            )
                    logger.info(
                        f"G2 export kernel: Saved fitting results file: {fit_filename}"
                    )

                    # Export fitted curves if available
                    if "fit_line" in xf.fit_summary and "fit_x" in xf.fit_summary:
                        fit_x = xf.fit_summary["fit_x"]
                        fit_line = xf.fit_summary["fit_line"]

                        # Create header for fitted curves
                        # only curves that exist; one column per fitted q
                        n_curves = min(len(fit_q), fit_line.shape[0])
                        header = "tau(s)"
                        for i in range(n_curves):
                            header += f"\tG2_fit_q{i:03d}(q={fit_q[i]:.6f})"
                            if fit_ids is not None:
                                header += _bin_tag(fit_ids[i])
                            if not fit_ok[i]:
                                header += "[failed]"

                        curve_data = [fit_x]
                        curve_data.extend(
                            fit_line[i, :] if fit_ok[i] else np.full(len(fit_x), np.nan)
                            for i in range(n_curves)
                        )

                        curve_matrix = np.column_stack(curve_data)
                        np.savetxt(
                            curve_filename, curve_matrix, header=header, delimiter="\t"
                        )
                        logger.info(
                            f"G2 export kernel: Saved fitted curves file: {curve_filename}"
                        )

                successful_exports += 1
                logger.info(f"Successfully exported G2 data for {xf.label}")

            except ExportCollisionError as e:
                # This file is refused whole; the others are still exported
                logger.error(f"G2 export of {xf.label} refused: {e}")
                continue
            except ValueError as e:
                # Handle specific data validation errors (like invalid Q indices)
                logger.error(f"Data validation error for {xf.label}: {e}")
                continue
            except (OSError, PermissionError) as e:
                # Handle file system errors
                logger.error(f"File system error exporting {xf.label}: {e}")
                raise  # Re-raise file system errors as they affect the entire export
            except Exception as e:
                # Handle unexpected errors with more detail
                logger.error(
                    f"Unexpected error exporting G2 data for {xf.label}: {type(e).__name__}: {e}"
                )
                import traceback

                logger.debug(f"Full traceback for {xf.label}: {traceback.format_exc()}")
                continue

        # Export summary
        if successful_exports == total_files:
            logger.info(
                f"G2 export completed successfully: {successful_exports}/{total_files} files exported"
            )
        elif successful_exports > 0:
            logger.warning(
                f"G2 export partially completed: {successful_exports}/{total_files} files exported successfully"
            )
        else:
            raise ValueError(
                f"G2 export failed: 0/{total_files} files exported successfully"
            )

    @log_timing(threshold_ms=1000)
    def export_diffusion(self, folder, rows=None):
        """
        Export tau-q power law fitting results to text files.

        Parameters
        ----------
        folder : str
            Directory path where files will be saved
        rows : list, optional
            Selected row indices. If None, uses all files.
        """
        import os
        import time

        # same fit selection as the plot (_diffusion_fit): Bayesian when present
        xf_list = [
            xf
            for xf in self.get_xf_list(rows=rows, filter_atype="Multitau")
            if _diffusion_fit(xf)[0] is not None
        ]

        if not xf_list:
            raise ValueError("No files with tau-q fitting results found")

        # tauq_success is the single truth: a failed fit has no valid parameters
        valid_xf_list = []
        fits: dict[int, tuple[dict, str]] = {}  # id(xf) -> (summary, method)
        n_failed = 0
        for xf in xf_list:
            fs, method = _diffusion_fit(xf)
            if not fs or not ("tauq_fit_val" in fs or "tauq_success" in fs):
                continue
            if fs.get("tauq_success", False):
                valid_xf_list.append(xf)
                fits[id(xf)] = (fs, method)
            else:
                n_failed += 1
                logger.error(
                    f"Tau-q fit failed for {xf.label}; fit parameters not exported"
                )

        if not valid_xf_list:
            if n_failed:
                raise ValueError(
                    f"Tau-q power law fit failed for all {n_failed} selected files; "
                    "nothing to export."
                )
            raise ValueError(
                "No files with tau-q power law fitting results found. Please perform diffusion analysis first."
            )

        successful_exports = 0
        total_files = len(valid_xf_list) + n_failed
        written = []  # files whose individual result was written (summary rows)
        # One guard for the whole export (inputs: every selected file): no
        # destination may be (a link to) an input or an earlier output of this run.
        guard = ExportGuard(xf.fname for xf in xf_list)

        # Export individual file results
        for xf, stem in zip(valid_xf_list, _export_stems(valid_xf_list), strict=True):
            try:
                filename = guard.claim(
                    os.path.join(folder, f"{stem}_diffusion_fitting.txt"),
                    str(xf.fname),
                )

                fs, method = fits[id(xf)]
                tauq_data = fs[
                    "tauq_fit_val"
                ]  # Shape: (2, 2) - [[a_val, b_val], [a_err, b_err]]

                with open(filename, "w") as f:
                    f.write(f"# Power Law Fitting Results for {xf.label}\n")
                    f.write("# Model: tau = a * q^b\n")
                    f.write(f"# File: {xf.fname}\n")
                    f.write(f"# Fit method: {method}\n")
                    for line in _DIFFUSION_PROVENANCE[method]:
                        f.write(f"# {line}\n")
                    f.write(f"# G2 fit function: {fs.get('fit_func', 'N/A')}\n")
                    if fs.get("fname") is not None:  # Bayesian summaries record it
                        f.write(f"# G2 fit source file: {fs['fname']}\n")
                    f.write(f"# Q-range: {fs.get('tauq_q_range', 'N/A')}\n")
                    f.write(f"# Bounds: {fs.get('tauq_bounds', 'N/A')}\n")
                    f.write(f"# Fit flags: {fs.get('tauq_fit_flag', 'N/A')}\n\n")

                    f.write("parameter\tvalue\terror\n")
                    f.write(f"a\t{tauq_data[0, 0]:.6e}\t{tauq_data[1, 0]:.6e}\n")
                    f.write(f"b\t{tauq_data[0, 1]:.6f}\t{tauq_data[1, 1]:.6f}\n")

                successful_exports += 1
                written.append(xf)
                logger.info(
                    f"Successfully exported diffusion fitting results for {xf.label}"
                )

            except ExportCollisionError as e:
                logger.error(f"Diffusion export of {xf.label} refused: {e}")
                continue
            except (OSError, PermissionError) as e:
                logger.error(
                    f"File system error exporting diffusion data for {xf.label}: {e}"
                )
                raise  # Re-raise file system errors as they affect the entire export
            except Exception as e:
                logger.error(
                    f"Unexpected error exporting diffusion data for {xf.label}: {type(e).__name__}: {e}"
                )
                import traceback

                logger.debug(f"Full traceback for {xf.label}: {traceback.format_exc()}")
                continue

        if not written:  # nothing to summarise
            raise ValueError(
                f"Diffusion export failed: 0/{total_files} files exported successfully"
            )

        # Export summary table
        try:
            summary_filename = guard.claim(
                os.path.join(folder, "diffusion_summary.txt"), "diffusion summary"
            )

            with open(summary_filename, "w") as f:
                f.write("# Power Law Fitting Summary\n")
                f.write("# Model: tau = a * q^b\n")
                f.write(f"# Export date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Total files: {len(written)}\n")
                for method in sorted({m for _, m in fits.values()}):
                    f.write(f"# Fit method {method}:\n")
                    for line in _DIFFUSION_PROVENANCE[method]:
                        f.write(f"#   {line}\n")
                f.write("\n")

                f.write(
                    "file_label\tfit_method\ta_value\ta_error\tb_value\tb_error\t"
                    "file_name\n"
                )

                logger.info(f"Creating summary table for {len(written)} files")
                for xf in written:
                    fs, method = fits[id(xf)]
                    if "tauq_fit_val" in fs:
                        tauq_data = fs["tauq_fit_val"]
                        f.write(
                            f"{xf.label}\t{method}\t{tauq_data[0, 0]:.6e}\t{tauq_data[1, 0]:.6e}\t{tauq_data[0, 1]:.6f}\t{tauq_data[1, 1]:.6f}\t{xf.fname}\n"
                        )
                        logger.info(f"Added summary data for {xf.label}")
                    else:
                        logger.warning(f"No tauq_fit_val found for {xf.label}")

            logger.info(f"Summary table exported to {summary_filename}")

        except ExportCollisionError as e:
            logger.error(f"Diffusion summary table refused: {e}")
        except Exception as e:
            logger.warning(f"Failed to create summary table: {e}")
            # Don't fail the entire export if summary fails

        # Export summary
        if successful_exports == total_files:
            logger.info(
                f"Diffusion export completed successfully: {successful_exports}/{total_files} files exported"
            )
        elif successful_exports > 0:
            logger.warning(
                f"Diffusion export partially completed: {successful_exports}/{total_files} files exported successfully"
            )
        else:
            raise ValueError(
                f"Diffusion export failed: 0/{total_files} files exported successfully"
            )


if __name__ == "__main__":
    flist = os.listdir("./data")
    dv = ViewerKernel("./data", flist)
