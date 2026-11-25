"""Core data container for XPCS datasets.

This module provides the XpcsFile class, the primary interface for loading
and accessing XPCS data from HDF5/NeXus files.

Example:
    >>> from xpcsviewer import XpcsFile
    >>> xf = XpcsFile('data.h5')
    >>> print(f"Analysis type: {xf.atype}")
    >>> print(f"G2 shape: {xf.g2.shape}")
"""

from __future__ import annotations

# Standard library imports
import hashlib
import os
import re
import uuid
import warnings
from typing import ClassVar

# Third-party imports
import numpy as np
import pyqtgraph as pg

from xpcsviewer.constants import (
    DOUBLE_EXP_PARAMS,
    MAX_PARALLEL_WORKERS,
    MEMORY_EFFICIENCY_HIGH,
    MEMORY_EFFICIENCY_LOW,
    MEMORY_EFFICIENCY_MEDIUM,
    MEMORY_WARNING_THRESHOLD_MB,
    MIN_DISPLAY_POINTS,
    MIN_DOWNSAMPLE_POINTS,
    NDIM_3D,
    SINGLE_EXP_PARAMS,
    WORKER_THRESHOLD_LARGE,
    WORKER_THRESHOLD_MEDIUM,
)

# Import extracted utilities from xpcs_file subpackage
from xpcsviewer.xpcs_file.memory import MemoryMonitor

# Local imports
from .fileIO.hdf_reader import (
    _file_identity,
    batch_read_fields,
    get,
    get_analysis_type,
    get_chunked_dataset,
    get_file_info,
    read_metadata_to_dict,
)
from .fileIO.hdf_reader_enhanced import get_enhanced_hdf5_reader
from .fileIO.qmap_utils import Q_UNIT_DISPLAY, get_qmap
from .module.twotime_utils import get_c2_stream, get_single_c2_from_hdf
from .utils.exceptions import XPCSFileError, convert_exception
from .utils.export_guard import ExportGuard
from .utils.lazy_loader import LazyHDF5Array, get_lazy_loader
from .utils.log_utils import log_timing
from .utils.logging_config import get_logger
from .utils.memory_manager import CacheType, MemoryPressure, get_memory_manager
from .utils.memory_predictor import (
    get_memory_predictor,
    predict_operation_memory,
    record_operation_memory,
)
from .utils.streaming_processor import AdaptiveChunkSizer, process_saxs_log_streaming

logger = get_logger(__name__)

# OPT-004: module-level JIT-compiled SAXS log kernel.
# Defined once at module level so JAX reuses the same compiled XLA program
# across all calls — avoids per-call recompilation overhead.
# Lazy: stays None until first successful JAX import.
_jax_saxs_log_fn = None


def _get_jax_saxs_log():
    """Return the JAX-jitted SAXS log kernel, compiling it on first call."""
    global _jax_saxs_log_fn
    if _jax_saxs_log_fn is None:
        import jax
        import jax.numpy as jnp

        @jax.jit
        def _kernel(data):
            pos_mask = data > 0
            min_pos = jnp.where(pos_mask, data, jnp.inf).min()
            # Guard: if no positive values, substitute 1.0 to avoid log10(inf)
            min_pos = jnp.where(jnp.isinf(min_pos), 1.0, min_pos)
            # Match NumPy behavior: replace non-positive pixels with min_pos
            # (same as np.maximum(data, min_val)) then take log10 uniformly.
            safe = jnp.maximum(data, min_pos)
            return jnp.log10(safe).astype(jnp.float32)

        _jax_saxs_log_fn = _kernel
    return _jax_saxs_log_fn


# Module-level cached singletons — avoid per-instance lookups in __init__.
_cached_memory_manager = None
_cached_memory_predictor = None
_cached_lazy_loader = None
_cached_enhanced_hdf5_reader = None


def _get_cached_singletons():
    """Return cached (memory_manager, memory_predictor, lazy_loader, hdf5_reader)."""
    global _cached_memory_manager, _cached_memory_predictor
    global _cached_lazy_loader, _cached_enhanced_hdf5_reader
    if _cached_memory_manager is None:
        _cached_memory_manager = get_memory_manager()
        _cached_memory_predictor = get_memory_predictor()
        _cached_lazy_loader = get_lazy_loader()
        _cached_enhanced_hdf5_reader = get_enhanced_hdf5_reader()
    return (
        _cached_memory_manager,
        _cached_memory_predictor,
        _cached_lazy_loader,
        _cached_enhanced_hdf5_reader,
    )


def _tauq_failure() -> dict:
    """tau-q result for a failed fit: overwrites every field of an earlier fit.

    The observation arrays are emptied too, so no stale point from a previous
    q range survives to be plotted or exported.
    """
    return {
        "tauq_success": False,
        "tauq_q": np.empty(0),
        "tauq_tau": np.empty(0),
        "tauq_tau_err": np.empty(0),
        "tauq_fit_line": None,
        "tauq_fit_val": np.full((2, 2), np.nan),  # rows: value, error; cols: a, b
    }


def _positive_delay_fit_data(t_el, g2, sigma):
    """Restrict a G2 fit to finite, strictly positive delays.

    Rule: the models are functions of tau / delay with a non-integer stretch
    exponent, so delay 0 (e.g. the lag-0 point of a two-time file) gives an
    invalid derivative, and the fit grid is logarithmic. Such points, and any
    non-finite delay, are dropped before fitting and the count is logged.
    """
    t = np.asarray(t_el, dtype=float)
    keep = np.isfinite(t) & (t > 0)
    n_drop = int(t.size - np.count_nonzero(keep))
    if n_drop:
        logger.warning(
            "G2 fit: dropped %d of %d delay point(s) that are non-positive or "
            "non-finite (fits use finite, strictly positive delays only)",
            n_drop,
            t.size,
        )
    if not keep.any():
        raise ValueError(
            "Cannot fit G2: no finite positive delays in the selected range"
        )
    return t[keep], np.asarray(g2)[keep], np.asarray(sigma)[keep]


def _blank_failed_fits(fit_val, fit_line, ok):
    """Copies of ``fit_val`` / ``fit_line`` with NaN rows for every failed Q.

    Solver placeholders (bounds midpoints, zero errors, last iterates of a
    non-converged solve) are not measurements, so nothing downstream can
    mistake them for values. A "successful" row with non-finite parameters is
    failed too. Returns ``(fit_val, fit_line, ok)`` with ``ok`` a bool array.
    """
    fit_val = np.array(fit_val, dtype=float)
    fit_line = np.array(fit_line, dtype=float)
    ok = np.asarray(ok, dtype=bool).copy()
    ok &= np.isfinite(fit_val[:, 0, :]).all(axis=1)
    fit_val[~ok] = np.nan
    fit_line[~ok] = np.nan
    n_failed = int(np.count_nonzero(~ok))
    if n_failed:
        logger.warning(
            "G2 fit failed for %d of %d Q bin(s); their values are NaN",
            n_failed,
            ok.size,
        )
    return fit_val, fit_line, ok


def _g2_fit_digest(fit_summary: dict) -> str:
    """Identity of the G2 fit that a tau-q fit is derived from (q and fit values)."""
    h = hashlib.md5(usedforsecurity=False)
    for key in ("q_val", "fit_val"):
        try:
            arr = np.ascontiguousarray(
                np.asarray(fit_summary.get(key, ()), dtype=float)
            )
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
        except (TypeError, ValueError):  # non-numeric entries: hash their repr
            h.update(repr(fit_summary.get(key)).encode())
    return h.hexdigest()


def create_id(fname, label_style=None, simplify_flag=True):
    """
    Generate a simplified or customized ID string from a filename.

    Parameters
    ----------
    fname : str
        Input file name, possibly with path and extension.
    label_style : str or None, optional
        Comma-separated string of indices to extract specific components from the file name.
    simplify_flag : bool, optional
        Whether to simplify the file name by removing leading zeros and stripping suffixes.

    Returns
    -------
    str
        A simplified or customized ID string derived from the input filename.
    """
    fname = os.path.basename(fname)

    if simplify_flag:
        # Remove leading zeros from structured parts like '_t0600' → '_t600'
        fname = re.sub(r"_(\w)0+(\d+)", r"_\1\2", fname)
        # Remove trailing _results and file extension
        fname = re.sub(r"(_results)?\.hdf$", "", fname, flags=re.IGNORECASE)

    if len(fname) < MIN_DISPLAY_POINTS or not label_style:
        return fname

    try:
        selection = [int(x.strip()) for x in label_style.split(",")]
        if not selection:
            warnings.warn(
                "Empty label_style selection. Returning simplified filename.",
                stacklevel=2,
            )
            return fname
    except ValueError:
        warnings.warn(
            "Invalid label_style format. Must be comma-separated integers.",
            stacklevel=2,
        )
        return fname

    segments = fname.split("_")
    selected_segments = []

    for i in selection:
        if i < len(segments):
            selected_segments.append(segments[i])
        else:
            warnings.warn(
                f"Index {i} out of range for segments {segments}", stacklevel=2
            )

    if not selected_segments:
        return fname  # fallback if nothing valid was selected

    return "_".join(selected_segments)


class XpcsFile:
    """
    XpcsFile is a class that wraps an Xpcs analysis hdf file;
    """

    _QMAP_KEYS = frozenset(
        {
            "sqlist",
            "dqlist",
            "dqmap",
            "sqmap",
            "mask",
            "bcx",
            "bcy",
            "det_dist",
            "pixel_size",
            "X_energy",
            "splist",
            "dplist",
            "static_num_pts",
            "dynamic_num_pts",
            "map_names",
            "map_units",
            "get_qbin_label",
        }
    )

    _G2_PARTIAL_KEYS = frozenset({"g2_partial", "g2_partial_err", "g2_partial_labels"})

    # An imported Q-map partition (SimpleMask) is DISPLAY ONLY. xf.dqmap and
    # xf.sqmap always stay the file's own maps (delegated to xf.qmap) for every
    # selection and numerical path; only the designated Q-map displays read the
    # import, through get_display_qmap().
    display_dqmap = None
    display_sqmap = None

    def get_display_qmap(self, target):
        """Map for the Q-map displays: the imported partition, else the file's own.

        ``target`` is "dqmap" (dynamic) or "sqmap" (static).
        """
        if target not in ("dqmap", "sqmap"):
            raise ValueError(f"target must be 'dqmap' or 'sqmap', got '{target}'")
        override = getattr(self, f"display_{target}")
        return getattr(self, target) if override is None else override

    def __init__(self, fname, fields=None, label_style=None, qmap_manager=None):
        self.fname = fname
        if qmap_manager is None:
            self.qmap = get_qmap(self.fname)
        else:
            self.qmap = qmap_manager.get_qmap(self.fname)
        self.atype = get_analysis_type(self.fname)
        self.label = self.update_label(label_style)
        payload_dictionary = self.load_data(fields)
        # Guard against HDF5 payload overwriting existing instance attributes.
        # Attributes set before this point (e.g. qmap, label, atype, fname)
        # are owned by the instance and must not be silently overwritten by
        # arbitrary keys returned from the HDF5 file. (BUG-015)
        _PROTECTED = frozenset(self.__dict__.keys())
        for k, v in payload_dictionary.items():
            if k == "c2_processed_bins" and v is not None:
                # a single processed bin stored as shape (1, 1) is read back as a
                # Python scalar; consumers expect a 1-D sequence of bin IDs
                v = np.atleast_1d(np.ravel(v))
            if k not in _PROTECTED:
                self.__dict__[k] = v
        self.hdf_info = None
        self.fit_summary = None
        self.bayesian_fit_summary = None  # Bayesian posterior means (legacy format)
        self.bayesian_results = None  # dict[int, FitResult] per-Q posteriors
        self.c2_all_data = None
        self.c2_kwargs = None

        # Generate a single UUID for this instance (used for cleanup registration and cache prefix)
        self._instance_id = str(uuid.uuid4())

        # Register for optimized cleanup
        try:
            from .threading.cleanup_optimized import register_for_cleanup

            register_for_cleanup(f"XpcsFile_{self._instance_id}", self)
        except ImportError:
            pass  # Optimized cleanup system not available

        # Placeholder for lazy-loaded saxs_2d data
        self._saxs_2d_data = None
        self._saxs_2d_log_data = None
        self._saxs_data_loaded = False

        # Use module-level cached singletons instead of per-instance lookups
        mm, mp, ll, hr = _get_cached_singletons()
        self._memory_manager = mm
        self._cache_prefix = f"xpcs_{self._instance_id}"
        self._memory_predictor = mp
        self._lazy_loader = ll
        self._use_lazy_loading = True  # Enable by default for large datasets
        self._hdf5_reader = hr
        self._use_enhanced_hdf5 = True  # Enable enhanced HDF5 reading

    @property
    def saxs_2d_data(self):
        """
        Access SAXS 2D data with transparent lazy loading support.

        Returns either regular numpy array or lazy-loaded array proxy.
        The lazy proxy supports array operations and slicing.
        """
        return self._saxs_2d_data

    @saxs_2d_data.setter
    def saxs_2d_data(self, value):
        """Set SAXS 2D data (regular array or lazy proxy)."""
        self._saxs_2d_data = value

    @property
    def saxs_2d_log_data(self):
        """
        Access SAXS 2D log data with transparent lazy loading support.
        """
        return self._saxs_2d_log_data

    @saxs_2d_log_data.setter
    def saxs_2d_log_data(self, value):
        """Set SAXS 2D log data (regular array or lazy proxy)."""
        self._saxs_2d_log_data = value

    @property
    def saxs_2d(self):
        """
        Backward compatibility property for SAXS 2D data access.

        This property ensures existing code continues to work while providing
        lazy loading benefits. It automatically loads data when accessed if
        using lazy loading.
        """
        if self._saxs_2d_data is None:
            # Try to load SAXS data if not already loaded
            if not self._saxs_data_loaded:
                self._load_saxs_data_batch()
            return self._saxs_2d_data

        # Handle lazy-loaded data by converting to array when needed
        if isinstance(self._saxs_2d_data, LazyHDF5Array):
            # This will trigger loading if not already loaded
            return np.array(self._saxs_2d_data)
        return self._saxs_2d_data

    @saxs_2d.setter
    def saxs_2d(self, value):
        """Set SAXS 2D data (backward compatibility)."""
        self._saxs_2d_data = value

    def update_label(self, label_style):
        self.label = create_id(self.fname, label_style=label_style)
        return self.label

    def __str__(self):
        ans = ["File:" + str(self.fname)]
        for key, val in self.__dict__.items():
            # omit those to avoid lengthy output
            if key == "hdf_info":
                continue
            display_val = (
                str(val.shape)
                if isinstance(val, np.ndarray) and val.size > 1
                else str(val)
            )
            ans.append(f"   {key.ljust(12)}: {display_val.ljust(30)}")
        return "\n".join(ans)

    def __repr__(self):
        ans = str(type(self))
        ans = "\n".join([ans, self.__str__()])
        return ans

    def __enter__(self):
        """
        Enter context manager.

        Returns:
            self: The XpcsFile instance for use in with statement

        Example:
            >>> with XpcsFile("data.hdf5") as xfile:
            ...     data = xfile.get_g2_data()
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit context manager, ensuring cleanup of resources.

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred

        Returns:
            False: Don't suppress exceptions
        """
        self.close()
        return False  # Don't suppress exceptions

    def close(self):
        """
        Explicitly close and clean up all resources held by this XpcsFile.

        This method clears all caches, releases large data arrays, and prepares
        the object for garbage collection. Safe to call multiple times.

        Note:
            HDF5 file handles are managed by the global connection pool and
            will be closed automatically when evicted from the pool.
        """
        try:
            # Clear all internal caches
            self.clear_cache()

            # Release large data arrays
            self._saxs_2d_data = None
            self._saxs_2d_log_data = None
            self._saxs_data_loaded = False

            # Clear other cached data
            self.c2_all_data = None

            logger.debug(f"Closed XpcsFile: {self.fname}")
        except Exception as e:
            logger.warning(f"Error during XpcsFile cleanup: {e}")

    def get_hdf_info(self, fstr=None):
        """
        get a text representation of the xpcs file; the entries are organized
        in a tree structure;
        :param fstr: list of filter strings, ["string_1", "string_2", ...]
        :return: a list strings
        """
        # cache the data because it may take long time to generate the str
        if self.hdf_info is None:
            self.hdf_info = read_metadata_to_dict(self.fname)
        return self.hdf_info

    @log_timing(threshold_ms=500)
    def load_data(self, extra_fields=None):
        # default common fields for both twotime and multitau analysis;
        fields = ["saxs_1d", "Iqp", "Int_t", "t0", "t1", "start_time"]

        if "Multitau" in self.atype:
            fields = [*fields, "tau", "g2", "g2_err", "stride_frame", "avg_frame"]
        if "Twotime" in self.atype:
            fields = [
                *fields,
                "c2_g2",
                "c2_g2_segments",
                "c2_processed_bins",
                "c2_stride_frame",
                "c2_avg_frame",
            ]

        # append other extra fields, eg "G2", "IP", "IF"
        if isinstance(extra_fields, list):
            fields += extra_fields

        # avoid duplicated keys
        fields = list(set(fields))

        # Use enhanced HDF5 reader for optimized batch reading
        # Reuse the cached singleton set in __init__ to avoid repeated sys.modules lookups.
        try:
            enhanced_reader = self._hdf5_reader
            if enhanced_reader is None:
                raise AttributeError("No enhanced HDF5 reader available")
            batch_cache_key = f"{self._cache_prefix}_batch_metadata"

            logger.debug(
                f"Loading metadata fields using enhanced HDF5 reader: {fields}"
            )
            ret = enhanced_reader.read_multiple_datasets(
                self.fname,
                fields,
                cache_key=batch_cache_key,
                access_pattern="sequential",
                use_aliases=True,
            )

        except (ImportError, ModuleNotFoundError, AttributeError) as e:
            # Expected - enhanced reader not available, use fallback
            logger.debug(
                f"Enhanced HDF5 reader not available, using standard method: {e}"
            )
            ret = batch_read_fields(
                self.fname, fields, "alias", ftype="nexus", use_pool=True
            )
        except (OSError, PermissionError) as e:
            # File access issues - convert to XPCSFileError
            raise XPCSFileError(
                f"Failed to read metadata from HDF5 file: {e}", file_path=self.fname
            ).add_recovery_suggestion("Check file permissions and disk space") from e
        except Exception as enhanced_error:
            # Unexpected enhanced reader errors - log and fallback
            xpcs_error = convert_exception(
                enhanced_error, "Enhanced HDF5 reader failed for metadata loading"
            )
            logger.warning(
                f"Enhanced reader error: {xpcs_error}, falling back to standard method"
            )
            try:
                ret = batch_read_fields(
                    self.fname, fields, "alias", ftype="nexus", use_pool=True
                )
            except Exception as fallback_error:
                # Both methods failed - this is critical
                raise (
                    XPCSFileError(
                        "Both enhanced and standard HDF5 readers failed",
                        file_path=self.fname,
                    )
                    .add_context("enhanced_error", str(enhanced_error))
                    .add_context("fallback_error", str(fallback_error))
                    .add_recovery_suggestion("Verify HDF5 file integrity")
                ) from fallback_error

        if "Twotime" in self.atype:
            stride_frame = ret.pop("c2_stride_frame")
            avg_frame = ret.pop("c2_avg_frame")
            ret["c2_t0"] = ret["t0"] * stride_frame * avg_frame
        if "Multitau" in self.atype:
            # correct g2_err to avoid fitting divergence
            # ret['g2_err_mod'] = self.correct_g2_err(ret['g2_err'])
            if "g2_err" in ret:
                ret["g2_err"] = self.correct_g2_err(ret["g2_err"])
            if "stride_frame" in ret and "avg_frame" in ret:
                stride_frame = ret.pop("stride_frame")
                avg_frame = ret.pop("avg_frame")
                ret["t0"] = ret["t0"] * stride_frame * avg_frame
                if "tau" in ret:
                    ret["t_el"] = ret["tau"] * ret["t0"]
                ret["g2_t0"] = ret["t0"]

        # Handle qmap reshaping with fallback for when qmap is a dict (minimal mode)
        if hasattr(self.qmap, "reshape_phi_analysis"):
            if "saxs_1d" in ret:
                ret["saxs_1d"] = self.qmap.reshape_phi_analysis(
                    ret["saxs_1d"], self.label, mode="saxs_1d"
                )
            if "Iqp" in ret:
                ret["Iqp"] = self.qmap.reshape_phi_analysis(
                    ret["Iqp"], self.label, mode="stability"
                )
        else:
            # Fallback for minimal qmap mode (when qmap is a dict)
            if "saxs_1d" in ret:
                ret["saxs_1d"] = (
                    np.array(ret["saxs_1d"])
                    if isinstance(ret["saxs_1d"], (list, np.ndarray))
                    and len(ret["saxs_1d"]) > 0
                    else ret["saxs_1d"]
                )
            if "Iqp" in ret:
                ret["Iqp"] = (
                    np.array(ret["Iqp"])
                    if isinstance(ret["Iqp"], (list, np.ndarray))
                    and len(ret["Iqp"]) > 0
                    else ret["Iqp"]
                )

        ret["abs_cross_section_scale"] = 1.0
        return ret

    def _load_saxs_data_batch(
        self,
        use_memory_mapping: bool | None = None,
        chunk_processing: bool | None = None,
    ):
        """
        Enhanced SAXS 2D data loading with chunked processing and memory mapping support.

        Parameters
        ----------
        use_memory_mapping : bool, optional
            Whether to use memory mapping for very large files. If None, decides automatically.
        chunk_processing : bool, optional
            Whether to use chunked processing for log computation. If None, decides automatically.
        """
        if self._saxs_data_loaded:
            return

        # Early cache check: if the data is already in the unified memory manager
        # cache, skip all memory-pressure bookkeeping and file-info I/O.
        _early_cache_key = f"{self._cache_prefix}_saxs_2d"
        _cached_early = self._memory_manager.cache_get(
            _early_cache_key, CacheType.ARRAY_DATA
        )
        if _cached_early is not None:
            logger.info(
                "Using cached SAXS 2D data from unified memory manager (early check)"
            )
            self.saxs_2d_data = _cached_early
            _log_cache_key = f"{self._cache_prefix}_saxs_2d_log"
            _cached_log_early = self._memory_manager.cache_get(
                _log_cache_key, CacheType.ARRAY_DATA
            )
            if _cached_log_early is not None:
                self.saxs_2d_log_data = _cached_log_early
            else:
                self.saxs_2d_log_data = self._compute_saxs_log_jax(self.saxs_2d_data)
            self._saxs_data_loaded = True
            return

        # Check memory pressure before loading large data
        memory_before_mb, _available_mb = MemoryMonitor.get_memory_usage()
        memory_pressure = MemoryMonitor.get_memory_pressure()

        if memory_pressure > MEMORY_EFFICIENCY_HIGH:
            logger.warning(
                f"High memory pressure ({memory_pressure * 100:.1f}%) before loading SAXS data"
            )
            # Cache cleanup is handled by the global memory manager when available.

        # Get file information to make intelligent loading decisions
        try:
            file_info = get_file_info(self.fname, use_pool=True)
            saxs_2d_info = None

            # Find saxs_2d dataset info
            for dataset_info in file_info.get("large_datasets", []):
                if "scattering_2d" in dataset_info["path"]:
                    saxs_2d_info = dataset_info
                    break

            if saxs_2d_info:
                estimated_saxs_size_mb = saxs_2d_info["estimated_size_mb"]
                logger.info(
                    f"SAXS 2D dataset estimated size: {estimated_saxs_size_mb:.1f}MB"
                )
            else:
                estimated_saxs_size_mb = 0

        except Exception as e:
            logger.warning(f"Could not get file info for optimization: {e}")
            estimated_saxs_size_mb = 0

        # Decide on loading strategy based on file size and available memory
        if use_memory_mapping is None:
            # Use memory mapping for very large files (>1GB) when memory pressure is high
            use_memory_mapping = (
                estimated_saxs_size_mb > WORKER_THRESHOLD_LARGE
                and memory_pressure > MEMORY_EFFICIENCY_LOW
            )

        if chunk_processing is None:
            # Use chunk processing for large files or when memory pressure is high
            chunk_processing = (
                estimated_saxs_size_mb > WORKER_THRESHOLD_MEDIUM
                or memory_pressure > MEMORY_EFFICIENCY_MEDIUM
            )

        # Predictive memory management before loading
        memory_before_mb, _ = MemoryMonitor.get_memory_usage()

        # Predict memory requirements for SAXS loading
        prediction = predict_operation_memory("load_saxs_2d", estimated_saxs_size_mb)

        logger.info(
            f"SAXS 2D loading prediction for {self.fname}: "
            f"estimated data: {estimated_saxs_size_mb:.1f}MB, "
            f"predicted memory: {prediction.predicted_mb:.1f}MB, "
            f"pressure: {prediction.pressure_level.value}, "
            f"confidence: {prediction.confidence:.2f}"
        )

        # Act on memory pressure predictions
        if prediction.pressure_level in [MemoryPressure.HIGH, MemoryPressure.CRITICAL]:
            logger.warning("High memory pressure predicted for SAXS loading")
            for action in prediction.recommended_actions:
                logger.warning(f"Recommendation: {action}")

            # Proactive cleanup based on predictions
            if prediction.pressure_level == MemoryPressure.CRITICAL:
                logger.warning(
                    "Critical memory pressure - performing emergency cleanup"
                )
                self._memory_manager._emergency_cleanup()
            else:
                logger.warning("High memory pressure - performing aggressive cleanup")
                self._memory_manager._aggressive_cleanup()

        # No lazy branch: LazyHDF5Array lacks ndim/size/dtype that callers use, and
        # the log image needs the full array anyway. Always use the chunked/eager
        # readers below (they honour use_memory_mapping/chunk_processing).

        # Check if we should use unified memory manager cache
        cache_key = f"{self._cache_prefix}_saxs_2d"
        cached_data = self._memory_manager.cache_get(cache_key, CacheType.ARRAY_DATA)

        if cached_data is not None:
            logger.info("Using cached SAXS 2D data from unified memory manager")
            self.saxs_2d_data = cached_data
        else:
            # Load saxs_2d data using optimized method
            try:
                if use_memory_mapping and estimated_saxs_size_mb > 1000:
                    logger.info(
                        f"Loading large SAXS dataset ({estimated_saxs_size_mb:.1f}MB) with enhanced memory mapping"
                    )
                    try:
                        # Use enhanced HDF5 reader with chunked access
                        from .fileIO.aps_8idi import key as hdf_key
                        from .fileIO.hdf_reader_enhanced import get_enhanced_reader

                        enhanced_reader = get_enhanced_reader()
                        saxs_path = hdf_key["nexus"]["saxs_2d"]

                        logger.debug(
                            f"Loading large SAXS dataset using enhanced chunked reader: {saxs_path}"
                        )
                        self.saxs_2d_data = enhanced_reader.read_dataset(
                            self.fname, saxs_path, enable_read_ahead=True
                        )

                    except Exception as enhanced_error:
                        logger.warning(
                            f"Enhanced chunked reader failed, falling back to standard method: {enhanced_error}"
                        )
                        # Fallback to original chunked dataset reading
                        from .fileIO.aps_8idi import key as hdf_key

                        saxs_path = hdf_key["nexus"]["saxs_2d"]
                        self.saxs_2d_data = get_chunked_dataset(
                            self.fname, saxs_path, use_pool=True
                        )
                else:
                    # Use enhanced HDF5 reader with intelligent caching
                    try:
                        from .fileIO.hdf_reader_enhanced import get_enhanced_reader

                        enhanced_reader = get_enhanced_reader()

                        # Get the SAXS 2D dataset path
                        from .fileIO.aps_8idi import key as hdf_key

                        saxs_path = hdf_key["nexus"]["saxs_2d"]

                        logger.debug(
                            f"Loading SAXS 2D data using enhanced HDF5 reader: {saxs_path}"
                        )
                        self.saxs_2d_data = enhanced_reader.read_dataset(
                            self.fname, saxs_path, enable_read_ahead=True
                        )

                    except Exception as enhanced_error:
                        logger.warning(
                            f"Enhanced HDF5 reader failed, falling back to standard method: {enhanced_error}"
                        )
                        # Fallback to standard optimized batch reading
                        ret = batch_read_fields(
                            self.fname,
                            ["saxs_2d"],
                            "alias",
                            ftype="nexus",
                            use_pool=True,
                        )
                        self.saxs_2d_data = ret["saxs_2d"]

                # Canonical NeXus layout is (1, rows, cols); drop that singleton
                # axis for every loading strategy (the standard reader does the
                # same). Genuine multi-frame data is left untouched.
                saxs_loaded = self.saxs_2d_data
                if getattr(saxs_loaded, "ndim", 0) == 3 and saxs_loaded.shape[0] == 1:
                    self.saxs_2d_data = saxs_loaded[0]

                # Cache the loaded data using unified memory manager
                if (
                    estimated_saxs_size_mb < MAX_PARALLEL_WORKERS
                ):  # Only cache if less than 800MB
                    self._memory_manager.cache_put(
                        cache_key, self.saxs_2d_data, CacheType.ARRAY_DATA
                    )

            except Exception as e:
                logger.error(f"Error loading SAXS 2D data: {e}")
                # Fallback to original method
                ret = get(
                    self.fname, ["saxs_2d"], "alias", ftype="nexus", use_pool=True
                )
                self.saxs_2d_data = ret["saxs_2d"]

        # Compute log version with optional chunked processing
        log_cache_key = f"{self._cache_prefix}_saxs_2d_log"
        cached_log_data = self._memory_manager.cache_get(
            log_cache_key, CacheType.ARRAY_DATA
        )

        if cached_log_data is not None:
            logger.debug("Using cached SAXS 2D log data from unified memory manager")
            self.saxs_2d_log_data = cached_log_data
        else:
            if self.saxs_2d_data is None:
                logger.error("SAXS 2D data is None; cannot compute log version")
                return
            if chunk_processing and self.saxs_2d_data.size > 10**7:  # >10M elements
                logger.info("Computing SAXS 2D log data using optimized streaming")
                self.saxs_2d_log_data = self._compute_saxs_log_streaming(
                    self.saxs_2d_data
                )
            else:
                # JAX GPU log computation (falls back to NumPy on any failure)
                self.saxs_2d_log_data = self._compute_saxs_log_jax(self.saxs_2d_data)

            # Cache log data using unified memory manager if reasonable size
            if (
                estimated_saxs_size_mb < MEMORY_WARNING_THRESHOLD_MB
            ):  # Log data is typically smaller
                self._memory_manager.cache_put(
                    log_cache_key, self.saxs_2d_log_data, CacheType.ARRAY_DATA
                )

        # Log memory usage after loading
        memory_after_mb, _ = MemoryMonitor.get_memory_usage()
        memory_used = memory_after_mb - memory_before_mb

        if memory_used > MIN_DOWNSAMPLE_POINTS:  # Log significant memory usage
            logger.info(
                f"SAXS 2D data loaded: {memory_used:.1f}MB memory used (strategy: {'memory_mapped' if use_memory_mapping else 'chunked' if chunk_processing else 'standard'})"
            )

        # Check for memory pressure after loading
        final_memory_pressure = MemoryMonitor.get_memory_pressure()
        if final_memory_pressure > MEMORY_EFFICIENCY_HIGH:
            logger.warning(
                f"High memory pressure after SAXS data loading: {final_memory_pressure * 100:.1f}%"
            )

        # Record operation for memory predictor learning
        record_operation_memory(
            operation_type="load_saxs_2d",
            input_size_mb=estimated_saxs_size_mb,
            memory_before_mb=memory_before_mb,
            memory_after_mb=memory_after_mb,
            duration_seconds=2.0,  # Approximate duration - could be improved with actual timing
        )

        self._saxs_data_loaded = True

    def _compute_saxs_log_standard(self, saxs_data: np.ndarray) -> np.ndarray:
        """Standard log computation for SAXS data."""
        if not np.any(saxs_data > 0):
            return np.zeros_like(saxs_data, dtype=np.uint8)
        min_val = np.min(saxs_data[saxs_data > 0])
        return np.log10(np.maximum(saxs_data, min_val)).astype(np.float32)

    def _compute_saxs_log_jax(self, saxs_data: np.ndarray) -> np.ndarray:
        """JAX GPU-accelerated log computation for SAXS data (OPT-004).

        Uses a module-level JIT-compiled kernel for 2.9× speedup vs NumPy
        on GPU (full host↔device round-trip, 2048×2048 float32).
        Falls back to _compute_saxs_log_standard on any JAX failure.
        """
        try:
            import jax.numpy as jnp

            from xpcsviewer.backends._conversions import ensure_numpy

            kernel = _get_jax_saxs_log()
            data_f32 = (
                saxs_data.astype(np.float32)
                if saxs_data.dtype != np.float32
                else saxs_data
            )
            result = ensure_numpy(kernel(jnp.asarray(data_f32)))
            logger.debug("SAXS log: JAX GPU path used")
            return result
        except Exception:
            logger.debug("SAXS log: JAX unavailable, falling back to NumPy")
            return self._compute_saxs_log_standard(saxs_data)

    def _compute_saxs_log_streaming(self, saxs_data: np.ndarray) -> np.ndarray:
        """
        Memory-efficient streaming log computation for large SAXS data.

        Uses adaptive chunk sizing and streaming processing to minimize
        memory footprint while computing logarithmic transformation.
        """
        logger.info(
            f"Starting streaming SAXS log computation for data shape {saxs_data.shape}"
        )

        # Record operation start for memory tracking
        memory_before_mb, _ = MemoryMonitor.get_memory_usage()

        # Use adaptive chunk sizing based on current memory pressure
        sizer = AdaptiveChunkSizer()
        optimal_chunk_size = sizer.get_optimal_chunk_size(
            saxs_data.shape, saxs_data.dtype
        )

        logger.debug(
            f"Using adaptive chunk size: {optimal_chunk_size:.1f}MB for streaming log computation"
        )

        # Process using streaming processor
        try:
            result = process_saxs_log_streaming(
                data=saxs_data, chunk_size_mb=optimal_chunk_size, epsilon=1e-10
            )

            # Record operation completion
            memory_after_mb, _ = MemoryMonitor.get_memory_usage()
            memory_used = memory_after_mb - memory_before_mb

            logger.info(
                f"Streaming SAXS log computation completed, "
                f"memory used: {memory_used:+.1f}MB"
            )

            # Record operation for memory predictor learning
            record_operation_memory(
                operation_type="compute_saxs_log",
                input_size_mb=saxs_data.nbytes / (1024 * 1024),
                memory_before_mb=memory_before_mb,
                memory_after_mb=memory_after_mb,
                duration_seconds=2.0,  # Approximate duration
            )

            return result

        except Exception as e:
            logger.error(f"Streaming SAXS log computation failed: {e}")
            logger.warning("Falling back to standard log computation")
            return self._compute_saxs_log_standard(saxs_data)

    def _compute_saxs_log_chunked(
        self, saxs_data: np.ndarray, chunk_size: int = 1024
    ) -> np.ndarray:
        """
        Chunked log computation for large SAXS data to manage memory usage.

        Parameters
        ----------
        saxs_data : np.ndarray
            Input SAXS data
        chunk_size : int
            Size of chunks for processing (along first dimension)

        Returns
        -------
        np.ndarray
            Log-transformed SAXS data
        """
        logger.debug(f"Computing log transform in chunks of size {chunk_size}")

        # Find global minimum value for zero pixels
        # Process in chunks to find minimum
        global_min = np.inf
        num_chunks = (saxs_data.shape[0] + chunk_size - 1) // chunk_size

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, saxs_data.shape[0])
            chunk = saxs_data[start_idx:end_idx]

            positive_values = chunk[chunk > 0]
            if len(positive_values) > 0:
                chunk_min = np.min(positive_values)
                global_min = min(global_min, chunk_min)

        if global_min == np.inf:
            # No positive values found
            return np.zeros_like(saxs_data, dtype=np.float32)

        # Process log transformation in chunks
        result = np.empty_like(saxs_data, dtype=np.float32)

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, saxs_data.shape[0])

            chunk = saxs_data[start_idx:end_idx].astype(np.float32)

            # Replace non-positive values with global minimum
            chunk[chunk <= 0] = global_min

            # Compute log transform
            result[start_idx:end_idx] = np.log10(chunk)

            # Periodic memory check during processing
            if i % 10 == 0 and MemoryMonitor.is_memory_pressure_high(threshold=0.90):
                logger.warning(
                    f"High memory pressure during chunked log computation at chunk {i}/{num_chunks}"
                )
                # Schedule smart garbage collection
                try:
                    from .threading.cleanup_optimized import smart_gc_collect

                    smart_gc_collect("memory_pressure_chunked_log")
                except ImportError:
                    # Fallback to manual GC if optimized system not available
                    import gc

                    gc.collect()

        return result

    def __getattr__(self, key):
        # keys from qmap — O(1) frozenset lookup
        if key in self._QMAP_KEYS:
            return getattr(self.qmap, key)
        # delayed loading of saxs_2d due to its large size - now using connection pool and batch loading
        if key == "saxs_2d":
            if not self._saxs_data_loaded:
                self._load_saxs_data_batch()
            return self.saxs_2d_data
        if key == "saxs_2d_log":
            if not self._saxs_data_loaded:
                self._load_saxs_data_batch()
            return self.saxs_2d_log_data
        if key == "Int_t_fft":
            return self._compute_int_t_fft_cached()
        # G2 partial data for stability analysis (lazy loading)
        if key in self._G2_PARTIAL_KEYS:
            if key not in self.__dict__:
                try:
                    ret = get(self.fname, [key], "alias", ftype="nexus", use_pool=True)
                    self.__dict__[key] = ret[key]
                except Exception as e:
                    self.__dict__[key] = None
                    logger.warning("Cannot load %s due to %s", key, str(e))
            return self.__dict__[key]
        if key in self.__dict__:
            return self.__dict__[key]
        # Raise AttributeError so hasattr() works correctly
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{key}'"
        )

    def get_info_at_position(self, x, y):
        x, y = int(x), int(y)
        shape = self.saxs_2d.shape
        if x < 0 or x >= shape[1] or y < 0 or y >= shape[0]:
            return None
        scat_intensity = self.saxs_2d[y, x]
        qmap_info = self.qmap.get_qmap_at_pos(x, y)
        return f"I={scat_intensity:.4e} {qmap_info}"

    def get_detector_extent(self):
        return self.qmap.extent

    def get_qbin_label(self, qbin: int, append_qbin: bool = False):
        return self.qmap.get_qbin_label(qbin, append_qbin=append_qbin)

    def get_qbinlist_at_qindex(self, qindex, zero_based=True):
        return self.qmap.get_qbinlist_at_qindex(qindex, zero_based=zero_based)

    @log_timing(threshold_ms=200)
    def get_g2_data(self, qrange=None, trange=None):
        # Support both Multitau and Twotime analysis types
        supported_types = ["Multitau", "Twotime"]
        if not any(atype in self.atype for atype in supported_types):
            raise ValueError(
                f"Analysis type {self.atype} not supported for G2 plotting. Supported types: {supported_types}"
            )

        # Handle different analysis types - Multitau vs Twotime
        has_multitau = "Multitau" in self.atype
        has_twotime = "Twotime" in self.atype

        # For Multitau analysis, we need g2, g2_err, t_el
        # For Twotime analysis, we can use c2_g2 and compute from c2_delay
        missing_attrs = []

        if has_multitau:
            # Check Multitau-specific attributes
            multitau_attrs = ["g2", "g2_err"]
            missing_multitau = [
                attr for attr in multitau_attrs if not hasattr(self, attr)
            ]

            # Special handling for t_el - compute it if missing but tau and t0/g2_t0 are available
            if not hasattr(self, "t_el"):
                if hasattr(self, "tau"):
                    # Try g2_t0 first (preferred), then fall back to t0
                    if hasattr(self, "g2_t0"):
                        logger.info("Computing t_el from tau and g2_t0 for G2 plotting")
                        self.t_el = self.tau * self.g2_t0
                    elif hasattr(self, "t0"):
                        logger.info("Computing t_el from tau and t0 for G2 plotting")
                        self.t_el = self.tau * self.t0
                    else:
                        missing_multitau.append("t_el")
                        logger.warning(
                            "Cannot compute t_el: tau available, but neither g2_t0 nor t0 available"
                        )
                else:
                    missing_multitau.append("t_el")
                    logger.warning("Cannot compute t_el: tau not available")

            if missing_multitau and not has_twotime:
                missing_attrs.extend(missing_multitau)

        if has_twotime and not (has_multitau and not missing_attrs):
            # Check Twotime-specific attributes as fallback
            twotime_attrs = ["c2_g2"]
            missing_twotime = [
                attr for attr in twotime_attrs if not hasattr(self, attr)
            ]

            if missing_twotime:
                missing_attrs.extend(missing_twotime)
                logger.error(
                    "Neither Multitau nor Twotime data available for G2 plotting"
                )

        # Check common required attributes
        if not hasattr(self, "qmap"):
            missing_attrs.append("qmap")

        if missing_attrs:
            # Provide detailed diagnostic information
            available_attrs = [attr for attr in dir(self) if not attr.startswith("_")]
            data_attrs = [
                attr for attr in available_attrs if not callable(getattr(self, attr))
            ]
            logger.error(f"Available data attributes: {sorted(data_attrs)}")
            raise AttributeError(
                f"Missing required attributes for G2 plotting: {missing_attrs}"
            )

        # qrange can be None
        qindex_selected, qvalues = self.qmap.get_qbin_in_qrange(qrange, zero_based=True)
        labels = [self.qmap.get_qbin_label(qbin + 1) for qbin in qindex_selected]

        # Extract data based on analysis type
        if has_multitau and hasattr(self, "g2") and hasattr(self, "g2_err"):
            # Use Multitau data
            # Validate qindex_selected bounds before array access
            max_q_index = min(self.g2.shape[1], self.g2_err.shape[1]) - 1
            valid_qindex = [idx for idx in qindex_selected if 0 <= idx <= max_q_index]
            invalid_qindex = [idx for idx in qindex_selected if idx not in valid_qindex]

            if invalid_qindex:
                logger.warning(
                    f"Filtering out invalid Q indices {invalid_qindex} (max valid: {max_q_index}) for file {self.label}"
                )

            # An empty selection (Q range matching no bin) is a valid empty result.
            if not valid_qindex and len(qindex_selected) > 0:
                raise ValueError(
                    f"No valid Q indices found in qindex_selected {qindex_selected} for file {self.label}"
                )

            # Use validated indices for array access
            g2 = self.g2[:, valid_qindex]
            g2_err = self.g2_err[:, valid_qindex]
            t_el = self.t_el

            # Update corresponding qvalues and labels to match filtered indices
            qvalues = [
                qvalues[i]
                for i, idx in enumerate(qindex_selected)
                if idx in valid_qindex
            ]
            labels = [
                labels[i]
                for i, idx in enumerate(qindex_selected)
                if idx in valid_qindex
            ]

            logger.info(
                f"Using Multitau G2 data for plotting ({len(valid_qindex)} valid Q indices)"
            )
        elif has_twotime and hasattr(self, "c2_g2"):
            # Use Twotime data as fallback
            logger.info("Using Twotime G2 data for plotting")

            # Packed c2_g2 columns follow c2_processed_bins (1-based detector bin
            # IDs), not the dynamic Q-bin index, so map index -> column explicitly.
            n_cols = self.c2_g2.shape[1]
            processed = getattr(self, "c2_processed_bins", None)
            if processed is None:  # legacy files: every bin packed in order
                col_of = {i: i for i in range(n_cols)}
            else:
                if np.size(processed) != n_cols:
                    logger.warning(
                        f"c2_processed_bins ({np.size(processed)}) and c2_g2 "
                        f"columns ({n_cols}) differ for {self.label}"
                    )
                col_of = {
                    int(b) - 1: c
                    for c, b in enumerate(np.ravel(processed))
                    if c < n_cols
                }
            valid_qindex = [idx for idx in qindex_selected if int(idx) in col_of]
            invalid_qindex = [idx for idx in qindex_selected if idx not in valid_qindex]

            if invalid_qindex:
                logger.warning(
                    f"Filtering out unprocessed Q indices {invalid_qindex} for twotime file {self.label}"
                )

            if not valid_qindex and len(qindex_selected) > 0:
                raise ValueError(
                    f"No valid Q indices found in qindex_selected {qindex_selected} for twotime file {self.label}"
                )

            cols = [col_of[int(idx)] for idx in valid_qindex]
            g2 = self.c2_g2[:, cols]

            # For twotime, we might not have error data - create placeholder
            if hasattr(self, "c2_g2_err"):
                # Validate error array bounds too
                max_q_index_err = self.c2_g2_err.shape[1] - 1
                keep = [i for i, c in enumerate(cols) if c <= max_q_index_err]
                if len(keep) != len(cols):
                    logger.warning(
                        f"Error array has different size, using available indices for {self.label}"
                    )
                    valid_qindex = [valid_qindex[i] for i in keep]
                    cols = [cols[i] for i in keep]
                    g2 = self.c2_g2[:, cols]

                g2_err = self.c2_g2_err[:, cols]
            else:
                logger.warning("No Twotime G2 error data available - using zeros")
                g2_err = np.zeros_like(g2)

            # Update corresponding qvalues and labels to match filtered indices
            qvalues = [
                qvalues[i]
                for i, idx in enumerate(qindex_selected)
                if idx in valid_qindex
            ]
            labels = [
                labels[i]
                for i, idx in enumerate(qindex_selected)
                if idx in valid_qindex
            ]

            # For twotime delay, we need to compute t_el from available delay data
            if hasattr(self, "c2_delay"):
                t_el = self.c2_delay
            elif hasattr(self, "c2_t0"):
                # Fallback - create delay array assuming regular spacing
                logger.warning(
                    "No c2_delay available - creating delay array from c2_t0"
                )
                t_el = np.arange(g2.shape[0]) * self.c2_t0
            else:
                logger.error("Cannot determine time delays for Twotime data")
                raise AttributeError(
                    "Missing delay information for Twotime G2 plotting"
                )
        else:
            raise AttributeError("No valid G2 data found for plotting")

        # Apply time range filtering if specified
        if trange is not None and len(t_el) > 0:
            t_roi = (t_el >= trange[0]) * (t_el <= trange[1])
            g2 = g2[t_roi]
            g2_err = g2_err[t_roi]
            t_el = t_el[t_roi]

        return qvalues, t_el, g2, g2_err, labels

    def get_g2_stability_data(self, qrange=None, trange=None):
        """
        Get G2 stability data from g2_partial for frame-by-frame analysis.

        Parameters
        ----------
        qrange : tuple or None
            Q-range filter (qmin, qmax). If None, use all Q-bins.
        trange : tuple or None
            Time range filter (tmin, tmax). If None, use all delay times.

        Returns
        -------
        tuple
            (qvalues, t_el, g2, g2_err, qbin_labels, labels) where:
            - qvalues: Q-values for selected bins
            - t_el: delay times
            - g2: G2 partial data [frames, delays, q_bins]
            - g2_err: G2 partial errors
            - qbin_labels: labels for Q-bins
            - labels: frame labels
        """
        if "Multitau" not in self.atype:
            raise ValueError("G2 stability data only available for Multitau analysis")

        # Check if g2_partial data is available
        if self.g2_partial is None:
            raise AttributeError(
                "g2_partial data not available in this file. "
                "G2 stability analysis requires partial G2 data."
            )

        # qrange can be None
        qindex_selected, qvalues = self.qmap.get_qbin_in_qrange(qrange, zero_based=True)
        g2 = self.g2_partial[:, :, qindex_selected]
        g2_err = (
            self.g2_partial_err[:, :, qindex_selected]
            if self.g2_partial_err is not None
            else np.zeros_like(g2)
        )

        qbin_labels = [
            f"qbin={qbin + 1}, {self.qmap.get_qbin_label(qbin + 1)}"
            for qbin in qindex_selected
        ]
        labels = (
            self.g2_partial_labels
            if self.g2_partial_labels is not None
            else np.arange(g2.shape[0])
        )

        if trange is not None:
            t_roi = (self.t_el >= trange[0]) * (self.t_el <= trange[1])
            g2 = g2[:, t_roi]
            g2_err = g2_err[:, t_roi]
            t_el = self.t_el[t_roi]
        else:
            t_el = self.t_el

        return qvalues, t_el, g2, g2_err, qbin_labels, labels

    def get_cropped_qmap(self, target="dqmap", enabled=True):
        """
        Get cropped Q-map array, trimmed to non-zero region.

        Parameters
        ----------
        target : str
            Either "dqmap" (dynamic) or "sqmap" (static)
        enabled : bool
            If True, crop to non-zero region. If False, return full array.

        Returns
        -------
        np.ndarray
            Cropped or full Q-map array
        """
        if target not in ["dqmap", "sqmap"]:
            raise ValueError(f"target must be 'dqmap' or 'sqmap', got '{target}'")

        obj = getattr(self, target).copy()
        if enabled:
            idx = np.nonzero(obj >= 1)
            if len(idx[0]) > 0 and len(idx[1]) > 0:
                sl_v = slice(np.min(idx[0]), np.max(idx[0]) + 1)
                sl_h = slice(np.min(idx[1]), np.max(idx[1]) + 1)
                obj = obj[sl_v, sl_h]
        return obj

    def get_offseted_g2(self, normalization=False):
        """
        Get G2 data with optional baseline normalization for G2 map visualization.

        Parameters
        ----------
        normalization : bool
            If True, normalize G2 by subtracting baseline and shifting to mean.

        Returns
        -------
        np.ndarray
            G2 data array, optionally normalized
        """
        g2 = self.g2.copy()
        if normalization:
            g2_baseline = g2[0]
            g2 = g2 - g2_baseline + np.mean(g2_baseline)
        return g2

    def _saxs_axes_mismatch(self, bkg_xf, bkg_rows):
        """Why ``bkg_xf``'s SAXS curves are not on this file's axes, or None.

        Equal Q values and row counts do not make the rows comparable: the
        static map's axis names/units say what the numbers measure, and when the
        background has sector curves (rows 1..n of ``saxs_1d``, one per
        ``splist`` entry) each must be measured at the sample's azimuth. A
        single-row background is one curve broadcast over every sample row.
        """
        mine, theirs = self.qmap, bkg_xf.qmap
        for attr, what in (("map_names", "names"), ("map_units", "units")):
            a, b = getattr(mine, attr, None), getattr(theirs, attr, None)
            if a is not None and b is not None and list(a) != list(b):
                return (
                    f"its static-map axis {what} {list(b)} differ from the "
                    f"sample's {list(a)}"
                )
        if bkg_rows > 1:
            sample_phi = np.atleast_1d(np.asarray(self.splist, dtype=float))
            bkg_phi = np.atleast_1d(np.asarray(bkg_xf.splist, dtype=float))
            if sample_phi.shape != bkg_phi.shape or not np.allclose(
                sample_phi, bkg_phi, rtol=0.0, atol=1e-6
            ):
                return (
                    f"its azimuth sectors {bkg_phi.tolist()} differ from the "
                    f"sample's {sample_phi.tolist()}"
                )
        return None

    @log_timing(threshold_ms=200)
    def get_saxs1d_data(
        self,
        bkg_xf=None,
        bkg_weight=1.0,
        qrange=None,
        sampling=1,
        use_absolute_crosssection=False,
        norm_method=None,
        target="saxs1d",
    ):
        assert target in ["saxs1d", "saxs1d_partial"]
        if target == "saxs1d":
            q, Iq = self.saxs_1d["q"], self.saxs_1d["Iq"]
        else:
            q, Iq = self.saxs_1d["q"], self.Iqp
        if bkg_xf is not None:
            bkg_q, bkg_iq = bkg_xf.saxs_1d["q"], bkg_xf.saxs_1d["Iq"]
            # Compare shapes before values: allclose raises on different Q counts.
            if not (
                np.shape(q) == np.shape(bkg_q)
                and np.shape(bkg_iq)[-1] == np.shape(Iq)[-1]
                and np.shape(bkg_iq)[0] in (1, np.shape(Iq)[0])
                and np.allclose(q, bkg_q)
            ):
                problem = (
                    "its q axis does not match the sample's "
                    f"({np.shape(bkg_q)[-1:] or 'scalar'} vs "
                    f"{np.shape(q)[-1:] or 'scalar'} Q bins)"
                )
            else:
                problem = self._saxs_axes_mismatch(bkg_xf, np.shape(bkg_iq)[0])
            if problem is None:
                Iq = Iq - bkg_weight * bkg_iq
                Iq[Iq < 0] = np.nan
            else:
                logger.warning(
                    "background subtraction is not applied for %s because %s; "
                    "the sample trace is kept",
                    getattr(bkg_xf, "label", "background"),
                    problem,
                )
        if qrange is not None:
            q_roi = (q >= qrange[0]) * (q <= qrange[1])
            if q_roi.sum() > 0:
                q = q[q_roi]
                Iq = Iq[:, q_roi]
            else:
                logger.warning("qrange is not applied because it is out of range")
        if use_absolute_crosssection and self.abs_cross_section_scale is not None:
            Iq = (
                Iq * self.abs_cross_section_scale
            )  # not in place: Iq may be stored data

        # apply sampling (Iq is 2D (n_curves, n_q); sample the q-axis to stay
        # aligned with q)
        if sampling > 1:
            q, Iq = q[::sampling], Iq[:, ::sampling]
        # apply normalization
        q, Iq, xlabel, ylabel = self.norm_saxs_data(q, Iq, norm_method=norm_method)
        return q, Iq, xlabel, ylabel

    def _static_axis(self):
        """``(name, unit)`` of the file's static-map axis the SAXS q values are on.

        The numbers in ``saxs_1d["q"]``, ``sqlist`` and the Pie profile are
        stored in this unit (``1/A``, ``1/nm``, ...); they are labelled with it
        and never converted. Falls back to ``q`` / inverse angstrom only when
        the file's map carries no axis names or units.
        """
        qmap = getattr(self, "qmap", None)  # a bare object may carry no map
        names = getattr(qmap, "map_names", None)
        units = getattr(qmap, "map_units", None)
        name = str(names[0]) if names is not None and len(names) else "q"
        unit = str(units[0]) if units is not None and len(units) else Q_UNIT_DISPLAY
        return name, unit

    def norm_saxs_data(self, q, Iq, norm_method=None):
        assert norm_method in (None, "q2", "q4", "I0")
        name, unit = self._static_axis()
        if norm_method is None:
            return q, Iq, f"{name} ({unit})", "Intensity"
        ylabel = "Intensity"
        if norm_method == "q2":
            Iq = Iq * np.square(q)
            ylabel = ylabel + " * q^2"
        elif norm_method == "q4":
            Iq = Iq * np.square(np.square(q))
            ylabel = ylabel + " * q^4"
        elif norm_method == "I0":
            # each curve (row) is divided by its own first valid non-zero point
            # along Q; a curve without one cannot be normalised and becomes NaN
            Iq = np.asarray(Iq, dtype=float)
            usable = np.isfinite(Iq) & (Iq != 0)
            first = np.argmax(usable, axis=-1)
            ref = np.take_along_axis(Iq, first[..., None], axis=-1)
            ref = np.where(usable.any(axis=-1, keepdims=True), ref, np.nan)
            with np.errstate(invalid="ignore"):
                Iq = Iq / ref
            ylabel = ylabel + " / I_0"
        xlabel = f"{name} ({unit})"
        return q, Iq, xlabel, ylabel

    def get_twotime_qbin_labels(self):
        qbin_labels = []
        for qbin in np.ravel(self.c2_processed_bins).tolist():
            qbin_labels.append(self.get_qbin_label(qbin, append_qbin=True))
        return qbin_labels

    @log_timing(threshold_ms=200)
    def get_twotime_maps(
        self, scale="log", auto_crop=True, highlight_xy=None, selection=None
    ):
        # Two-time bins are bound to the file's own dynamic map (xf.dqmap is
        # always canonical; an imported partition lives in xf.display_dqmap).
        # Work on a view until we need to mutate (astype creates a copy)
        dqmap = self.dqmap
        saxs = self.saxs_2d_log if scale == "log" else self.saxs_2d

        # Handle 3D SAXS data by taking the first frame if needed
        if saxs.ndim == NDIM_3D and saxs.shape[0] == 1:
            saxs = saxs[0]
            logger.debug(f"Converted 3D SAXS data to 2D: new shape={saxs.shape}")

        if auto_crop:
            idx = np.nonzero(dqmap >= 1)
            # Check if there are any valid indices before cropping
            if len(idx[0]) > 0 and len(idx[1]) > 0:
                sl_v = slice(np.min(idx[0]), np.max(idx[0]) + 1)
                sl_h = slice(np.min(idx[1]), np.max(idx[1]) + 1)
                dqmap_cropped = dqmap[sl_v, sl_h]
                saxs_cropped = saxs[sl_v, sl_h]

                # Validate that cropping didn't result in empty arrays
                if dqmap_cropped.size > 0 and saxs_cropped.size > 0:
                    dqmap = dqmap_cropped
                    saxs = saxs_cropped
                else:
                    logger.warning(
                        f"Auto-crop resulted in empty arrays for {self.label}, using original data"
                    )
            else:
                logger.warning(
                    f"No valid qmap data found for auto_crop in {self.label}"
                )
                # Use original data without cropping

        if saxs.size == 0:
            logger.warning(f"SAXS data is empty for {self.label} after processing")
        if dqmap.size == 0:
            logger.warning(f"DQMAP data is empty for {self.label} after processing")

        # Check for valid dqmap before computing max/unique operations
        if dqmap.size == 0:
            logger.warning(f"Cannot process empty dqmap for {self.label}")
            return dqmap, saxs, None

        qindex_max = np.max(dqmap)
        dqlist = np.unique(dqmap)
        dqlist = dqlist[dqlist > 0]

        # Validate dqlist is not empty
        if len(dqlist) == 0:
            logger.warning(f"No valid Q-bins found for {self.label}")
            dqmap = dqmap.astype(np.float32)
            dqmap[dqmap == 0] = np.nan
            dqmap_disp = np.flipud(dqmap).copy()
            return dqmap_disp, saxs, None

        # astype returns a new array, so dqmap is already a copy here
        dqmap = dqmap.astype(np.float32)
        dqmap[dqmap == 0] = np.nan

        # flipud returns a view; copy so downstream mutations are safe
        dqmap_disp = np.flipud(dqmap).copy()

        dq_bin = None
        if highlight_xy is not None:
            x, y = highlight_xy
            if x >= 0 and y >= 0 and x < dqmap.shape[1] and y < dqmap.shape[0]:
                dq_bin = dqmap_disp[y, x]
        # The selection is a position in c2_processed_bins (the C2 dataset order),
        # so resolve detector bin IDs against it, not against the pixels' IDs.
        processed = getattr(self, "c2_processed_bins", None)
        bins = dqlist if processed is None else np.ravel(processed)
        if highlight_xy is None and selection is not None and selection < len(bins):
            dq_bin = bins[selection]

        matching_indices = (
            np.where(bins == dq_bin)[0]
            if dq_bin is not None and not np.isnan(dq_bin) and dq_bin > 0
            else []
        )
        if len(matching_indices) > 0:
            # highlight the selected qbin only if it was processed
            dqmap_disp[dqmap_disp == dq_bin] = qindex_max + 1
            selection = matching_indices[0]
        else:
            selection = None
        return dqmap_disp, saxs, selection

    def get_twotime_c2(self, selection=0, correct_diag=True, max_size=32678):
        dq_processed = tuple(np.ravel(self.c2_processed_bins).tolist())
        assert selection >= 0 and selection < len(dq_processed), (
            f"selection {selection} out of range {dq_processed}"
        )
        # the file revision is part of the key: a replaced file must not reuse C2
        config = (selection, correct_diag, max_size, _file_identity(self.fname))
        if self.c2_kwargs == config:
            return self.c2_all_data
        # Monitor memory for large two-time correlation data
        memory_before_mb, _ = MemoryMonitor.get_memory_usage()

        # Estimate memory needed for C2 data
        estimated_mb = MemoryMonitor.estimate_array_memory(
            (max_size, max_size), np.float64
        )

        if estimated_mb > 500:  # Large C2 matrix
            logger.info(f"Loading large C2 matrix: estimated {estimated_mb:.1f}MB")

        if MemoryMonitor.is_memory_pressure_high(threshold=0.75):
            logger.warning("High memory pressure before C2 loading, clearing caches")
            self.clear_cache("saxs")
            # Also trigger cleanup in unified memory manager
            if self._memory_manager.get_memory_pressure() in [
                MemoryPressure.HIGH,
                MemoryPressure.CRITICAL,
            ]:
                self._memory_manager._aggressive_cleanup()
        c2_result = get_single_c2_from_hdf(
            self.fname,
            selection=selection,
            max_size=max_size,
            t0=self.c2_t0,  # effective period: t0 * c2_stride_frame * c2_avg_frame
            correct_diag=correct_diag,
        )
        self.c2_all_data = c2_result
        self.c2_kwargs = config

        # Log memory usage after C2 data loading
        memory_after_mb, _ = MemoryMonitor.get_memory_usage()
        memory_used = memory_after_mb - memory_before_mb

        if memory_used > 50:  # Log significant memory usage
            logger.info(f"C2 data loaded: {memory_used:.1f}MB memory used")

        return c2_result

    def get_twotime_stream(self, **kwargs):
        return get_c2_stream(self.fname, **kwargs)

    # def get_g2_fitting_line(self, q, tor=1e-6):
    #     """
    #     get the fitting line for q, within tor
    #     """
    #     if self.fit_summary is None:
    #         return None, None
    #     idx = np.argmin(np.abs(self.fit_summary["q_val"] - q))
    #     if abs(self.fit_summary["q_val"][idx] - q) > tor:
    #         return None, None

    #     fit_x = self.fit_summary["fit_line"][idx]["fit_x"]
    #     fit_y = self.fit_summary["fit_line"][idx]["fit_y"]
    #     return fit_x, fit_y

    def get_fitting_info(self, mode="g2_fitting"):
        if self.fit_summary is None:
            return f"fitting is not ready for {self.label}"

        if mode == "g2_fitting":
            result = self.fit_summary.copy()
            # fit_line is not useful to display
            result.pop("fit_line", None)
            val = result.pop("fit_val", None)
            if result["fit_func"] == "single":
                prefix = ["a", "b", "c", "d"]
            else:
                prefix = ["a", "b", "c", "d", "b2", "c2", "f"]

            from xpcsviewer.fitting.legacy import fit_success_mask

            fit_ok = fit_success_mask(self.fit_summary)
            msg = []
            for n in range(val.shape[0]):
                if not fit_ok[n]:
                    msg.append("fit failed: no valid parameters")
                    continue
                temp = []
                for m in range(len(prefix)):
                    value = val[n, 0, m]
                    error = val[n, 1, m]

                    # Handle infinite or NaN error values gracefully
                    if np.isfinite(error) and error > 0:
                        temp.append(f"{prefix[m]} = {value:f} ± {error:f}")
                    elif np.isnan(error):
                        temp.append(f"{prefix[m]} = {value:f} ± NaN")
                    else:  # infinite, negative, or zero error
                        temp.append(f"{prefix[m]} = {value:f} ± --")
                msg.append(", ".join(temp))
            result["fit_val"] = np.array(msg)

        elif mode == "tauq_fitting":
            if "tauq_fit_val" not in self.fit_summary:
                result = "tauq fitting is not available"
            elif not self.fit_summary.get("tauq_success", True):
                result = "tauq fitting failed; no fit parameters available"
            else:
                v = self.fit_summary["tauq_fit_val"]

                # Handle infinite or NaN error values gracefully for tauq fitting
                a_val, a_err = v[0, 0], v[1, 0]
                b_val, b_err = v[0, 1], v[1, 1]

                # Format 'a' parameter
                if np.isfinite(a_err) and a_err > 0:
                    a_str = f"a = {a_val:e} ± {a_err:e}"
                elif np.isnan(a_err):
                    a_str = f"a = {a_val:e} ± NaN"
                else:
                    a_str = f"a = {a_val:e} ± --"

                # Format 'b' parameter
                if np.isfinite(b_err) and b_err > 0:
                    b_str = f"b = {b_val:f} ± {b_err:f}"
                elif np.isnan(b_err):
                    b_str = f"b = {b_val:f} ± NaN"
                else:
                    b_str = f"b = {b_val:f} ± --"

                result = f"{a_str}; {b_str}"
        else:
            raise ValueError("mode not supported.")

        return result

    def fit_g2(
        self,
        q_range=None,
        t_range=None,
        bounds=None,
        fit_flag=None,
        fit_func="single",
        robust_fitting=False,
        diagnostic_level="standard",
        bootstrap_samples=None,
        force_refit=False,
        use_parallel=False,
        max_workers=None,
    ):
        """
        Optimized g2 fitting with caching and improved parameter initialization.
        Enhanced with robust fitting capabilities and optional parallel processing.

        :param q_range: a tuple of q lower bound and upper bound
        :param t_range: a tuple of t lower bound and upper bound
        :param bounds: bounds for fitting;
        :param fit_flag: tuple of bools; True to fit and False to float
        :param fit_func: ["single" | "double"]: to fit with single exponential
            or double exponential function
        :param robust_fitting: bool, whether to use robust fitting methods
        :param diagnostic_level: str, level of diagnostics ('basic', 'standard', 'comprehensive')
        :param bootstrap_samples: int, number of bootstrap samples for uncertainty estimation
        :param force_refit: bool, whether to bypass cache and force new fitting
        :param use_parallel: bool, whether to use parallel processing for multiple q-values
        :param max_workers: int, maximum number of parallel workers (None for auto)
        :return: dictionary with the fitting result;
        """
        # Monitor memory usage during fitting
        memory_before_mb, _ = MemoryMonitor.get_memory_usage()

        # Predictive memory management for G2 fitting
        g2_size_mb = (
            (self.g2.nbytes / (1024 * 1024))
            if hasattr(self, "g2") and self.g2 is not None
            else 10.0
        )
        fit_operation = "fit_g2_parallel" if robust_fitting else "fit_g2"
        prediction = predict_operation_memory(fit_operation, g2_size_mb)

        logger.info(
            f"G2 fitting prediction: operation={fit_operation}, "
            f"G2 data={g2_size_mb:.1f}MB, predicted={prediction.predicted_mb:.1f}MB, "
            f"pressure={prediction.pressure_level.value}"
        )

        # Act on memory pressure predictions
        if prediction.pressure_level in [MemoryPressure.HIGH, MemoryPressure.CRITICAL]:
            logger.warning("High memory pressure predicted for G2 fitting")
            for action in prediction.recommended_actions:
                logger.warning(f"Recommendation: {action}")

            # Consider switching to standard fitting if memory pressure is critical
            if prediction.pressure_level == MemoryPressure.CRITICAL and robust_fitting:
                logger.warning(
                    "Switching from robust to standard fitting due to memory pressure"
                )
                robust_fitting = False

        # Check memory pressure and clean cache if needed
        if MemoryMonitor.is_memory_pressure_high(threshold=0.80):
            logger.warning(
                "Memory pressure detected before G2 fitting, clearing caches"
            )
            self.clear_cache("computation")
            # Force cleanup in unified memory manager if under pressure
            if self._memory_manager.get_memory_pressure() in [
                MemoryPressure.HIGH,
                MemoryPressure.CRITICAL,
            ]:
                self._memory_manager._aggressive_cleanup()
        # Generate cache key for fitting parameters
        cache_key = self._generate_cache_key(
            "fit_g2", q_range, t_range, bounds, fit_flag, fit_func
        )

        # Check if fitting result is already cached (unless force_refit is True)
        fit_cache_key = f"{self._cache_prefix}_fit_summary_{cache_key}"
        if not force_refit:
            cached_fit = self._memory_manager.cache_get(
                fit_cache_key, CacheType.COMPUTATION
            )
            if cached_fit is not None:
                self.fit_summary = cached_fit
                return self.fit_summary

        from xpcsviewer.fitting import (
            fit_with_fixed,
            fit_with_fixed_parallel,
            initial_fit_guess,
        )
        from xpcsviewer.xpcs_file.fitting import double_exp_all, single_exp_all

        if bounds is None:
            raise ValueError("bounds must be provided for G2 fitting")
        assert len(bounds) == 2
        if fit_func == "single":
            assert len(bounds[0]) == SINGLE_EXP_PARAMS, (
                "for single exp, the shape of bounds must be (2, 4)"
            )
            if fit_flag is None:
                fit_flag = [True for _ in range(SINGLE_EXP_PARAMS)]
            func = single_exp_all
        else:
            assert len(bounds[0]) == DOUBLE_EXP_PARAMS, (
                "for double exp, the shape of bounds must be (2, 7)"
            )
            if fit_flag is None:
                fit_flag = [True for _ in range(DOUBLE_EXP_PARAMS)]
            func = double_exp_all

        # Use optimized get_g2_data (which now has caching)
        q_val, t_el, g2, sigma, label = self.get_g2_data(qrange=q_range, trange=t_range)
        if np.shape(g2)[-1] == 0:
            raise ValueError(f"Cannot fit G2: no Q bins in range {q_range}")
        t_el, g2, sigma = _positive_delay_fit_data(t_el, g2, sigma)

        # The fit UI has no initial-value entry: the bounds midpoint (geometric
        # mean for tau) is both the starting point and the documented fixed value.
        p0 = initial_fit_guess(bounds, [1, 4] if fit_func == "double" else [1])

        # Optimized fit_x generation - cache if t_el doesn't change
        # (t_el is strictly positive here, so the log grid is finite)
        t_range_key = (
            f"{self._cache_prefix}_fit_x_{np.min(t_el):.6e}_{np.max(t_el):.6e}"
        )
        cached_fit_x = self._memory_manager.cache_get(
            t_range_key, CacheType.COMPUTATION
        )
        if cached_fit_x is not None:
            fit_x = cached_fit_x
        else:
            fit_x = np.logspace(
                np.log10(np.min(t_el)) - 0.5, np.log10(np.max(t_el)) + 0.5, 128
            )
            self._memory_manager.cache_put(t_range_key, fit_x, CacheType.COMPUTATION)

        # Perform the fitting - use robust fitting if requested
        if robust_fitting:
            # Use comprehensive robust fitting for enhanced reliability
            fit_line, fit_val, fit_ok = self._perform_robust_g2_fitting(
                func,
                t_el,
                g2,
                sigma,
                bounds,
                fit_flag,
                fit_x,
                p0,
                diagnostic_level,
                bootstrap_samples,
            )
        else:
            # Check if we should use parallel processing
            num_qvals = g2.shape[1] if g2.ndim > 1 else 1
            use_parallel_actual = (
                use_parallel
                and num_qvals
                > SINGLE_EXP_PARAMS  # Only use parallel for multiple q-values
                and prediction.pressure_level
                != MemoryPressure.CRITICAL  # Avoid parallel if memory critical
            )

            if use_parallel_actual:
                logger.info(
                    f"Using parallel G2 fitting for {num_qvals} q-values with {max_workers or 'auto'} workers"
                )
                fit_line, fit_val, fit_ok = fit_with_fixed_parallel(
                    func,
                    t_el,
                    g2,
                    sigma,
                    bounds,
                    fit_flag,
                    fit_x,
                    p0=p0,
                    max_workers=max_workers,
                    use_threads=True,
                    return_status=True,
                )
            else:
                # Standard sequential fitting
                if use_parallel and num_qvals <= 4:
                    logger.debug(
                        f"Using sequential fitting for small dataset ({num_qvals} q-values)"
                    )
                elif (
                    use_parallel
                    and prediction.pressure_level == MemoryPressure.CRITICAL
                ):
                    logger.warning(
                        "Using sequential fitting due to critical memory pressure"
                    )

                fit_line, fit_val, fit_ok = fit_with_fixed(
                    func,
                    t_el,
                    g2,
                    sigma,
                    bounds,
                    fit_flag,
                    fit_x,
                    p0=p0,
                    return_status=True,
                )

        # Failed Q bins carry NaN, never solver placeholders, and are listed in
        # "fit_success" (one bool per fitted Q, same order as "q_val").
        fit_val, fit_line, fit_ok = _blank_failed_fits(fit_val, fit_line, fit_ok)

        # Create optimized fit summary
        self.fit_summary = {
            "fit_func": fit_func,
            "fit_val": fit_val,
            "fit_success": fit_ok,
            "t_el": t_el,
            "q_val": np.asarray(q_val),  # Ensure q_val is always a numpy array
            "q_range": str(q_range),
            "t_range": str(t_range),
            "bounds": bounds,
            "fit_flag": str(fit_flag),
            "fit_line": fit_line,
            "fit_x": fit_x,  # Include the x-values used for fitting
            "label": label,
        }

        # Cache the fit summary for future calls using unified memory manager
        self._memory_manager.cache_put(
            fit_cache_key, self.fit_summary, CacheType.COMPUTATION
        )

        # Log memory usage after fitting
        memory_after_mb, _ = MemoryMonitor.get_memory_usage()
        memory_used = memory_after_mb - memory_before_mb

        if memory_used > 10:  # Log if fitting used significant memory
            logger.debug(f"G2 fitting completed: {memory_used:.1f}MB memory used")

        # Record operation for memory predictor learning
        record_operation_memory(
            operation_type=fit_operation,
            input_size_mb=g2_size_mb,
            memory_before_mb=memory_before_mb,
            memory_after_mb=memory_after_mb,
            duration_seconds=5.0,  # Approximate fitting duration
        )

        return self.fit_summary

    def _perform_robust_g2_fitting(
        self,
        func,
        t_el,
        g2,
        sigma,
        bounds,
        fit_flag,
        fit_x,
        p0,
        diagnostic_level="standard",
        bootstrap_samples=None,
    ):
        """
        Perform robust G2 fitting using sequential method approach.

        Uses the enhanced fitting sequence: robust -> least squares -> differential evolution
        for improved reliability and convergence.

        Returns ``(fit_line, fit_val, success)``; ``success`` is the per-Q status
        mask under the same rules as every other solver path.
        """
        logger.info(f"Starting sequential G2 fitting with {g2.shape[1]} q-values")

        from xpcsviewer.fitting import fit_with_fixed, fit_with_fixed_sequential

        # Use the new sequential fitting approach
        try:
            fit_line, fit_val, _fit_methods, fit_ok = fit_with_fixed_sequential(
                func,
                t_el,
                g2,
                sigma,
                bounds,
                fit_flag,
                fit_x,
                p0=p0,
                return_status=True,
            )

            # fit_val is already in the correct format (n_q, 2, n_params)
            logger.info("Sequential G2 fitting completed")
            return fit_line, fit_val, fit_ok

        except Exception as e:
            logger.error(f"Sequential fitting failed: {e}")
            # Fallback to original method
            logger.info("Falling back to standard fitting")
            return fit_with_fixed(
                func,
                t_el,
                g2,
                sigma,
                bounds,
                fit_flag,
                fit_x,
                p0=p0,
                return_status=True,
            )

    def fit_g2_robust(
        self,
        q_range=None,
        t_range=None,
        bounds=None,
        fit_flag=None,
        fit_func="single",
        diagnostic_level="standard",
        bootstrap_samples=500,
        enable_caching=True,
        enable_diagnostics=None,
        force_refit=False,
        **kwargs,
    ):
        """
        Dedicated robust G2 fitting method with comprehensive diagnostics.

        This method provides a high-level interface for robust fitting that scientists
        can use when they need enhanced reliability and detailed uncertainty analysis.

        :param q_range: a tuple of q lower bound and upper bound
        :param t_range: a tuple of t lower bound and upper bound
        :param bounds: bounds for fitting
        :param fit_flag: tuple of bools; True to fit and False to float
        :param fit_func: ["single" | "double"]: fitting function type
        :param diagnostic_level: ['basic', 'standard', 'comprehensive'] diagnostics detail
        :param bootstrap_samples: number of bootstrap samples for uncertainty estimation
        :param enable_caching: whether to cache results for performance
        :return: dictionary with enhanced fitting results and diagnostics
        """
        # Map enable_diagnostics to diagnostic_level for compatibility
        if enable_diagnostics is not None:
            diagnostic_level = "comprehensive" if enable_diagnostics else "basic"

        return self.fit_g2(
            q_range=q_range,
            t_range=t_range,
            bounds=bounds,
            fit_flag=fit_flag,
            fit_func=fit_func,
            robust_fitting=True,
            diagnostic_level=diagnostic_level,
            bootstrap_samples=bootstrap_samples,
            force_refit=force_refit,
        )

    def fit_g2_high_performance(
        self,
        q_range=None,
        t_range=None,
        bounds=None,
        fit_flag=None,
        fit_func="single",
        bootstrap_samples=500,
        diagnostic_level="standard",
        max_memory_mb=2048,
        max_workers=None,
        use_parallel=True,
    ):
        """
        High-performance G2 fitting optimized for large XPCS datasets with parallel processing.

        This method uses advanced performance optimizations including adaptive memory
        management, intelligent parallelization across q-values, and chunked processing
        for datasets that exceed memory constraints.

        :param q_range: a tuple of q lower bound and upper bound
        :param t_range: a tuple of t lower bound and upper bound
        :param bounds: bounds for fitting
        :param fit_flag: tuple of bools; True to fit and False to float
        :param fit_func: ["single" | "double"]: fitting function type
        :param bootstrap_samples: number of bootstrap samples for uncertainty estimation
        :param diagnostic_level: ['basic', 'standard', 'comprehensive'] diagnostics detail
        :param max_memory_mb: maximum memory usage threshold in MB
        :param max_workers: maximum number of parallel workers (None for auto)
        :param use_parallel: whether to use parallel processing
        :return: dictionary with comprehensive performance-optimized results
        """
        # Monitor memory usage during fitting
        memory_before_mb, _ = MemoryMonitor.get_memory_usage()

        # Get G2 data
        q_val, t_el, g2, sigma, label = self.get_g2_data(qrange=q_range, trange=t_range)
        if np.shape(g2)[-1] == 0:
            raise ValueError(f"Cannot fit G2: no Q bins in range {q_range}")
        t_el, g2, sigma = _positive_delay_fit_data(t_el, g2, sigma)

        # Check if we should use parallel processing
        num_qvals = g2.shape[1] if g2.ndim > 1 else 1

        # Predictive memory management for high-performance fitting
        g2_size_mb = g2.nbytes / (1024 * 1024) if hasattr(g2, "nbytes") else 10.0
        prediction = predict_operation_memory("fit_g2_parallel", g2_size_mb)

        logger.info(
            f"High-performance G2 fitting: {num_qvals} q-values, "
            f"predicted memory: {prediction.predicted_mb:.1f}MB, "
            f"pressure: {prediction.pressure_level.value}"
        )

        # Decide on parallel processing based on dataset size and memory pressure
        use_parallel_actual = (
            use_parallel
            and num_qvals > 4  # Only use parallel for multiple q-values
            and prediction.pressure_level
            != MemoryPressure.CRITICAL  # Avoid parallel if memory critical
        )

        if not use_parallel_actual:
            logger.info(
                "Using sequential fitting due to small dataset or memory constraints"
            )

        # Same a,b,c,d (single) / 7-parameter (double) models as fit_g2, so the
        # stored summary has the column layout every fit_val reader assumes.
        from xpcsviewer.xpcs_file.fitting import double_exp_all, single_exp_all

        if fit_func == "single":
            func = single_exp_all
        elif fit_func == "double":
            func = double_exp_all
        else:
            raise ValueError(f"Unknown fit function: {fit_func}")

        # Generate fit_x for evaluation (t_el is strictly positive here)
        fit_x = np.logspace(
            np.log10(np.min(t_el)) - 0.5, np.log10(np.max(t_el)) + 0.5, 128
        )

        # Perform fitting using parallel or sequential method
        from xpcsviewer.fitting import (
            fit_with_fixed,
            fit_with_fixed_parallel,
            initial_fit_guess,
        )

        # Same starting point / fixed values as fit_g2.
        if fit_flag is None:
            fit_flag = [True] * len(bounds[0])
        p0 = initial_fit_guess(bounds, [1, 4] if fit_func == "double" else [1])

        if use_parallel_actual:
            logger.info(
                f"Using parallel G2 fitting with {max_workers or 'auto'} workers"
            )
            fit_line, fit_val, fit_ok = fit_with_fixed_parallel(
                func,
                t_el,
                g2,
                sigma,
                bounds,
                fit_flag,
                fit_x,
                p0=p0,
                max_workers=max_workers,
                use_threads=True,
                return_status=True,
            )
        else:
            logger.info("Using sequential G2 fitting")
            fit_line, fit_val, fit_ok = fit_with_fixed(
                func,
                t_el,
                g2,
                sigma,
                bounds,
                fit_flag,
                fit_x,
                p0=p0,
                return_status=True,
            )

        # Failed Q bins carry NaN, never solver placeholders ("fit_success" lists them)
        fit_val, fit_line, fit_ok = _blank_failed_fits(fit_val, fit_line, fit_ok)

        # Create results structure
        results = {
            "fit_line": fit_line,
            "fit_results": fit_val,
            "fit_success": fit_ok,
            "q_values": q_val,
            "tau": t_el,
            "g2_data": g2,
            "g2_errors": sigma,
            "parallel_used": use_parallel_actual,
            "num_qvals": num_qvals,
            "fit_x": fit_x,
        }

        # Convert results to XPCS format for compatibility
        fit_summary = self._convert_optimized_results_to_xpcs_format(
            results, q_val, t_el, fit_func, bounds, fit_flag, label, q_range, t_range
        )

        # Record operation for memory predictor learning
        memory_after_mb, _ = MemoryMonitor.get_memory_usage()
        operation_type = "fit_g2_parallel" if use_parallel_actual else "fit_g2"
        record_operation_memory(
            operation_type=operation_type,
            input_size_mb=g2_size_mb,
            memory_before_mb=memory_before_mb,
            memory_after_mb=memory_after_mb,
            duration_seconds=2.0,  # Approximate - could track actual duration
        )

        # Cache the high-performance results
        cache_key = self._generate_cache_key(
            "fit_g2_hp", q_range, t_range, bounds, fit_flag, fit_func, bootstrap_samples
        )
        fit_cache_key = f"{self._cache_prefix}_hp_fit_summary_{cache_key}"
        self._memory_manager.cache_put(
            fit_cache_key, fit_summary, CacheType.COMPUTATION
        )

        # Store as primary fit summary
        self.fit_summary = fit_summary

        return fit_summary

    def _convert_optimized_results_to_xpcs_format(
        self, results, q_val, t_el, fit_func, bounds, fit_flag, label, q_range, t_range
    ):
        """
        Convert high-performance fitting results to standard XPCS format.

        ``fit_with_fixed`` already returns ``fit_val`` shaped (n_q, 2, n_params)
        and ``fit_line`` shaped (n_q, n_x), the same arrays ``fit_g2`` stores, so
        they are passed through unchanged together with the run metadata.
        """
        return {
            "fit_func": fit_func,
            "fit_val": results["fit_results"],
            "fit_success": results["fit_success"],
            "t_el": t_el,
            "q_val": np.asarray(q_val),  # Ensure q_val is always a numpy array
            "q_range": str(q_range),
            "t_range": str(t_range),
            "bounds": bounds,
            "fit_flag": str(fit_flag),
            "fit_line": results["fit_line"],
            "fit_x": results["fit_x"],
            "label": label,
            "performance_info": results.get("performance_info", {}),
            "timing": results.get("timing", {}),
            "optimization_summary": results.get("optimization_summary", {}),
            "high_performance_fit": True,
        }

    @staticmethod
    def correct_g2_err(g2_err=None, threshold=1e-6):
        # correct the err for some data points with really small error, which
        # may cause the fitting to blowup

        valid_mask = g2_err > threshold

        # Column averages of valid values; fallback to threshold if no valid data
        masked_data = np.ma.masked_where(~valid_mask, g2_err)
        column_averages = np.ma.filled(np.ma.mean(masked_data, axis=0), threshold)

        # np.where avoids an explicit copy — invalid values replaced in-place
        return np.where(valid_mask, g2_err, column_averages)

    def fit_tauq(self, q_range, bounds, fit_flag, force_refit=False):
        """
        Optimized tau-q fitting with improved data filtering and caching.
        """
        if self.fit_summary is None:
            return None

        # Generate cache key for tauq fitting
        # The G2 fit it derives from is part of the key: a re-fit must not reuse it.
        cache_key = self._generate_cache_key(
            "fit_tauq", q_range, bounds, fit_flag, _g2_fit_digest(self.fit_summary)
        )
        tauq_cache_key = f"tauq_fit_{cache_key}"

        # Check if tauq fitting result is already cached (unless force_refit is True)
        tauq_full_cache_key = f"{self._cache_prefix}_{tauq_cache_key}"
        if not force_refit:
            cached_result = self._memory_manager.cache_get(
                tauq_full_cache_key, CacheType.COMPUTATION
            )
            if cached_result is not None:
                self.fit_summary.update(cached_result)
                return self.fit_summary

        try:
            x = self.fit_summary["q_val"]
            fit_val = self.fit_summary["fit_val"]

            # Ensure x is a numpy array (convert from list if necessary)
            if not hasattr(x, "shape"):
                x = np.array(x)

            # Ensure fit_val is a numpy array (convert from list if necessary)
            if not hasattr(fit_val, "shape"):
                fit_val = np.array(fit_val)

            # Validate array dimensions and ensure compatibility
            if len(x.shape) > 1:
                x = x.flatten()

            # Ensure x and fit_val have compatible first dimension
            min_length = min(len(x), fit_val.shape[0])
            x = x[:min_length]
            fit_val = fit_val[:min_length]

            # Vectorized q-range filtering using boolean indexing
            q_slice = (x >= q_range[0]) & (x <= q_range[1])

            # Validate boolean index dimensions
            if len(q_slice) != fit_val.shape[0]:
                logger.warning(
                    f"Boolean index size mismatch: q_slice={len(q_slice)}, fit_val={fit_val.shape[0]}. Adjusting."
                )
                min_slice_len = min(len(q_slice), fit_val.shape[0])
                q_slice = q_slice[:min_slice_len]
                fit_val = fit_val[:min_slice_len]

            # Validate that we have data after filtering
            if not np.any(q_slice):
                logger.warning(f"No data points in q_range {q_range}")
                tauq_result = _tauq_failure()
                self._memory_manager.cache_put(
                    tauq_full_cache_key, tauq_result, CacheType.COMPUTATION
                )
                self.fit_summary.update(tauq_result)
                return self.fit_summary

            # Only Q bins whose G2 fit succeeded are tau(q) observations.
            from xpcsviewer.fitting.legacy import fit_success_mask

            g2_ok = fit_success_mask({**self.fit_summary, "fit_val": fit_val})
            g2_ok = g2_ok[: len(q_slice)]
            n_failed = int(np.count_nonzero(q_slice & ~g2_ok))
            if n_failed:
                logger.warning(
                    "fit_tauq: ignoring %d Q bin(s) whose G2 fit failed", n_failed
                )
            q_slice = q_slice & g2_ok

            x = x[q_slice]
            y = fit_val[q_slice, 0, 1]
            sigma = fit_val[q_slice, 1, 1]

        except (IndexError, KeyError, ValueError) as e:
            logger.error(f"Array indexing error in fit_tauq: {e}")
            tauq_result = _tauq_failure()
            self._memory_manager.cache_put(
                tauq_full_cache_key, tauq_result, CacheType.COMPUTATION
            )
            self.fit_summary.update(tauq_result)
            return self.fit_summary

        # Optimized filtering: combine validity checks
        valid_idx = (sigma > 0) & np.isfinite(y) & np.isfinite(sigma)
        # a*q**b lives on the log-log domain: drop q<=0, tau<=0 and infinite q
        # from the start values, the solver and the plot grid alike.
        domain = np.isfinite(x) & (x > 0) & (y > 0)
        n_dropped = int(np.count_nonzero(valid_idx & ~domain))
        if n_dropped:
            logger.warning(
                "fit_tauq: dropped %d sample(s) with non-positive or non-finite "
                "q or tau",
                n_dropped,
            )
        valid_idx &= domain

        if np.sum(valid_idx) == 0:
            tauq_result = _tauq_failure()
            self._memory_manager.cache_put(
                tauq_full_cache_key, tauq_result, CacheType.COMPUTATION
            )
            self.fit_summary.update(tauq_result)
            return self.fit_summary

        # a*q**b needs two distinct q to separate a from b; one q is rank-deficient.
        if np.unique(x[valid_idx]).size < 2 and np.all(fit_flag):
            logger.warning(
                "fit_tauq: fewer than 2 distinct positive q values; "
                "cannot fit both parameters"
            )
            tauq_result = _tauq_failure()
            self._memory_manager.cache_put(
                tauq_full_cache_key, tauq_result, CacheType.COMPUTATION
            )
            self.fit_summary.update(tauq_result)
            return self.fit_summary

        # Apply filtering efficiently
        x_valid = x[valid_idx]
        y_valid = y[valid_idx]
        sigma_valid = sigma[valid_idx]

        # Reshape arrays more efficiently
        y_reshaped = y_valid[:, np.newaxis]
        sigma_reshaped = sigma_valid[:, np.newaxis]

        # Improved initial parameter guess based on data characteristics
        # (every sample is strictly positive here, so log10 is finite)
        if len(x_valid) > 1:
            # Use data-driven initial guess for better convergence
            log_x = np.log10(x_valid)
            log_y = np.log10(y_valid)

            # Linear regression for initial slope estimate
            slope = np.polyfit(log_x, log_y, 1)[0] if len(log_x) > 1 else -2.0
            intercept = np.mean(log_y) - slope * np.mean(log_x)

            # Calculate initial guess and constrain to bounds
            p0_a = 10**intercept
            p0_b = slope

            # Constrain initial guess to be within bounds
            if bounds is not None and len(bounds) >= 2:
                # bounds structure: [[a_min, b_min], [a_max, b_max]]
                a_min, b_min = bounds[0]
                a_max, b_max = bounds[1]

                # Clamp p0_a to bounds
                p0_a = np.clip(p0_a, a_min, a_max)
                # Clamp p0_b to bounds
                p0_b = np.clip(p0_b, b_min, b_max)

                logger.debug(
                    f"Initial guess constrained: a={p0_a:.6e} (bounds: {a_min:.6e}-{a_max:.6e}), b={p0_b:.3f} (bounds: {b_min:.3f}-{b_max:.3f})"
                )

            p0 = [p0_a, p0_b]
        else:
            # Fallback initial guess - ensure it's within bounds if provided
            p0_a, p0_b = 1.0e-7, -2.0
            if bounds is not None and len(bounds) >= 2:
                a_min, b_min = bounds[0]
                a_max, b_max = bounds[1]
                p0_a = np.clip(p0_a, a_min, a_max)
                p0_b = np.clip(p0_b, b_min, b_max)
                logger.debug(
                    f"Fallback initial guess constrained: a={p0_a:.6e}, b={p0_b:.3f}"
                )
            p0 = [p0_a, p0_b]

        # Cache fit_x calculation
        x_range_key = f"{self._cache_prefix}_tauq_fit_x_{np.min(x_valid):.6e}_{np.max(x_valid):.6e}"
        cached_tauq_fit_x = self._memory_manager.cache_get(
            x_range_key, CacheType.COMPUTATION
        )
        if cached_tauq_fit_x is not None:
            fit_x = cached_tauq_fit_x
        else:
            fit_x = np.logspace(
                np.log10(np.min(x_valid) / 1.1), np.log10(np.max(x_valid) * 1.1), 128
            )
            self._memory_manager.cache_put(x_range_key, fit_x, CacheType.COMPUTATION)

        from xpcsviewer.fitting import fit_with_fixed
        from xpcsviewer.xpcs_file.fitting import power_law

        fit_line, fit_val, solver_ok = fit_with_fixed(
            power_law,
            x_valid,
            y_reshaped,
            sigma_reshaped,
            bounds,
            fit_flag,
            fit_x,
            p0=p0,
            return_status=True,
        )

        # Store tauq fitting results
        # The solver's own status comes first: on failure fit_with_fixed returns
        # finite bounds-midpoint placeholders that must not be reported as a fit.
        fitting_success = (
            bool(np.all(solver_ok))
            and fit_line is not None
            and fit_val is not None
            and fit_line.shape[0] > 0
            and fit_val.shape[0] > 0
            and np.all(np.isfinite(fit_line[0]))
            and np.all(np.isfinite(fit_val[0, 0, :]))
        )

        tauq_result = {
            "tauq_success": fitting_success,
            "tauq_q": x_valid,
            "tauq_tau": y_valid,
            "tauq_tau_err": sigma_valid,
            "tauq_fit_line": {"fit_x": fit_x, "fit_y": fit_line[0]}
            if fitting_success
            else None,
            "tauq_fit_val": fit_val[0]
            if fitting_success
            else np.full((2, len(fit_flag)), np.nan),
        }

        # Cache the result using unified memory manager
        self._memory_manager.cache_put(
            tauq_full_cache_key, tauq_result, CacheType.COMPUTATION
        )
        self.fit_summary.update(tauq_result)

        return self.fit_summary

    def get_roi_data_vectorized(
        self, roi_parameter, phi_num=180, use_vectorized=True, use_parallel=False
    ):
        """Compatibility entry point; the ROI profile comes from ``get_roi_data``.

        The vectorized calculators in ``utils.vectorized_roi`` need static bin
        edges and angle maps that a loaded file does not provide, so there is
        one ROI implementation (``get_roi_data``) and these arguments are
        accepted but ignored.
        """
        return self.get_roi_data(roi_parameter, phi_num)

    def get_multiple_roi_data_parallel(self, roi_list, phi_num=180, max_workers=None):
        """Return ``[(x, y), ...]`` for each ROI in ``roi_list`` (see ``get_roi_data``)."""
        return [self.get_roi_data(roi, phi_num) for roi in roi_list]

    def _roi_geometry(self):
        """Calibrated per-pixel geometry for ROI profiles, from the file's own QMap.

        Returns ``(r_pixel, phi, qbin, qlist, mask)``:

        - ``r_pixel``: distance from the beam centre [pixel]
        - ``phi``: azimuth of each pixel [degree], wrapped to [0, 360)
        - ``qbin``: 1-based index into ``qlist`` of the static q bin that owns
          the pixel (0 = in no static bin). Read from the canonical ``sqmap``
          through ``static_index_mapping``: bin k of ``sqmap`` is flat index
          ``static_index_mapping[k - 1]`` of the row-major
          ``(len(sqlist), len(splist))`` static grid (the layout of
          ``QMap.reshape_phi_analysis``), i.e. q bin ``index // len(splist)``.
        - ``qlist``: q of each static q bin [the file's static-axis unit, see
          ``_static_axis``]
        - ``mask``: detector mask (> 0 is a valid pixel)
        """
        maps = self.qmap.qmap
        if "r_pixel" not in maps or "phi" not in maps:
            raise ValueError(
                f"{self.fname}: the Q map has no per-pixel geometry (it failed to "
                "load); ROI profiles are unavailable"
            )
        qlist = np.asarray(self.sqlist, dtype=float)
        sqmap = np.asarray(self.sqmap)
        q_of_bin = np.asarray(self.qmap.static_index_mapping) // max(
            len(self.splist), 1
        )
        if q_of_bin.size and q_of_bin.max() >= len(qlist):
            raise ValueError(
                f"{self.fname}: static_index_mapping addresses q bin "
                f"{q_of_bin.max()} but the file lists {len(qlist)} static q values"
            )
        owned = (sqmap >= 1) & (sqmap <= len(q_of_bin))
        qbin = np.zeros(sqmap.shape, dtype=np.int64)
        qbin[owned] = q_of_bin[sqmap[owned].astype(np.int64) - 1] + 1
        return (
            np.asarray(maps["r_pixel"], dtype=float),
            np.mod(np.asarray(maps["phi"], dtype=float), 360.0),
            qbin,
            qlist,
            np.asarray(self.mask),
        )

    @staticmethod
    def _binned_mean(index, values, nbins, chunk=None):
        """Mean of ``values`` per 1-based bin of ``index`` (0 = excluded).

        Bins that received no pixel are NaN. ``chunk`` accumulates the
        histograms in slices of that many pixels; the result is identical.
        """
        total = np.zeros(nbins)
        count = np.zeros(nbins)
        step = max(1, index.size if chunk is None else int(chunk))
        for start in range(0, index.size, step):
            idx = index[start : start + step]
            total += np.bincount(idx, values[start : start + step], nbins + 1)[1:]
            count += np.bincount(idx, minlength=nbins + 1)[1:]
        return np.divide(total, count, out=np.full(nbins, np.nan), where=count > 0)

    # The frame a ROI is DRAWN in. ``ImageItem`` draws pixel (row, col) over the
    # square [col, col+1] x [row, row+1] of its own coordinates, so that pixel's
    # centre is (col + 0.5, row + 0.5) there, and a ROI added to the same view
    # lives in the image item's PARENT frame, reached through the item's
    # transform. ``ImageViewDev.get_roi_list`` reads that mapping from the real
    # item and puts it in each descriptor as ``pixel_frame``; this default is the
    # frame of an unmodified item (identity transform), used only for a
    # descriptor that carries none. ``origin`` is where pixel (0, 0)'s centre is,
    # ``col_step`` / ``row_step`` are the displacements for one more column / row.
    #
    # Calibration (``QMap.compute_qmap``) puts pixel i at offset ``i - bcx``, i.e.
    # the beam centre is in a pixel-INDEX frame where pixel i's centre is i. The
    # display frame differs from that by a constant half pixel. Nothing in the
    # file says whether the stored ``bcx`` / ``bcy`` follow the index convention
    # (centre of pixel 0 at 0) or the display convention (corner of pixel 0 at
    # 0), so the export does not guess: it selects what is drawn, and the Q of
    # each selected pixel is the file's own ``sqmap`` bin.
    _DISPLAY_PIXEL_FRAME: ClassVar[dict[str, tuple[float, float]]] = {
        "origin": (0.5, 0.5),
        "col_step": (1.0, 0.0),
        "row_step": (0.0, 1.0),
    }

    @classmethod
    def _pixel_centres(cls, shape, pixel_frame=None):
        """``(x, y)`` arrays of shape ``shape``: each pixel centre in the ROI frame."""
        frame = pixel_frame or cls._DISPLAY_PIXEL_FRAME
        rows, cols = np.indices(shape)
        (ox, oy), (cx, cy), (rx, ry) = (
            frame["origin"],
            frame["col_step"],
            frame["row_step"],
        )
        return ox + cols * cx + rows * rx, oy + cols * cy + rows * ry

    @classmethod
    def _pixels_in_polygon(
        cls, polygon, shape, origin=(0.0, 0.0), rotation=0.0, pixel_frame=None
    ):
        """Boolean ``shape = (rows, cols)`` mask of the pixels inside a polygon.

        ``polygon`` is ``[(x, y), ...]`` in a local frame that sits at ``origin``
        in the view frame and is rotated by ``rotation`` [degree], exactly as a
        ``pg.ROI``: ``view = origin + R(rotation) @ local`` with
        ``R = [[cos, -sin], [sin, cos]]``. A pixel belongs to the polygon iff its
        CENTRE, placed in the view frame by ``pixel_frame`` (see
        ``_DISPLAY_PIXEL_FRAME``), is inside it. Each centre is moved into the
        local frame and tested with the even-odd rule and scan-conversion
        convention of ``QPainterPath.contains`` (a ray towards -x; an edge counts
        for ``y1 <= y < y2``; crossings at ``x <= px``; horizontal edges ignored),
        so a centre exactly on an edge is decided as Qt decides it for the drawn
        shape. (Qt's rotation is exact for multiples of 90 degrees; so is this.)
        """
        pts = np.asarray(polygon, dtype=float)
        centre_x, centre_y = cls._pixel_centres(shape, pixel_frame)
        turn = float(rotation) % 360.0
        exact = {
            0.0: (1.0, 0.0),
            90.0: (0.0, 1.0),
            180.0: (-1.0, 0.0),
            270.0: (0.0, -1.0),
        }
        cos, sin = exact.get(turn) or (
            np.cos(np.radians(turn)),
            np.sin(np.radians(turn)),
        )
        dx, dy = centre_x - origin[0], centre_y - origin[1]
        px, py = dx * cos + dy * sin, -dx * sin + dy * cos  # R(-rotation) @ (dx, dy)
        winding = np.zeros(shape, dtype=np.int32)
        for (x1, y1), (x2, y2) in zip(pts, np.roll(pts, -1, axis=0), strict=True):
            if np.isclose(y1, y2, rtol=1e-12, atol=0.0):
                continue
            direction = 1
            if y2 < y1:
                x1, y1, x2, y2, direction = x2, y2, x1, y1, -1
            crossing = (py >= y1) & (py < y2)
            x_at = x1 + (x2 - x1) / (y2 - y1) * (py - y1)
            winding += direction * (crossing & (x_at <= px))
        in_box = (
            (px >= pts[:, 0].min())
            & (px <= pts[:, 0].max())
            & (py >= pts[:, 1].min())
            & (py <= pts[:, 1].max())
        )
        return in_box & (winding % 2 != 0)

    def _roi_profile(self, roi_parameter, phi_num, chunk=None):
        """Pie or Ring ROI profile ``(x, y)``; None for other ROI types."""
        sl_type = roi_parameter["sl_type"]
        if sl_type not in ("Pie", "Ring"):
            return None
        r_pixel, phi, qbin, qlist, mask = self._roi_geometry()
        saxs = np.asarray(self.saxs_2d).ravel()

        if sl_type == "Pie" and roi_parameter.get("polygon") is not None:
            # The drawn region itself: exactly the (unmasked) detector pixels
            # inside it, binned by the file's static q bin. No further angle or
            # radius cut: the polygon already is the finite wedge. Bins that
            # receive no pixel are NaN.
            inside = self._pixels_in_polygon(
                roi_parameter["polygon"],
                qbin.shape,
                roi_parameter.get("origin", (0.0, 0.0)),
                roi_parameter.get("rotation", 0.0),
                roi_parameter.get("pixel_frame"),
            )
            y = self._binned_mean(
                np.where(inside & (mask > 0), qbin, 0).ravel(), saxs, len(qlist), chunk
            )
            return qlist, np.where(y <= 0, np.nan, y)

        if sl_type == "Pie":
            # Legacy descriptor without the drawn polygon: an angular wedge about
            # the beam centre cut at `dist`. The cut-off below is a q in inverse
            # angstrom, so it can only be compared with a static axis in that
            # unit; no GUI path produces this descriptor any more.
            _, unit = self._static_axis()
            if unit != Q_UNIT_DISPLAY:
                raise ValueError(
                    f"{self.fname}: a Pie ROI without its drawn polygon is cut at a "
                    f"q in {Q_UNIT_DISPLAY}, but this file's static q axis is in "
                    f"{unit}; draw the Pie ROI in the viewer instead"
                )
            # `angle_range` [degree] is in the frame of PieROI.get_parameter:
            # the direction (column, row) of the pixel from the beam centre,
            # theta = atan2(row, column), measured from 6 o'clock, theta - 90.
            # QMap phi = -theta, so that angle is (-phi - 90) mod 360.
            phi = np.mod(-phi - 90.0, 360.0)
            pmin, pmax = roi_parameter["angle_range"]
            if pmax < pmin:  # the wedge crosses 0 degree
                pmax += 360.0
                phi = np.where(phi < pmin, phi + 360.0, phi)
            inside = (phi >= pmin) & (phi < pmax) & (mask > 0)
            y = self._binned_mean(
                np.where(inside, qbin, 0).ravel(), saxs, len(qlist), chunk
            )
            # `dist` is the wedge radius in pixels. A pixel `dist` from the beam
            # centre scatters by alpha = arctan(dist * pixel_size / det_dist)
            # (pixel_size and det_dist in the same length unit), and
            # q = k0 sin(alpha) with k0 = 2 pi / wavelength [1/Angstrom]; this
            # is the relation QMap.compute_qmap uses for the q map itself.
            qmax = self.qmap.k0 * np.sin(
                np.arctan(roi_parameter["dist"] * self.pixel_size / self.det_dist)
            )
            return qlist, np.where((qlist >= qmax) | (y <= 0), np.nan, y)

        rmin, rmax = sorted(roi_parameter["radius"])  # pixels
        if roi_parameter.get("center") is not None:
            # Annulus about the CENTRE THAT WAS DRAWN, in the frame it was drawn
            # in (view x = column direction, y = row direction), tested at each
            # pixel's centre in that frame (see _DISPLAY_PIXEL_FRAME). It is not
            # necessarily this file's beam centre; phi follows QMap.compute_qmap:
            # phi = -atan2(y - cy, x - cx).
            cx, cy = roi_parameter["center"]
            frame = roi_parameter.get("pixel_frame") or self._DISPLAY_PIXEL_FRAME
            px, py = self._pixel_centres(r_pixel.shape, frame)
            r_pixel = np.hypot(px - cx, py - cy)
            # phi is the azimuth in the DETECTOR frame (QMap phi), so the drawn
            # centre goes back to (column, row) through the display mapping: a
            # rotated display must not rotate phi.
            steps = np.array([frame["col_step"], frame["row_step"]], dtype=float).T
            col0, row0 = np.linalg.solve(
                steps, np.array([cx, cy]) - np.asarray(frame["origin"], dtype=float)
            )
            rows, cols = np.indices(r_pixel.shape)
            phi = np.mod(-np.degrees(np.arctan2(rows - row0, cols - col0)), 360.0)
        annulus = (r_pixel >= rmin) & (r_pixel < rmax) & (mask > 0)
        if not annulus.any():
            logger.warning(f"Ring ROI radius {(rmin, rmax)} contains no pixels")
            return np.empty(0), np.empty(0)
        phi_min, phi_max = phi[annulus].min(), phi[annulus].max()
        if phi_min == phi_max:
            # zero angular span: a single angle bin holding the mean intensity
            return np.array([phi_min]), np.array([np.mean(saxs[annulus.ravel()])])
        index = ((phi - phi_min) / ((phi_max - phi_min) / phi_num)).astype(np.int64)
        index = np.where(annulus, np.clip(index, 0, phi_num - 1) + 1, 0).ravel()
        x = np.linspace(phi_min, phi_max, phi_num)  # degree
        return x, self._binned_mean(index, saxs, phi_num, chunk)

    def get_roi_data(self, roi_parameter, phi_num=180):
        """Profile of a Pie ROI (I vs q) or Ring ROI (I vs phi [degree]).

        q is in the file's static-axis unit (``_static_axis``). ``roi_parameter``
        is a ROI descriptor from ``ImageViewDev.get_roi_list``: Pie
        ``{"sl_type", "polygon", "origin", "rotation", "pixel_frame", ...}`` and
        Ring ``{"sl_type", "radius", "center", "pixel_frame"}`` (the drawn region
        and the image item's pixel-centre frame, see ``_DISPLAY_PIXEL_FRAME``).
        Returns None for other ROI types.
        """
        memory_before_mb, _ = MemoryMonitor.get_memory_usage()

        # Check if we should use streaming for very large datasets
        saxs_size_mb = (
            self.saxs_2d.nbytes / (1024 * 1024)
            if hasattr(self.saxs_2d, "nbytes")
            else 0
        )
        use_streaming = (
            saxs_size_mb > 500  # Large dataset > 500MB
            and self._memory_manager.get_memory_pressure()
            in [MemoryPressure.HIGH, MemoryPressure.CRITICAL]
        )

        if use_streaming:
            logger.info(
                f"Using streaming ROI calculation for large dataset ({saxs_size_mb:.1f}MB)"
            )
            streamed = self._get_roi_data_streaming(roi_parameter, phi_num)
            if streamed is not None:
                return streamed
            logger.info(f"No streaming ROI for {roi_parameter['sl_type']}; using eager")

        result = self._roi_profile(roi_parameter, phi_num)
        memory_used = MemoryMonitor.get_memory_usage()[0] - memory_before_mb
        if memory_used > 5:  # Log if ROI calculation used significant memory
            logger.debug(f"ROI calculation completed: {memory_used:.1f}MB memory used")
        return result

    def _get_roi_data_streaming(self, roi_parameter, phi_num=180):
        """``get_roi_data`` with chunked histograms for very large SAXS images.

        Same result as the eager path; None for ROI types it does not cover.
        """
        logger.info("Starting streaming ROI calculation")
        chunk = max(1, min(1_000_000, np.asarray(self.saxs_2d).size // 10))
        return self._roi_profile(roi_parameter, phi_num, chunk)

    def _q_header_label(self):
        """``name(unit)`` of the static axis, ASCII only (``np.savetxt`` header)."""
        name, unit = self._static_axis()
        ascii_units = {Q_UNIT_DISPLAY: "1/Angstrom", "nm⁻¹": "1/nm"}
        unit = ascii_units.get(unit) or unit.encode("ascii", "replace").decode()
        return f"{name}({unit.replace(' ', '_')})"

    def export_saxs1d(
        self, roi_list, folder, basename=None, guard: ExportGuard | None = None
    ):
        """Write ROI profiles and the full SAXS 1D curves into ``folder``.

        ``basename`` prefixes every output name. ViewerKernel passes a stem that
        is unique across the whole export; the default is the source file's own
        name (display labels are not unique). Every destination is claimed on
        ``guard`` before it is written: the kernel passes one guard per export
        run; without one, this file is the only protected input.
        """
        if guard is None:
            guard = ExportGuard([self.fname])
        source = str(self.fname)
        if basename is None:
            basename = os.path.splitext(os.path.basename(source))[0]
        q_label = self._q_header_label()
        # export ROI
        for idx, roi in enumerate(roi_list):
            profile = self.get_roi_data(roi)
            if profile is None:
                logger.warning(f"ROI type {roi['sl_type']} cannot be exported; skipped")
                continue
            fname = os.path.join(
                folder, basename + "_" + roi["sl_type"] + f"_{idx:03d}.txt"
            )
            if roi["sl_type"] == "Ring":
                header = "phi(degree) Intensity"
            else:
                header = f"{q_label} Intensity"
            guard.claim(fname, source=source)
            np.savetxt(fname, np.vstack(profile).T, header=header)

        # export all saxs1d
        fname = os.path.join(folder, basename + "_" + "saxs1d.txt")
        Iq, q = self.saxs_1d["Iq"], self.saxs_1d["q"]
        header = f"{q_label} Intensity"
        for n in range(Iq.shape[0] - 1):
            header += f" Intensity_phi{n + 1:03d}"
        guard.claim(fname, source=source)
        np.savetxt(fname, np.vstack([q, Iq]).T, header=header)

    def get_pg_tree(self):
        data = self.load_data()
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                if val.size > 4096:
                    data[key] = "data size is too large"
                # suqeeze one-element array
                if val.size == 1:
                    data[key] = float(val)
        data["analysis_type"] = self.atype
        data["label"] = self.label
        tree = pg.DataTreeWidget(data=data)
        tree.setWindowTitle(self.fname)
        tree.resize(600, 800)
        return tree

    def get_memory_usage_report(self) -> dict:
        """
        Generate a memory usage report for the loaded data.

        Returns
        -------
        dict
            Memory usage information
        """
        report = {
            "file": self.fname,
            "label": self.label,
            "arrays": {},
            "total_memory_mb": 0.0,
        }

        # Check memory usage of major arrays
        array_attrs = [
            "saxs_2d_data",
            "saxs_2d_log_data",
            "g2",
            "g2_err",
            "tau",
            "t_el",
        ]

        for attr in array_attrs:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                arr = getattr(self, attr)
                if isinstance(arr, np.ndarray):
                    memory_mb = arr.nbytes / (1024 * 1024)
                    optimal_dtype = arr.dtype  # Use current dtype as fallback

                    report["arrays"][attr] = {
                        "shape": arr.shape,
                        "dtype": str(arr.dtype),
                        "memory_mb": memory_mb,
                        "optimal_dtype": str(optimal_dtype),
                        "can_optimize": optimal_dtype != arr.dtype,
                    }
                    report["total_memory_mb"] += memory_mb

        return report

    def optimize_memory_usage(self, optimize_dtypes: bool = True) -> dict:
        """
        Optimize memory usage of loaded arrays.

        Parameters
        ----------
        optimize_dtypes : bool
            Whether to optimize data types

        Returns
        -------
        dict
            Optimization results
        """
        results = {"optimizations_applied": [], "memory_saved_mb": 0.0, "errors": []}

        if optimize_dtypes:
            # Optimize dtypes for suitable arrays
            array_attrs = ["g2_err", "tau"]  # Start with safer arrays

            for attr in array_attrs:
                if hasattr(self, attr) and getattr(self, attr) is not None:
                    try:
                        arr = getattr(self, attr)
                        if isinstance(arr, np.ndarray):
                            original_memory = arr.nbytes / (1024 * 1024)
                            optimal_dtype = arr.dtype  # Use current dtype as fallback

                            if optimal_dtype != arr.dtype:
                                optimized_arr = arr.astype(optimal_dtype)
                                setattr(self, attr, optimized_arr)

                                new_memory = optimized_arr.nbytes / (1024 * 1024)
                                saved_mb = original_memory - new_memory

                                results["optimizations_applied"].append(
                                    {
                                        "array": attr,
                                        "old_dtype": str(arr.dtype),
                                        "new_dtype": str(optimal_dtype),
                                        "memory_saved_mb": saved_mb,
                                    }
                                )
                                results["memory_saved_mb"] += saved_mb

                    except Exception as e:
                        results["errors"].append(f"Failed to optimize {attr}: {e!s}")

        return results

    def _generate_cache_key(self, method_name, *args, **kwargs):
        """Generate a cache key from method parameters."""
        # Convert all parameters to a string representation
        key_parts = [method_name]

        for arg in args:
            if arg is None:
                key_parts.append("None")
            elif isinstance(arg, (list, tuple)):
                # Handle lists/tuples by converting to string
                # Order matters (e.g. fit_flag [T, F] vs [F, T]); never sort.
                key_parts.append(str(arg))
            elif isinstance(arg, (dict,)):
                # Handle dictionaries by sorting keys
                key_parts.append(str(sorted(arg.items())))
            else:
                key_parts.append(str(arg))

        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")

        # Create hash from concatenated parts
        cache_string = "|".join(key_parts)
        return hashlib.md5(cache_string.encode(), usedforsecurity=False).hexdigest()

    def _compute_int_t_fft_cached(self):
        """
        Compute and cache FFT of intensity vs time data.
        Returns tuple of (frequencies, fft_magnitudes)
        """
        cache_key = f"{self._cache_prefix}_int_t_fft"
        cached_result = self._memory_manager.cache_get(cache_key, CacheType.COMPUTATION)
        if cached_result is not None:
            return cached_result

        try:
            # Get intensity vs time data
            if not hasattr(self, "Int_t") or self.Int_t is None:
                return np.array([]), np.array([])

            # Int_t[1] contains the intensity data
            intensity_data = self.Int_t[1]

            if len(intensity_data) == 0:
                return np.array([]), np.array([])

            # Compute FFT
            fft_values = np.fft.fft(intensity_data)
            fft_magnitudes = np.abs(fft_values)

            # Create frequency array
            n_samples = len(intensity_data)
            sample_rate = 1.0  # Assuming 1 Hz sample rate, adjust if needed
            frequencies = np.fft.fftfreq(n_samples, 1.0 / sample_rate)

            # Take only positive frequencies
            positive_freq_mask = frequencies > 0
            frequencies = frequencies[positive_freq_mask]
            fft_magnitudes = fft_magnitudes[positive_freq_mask]

            result = (frequencies, fft_magnitudes)

            # Cache the result using unified memory manager
            self._memory_manager.cache_put(cache_key, result, CacheType.COMPUTATION)

            return result

        except Exception as e:
            logger.error(f"Failed to compute Int_t FFT: {e}")
            return np.array([]), np.array([])

    @log_timing(threshold_ms=100)
    def clear_cache(self, cache_type: str = "all"):
        """
        Clear cached data to free memory.

        Parameters
        ----------
        cache_type : str
            Type of cache to clear:
            - 'all': Clear all caches (default)
            - 'computation': Clear only computation cache
            - 'saxs': Clear only SAXS 2D data
            - 'fit': Clear only fitting results
        """
        memory_before = MemoryMonitor.get_memory_usage()[0]

        if cache_type in ("all", "computation"):
            # Clear computation cache entries for this instance from unified memory manager
            stats_before = self._memory_manager.get_cache_stats()
            stats_before.get("computation_entries", 0)

            # We can't directly clear specific entries by prefix, so we'll track what we had
            # This is a limitation of the current interface that could be improved
            logger.info(
                "Clearing computation cache (unified memory manager handles cleanup)"
            )

        if cache_type in ("all", "saxs") and self._saxs_data_loaded:
            # Handle both regular arrays and lazy-loaded data
            if isinstance(self._saxs_2d_data, LazyHDF5Array):
                # For lazy-loaded data, just clear the loaded data, keep the proxy
                if hasattr(self._saxs_2d_data, "_loaded_data"):
                    self._saxs_2d_data._loaded_data = None
                logger.info("Cleared lazy-loaded SAXS 2D data cache")
            else:
                self._saxs_2d_data = None
                logger.info("Cleared SAXS 2D data cache")

            if isinstance(self._saxs_2d_log_data, LazyHDF5Array):
                if hasattr(self._saxs_2d_log_data, "_loaded_data"):
                    self._saxs_2d_log_data._loaded_data = None
                logger.info("Cleared lazy-loaded SAXS 2D log data cache")
            else:
                self._saxs_2d_log_data = None
                logger.info("Cleared SAXS 2D log data cache")

            self._saxs_data_loaded = False

        if cache_type in ("all", "fit"):
            if self.fit_summary is not None:
                self.fit_summary = None
                logger.info("Cleared fitting results cache")

            if self.c2_all_data is not None:
                self.c2_all_data = None
                self.c2_kwargs = None
                logger.info("Cleared two-time C2 data cache")

        # Clear unified memory manager cache entries for this instance
        # Note: This is a simplified approach - in a full implementation we might add
        # a method to clear entries by prefix to the memory manager
        logger.debug(f"Cleared cache for instance {self._cache_prefix}")

        memory_after = MemoryMonitor.get_memory_usage()[0]
        memory_freed = memory_before - memory_after

        if memory_freed > 1.0:  # Only log if significant memory was freed
            logger.info(f"Cache clear freed {memory_freed:.2f}MB of memory")

        # Schedule smart garbage collection to ensure memory is released
        try:
            from .threading.cleanup_optimized import smart_gc_collect

            smart_gc_collect("cache_clear")
        except ImportError:
            # Fallback to manual GC if optimized system not available
            import gc

            gc.collect()

    def get_cache_stats(self) -> dict:
        """
        Get statistics about cached data.

        Returns
        -------
        dict
            Cache statistics including memory usage
        """
        stats = {
            "file": self.fname,
            "unified_memory_manager": self._memory_manager.get_cache_stats(),
            "lazy_loader_stats": self._lazy_loader.get_memory_stats(),
            "saxs_data_loaded": self._saxs_data_loaded,
            "saxs_2d_is_lazy": isinstance(self._saxs_2d_data, LazyHDF5Array),
            "saxs_2d_log_is_lazy": isinstance(self._saxs_2d_log_data, LazyHDF5Array),
            "has_fit_summary": self.fit_summary is not None,
            "has_c2_data": self.c2_all_data is not None,
            "estimated_memory_mb": 0.0,
        }

        # Estimate memory usage of major cached items
        if self._saxs_data_loaded and self._saxs_2d_data is not None:
            if isinstance(self._saxs_2d_data, LazyHDF5Array):
                # For lazy data, only count loaded portion
                if (
                    hasattr(self._saxs_2d_data, "_loaded_data")
                    and self._saxs_2d_data._loaded_data is not None
                ):
                    stats["saxs_2d_memory_mb"] = MemoryMonitor.estimate_array_memory(
                        self._saxs_2d_data._loaded_data.shape,
                        self._saxs_2d_data._loaded_data.dtype,
                    )
                    stats["estimated_memory_mb"] += stats["saxs_2d_memory_mb"]
                else:
                    stats["saxs_2d_memory_mb"] = 0.0  # Not loaded yet
            else:
                stats["saxs_2d_memory_mb"] = MemoryMonitor.estimate_array_memory(
                    self._saxs_2d_data.shape, self._saxs_2d_data.dtype
                )
                stats["estimated_memory_mb"] += stats["saxs_2d_memory_mb"]

        if self._saxs_data_loaded and self._saxs_2d_log_data is not None:
            if isinstance(self._saxs_2d_log_data, LazyHDF5Array):
                if (
                    hasattr(self._saxs_2d_log_data, "_loaded_data")
                    and self._saxs_2d_log_data._loaded_data is not None
                ):
                    stats["saxs_2d_log_memory_mb"] = (
                        MemoryMonitor.estimate_array_memory(
                            self._saxs_2d_log_data._loaded_data.shape,
                            self._saxs_2d_log_data._loaded_data.dtype,
                        )
                    )
                    stats["estimated_memory_mb"] += stats["saxs_2d_log_memory_mb"]
                else:
                    stats["saxs_2d_log_memory_mb"] = 0.0  # Not loaded yet
            else:
                stats["saxs_2d_log_memory_mb"] = MemoryMonitor.estimate_array_memory(
                    self._saxs_2d_log_data.shape, self._saxs_2d_log_data.dtype
                )
                stats["estimated_memory_mb"] += stats["saxs_2d_log_memory_mb"]

        return stats


def test1():
    cwd = "../../../xpcs_data"
    af = XpcsFile(fname="N077_D100_att02_0128_0001-100000.hdf", cwd=cwd)
    af.plot_saxs2d()


if __name__ == "__main__":
    test1()
