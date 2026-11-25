# Standard library imports
import json
import os
import shutil
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any
from uuid import uuid4

# Third-party imports
import numpy as np
import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter

# Qt imports via compatibility layer
from xpcsviewer.gui.qt_compat import (
    QAction,
    QActionGroup,
    QDesktopServices,
    QKeySequence,
    QShortcut,
    QtCore,
    QtGui,
    QtWidgets,
)

# Import centralized logging
from .backends._conversions import ensure_numpy

# Import async components
from .constants import MIN_AVERAGING_FILES
from .fileIO.g2_regroup import G2RegroupResult
from .fileIO.hdf_reader import _file_identity
from .fileIO.qmap_utils import Q_UNIT_DISPLAY
from .gui.g2_regroup_dialog import G2RegroupDialog, G2RegroupExportWorker
from .gui.layout_helpers import apply_all_layout_improvements
from .gui.shortcuts.shortcut_manager import ShortcutManager

# Import GUI components
from .gui.state.recent_paths import RecentPathsManager
from .gui.state.session_manager import (
    AnalysisParameters,
    FileEntry,
    SessionManager,
    SessionState,
    WindowGeometry,
)
from .gui.theme.manager import ThemeManager
from .gui.widgets.command_palette import CommandPalette
from .gui.widgets.drag_drop_list import DragDropListView
from .gui.widgets.toast_notification import ToastManager

# Import PlotWidgetDev for dynamic tab creation
from .plothandler import PlotWidgetDev

# Local imports
from .simplemask import SimpleMaskWindow
from .threading.async_kernel import AsyncDataPreloader, AsyncViewerKernel
from .threading.progress_manager import ProgressManager
from .utils import get_logger, log_system_info, sanitize_path, setup_exception_logging
from .utils.log_utils import RateLimitedLogger
from .viewer_kernel import ViewerKernel
from .viewer_ui import Ui_mainWindow as Ui

if TYPE_CHECKING:  # the fitting package is imported lazily (JAX start-up cost)
    from .fitting.bayesian_assembly import BayesianBundle

# Initialize centralized logging and get logger
logger = get_logger(__name__)

# Cap on remembered per-file twotime qbin selections (LRU-evicted).
_TWOTIME_QBIN_MEMORY_MAX = 256


def _remember_twotime_qbin(memory, fname, index):
    memory[fname] = index
    memory.move_to_end(fname)
    if len(memory) > _TWOTIME_QBIN_MEMORY_MAX:
        memory.popitem(last=False)


# Setup exception logging
setup_exception_logging()

# Ensure home directory exists (moved after logging setup)
home_dir = os.path.join(os.path.expanduser("~"), ".xpcsviewer")
if not os.path.isdir(home_dir):
    os.mkdir(home_dir)
    logger.info(f"Created home directory: {home_dir}")

# Log system information for debugging
log_system_info()


tab_mapping = {
    0: "saxs_2d",
    1: "saxs_1d",
    2: "stability",
    3: "intensity_t",
    4: "g2",
    5: "g2_fitting",  # New g2 fitting tab
    6: "g2_map",
    7: "diffusion",
    8: "twotime",
    9: "qmap",
    10: "average",
    11: "metadata",
}


def _revision_of(path: str | None) -> tuple[int, ...] | None:
    """On-disk revision of a data file: the identity the locator records for
    the file it loaded (inode, device, mtime, size, ctime); None if unreadable."""
    return None if path is None else _file_identity(os.path.normpath(path))


def _detector_shape(xf) -> tuple[int, ...] | None:
    """Shape of the file's detector image (None if it has none)."""
    image = getattr(xf, "saxs_2d", None)
    return None if image is None else tuple(np.shape(image))


def _split_by_detector_shape(xf_list, arrays):
    """Partition ``xf_list`` into files an import may be applied to and the rest.

    One array (a SimpleMask mask or partition map) is pushed onto several
    targets whose detectors can differ. A file takes it only if EVERY array in
    ``arrays`` has that file's own detector shape; this is the only place that
    decides, so no import can leave a file holding an array of the wrong shape.

    Returns ``(accepted, skipped)``; ``skipped`` is ``[(file name, reason)]``.
    """
    accepted, skipped = [], []
    for xf in xf_list:
        shape = _detector_shape(xf)
        name = os.path.basename(str(getattr(xf, "fname", "?")))
        wrong = [tuple(np.shape(a)) for a in arrays if tuple(np.shape(a)) != shape]
        if shape is None:
            skipped.append((name, "its detector shape is unknown"))
        elif wrong:
            skipped.append(
                (name, f"detector {shape} does not match the imported {wrong[0]}")
            )
        else:
            accepted.append(xf)
    return accepted, skipped


def _skipped_text(skipped) -> str:
    return "; ".join(f"{name} ({reason})" for name, reason in skipped)


def _tauq_rows(fit_summary, q_range):
    """``(q, tau, tau_err)`` rows of the tau-q power-law fit for *q_range*.

    Source of truth: ``XpcsFile.fit_tauq`` (xpcs_file.py).  This is the same
    selection, kept identical on purpose (a test compares both on a real file):
    only Q bins inside *q_range* whose G2 fit succeeded
    (``fit_success_mask``), with finite positive q and tau and a positive finite
    uncertainty.  ``fit_tauq`` and the Bayesian launch must never disagree about
    which rows are observations.
    """
    from .fitting.legacy import fit_success_mask

    x = np.asarray(fit_summary["q_val"]).ravel()
    fit_val = np.asarray(fit_summary["fit_val"])
    n = min(len(x), fit_val.shape[0])
    x, fit_val = x[:n], fit_val[:n]
    in_range = (x >= q_range[0]) & (x <= q_range[1])
    in_range &= fit_success_mask({**fit_summary, "fit_val": fit_val})[:n]
    q, tau, err = x[in_range], fit_val[in_range, 0, 1], fit_val[in_range, 1, 1]
    valid = (err > 0) & np.isfinite(tau) & np.isfinite(err)
    valid &= np.isfinite(q) & (q > 0) & (tau > 0)
    return q[valid], tau[valid], err[valid]


@dataclass(frozen=True)
class _BayesianExportSnapshot:
    """What a modeless All-Q dialog shows: summary, results, file, revision.

    Taken when the figure is created; the figure and every export of that
    dialog read it, never the viewer's current (mutable) state.
    """

    summary: Mapping[str, Any] | None
    results: Mapping[Any, Any] | None
    fname: str | None
    revision: tuple[int, ...] | None


def _bayesian_export_snapshot(xf) -> _BayesianExportSnapshot:
    """Freeze what the All-Q figure about to be shown was built from."""
    summary = getattr(xf, "bayesian_fit_summary", None)
    results = getattr(xf, "bayesian_results", None)
    return _BayesianExportSnapshot(
        summary=None if summary is None else MappingProxyType(dict(summary)),
        results=None if results is None else MappingProxyType(dict(results)),
        fname=xf.fname,
        revision=_revision_of(xf.fname),
    )


def create_param_tree(data_dict):
    """Convert a dictionary into PyQtGraph's ParameterTree format."""
    params = []
    for key, value in data_dict.items():
        if isinstance(value, dict):  # If value is a nested dictionary
            params.append(
                {"name": key, "type": "group", "children": create_param_tree(value)}
            )
        elif isinstance(value, (int, np.integer)):  # Integer types
            params.append({"name": key, "type": "int", "value": int(value)})
        elif isinstance(value, (float, np.floating)):  # Float types
            params.append(
                {
                    "name": key,
                    "type": "float",
                    "value": float(value),
                    "format": "{value:.10g}",
                }
            )
        elif isinstance(value, str):  # String types
            params.append({"name": key, "type": "str", "value": value})
        elif isinstance(value, np.ndarray):  # Numpy arrays
            params.append({"name": key, "type": "text", "value": str(value.tolist())})
        else:  # Default fallback
            params.append({"name": key, "type": "text", "value": str(value)})
    return params


class XpcsViewer(QtWidgets.QMainWindow, Ui):
    """
    Main XPCS Viewer application window.

    XpcsViewer provides a comprehensive GUI for analyzing X-ray Photon Correlation
    Spectroscopy (XPCS) datasets. It supports both multi-tau and two-time correlation
    analysis with interactive plotting and real-time data visualization.

    The viewer integrates multiple analysis modules including SAXS 2D/1D visualization,
    G2 correlation analysis, stability monitoring, intensity-vs-time analysis,
    two-time correlation maps, and data averaging capabilities.

    Parameters
    ----------
    path : str, optional
        Initial directory path to load XPCS data files. If None, uses home directory.
    label_style : str, optional
        Custom labeling style for file identification. Format: comma-separated indices.

    Attributes
    ----------
    vk : ViewerKernel
        Backend kernel managing data processing and file operations
    async_vk : AsyncViewerKernel
        Asynchronous processing kernel for non-blocking operations
    progress_manager : ProgressManager
        Manager for tracking and displaying operation progress
    thread_pool : QThreadPool
        Thread pool for background computations

    Notes
    -----
    The application uses a tab-based interface where each tab represents a different
    analysis mode. Heavy computations are performed in background threads to maintain
    GUI responsiveness. The viewer supports both synchronous and asynchronous plotting
    modes for optimal performance.

    Examples
    --------
    >>> viewer = XpcsViewer(path="/path/to/data")
    >>> viewer.show()
    """

    def __init__(self, path=None, label_style=None, path_explicit=None):
        super().__init__()
        self.setupUi(self)
        self.home_dir = home_dir
        self.label_style = label_style
        # True when the caller named the startup directory: it then wins over
        # the workspace stored in the saved session (see _restore_session).
        # A supplied path is explicit unless the caller marks it as an
        # automatic default with path_explicit=False.
        self._path_explicit = (
            (path is not None) if path_explicit is None else bool(path_explicit)
        )

        self.tabWidget.setCurrentIndex(0)  # show scattering 2d
        self.plot_kwargs_record = {}
        for _, v in tab_mapping.items():
            self.plot_kwargs_record[v] = {}

        self.thread_pool = QtCore.QThreadPool()
        logger.info("Maximal threads: %d", self.thread_pool.maxThreadCount())
        self._g2_regroup_dialog: G2RegroupDialog | None = None
        self._g2_regroup_export: G2RegroupExportWorker | None = None

        # Initialize async components
        self.progress_manager = ProgressManager(self)
        self.progress_manager.set_statusbar(self.statusbar)

        # Initialize theme manager (applies saved theme preference on startup)
        self.theme_manager = ThemeManager(parent=self)
        # Connect theme changes to plot refresh
        self.theme_manager.theme_changed.connect(self._on_theme_changed)

        # Initialize session manager for workspace persistence
        self.session_manager = SessionManager()

        # Initialize recent paths manager
        self.recent_paths_manager = RecentPathsManager()

        # Initialize toast notification manager for user feedback
        self.toast_manager = ToastManager(parent=self)

        # Initialize command palette and shortcut manager
        self.command_palette = CommandPalette(parent=self)
        self.shortcut_manager = ShortcutManager(parent=self)

        self.vk = None
        self.async_vk = None  # Will be initialized when vk is ready
        self.data_preloader = None

        # Track initialization state for performance optimization
        self._startup_complete = False

        # Track active async operations
        self.active_plot_operations = {}  # operation_id -> plot_type
        self._plot_op_requests = {}  # operation_id -> kwargs it was started with
        # Tabs whose newer request was suppressed while an older operation ran.
        self._plot_pending: set[str] = set()
        self._retiring_plots = False  # True while callers retire ops on purpose
        # Bumped whenever the display partition of the loaded files changes
        # (import, or loss of the override by a reload): part of every plot request.
        self._display_partition_version = 0
        self._g2_fit_bounds_initialized = False

        # list widget models
        self.source_model = None
        self.target_model = None

        # Rate-limited logger for high-frequency GUI events (mouse moves, etc.)
        self._rate_limited_logger = RateLimitedLogger(logger, rate_per_second=2.0)

        # UI polish
        self._init_spacing()
        self._init_menus_and_toolbar()
        self._register_shortcuts()
        self._setup_drag_drop_list()
        self._apply_button_styles()
        self.timer = QtCore.QTimer()

        # Debounce SAXS-2D file-selection spinbox: each tick otherwise
        # triggers a full image re-render + ROI recreation immediately.
        self._saxs2d_selection_timer = QtCore.QTimer(self)
        self._saxs2d_selection_timer.setSingleShot(True)
        self._saxs2d_selection_timer.setInterval(120)
        self._saxs2d_selection_timer.timeout.connect(self.plot_saxs_2d_selection)

        # Must be initialized before load_path() which checks it
        self._g2_batch_coordinator: object | None = None

        if path is not None:
            self.start_wd = path
            self.load_path(path)
        else:
            # use home directory
            self.start_wd = os.path.expanduser("~")

        self.start_wd = os.path.abspath(self.start_wd)
        logger.info(f"Start up directory is [{self.start_wd}]")

        # Prompt for start path only after start_wd is defined
        self._maybe_prompt_start_path()

        # Zero-argument adapters: clicked(bool) must not reach the plot
        # functions' first parameter (an index / the dryrun flag).
        self.pushButton_plot_saxs2d.clicked.connect(lambda: self.plot_saxs_2d())
        self.pushButton_plot_saxs1d.clicked.connect(lambda: self.plot_saxs_1d())
        self.pushButton_plot_stability.clicked.connect(lambda: self.plot_stability())
        self.pushButton_plot_intt.clicked.connect(lambda: self.plot_intensity_t())
        self.pushButton_8.clicked.connect(
            lambda: self.plot_diffusion()
        )  # Diffusion "fit plot" button
        # viewer_ui.py (generated, not editable) wires these signals straight to
        # slots whose first parameter is a path / the dryrun flag: the combo's
        # int index (> 0) made plot_twotime a dry run that drew nothing.
        self._rewire_without_args(
            self.cb_twotime_cmap.currentIndexChanged, "plot_twotime"
        )
        self._rewire_without_args(self.pushButton_12.clicked["bool"], "plot_twotime")
        self._rewire_without_args(self.pushButton_4.clicked, "plot_g2")
        self._rewire_without_args(self.pushButton.clicked, "load_path")
        # self.saxs1d_lb_type.currentIndexChanged.connect(self.switch_saxs1d_line)

        self.tabWidget.currentChanged.connect(self.update_plot)
        self.list_view_target.clicked.connect(self.update_plot)

        self.mp_2t_hdls = None
        # fname -> last selected qbin index; bounded LRU so a long session
        # analyzing many files doesn't grow this without limit.
        self._twotime_qbin_memory: OrderedDict[str, int] = OrderedDict()
        self._twotime_last_fname = None
        self._init_twotime_tab()
        self.init_twotime_plot_handler()

        # Initialize G2 Fitting tab (dynamic creation - must be before g2 map)
        self._init_g2_fitting_tab()

        # Initialize G2 Map tab (dynamic creation)
        self._init_g2map_tab()

        # Apply scientific tab labels (after all dynamic tabs are created)
        self._apply_tab_labels()

        # Initialize SimpleMask window reference (will be created on demand)
        self._simplemask_window: SimpleMaskWindow | None = None

        # Bayesian fitting state — G2 (single-Q)
        # Invariant: a posterior is only ever shown, diagnosed or plotted with
        # the exact observations and physical bin it was fitted to.  Results
        # are therefore committed as BayesianBundle objects (result + captured
        # observations + file + absolute bin ID), keyed by (file name,
        # absolute bin ID) -- never by the relative index in the current Q range.
        # The key also carries the file's on-disk revision at launch:
        # (file name, revision, absolute bin ID).
        self._g2_bayesian_store: dict[tuple, BayesianBundle] = {}
        self._g2_bayesian_launch: BayesianBundle | None = None  # fit in flight
        # Revision of the launch's file when it was launched.  A diagnosis
        # window carries its own identity (file, revision, bin, bundle); the
        # viewer never tracks "the bundle a window shows".
        self._g2_bayesian_launch_revision: tuple | None = None
        self._bayesian_plot_dialog = None
        self._bayesian_plot_snapshot: _BayesianExportSnapshot | None = None
        self._g2_diagnosis_window = None
        self._g2_bayesian_worker_active: bool = False
        self._g2_bayesian_worker: object | None = None  # current single-Q worker

        # Bayesian fitting state — G2 (batch all-Q): everything captured when
        # the batch was launched (None when idle or after a workspace reset).
        self._g2_batch_launch: dict | None = None

        # Grid metadata for Q-bin subplot highlighting
        self._g2_fitting_num_col: int = 4
        self._g2_fitting_plot_type: str = "multiple"

        # Bayesian fitting state — Diffusion (same bundle rule, keyed by file)
        # keyed (file name, revision at launch)
        self._diff_bayesian_store: dict[tuple, BayesianBundle] = {}
        self._diff_bayesian_launch: BayesianBundle | None = None
        self._diff_bayesian_launch_revision: tuple | None = None
        self._diff_bayesian_worker: object | None = None
        self._diff_diagnosis_window = None
        self._diff_bayesian_worker_active: bool = False

        self.avg_job_pop.clicked.connect(self.remove_avg_job)
        self.btn_submit_job.clicked.connect(self.submit_job)
        self.btn_start_avg_job.clicked.connect(self.start_avg_job)
        self.btn_set_average_save_path.clicked.connect(self.set_average_save_path)
        self.btn_set_average_save_name.clicked.connect(self.set_average_save_name)
        self.btn_avg_kill.clicked.connect(self.avg_kill_job)
        self.btn_avg_jobinfo.clicked.connect(self.show_avg_jobinfo)
        self.show_g2_fit_summary.clicked.connect(self.show_g2_fit_summary_func)
        self.btn_g2_export.clicked.connect(self.export_g2)
        self.btn_g2_refit.clicked.connect(self.refit_g2)
        self.btn_g2_refit.setText("Fit")
        self.pushButton_6.clicked.connect(self.saxs2d_roi_add)  # ROI Add button
        self.saxs2d_autolevel.stateChanged.connect(self.update_saxs2d_level)
        self.btn_deselect.clicked.connect(self.clear_target_selection)
        self.list_view_target.doubleClicked.connect(self.show_dataset)
        self.btn_select_bkgfile.clicked.connect(self.select_bkgfile)
        self.spinBox_saxs2d_selection.valueChanged.connect(
            lambda: self._saxs2d_selection_timer.start()
        )
        self.comboBox_twotime_selection.currentIndexChanged.connect(
            self.on_twotime_q_selection_changed
        )

        self.g2_fitting_function.currentIndexChanged.connect(
            self.update_g2_fitting_function
        )
        self.btn_up.clicked.connect(lambda: self.reorder_target("up"))
        self.btn_down.clicked.connect(lambda: self.reorder_target("down"))

        self.btn_export_saxs1d.clicked.connect(self.saxs1d_export)
        self.btn_export_diffusion.clicked.connect(self.export_diffusion)

        # Bayesian fitting connections
        self.btn_g2_bayesian.clicked.connect(self._fit_g2_bayesian)
        self.btn_g2_diagnosis.clicked.connect(self._show_g2_diagnosis)
        self.sb_g2_bayesian_qidx.valueChanged.connect(self._on_g2_qbin_changed)
        # Set initial button labels to reflect default Q-bin 0
        self.btn_g2_bayesian.setText("Fit Q-0")
        self.btn_g2_diagnosis.setText("Diagnosis Q-0 (no fit)")
        self.btn_diff_bayesian.clicked.connect(self._fit_diff_bayesian)
        self.btn_diff_diagnosis.clicked.connect(self._show_diff_diagnosis)

        # Batch Bayesian fitting UI — programmatically added (viewer_ui.py is
        # auto-generated and must not be edited).
        from qtpy.QtWidgets import QPushButton, QSpinBox

        self.btn_g2_bayesian_all = QPushButton("Fit All Q", self.groupBox_2)
        self.btn_g2_bayesian_all.setObjectName("btn_g2_bayesian_all")
        self.btn_g2_bayesian_all.setVisible(
            False
        )  # Hidden from GUI; kept for scripting

        self.sb_g2_bayesian_workers = QSpinBox(self.groupBox_2)
        self.sb_g2_bayesian_workers.setObjectName("sb_g2_bayesian_workers")
        self.sb_g2_bayesian_workers.setMinimum(1)
        self.sb_g2_bayesian_workers.setMaximum(8)
        self.sb_g2_bayesian_workers.setValue(4)
        self.sb_g2_bayesian_workers.setToolTip("Max concurrent Bayesian workers")
        self.gridLayout_12.addWidget(self.sb_g2_bayesian_workers, 2, 12, 1, 1)

        # Sampler configuration spinboxes
        self.sb_g2_bayesian_warmup = QSpinBox(self.groupBox_2)
        self.sb_g2_bayesian_warmup.setObjectName("sb_g2_bayesian_warmup")
        self.sb_g2_bayesian_warmup.setMinimum(100)
        self.sb_g2_bayesian_warmup.setMaximum(5000)
        self.sb_g2_bayesian_warmup.setValue(500)
        self.sb_g2_bayesian_warmup.setToolTip("NUTS warmup steps")

        self.sb_g2_bayesian_samples = QSpinBox(self.groupBox_2)
        self.sb_g2_bayesian_samples.setObjectName("sb_g2_bayesian_samples")
        self.sb_g2_bayesian_samples.setMinimum(200)
        self.sb_g2_bayesian_samples.setMaximum(10000)
        self.sb_g2_bayesian_samples.setValue(1000)
        self.sb_g2_bayesian_samples.setToolTip("NUTS sample count")

        self.sb_g2_bayesian_chains = QSpinBox(self.groupBox_2)
        self.sb_g2_bayesian_chains.setObjectName("sb_g2_bayesian_chains")
        self.sb_g2_bayesian_chains.setMinimum(1)
        self.sb_g2_bayesian_chains.setMaximum(8)
        self.sb_g2_bayesian_chains.setValue(4)
        self.sb_g2_bayesian_chains.setToolTip("NUTS chain count")

        self.btn_g2_bayesian_all.clicked.connect(self._fit_g2_bayesian_all)

        self.btn_g2_plot_all_q = QPushButton("Plot All Q", self.groupBox_2)
        self.btn_g2_plot_all_q.setObjectName("btn_g2_plot_all_q")
        self.btn_g2_plot_all_q.setEnabled(False)  # Enabled after batch completes
        self.btn_g2_plot_all_q.setVisible(False)  # Hidden from GUI; kept for scripting
        self.btn_g2_plot_all_q.clicked.connect(self._plot_g2_bayesian_all)

        # Disable Bayesian buttons if NumPyro is not available
        try:
            from .fitting.models import NUMPYRO_AVAILABLE

            if not NUMPYRO_AVAILABLE:
                for btn in (
                    self.btn_g2_bayesian,
                    self.btn_g2_diagnosis,
                    self.btn_diff_bayesian,
                    self.btn_diff_diagnosis,
                    self.btn_g2_bayesian_all,
                    self.btn_g2_plot_all_q,
                ):
                    btn.setEnabled(False)
                    btn.setToolTip("NumPyro not installed")
                self.sb_g2_bayesian_workers.setEnabled(False)
                self.sb_g2_bayesian_warmup.setEnabled(False)
                self.sb_g2_bayesian_samples.setEnabled(False)
                self.sb_g2_bayesian_chains.setEnabled(False)
        except ImportError:
            pass

        self.comboBox_qmap_target.currentIndexChanged.connect(self.update_plot)
        self.cb_qmap_cmap.currentIndexChanged.connect(self.update_plot)
        self.update_g2_fitting_function()

        self.pg_saxs.getView().scene().sigMouseMoved.connect(self.saxs2d_mouseMoved)

        # Connect progress manager to show/hide progress dialog
        # Add keyboard shortcut to show progress dialog
        self.setup_async_connections()
        self.setup_progress_shortcut()

        # Apply layout improvements AFTER all dynamic widgets are created
        # (e.g. btn_g2_bayesian_all at line ~322 must exist before rearrangement)
        self._apply_layout_improvements()

        self.load_default_setting()

        # Restore session state (window geometry, files, tab)
        self._restore_session()

        # Prefer maximized startup on real displays; keep deterministic size offscreen
        platform = os.environ.get("QT_QPA_PLATFORM", "").lower()
        if platform not in {"offscreen", "minimal"}:
            self.showMaximized()
        elif getattr(self, "_default_size", None):
            self.resize(*self._default_size)
            # offscreen needs an explicit show after resize
        self.show()

    def load_default_setting(self):
        if not os.path.isdir(self.home_dir):
            os.mkdir(self.home_dir)

        key_fname = os.path.join(self.home_dir, "default_setting.json")
        # copy the default values
        if not os.path.isfile(key_fname):
            from .default_setting import setting

            with open(key_fname, "w") as f:
                json.dump(setting, f, indent=4)

        # the display size might too big for some laptops
        with open(key_fname) as f:
            config = json.load(f)
            if "window_size_h" in config:
                new_size = (config["window_size_w"], config["window_size_h"])
                self._default_size = new_size
                # enforce a reasonable floor so controls are not squeezed pre-maximize
                self.setMinimumSize(*new_size)
            else:
                # fallback default if config missing height/width
                self._default_size = (1024, 800)
                self.setMinimumSize(*self._default_size)

        cache_dir = os.path.join(
            os.path.expanduser("~"), ".xpcsviewer", "joblib/xpcsviewer"
        )
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)

    def setup_async_connections(self):
        """Set up connections for async operations."""
        # Connect progress manager signals for operation cancellation
        self.progress_manager.operation_cancelled.connect(self.cancel_async_operation)

    # ---- UI polish helpers -------------------------------------------------

    def _init_spacing(self):
        layout = self.centralwidget.layout()
        if layout:
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(8)

    def _apply_button_styles(self):
        """Apply semantic button styling for better visual hierarchy.

        Uses QSS property selectors to differentiate button types:
        - Primary (default): Main action buttons like Plot, Fit
        - Secondary: Navigation buttons like Move up/down
        - Destructive: Remove/delete actions
        """
        from xpcsviewer.gui.theme.style_helpers import (
            apply_destructive_buttons,
            apply_secondary_buttons,
        )

        # Destructive buttons (remove/delete actions)
        destructive_buttons = [
            "pushButton_3",  # Remove from target
            "avg_job_pop",  # Remove average job
            "btn_avg_kill",  # Kill average job
        ]
        apply_destructive_buttons(self, destructive_buttons)

        # Secondary buttons (navigation, utility actions)
        secondary_buttons = [
            "btn_up",  # Move up
            "btn_down",  # Move down
            "btn_deselect",  # Clear selection
            "pushButton_11",  # Reload files
        ]
        apply_secondary_buttons(self, secondary_buttons)

    def _apply_layout_improvements(self):
        """Apply layout improvements for better visual hierarchy.

        Enhances spacing, margins, and visual organization of
        panels, control groups, and action buttons.
        """
        apply_all_layout_improvements(self)

    def _init_toolbar(self):
        """Initialize the main toolbar with custom SVG icons."""
        from xpcsviewer.gui.icons import get_icon, set_icon_color
        from xpcsviewer.gui.qt_compat import QSize, Qt, QToolBar, QToolButton

        # Set initial icon color from current theme
        text_color = self.theme_manager.get_color("text_primary")
        set_icon_color(text_color)

        toolbar = QToolBar("Main Toolbar", self)
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        toolbar.setIconSize(QSize(20, 20))
        self.addToolBar(toolbar)

        style = self.style()

        # Open folder action
        action_open = toolbar.addAction(get_icon("folder-open", style), "Open Folder")
        action_open.setToolTip("Open data folder (Ctrl+O)")
        action_open.triggered.connect(lambda: self.load_path(None))

        # Reload action
        action_reload = toolbar.addAction(get_icon("refresh", style), "Reload")
        action_reload.setToolTip("Reload files from current folder (Ctrl+R)")
        action_reload.triggered.connect(self.reload_source)

        toolbar.addSeparator()

        # Plot current tab action
        action_plot = toolbar.addAction(get_icon("play-circle", style), "Plot")
        action_plot.setToolTip("Plot current tab")
        action_plot.triggered.connect(self._plot_current_tab)

        toolbar.addSeparator()

        # Theme toggle button (icon swaps based on current theme)
        self.theme_toggle_btn = QToolButton()
        self.theme_toggle_btn.setToolTip("Toggle dark/light theme")
        self.theme_toggle_btn.clicked.connect(self._toggle_theme)
        toolbar.addWidget(self.theme_toggle_btn)
        self._update_theme_toggle_button()

        # Progress indicator
        action_progress = toolbar.addAction(get_icon("activity", style), "Progress")
        action_progress.setToolTip("Show progress dialog (Ctrl+Shift+P)")
        action_progress.triggered.connect(self.show_progress_dialog)

        # Store toolbar actions for icon refresh on theme change
        self._toolbar_actions = {
            "folder-open": action_open,
            "refresh": action_reload,
            "play-circle": action_plot,
            "activity": action_progress,
        }

        toolbar.addSeparator()

        # Mask Editor button
        action_mask_editor = toolbar.addAction(
            get_icon("grid-mask", style), "Mask Editor"
        )
        action_mask_editor.setToolTip("Open Mask Editor window")
        action_mask_editor.triggered.connect(self.open_simplemask)

    def _plot_current_tab(self):
        """Trigger plot for the currently active tab."""
        tab_index = self.tabWidget.currentIndex()
        tab_name = tab_mapping.get(tab_index, "")

        plot_methods = {
            "saxs_2d": self.update_plot,
            "saxs_1d": self.update_plot,
            "stability": self.update_plot,
            "intensity_t": self.update_plot,
            "g2": self.update_plot,
            "diffusion": self.update_plot,
            "twotime": self.update_plot,
            "qmap": self.update_plot,
            "average": self.update_plot,
            "metadata": self.update_plot,
        }

        if tab_name in plot_methods:
            plot_methods[tab_name]()

    def _toggle_theme(self):
        """Toggle between light and dark theme."""
        current = self.theme_manager.get_current_theme()
        new_theme = "dark" if current == "light" else "light"
        self._set_theme(new_theme)

    def _update_theme_toggle_button(self):
        """Update theme toggle button icon for the current theme."""
        if hasattr(self, "theme_toggle_btn"):
            from xpcsviewer.gui.icons import get_icon

            style = self.style()
            current = self.theme_manager.get_current_theme()
            if current == "dark":
                # Dark mode active: show sun icon (switch to light)
                self.theme_toggle_btn.setIcon(get_icon("sun", style))
                self.theme_toggle_btn.setToolTip("Switch to light theme")
            else:
                # Light mode active: show moon icon (switch to dark)
                self.theme_toggle_btn.setIcon(get_icon("moon", style))
                self.theme_toggle_btn.setToolTip("Switch to dark theme")

    def _init_menus_and_toolbar(self):
        # File menu
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")

        self.action_open = QAction("&Open…", self)
        self.action_open.setShortcut(QKeySequence("Ctrl+O"))
        self.action_open.triggered.connect(lambda: self.load_path(None))
        file_menu.addAction(self.action_open)

        sample_dir = self._sample_data_path()
        if sample_dir:
            action_sample = QAction("Open &Sample Data", self)
            action_sample.triggered.connect(lambda: self.load_path(str(sample_dir)))
            file_menu.addAction(action_sample)

        action_reload = QAction("&Reload", self)
        action_reload.setShortcut(QKeySequence("Ctrl+R"))
        action_reload.triggered.connect(self.reload_source)
        file_menu.addAction(action_reload)

        action_regroup = QAction("Regroup G2…", self)
        action_regroup.triggered.connect(self._open_g2_regroup)
        file_menu.addAction(action_regroup)

        # Recent directories submenu
        self.recent_menu = file_menu.addMenu("Recent &Directories")
        self._populate_recent_directories_menu()

        file_menu.addSeparator()
        action_quit = QAction("E&xit", self)
        action_quit.triggered.connect(QtWidgets.QApplication.instance().quit)
        file_menu.addAction(action_quit)

        # View/Help menu
        view_menu = menubar.addMenu("&View")

        # Theme submenu
        theme_menu = view_menu.addMenu("&Theme")
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)

        self.action_theme_light = QAction("&Light Mode", self, checkable=True)
        self.action_theme_light.triggered.connect(lambda: self._set_theme("light"))
        theme_group.addAction(self.action_theme_light)
        theme_menu.addAction(self.action_theme_light)

        self.action_theme_dark = QAction("&Dark Mode", self, checkable=True)
        self.action_theme_dark.triggered.connect(lambda: self._set_theme("dark"))
        theme_group.addAction(self.action_theme_dark)
        theme_menu.addAction(self.action_theme_dark)

        theme_menu.addSeparator()

        self.action_theme_system = QAction("&Follow System", self, checkable=True)
        self.action_theme_system.triggered.connect(lambda: self._set_theme("system"))
        theme_group.addAction(self.action_theme_system)
        theme_menu.addAction(self.action_theme_system)

        # Set initial check state based on current mode
        self._update_theme_menu_state()

        view_menu.addSeparator()

        action_progress = QAction("Show &Progress", self)
        action_progress.setShortcut(QKeySequence("Ctrl+Shift+P"))
        action_progress.triggered.connect(self.show_progress_dialog)
        view_menu.addAction(action_progress)

        help_menu = menubar.addMenu("&Help")
        action_logs = QAction("View &Logs", self)
        action_logs.setShortcut(QKeySequence("Ctrl+L"))
        action_logs.triggered.connect(self._open_logs_dir)
        help_menu.addAction(action_logs)

        # Create toolbar
        self._init_toolbar()

    def _register_shortcuts(self):
        # Add to target
        QShortcut(QKeySequence("Ctrl+Shift+A"), self, activated=self.add_target)
        # Clear target selection
        QShortcut(QKeySequence("Esc"), self, activated=self.clear_target_selection)
        # Show docs (opens README)
        QShortcut(QKeySequence("F1"), self, activated=self._open_docs)

        # Command palette (Ctrl+P)
        self.shortcut_manager.register_shortcut(
            "command_palette", "Ctrl+P", self._show_command_palette
        )

        # Tab navigation shortcuts
        self.shortcut_manager.register_shortcut(
            "next_tab", "Ctrl+PgDown", self._next_tab
        )
        self.shortcut_manager.register_shortcut("prev_tab", "Ctrl+PgUp", self._prev_tab)

        # Register actions in command palette
        self._register_command_palette_actions()

    def _show_command_palette(self) -> None:
        """Show the command palette dialog."""
        self.command_palette.show()

    def _next_tab(self) -> None:
        """Switch to next tab."""
        current = self.tabWidget.currentIndex()
        count = self.tabWidget.count()
        next_idx = (current + 1) % count
        logger.debug(
            f"Tab navigation: next ({tab_mapping.get(current, '')} -> {tab_mapping.get(next_idx, '')})"
        )
        self.tabWidget.setCurrentIndex(next_idx)

    def _prev_tab(self) -> None:
        """Switch to previous tab."""
        current = self.tabWidget.currentIndex()
        count = self.tabWidget.count()
        prev_idx = (current - 1) % count
        logger.debug(
            f"Tab navigation: prev ({tab_mapping.get(current, '')} -> {tab_mapping.get(prev_idx, '')})"
        )
        self.tabWidget.setCurrentIndex(prev_idx)

    def _register_command_palette_actions(self) -> None:
        """Register actions in the command palette."""
        # File actions
        self.command_palette.register_action(
            "file.open",
            "Open Folder",
            "File",
            lambda: self.load_path(None),
            shortcut="Ctrl+O",
        )
        self.command_palette.register_action(
            "file.reload", "Reload", "File", self.reload_source, shortcut="Ctrl+R"
        )

        # View actions
        self.command_palette.register_action(
            "view.light_theme", "Light Theme", "View", lambda: self._set_theme("light")
        )
        self.command_palette.register_action(
            "view.dark_theme", "Dark Theme", "View", lambda: self._set_theme("dark")
        )
        self.command_palette.register_action(
            "view.system_theme",
            "System Theme",
            "View",
            lambda: self._set_theme("system"),
        )

        # Tab navigation actions
        for idx, name in tab_mapping.items():
            display_name = name.replace("_", " ").title()
            self.command_palette.register_action(
                f"tab.{name}",
                f"Go to {display_name}",
                "Navigate",
                lambda i=idx: self.tabWidget.setCurrentIndex(i),
            )

    def _populate_recent_directories_menu(self) -> None:
        """Populate the Recent Directories submenu."""
        self.recent_menu.clear()

        recent_paths = self.recent_paths_manager.get_recent_paths()

        if not recent_paths:
            action = self.recent_menu.addAction("(No recent directories)")
            action.setEnabled(False)
            return

        for recent in recent_paths:
            path = recent.path
            # Use basename for display, full path in status tip
            display = os.path.basename(path) or path
            action = self.recent_menu.addAction(display)
            action.setStatusTip(path)
            action.triggered.connect(
                lambda checked=False, p=path: self._open_recent_directory(p)
            )

        self.recent_menu.addSeparator()
        clear_action = self.recent_menu.addAction("Clear Recent")
        clear_action.triggered.connect(self._clear_recent_directories)

    def _open_recent_directory(self, path: str) -> None:
        """Open a directory from the recent list.

        Args:
            path: Directory path to open
        """
        if os.path.isdir(path):
            self.load_path(path)
        else:
            # Path no longer exists
            self.toast_manager.show_warning(f"Directory not found: {path}")
            self.recent_paths_manager.remove_invalid_path(path)
            self._populate_recent_directories_menu()

    def _clear_recent_directories(self) -> None:
        """Clear all recent directories."""
        self.recent_paths_manager.clear()
        self._populate_recent_directories_menu()
        self.toast_manager.show_info("Recent directories cleared")

    def _sample_data_path(self):
        candidates = [
            Path("tests/fixtures/reference_data"),
            Path("tests/fixtures"),
        ]
        for c in candidates:
            if c.exists() and c.is_dir():
                return c.resolve()
        return None

    def _open_logs_dir(self):
        log_dir = Path.home() / ".xpcsviewer" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(log_dir)))

    def _set_theme(self, mode: str) -> None:
        """Set the application theme.

        Args:
            mode: One of "light", "dark", or "system"
        """
        self.theme_manager.set_theme(mode)
        self._update_theme_menu_state()
        logger.info(f"Theme changed to {mode}")

    def _update_theme_menu_state(self) -> None:
        """Update theme menu check states based on current mode."""
        current_mode = self.theme_manager.get_current_mode()
        self.action_theme_light.setChecked(current_mode == "light")
        self.action_theme_dark.setChecked(current_mode == "dark")
        self.action_theme_system.setChecked(current_mode == "system")

    def _on_theme_changed(self, theme: str) -> None:
        """Handle theme change signal.

        Args:
            theme: The new theme name ("light" or "dark")
        """
        logger.debug(f"Theme changed to {theme}, refreshing plots")
        # Recolor SVG icons for the new theme
        from xpcsviewer.gui.icons import clear_icon_cache, get_icon, set_icon_color

        text_color = self.theme_manager.get_color("text_primary")
        set_icon_color(text_color)
        clear_icon_cache()
        # Refresh toolbar action icons with new theme color
        if hasattr(self, "_toolbar_actions"):
            style = self.style()
            for icon_name, action in self._toolbar_actions.items():
                action.setIcon(get_icon(icon_name, style))
        # Update tab bar separator color for the new theme
        if hasattr(self, "_category_tab_bar"):
            self._category_tab_bar.set_dark_mode(theme == "dark")
        # Recolor per-tab category icons (else they stay the old theme's color
        # and become invisible against the inverted background).
        self._refresh_tab_icons()
        # Update theme toggle button icon (uses freshly cleared cache)
        self._update_theme_toggle_button()
        self._refresh_all_plots(theme)

    def _refresh_all_plots(self, theme: str) -> None:
        """Refresh all plot widgets with new theme colors.

        This method updates backgrounds and colors for all plot widgets
        (both PyQtGraph and Matplotlib) when the theme changes.

        Args:
            theme: The new theme name ("light" or "dark")
        """
        # Apply theme to PyQtGraph global config
        self.theme_manager.apply_to_pyqtgraph()

        # Update individual plot widgets that have apply_theme method
        plot_widgets = [
            # PyQtGraph widgets
            getattr(self, "pg_saxs", None),  # SAXS 2D ImageView
            getattr(self, "mp_saxs", None),  # SAXS 1D PlotWidget
            getattr(self, "mp_stab", None),  # Stability PlotWidget
            getattr(self, "pg_intt", None),  # Intensity PlotWidgetDev
            getattr(self, "mp_g2", None),  # G2 PlotWidgetDev
            getattr(self, "mp_2t", None),  # Two-time ImageView
            getattr(self, "pg_qmap", None),  # Q-map ImageView
            getattr(self, "mp_avg_g2", None),  # Average G2 PlotWidgetDev
            # Matplotlib widgets
            getattr(self, "mp_tauq", None),  # Diffusion MplCanvasBarV
            getattr(self, "mp_tauq_pre", None),  # Diffusion preview MplCanvasBarV
        ]

        for widget in plot_widgets:
            if widget is not None and hasattr(widget, "apply_theme"):
                try:
                    widget.apply_theme(theme)
                except Exception as e:
                    logger.warning(
                        f"Failed to apply theme to {type(widget).__name__}: {e}"
                    )

        logger.debug(f"Refreshed all plots with {theme} theme")

    def _setup_drag_drop_list(self) -> None:
        """Replace the target list view with a drag-drop enabled version."""
        # Get the parent (box_target) and its grid layout
        parent = self.list_view_target.parent()

        # Create new drag-drop list view with same parent
        new_list_view = DragDropListView(parent)
        new_list_view.setObjectName("list_view_target")

        # Copy selection mode from old list view
        new_list_view.setSelectionMode(self.list_view_target.selectionMode())

        # Get the grid layout (gridLayout_4)
        layout = self.gridLayout_4

        # Remove old widget from layout
        layout.removeWidget(self.list_view_target)
        self.list_view_target.deleteLater()

        # Add new widget at same grid position (row 0, col 0)
        layout.addWidget(new_list_view, 0, 0, 1, 1)

        # Replace reference
        self.list_view_target = new_list_view

        # Connect the items_reordered signal
        self.list_view_target.items_reordered.connect(self._on_target_list_reordered)

        logger.debug("Replaced target list with DragDropListView")

    def _on_target_list_reordered(self, from_index: int, to_index: int) -> None:
        """Handle drag-drop reordering in the target list.

        Args:
            from_index: Original position of the item
            to_index: New position of the item
        """
        logger.info(f"Target list reordered: {from_index} -> {to_index}")
        if self.vk is None:
            return
        # The target model is a ListDataModel (== self.vk.target), which has no
        # Qt drag-drop row-move support, so the underlying list is NOT moved for
        # us — do it here via the list-like API so the backend order matches the
        # view. ponytail: the up/down buttons (reorder_target) are the reliable
        # reorder path; full Qt InternalMove would need flags()/moveRows() on
        # ListDataModel, which is out of scope for this fix.
        model = self.list_view_target.model()
        if model is None or not (hasattr(model, "pop") and hasattr(model, "insert")):
            return
        size = len(model)
        if from_index == to_index or not (0 <= from_index < size) or to_index < 0:
            return
        item = model.pop(from_index)
        model.insert(min(to_index, len(model)), item)
        self.update_plot()

    # ---- Session management -------------------------------------------------

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Override close event to save session state.

        Args:
            event: The close event
        """
        self._close_g2_regroup()

        # Close SimpleMask window if open
        if hasattr(self, "_simplemask_window") and self._simplemask_window is not None:
            self._simplemask_window.close()

        # Close Bayesian diagnosis windows if open
        if (
            hasattr(self, "_g2_diagnosis_window")
            and self._g2_diagnosis_window is not None
        ):
            self._g2_diagnosis_window.close()
        if (
            hasattr(self, "_diff_diagnosis_window")
            and self._diff_diagnosis_window is not None
        ):
            self._diff_diagnosis_window.close()

        # Shutdown async kernel if available
        if hasattr(self, "async_vk") and self.async_vk is not None:
            try:
                self.async_vk.cancel_all_operations()
            except Exception:
                logger.debug("async_vk shutdown failed during close", exc_info=True)

        # Shutdown UnifiedThreadingManager BEFORE waiting for QThreadPool workers.
        # Its _monitor_system() thread runs outside the pool and can fire
        # QMetaObject.invokeMethod calls after Qt objects begin teardown if we
        # drain the pool first (P1-14).
        try:
            from xpcsviewer.threading.unified_threading import (
                shutdown_unified_threading,
            )

            shutdown_unified_threading()
        except Exception:
            logger.debug(
                "UnifiedThreadingManager shutdown failed during close", exc_info=True
            )

        # Wait for all thread-pool workers to finish before allowing the GUI
        # objects to be destroyed.  Without this, signals fired from
        # background threads can reach already-deleted C++ QObjects and
        # cause a segfault.  Use a generous but bounded timeout so the
        # application never hangs indefinitely on shutdown. (BUG-013)
        _SHUTDOWN_TIMEOUT_MS = 5000
        if hasattr(self, "thread_pool") and self.thread_pool is not None:
            self.thread_pool.waitForDone(_SHUTDOWN_TIMEOUT_MS)

        try:
            session = self._collect_session_state()
            self.session_manager.save_session(session)
            logger.debug("Session saved on close")
        except Exception as e:
            logger.warning(f"Failed to save session on close: {e}")

        # Shutdown singleton daemon managers so background threads stop
        # promptly instead of lingering until atexit.  (SRE-4)
        for _shutdown_path in (
            "xpcsviewer.threading.unified_threading.shutdown_unified_threading",
            "xpcsviewer.utils.memory_manager.shutdown_memory_manager",
            "xpcsviewer.utils.lazy_loader.shutdown_lazy_loader",
        ):
            try:
                _mod_path, _fn_name = _shutdown_path.rsplit(".", 1)
                import importlib

                _mod = importlib.import_module(_mod_path)
                getattr(_mod, _fn_name)()
            except Exception:
                logger.debug(
                    "Shutdown call %s failed (expected during teardown)",
                    _shutdown_path,
                    exc_info=True,
                )

        # Stop health monitor if it was started
        try:
            from xpcsviewer.utils.health_monitor import get_health_monitor

            get_health_monitor().stop_monitoring()
        except Exception:
            logger.debug(
                "Health monitor stop failed (expected during teardown)",
                exc_info=True,
            )

        # Stop background state validation if active
        try:
            from xpcsviewer.utils.state_validator import get_state_validator

            get_state_validator().stop_background_validation()
        except Exception:
            logger.debug(
                "State validator stop failed (expected during teardown)",
                exc_info=True,
            )

        # Call parent implementation
        super().closeEvent(event)

    def _collect_session_state(self) -> SessionState:
        """Collect current workspace state for session persistence.

        Returns:
            SessionState with current UI and file state
        """
        # Collect target files. The target model is a ListDataModel (list-like
        # API: len()/[]); it has no QStandardItemModel-style item()/text().
        # Its entries are filenames relative to vk.path, so persist the
        # *absolute* path to honor FileEntry.path's contract and survive a
        # restart whose working directory differs from the data directory.
        target_files = []
        # Prefer the installed GUI model, but fall back to the kernel target
        # list so a restored-but-not-yet-reattached model is never silently
        # dropped on save.
        target_list = self.target_model
        if target_list is None and self.vk is not None:
            target_list = self.vk.target
        if target_list is not None:
            base = self.vk.path if self.vk is not None else ""
            for row in range(len(target_list)):
                name = target_list[row]
                if name:
                    abs_path = os.path.abspath(os.path.join(base, str(name)))
                    target_files.append(FileEntry(path=abs_path, order=row))

        # Collect window geometry
        geom = self.geometry()
        window_geometry = WindowGeometry(
            x=geom.x(),
            y=geom.y(),
            width=geom.width(),
            height=geom.height(),
            maximized=self.isMaximized(),
        )

        # Helper to safely get checkbox state
        def get_checkbox_state(attr_name: str, default: bool = True) -> bool:
            widget = getattr(self, attr_name, None)
            if widget is not None and hasattr(widget, "isChecked"):
                return widget.isChecked()
            return default

        # Helper to safely get combobox text
        def get_combobox_text(attr_name: str, default: str = "") -> str:
            widget = getattr(self, attr_name, None)
            if widget is not None and hasattr(widget, "currentText"):
                return widget.currentText()
            return default

        # Helper to safely get combobox index
        def get_combobox_index(attr_name: str, default: int = 0) -> int:
            widget = getattr(self, attr_name, None)
            if widget is not None and hasattr(widget, "currentIndex"):
                return widget.currentIndex()
            return default

        # Map stability combobox index to metric name
        _stab_metric_map = {0: "mean", 1: "sigma", 2: "contrast"}
        stab_idx = get_combobox_index("cb_stab_type", 0)

        # Collect analysis parameters from UI widgets — extract VALUES, not widgets.
        # Fields without widget mappings (intensity_t_normalize, diffusion_model,
        # twotime_symmetric, qmap_show_rings) keep their dataclass defaults.
        analysis_params = AnalysisParameters(
            saxs2d_colormap=get_combobox_text("cb_saxs2D_cmap", "viridis"),
            saxs2d_auto_level=get_checkbox_state("saxs2d_autolevel", True),
            saxs2d_log_scale=get_checkbox_state("saxs2d_log_scale", False),
            saxs1d_log_x=get_checkbox_state("saxs1d_log_x", False),
            saxs1d_log_y=get_checkbox_state("saxs1d_log_y", True),
            stability_metric=_stab_metric_map.get(stab_idx, "mean"),
            g2_fit_function=get_combobox_text("g2_fitting_function", "single_exp"),
            g2_q_index=get_combobox_index("g2_q_selection", 0),
            g2_show_fit=get_checkbox_state("g2_show_fit", True),
            twotime_selected_q=get_combobox_index("comboBox_twotime_selection", 0),
            twotime_colormap=get_combobox_text("cb_twotime_cmap", "viridis"),
            qmap_colormap=get_combobox_text("cb_qmap_cmap", "viridis"),
        )

        return SessionState(
            data_path=(
                os.path.abspath(self.work_dir.text())
                if hasattr(self, "work_dir") and self.work_dir.text()
                else None
            ),
            target_files=target_files,
            active_tab=self.tabWidget.currentIndex(),
            window_geometry=window_geometry,
            analysis_params=analysis_params,
        )

    def _restore_session(self) -> None:
        """Restore workspace state from saved session."""
        session = self.session_manager.load_session()
        if session is None:
            logger.debug("No session to restore")
            return

        # Show warnings for missing files as toast notifications
        warnings = self.session_manager.get_warnings()
        for warning in warnings:
            logger.warning(warning)
            self.toast_manager.show_warning(warning)

        # Restore window geometry (but don't override maximized state)
        if not session.window_geometry.maximized:
            self.setGeometry(
                session.window_geometry.x,
                session.window_geometry.y,
                session.window_geometry.width,
                session.window_geometry.height,
            )

        # An explicitly requested startup directory wins over the session's
        # workspace; the session's target files belong to that workspace only.
        restore_workspace = not (
            getattr(self, "_path_explicit", False)
            and session.data_path
            and os.path.abspath(session.data_path)
            != os.path.abspath(self.work_dir.text() or ".")
        )

        # Restore data path if it still exists
        if restore_workspace and session.data_path and os.path.isdir(session.data_path):
            self.load_path(session.data_path)

        # Restore target files from session.
        # Guard against self.vk being None when data_path was absent or
        # load_path() was skipped (e.g. directory no longer exists).
        if (
            restore_workspace
            and self.vk is not None
            and hasattr(session, "target_files")
            and session.target_files
        ):
            sorted_files = sorted(session.target_files, key=lambda f: f.order)
            restored_any = False
            for entry in sorted_files:
                # FileEntry.path is absolute (current sessions); tolerate
                # legacy relative entries by resolving against the data dir.
                abs_path = (
                    entry.path
                    if os.path.isabs(entry.path)
                    else os.path.join(self.vk.path, entry.path)
                )
                if os.path.isfile(abs_path):
                    # add_target expects names relative to vk.path.
                    self.vk.add_target([os.path.relpath(abs_path, self.vk.path)])
                    restored_any = True
            # Attach the restored targets to the GUI target model so they are
            # both visible in the target list AND persisted on the next save;
            # otherwise self.target_model stays None and _collect_session_state
            # would drop the restored targets on the next close.
            if restored_any:
                self.update_box(self.vk.target, mode="target")

        # Restore active tab (block signals to prevent spurious plot updates)
        if 0 <= session.active_tab < self.tabWidget.count():
            self.tabWidget.blockSignals(True)
            self.tabWidget.setCurrentIndex(session.active_tab)
            self.tabWidget.blockSignals(False)

        # Restore analysis parameters
        params = session.analysis_params
        if hasattr(self, "saxs2d_autolevel"):
            self.saxs2d_autolevel.setChecked(params.saxs2d_auto_level)
        if hasattr(self, "g2_fitting_function"):
            idx = self.g2_fitting_function.findText(params.g2_fit_function)
            if idx >= 0:
                self.g2_fitting_function.setCurrentIndex(idx)

        logger.info("Session restored successfully")

    def _open_docs(self):
        docs_path = Path(__file__).resolve().parent.parent / "docs" / "README.rst"
        if docs_path.exists():
            QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(docs_path)))

    def _maybe_prompt_start_path(self):
        """Show startup dialog with recent directories and options."""
        # Skip in headless/offscreen runs
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            return

        # Skip if a path was provided programmatically
        if self.work_dir.text():
            return

        # Get recent paths
        recent_paths = self.recent_paths_manager.get_recent_paths()
        sample_dir = self._sample_data_path()

        # Create custom startup dialog
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Welcome to XPCS Viewer")
        dialog.setMinimumWidth(450)
        dialog.setModal(True)

        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Header
        header = QtWidgets.QLabel("Choose a data folder to start")
        header.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(header)

        # Recent directories section
        if recent_paths:
            recent_label = QtWidgets.QLabel("Recent Directories:")
            recent_label.setStyleSheet("font-size: 13px; color: #666; margin-top: 8px;")
            layout.addWidget(recent_label)

            recent_list = QtWidgets.QListWidget()
            recent_list.setMaximumHeight(150)
            recent_list.setAlternatingRowColors(True)

            for recent in recent_paths[:5]:  # Show max 5 recent
                item = QtWidgets.QListWidgetItem(recent.path)
                item.setToolTip(recent.path)
                recent_list.addItem(item)

            layout.addWidget(recent_list)

            # Double-click to open
            def open_recent_item(item):
                dialog.accept()
                self.load_path(item.text())

            recent_list.itemDoubleClicked.connect(open_recent_item)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setSpacing(8)

        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.setDefault(True)
        browse_btn.clicked.connect(lambda: (dialog.accept(), self.load_path(None)))
        button_layout.addWidget(browse_btn)

        if sample_dir:
            sample_btn = QtWidgets.QPushButton("Open Sample Data")
            sample_btn.clicked.connect(
                lambda: (dialog.accept(), self.load_path(str(sample_dir)))
            )
            button_layout.addWidget(sample_btn)

        button_layout.addStretch()

        skip_btn = QtWidgets.QPushButton("Skip")
        skip_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(skip_btn)

        layout.addLayout(button_layout)

        # If recent paths exist, add "Open Selected" button
        if recent_paths:

            def open_selected():
                items = recent_list.selectedItems()
                if items:
                    dialog.accept()
                    self.load_path(items[0].text())

            open_selected_btn = QtWidgets.QPushButton("Open Selected")
            open_selected_btn.setEnabled(False)
            recent_list.itemSelectionChanged.connect(
                lambda: open_selected_btn.setEnabled(
                    len(recent_list.selectedItems()) > 0
                )
            )
            open_selected_btn.clicked.connect(open_selected)
            button_layout.insertWidget(1, open_selected_btn)

        dialog.exec()

    def setup_progress_shortcut(self):
        """Set up keyboard shortcut to show progress dialog."""
        from xpcsviewer.gui.qt_compat import QKeySequence, QShortcut

        # Ctrl+Shift+P to show progress dialog (Ctrl+P is command palette)
        shortcut = QShortcut(QKeySequence("Ctrl+Shift+P"), self)
        shortcut.activated.connect(self.show_progress_dialog)

    def show_progress_dialog(self):
        """Show the progress dialog."""
        self.progress_manager.show_progress_dialog()

    def init_async_kernel(self):
        """Initialize async kernel when viewer kernel is ready."""
        if self.vk and not self.async_vk:
            self.async_vk = AsyncViewerKernel(self.vk, self.thread_pool)
            self.data_preloader = AsyncDataPreloader(self.async_vk)

            # Connect async signals
            self.async_vk.plot_ready.connect(self.on_async_plot_ready)
            self.async_vk.operation_error.connect(self.on_async_operation_error)
            self.async_vk.operation_cancelled.connect(self.on_async_operation_cancelled)
            self.async_vk.operation_progress.connect(
                self.progress_manager.update_progress
            )

            logger.info("Async kernel initialized")

    def cancel_async_operation(self, operation_id: str):
        """Cancel an async operation."""
        if self.async_vk:
            self.async_vk.cancel_operation(operation_id)

        # An explicit retirement (selection changed, workspace reset) is
        # followed by the caller's own redraw; a user cancel is not.
        self._settle_plot_operation(
            operation_id, failed=True, reschedule=not self._retiring_plots
        )

    def _settle_plot_operation(
        self, operation_id: str, *, failed: bool, reschedule: bool = True
    ) -> str | None:
        """End bookkeeping for a plot operation; whether it finished, failed
        or was cancelled, the newest request for its tab must still be drawn.

        pop() avoids KeyError if another path already removed the entry
        (P1-02).  Returns the operation's tab name, or None if unknown.
        """
        plot_type = self.active_plot_operations.pop(operation_id, None)
        request = self._plot_op_requests.pop(operation_id, None)
        if plot_type is None:
            return None
        pending = plot_type in self._plot_pending
        self._plot_pending.discard(plot_type)
        if failed and request is not None and not self._retiring_plots:
            # A failed/cancelled request drew nothing: let the same request
            # be issued again instead of being skipped as "unchanged".  (A
            # deliberate retirement is followed by the caller's own redraw.)
            if self.plot_kwargs_record.get(plot_type) == request:
                self.plot_kwargs_record[plot_type] = {}
        # After a success the staleness was already checked before drawing, and
        # drawing itself may change the kwargs (e.g. auto bounds): only a
        # request suppressed meanwhile (`pending`) is replayed then.
        if reschedule and (
            pending
            or (
                failed
                and request is not None
                and self._plot_request_is_stale(plot_type, request)
            )
        ):
            self.update_plot_async(plot_type)
        return plot_type

    def _plot_request_is_stale(self, plot_type, request) -> bool:
        """True if the current selection no longer yields *request*'s kwargs."""
        try:
            current = getattr(self, "plot_" + plot_type)(dryrun=True)
            if not current:
                return True
            current["target_timestamp"] = self.vk.timestamp
            return bool(self._plot_request(current) != request)
        except Exception:
            logger.debug("stale-plot check failed for %s", plot_type, exc_info=True)
            return False

    def on_async_plot_ready(self, operation_id: str, result):
        """Handle async plot completion."""
        if operation_id not in self.active_plot_operations:
            return

        plot_type = self.active_plot_operations[operation_id]

        # Selection/targets changed while the worker ran: drop the result and
        # queue a refresh for the newest selection instead of painting old data.
        request = self._plot_op_requests.get(operation_id)
        if request is not None and self._plot_request_is_stale(plot_type, request):
            logger.info("Discarding stale %s plot result", plot_type)
            self.progress_manager.complete_operation(operation_id, True)
            self._settle_plot_operation(operation_id, failed=True)
            return

        try:
            # Apply the plot result to the GUI
            if plot_type == "saxs_2d":
                self.apply_saxs_2d_result(result)
            elif plot_type == "g2":
                self.apply_g2_result(result)
            elif plot_type == "twotime":
                self.apply_twotime_result(result)
            elif plot_type == "intensity_t":
                self.apply_intensity_result(result)
            elif plot_type == "stability":
                self.apply_stability_result(result)
            elif plot_type == "qmap":
                self.apply_qmap_result(result)

            self.progress_manager.complete_operation(operation_id, True)
            self.toast_manager.show_success(
                f"{plot_type.replace('_', ' ').title()} plot updated"
            )

        except Exception as e:
            logger.error(f"Error applying {plot_type} plot result: {e}")
            self.progress_manager.complete_operation(
                operation_id, False, f"Plot error: {e!s}"
            )
            self.toast_manager.show_error(f"Plot error: {e!s}")

        # Clean up; a request suppressed while this operation ran is drawn now.
        self._settle_plot_operation(operation_id, failed=False)

    def on_async_operation_error(
        self, operation_id: str, error_msg: str, traceback_str: str
    ):
        """Handle async operation errors."""
        if (
            operation_id.startswith("plot_")
            and operation_id not in self.active_plot_operations
        ):
            return

        logger.error(f"Async operation {operation_id} failed: {error_msg}")
        logger.debug(f"Traceback: {traceback_str}")

        self.progress_manager.complete_operation(operation_id, False, error_msg)
        self.toast_manager.show_error(f"Operation failed: {error_msg}")

        # The newest request for the tab must still be drawn.
        self._settle_plot_operation(operation_id, failed=True)

    def on_async_operation_cancelled(self, operation_id: str):
        """A worker was cancelled without the viewer asking for it."""
        if operation_id not in self.active_plot_operations:
            return
        self.progress_manager.complete_operation(operation_id, False, "Cancelled")
        self._settle_plot_operation(
            operation_id, failed=True, reschedule=not self._retiring_plots
        )

    def apply_saxs_2d_result(self, result):
        """Apply SAXS 2D plot result to the GUI."""
        if result is None:
            logger.debug("No SAXS 2D data to plot")
            return

        # Handle info messages
        if isinstance(result, dict) and result.get("type") == "info":
            logger.info(
                f"SAXS 2D plotting info: {result.get('message', 'Unknown info')}"
            )
            return

        # The worker's frame (image, orientation, beam centre of ONE file) and
        # the options it was made for are drawn by the same function as the
        # synchronous plot; nothing is recombined here.
        from .module import saxs2d

        saxs2d.draw(self.pg_saxs, result["frame"], **result["display"])

    def apply_g2_result(self, result):
        """Apply G2 plot result to the GUI.

        Uses the pre-computed data returned by the async worker directly,
        avoiding redundant synchronous re-computation on the main thread
        (BUG-014).
        """
        if result is None:
            logger.debug("No G2 data to plot")
            return

        # Handle info messages
        if isinstance(result, dict) and result.get("type") == "info":
            logger.info(f"G2 plotting info: {result.get('message', 'Unknown info')}")
            return

        # Use the pre-computed data from the worker result directly.
        # The worker has already extracted q, tel, g2, g2_err, labels and
        # computed the plot geometry on the background thread.  We only
        # need to call the rendering step on the main thread.
        try:
            from .module import g2mod

            plot_params = result.get("plot_params", {})
            q = result.get("q")
            tel = result.get("tel")
            g2 = result.get("g2")
            g2_err = result.get("g2_err")
            labels = result.get("labels")
            num_figs = result.get("num_figs")
            fit_results = result.get("fit_results")

            if q is None or g2 is None:
                logger.warning("G2 worker result missing required data fields")
                return

            # Render the pre-computed data into the plot handler without
            # re-calling get_data() or get_xf_list() on the main thread.
            g2mod.pg_plot_from_data(
                self.mp_g2,
                q=q,
                tel=tel,
                g2=g2,
                g2_err=g2_err,
                labels=labels,
                num_figs=num_figs,
                fit_results=fit_results,
                # t_range/y_range are forwarded so the renderer applies the manual
                # axis ranges (it ignores them itself when t_auto/y_auto is set).
                **{
                    k: v
                    for k, v in plot_params.items()
                    if k not in {"q_range", "rows", "target_timestamp"}
                },
            )
            self._initialize_g2_fit_bounds(result.get("fit_delay_extent"))
            logger.info("G2 plot applied successfully from async worker result")

        except Exception as e:
            logger.error(f"Failed to apply G2 result: {e}", exc_info=True)

    def apply_twotime_result(self, result):
        """Apply two-time plot result to the GUI."""
        if result is None:
            logger.debug("No twotime data to plot")
            return

        # A late worker callback can fire after the workspace was reset/closed.
        if self.vk is None:
            logger.debug("ViewerKernel is None; skipping twotime result")
            return

        # Handle info messages
        if isinstance(result, dict) and result.get("type") == "info":
            logger.info(
                f"Two-time plotting info: {result.get('message', 'Unknown info')}"
            )
            return

        c2_result = result["c2_result"]
        new_qbin_labels = result["new_qbin_labels"]
        fname = getattr(result.get("xfile"), "fname", None)

        # Update the two-time plots using lazy loading
        self.vk.get_module("twotime").plot_twotime_g2(self.mp_2t_hdls, c2_result)

        if new_qbin_labels:
            logger.debug(
                f"[ASYNC] Repopulating ComboBox with new labels: {new_qbin_labels}"
            )
            # Preserve current selection when repopulating; on a dataset
            # switch, prefer this file's remembered qbin over whatever the
            # combobox still shows from the previous file.
            current_selection = self.comboBox_twotime_selection.currentIndex()
            if fname is not None and fname != self._twotime_last_fname:
                current_selection = self._twotime_qbin_memory.get(fname, 0)
            logger.debug(
                f"COMBOBOX POPULATE: [ASYNC] Current selection before repopulation: {current_selection}"
            )
            logger.debug(
                f"COMBOBOX POPULATE: [ASYNC] Adding {len(new_qbin_labels)} items to ComboBox: {new_qbin_labels[:5]}..."
            )  # Show first 5 items

            # Block signals to prevent recursive updates
            self.comboBox_twotime_selection.blockSignals(True)
            self.horizontalSlider_twotime_selection.blockSignals(True)

            try:
                self.comboBox_twotime_selection.clear()
                self.comboBox_twotime_selection.addItems(new_qbin_labels)
                self.horizontalSlider_twotime_selection.setMaximum(
                    len(new_qbin_labels) - 1
                )

                # Restore selection if it's valid
                if 0 <= current_selection < len(new_qbin_labels):
                    self.comboBox_twotime_selection.setCurrentIndex(current_selection)
                    self.horizontalSlider_twotime_selection.setValue(current_selection)
                    logger.debug(f"[ASYNC] Restored selection to: {current_selection}")
                else:
                    logger.debug(
                        f"[ASYNC] Selection {current_selection} is invalid for {len(new_qbin_labels)} items, defaulting to 0"
                    )
            finally:
                # Always restore signal connections
                self.comboBox_twotime_selection.blockSignals(False)
                self.horizontalSlider_twotime_selection.blockSignals(False)
        if fname is not None:
            self._twotime_last_fname = fname

    def apply_intensity_result(self, result):
        """Apply intensity plot result to the GUI."""
        if result is None:
            logger.debug("No intensity data to plot")
            return

        # A late worker callback can fire after the workspace was reset/closed.
        if self.vk is None:
            logger.debug("ViewerKernel is None; skipping intensity result")
            return

        # Handle info messages
        if isinstance(result, dict) and result.get("type") == "info":
            logger.info(
                f"Intensity plotting info: {result.get('message', 'Unknown info')}"
            )
            return

        try:
            # Use the pre-computed plot_params from the async worker
            # to ensure consistency with the original request
            plot_params = result["plot_params"]
            rows = plot_params.get("rows", [])

            # Render using the viewer kernel with the original async parameters
            filtered_params = {
                k: v
                for k, v in plot_params.items()
                if k not in ["rows", "target_timestamp"]
            }
            self.vk.plot_intt(self.pg_intt, rows=rows, **filtered_params)
            logger.info("Intensity plot applied successfully")

        except Exception as e:
            logger.error(f"Failed to apply intensity result: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")

    def apply_stability_result(self, result):
        """Apply stability plot result to the GUI."""
        if result is None:
            logger.debug("No stability data to plot")
            return

        # Handle info messages
        if isinstance(result, dict) and result.get("type") == "info":
            logger.info(
                f"Stability plotting info: {result.get('message', 'Unknown info')}"
            )
            return

        try:
            # Use the pre-computed plot_params from the async worker
            # to ensure consistency with the original request
            plot_params = result["plot_params"]
            rows = plot_params.get("rows", [])

            # Render using the viewer kernel with the original async parameters
            filtered_params = {
                k: v
                for k, v in plot_params.items()
                if k not in ["rows", "target_timestamp"]
            }
            if self.vk:
                self.vk.plot_stability(self.mp_stab, rows=rows, **filtered_params)
            logger.info("Stability plot applied successfully")

        except Exception as e:
            logger.error(f"Failed to apply stability result: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")

    def apply_qmap_result(self, result):
        """Apply Q-map plot result to the GUI."""
        if result is None:
            logger.debug("No Q-map data to plot")
            return

        # Handle info messages
        if isinstance(result, dict) and result.get("type") == "info":
            logger.info(f"Q-map plotting info: {result.get('message', 'Unknown info')}")
            return

        image_data = result["image_data"]
        levels = result["levels"]

        # Convert JAX arrays to NumPy before passing to PyQtGraph (BUG-027)
        self.pg_qmap.setImage(ensure_numpy(image_data), levels=levels)

    def get_selected_rows(self):
        selected_index = self.list_view_target.selectedIndexes()
        selected_row = [x.row() for x in selected_index]
        # the selected index is ordered;
        selected_row.sort()

        # If no specific rows are selected but files exist in target list,
        # default to using all files for better user experience
        if not selected_row and self.vk and self.vk.target and len(self.vk.target) > 0:
            selected_row = list(range(len(self.vk.target)))
            logger.debug(
                f"No specific rows selected, defaulting to all {len(selected_row)} files in target list"
            )

        return selected_row

    @QtCore.Slot()
    def _open_g2_regroup(self) -> None:
        if self._g2_regroup_dialog is not None:
            self._g2_regroup_dialog.raise_()
            self._g2_regroup_dialog.activateWindow()
            return

        selected = self.list_view_target.selectedIndexes()
        if self.vk is None or len(selected) != 1:
            self.statusbar.showMessage(
                "Select one result in the Target list to regroup G2.", 5000
            )
            return

        source = (Path(self.vk.path) / self.vk.target[selected[0].row()]).resolve()
        dialog = G2RegroupDialog(source, self.thread_pool, self)
        self._g2_regroup_dialog = dialog
        dialog.set_export_active(self._g2_regroup_export is not None)
        dialog.finished.connect(self._g2_regroup_closed)
        dialog.save_requested.connect(self._start_g2_regroup_export)
        dialog.open()

    def _close_g2_regroup(self) -> None:
        if self._g2_regroup_dialog is not None:
            self._g2_regroup_dialog.reject()

    @QtCore.Slot(int)
    def _g2_regroup_closed(self, _result: int) -> None:
        self._g2_regroup_dialog = None

    @QtCore.Slot(object, object)
    def _start_g2_regroup_export(
        self, result: G2RegroupResult, destination: Path
    ) -> None:
        if self._g2_regroup_export is not None:
            return
        from .utils.export_guard import ExportCollisionError, ExportGuard

        try:  # the source file is an input of this export
            ExportGuard([result.source_path]).claim(destination, "regrouped G2")
        except ExportCollisionError as exc:
            logger.error("Regroup export refused: %s", exc)
            message = f"Export refused: {exc}"
            self.statusbar.showMessage(message, 10000)
            self.toast_manager.show_error(message)
            if self._g2_regroup_dialog is not None:
                self._g2_regroup_dialog.set_export_active(False, message)
            return
        worker = G2RegroupExportWorker(result, destination)
        self._g2_regroup_export = worker
        if self._g2_regroup_dialog is not None:
            self._g2_regroup_dialog.set_export_active(True)
        queued = QtCore.Qt.ConnectionType.QueuedConnection
        worker.signals.finished.connect(self._g2_regroup_export_finished, queued)
        worker.signals.error.connect(self._g2_regroup_export_failed, queued)
        self.thread_pool.start(worker)

    def _release_g2_regroup_export(self, message: str) -> None:
        self._g2_regroup_export = None
        if self._g2_regroup_dialog is not None:
            self._g2_regroup_dialog.set_export_active(False, message)

    @QtCore.Slot(object)
    def _g2_regroup_export_finished(self, destination: Path) -> None:
        assert self._g2_regroup_export is not None
        source = self._g2_regroup_export.result.source_path.name
        message = f"Saved regrouped G2 from {source} to {destination}"
        self._release_g2_regroup_export(message)
        self.toast_manager.show_success(message)

    @QtCore.Slot(str, str, str, int)
    def _g2_regroup_export_failed(
        self, worker_id: str, error: str, _traceback: str, _retry: int
    ) -> None:
        assert self._g2_regroup_export is not None
        source = self._g2_regroup_export.result.source_path.name
        reason = error.removeprefix(f"Error in worker {worker_id}: ")
        message = (
            f"Could not save regrouped G2 from {source}. {reason}\n"
            "Choose a new writable destination to retry."
        )
        self._release_g2_regroup_export(message)
        self.toast_manager.show_error(message)

    def update_plot(self):
        """
        Update plot display based on current tab and selected files.

        Automatically determines whether to use synchronous or asynchronous
        plotting based on the current tab and available async kernel.
        Supported async tabs include SAXS 2D/1D, G2, two-time, intensity,
        stability, and Q-map visualizations.

        Notes
        -----
        - Checks plot parameters for changes before updating
        - Uses async plotting for performance-critical visualizations
        - Falls back to synchronous plotting for unsupported tabs
        - Updates plot kwargs record for caching
        """
        idx = self.tabWidget.currentIndex()
        tab_name = tab_mapping.get(idx, "")
        logger.debug(f"update_plot called for tab: {tab_name} (index: {idx})")

        if not tab_name:
            logger.debug(f"No tab mapping for index {idx}, skipping update")
            return

        if tab_name == "average":
            return

        # Check if files are selected before attempting to plot
        # Some tabs like diffusion should always update to show proper empty state
        tabs_requiring_files = [
            "saxs_2d",
            "saxs_1d",
            "stability",
            "intensity_t",
            "g2",
            "g2_fitting",
            "twotime",
            "qmap",
            "g2_map",
        ]
        if (
            not self.vk or not self.vk.target or len(self.vk.target) == 0
        ) and tab_name in tabs_requiring_files:
            logger.debug(
                f"No files selected for {tab_name} plotting, "
                f"clearing plot and skipping update"
            )
            # Clear the plot to show empty state
            self._clear_plot_for_tab(tab_name)
            return

        # Check if we should use async plotting
        # Note: twotime removed from async list to fix automatic plotting issues
        use_async = self.async_vk is not None and tab_name in [
            "saxs_2d",
            "g2",
            "intensity_t",
            "stability",
            "qmap",
        ]

        logger.debug(
            f"Using {'async' if use_async else 'sync'} plotting for {tab_name}"
        )

        if use_async:
            self.update_plot_async(tab_name)
        else:
            self.update_plot_sync(tab_name)

    def _plot_request(self, kwargs: dict) -> dict:
        """The plot kwargs plus everything else that decides what is drawn.

        Recorded and compared instead of the bare kwargs, so a request is only
        "unchanged" if the same rows still hold the same on-disk revisions of
        their files and the display partition is the same.  (The call itself
        keeps using *kwargs*.)
        """
        rows = kwargs.get("rows")
        if rows is None:
            rows = self.get_selected_rows()
        names = self.vk.target
        revisions = tuple(
            (names[r], self.vk.file_revision(names[r]))
            for r in rows
            if isinstance(r, int) and not isinstance(r, bool) and 0 <= r < len(names)
        )
        return {
            **kwargs,
            "source_revisions": revisions,
            "display_partition": self._display_partition_version,
            # a replaced or re-selected SAXS background changes what SAXS-1D draws
            "saxs1d_background": self.vk.background_revision() if self.vk else None,
        }

    def _invalidate_plots(self) -> None:
        """Forget what is drawn: in-flight plot operations are retired and no
        tab's last request is "unchanged" any more (caller redraws)."""
        self._retire_plot_operations()
        self.plot_kwargs_record = {v: {} for _, v in tab_mapping.items()}

    def update_plot_sync(self, tab_name):
        """Synchronous plot update (original behavior)."""
        # Special handling for diffusion tab - update the pre-plot only
        if tab_name == "diffusion":
            try:
                self.init_diffusion()
                logger.debug("Diffusion pre-plot updated")
            except Exception as e:
                logger.exception(f"Failed to update diffusion pre-plot: {e}")
            return  # View-only on tab switch; fitting via "Fit Plot" button

        # g2 fitting tab is view-only on switch; fitting via "Fit" button
        if tab_name == "g2_fitting":
            return

        func = getattr(self, "plot_" + tab_name)
        try:
            if not self.vk:
                logger.debug(f"No viewer kernel; skipping {tab_name} update")
                return

            kwargs = func(dryrun=True)
            if not kwargs:
                logger.info(
                    "No parameters returned for %s (likely no target/fit); skipping update",
                    tab_name,
                )
                return

            kwargs["target_timestamp"] = self.vk.timestamp
            request = self._plot_request(kwargs)
            if self.plot_kwargs_record[tab_name] != request:
                logger.debug(f"Plot parameters changed for {tab_name}")
                logger.debug(f"Old params: {self.plot_kwargs_record[tab_name]}")
                logger.debug(f"New params: {request}")
                self.plot_kwargs_record[tab_name] = request
                func(dryrun=False)
            else:
                logger.debug(f"No parameter changes for {tab_name}, skipping update")
        except Exception as e:
            logger.error(f"update selection in [{tab_name}] failed")
            logger.exception(e)

    def _clear_plot_for_tab(self, tab_name):
        """Clear plot display for specified tab when no files are selected."""
        try:
            plot_handlers = {
                "saxs_2d": self.pg_saxs if hasattr(self, "pg_saxs") else None,
                "saxs_1d": self.mp_saxs1d if hasattr(self, "mp_saxs1d") else None,
                "stability": self.mp_stab if hasattr(self, "mp_stab") else None,
                "intensity_t": self.pg_intt if hasattr(self, "pg_intt") else None,
                "g2": self.mp_g2 if hasattr(self, "mp_g2") else None,
                "g2_fitting": self.mp_g2_fitting
                if hasattr(self, "mp_g2_fitting")
                else None,
                "twotime": self.mp_2t_hdls if hasattr(self, "mp_2t_hdls") else None,
                "qmap": self.pg_qmap if hasattr(self, "pg_qmap") else None,
                "g2_map": self.pg_g2map if hasattr(self, "pg_g2map") else None,
            }

            handler = plot_handlers.get(tab_name)
            if handler:
                if hasattr(handler, "clear"):
                    handler.clear()
                elif hasattr(handler, "setImage"):
                    # For ImageView widgets, set empty image
                    empty_image = np.zeros((50, 50))
                    handler.setImage(empty_image)
                logger.debug(f"Cleared plot for {tab_name}")
                # Show status message to user for guidance
                self.statusbar.showMessage(
                    f"No files selected for {tab_name.replace('_', ' ').title()} - please select files from source list and click 'Add Target'",  # nosec B608
                    5000,
                )
        except Exception as e:
            logger.warning(f"Failed to clear plot for {tab_name}: {e}")

    def _guard_no_data(self, tab_name: str) -> bool:
        """Return False and show guidance when no target data is available."""
        has_data = bool(self.vk and getattr(self.vk, "target", []))
        if has_data:
            return True

        logger.info("Skipping %s plot because no target data is available", tab_name)
        self._clear_plot_for_tab(tab_name)
        if self.statusbar:
            self.statusbar.showMessage(
                f"No data for {tab_name.replace('_', ' ')}. Add files to the target list to plot.",
                4000,
            )
        return False

    def update_plot_async(self, tab_name):
        """Asynchronous plot update with progress indication."""
        # Check if there's already an active operation for this plot type
        for _op_id, plot_type in list(self.active_plot_operations.items()):
            if plot_type == tab_name:
                logger.info(f"Async {tab_name} plot already in progress, skipping")
                # Remember the newer request: whichever way the running
                # operation ends (done, failed, cancelled), it is drawn then.
                self._plot_pending.add(tab_name)
                return

        operation_id = None
        try:
            # Get plot parameters
            func = getattr(self, "plot_" + tab_name)
            if not self.vk:
                logger.debug(f"No viewer kernel; skipping async {tab_name} update")
                return

            kwargs = func(dryrun=True)
            if not kwargs:
                logger.info(
                    "No parameters for async %s (likely no target/fit); skipping",
                    tab_name,
                )
                return

            kwargs["target_timestamp"] = self.vk.timestamp
            request = self._plot_request(kwargs)

            # Check if parameters changed
            if self.plot_kwargs_record[tab_name] == request:
                logger.debug(f"No parameter changes for async {tab_name}, skipping")
                return  # No change needed
            logger.debug(f"Async plot parameters changed for {tab_name}")
            logger.debug(f"Old params: {self.plot_kwargs_record[tab_name]}")
            logger.debug(f"New params: {request}")

            self.plot_kwargs_record[tab_name] = request

            # Start async plotting
            operation_id = f"plot_{tab_name}_{uuid4().hex}"

            # Start progress tracking
            description = f"Plotting {tab_name.replace('_', ' ').title()}"
            self.progress_manager.start_operation(
                operation_id, description, show_in_statusbar=True, is_cancellable=True
            )

            # Track this operation
            self.active_plot_operations[operation_id] = tab_name
            self._plot_op_requests[operation_id] = request

            # Map tab names to async methods
            # Note: twotime is included here for manual async calls but is
            # excluded from the auto-async list in update_plot() to avoid
            # automatic plotting issues.
            async_method_map = {
                "saxs_2d": self.async_vk.plot_saxs_2d_async,
                "g2": self.async_vk.plot_g2_async,
                "twotime": self.async_vk.plot_twotime_async,
                "intensity_t": self.async_vk.plot_intensity_async,
                "stability": self.async_vk.plot_stability_async,
                "qmap": self.async_vk.plot_qmap_async,
            }

            # Get the appropriate plot handler
            handler_map = {
                "saxs_2d": self.pg_saxs,
                "g2": self.mp_g2,
                "twotime": self.mp_2t_hdls,
                "intensity_t": self.pg_intt,
                "stability": self.mp_stab,
                "qmap": self.pg_qmap,
            }

            if tab_name in async_method_map:
                plot_handler = handler_map[tab_name]
                async_method = async_method_map[tab_name]

                # Start the async operation
                async_method(plot_handler, operation_id=operation_id, **kwargs)

                logger.info(f"Started async {tab_name} plotting: {operation_id}")

        except Exception as e:
            logger.error(f"Failed to start async {tab_name} plotting: {e}")
            # Clear any partially-registered operation so this tab is not left
            # permanently marked in-progress (which would skip all future
            # updates via the guard at the top of this method).
            if operation_id is not None:
                self.active_plot_operations.pop(operation_id, None)
                try:
                    self.progress_manager.complete_operation(
                        operation_id, False, str(e)
                    )
                except Exception:
                    logger.debug("progress cleanup failed", exc_info=True)
            # Fall back to synchronous plotting
            self.update_plot_sync(tab_name)

    def plot_metadata(self, dryrun=False):
        if not self.vk:
            return None

        kwargs = {"rows": self.get_selected_rows()}
        if dryrun:
            return kwargs

        xf_list = self.vk.get_xf_list(**kwargs)
        if not xf_list:
            logger.warning("No files available for metadata display")
            # Clear the metadata display
            empty_params = Parameter.create(name="Settings", type="group", children=[])
            self.hdf_info.setParameters(empty_params, showTop=True)
            return None

        msg = xf_list[0].get_hdf_info()
        hdf_info_data = create_param_tree(msg)
        hdf_params = Parameter.create(
            name="Settings", type="group", children=hdf_info_data
        )
        self.hdf_info.setParameters(hdf_params, showTop=True)
        return None

    def saxs2d_mouseMoved(self, pos):
        if self.vk is None:
            return
        if self.pg_saxs.view.sceneBoundingRect().contains(pos):
            mouse_point = self.pg_saxs.getView().mapSceneToView(pos)
            # the pixel drawn there, through the item's own (possibly rotated) mapping
            x, y = self.pg_saxs.pixel_index_at(mouse_point.x(), mouse_point.y())
            rows = self.get_selected_rows()
            payload = self.vk.get_info_at_mouse(rows, x, y)
            if payload:
                self.saxs2d_display.setText(payload)
                self._rate_limited_logger.debug(f"SAXS 2D mouse at ({x}, {y})")

    def plot_saxs_2d_selection(self):
        selection = self.spinBox_saxs2d_selection.value()
        # The direct draw below supersedes any in-flight async SAXS plot, whose
        # request (target rows) cannot tell it apart from this selection.
        self._retire_plot_operations("saxs_2d")
        self.plot_saxs_2d(selection=selection)

    def _current_twotime_fname(self):
        """Return fname of the currently selected Twotime file, or None."""
        if self.vk is None:
            return None
        xf_list = self.vk.get_xf_list(self.get_selected_rows(), filter_atype="Twotime")
        return xf_list[0].fname if xf_list else None

    def on_twotime_q_selection_changed(self, index):
        """Handle Q value selection changes in twotime ComboBox."""
        logger.debug(f"TWOTIME Q SELECTION CHANGED: ComboBox index changed to {index}")
        logger.debug(
            f"TWOTIME Q SELECTION CHANGED: ComboBox current text = '{self.comboBox_twotime_selection.currentText()}'"
        )
        logger.debug(
            f"TWOTIME Q SELECTION CHANGED: ComboBox item count = {self.comboBox_twotime_selection.count()}"
        )
        logger.debug(
            f"TWOTIME Q SELECTION CHANGED: Current tab = {self.tabWidget.currentIndex()}"
        )

        # Also update the slider to stay in sync
        if 0 <= index < self.comboBox_twotime_selection.count():
            self.horizontalSlider_twotime_selection.blockSignals(True)
            self.horizontalSlider_twotime_selection.setValue(index)
            self.horizontalSlider_twotime_selection.blockSignals(False)

        # Remember this qbin choice per-file so switching datasets and back
        # restores it, instead of carrying over whatever index the combobox
        # happened to be showing.
        fname = self._current_twotime_fname()
        if fname is not None and index >= 0:
            _remember_twotime_qbin(self._twotime_qbin_memory, fname, index)

        # Now trigger the plot update
        logger.debug("TWOTIME Q SELECTION CHANGED: Calling update_plot()")
        self.update_plot()
        logger.debug("TWOTIME Q SELECTION CHANGED: update_plot() completed")

    def plot_saxs_2d(self, selection=None, dryrun=False):
        kwargs = {
            "plot_type": self.cb_saxs2D_type.currentText(),
            "cmap": self.cb_saxs2D_cmap.currentText(),
            "rotate": self.saxs2d_rotate.isChecked(),
            "autolevel": self.saxs2d_autolevel.isChecked(),
            "vmin": self.saxs2d_min.value(),
            "vmax": self.saxs2d_max.value(),
        }
        if selection is not None and selection >= 0:  # row 0 is a row
            kwargs["rows"] = [selection]
        else:
            kwargs["rows"] = self.get_selected_rows()

        if not dryrun and not self._guard_no_data("saxs_2d"):
            return None

        if dryrun:
            return kwargs
        self.vk.plot_saxs_2d(pg_hdl=self.pg_saxs, **kwargs)
        return None

    def saxs2d_roi_add(self):
        if self.vk is None:
            self.statusbar.showMessage("No directory loaded", 2000)
            return
        sl_type_idx = self.cb_saxs2D_roi_type.currentIndex()
        color = ("g", "y", "b", "r", "c", "m", "k", "w")[
            self.cb_saxs2D_roi_color.currentIndex()
        ]

        # Map UI ROI types correctly: Q-Wedge (0), Phi-Ring (1)
        roi_types = ("Q-Wedge", "Phi-Ring")

        kwargs = {
            "sl_type": roi_types[sl_type_idx],
            "width": self.sb_saxs2D_roi_width.value(),
            "color": color,
        }

        logger.debug(f"GUI: Attempting to add ROI with kwargs: {kwargs}")
        result = self.vk.add_roi(self.pg_saxs, rows=self.get_selected_rows(), **kwargs)
        if result is None:
            logger.warning(f"GUI: ROI addition failed for type '{kwargs['sl_type']}'")
        else:
            logger.debug(f"GUI: ROI addition succeeded, returned: {result}")

    def plot_saxs_1d(self, dryrun=False):
        kwargs = {
            "plot_type": self.cb_saxs_type.currentIndex(),
            "plot_offset": self.sb_saxs_offset.value(),
            "plot_norm": self.cb_saxs_norm.currentIndex(),
            "rows": self.get_selected_rows(),
            "qmin": self.saxs1d_qmin.value(),
            "qmax": self.saxs1d_qmax.value(),
            "loc": self.saxs1d_legend_loc.currentText(),
            "marker_size": self.sb_saxs_marker_size.value(),
            "sampling": self.saxs1d_sampling.value(),
            "all_phi": self.box_all_phi.isChecked(),
            "absolute_crosssection": self.cbox_use_abs.isChecked(),
            "subtract_background": self.cb_sub_bkg.isChecked(),
            "weight": self.bkg_weight.value(),
            "show_roi": self.box_show_roi.isChecked(),
            "show_phi_roi": self.box_show_phi_roi.isChecked(),
        }
        if kwargs["qmin"] >= kwargs["qmax"]:
            self.statusbar.showMessage("check qmin and qmax")
            return None

        if not dryrun and not self._guard_no_data("saxs_1d"):
            return None

        if dryrun:
            return kwargs
        self.vk.plot_saxs_1d(self.pg_saxs, self.mp_saxs, **kwargs)
        # adjust the line behavior
        self.switch_saxs1d_line()
        return None

    def switch_saxs1d_line(self):
        lb_type = self.saxs1d_lb_type.currentIndex()
        lb_type = [None, "slope", "hline"][lb_type]
        self.vk.switch_saxs1d_line(self.mp_saxs, lb_type)

    def saxs1d_export(self):
        if self.vk is None:
            self.statusbar.showMessage("No directory loaded", 2000)
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, caption="select a folder to export SAXS profiles"
        )

        if folder in [None, ""]:
            return

        self.vk.export_saxs_1d(self.pg_saxs, folder)

    def _init_twotime_tab(self):
        """Rebuild the two-time tab with a side-panel layout.

        Restructures from the original vertical split
        (top: SAXS+dqmap+G2 in a row, bottom: C2 heatmap) to a side-panel
        layout that maximizes the C2 heatmap display area:

        Left panel (narrow):  SAXS + dqmap stacked vertically
        Right panel (wide):   C2 heatmap (main) + G2 curves (below)
        Bottom:               Controls (unchanged)
        """
        # Remove old splitter from the tab grid layout
        self.gridLayout_33.removeWidget(self.splitter_4)

        # Detach existing plot widgets from old splitter
        self.mp_2t_map.setParent(None)
        self.mp_2t.setParent(None)

        # Discard the old vertical splitter
        self.splitter_4.deleteLater()

        # --- Main horizontal splitter (left nav | right main view) ---
        self.splitter_2t_main = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter_2t_main.setObjectName("splitter_2t_main")

        # LEFT PANEL: SAXS + dqmap navigation images (stacked vertically)
        left_policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        left_policy.setHorizontalStretch(1)
        self.mp_2t_map.setSizePolicy(left_policy)
        self.mp_2t_map.setMinimumSize(QtCore.QSize(180, 200))
        self.splitter_2t_main.addWidget(self.mp_2t_map)

        # RIGHT PANEL: C2 heatmap + G2 curves in a vertical splitter
        self.splitter_2t_right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.splitter_2t_right.setObjectName("splitter_2t_right")

        # C2 heatmap — primary visualization, gets most vertical space
        c2_policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        c2_policy.setVerticalStretch(3)
        self.mp_2t.setSizePolicy(c2_policy)
        self.mp_2t.setMinimumSize(QtCore.QSize(300, 250))
        self.splitter_2t_right.addWidget(self.mp_2t)

        # G2 curves — below C2
        self.mp_2t_g2_plot = pg.PlotWidget()
        self.mp_2t_g2_plot.setObjectName("mp_2t_g2_plot")
        self.mp_2t_g2_plot.setBackground("w")
        g2_policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        g2_policy.setVerticalStretch(1)
        self.mp_2t_g2_plot.setSizePolicy(g2_policy)
        self.mp_2t_g2_plot.setMinimumSize(QtCore.QSize(300, 120))
        self.splitter_2t_right.addWidget(self.mp_2t_g2_plot)

        # Right panel size policy
        right_policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        right_policy.setHorizontalStretch(3)
        self.splitter_2t_right.setSizePolicy(right_policy)
        self.splitter_2t_main.addWidget(self.splitter_2t_right)

        # Set initial proportions (~25% left, ~75% right)
        self.splitter_2t_main.setSizes([250, 750])
        self.splitter_2t_main.setStretchFactor(0, 1)
        self.splitter_2t_main.setStretchFactor(1, 3)
        self.splitter_2t_right.setStretchFactor(0, 3)
        self.splitter_2t_right.setStretchFactor(1, 1)

        # Place new splitter at the same grid position as the old one
        self.gridLayout_33.addWidget(self.splitter_2t_main, 0, 0, 1, 1)

    def init_twotime_plot_handler(self):
        self.mp_2t_map.clear()
        self.mp_2t_hdls = {}
        labels = ["saxs", "dqmap"]
        titles = ["scattering", "dynamic_qmap"]
        cmaps = ["viridis", "tab20"]
        self.mp_2t_map.setBackground("w")
        for n in range(2):
            # Stack SAXS and dqmap vertically in the left panel
            plot_item = self.mp_2t_map.addPlot(row=n, col=0)
            # Remove axes
            plot_item.hideAxis("left")
            plot_item.hideAxis("bottom")
            plot_item.getViewBox().setDefaultPadding(0)

            plot_item.setMouseEnabled(x=False, y=False)
            image_item = pg.ImageItem(np.ones((128, 128)))
            image_item.setOpts(axisOrder="row-major")  # Set to row-major order

            plot_item.setTitle(titles[n])
            plot_item.addItem(image_item)
            plot_item.setAspectLocked(True)

            cmap = pg.colormap.getFromMatplotlib(cmaps[n])
            if n == 1:
                positions = cmap.pos
                colors = cmap.color
                new_color = [0, 0, 1, 1.0]
                colors[-1] = new_color
                # need to convert to 0-255 range for pyqtgraph ColorMap
                cmap = pg.ColorMap(positions, colors * 255)
            colorbar = plot_item.addColorBar(image_item, colorMap=cmap)
            self.mp_2t_hdls[labels[n]] = image_item
            self.mp_2t_hdls[labels[n] + "_colorbar"] = colorbar

        # G2 curves use a dedicated PlotWidget on the right panel
        c2g2_plot = self.mp_2t_g2_plot.getPlotItem()
        c2g2_plot.setLabel("left", "g2")
        c2g2_plot.setLabel("bottom", "t (s)")
        self.mp_2t_hdls["c2g2"] = c2g2_plot

        self.mp_2t_hdls["dqmap"].mouseClickEvent = self.pick_twotime_index
        self.mp_2t_hdls["saxs"].mouseClickEvent = self.pick_twotime_index
        self.mp_2t.ui.graphicsView.setBackground("w")
        self.mp_2t_hdls["tt"] = self.mp_2t
        self.mp_2t_hdls["tt"].view.invertY(False)
        self.mp_2t.view.setLabel("left", "t2", units="s")
        self.mp_2t.view.setLabel("bottom", "t1", units="s")

    def pick_twotime_index(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            pos = event.pos()
            x, y = int(pos.x()), int(pos.y())
            self.plot_twotime(highlight_xy=(x, y))
        event.accept()  # Mark the event as handled

    def _init_g2map_tab(self):
        """Initialize the G2 Map tab with dynamically created widgets."""
        # Create the G2 Map tab widget
        self.tab_g2map = QtWidgets.QWidget()
        self.tab_g2map.setObjectName("tab_g2map")

        # Main layout for the tab
        main_layout = QtWidgets.QVBoxLayout(self.tab_g2map)

        # Create a splitter for the three-panel layout
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setObjectName("splitter_g2map")

        # Panel 1: G2 Map (2D view)
        g2map_widget = QtWidgets.QWidget()
        g2map_layout = QtWidgets.QVBoxLayout(g2map_widget)
        g2map_label = QtWidgets.QLabel("G2 Map")
        g2map_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        g2map_plot = pg.PlotItem(
            labels={"bottom": "Q-bin index", "left": "Delay index"}
        )
        self.pg_g2map = pg.ImageView(view=g2map_plot)
        self.pg_g2map.ui.histogram.hide()
        self.pg_g2map.ui.roiBtn.hide()
        self.pg_g2map.ui.menuBtn.hide()
        g2map_layout.addWidget(g2map_label)
        g2map_layout.addWidget(self.pg_g2map)
        splitter.addWidget(g2map_widget)

        # Panel 2: Q-map (cropped)
        qmap_widget = QtWidgets.QWidget()
        qmap_layout = QtWidgets.QVBoxLayout(qmap_widget)
        qmap_label = QtWidgets.QLabel("Q-map (cropped)")
        qmap_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.pg_g2map_qmap = pg.ImageView()
        self.pg_g2map_qmap.ui.histogram.hide()
        self.pg_g2map_qmap.ui.roiBtn.hide()
        self.pg_g2map_qmap.ui.menuBtn.hide()
        qmap_layout.addWidget(qmap_label)
        qmap_layout.addWidget(self.pg_g2map_qmap)
        splitter.addWidget(qmap_widget)

        # Panel 3: G2 Profile
        profile_widget = QtWidgets.QWidget()
        profile_layout = QtWidgets.QVBoxLayout(profile_widget)
        profile_label = QtWidgets.QLabel("G2 Profile at Q-bin")
        profile_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.pg_g2map_profile = pg.PlotWidget()
        self.pg_g2map_profile.setLabel("left", "g2")
        self.pg_g2map_profile.setLabel("bottom", "tau", units="s")
        profile_layout.addWidget(profile_label)
        profile_layout.addWidget(self.pg_g2map_profile)
        splitter.addWidget(profile_widget)

        # Equal space for all three panels
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)
        splitter.setChildrenCollapsible(False)

        main_layout.addWidget(splitter)

        # Controls at the bottom
        controls_widget = QtWidgets.QGroupBox("G2 Map Controls")
        controls_layout = QtWidgets.QHBoxLayout(controls_widget)

        # Q-bin selector
        controls_layout.addWidget(QtWidgets.QLabel("Q-bin:"))
        self.spinBox_g2map_qbin = QtWidgets.QSpinBox()
        self.spinBox_g2map_qbin.setMinimum(0)
        self.spinBox_g2map_qbin.setMaximum(999)
        self.spinBox_g2map_qbin.setValue(0)
        controls_layout.addWidget(self.spinBox_g2map_qbin)

        # Normalization checkbox
        self.checkBox_g2map_normalize = QtWidgets.QCheckBox("Normalize baseline")
        controls_layout.addWidget(self.checkBox_g2map_normalize)

        controls_layout.addStretch()

        # G2 map colormap
        _g2map_cmaps = ["jet", "hot", "plasma", "viridis", "magma", "gray"]
        controls_layout.addWidget(QtWidgets.QLabel("G2 cmap:"))
        self.cb_g2map_cmap = QtWidgets.QComboBox()
        self.cb_g2map_cmap.addItems(_g2map_cmaps)
        self.cb_g2map_cmap.setCurrentText("jet")
        controls_layout.addWidget(self.cb_g2map_cmap)

        # G2 map scale (linear / log)
        controls_layout.addWidget(QtWidgets.QLabel("Scale:"))
        self.cb_g2map_scale = QtWidgets.QComboBox()
        self.cb_g2map_scale.addItems(["linear", "log"])
        controls_layout.addWidget(self.cb_g2map_scale)

        # Q-map colormap
        _qmap_cmaps = ["tab20b", "jet", "hot", "plasma", "viridis", "magma", "gray"]
        controls_layout.addWidget(QtWidgets.QLabel("Q cmap:"))
        self.cb_g2map_qmap_cmap = QtWidgets.QComboBox()
        self.cb_g2map_qmap_cmap.addItems(_qmap_cmaps)
        self.cb_g2map_qmap_cmap.setCurrentText("tab20b")
        controls_layout.addWidget(self.cb_g2map_qmap_cmap)

        # Plot button
        self.btn_plot_g2map = QtWidgets.QPushButton("Update G2 Map")
        self.btn_plot_g2map.clicked.connect(lambda: self.plot_g2_map())
        controls_layout.addWidget(self.btn_plot_g2map)

        main_layout.addWidget(controls_widget)

        # Insert the tab right after g2 fitting (index 6)
        self.tabWidget.insertTab(6, self.tab_g2map, "g2 map")

        # Connect Q-bin spinbox to update profile
        self.spinBox_g2map_qbin.valueChanged.connect(self._update_g2map_profile)

        # Changing colormap or scale re-plots immediately (same as SAXS2D)
        # (the combo's int index must not become plot_g2_map's dryrun flag)
        self.cb_g2map_cmap.currentIndexChanged.connect(lambda: self.plot_g2_map())
        self.cb_g2map_scale.currentIndexChanged.connect(lambda: self.plot_g2_map())
        self.cb_g2map_qmap_cmap.currentIndexChanged.connect(lambda: self.plot_g2_map())

        logger.info("g2 map tab initialized")

    def _refresh_tab_icons(self):
        """(Re)apply theme-colored category icons to each tab.

        Pulls freshly recolored SVG icons from the (possibly just-cleared) icon
        cache, so it must be called after a theme change as well as at startup.
        """
        from xpcsviewer.gui.icons import get_icon
        from xpcsviewer.gui.widgets.category_tab_bar import TAB_INDEX_CATEGORY

        category_icons = {
            "Scattering": get_icon("tab-scattering"),
            "Diagnostics": get_icon("tab-diagnostics"),
            "Correlation": get_icon("tab-correlation"),
            "Analysis": get_icon("tab-analysis"),
            "Setup": get_icon("tab-setup"),
        }
        for i in range(self.tabWidget.count()):
            category = TAB_INDEX_CATEGORY.get(i)
            if category and category in category_icons:
                self.tabWidget.setTabIcon(i, category_icons[category])

    def _apply_tab_labels(self):
        """Apply scientific notation tab labels with category tooltips.

        Also installs the CategoryTabBar for visual group separators and
        sets small category icons on each tab group's first tab.
        """
        from xpcsviewer.gui.widgets.category_tab_bar import (
            CategorySeparatorFilter,
        )

        tab_labels = [
            ("SAXS 2D", "2D scattering pattern | Scattering"),
            ("SAXS 1D", "1D azimuthal integration | Scattering"),
            ("Stability", "Frame stability analysis | Diagnostics"),
            ("I(t)", "Integrated intensity vs time | Diagnostics"),
            ("\u0067\u2082", "g\u2082 autocorrelation | Correlation"),
            ("\u0067\u2082 Fit", "g\u2082 curve fitting | Correlation"),
            ("\u0067\u2082 Map", "g\u2082 parameter map | Correlation"),
            ("Diffusion", "Diffusion coefficient | Analysis"),
            ("Two-Time", "Two-time correlation | Correlation"),
            ("Q-Map", "Reciprocal space map | Setup"),
            ("Average", "Frame averaging | Setup"),
            ("Metadata", "HDF5 metadata browser | Setup"),
        ]

        for i, (label, tooltip) in enumerate(tab_labels):
            if i < self.tabWidget.count():
                self.tabWidget.setTabText(i, label)
                self.tabWidget.setTabToolTip(i, tooltip)

        # Apply theme-colored category icons to each tab.
        self._refresh_tab_icons()

        # Install event filter for category separators (avoids setTabBar
        # which destroys existing tabs)
        existing_bar = self.tabWidget.tabBar()
        sep_filter = CategorySeparatorFilter(existing_bar)
        is_dark = self.theme_manager.get_current_theme() == "dark"
        sep_filter.set_dark_mode(is_dark)
        existing_bar.installEventFilter(sep_filter)
        self._category_tab_bar = sep_filter

    def _init_g2_fitting_tab(self):
        """Initialize the G2 Fitting tab with plot and fitting controls.

        This method:
        1. Creates a new tab with a PlotWidgetDev for fitted g2 display
        2. Moves the groupBox_2 (g2 fitting controls) from tab_6 to the new tab
        3. Inserts the tab right after the g2 tab (index 5)
        """
        # Create the G2 Fitting tab widget
        self.tab_g2_fitting = QtWidgets.QWidget()
        self.tab_g2_fitting.setObjectName("tab_g2_fitting")

        # Main layout for the tab
        main_layout = QtWidgets.QGridLayout(self.tab_g2_fitting)
        main_layout.setObjectName("gridLayout_g2_fitting")

        # Create scroll area for the plot (matching tab_6 structure)
        scroll_area = QtWidgets.QScrollArea(self.tab_g2_fitting)
        scroll_area.setObjectName("scrollArea_g2_fitting")
        size_policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        size_policy.setVerticalStretch(30)
        scroll_area.setSizePolicy(size_policy)
        scroll_area.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )
        scroll_area.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents
        )
        scroll_area.setWidgetResizable(True)

        # Scroll area content widget
        scroll_content = QtWidgets.QWidget()
        scroll_content.setObjectName("g2_fitting_scroll_area")
        scroll_layout = QtWidgets.QGridLayout(scroll_content)
        scroll_layout.setObjectName("gridLayout_g2_fitting_scroll")

        # Create PlotWidgetDev for fitted g2 display
        self.mp_g2_fitting = PlotWidgetDev(scroll_content)
        self.mp_g2_fitting.setObjectName("mp_g2_fitting")
        plot_size_policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.mp_g2_fitting.setSizePolicy(plot_size_policy)
        scroll_layout.addWidget(self.mp_g2_fitting, 0, 0, 1, 1)

        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area, 0, 0, 1, 1)

        # Move groupBox_2 (g2 fitting controls) from tab_6 splitter to new tab
        # First, remove it from the splitter
        self.groupBox_2.setParent(None)

        # Add it to the new tab's layout
        main_layout.addWidget(self.groupBox_2, 1, 0, 1, 1)

        # Insert the tab right after g2 (index 5)
        self.tabWidget.insertTab(5, self.tab_g2_fitting, "g2 fitting")

        # Hide g2_show_fit checkbox since fit is always shown on the fitting tab
        # and never on the raw g2 tab
        self.g2_show_fit.setVisible(False)

        logger.info("g2 fitting tab initialized")

    def open_simplemask(self):
        """Open or focus the SimpleMask window with current detector data.

        Extracts detector image and geometry metadata from the currently
        selected file and passes it to SimpleMaskWindow.
        """
        # Create window if it doesn't exist (reuse existing instance if open)
        if self._simplemask_window is None:
            self._simplemask_window = SimpleMaskWindow(parent_viewer=self)
            self._simplemask_window.mask_exported.connect(self.import_mask)
            self._simplemask_window.qmap_exported.connect(self.import_partition)
            logger.info("Created new SimpleMask window")

        if self._simplemask_window.isVisible():
            # Reloading would discard the editor's unsaved mask edits; just focus.
            self._simplemask_window.raise_()
            self._simplemask_window.activateWindow()
            return

        # Get detector image and metadata from current file
        detector_image, metadata = self._get_simplemask_data()

        # Load data into the window
        self._simplemask_window.load_from_viewer(detector_image, metadata)

        # Show and bring to front
        self._simplemask_window.show()
        self._simplemask_window.raise_()
        self._simplemask_window.activateWindow()

    def _get_simplemask_data(self) -> tuple:
        """Get detector image and metadata for SimpleMask.

        Returns:
            Tuple of (detector_image, metadata) or (None, None) if no data loaded
        """
        if not self.vk or not self.vk.target or len(self.vk.target) == 0:
            logger.debug("No files loaded for SimpleMask")
            return None, None

        try:
            xf_list = self.vk.get_xf_list()
            if not xf_list:
                return None, None

            xf = xf_list[0]  # Use first file

            # Get detector image (SAXS 2D)
            detector_image = xf.saxs_2d

            # Squeeze leading singleton dimension (e.g. (1, H, W) → (H, W))
            if (
                detector_image is not None
                and detector_image.ndim == 3
                and detector_image.shape[0] == 1
            ):
                detector_image = detector_image[0]

            # Extract geometry metadata. QMap stores lengths in metres; the
            # SimpleMask kernel expects millimetres.
            det_dist = getattr(xf, "det_dist", None)
            pix_dim = getattr(xf, "pixel_size", None)
            metadata = {
                "bcx": getattr(xf, "bcx", None),
                "bcy": getattr(xf, "bcy", None),
                "det_dist": det_dist * 1000.0 if det_dist is not None else None,
                "pix_dim": pix_dim * 1000.0 if pix_dim is not None else None,
                "energy": getattr(xf, "X_energy", None),
                "shape": detector_image.shape if detector_image is not None else None,
            }

            logger.debug(
                f"SimpleMask data: shape={metadata.get('shape')}, "
                f"bcx={metadata.get('bcx')}, bcy={metadata.get('bcy')}"
            )
            return detector_image, metadata

        except Exception as e:
            logger.error(f"Failed to get SimpleMask data: {e}")
            return None, None

    def import_mask(self, mask: np.ndarray) -> None:
        """Import mask from SimpleMask into current analysis.

        Args:
            mask: Boolean mask array from SimpleMask (True = keep, False = masked)
        """
        if not self.vk or not self.vk.target or len(self.vk.target) == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "No Data Loaded",
                "Load data first before importing a mask.",
            )
            return

        try:
            xf_list = self.vk.get_xf_list()
            if not xf_list:
                return

            # Each target is checked against its OWN detector: files of another
            # shape are left untouched, never given a mask of the wrong shape.
            accepted, skipped = _split_by_detector_shape(xf_list, [mask])
            if skipped:
                logger.warning(
                    f"Mask {np.shape(mask)} not applied to {len(skipped)} "
                    f"file(s): {_skipped_text(skipped)}"
                )
            if not accepted:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Dimension Mismatch",
                    f"Mask dimensions {np.shape(mask)} match no loaded file:\n"
                    f"{_skipped_text(skipped)}\n\nCannot import mask.",
                )
                return

            # Apply mask to the matching loaded XpcsFile objects (BUG-001 fix)
            for xf in accepted:
                xf.mask = mask

            logger.info(
                f"Mask exported from SimpleMask: shape {mask.shape}, "
                f"{np.sum(mask)} pixels unmasked - applied to {len(accepted)} file(s)"
            )

            message = (
                f"Mask imported: {np.sum(mask):,} of {mask.size:,} pixels unmasked"
            )
            if skipped:
                message += f". Skipped (other detector shape): {_skipped_text(skipped)}"
            self.update_plot()
            # after the redraw: starting a plot writes its own status message
            self.statusbar.showMessage(message)

        except Exception as e:
            logger.error(f"Failed to import mask: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Import Error",
                f"Failed to import mask:\n{e}",
            )

    def import_partition(self, partition: dict) -> None:
        """Import Q-map partition from SimpleMask.

        Args:
            partition: Partition dictionary from SimpleMask containing:
                - dynamic_roi_map: Dynamic partition map
                - static_roi_map: Static partition map
                - mask: Associated mask
                - beam_center_x/y: Beam center coordinates
                - and other partition metadata
        """
        if not self.vk or not self.vk.target or len(self.vk.target) == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "No Data Loaded",
                "Load data first before importing a partition.",
            )
            return

        try:
            dq_num = (
                partition.get("dynamic_roi_map", np.array([])).max()
                if "dynamic_roi_map" in partition
                else 0
            )
            sq_num = (
                partition.get("static_roi_map", np.array([])).max()
                if "static_roi_map" in partition
                else 0
            )

            xf_list = self.vk.get_xf_list() or []
            dqmap = partition.get("dynamic_roi_map")
            sqmap = partition.get("static_roi_map")
            # Each target takes the maps only if they have ITS detector shape.
            accepted, skipped = _split_by_detector_shape(
                xf_list, [m for m in (dqmap, sqmap) if m is not None]
            )
            if skipped:
                logger.warning(
                    f"Partition not applied to {len(skipped)} file(s): "
                    f"{_skipped_text(skipped)}"
                )
            if xf_list and not accepted:
                self.statusbar.showMessage(
                    f"Partition not imported: {_skipped_text(skipped)}"
                )
                return

            # Apply partition maps to the matching loaded XpcsFile objects
            # (BUG-001 fix). DISPLAY ONLY: stored under separate names that only
            # the Q-map displays read (XpcsFile.get_display_qmap). xf.dqmap /
            # xf.sqmap stay the file's own maps, which the G2/two-time bins, the
            # ROI calculations and the G2 map's detector image are bound to.
            for xf in accepted:
                if dqmap is not None:
                    xf.display_dqmap = dqmap
                if sqmap is not None:
                    xf.display_sqmap = sqmap

            self._display_partition_version += 1
            self._invalidate_plots()

            logger.info(
                f"Partition exported from SimpleMask: {dq_num} dynamic bins, "
                f"{sq_num} static bins - applied to {len(accepted)} file(s)"
            )

            message = (
                f"Partition imported for display: {dq_num} dynamic Q-bins, "
                f"{sq_num} static Q-bins (existing G2/two-time results keep "
                "the file's own Q bins)"
            )
            if skipped:
                message += f". Skipped (other detector shape): {_skipped_text(skipped)}"
            self.update_plot()
            # after the redraw: starting a plot writes its own status message
            self.statusbar.showMessage(message)

        except Exception as e:
            logger.error(f"Failed to import partition: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Import Error",
                f"Failed to import partition:\n{e}",
            )

    def _update_g2map_profile(self, qbin):
        """Update only the G2 profile when Q-bin changes."""
        if self.vk is None:
            return
        rows = self.get_selected_rows()
        if not rows:
            return
        try:
            xf_list = self.vk.get_xf_list(rows=rows)
            if not xf_list:
                return
            xf_obj = xf_list[0]
            # Update profile
            self.pg_g2map_profile.clear()
            color = (0, 128, 255)
            pen = pg.mkPen(color=color, width=2)
            x = xf_obj.t_el
            if qbin >= xf_obj.g2.shape[1]:
                qbin = xf_obj.g2.shape[1] - 1
            y = xf_obj.g2[:, qbin]
            dy = xf_obj.g2_err[:, qbin]

            # Filter valid data points (positive x for log scale)
            valid_mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
            if not np.any(valid_mask):
                return

            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            dy_valid = dy[valid_mask] if dy is not None else np.zeros_like(y_valid)

            line = pg.ErrorBarItem(
                x=np.log10(x_valid), y=y_valid, top=dy_valid, bottom=dy_valid, pen=pen
            )
            pen_symbol = pg.mkPen(color=color, width=1)
            self.pg_g2map_profile.plot(
                x_valid,
                y_valid,
                pen=None,
                symbol="o",
                name=f"qbin={qbin}",
                symbolSize=3,
                symbolPen=pen_symbol,
                symbolBrush=pg.mkBrush(color=(*color, 0)),
            )
            self.pg_g2map_profile.setLogMode(x=True, y=None)
            self.pg_g2map_profile.addItem(line)
        except Exception as e:
            logger.warning(f"Failed to update G2 map profile: {e}")

    def plot_g2_map(self, dryrun=False):
        """Plot the G2 Map visualization."""
        rows = self.get_selected_rows()
        qbin = (
            self.spinBox_g2map_qbin.value()
            if hasattr(self, "spinBox_g2map_qbin")
            else 0
        )
        normalize = (
            self.checkBox_g2map_normalize.isChecked()
            if hasattr(self, "checkBox_g2map_normalize")
            else False
        )
        g2map_cmap = (
            self.cb_g2map_cmap.currentText()
            if hasattr(self, "cb_g2map_cmap")
            else "jet"
        )
        g2map_scale = (
            self.cb_g2map_scale.currentText()
            if hasattr(self, "cb_g2map_scale")
            else "linear"
        )
        qmap_cmap = (
            self.cb_g2map_qmap_cmap.currentText()
            if hasattr(self, "cb_g2map_qmap_cmap")
            else "tab20b"
        )

        kwargs = {
            "rows": rows,
            "qbin": qbin,
            "normalize": normalize,
            "g2map_cmap": g2map_cmap,
            "g2map_scale": g2map_scale,
            "qmap_cmap": qmap_cmap,
        }

        if dryrun:
            return kwargs

        if self.vk is None:
            self.statusbar.showMessage("No data available for G2 map")
            return None

        if not rows:
            self.statusbar.showMessage("Please select files to plot G2 map")
            return None

        try:
            self.vk.plot_g2map(
                g2map_hdl=self.pg_g2map,
                qmap_hdl=self.pg_g2map_qmap,
                g2_hdl=self.pg_g2map_profile,
                rows=rows,
                qbin=qbin,
                normalization=normalize,
                g2map_cmap=g2map_cmap,
                g2map_scale=g2map_scale,
                qmap_cmap=qmap_cmap,
            )
            self.statusbar.showMessage("G2 map updated successfully")
        except Exception as e:
            logger.error(f"Error plotting G2 map: {e}")
            self.statusbar.showMessage(f"Error plotting G2 map: {e}")
        return None

    def plot_qmap(self, dryrun=False):
        kwargs = {
            "rows": self.get_selected_rows(),
            "target": self.comboBox_qmap_target.currentText(),
            "cmap": self.cb_qmap_cmap.currentText(),
        }
        if not dryrun and not self._guard_no_data("qmap"):
            return None
        if dryrun:
            return kwargs
        self.vk.plot_qmap(self.pg_qmap, **kwargs)
        return None

    def plot_twotime(self, dryrun=False, highlight_xy=None):
        current_selection = self.comboBox_twotime_selection.currentIndex()
        fname = self._current_twotime_fname()
        switched_dataset = fname is not None and fname != self._twotime_last_fname
        if switched_dataset:
            # New file: use its remembered qbin (default 0), not the old
            # combobox index left over from the previous file.
            current_selection = self._twotime_qbin_memory.get(fname, 0)
        logger.debug(
            f"plot_twotime called: dryrun={dryrun}, ComboBox currentIndex={current_selection}, count={self.comboBox_twotime_selection.count()}"
        )

        kwargs = {
            "rows": self.get_selected_rows(),
            "auto_crop": self.twotime_autocrop.isChecked(),
            "highlight_xy": highlight_xy,
            "cmap": self.cb_twotime_cmap.currentText(),
            "vmin": self.c2_min.value(),
            "vmax": self.c2_max.value(),
            "correct_diag": self.twotime_correct_diag.isChecked(),
            "autolevel": self.checkBox_twotime_autolevel.isChecked(),
            "selection": max(0, current_selection),
        }

        logger.debug(f"Twotime plot kwargs: {kwargs}")
        if dryrun:
            return kwargs

        if not self._guard_no_data("twotime"):
            return None

        if not self.mp_2t_hdls:
            self.init_twotime_plot_handler()
        new_labels = self.vk.plot_twotime(self.mp_2t_hdls, **kwargs)
        if highlight_xy is not None:
            # The renderer plotted the bin under the click; make the controls and
            # the remembered selection show that bin, not the previous one.
            xf_list = self.vk.get_xf_list(kwargs["rows"], filter_atype="Twotime")
            if xf_list:
                resolved = xf_list[0].get_twotime_maps(
                    auto_crop=kwargs["auto_crop"],
                    highlight_xy=highlight_xy,
                    selection=current_selection,
                )[2]
                if resolved is not None:
                    current_selection = int(resolved)
                    if fname is not None:
                        _remember_twotime_qbin(
                            self._twotime_qbin_memory, fname, current_selection
                        )
        if new_labels is not None:
            logger.debug(f"Repopulating ComboBox with new labels: {new_labels}")
            # current_selection already holds the right target: the
            # remembered qbin for this file on a dataset switch, or the
            # combobox's current index when replotting the same file.
            logger.debug(
                f"COMBOBOX POPULATE: [SYNC] Current selection before repopulation: {current_selection}"
            )
            logger.debug(
                f"COMBOBOX POPULATE: [SYNC] Adding {len(new_labels)} items to ComboBox: {new_labels[:5]}..."
            )  # Show first 5 items

            # Block signals to prevent recursive updates
            self.comboBox_twotime_selection.blockSignals(True)
            self.horizontalSlider_twotime_selection.blockSignals(True)

            try:
                self.comboBox_twotime_selection.clear()
                self.comboBox_twotime_selection.addItems(new_labels)
                self.horizontalSlider_twotime_selection.setMaximum(len(new_labels) - 1)

                # Restore selection if it's valid
                if 0 <= current_selection < len(new_labels):
                    self.comboBox_twotime_selection.setCurrentIndex(current_selection)
                    self.horizontalSlider_twotime_selection.setValue(current_selection)
                    logger.debug(f"Restored selection to: {current_selection}")
                else:
                    logger.debug(
                        f"Selection {current_selection} is invalid for {len(new_labels)} items, defaulting to 0"
                    )
            finally:
                # Always restore signal connections
                self.comboBox_twotime_selection.blockSignals(False)
                self.horizontalSlider_twotime_selection.blockSignals(False)
        if fname is not None:
            self._twotime_last_fname = fname
        return None

    def show_dataset(self):
        if self.vk is None:
            return
        rows = self.get_selected_rows()
        self.tree = self.vk.get_pg_tree(rows)
        if self.tree:
            self.tree.show()

    def plot_stability(self, dryrun=False):
        kwargs = {
            "plot_type": self.cb_stab_type.currentIndex(),
            "plot_norm": self.cb_stab_norm.currentIndex(),
            "rows": self.get_selected_rows(),
            "loc": self.stab_legend_loc.currentText(),
        }
        if not dryrun and not self._guard_no_data("stability"):
            return None
        if dryrun:
            return kwargs
        if self.vk:
            self.vk.plot_stability(self.mp_stab, **kwargs)
        return None

    def plot_intensity_t(self, dryrun=False):
        kwargs = {
            "sampling": max(1, self.sb_intt_sampling.value()),
            "window": self.sb_window.value(),
            "rows": self.get_selected_rows(),
            "xlabel": self.intt_xlabel.currentText(),
        }
        if not dryrun and not self._guard_no_data("intensity_t"):
            return None
        if dryrun:
            return kwargs
        self.vk.plot_intt(self.pg_intt, **kwargs)
        return None

    def init_diffusion(self):
        logger.info("init_diffusion called - updating left window (mp_tauq_pre)")
        rows = self.get_selected_rows()
        logger.info(f"Selected rows for pre-plot: {rows}")

        # Debug mp_tauq_pre widget state
        logger.info(
            f"mp_tauq_pre visible: {self.mp_tauq_pre.isVisible()}, size: {self.mp_tauq_pre.size()}"
        )
        logger.info(f"mp_tauq_pre canvas: {self.mp_tauq_pre.hdl}")

        self.vk.plot_tauq_pre(hdl=self.mp_tauq_pre.hdl, rows=rows)

        logger.info("init_diffusion completed")

    def plot_diffusion(self, dryrun=False):
        logger.info("plot_diffusion method called")

        try:
            keys = [self.tauq_amin, self.tauq_bmin, self.tauq_amax, self.tauq_bmax]
            bounds = np.array([float(x.text()) for x in keys]).reshape(2, 2)

            fit_flag = [self.tauq_afit.isChecked(), self.tauq_bfit.isChecked()]

            if sum(fit_flag) == 0:
                logger.warning("No fit flags selected")
                self.statusbar.showMessage("nothing to fit, really?", 1000)
                return {} if dryrun else None

            if not self._guard_no_data("diffusion"):
                return {} if dryrun else None

            tauq = [self.tauq_qmin, self.tauq_qmax]
            q_range = [float(x.text()) for x in tauq]

            kwargs = {
                "bounds": bounds.tolist(),
                "fit_flag": fit_flag,
                "offset": self.sb_tauq_offset.value(),
                "rows": self.get_selected_rows(),
                "q_range": q_range,
                "plot_type": self.cb_tauq_type.currentIndex(),
            }

            if dryrun:
                return kwargs

            # CRITICAL FIX: Force fresh G2 fitting first, then tau-q fitting
            # This ensures tau-q gets fresh G2 results as input
            rows = self.get_selected_rows()
            xf_list = self.vk.get_xf_list(rows)

            # Force fresh G2 fitting for all files to ensure fresh input data
            for xf in xf_list:
                if hasattr(xf, "fit_summary") and xf.fit_summary is not None:
                    # Get current G2 parameters from UI
                    g2_bounds, g2_fit_flag, g2_fit_func = self.check_g2_fitting_number()
                    g2_params = self.check_g2_number()
                    g2_q_range = (g2_params[0], g2_params[1])
                    g2_t_range = (g2_params[2], g2_params[3])

                    # Force fresh G2 fitting to provide fresh input for tau-q
                    logger.info(
                        f"Forcing fresh G2 fitting for {xf.label} before tau-q analysis"
                    )
                    xf.fit_g2(
                        g2_q_range,
                        g2_t_range,
                        g2_bounds,
                        g2_fit_flag,
                        g2_fit_func,
                        force_refit=True,
                    )

            # Now force fresh tau-q fitting with the fresh G2 results
            kwargs["force_refit"] = True
            msg = self.vk.plot_tauq(hdl=self.mp_tauq.hdl, **kwargs)

            # Update BOTH display areas: right panel AND left panel
            self.tauq_msg.clear()
            self.tauq_msg.setData(msg)
            self.tauq_msg.parent().repaint()

            # ALSO update left panel parameter plots with fresh data
            logger.info("Updating diffusion parameter plots with fresh results")
            self.init_diffusion()

            logger.info("plot_diffusion completed successfully")
            return None

        except Exception as e:
            logger.exception(f"Error in plot_diffusion: {e}")
            return None

    def select_bkgfile(self):
        if self.vk is None:
            return
        path = self.work_dir.text()
        f = QtWidgets.QFileDialog.getOpenFileName(
            self, caption="select the file for background subtraction", dir=path
        )[0]
        if os.path.isfile(f):
            self.le_bkg_fname.setText(f)
            self.vk.select_bkgfile(f)
        else:
            return

    def remove_avg_job(self):
        if self.vk is None or self.vk.avg_worker is None:
            return
        self.vk.remove_job()

    def set_average_save_path(self):
        save_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open directory")
        self.avg_save_path.clear()
        self.avg_save_path.setText(save_path)

    def set_average_save_name(self):
        save_name = QtWidgets.QFileDialog.getSaveFileName(self, "Save as")
        self.avg_save_name.clear()
        self.avg_save_name.setText(os.path.basename(save_name[0]))

    def init_average(self):
        if self.vk is None:
            return
        if len(self.vk.target) > 0:
            save_path = self.avg_save_path.text()
            if save_path == "":
                self.avg_save_path.setText(self.work_dir.text())
            else:
                logger.info("use the previous save path")

            # Only set the default save_name when the field is currently
            # empty.  If the user has already typed a name it must be
            # preserved so that repeated clicks on "init" do not clobber
            # the user's entry. (BUG-016)
            save_name = self.avg_save_name.text()
            if save_name == "":
                save_name = "Avg" + self.vk.target[0]
                self.avg_save_name.setText(save_name)

    def submit_job(self):
        if self.vk is None:
            self.statusbar.showMessage("No directory loaded", 2000)
            return
        if len(self.vk.target) < MIN_AVERAGING_FILES:
            self.statusbar.showMessage("select at least 2 files for averaging", 1000)
            return

        self.thread_pool.setMaxThreadCount(self.max_thread_count.value())

        save_path = self.avg_save_path.text()
        save_name = self.avg_save_name.text()

        if not os.path.isdir(save_path):
            logger.info("the average save_path doesn't exist; creating one")
            try:
                os.mkdir(save_path)
            except OSError as e:
                logger.error("cannot create the folder %s: %s", save_path, e)
                return

        avg_fields = []
        if self.bx_avg_G2IPIF.isChecked():
            avg_fields.extend(["G2", "IP", "IF"])
        if self.bx_avg_g2g2err.isChecked():
            avg_fields.extend(["g2", "g2_err"])
        if self.bx_avg_saxs.isChecked():
            avg_fields.extend(["saxs_1d", "saxs_2d"])

        if len(avg_fields) == 0:
            self.statusbar.showMessage("No average field is selected. quit", 1000)
            return

        kwargs = {
            "save_path": os.path.join(save_path, save_name),
            "chunk_size": int(self.cb_avg_chunk_size.currentText()),
            "avg_blmin": self.avg_blmin.value(),
            "avg_blmax": self.avg_blmax.value(),
            "avg_qindex": self.avg_qindex.value(),
            "avg_window": self.avg_window.value(),
            "fields": avg_fields,
        }

        if kwargs["avg_blmax"] <= kwargs["avg_blmin"]:
            self.statusbar.showMessage("check avg min/max values.", 1000)
            return

        self._reset_g2_fit_bounds()
        self.vk.submit_job(**kwargs)
        if not self.vk.target:
            self._close_g2_regroup()
        self.update_box(self.vk.target, mode="target")

    def update_avg_info(self):
        if self.vk is None:
            return
        worker = self.vk.avg_worker
        if worker is None:
            self.statusbar.showMessage("no averaging job exists", 1000)
            return

        self.timer.stop()
        self.timer.blockSignals(True)
        try:
            self.timer.timeout.disconnect()
        except (RuntimeError, TypeError):
            pass
        self.timer.blockSignals(False)

        self.timer.setInterval(1000)
        worker.initialize_plot(self.mp_avg_g2)
        self.timer.timeout.connect(self._poll_avg_job)
        self.timer.start()

    def _poll_avg_job(self) -> None:
        """1 Hz tick: refresh the plot, stop on a terminal job state."""
        if self.vk is None or self.vk.avg_worker is None:
            self.timer.stop()
            return
        self.vk.update_avg_info()
        worker = self.vk.avg_worker
        if worker.status in ("finished", "killed"):
            self.timer.stop()
            if worker.status == "finished" and worker.n_accepted == 0:
                self.statusbar.showMessage(
                    "Averaging finished: no valid dataset; no output written", 8000
                )

    def start_avg_job(self):
        if self.vk is None:
            return
        worker = self.vk.avg_worker
        if worker is None:
            self.statusbar.showMessage("no averaging job exists", 1000)
            return
        # Only a fresh job may start: a queued job started twice segfaults, and
        # a run worker is auto-deleted, so restarting it raises.
        if worker.status != "wait":
            self.statusbar.showMessage(f"this job is {worker.status}", 1000)
            return
        worker.status = "queued"  # do_average sets "running" when it starts

        # Disconnect any previously connected slot before reconnecting (BUG-037):
        # prevents lambda slot accumulation on repeated calls.
        try:
            worker.signals.values.disconnect(self.vk.update_avg_values)
        except (RuntimeError, TypeError):
            pass  # Signal was not connected; safe to ignore
        worker.signals.values.connect(self.vk.update_avg_values)
        self.thread_pool.start(worker)
        self.vk.avg_worker_active[worker.jid] = None
        self.update_avg_info()

    def avg_kill_job(self):
        if self.vk is None:
            return
        worker = self.vk.avg_worker
        if worker is None:
            self.statusbar.showMessage("no averaging job exists", 1000)
            return
        if worker.status not in ("queued", "running"):
            self.statusbar.showMessage("the selected job isn't running", 1000)
            return
        # The 1 Hz poll stops itself once the worker reports "killed" (a queued
        # job only gets there when the pool starts it), so keep it running.
        worker.kill()

    def show_g2_fit_summary_func(self):
        rows = self.get_selected_rows()
        self.tree = self.vk.get_fitting_tree(rows)
        self.tree.show()

    def show_avg_jobinfo(self):
        if self.vk is None:
            return
        worker = self.vk.avg_worker
        if worker is None:
            logger.info("no averaging job to show info for")
            return
        self.tree = worker.get_pg_tree()
        self.tree.show()

    def _initialize_g2_fit_bounds(self, extent: tuple[float, float] | None) -> None:
        if not self._g2_fit_bounds_initialized and extent is not None:
            t_min, t_max = extent
            # Mirror the spin boxes' rounding and clamping to spot inverted bounds.
            lo, hi = (
                min(max(round(value, box.decimals()), box.minimum()), box.maximum())
                for box, value in (
                    (self.g2_bmin, t_min / 20),
                    (self.g2_bmax, t_max * 10),
                )
            )
            if hi <= lo:
                logger.warning(
                    "G2 fit bounds not initialised: delays %s..%s give tau bounds "
                    "%s..%s after spin-box rounding",
                    t_min,
                    t_max,
                    lo,
                    hi,
                )
                return
            self._g2_fit_bounds_initialized = True
            self.g2_bmin.setValue(lo)
            self.g2_bmax.setValue(hi)

    def init_g2(self, qd_per_file, tel):
        from .module import g2mod

        if qd_per_file is None or tel is None:
            return
        if len(qd_per_file) == 0 or len(tel) == 0:
            return

        # Update Bayesian Q-index spinbox range.
        # qd_per_file is a list of per-file Q-arrays; use the first file's array length
        # to get the actual number of Q-bins.
        if hasattr(self, "sb_g2_bayesian_qidx") and len(qd_per_file) > 0:
            num_qbins = len(qd_per_file[0])
            self.sb_g2_bayesian_qidx.setMaximum(max(num_qbins - 1, 0))

        # Store grid metadata for Q-bin subplot highlighting
        self._g2_fitting_num_col = self.sb_g2_column.value()
        self._g2_fitting_plot_type = self.g2_plot_type.currentText()

        q_auto = self.g2_qauto.isChecked()
        t_auto = self.g2_tauto.isChecked()

        extent = g2mod.get_positive_delay_extent(tel)
        self._initialize_g2_fit_bounds(extent)

        if t_auto and extent is not None:
            t_min, t_max = extent
            self.g2_tmin.setText(f"{t_min / 1.1:.2e}")
            self.g2_tmax.setText(f"{t_max * 1.1:.2e}")

        if q_auto:
            # qd_per_file arrays may be ragged (different Q-bin counts); reduce
            # per file so np.min/np.max don't choke on a ragged object array.
            q_min = min(np.min(q) for q in qd_per_file)
            q_max = max(np.max(q) for q in qd_per_file)
            self.g2_qmin.setValue(q_min / 1.1)
            self.g2_qmax.setValue(q_max * 1.1)

    def plot_g2(self, dryrun=False):
        """
        Plot G2 correlation functions with optional fitting.

        Generates multi-tau correlation function plots for selected files
        with configurable parameters including Q-range, time-range,
        fitting functions, and display options.

        Parameters
        ----------
        dryrun : bool, optional
            If True, returns parameters without plotting (default: False).

        Returns
        -------
        dict or tuple
            If dryrun=True, returns plot parameters dictionary.
            If dryrun=False, returns (q_values, time_delays) tuple.

        Notes
        -----
        - Supports single and double exponential fitting
        - Automatically initializes fitting parameters based on data range
        - Updates diffusion analysis tab when fitting is enabled
        - Validates fitting parameters and bounds before processing
        """
        p = self.check_g2_number()
        bounds, fit_flag, fit_func = self.check_g2_fitting_number()

        kwargs = {
            "num_col": self.sb_g2_column.value(),
            "offset": self.sb_g2_offset.value(),
            "show_fit": False,  # G2 tab shows raw data only (fitting is on g2 fitting tab)
            "show_label": self.g2_show_label.isChecked(),
            "plot_type": self.g2_plot_type.currentText(),
            "q_range": (p[0], p[1]),
            "t_range": (p[2], p[3]),
            "y_range": (p[4], p[5]),
            "y_auto": self.g2_yauto.isChecked(),
            "q_auto": self.g2_qauto.isChecked(),
            "t_auto": self.g2_tauto.isChecked(),
            "rows": self.get_selected_rows(),
            "bounds": bounds,
            "fit_flag": fit_flag,
            "marker_size": self.g2_marker_size.value(),
            "subtract_baseline": self.g2_sub_baseline.isChecked(),
            "fit_func": fit_func,
            "robust_fitting": True,  # Always use robust sequential fitting for better reliability
        }
        if dryrun:
            return kwargs
        self.pushButton_4.setDisabled(True)
        self.pushButton_4.setText("plotting")
        try:
            qd_per_file, tel = self.vk.plot_g2(handler=self.mp_g2, **kwargs)
            self.init_g2(qd_per_file, tel)
            # Note: Fitting/diffusion is handled in the g2 fitting tab now
        except Exception:
            logger.exception("Error in plot_g2")
        finally:
            self.pushButton_4.setEnabled(True)
            self.pushButton_4.setText("plot")

    def refit_g2(self):
        """
        Refit G2 correlation functions with force_refit=True to bypass cache.

        This method forces a complete refitting of G2 data, ensuring that
        parameter values are updated even if fitting parameters haven't changed.
        Plots to the g2 fitting tab.
        """
        logger.info("Fit G2 button clicked - forcing new fit calculation")

        p = self.check_g2_number()
        bounds, fit_flag, fit_func = self.check_g2_fitting_number()

        kwargs = {
            "num_col": self.sb_g2_column.value(),
            "offset": self.sb_g2_offset.value(),
            "show_fit": True,  # Always show fit when refitting
            "show_label": self.g2_show_label.isChecked(),
            "plot_type": self.g2_plot_type.currentText(),
            "q_range": (p[0], p[1]),
            "t_range": (p[2], p[3]),
            "y_range": (p[4], p[5]),
            "y_auto": self.g2_yauto.isChecked(),
            "q_auto": self.g2_qauto.isChecked(),
            "t_auto": self.g2_tauto.isChecked(),
            "rows": self.get_selected_rows(),
            "bounds": bounds,
            "fit_flag": fit_flag,
            "marker_size": self.g2_marker_size.value(),
            "subtract_baseline": self.g2_sub_baseline.isChecked(),
            "fit_func": fit_func,
            "robust_fitting": True,
            "force_refit": True,  # KEY: Force refitting to bypass cache
        }

        if sum(kwargs["fit_flag"]) == 0:
            self.statusbar.showMessage("No fitting parameters enabled", 1000)
            return

        self.btn_g2_refit.setDisabled(True)
        self.btn_g2_refit.setText("fitting...")
        try:
            qd_per_file, tel = self.vk.plot_g2(handler=self.mp_g2_fitting, **kwargs)
            self.init_g2(qd_per_file, tel)
            self.init_diffusion()
            self.statusbar.showMessage("G2 fitting completed successfully", 2000)
        except Exception:
            logger.exception("Error in refit_g2")
            self.statusbar.showMessage("G2 fitting failed", 2000)
        finally:
            self.btn_g2_refit.setEnabled(True)
            self.btn_g2_refit.setText("Fit")

    def plot_g2_fitting(self, dryrun=False):
        """
        Plot G2 correlation functions with fitting overlay on the g2 fitting tab.

        This is similar to plot_g2 but:
        - Always shows fitting results
        - Plots to the mp_g2_fitting widget

        Parameters
        ----------
        dryrun : bool, optional
            If True, returns parameters without plotting (default: False).

        Returns
        -------
        dict or tuple
            If dryrun=True, returns plot parameters dictionary.
            If dryrun=False, returns (q_values, time_delays) tuple.
        """
        p = self.check_g2_number()
        bounds, fit_flag, fit_func = self.check_g2_fitting_number()

        kwargs = {
            "num_col": self.sb_g2_column.value(),
            "offset": self.sb_g2_offset.value(),
            "show_fit": True,  # Always show fit on the fitting tab
            "show_label": self.g2_show_label.isChecked(),
            "plot_type": self.g2_plot_type.currentText(),
            "q_range": (p[0], p[1]),
            "t_range": (p[2], p[3]),
            "y_range": (p[4], p[5]),
            "y_auto": self.g2_yauto.isChecked(),
            "q_auto": self.g2_qauto.isChecked(),
            "t_auto": self.g2_tauto.isChecked(),
            "rows": self.get_selected_rows(),
            "bounds": bounds,
            "fit_flag": fit_flag,
            "marker_size": self.g2_marker_size.value(),
            "subtract_baseline": self.g2_sub_baseline.isChecked(),
            "fit_func": fit_func,
            "robust_fitting": True,
        }

        if sum(kwargs["fit_flag"]) == 0:
            self.statusbar.showMessage("No fitting parameters enabled", 1000)
            return None

        if dryrun:
            return kwargs

        try:
            qd_per_file, tel = self.vk.plot_g2(handler=self.mp_g2_fitting, **kwargs)
            self.init_g2(qd_per_file, tel)
            self.init_diffusion()
        except Exception:
            logger.exception("Error in plot_g2_fitting")

    def export_g2(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, caption="select a folder to export G2 data and fitting results"
        )

        if folder in [None, ""]:
            return

        try:
            rows = self.get_selected_rows()
            logger.info(f"G2 export: Selected rows: {rows}")

            # Check if there are any files in the target list at all
            all_available_files = self.vk.get_xf_list()
            logger.info(
                f"G2 export: Total files available in target list: {len(all_available_files)}"
            )

            if not rows:
                if not all_available_files:
                    self.statusbar.showMessage(
                        "No files loaded. Please add XPCS files to the target list first.",
                        5000,
                    )
                    logger.warning(
                        "G2 export: No files in target list. User needs to load XPCS files first."
                    )
                else:
                    # Show helpful message about file selection
                    multitau_available = self.vk.get_xf_list(filter_atype="Multitau")
                    if multitau_available:
                        self.statusbar.showMessage(
                            f"Please select files for export. {len(multitau_available)} Multitau files available in target list.",
                            5000,
                        )
                        logger.warning(
                            f"G2 export: No files selected. {len(multitau_available)} Multitau files available for export."
                        )
                    else:
                        self.statusbar.showMessage(
                            "No Multitau files found. G2 export requires Multitau analysis files.",
                            5000,
                        )
                        logger.warning(
                            f"G2 export: No Multitau files available. Found {len(all_available_files)} files but none are Multitau type."
                        )
                return

            # Additional diagnostics
            all_files = self.vk.get_xf_list(rows=rows)
            multitau_files = self.vk.get_xf_list(rows=rows, filter_atype="Multitau")
            logger.info(f"G2 export: Total files from selection: {len(all_files)}")
            logger.info(
                f"G2 export: Multitau files from selection: {len(multitau_files)}"
            )
            for i, xf in enumerate(all_files):
                logger.info(
                    f"G2 export: File {i}: {xf.label} - atype: {getattr(xf, 'atype', 'unknown')}"
                )

            # Check if selected files include any Multitau files
            if not multitau_files:
                # Check if there are any Multitau files available but not selected
                all_multitau_available = self.vk.get_xf_list(filter_atype="Multitau")
                if all_multitau_available:
                    self.statusbar.showMessage(
                        f"Selected files are not Multitau type. Please select from {len(all_multitau_available)} available Multitau files.",  # nosec B608
                        5000,
                    )
                    logger.warning(
                        f"G2 export: Selected {len(all_files)} files but none are Multitau. {len(all_multitau_available)} Multitau files available."
                    )
                else:
                    self.statusbar.showMessage(
                        "No Multitau files available for G2 export.", 5000
                    )
                    logger.warning(
                        "G2 export: No Multitau files available in entire target list."
                    )
                return

            self.vk.export_g2(folder, rows)
            self.statusbar.showMessage(f"G2 data exported to {folder}", 5000)
        except ValueError as e:
            # Handle data validation errors with specific guidance
            if "No files with G2 data found" in str(e):
                self.statusbar.showMessage(
                    "No files with G2 correlation data found. Please select multitau analysis files.",
                    5000,
                )
            elif "No valid Q indices" in str(e):
                self.statusbar.showMessage(
                    "Export failed: Q-range selection resulted in invalid indices. Try adjusting Q-range.",
                    5000,
                )
            else:
                self.statusbar.showMessage(f"Data validation error: {e}", 5000)
            logger.error(f"G2 export validation error: {e}")
        except (OSError, PermissionError) as e:
            # Handle file system errors
            self.statusbar.showMessage(
                f"File system error: Cannot write to {folder}. Check permissions.", 5000
            )
            logger.error(f"G2 export file system error: {e}")
        except Exception as e:
            # Handle unexpected errors
            self.statusbar.showMessage(
                f"Export failed with unexpected error: {type(e).__name__}: {e}", 5000
            )
            logger.error(f"G2 export unexpected error: {e}", exc_info=True)

    def export_diffusion(self):
        """Export power law fitting results from diffusion analysis."""
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, caption="Select folder to export diffusion fitting results"
        )

        if folder in [None, ""]:
            return

        try:
            rows = self.get_selected_rows()
            if not rows:
                self.statusbar.showMessage("No files selected for export", 3000)
                return

            self.vk.export_diffusion(folder, rows)
            self.statusbar.showMessage(
                f"Diffusion fitting results exported to {folder}", 5000
            )
        except ValueError as e:
            # Handle data validation errors with specific guidance
            if "No files with tau-q fitting results found" in str(e):
                self.statusbar.showMessage(
                    "No files with tau-q fitting results found. Please select multitau analysis files.",
                    5000,
                )
            elif "No files with tau-q power law fitting results found" in str(e):
                self.statusbar.showMessage(
                    "No diffusion analysis results found. Please perform diffusion analysis first in the Diffusion tab.",
                    5000,
                )
            else:
                self.statusbar.showMessage(f"Data validation error: {e}", 5000)
            logger.error(f"Diffusion export validation error: {e}")
        except (OSError, PermissionError) as e:
            # Handle file system errors
            self.statusbar.showMessage(
                f"File system error: Cannot write to {folder}. Check permissions.", 5000
            )
            logger.error(f"Diffusion export file system error: {e}")
        except Exception as e:
            # Handle unexpected errors
            self.statusbar.showMessage(
                f"Export failed with unexpected error: {type(e).__name__}: {e}", 5000
            )
            logger.error(f"Diffusion export unexpected error: {e}", exc_info=True)

    def reload_source(self):
        logger.debug("reload_source: starting reload")
        self.pushButton_11.setText("loading")
        self.pushButton_11.setDisabled(True)
        self.pushButton_11.parent().repaint()

        try:
            path = self.work_dir.text() if hasattr(self, "work_dir") else ""
            if not path or not os.path.isdir(path):
                self.statusbar.showMessage(
                    "Please select a valid data folder before reloading.", 5000
                )
                logger.warning("Reload requested with invalid path: %r", path)
                return
            if self.vk is None:
                self.statusbar.showMessage(
                    "No data folder loaded. Use File -> Open Directory.", 5000
                )
                logger.warning("Reload requested before viewer kernel initialized")
                return

            if os.path.abspath(path) != os.path.abspath(self.vk.path):
                self.load_path(path)
                return

            logger.debug(f"reload_source: building from {sanitize_path(path)}")
            built = self.vk.build(path=path, sort_method=self.sort_method.currentText())
            if not built:
                self.statusbar.showMessage("Reload failed: invalid data folder.", 5000)
                logger.warning("reload_source: build failed for path")
                return

            self.update_box(self.vk.source, mode="source")
            self.apply_filter_to_source()
            logger.info(f"reload_source: completed, {len(self.vk.source)} files loaded")
            self._on_reloaded_files()
        finally:
            self.pushButton_11.setText("reload")
            self.pushButton_11.setEnabled(True)
            self.pushButton_11.parent().repaint()

    def _on_reloaded_files(self) -> None:
        """React to Reload: everything shown or stored for a target file whose
        revision changed on disk belongs to the old revision and is dropped."""
        replaced = self.vk.replaced_paths
        changed = [
            n
            for n in self.vk.target
            if os.path.normpath(os.path.join(self.vk.path, n)) in replaced
        ]
        bkg_replaced = bool(getattr(self.vk, "background_replaced", False))
        if bkg_replaced and self.vk.meta.get("saxs1d_bkg_path") is None:
            # the kernel dropped a vanished background: do not show its path
            self.le_bkg_fname.clear()
        if not self.vk.target:
            return
        # Bundles of any other stale revision (file rewritten, then read again
        # before this reload) go too.
        evicted = self._evict_stale_bayesian()
        if not changed and not evicted:
            if bkg_replaced:
                # SAXS-1D's request carries the background revision, so it is redrawn
                self.update_plot()
            return
        # The loaded files' own display partition died with their old objects.
        self._display_partition_version += 1
        self._invalidate_plots()
        if changed and hasattr(self, "mp_g2_fitting"):
            self.mp_g2_fitting.clear()  # a fit of the old revision (view-only tab)
        self.update_plot()

    def load_path(self, path=None, debug=False):
        """
        Load XPCS data from specified directory path.

        This method initializes the viewer kernel with the specified path,
        builds the file list, and sets up the GUI for data analysis.

        Parameters
        ----------
        path : str, optional
            Directory path containing XPCS data files. If None, opens file dialog.
        debug : bool, optional
            Enable debug mode for additional logging (default: False).

        Notes
        -----
        - Creates new ViewerKernel instance or resets existing one
        - Initializes async kernel for background processing
        - Updates source and target file lists in GUI
        - Sets up averaging job table model
        """
        logger.debug(
            f"load_path called with path={sanitize_path(path) if path else None}"
        )

        if path in [None, False]:
            logger.debug("Opening directory selection dialog")
            # DontUseNativeDialog is used so files are shown along with dirs;
            folder = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                "Open directory",
                self.start_wd,
                QtWidgets.QFileDialog.DontUseNativeDialog,
            )
            if folder:
                logger.debug(f"User selected directory: {sanitize_path(folder)}")
            else:
                logger.debug("Directory selection cancelled")
                return
        else:
            folder = path

        if not os.path.isdir(folder):
            logger.warning(f"Invalid directory: {sanitize_path(folder)}")
            self.statusbar.showMessage(f"{folder} is not a folder.")
            folder = self.start_wd

        # Normalize to an absolute path so vk.path, work_dir, and any persisted
        # session paths survive a restart from a different working directory.
        folder = os.path.abspath(folder)
        self._close_g2_regroup()
        self._reset_g2_fit_bounds()
        # New directory: no in-flight plot (any tab) may repaint the new workspace.
        self._retire_plot_operations()
        self.work_dir.setText(folder)
        logger.info(f"Loading data from: {sanitize_path(folder)}")

        if self.vk is None:
            logger.debug("Creating new ViewerKernel")
            self.vk = ViewerKernel(folder, self.statusbar)
        else:
            logger.debug("Resetting existing ViewerKernel")
            self.vk.set_path(folder)
            self.vk.clear()
            # clear() only wipes source/source_search; reset_kernel() also clears
            # the stale target list, cached XpcsFile objects and metadata from the
            # previous directory so the new dataset starts from a clean state.
            self.vk.reset_kernel()

        # Initialize async kernel after viewer kernel is ready
        self.init_async_kernel()

        self.reload_source()
        self.source_model = self.vk.source
        self.update_box(self.vk.source, mode="source")

        # Update recent paths and menu
        self.recent_paths_manager.add_path(folder)
        self._populate_recent_directories_menu()

        logger.debug(
            f"Loaded {len(self.vk.source) if self.vk.source else 0} files from directory"
        )

        # Reset per-tab plot kwargs so stale parameters from a previous dataset
        # do not leak into the new session.
        self.plot_kwargs_record = {v: {} for _, v in tab_mapping.items()}

        self._reset_bayesian_state()

        # Trigger plot update to show proper empty states since no files are auto-added
        self.update_plot()

    def _reset_bayesian_state(self):
        """Forget every Bayesian result and in-flight fit of the old dataset.

        Running fits are asked to stop and their launch records are dropped,
        so a late completion callback no longer matches and is ignored.
        """
        # Stop the running batch BEFORE clearing the state its callbacks use.
        if self._g2_batch_coordinator is not None:
            self._cancel_g2_bayesian_batch()
        self._g2_batch_launch = None

        # Buttons are restored only for a fit that was actually running, so a
        # button disabled at start-up (NumPyro missing) stays disabled.
        g2_was_running = getattr(self, "_g2_bayesian_worker_active", False)
        diff_was_running = getattr(self, "_diff_bayesian_worker_active", False)
        for name in ("_g2_bayesian_worker", "_diff_bayesian_worker"):
            worker = getattr(self, name, None)
            if worker is not None and hasattr(worker, "cancel"):
                worker.cancel()
            setattr(self, name, None)
        self._g2_bayesian_worker_active = False
        self._diff_bayesian_worker_active = False
        self._g2_bayesian_launch = None
        self._diff_bayesian_launch = None
        self._g2_bayesian_store = {}
        self._diff_bayesian_store = {}
        self._close_bayesian_windows(everything=True)
        if g2_was_running:
            self.btn_g2_bayesian.setEnabled(True)
            self.btn_g2_bayesian.setText(f"Fit Q-{self.sb_g2_bayesian_qidx.value()}")
        if diff_was_running:
            self.btn_diff_bayesian.setEnabled(True)
            self.btn_diff_bayesian.setText("Fit Bayesian")
        if hasattr(self, "btn_g2_diagnosis"):
            self.btn_g2_diagnosis.setEnabled(False)
        if hasattr(self, "btn_diff_diagnosis"):
            self.btn_diff_diagnosis.setEnabled(False)
        if hasattr(self, "btn_g2_plot_all_q"):
            self.btn_g2_plot_all_q.setEnabled(False)

    # ------------------------------------------------------------------
    # Bayesian state follows the revision / membership of its files
    # ------------------------------------------------------------------

    def _rewire_without_args(self, signal, slot_name: str) -> None:
        """Replace ``signal -> self.<slot_name>`` by a zero-argument adapter."""
        try:
            signal.disconnect(getattr(self, slot_name))
        except (RuntimeError, TypeError):
            logger.warning("could not re-wire %s", slot_name, exc_info=True)
            return
        signal.connect(lambda: getattr(self, slot_name)())

    def _forget_window(self, attr: str, window) -> None:
        """Drop ``self.<attr>`` if it is still *window* (a destroyed old window
        must not clear the reference to the newer one that replaced it)."""
        if getattr(self, attr, None) is window:
            setattr(self, attr, None)
            if attr == "_bayesian_plot_dialog":
                self._bayesian_plot_snapshot = None

    def _close_bayesian_windows(self, *, everything=False, is_gone=None) -> int:
        """Close the diagnosis / All-Q windows whose OWN identity satisfies
        ``is_gone(file name, revision)`` (every one when *everything*).
        Returns how many were closed.

        The windows are asked, not the stores: a window keeps showing the
        bundle it was built from even after a refit replaced it in the store.
        """
        closed = 0
        for win_attr in ("_g2_diagnosis_window", "_diff_diagnosis_window"):
            win = getattr(self, win_attr, None)
            if win is not None and (
                everything or (is_gone is not None and win.matches(is_gone))
            ):
                # closed AND released: a hidden window would keep the evicted
                # posterior and its observations alive
                win.close()
                win.deleteLater()
                setattr(self, win_attr, None)
                closed += 1
        dialog = getattr(self, "_bayesian_plot_dialog", None)
        snapshot = getattr(self, "_bayesian_plot_snapshot", None)
        if dialog is not None and (
            everything
            or (
                is_gone is not None
                and snapshot is not None
                and is_gone(snapshot.fname, snapshot.revision)
            )
        ):
            dialog.close()
            self._forget_window("_bayesian_plot_dialog", dialog)
            closed += 1
        return closed

    def _drop_bayesian(self, is_gone) -> int:
        """Evict every Bayesian result of a ``(file name, revision)`` for which
        ``is_gone(fname, revision)`` holds: stored bundles, open windows that
        show such a file (asked directly), fits in flight (cancelled; their late completion no longer
        matches a launch and is ignored) and an All-Q dialog.  Returns how many
        things were dropped."""
        n = 0
        for store in (self._g2_bayesian_store, self._diff_bayesian_store):
            for key in [k for k in store if is_gone(k[0], k[1])]:
                del store[key]
                n += 1

        launch = self._g2_bayesian_launch
        if launch is not None and is_gone(
            launch.fname, self._g2_bayesian_launch_revision
        ):
            worker = self._g2_bayesian_worker
            if worker is not None and hasattr(worker, "cancel"):
                worker.cancel()
            self._g2_bayesian_worker = None
            self._g2_bayesian_launch = None
            self._g2_bayesian_worker_active = False
            self.btn_g2_bayesian.setEnabled(True)
            self.btn_g2_bayesian.setText(f"Fit Q-{self.sb_g2_bayesian_qidx.value()}")
            n += 1
        launch = self._diff_bayesian_launch
        if launch is not None and is_gone(
            launch.fname, self._diff_bayesian_launch_revision
        ):
            worker = self._diff_bayesian_worker
            if worker is not None and hasattr(worker, "cancel"):
                worker.cancel()
            self._diff_bayesian_worker = None
            self._diff_bayesian_launch = None
            self._diff_bayesian_worker_active = False
            self.btn_diff_bayesian.setEnabled(True)
            self.btn_diff_bayesian.setText("Fit Bayesian")
            n += 1
        batch = self._g2_batch_launch
        if batch is not None and is_gone(batch["fname"], batch.get("revision")):
            self._cancel_g2_bayesian_batch()
            self._g2_batch_launch = None  # late per-bin results are ignored
            n += 1

        n += self._close_bayesian_windows(is_gone=is_gone)

        if n:
            self._refresh_bayesian_buttons()
        return n

    def _refresh_bayesian_buttons(self) -> None:
        q_idx = self.sb_g2_bayesian_qidx.value()
        has_result = self._g2_current_bundle() is not None
        self.btn_g2_diagnosis.setEnabled(has_result)
        self.btn_g2_diagnosis.setText(
            f"Diagnosis Q-{q_idx}" if has_result else f"Diagnosis Q-{q_idx} (no fit)"
        )
        self.btn_diff_diagnosis.setEnabled(bool(self._diff_bayesian_store))

    def _evict_stale_bayesian(self) -> int:
        """Drop results whose file is no longer the revision they came from."""
        return self._drop_bayesian(lambda fname, rev: _revision_of(fname) != rev)

    def _sync_bayesian_to_targets(self) -> int:
        """Drop results of files that are no longer in the target list."""
        if self.vk is None or not hasattr(self, "_diff_bayesian_store"):
            return 0
        live = {os.path.normpath(os.path.join(self.vk.path, n)) for n in self.vk.target}
        return self._drop_bayesian(
            lambda fname, _rev: fname is None or os.path.normpath(fname) not in live
        )

    def update_box(self, file_list, mode="source"):
        if file_list is None:
            return
        if mode == "source":
            self.list_view_source.setModel(file_list)
            self.box_source.setTitle(f"Source: {len(file_list):5d}")
            self.box_source.parent().repaint()
            self.list_view_source.parent().repaint()
        elif mode == "target":
            self.list_view_target.setModel(file_list)
            # Track the installed model so session save can read the target list.
            self.target_model = file_list
            self.box_target.setTitle(f"Target: {len(file_list):5d}")
            # on macos, the target box doesn't seem to update; force it
            file_list.layoutChanged.emit()
            self.box_target.repaint()
            self.list_view_target.repaint()
            self._sync_bayesian_to_targets()
            max_size = len(file_list) - 1
            self.horizontalSlider_saxs2d_selection.setMaximum(max_size)
            self.horizontalSlider_saxs2d_selection.setValue(0)
            self.spinBox_saxs2d_selection.setMaximum(max_size)
            self.spinBox_saxs2d_selection.setValue(0)
        self.statusbar.showMessage("Target file list updated.", 1000)
        return

    def add_target(self):
        target = []
        for x in self.list_view_source.selectedIndexes():
            # in some cases, it will return None
            val = x.data()
            if val is not None:
                target.append(val)
        if target == []:
            logger.debug("add_target: No files selected in source list")
            return

        logger.debug(f"add_target: Adding {len(target)} files to target list")

        tab_id = self.tabWidget.currentIndex()
        tab_name = tab_mapping[tab_id]
        preload = tab_name != "average"
        self.vk.add_target(target, preload=preload)
        self.list_view_source.clearSelection()
        self.update_box(self.vk.target, mode="target")

        # Update tab availability based on file formats
        self.update_tab_availability()

        logger.debug(f"add_target: Target list now has {len(self.vk.target)} files")

        if tab_name == "average":
            self.init_average()
        else:
            self.update_plot()

    def update_tab_availability(self):
        """
        Update tab availability based on the file formats in the target list.

        Disables incompatible tabs to prevent application freezing:
        - Multitau only: Disables "Two Time" tab
        - Twotime only: Disables "g2", "g2 fitting", "g2 map", and "Diffusion" tabs
        - Both present (a combined "both" file or a multi-tau + two-time mix):
          Enables all tabs
        """
        if not self.vk.target or len(self.vk.target) == 0:
            # No files selected, enable all tabs
            self._enable_all_tabs()
            self.statusbar.showMessage("No files selected - all tabs available", 3000)
            return

        # Analyze file formats in target list
        file_formats = set()
        multitau_count = 0
        twotime_count = 0

        try:
            # Get XpcsFile objects to check their analysis types
            xf_list = self.vk.get_xf_list()
            if not xf_list:
                self._enable_all_tabs()
                return

            for xf in xf_list:
                if hasattr(xf, "atype") and xf.atype:
                    for atype in xf.atype:
                        file_formats.add(atype)
                        if atype == "Multitau":
                            multitau_count += 1
                        elif atype == "Twotime":
                            twotime_count += 1

            logger.debug(
                f"File format analysis: {file_formats}, Multitau: {multitau_count}, Twotime: {twotime_count}"
            )

            # Determine tab availability based on which data kinds are present
            # across the target list. A single "both" file (atype contains both
            # "Multitau" and "Twotime") or a mix of single-type files both make
            # all tabs valid; gating on presence, not the count of distinct
            # formats, keeps "both" from being mislabeled as an accidental mix.
            has_multitau = "Multitau" in file_formats
            has_twotime = "Twotime" in file_formats

            if has_multitau and has_twotime:
                # Combined multi-tau + two-time data: every tab is usable.
                self._enable_all_tabs()
                self.statusbar.showMessage(
                    f"Multitau + Two-time data detected - all tabs available "
                    f"(Multitau: {multitau_count}, Twotime: {twotime_count})",
                    5000,
                )
            elif has_multitau:
                self._configure_for_multitau()
                self.statusbar.showMessage(
                    f"Multitau format detected - Two Time tab disabled ({multitau_count} files)",
                    5000,
                )
            elif has_twotime:
                self._configure_for_twotime()
                self.statusbar.showMessage(
                    f"Twotime format detected - G2 and Diffusion tabs disabled ({twotime_count} files)",
                    5000,
                )
            else:
                # No recognized formats, enable all tabs
                self._enable_all_tabs()
                self.statusbar.showMessage(
                    "Warning: File format not recognized - proceed with caution", 5000
                )

        except Exception as e:
            logger.error(f"Error analyzing file formats for tab management: {e}")
            self._enable_all_tabs()
            self.statusbar.showMessage(
                "Error analyzing file formats - all tabs enabled", 5000
            )

    def _enable_all_tabs(self):
        """Enable all tabs."""
        for tab_index in range(self.tabWidget.count()):
            self.tabWidget.setTabEnabled(tab_index, True)
        self._clear_tab_tooltips()  # Clear tooltips when all tabs are enabled
        logger.debug("All tabs enabled")

    def _configure_for_multitau(self):
        """Configure tabs for multitau format: disable Two Time tab."""
        self._enable_all_tabs()  # First enable all
        self._clear_tab_tooltips()  # Clear any existing tooltips

        # Disable Two Time tab (index 8 based on tab_mapping)
        twotime_tab_index = 8
        self.tabWidget.setTabEnabled(twotime_tab_index, False)
        self.tabWidget.setTabToolTip(
            twotime_tab_index,
            "Two Time analysis not available for multi-tau format files",
        )

        # If user is currently on Two Time tab, switch to G2 tab
        if self.tabWidget.currentIndex() == twotime_tab_index:
            self.tabWidget.setCurrentIndex(4)  # Switch to G2 tab
            logger.info("Switched from Two Time to G2 tab (multi-tau format)")

        logger.debug("Configured tabs for multi-tau format: Two Time tab disabled")

    def _configure_for_twotime(self):
        """Configure tabs for twotime format: disable G2, G2 Fitting, G2 Map, and Diffusion tabs.

        Multi-tau related tabs (g2, g2_fitting, g2_map, diffusion) are grouped and disabled
        when two-time data is loaded since they require multi-tau correlation data.
        """
        self._enable_all_tabs()  # First enable all
        self._clear_tab_tooltips()  # Clear any existing tooltips

        # Disable multi-tau related tabs (based on tab_mapping indices)
        # These tabs require multi-tau correlation data which isn't available in two-time files
        g2_tab_index = 4
        g2_fitting_tab_index = 5
        g2_map_tab_index = 6
        diffusion_tab_index = 7
        twotime_tab_index = 8

        multitau_tabs = [
            g2_tab_index,
            g2_fitting_tab_index,
            g2_map_tab_index,
            diffusion_tab_index,
        ]

        for tab_index in multitau_tabs:
            self.tabWidget.setTabEnabled(tab_index, False)

        self.tabWidget.setTabToolTip(
            g2_tab_index,
            "G2 analysis requires multi-tau data (not available in two-time format files)",
        )
        self.tabWidget.setTabToolTip(
            g2_fitting_tab_index,
            "G2 Fitting requires multi-tau data (not available in two-time format files)",
        )
        self.tabWidget.setTabToolTip(
            g2_map_tab_index,
            "G2 Map requires multi-tau data (not available in two-time format files)",
        )
        self.tabWidget.setTabToolTip(
            diffusion_tab_index,
            "Diffusion analysis requires multi-tau data (not available in two-time format files)",
        )

        # If user is currently on disabled tab, switch to Two Time tab
        current_tab = self.tabWidget.currentIndex()
        if current_tab in multitau_tabs:
            self.tabWidget.setCurrentIndex(twotime_tab_index)
            logger.info("Switched from disabled tab to Two Time tab (two-time format)")

        logger.debug(
            "Configured tabs for two-time format: G2, G2 Fitting, G2 Map, and Diffusion tabs disabled"
        )

    def _clear_tab_tooltips(self):
        """Clear all tab tooltips."""
        for tab_index in range(self.tabWidget.count()):
            self.tabWidget.setTabToolTip(tab_index, "")

    def reorder_target(self, direction="up"):
        rows = self.get_selected_rows()
        if self.vk is None or len(rows) != 1 or len(self.vk.target) <= 1:
            return
        idx = self.vk.reorder_target(rows[0], direction)
        # idx is the new int row index, or -1 for a boundary no-op. Only reselect
        # for a valid index; setCurrentIndex needs a QModelIndex, not an int.
        if isinstance(idx, int) and idx >= 0:
            model = self.list_view_target.model()
            if model is not None:
                self.list_view_target.setCurrentIndex(model.index(idx, 0))
        self.list_view_target.repaint()
        self.update_plot()
        return

    def remove_target(self):
        if self.vk is None:
            return
        rmv_list = []
        for x in self.list_view_target.selectedIndexes():
            rmv_list.append(x.data())
        if not rmv_list:
            return

        logger.debug(f"remove_target: Removing {len(rmv_list)} files from target list")

        if len(rmv_list) == len(self.vk.target):
            self._reset_g2_fit_bounds()
        self.vk.remove_target(rmv_list)
        # clear selection to avoid the bug: when the last one is selected, then
        # the list will out of bounds
        self.clear_target_selection()

        # Update tab availability after removing files
        self.update_tab_availability()

        # if all files are removed; then go to state 1
        if self.vk.target in [[], None] or len(self.vk.target) == 0:
            logger.debug("remove_target: All files removed, resetting GUI")
            self.reset_gui()
        self.update_box(self.vk.target, mode="target")

    def _retire_plot_operations(self, plot_type: str | None = None) -> None:
        """Cancel in-flight plot operations (all, or only those of *plot_type*).

        The caller redraws (or has reset) the affected tabs itself, so no
        deferred request is replayed from here.
        """
        self._retiring_plots = True
        try:
            for operation_id, op_type in list(self.active_plot_operations.items()):
                if plot_type is not None and op_type != plot_type:
                    continue
                if self.progress_manager.is_operation_active(operation_id):
                    self.progress_manager.cancel_operation_ui(operation_id)
                else:
                    self.cancel_async_operation(operation_id)
                self._plot_pending.discard(op_type)
        finally:
            self._retiring_plots = False

    def _reset_g2_fit_bounds(self) -> None:
        self._retire_plot_operations("g2")
        self._g2_fit_bounds_initialized = False

    def reset_gui(self):
        self._close_g2_regroup()
        self._reset_g2_fit_bounds()
        self.vk.reset_kernel()
        for x in [
            # self.pg_saxs,
            self.pg_intt,
            self.mp_tauq,
            self.mp_g2,
            self.mp_saxs,
            self.mp_stab,
        ]:
            x.clear()
        self.le_bkg_fname.clear()

        # Enable all tabs when no files are loaded
        self._enable_all_tabs()

    def apply_filter_to_source(self):
        min_length = 1
        val = self.filter_str.text()
        if len(val) == 0:
            self.source_model = self.vk.source
            self.update_box(self.vk.source, mode="source")
            return
        # avoid searching when the filter lister is too short
        if len(val) < min_length:
            self.statusbar.showMessage(
                f"Please enter at least {min_length} characters", 1000
            )
            return

        filter_type = ["prefix", "substr"][self.filter_type.currentIndex()]
        self.vk.search(val, filter_type)
        self.source_model = self.vk.source_search
        self.update_box(self.source_model, mode="source")
        self.list_view_source.selectAll()

    def check_g2_number(self, default_val=(0, 0.0092, 1e-8, 1, 0.95, 1.35)):
        keys = (
            self.g2_qmin,
            self.g2_qmax,
            self.g2_tmin,
            self.g2_tmax,
            self.g2_ymin,
            self.g2_ymax,
        )
        vals = [None] * len(keys)
        for n, key in enumerate(keys):
            if isinstance(key, QtWidgets.QDoubleSpinBox):
                val = key.value()
            elif isinstance(key, QtWidgets.QLineEdit):
                try:
                    val = float(key.text())
                except Exception:
                    key.setText(str(default_val[n]))
                    val = default_val[
                        n
                    ]  # BUG-002 fix: assign val so return is never None
                    self.statusbar.showMessage("g2 number is invalid", 1000)
            vals[n] = val

        def swap_min_max(id1, id2):
            if vals[id1] > vals[id2]:
                keys[id1].setValue(vals[id2])
                keys[id2].setValue(vals[id1])
                vals[id1], vals[id2] = vals[id2], vals[id1]

        swap_min_max(0, 1)
        swap_min_max(4, 5)

        return vals

    def check_g2_fitting_number(self):
        fit_func = ["single", "double"][self.g2_fitting_function.currentIndex()]
        keys = (
            self.g2_amin,
            self.g2_amax,
            self.g2_bmin,
            self.g2_bmax,
            self.g2_cmin,
            self.g2_cmax,
            self.g2_dmin,
            self.g2_dmax,
            self.g2_b2min,
            self.g2_b2max,
            self.g2_c2min,
            self.g2_c2max,
            self.g2_fmin,
            self.g2_fmax,
        )

        vals = [None] * len(keys)
        for n, key in enumerate(keys):
            vals[n] = key.value()

        def swap_min_max(id1, id2):
            if vals[id1] > vals[id2]:
                keys[id1].setValue(vals[id2])
                keys[id2].setValue(vals[id1])
                vals[id1], vals[id2] = vals[id2], vals[id1]

        for n in range(7):
            swap_min_max(2 * n, 2 * n + 1)

        vals = np.array(vals).reshape(len(keys) // 2, 2)
        bounds = vals.T

        fit_keys = (
            self.g2_afit,
            self.g2_bfit,
            self.g2_cfit,
            self.g2_dfit,
            self.g2_b2fit,
            self.g2_c2fit,
            self.g2_ffit,
        )
        fit_flag = [x.isChecked() for x in fit_keys]

        if fit_func == "single":
            fit_flag = fit_flag[0:4]
            bounds = bounds[:, 0:4]
        bounds = bounds.tolist()
        return bounds, fit_flag, fit_func

    def update_saxs2d_level(self, flag=True):
        if not flag:
            vmin = self.pg_saxs.levelMin
            vmax = self.pg_saxs.levelMax
            if vmin is not None:
                self.saxs2d_min.setValue(vmin)
            if vmax is not None:
                self.saxs2d_max.setValue(vmax)
            self.saxs2d_min.setEnabled(True)
            self.saxs2d_max.setEnabled(True)
        else:
            self.saxs2d_min.setDisabled(True)
            self.saxs2d_max.setDisabled(True)

        self.saxs2d_min.parent().repaint()

    def clear_target_selection(self):
        self.list_view_target.clearSelection()

    def update_g2_fitting_function(self):
        idx = self.g2_fitting_function.currentIndex()
        title = [
            "g2 fitting with Single Exp:  y = a·exp[-2(x/b)^c]+d",
            "g2 fitting with Double Exp:  y = a·[f·exp[-(x/b)^c +"
            + "(1-f)·exp[-(x/b2)^c2]^2+d",
        ]
        self.groupBox_2.setTitle(title[idx])

        pvs = [
            [self.g2_b2min, self.g2_b2max, self.g2_b2fit],
            [self.g2_c2min, self.g2_c2max, self.g2_c2fit],
            [self.g2_fmin, self.g2_fmax, self.g2_ffit],
        ]

        # change to single
        if idx == 0:  # single
            for n in range(3):
                pvs[n][0].setDisabled(True)
                pvs[n][1].setDisabled(True)
                pvs[n][2].setDisabled(True)
        # change to double
        elif idx == 1:  # double
            for n in range(3):
                pvs[n][2].setEnabled(True)
                pvs[n][1].setEnabled(True)
                if pvs[n][2].isChecked():
                    pvs[n][0].setEnabled(True)

    # ------------------------------------------------------------------
    # Bayesian fitting — G2 tab
    # ------------------------------------------------------------------

    def _g2_bin_table(self):
        """``(fname, bins)`` of the first selected file for the current Q range.

        ``bins[j]`` is ``(absolute bin ID, q, label)`` of G2 column ``j``; the
        Q-bin spinbox is only a position in this table.  ``None`` if no file
        or bin table is available.
        """
        if not self.vk:
            return None
        rows = self.get_selected_rows()
        xf_list = self.vk.get_xf_list(rows) if rows else []
        if not xf_list:
            return None
        from .fitting.bayesian_assembly import resolve_g2_bins

        try:
            p = self.check_g2_number()
            return xf_list[0].fname, resolve_g2_bins(xf_list[0], (p[0], p[1]))
        except Exception:
            logger.debug("Could not resolve the G2 Q-bin table", exc_info=True)
            return None

    def _g2_current_bundle(self):
        """Stored Bayesian bundle of the currently selected physical bin, or
        ``None`` (no fit yet for *this* bin of *this* file)."""
        table = self._g2_bin_table()
        q_idx = self.sb_g2_bayesian_qidx.value()
        if table is None or not 0 <= q_idx < len(table[1]):
            return None
        fname, bins = table
        return self._g2_bayesian_store.get((fname, _revision_of(fname), bins[q_idx][0]))

    def _extract_g2_for_bayesian(self):
        """Capture the observations and physical identity of the selected Q-bin.

        Returns
        -------
        tuple or None
            ``(bundle, q_idx)``: a :class:`BayesianBundle` without result
            (observations, file name, absolute bin ID, label, model name) and
            the relative spinbox position; ``None`` if data cannot be
            extracted or the bin cannot be identified.
        """
        if not self._guard_no_data("g2"):
            return None

        rows = self.get_selected_rows()
        xf_list = self.vk.get_xf_list(rows)

        from .module import g2mod

        p = self.check_g2_number()
        q_range = (p[0], p[1])
        t_range = (p[2], p[3])

        result = g2mod.get_data(xf_list, q_range=q_range, t_range=t_range)
        if result[0] is False or len(result[0]) == 0:
            self.statusbar.showMessage("No G2 data available", 2000)
            return None

        q, tel, g2, g2_err, _labels = result

        q_idx = self.sb_g2_bayesian_qidx.value()
        # Use first file
        q_arr = np.asarray(q[0])
        if q_idx >= len(q_arr):
            self.statusbar.showMessage(
                f"Q-index {q_idx} out of range (max {len(q_arr) - 1})", 2000
            )
            return None

        bins = self._g2_resolved_bins(xf_list[0], p, q_arr)
        if bins is None:
            return None
        bin_id, _, label = bins[q_idx]

        q_value = float(q_arr[q_idx])
        x = np.asarray(tel[0])
        y = np.asarray(g2[0][:, q_idx])
        yerr = np.asarray(g2_err[0][:, q_idx])

        # Filter invalid points
        valid = np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
        if not np.any(valid):
            self.statusbar.showMessage("No valid G2 data for this Q-index", 2000)
            return None

        from .fitting.bayesian_assembly import BayesianBundle

        _, _, fit_func_name = self.check_g2_fitting_number()
        bundle = BayesianBundle.create(
            xf_list[0].fname,
            bin_id,
            label,
            q_value,
            fit_func_name,
            x[valid],
            y[valid],
            yerr[valid],
        )
        return bundle, q_idx

    def _g2_resolved_bins(self, xf, p, q_arr):
        """Absolute bin table of *xf* checked against the extracted columns."""
        from .fitting.bayesian_assembly import resolve_g2_bins

        try:
            bins = resolve_g2_bins(xf, (p[0], p[1]))
            ok = len(bins) == len(q_arr) and np.allclose([b[1] for b in bins], q_arr)
        except Exception:
            logger.debug("Could not resolve the G2 Q-bin table", exc_info=True)
            ok = False
        if not ok:
            self.statusbar.showMessage(
                "Cannot identify the physical Q-bins of this selection", 3000
            )
            return None
        return bins

    def _on_g2_qbin_changed(self, q_idx):
        """Handle Q-bin spinbox value change in g2 fitting tab.

        Updates button labels, highlights the subplot, shows Q-value
        in the status bar, and refreshes the diagnosis window.
        """
        # 1. Update button labels to reflect the selected Q-bin.
        #    This provides immediate, visible confirmation that the
        #    spinbox value is connected to the Bayesian actions.
        self.btn_g2_bayesian.setText(f"Fit Q-{q_idx}")
        bundle = self._g2_current_bundle()
        has_result = bundle is not None
        self.btn_g2_diagnosis.setEnabled(has_result)
        if has_result:
            self.btn_g2_diagnosis.setText(f"Diagnosis Q-{q_idx}")
        else:
            self.btn_g2_diagnosis.setText(f"Diagnosis Q-{q_idx} (no fit)")

        # 2. Highlight subplot (works even without file selection)
        highlighted = self._highlight_g2_subplot(q_idx)

        # 3. Show Q-value in status bar
        # The spinbox addresses a position in the current Q range: name the
        # physical bin from the same table Fit and Diagnosis use.
        q_label = ""
        table = self._g2_bin_table()
        if table is not None and 0 <= q_idx < len(table[1]):
            _, q_bin, label = table[1][q_idx]
            q_label = f"q = {q_bin:.4g} {Q_UNIT_DISPLAY} ({label})"

        if q_label:
            self.statusbar.showMessage(f"Q-bin {q_idx}: {q_label}", 3000)
        elif not highlighted:
            self.statusbar.showMessage(
                f"Q-bin {q_idx} selected \u2014 click 'Fit' to generate the plot",
                3000,
            )

        # 4. Auto-update diagnosis window if open
        window = self._g2_diagnosis_window
        if window is not None and window.isVisible():
            if has_result:
                # The stored observations are shown with the stored posterior;
                # nothing is re-extracted from the current selection.
                self._update_g2_diagnosis(bundle)
            else:
                # Explicit empty state: never another bin's posterior under
                # this bin's title.
                window.clear(f"G2 Bayesian Diagnosis \u2014 Q-bin {q_idx} (no fit yet)")

    def _highlight_g2_subplot(self, q_idx):
        """Highlight the selected Q-bin subplot in the g2 fitting plot grid.

        Returns True if a subplot was highlighted, False otherwise.
        """
        if not hasattr(self, "mp_g2_fitting"):
            return False
        hdl = self.mp_g2_fitting

        # Only meaningful for "multiple" plot type (one panel per Q-bin)
        plot_type = getattr(self, "_g2_fitting_plot_type", "multiple")
        if plot_type != "multiple":
            return False

        # No subplots yet (plot not generated)
        if not hdl.ci.rows:
            return False

        num_col = getattr(self, "_g2_fitting_num_col", 4)

        # Clear all subplot borders, then highlight selected one
        for row_dict in hdl.ci.rows.values():
            for item in row_dict.values():
                if hasattr(item, "getViewBox"):
                    item.getViewBox().setBorder(None)

        # Compute grid position for selected Q-bin
        target_row = q_idx // num_col
        target_col = q_idx % num_col
        item = hdl.getItem(target_row, target_col)
        if item is None or not hasattr(item, "getViewBox"):
            return False

        item.getViewBox().setBorder(pg.mkPen("#FF5722", width=4))

        # Auto-scroll the fitting tab scroll area so the subplot is visible
        self._scroll_to_g2_subplot(hdl, item)
        return True

    def _scroll_to_g2_subplot(self, hdl, item):
        """Scroll the g2 fitting scroll area to ensure *item* is visible."""
        try:
            scene_rect = item.getViewBox().sceneBoundingRect()
            widget_point = hdl.mapFromScene(scene_rect.center())

            # Walk up from the plot widget to find the enclosing QScrollArea
            scroll_area = hdl.parent()
            while scroll_area is not None:
                if isinstance(scroll_area, QtWidgets.QScrollArea):
                    break
                scroll_area = scroll_area.parent()

            if scroll_area is not None:
                viewport_h = scroll_area.viewport().height()
                scroll_area.ensureVisible(
                    int(widget_point.x()),
                    int(widget_point.y()),
                    0,
                    viewport_h // 3,
                )
        except Exception:
            pass  # Non-critical: highlighting still works without scroll

    def _fit_g2_bayesian(self):
        """Launch or cancel Bayesian (NUTS) fit for the selected Q-bin.

        First click starts the fit; second click while running requests
        cancellation (same toggle pattern as the batch "Fit All Q" button).
        Note: NUTS sampling is not interruptible mid-chain — cancellation
        is honoured at the next checkpoint.
        """
        # Toggle-cancel: if running, request cancellation
        if self._g2_bayesian_worker_active:
            if self._g2_bayesian_worker is not None:
                self._g2_bayesian_worker.cancel()
                self.statusbar.showMessage(
                    "Cancelling — NUTS chain must finish current iteration", 3000
                )
                self.btn_g2_bayesian.setText("Cancelling...")
                self.btn_g2_bayesian.setEnabled(False)
            return

        extracted = self._extract_g2_for_bayesian()
        if extracted is None:
            return

        bundle, q_idx = extracted
        x, y, yerr, q_value = bundle.x, bundle.y, bundle.yerr, bundle.q_value
        self._g2_bayesian_launch_revision = _revision_of(bundle.fname)

        from .fitting import fit_double_exp, fit_single_exp
        from .threading.bayesian_worker import BayesianFitWorker

        fit_func = fit_single_exp if bundle.model_name == "single" else fit_double_exp

        warmup = self.sb_g2_bayesian_warmup.value()
        samples = self.sb_g2_bayesian_samples.value()
        chains = self.sb_g2_bayesian_chains.value()
        logger.info(
            "Fit Q-%d: spinbox values warmup=%d, samples=%d, chains=%d",
            q_idx,
            warmup,
            samples,
            chains,
        )
        # Private writable copies: the bundle's own arrays stay untouched.
        worker = BayesianFitWorker(
            fit_func=fit_func,
            x=np.array(x),
            y=np.array(y),
            yerr=np.array(yerr),
            q_index=q_idx,
            q_value=q_value,
            context="g2",
            sampler_kwargs={
                "num_warmup": warmup,
                "num_samples": samples,
                "num_chains": chains,
            },
            launch=bundle,
        )
        worker.signals.finished.connect(self._on_g2_bayesian_finished)
        worker.signals.error.connect(self._on_g2_bayesian_error)
        worker.signals.cancelled.connect(self._on_g2_bayesian_cancelled)

        self._g2_bayesian_launch = bundle
        self._g2_bayesian_worker = worker
        self._g2_bayesian_worker_active = True
        self.btn_g2_bayesian.setText(f"Cancel Q-{q_idx}")
        self.statusbar.showMessage(f"Bayesian fit running for Q={q_value:.4g}...", 0)
        self.thread_pool.start(worker)

    def _on_g2_bayesian_finished(self, result):
        """Handle successful G2 Bayesian fit."""
        launch = result.get("launch") if isinstance(result, dict) else None
        if result is not None and (
            launch is None or launch is not self._g2_bayesian_launch
        ):
            # A superseded fit (workspace reset, newer launch): drop it.
            logger.info("Ignoring Bayesian G2 result of a superseded fit")
            return
        launch_revision = self._g2_bayesian_launch_revision
        self._g2_bayesian_worker_active = False
        self._g2_bayesian_worker = None
        self._g2_bayesian_launch = None
        self.btn_g2_bayesian.setEnabled(True)
        q_idx_current = self.sb_g2_bayesian_qidx.value()
        self.btn_g2_bayesian.setText(f"Fit Q-{q_idx_current}")

        if result is None:
            self.statusbar.showMessage("Bayesian fit returned no result", 2000)
            return

        if _revision_of(launch.fname) != launch_revision:
            logger.info("Dropping Bayesian G2 result: its data file was replaced")
            self.statusbar.showMessage(
                "Bayesian fit discarded: its data file changed on disk while it ran",
                5000,
            )
            return

        fit_result = result["fit_result"]
        q_value = launch.q_value

        # Commit posterior + the observations/bin it was fitted to as one unit,
        # keyed by the file revision it was fitted from.
        done = launch.with_result(fit_result)
        self._g2_bayesian_store[(done.fname, launch_revision, done.bin_id)] = done
        self._evict_stale_bayesian()

        # Enable diagnosis only if the CURRENT spinbox Q-bin has a result
        # (user may have changed spinbox while the fit was running)
        has_current = self._g2_current_bundle() is not None
        self.btn_g2_diagnosis.setEnabled(has_current)
        self.btn_g2_diagnosis.setText(
            f"Diagnosis Q-{q_idx_current}"
            if has_current
            else f"Diagnosis Q-{q_idx_current} (no fit)"
        )

        self._announce_bayesian_fit(f"Q={q_value:.4g}", fit_result)

    def _announce_bayesian_fit(self, what: str, fit_result) -> None:
        """Status-bar text (and a warning toast) for a finished Bayesian fit.

        A fit whose diagnostics did not converge is kept for the diagnosis
        views only; it is not a measurement.  A single chain cannot be assessed.
        """
        if fit_result.diagnostics.converged:
            self.statusbar.showMessage(
                f"Bayesian fit done for {what} (converged)", 4000
            )
            return
        config = getattr(fit_result, "config", None)
        single = getattr(config, "num_chains", None) == 1
        text = f"Bayesian fit for {what} did NOT converge: not a measurement"
        if single:
            text += "; single-chain runs cannot be assessed (use 2 or more chains)"
        text += " (kept for diagnosis only)"
        self.statusbar.showMessage(text, 8000)
        self.toast_manager.show_warning(text)

    def _is_current_g2_worker(self, worker_id):
        """True if *worker_id* is the single-Q fit in flight (not a superseded one)."""
        worker = self._g2_bayesian_worker
        return worker is not None and getattr(worker, "worker_id", None) == worker_id

    def _on_g2_bayesian_error(self, worker_id, error_msg, _tb, _retry):
        """Handle G2 Bayesian fit error."""
        if not self._is_current_g2_worker(worker_id):
            return
        self._g2_bayesian_worker_active = False
        self._g2_bayesian_worker = None
        self._g2_bayesian_launch = None
        self.btn_g2_bayesian.setEnabled(True)
        q_idx_current = self.sb_g2_bayesian_qidx.value()
        self.btn_g2_bayesian.setText(f"Fit Q-{q_idx_current}")
        logger.error("Bayesian G2 fit failed: %s", error_msg)
        self.statusbar.showMessage(f"Bayesian fit failed: {error_msg[:80]}", 4000)

    def _on_g2_bayesian_cancelled(self, worker_id, reason):
        """Handle G2 Bayesian fit cancellation."""
        if not self._is_current_g2_worker(worker_id):
            return
        self._g2_bayesian_worker_active = False
        self._g2_bayesian_worker = None
        self._g2_bayesian_launch = None
        self.btn_g2_bayesian.setEnabled(True)
        q_idx_current = self.sb_g2_bayesian_qidx.value()
        self.btn_g2_bayesian.setText(f"Fit Q-{q_idx_current}")
        logger.info("Bayesian G2 fit cancelled: %s", reason)
        self.statusbar.showMessage("Bayesian fit cancelled", 3000)

    # ------------------------------------------------------------------
    # Batched all-Q Bayesian fitting
    # ------------------------------------------------------------------

    def _extract_g2_all_for_bayesian(self):
        """Extract data for ALL valid Q-bins.

        Returns
        -------
        dict or None
            Everything the batch needs, captured now: ``specs`` (dicts for
            :class:`BatchBayesianCoordinator`, each carrying its own
            :class:`BayesianBundle` as ``launch``), ``q_arr``, ``t_el``,
            ``g2`` (the fitted observation matrix, NaN where not fitted), ``bin_ids``,
            ``fit_func_name``, ``fname``, ``xf`` and the Q/delay ranges;
            ``None`` on failure.
        """
        if not self._guard_no_data("g2"):
            return None

        rows = self.get_selected_rows()
        xf_list = self.vk.get_xf_list(rows)

        from .module import g2mod

        p = self.check_g2_number()
        q_range = (p[0], p[1])
        t_range = (p[2], p[3])

        result = g2mod.get_data(xf_list, q_range=q_range, t_range=t_range)
        if result[0] is False or len(result[0]) == 0:
            self.statusbar.showMessage("No G2 data available", 2000)
            return None

        q, tel, g2, g2_err, _labels = result

        q_arr = np.asarray(q[0])
        x = np.asarray(tel[0])

        _, _, fit_func_name = self.check_g2_fitting_number()

        bins = self._g2_resolved_bins(xf_list[0], p, q_arr)
        if bins is None:
            return None

        from .fitting import fit_double_exp, fit_single_exp
        from .fitting.bayesian_assembly import BayesianBundle, resolve_bin_phi

        fit_func = fit_single_exp if fit_func_name == "single" else fit_double_exp

        sampler_kwargs = {
            "num_warmup": self.sb_g2_bayesian_warmup.value(),
            "num_samples": self.sb_g2_bayesian_samples.value(),
            "num_chains": self.sb_g2_bayesian_chains.value(),
        }

        fname = xf_list[0].fname
        specs: list[dict] = []
        # Exactly the points that enter the fits (NaN elsewhere), for plotting.
        fitted_g2 = np.full(np.shape(g2[0]), np.nan)
        for q_idx in range(len(q_arr)):
            y = np.asarray(g2[0][:, q_idx])
            yerr = np.asarray(g2_err[0][:, q_idx])
            valid = np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
            if not np.any(valid):
                continue
            fitted_g2[valid, q_idx] = y[valid]
            bin_id, _, label = bins[q_idx]
            bundle = BayesianBundle.create(
                fname,
                bin_id,
                label,
                float(q_arr[q_idx]),
                fit_func_name,
                x[valid],
                y[valid],
                yerr[valid],
            )
            specs.append(
                {
                    "q_idx": q_idx,
                    "x": np.array(x[valid]),
                    "y": np.array(y[valid]),
                    "yerr": np.array(yerr[valid]),
                    "q_value": bundle.q_value,
                    "fit_func": fit_func,
                    "sampler_kwargs": sampler_kwargs,
                    "launch": bundle,
                }
            )

        if not specs:
            self.statusbar.showMessage("No valid Q-bins for Bayesian fitting", 2000)
            return None

        return {
            "specs": specs,
            "q_arr": q_arr,
            "t_el": x,
            "g2": fitted_g2,
            "bin_ids": [b[0] for b in bins],
            "bin_phi": resolve_bin_phi(xf_list[0], [b[0] for b in bins]),
            "fit_func_name": fit_func_name,
            "fname": fname,
            "revision": _revision_of(fname),
            "xf": xf_list[0],
            "q_range": q_range,
            "t_range": t_range,
        }

    def _fit_g2_bayesian_all(self):
        """Launch batched Bayesian (NUTS) fit for all Q-bins."""
        if self._g2_batch_coordinator is not None:
            self._cancel_g2_bayesian_batch()
            return

        extracted = self._extract_g2_all_for_bayesian()
        if extracted is None:
            return

        specs = extracted["specs"]

        from .fitting.models import double_exp_func, single_exp_func
        from .threading.batch_bayesian_coordinator import BatchBayesianCoordinator

        # Everything the assembly needs is captured now, as one record: the
        # target file, its bin IDs, the fitted ranges and observations.  Plots
        # later read them from the file's own summary, never from "the last
        # batch" or from the current selection.
        self._g2_batch_launch = {
            **extracted,
            "model_func": single_exp_func
            if extracted["fit_func_name"] == "single"
            else double_exp_func,
        }

        max_workers = self.sb_g2_bayesian_workers.value()
        coordinator = BatchBayesianCoordinator(
            q_specs=specs,
            thread_pool=self.thread_pool,
            max_concurrent=max_workers,
            parent=self,
        )
        self._g2_batch_coordinator = coordinator

        coordinator.progress.connect(self._on_g2_batch_progress)
        coordinator.all_finished.connect(self._on_g2_batch_finished)
        coordinator.single_q_finished.connect(self._on_g2_batch_single_q_done)
        coordinator.single_q_error.connect(self._on_g2_batch_single_q_error)

        # UI feedback
        self.btn_g2_bayesian_all.setText("Cancel")
        self.btn_g2_bayesian.setEnabled(False)
        self.sb_g2_bayesian_workers.setEnabled(False)
        self.sb_g2_bayesian_warmup.setEnabled(False)
        self.sb_g2_bayesian_samples.setEnabled(False)
        self.sb_g2_bayesian_chains.setEnabled(False)

        total = coordinator.total
        self.progress_manager.start_operation(
            "bayesian_batch_g2",
            f"Bayesian fitting {total} Q-bins",
            total=total,
            is_cancellable=True,
        )

        coordinator.start()

    def _on_g2_batch_progress(self, completed, total, msg):
        """Update progress during batch Bayesian fitting."""
        self.progress_manager.update_progress(
            "bayesian_batch_g2", completed, total, msg
        )

    def _on_g2_batch_single_q_done(self, q_idx, result):
        """Store individual Q-bin result for diagnosis access."""
        # No launch record when the dataset was reset mid-batch: drop the result
        if result is None or self._g2_batch_launch is None:
            return
        # The result travels with the bundle it was launched for (observations,
        # file, absolute bin ID); commit both as one unit.
        launch = result.get("launch")
        fit_result = result.get("fit_result")
        if launch is None or fit_result is None:
            return
        revision = self._g2_batch_launch.get("revision")
        if _revision_of(launch.fname) != revision:
            return  # fitted from a revision of the file that is gone
        done = launch.with_result(fit_result)
        self._g2_bayesian_store[(done.fname, revision, done.bin_id)] = done

    def _on_g2_batch_single_q_error(self, q_idx, error_msg):
        """Log individual Q-bin failure."""
        logger.warning("Batch Bayesian: Q-index %d failed: %s", q_idx, error_msg)

    def _on_g2_batch_finished(self, results):
        """Assemble fit_summary from batch results and refresh plots."""
        from .fitting.bayesian_assembly import assemble_fit_summary

        # The workspace was reset (state cleared) after this batch was started.
        launch = self._g2_batch_launch
        if launch is None:
            logger.debug("Ignoring batch Bayesian results: batch state was reset")
            return
        self._g2_batch_launch = None  # consumed: a repeated signal is ignored

        # Extract FitResult and per-Q timing from worker result dicts
        fit_results: dict[int, object | None] = {}
        timings: list[float] = []
        for q_idx, res in results.items():
            if res is not None:
                fit_results[q_idx] = res.get("fit_result")
                if "elapsed_s" in res:
                    timings.append(res["elapsed_s"])
            else:
                fit_results[q_idx] = None

        # Log aggregate timing stats
        if timings:
            total_time = sum(timings)
            mean_time = total_time / len(timings)
            max_time = max(timings)
            logger.info(
                "Batch timing: %d Q-bins, total=%.1fs, mean=%.1fs/Q, max=%.1fs/Q",
                len(timings),
                total_time,
                mean_time,
                max_time,
            )

        # A batch fitted from a revision of the file that has since been
        # replaced describes nothing that is loaded now: drop it.
        stale = _revision_of(launch["fname"]) != launch.get("revision")
        if stale:
            logger.info("Dropping Bayesian batch result: its data file was replaced")
            fit_results = {}

        if not stale:
            # The summary carries its own file, bin IDs, ranges and observations.
            fit_summary = assemble_fit_summary(
                results=fit_results,
                q_arr=launch["q_arr"],
                t_el=launch["t_el"],
                fit_func_name=launch["fit_func_name"],
                model_func=launch["model_func"],
                fname=launch["fname"],
                bin_ids=launch["bin_ids"],
                bin_phi=launch.get("bin_phi"),
                fit_q_range=launch["q_range"],
                fit_t_range=launch["t_range"],
                obs_g2=launch["g2"],
                obs_t_el=launch["t_el"],
            )

            # Store on the target file captured at batch start — DUAL STORAGE
            xf = launch["xf"]
            if xf is not None:
                xf.bayesian_fit_summary = fit_summary  # Bayesian owns this
                xf.bayesian_results = dict(fit_results)  # Per-Q FitResult objects
                # Do NOT touch xf.fit_summary — NLSQ owns that

        # Only a converged posterior is a measurement; the others are kept for
        # the diagnosis views (fit_summary's failed_mask says the same).
        succeeded = sum(
            1
            for v in fit_results.values()
            if v is not None
            and getattr(getattr(v, "diagnostics", None), "converged", False)
        )
        total = len(fit_results)

        # Complete progress with timing summary
        timing_str = f" ({sum(timings):.0f}s)" if timings else ""
        message = f"Bayesian fit: {succeeded}/{total} Q-bins converged{timing_str}"
        if stale:
            message = "Bayesian fit discarded: its data file changed on disk"
        self.progress_manager.complete_operation(
            "bayesian_batch_g2", success=not stale, final_message=message
        )

        # Restore UI state; the finished coordinator (and the results it holds)
        # is retired instead of lingering as a child of this window.
        coordinator, self._g2_batch_coordinator = self._g2_batch_coordinator, None
        if coordinator is not None:
            coordinator.dispose()
        self.btn_g2_bayesian_all.setText("Fit All Q")
        self.btn_g2_bayesian.setEnabled(True)
        self.sb_g2_bayesian_workers.setEnabled(True)
        self.sb_g2_bayesian_warmup.setEnabled(True)
        self.sb_g2_bayesian_samples.setEnabled(True)
        self.sb_g2_bayesian_chains.setEnabled(True)
        self.btn_g2_plot_all_q.setEnabled(True)
        q_idx_current = self.sb_g2_bayesian_qidx.value()
        self.btn_g2_bayesian.setText(f"Fit Q-{q_idx_current}")
        has_result = self._g2_current_bundle() is not None
        self.btn_g2_diagnosis.setEnabled(has_result)
        self.btn_g2_diagnosis.setText(
            f"Diagnosis Q-{q_idx_current}"
            if has_result
            else f"Diagnosis Q-{q_idx_current} (no fit)"
        )

        # Refresh tau-q tab
        try:
            self.init_diffusion()
        except Exception:
            logger.debug("Could not refresh tau-q tab after batch fit", exc_info=True)

        # Auto-refresh G2 Fit tab with Bayesian overlay
        try:
            self.plot_g2_fitting()
        except Exception:
            logger.debug("Could not refresh G2 Fit tab after batch fit", exc_info=True)

    def _cancel_g2_bayesian_batch(self):
        """Cancel the running batch Bayesian fitting."""
        if self._g2_batch_coordinator is not None:
            # cancel() mutes and retires the coordinator; it is deleted once
            # its still-running workers have reported.
            self._g2_batch_coordinator.cancel()
            self.progress_manager.complete_operation(
                "bayesian_batch_g2", success=False, final_message="Batch cancelled"
            )
            self._g2_batch_coordinator = None
            self._g2_batch_launch = None
            self.btn_g2_bayesian_all.setText("Fit All Q")
            self.btn_g2_bayesian.setEnabled(True)
            self.sb_g2_bayesian_workers.setEnabled(True)
            self.sb_g2_bayesian_warmup.setEnabled(True)
            self.sb_g2_bayesian_samples.setEnabled(True)
            self.sb_g2_bayesian_chains.setEnabled(True)
            self.btn_g2_plot_all_q.setEnabled(False)
            q_idx_current = self.sb_g2_bayesian_qidx.value()
            self.btn_g2_bayesian.setText(f"Fit Q-{q_idx_current}")

    def _plot_g2_bayesian_all(self):
        """Show all-Q Bayesian fit overlay in a matplotlib dialog."""
        rows = self.get_selected_rows()
        xf_list = self.vk.get_xf_list(rows)
        if not xf_list:
            self.statusbar.showMessage("No files selected", 2000)
            return

        xf = xf_list[0]
        bfs = getattr(xf, "bayesian_fit_summary", None)
        if bfs is None:
            self.statusbar.showMessage(
                "No Bayesian fit results \u2014 run 'Fit All Q' first", 3000
            )
            return

        # The summary is self-contained: it carries the bins, ranges and
        # observations it was fitted to, so the plot never re-extracts data
        # with the current (or another file's) selection.  The figure and the
        # dialog's exports are both built from one snapshot of it.
        from .fitting.viz import plot_bayesian_all_q

        snapshot = _bayesian_export_snapshot(xf)
        fig = plot_bayesian_all_q(dict(snapshot.summary))
        if fig is None:
            return

        self._show_bayesian_plot_dialog(fig, snapshot)

    def _show_bayesian_plot_dialog(self, fig, snapshot):
        """Display matplotlib figure in a QDialog with export button.

        *snapshot* is what the figure shows; the dialog's export reads it, not
        the viewer's current summary (a newer batch may have replaced that).
        """
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg,
            NavigationToolbar2QT,
        )
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QDialog, QHBoxLayout, QPushButton, QVBoxLayout

        # Close existing dialog if open (single-instance pattern)
        if getattr(self, "_bayesian_plot_dialog", None) is not None:
            self._bayesian_plot_dialog.close()

        dialog = QDialog(self)
        dialog.setWindowTitle("Bayesian All-Q Fit Results")
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.resize(1000, 700)

        layout = QVBoxLayout(dialog)
        canvas = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar2QT(canvas, dialog)

        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        # Export button row
        btn_row = QHBoxLayout()
        btn_export = QPushButton("Export Results")
        btn_export.clicked.connect(lambda: self._export_bayesian(snapshot, fig))
        btn_row.addStretch()
        btn_row.addWidget(btn_export)
        layout.addLayout(btn_row)

        # Track dialog lifecycle and clean up matplotlib figure on close
        self._bayesian_plot_dialog = dialog
        self._bayesian_plot_snapshot = snapshot
        dialog.destroyed.connect(lambda: plt.close(fig))
        dialog.destroyed.connect(
            lambda: self._forget_window("_bayesian_plot_dialog", dialog)
        )

        dialog.show()

    def _export_bayesian(self, snapshot, fig):
        """Export Bayesian results (CSV + PDF/PNG + netCDF) from *snapshot*,
        the summary and fit results the dialog's figure was built from."""
        from qtpy.QtWidgets import QFileDialog

        out_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not out_dir:
            return

        out_dir = Path(out_dir).resolve()
        if not out_dir.is_dir():
            self.statusbar.showMessage("Selected path is not a directory", 3000)
            return
        bfs = snapshot.summary
        if bfs is None:
            self.statusbar.showMessage("Fit summary is no longer available", 3000)
            return
        br = snapshot.results

        from .fitting.viz import export_bayesian_csv, export_bayesian_diagnostics
        from .utils.export_guard import ExportCollisionError, ExportGuard

        # One guard for the whole action: the source data file is an input and
        # every destination is claimed before it is written.
        guard = ExportGuard([snapshot.fname] if snapshot.fname else [])
        failures: list[str] = []
        refused: list[str] = []

        # 1. CSV parameters
        try:
            export_bayesian_csv(
                out_dir / "bayesian_params.csv",
                bfs["fit_val"],
                bfs["q_val"],
                bfs["fit_func"],
                failed_mask=bfs.get("failed_mask"),
                nonconverged_mask=bfs.get("nonconverged_mask"),
                bin_ids=bfs.get("bin_ids"),
                bin_phi=bfs.get("bin_phi"),
                guard=guard,
            )
        except ExportCollisionError as exc:
            logger.error("CSV export refused: %s", exc)
            refused.append(str(exc))
        except Exception:
            logger.exception("CSV export failed")
            failures.append("CSV")

        # 2. PDF figure — guard against closed figure (dialog may have been
        # destroyed between opening and clicking export)
        import matplotlib.pyplot as plt

        if not plt.fignum_exists(fig.number):
            logger.warning("Figure was already closed; skipping figure export")
            failures.append("PDF/PNG")
        else:
            for name, dpi in (("bayesian_all_q.pdf", 300), ("bayesian_all_q.png", 150)):
                try:
                    guard.claim(out_dir / name, "All-Q figure")
                    fig.savefig(out_dir / name, dpi=dpi, bbox_inches="tight")
                except ExportCollisionError as exc:
                    logger.error("Figure export refused: %s", exc)
                    refused.append(str(exc))
                except Exception:
                    logger.exception("Figure export failed")
                    failures.append("PDF/PNG")

        # 3. netCDF diagnostics (skipped, not failed, when no backend is
        # installed).  The writer drops the bins the summary marks failed.
        netcdf_skipped = False
        if br is not None:
            try:
                netcdf_skipped = not export_bayesian_diagnostics(
                    out_dir / "bayesian_diagnostics.nc",
                    br,
                    failed_mask=bfs.get("failed_mask"),
                    bin_ids=bfs.get("bin_ids"),
                    q_val=bfs.get("q_val"),
                    bin_phi=bfs.get("bin_phi"),
                    source=snapshot.fname,
                    guard=guard,
                )
            except ExportCollisionError as exc:
                logger.error("netCDF export refused: %s", exc)
                refused.append(str(exc))
            except Exception:
                logger.exception("netCDF export failed")
                failures.append("netCDF")

        if refused or failures:
            parts = []
            if refused:
                parts.append(f"Export refused ({'; '.join(refused)})")
            if failures:
                parts.append(
                    f"Export partially failed ({', '.join(failures)}); see log"
                )
            message = ". ".join(parts)
            self.statusbar.showMessage(message, 10000)
            if refused:
                self.toast_manager.show_warning(message)
        elif netcdf_skipped:
            self.statusbar.showMessage(
                f"Exported to {out_dir}; netCDF skipped "
                "(install netCDF4 or h5netcdf to enable it)",
                8000,
            )
        else:
            self.statusbar.showMessage(f"Exported Bayesian results to {out_dir}", 5000)

    def _show_g2_diagnosis(self):
        """Show or create the G2 Bayesian diagnosis window.

        Uses the current spinbox Q-bin value so the diagnosis window
        always reflects the user's current selection.
        """
        q_idx = self.sb_g2_bayesian_qidx.value()
        # Looked up by the physical bin (file, absolute bin ID) that the
        # spinbox position currently addresses.  The stored bundle brings its
        # own observations; nothing is re-extracted to pair with it.
        bundle = self._g2_current_bundle()
        if bundle is None:
            self.statusbar.showMessage(
                f"No Bayesian fit result for Q-bin {q_idx} of the selected file "
                "— run Fit first",
                3000,
            )
            return
        self._update_g2_diagnosis(bundle)

    def _update_g2_diagnosis(self, bundle):
        """Create/update the G2 diagnosis window from a stored bundle."""
        from .fitting.models import double_exp_func, single_exp_func
        from .gui.bayesian_diagnosis import BayesianDiagnosisWindow, DiagnosisIdentity

        fit_result, x, y, yerr = bundle.result, bundle.x, bundle.y, bundle.yerr
        q_value = bundle.q_value

        if self._g2_diagnosis_window is None:
            self._g2_diagnosis_window = BayesianDiagnosisWindow(
                parent=self, title="G2 Bayesian Diagnosis"
            )
            window = self._g2_diagnosis_window
            window.destroyed.connect(
                lambda: self._forget_window("_g2_diagnosis_window", window)
            )

        # The model is the one the bundle was fitted with (not whatever the
        # fitting-function combo box says now).
        model_func = {"single": single_exp_func, "double": double_exp_func}.get(
            bundle.model_name
        )
        if model_func is None:
            self.statusbar.showMessage(
                "No model function available — re-fit first", 3000
            )
            return

        win = self._g2_diagnosis_window
        win.set_axis_labels("Delay time (s)", "g2")
        win.update_results(
            result=fit_result,
            model_func=model_func,
            x_data=x,
            y_data=y,
            yerr=yerr,
            q_value=q_value,
            title=(
                f"G2 Bayesian Diagnosis - Q={q_value:.4g} "
                f"({bundle.label}, bin {bundle.bin_id})"
            ),
            identity=DiagnosisIdentity(
                bundle.fname, _revision_of(bundle.fname), bundle.bin_id, bundle
            ),
        )
        win.show()
        win.raise_()

    # ------------------------------------------------------------------
    # Bayesian fitting — Diffusion tab
    # ------------------------------------------------------------------

    def _extract_diffusion_for_bayesian(self):
        """Extract (q, tau, tau_err) from fit_summary for Bayesian power-law fit.

        Uses the deterministic tau-q fit's row selection with the current GUI
        Q range (``_tauq_rows``).

        Returns
        -------
        BayesianBundle or None
            The captured observations ``(q, tau, tau_err)`` of the first
            selected file and the Q range that selected them, as a result-less
            bundle; ``None`` if fewer than two distinct valid Q remain.
        """
        if not self._guard_no_data("diffusion"):
            return None

        rows = self.get_selected_rows()
        xf_list = self.vk.get_xf_list(rows)

        if not xf_list:
            self.statusbar.showMessage("No files selected", 2000)
            return None

        xf = xf_list[0]
        if not hasattr(xf, "fit_summary") or xf.fit_summary is None:
            self.statusbar.showMessage("Run G2 fitting first (no fit_summary)", 3000)
            return None

        # The rows are selected NOW with the GUI's Q range, by the deterministic
        # fit's own rule (_tauq_rows); rows of an earlier tau-q fit are never
        # reused.
        try:
            q_range = (float(self.tauq_qmin.text()), float(self.tauq_qmax.text()))
        except (ValueError, AttributeError):
            self.statusbar.showMessage("Invalid Q range for the diffusion fit", 3000)
            return None
        q, tau, tau_err = _tauq_rows(xf.fit_summary, q_range)

        if q.size == 0:
            self.statusbar.showMessage("No valid diffusion data in Q-range", 2000)
            return None
        if np.unique(q).size < 2:
            # same rule as XpcsFile.fit_tauq: with one Q the exponent is not
            # identifiable and the "posterior" would be the prior
            self.statusbar.showMessage(
                "Bayesian diffusion needs at least 2 distinct positive Q values "
                "in the Q range (fewer than 2: the exponent is not identifiable)",
                6000,
            )
            return None

        from .fitting.bayesian_assembly import BayesianBundle

        return BayesianBundle.create(
            xf.fname,
            "tau-q",
            "tau(q) power law",
            0.0,
            "power_law",
            q,
            tau,
            tau_err,
            q_range=q_range,
        )

    def _fit_diff_bayesian(self):
        """Launch Bayesian (NUTS) power-law fit for diffusion data."""
        if self._diff_bayesian_worker_active:
            self.statusbar.showMessage("Bayesian fit already running", 2000)
            return

        extracted = self._extract_diffusion_for_bayesian()
        if extracted is None:
            return

        # The observations are captured with the launch and committed with the
        # result; the displayed (previous) result keeps its own observations.
        bundle = extracted
        q, tau, tau_err = bundle.x, bundle.y, bundle.yerr
        self._diff_bayesian_launch_revision = _revision_of(bundle.fname)

        from .fitting import fit_power_law
        from .threading.bayesian_worker import BayesianFitWorker

        # Forward MCMC config from G2 Bayesian spinboxes (shared controls).
        sampler_kwargs: dict = {
            "num_warmup": self.sb_g2_bayesian_warmup.value(),
            "num_samples": self.sb_g2_bayesian_samples.value(),
            "num_chains": self.sb_g2_bayesian_chains.value(),
        }

        # Convert UI bounds (NLSQ b-convention) to Bayesian alpha-convention.
        # NLSQ: a * x^b  (b < 0 for diffusion)
        # Bayesian: tau0 * q^(-alpha)  (alpha > 0), so alpha = -b
        try:
            a_min = float(self.tauq_amin.text())
            a_max = float(self.tauq_amax.text())
            b_min = float(self.tauq_bmin.text())
            b_max = float(self.tauq_bmax.text())
            # b ∈ [b_min, b_max] → alpha ∈ [-b_max, -b_min]
            alpha_min = max(0.0, -b_max)
            alpha_max = max(alpha_min + 0.01, -b_min)
            sampler_kwargs["bounds"] = {
                "tau0": (max(1e-6, a_min), max(a_min + 1e-6, a_max)),
                "alpha": (alpha_min, alpha_max),
            }
        except (ValueError, AttributeError):
            pass  # Use defaults in run_power_law_fit

        worker = BayesianFitWorker(
            fit_func=fit_power_law,
            x=np.array(q),
            y=np.array(tau),
            yerr=np.array(tau_err),
            q_index=-1,
            q_value=0.0,
            context="diffusion",
            sampler_kwargs=sampler_kwargs,
            launch=bundle,
        )
        worker.signals.finished.connect(self._on_diff_bayesian_finished)
        worker.signals.error.connect(self._on_diff_bayesian_error)
        worker.signals.cancelled.connect(self._on_diff_bayesian_cancelled)

        self._diff_bayesian_launch = bundle
        self._diff_bayesian_worker = worker
        self._diff_bayesian_worker_active = True
        self.btn_diff_bayesian.setEnabled(False)
        self.btn_diff_bayesian.setText("fitting...")
        self.statusbar.showMessage("Bayesian power-law fit running...", 0)
        self.thread_pool.start(worker)

    def _on_diff_bayesian_finished(self, result):
        """Handle successful diffusion Bayesian fit."""
        launch = result.get("launch") if isinstance(result, dict) else None
        if result is not None and (
            launch is None or launch is not self._diff_bayesian_launch
        ):
            # A superseded fit (workspace reset, newer launch): drop it.
            logger.info("Ignoring Bayesian diffusion result of a superseded fit")
            return
        launch_revision = self._diff_bayesian_launch_revision
        self._diff_bayesian_worker_active = False
        self._diff_bayesian_worker = None
        self._diff_bayesian_launch = None
        self.btn_diff_bayesian.setEnabled(True)
        self.btn_diff_bayesian.setText("Fit Bayesian")

        if result is None:
            self.statusbar.showMessage("Bayesian fit returned no result", 2000)
            return

        if _revision_of(launch.fname) != launch_revision:
            logger.info("Dropping Bayesian diffusion result: its file was replaced")
            self.statusbar.showMessage(
                "Bayesian fit discarded: its data file changed on disk while it ran",
                5000,
            )
            return

        # Commit posterior + the observations it was fitted to as one unit,
        # keyed by the file revision it was fitted from.
        done = launch.with_result(result["fit_result"])
        self._diff_bayesian_store[(done.fname, launch_revision)] = done
        self.btn_diff_diagnosis.setEnabled(True)

        if not done.result.diagnostics.converged:
            # kept for the diagnosis view only: no parameter values are shown
            self._announce_bayesian_fit("the power law", done.result)
            return

        # Show parameter estimates with NLSQ-equivalent conversion:
        # Bayesian uses tau0 * q^(-alpha), NLSQ uses a * x^b, so b = -alpha.
        summary = done.result.summary
        try:
            alpha = summary.loc["alpha", "mean"]
            tau0 = summary.loc["tau0", "mean"]
            param_str = f"  tau0={tau0:.3g}, alpha={alpha:.3f} (b={-alpha:.3f})"
        except (KeyError, TypeError, AttributeError):
            param_str = ""
        self.statusbar.showMessage(
            f"Bayesian power-law fit done (converged){param_str}", 6000
        )

    def _is_current_diff_worker(self, worker_id):
        """True if *worker_id* is the diffusion fit in flight (not a superseded one)."""
        worker = self._diff_bayesian_worker
        return worker is not None and getattr(worker, "worker_id", None) == worker_id

    def _on_diff_bayesian_error(self, worker_id, error_msg, _tb, _retry):
        """Handle diffusion Bayesian fit error."""
        if not self._is_current_diff_worker(worker_id):
            return
        self._diff_bayesian_worker_active = False
        self._diff_bayesian_worker = None
        self._diff_bayesian_launch = None
        self.btn_diff_bayesian.setEnabled(True)
        self.btn_diff_bayesian.setText("Fit Bayesian")
        logger.error("Bayesian diffusion fit failed: %s", error_msg)
        self.statusbar.showMessage(f"Bayesian fit failed: {error_msg[:80]}", 4000)

    def _on_diff_bayesian_cancelled(self, worker_id, reason):
        """Handle diffusion Bayesian fit cancellation (P1-04)."""
        if not self._is_current_diff_worker(worker_id):
            return
        self._diff_bayesian_worker_active = False
        self._diff_bayesian_worker = None
        self._diff_bayesian_launch = None
        self.btn_diff_bayesian.setEnabled(True)
        self.btn_diff_bayesian.setText("Fit Bayesian")
        logger.info("Bayesian diffusion fit cancelled: %s", reason)
        self.statusbar.showMessage("Bayesian fit cancelled", 3000)

    def _show_diff_diagnosis(self):
        """Show or create the diffusion Bayesian diagnosis window."""
        # The stored bundle of the selected file brings its own observations.
        xf_list = (
            self.vk.get_xf_list(self.get_selected_rows())
            if self.vk and self.get_selected_rows()
            else []
        )
        bundle = None
        if xf_list:
            fname = xf_list[0].fname
            bundle = self._diff_bayesian_store.get((fname, _revision_of(fname)))
        if bundle is None:
            self.statusbar.showMessage(
                "No Bayesian diffusion fit for the selected file — run Fit first",
                2000,
            )
            return
        self._update_diff_diagnosis(bundle)

    def _update_diff_diagnosis(self, bundle):
        """Create/update the diffusion diagnosis window from a stored bundle."""
        from .fitting.models import power_law_func
        from .gui.bayesian_diagnosis import BayesianDiagnosisWindow, DiagnosisIdentity

        fit_result, q, tau, tau_err = bundle.result, bundle.x, bundle.y, bundle.yerr

        if self._diff_diagnosis_window is None:
            self._diff_diagnosis_window = BayesianDiagnosisWindow(
                parent=self, title="Diffusion Bayesian Diagnosis"
            )
            window = self._diff_diagnosis_window
            window.destroyed.connect(
                lambda: self._forget_window("_diff_diagnosis_window", window)
            )

        win = self._diff_diagnosis_window
        win.set_axis_labels("Q (1/A)", "tau (s)")
        win.update_results(
            result=fit_result,
            model_func=power_law_func,
            x_data=q,
            y_data=tau,
            yerr=tau_err,
            title=f"Diffusion Bayesian Diagnosis - Power Law ({bundle.fname})",
            identity=DiagnosisIdentity(
                bundle.fname, _revision_of(bundle.fname), None, bundle
            ),
        )
        win.show()
        win.raise_()


def setup_windows_icon():
    # reference: https://stackoverflow.com/questions/1551605
    import ctypes
    from ctypes import wintypes

    lpBuffer = wintypes.LPWSTR()
    AppUserModelID = ctypes.windll.shell32.GetCurrentProcessExplicitAppUserModelID
    AppUserModelID(ctypes.cast(ctypes.byref(lpBuffer), wintypes.LPWSTR))
    appid = lpBuffer.value
    ctypes.windll.kernel32.LocalFree(lpBuffer)
    if appid is None:
        appid = "aps.xpcs_viewer.viewer.0.20"
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(appid)


def main_gui(path=None, label_style=None, path_explicit=None):
    if os.name == "nt":
        setup_windows_icon()

    logger.info("Starting XPCS Viewer GUI application")
    if path:
        logger.info(f"Starting with path: {path}")

    # AA_EnableHighDpiScaling and AA_UseHighDpiPixmaps are always-on in Qt6
    # and were removed as configurable attributes. Setting them is unnecessary
    # and emits deprecation warnings.

    app = QtWidgets.QApplication([])

    # Work around macOS 26.3 bug: Apple's ImageIO PNGReadPlugin crashes
    # (EXC_ARM_DA_ALIGN at 0x0bad4007) when Core Text tries to render
    # emoji glyphs via CopyEmojiImage during QLabel paint events.
    # NoFontMerging prevents Qt from falling back to Apple Color Emoji.
    # Only apply on macOS — on Linux this breaks Unicode glyph rendering
    # (e.g. Å in "Å⁻¹") because the primary font may lack certain glyphs
    # and Qt cannot fall back to a font that has them.
    import sys

    if sys.platform == "darwin":
        font = app.font()
        font.setStyleStrategy(QtGui.QFont.StyleStrategy.NoFontMerging)
        app.setFont(font)

    logger.info("Qt Application created")

    try:
        # Store reference to prevent garbage collection of the Python wrapper
        # while the C++ QMainWindow is alive (PySide6 lifecycle requirement)
        _viewer = XpcsViewer(
            path=path, label_style=label_style, path_explicit=path_explicit
        )
        logger.info("XpcsViewer window created successfully")
        app.exec()  # exec_() is deprecated in PySide6
    except Exception as e:
        logger.critical(f"Failed to start application: {e}", exc_info=True)
        raise
    finally:
        logger.info("Application shutting down")


if __name__ == "__main__":
    main_gui()
