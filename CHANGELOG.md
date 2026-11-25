# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- Mask Editor: ROI vertex rasteriser ported from pySimpleMask (Apache-2.0); the wheel now ships `NOTICE` and `LICENSE-APACHE-2.0`
- Q-map: transmission geometry accepts horizontal and vertical detector swing angles (degrees, from the optional `swing_angle_horizontal`/`swing_angle_vertical` metadata keys; the APS 8-ID-I reader supplies them, other readers do not); zero swing leaves TTH, phi, qx, qy, x and y unchanged

### Changed

- Add `scikit-image>=0.26.0` dependency (used by the ROI rasteriser)
- Mask Editor: a custom frame range outside a single 2-D HDF5 image (start other than 0, or more than one frame) is now an error instead of being ignored
- Q-map: `q` is now the momentum-transfer magnitude `2*k0*sin(alpha/2)` instead of `k0*sin(alpha)`; `qx`/`qy` keep the `k0*sin(alpha)` projection. Saved zero-swing transmission Q-maps and q-based partitions shift slightly, by about `alpha^2/8` relative (more at wide angle). The interactive viewer's hover q readout (`fileIO/qmap_utils.py`, used via `xpcs_file`) still uses `k0*sin(alpha)` like upstream pyXpcsViewer, so at wide angle (about 1.5% at 20 deg) it no longer equals the SimpleMask partition q; aligning the viewer is a separate decision

### Fixed

- Mask Editor: APS 8-ID-I raw scans now carry their detector swing angles (`flightpath_swing`, `flightpath_swing_vertical`) into the Q-map; a swung scan previously opened with an unswung Q-map
- Version: `xpcsviewer.__version__` previously fell back to a hard-coded `0.1.10` (and the log banner to `unknown`); it is now looked up from the `xpcsviewer-gui` distribution, so installed wheels report their real version, with `0+unknown` only when package metadata is absent
- Fitting: in the G2 fit panel an unchecked parameter is still held at the midpoint of its min/max bounds (the geometric mean when the tau bounds are positive). Scripts calling `fit_with_fixed*` directly must now pass an explicit, finite `p0` whenever a parameter is fixed (`ValueError` otherwise) instead of getting a silent bounds-midpoint fallback; failed-fit placeholders keep the explicit fixed values
- Fitting: `fit_with_fixed*` treat integer `fit_flag` values such as `[1, 1, 0]` as booleans; integer flags previously gave wrong parameters
- Fitting: tau-q ignores samples with non-positive tau or q (or infinite q) for the starting values, the fit and the plot grid, and logs how many it dropped; such samples previously made the starting values NaN, biased the fitted amplitude or failed the fit
- Averaging: a job whose inputs are all rejected (by the baseline criteria or because they fail to load) now ends `finished` with no output file (it stayed `running` and kept polling); killed jobs stay `killed`, and the empty result is reported in the status bar
- Averaging: with 8 or more files (the parallel path) a scan is averaged only when all its fields match the shapes already accepted, as on the sequential path, and the job reports the number of scans actually averaged; a scan with a different lag count previously diluted G2 and its error bars
- Averaging: a second Start click on a queued, running, finished or killed job is ignored (it could crash the application)
- Averaging: an integer-typed first file no longer makes the sum fail or truncate on the sequential or parallel path (sums are float64), and a file is counted as accepted only after all of its fields were added
- Averaging: a queued job can now be killed with the Kill button; it ends `killed` without opening any file
- G2 fit panel: delay ranges so short that the tau bounds would round to an empty or inverted range no longer initialise the bounds, so a later, slower scan can still set usable defaults
- Fitting: tau-q with fewer than two distinct positive q values and both parameters free now reports a failed fit instead of raising a linear-algebra error or reporting the bounds midpoint as a success
- G2 regroup: a positive delay whose elapsed time underflows to zero seconds is now rejected with an error
- Q-map: the q-bin label lookup returns "invalid qbin" for booleans, non-integral numbers and non-numeric values, and accepts integral floats and NumPy integers
- Mask Editor: raw-frame averaging accumulates in float64 on every path, so one and several processes give the same image, and the averaged frame range (with a note when only a representative subset is used) is logged
- Mask Editor: a rejected mask import is reported as a failure in the status bar and log, without a modal dialog; datasets of unsupported type (strings, bytes, complex) are rejected the same way, and upper-case `.H5`/`.HDF5` extensions are accepted
- Mask Editor: draw and parameter edits replace their own mask layer instead of accumulating; Apply with nothing drawable is a no-op
- Mask Editor: ROIs are rasterised through their transform, so rotated shapes match pySimpleMask geometry; line ROIs (vertical and angled) created programmatically or as pyqtgraph `LineROI` items are rasterised the same way, but the toolbar Line tool still does not create an ROI (known issue, unchanged in this release); degenerate and off-image ROIs are ignored
- Mask Editor: Apply on a large curved ROI is slower than in 0.1.10 (about 2.4 s for a 2000-pixel ellipse on a 2167x2070 detector), because shapes are now rasterised as polygons to match pySimpleMask geometry

## [0.1.10] - 2026-08-11

### Added

- Mask Editor: Open Raw File action with off-thread loading and status-bar error reporting
- Mask Editor: APS 8-ID-I raw scattering-format reader stack (HDF5 loader, reader dispatch, NeXus metadata helpers, frame-range/parallel averaging)
- Mask Editor: automatic beam center detection UI wired to kernel

### Changed

- Add `optimistix` dependency, update CUDA install flow
- README: dynamic PyPI version badge
- SAXS-2D: debounce selection spinbox and tune two-time splitter ratios

### Fixed

- Two-time: preserve per-file q-bin selection across dataset switches
- Deps: keep `jax`/`jaxlib` lower bound at `>=0.8` (`interpax<=0.3.14` transitively caps `jax<0.11`; a `>=0.11` bound is unresolvable via a real `pip install`)
- CI: fix smoke-test collection gap, ruff format drift, missing Qt system libraries in smoke-test job
- Tests: stub startup dialog prompt in GUI conftest fixture; prune redundant GPU migration test classes; restore missing `ViewerKernel` mock fixture attributes
- pyproject.toml: `Changelog` project URL pointed at nonexistent `HISTORY.rst`; now points to `CHANGELOG.md`

## [0.1.9] - 2026-06-26

### Added

- CLI: `xv` and `xv-gui` short command aliases for `xpcsviewer` / `xpcsviewer-gui`
- Python 3.14 support
- OSTI GitLab mirror CI workflow
- Docs: README now covers command aliases and batch all-Q Bayesian fitting

### Changed

- Refine GUI tab availability logic
- Remove over-engineering and unused modules per repo-wide audit

### Fixed

- Session: persist absolute target paths so sessions restore correctly regardless of working directory
- GUI: resolve 13 verified defects from combined code-review pass
- Backends: drop stale `optimize` re-exports from `scipy_replacements`
- Resolve verified defects surfaced by multi-tool and adversarial review
- Docs (API): remove stale `backend` and `performance_monitor` module references

## [0.1.8] - 2026-05-07

### Changed

- Threading: fix race condition in `AsyncDataPreloader` — connect `@Slot` targets before worker submission to prevent signal loss under load
- Threading: add `threading.Lock` to `ViewerKernel` dataset cache to prevent `WeakValueDictionary` corruption under concurrent access
- Threading: LRU eviction in `WorkerManager` result/error storage (cap: 100 entries) to prevent unbounded memory growth
- Threading: optimised `UnifiedThreadingManager` shutdown — cancel pending futures before `pool.shutdown(wait=False)` to avoid hang on exit
- Schema: `G2Data.to_dict()` now returns defensive copies of all mutable array fields
- Performance benchmarks: mark `TestBottleneck*` with `@pytest.mark.flaky(reruns=3)` to tolerate CI timing variance
- Plothandler: `LazyMplCanvasBarV` in `plothandler/lazy_proxy.py` defers Matplotlib import until `showEvent` to reduce startup latency

## [0.1.7] - 2026-05-01

### Changed

- Constrain JAX versions and optimize performance benchmarks
- Add codecov configuration to ignore non-code paths

## [0.1.6] - 2026-03-12

### Fixed

- Python syntax error in Makefile `gpu-check` target (inline `if` with line continuations)

### Changed

- CI: bump `astral-sh/setup-uv` from v4 to v7 across all GitHub Actions workflows
- CI: bump GitHub Actions to latest majors (`checkout` v5→v6, `upload-artifact` v4→v7, `download-artifact` v5→v8, `cache` v4→v5, `codecov-action` v4→v5, `action-gh-release` v1→v2)

## [0.1.5] - 2026-03-04

### Added

- GPU acceleration setup: `gpu_cuda12` and `gpu_cuda13` optional dependency groups in pyproject.toml
- `check_plugin_conflicts()` in `utils/device.py` for detecting dual-plugin and version-mismatch issues
- `make gpu-diagnose` target for troubleshooting GPU plugin conflicts, version mismatches, and system CUDA
- `make gpu-check` now includes SVD computation test to verify GPU compute works end-to-end
- `make env-info` now includes GPU diagnostics output
- GPU troubleshooting table and platform support table in README.rst and installation docs

### Fixed

- P0: `trange(executor.map(...))` crash in averaging toolbox (trange expects int, not iterator)
- P0: `xf.at(key)` crash in averaging toolbox (XpcsFile has no `.at()` method)
- P1: Twotime HDF5 index off-by-one (`idx[3:]` → `idx[4:]` for `"dq_001"` keys)
- P1: `_computation_cache` attribute missing in XpcsFile tauq path (use `cache_put` instead)
- P1: Missing bounds `None` check before `assert len(bounds) == 2` in XpcsFile
- P1: `np.linspace` used for fit_x on log-x G2 plots in bayesian_assembly (use `np.logspace`)
- P1: `Normalize(vmin, vmax)` crash when `q_min == q_max` in fitting viz
- P1: `threading.Lock` deadlock in CleanupScheduler on re-entry (switched to `RLock`)
- P1: Race condition in singleton initialization for cleanup system (double-checked locking)
- P1: `_restore_session` crashes when `self.vk` is None due to missing data path
- P1: Missing leading `/` in `c2_g2` HDF5 path in `aps_8idi.py` key map
- P2: `statusbar()` call fails when statusbar is a QStatusBar widget, not callable
- P2: Null parent chain crash in matplot_qt and pyqtgraph plothandlers
- P2: Null `child_layout` access in `layout_helpers.py`
- P2: `self.vk` accessed without None check in 11 GUI methods (averaging, session, saxs2d, etc.)
- P2: `xf.__getattr__(key)` bypasses `@property` lazy-load in average_toolbox (use `getattr`)
- P2: `assert` on inconsistent C2 sampling rates replaced with warning + fallback
- P2: Division by zero in C2 statistics for 1x1 matrices
- P2: `saxs_2d_data.size` crash when saxs_2d_data is None
- P2: Off-by-one in `get_qbin_label` bounds check (`>` → `>=`)
- P2: `_connection_lock` held across `yield` in EnhancedHDF5Reader blocks all concurrent threads
- P2: Unhandled OSError in SimpleMask `save_mask()` h5py write
- P3: Dead `len()` call in `state_validator.py`
- Flaky window state signal test in GUI interactive tests

### Changed

- GPU install targets now uninstall all 6 JAX/CUDA packages before reinstalling (prevents PJRT plugin conflicts)
- `_print_gpu_warning()` in `utils/device.py` now shows full uninstall command and detected plugin issues
- `get_device_info()` returns `plugin_issues` field for runtime plugin conflict detection
- ArviZ `InferenceData` migrated to xarray `DataTree` for ArviZ 1.0+ compatibility (`FitResult.arviz_data`, `compute_bfmi`, `generate_arviz_diagnostics`, `bayesian_diagnosis`)
- 59 ruff auto-fixes: unused imports, `Callable` migration, unnecessary `else` after `return`
- Ruff auto-formatting applied across codebase

### Removed

- 16 flaky GUI tests: always-pass assertions (`assert True`, `len >= 0`, `or True`), race conditions (5–10 ms waits), and empty test bodies from `tests/gui_interactive/`
- Obsolete test frameworks, legacy validation scripts, and unused goldens (20+ files deleted)
- Offscreen golden snapshot test execution from CI workflow

### Documentation

- API docs coverage expanded: 8 utils modules (atomic_io, device, health_monitor, memory_predictor, reliability_manager, startup_optimizer, state_validator, visualization_optimizer), 3 scipy_replacements submodules, fitting sampler, fitting visualization, and schema validators
- LaTeX/PDF build configured: XeLaTeX engine with Unicode character support
- Fixed `curve_fit` docstring RST emphasis warning in scipy_replacements

### Build

- Consolidated `dev`, `docs`, and `test` dependency extra groups into base requirements
- CI: skip local hooks in pre-commit.ci, remove obsolete documentation group references
- Updated offscreen plotting golden image for snapshot tests
- Updated `sphinx-autodoc-typehints` and `tifffile` dependencies in `uv.lock`

## [0.1.3] - 2026-02-23

### Added

- "Plot All Q" button in G2 Fit tab for all-Q Bayesian overlay visualization
- Export button: CSV parameters, PDF/PNG figures, netCDF ArviZ diagnostics
- Sampler config spinboxes (warmup, samples, chains) for batch Bayesian fitting
- `fitting.viz` module: `plot_bayesian_all_q`, `export_bayesian_csv`, `export_bayesian_diagnostics`
- Batch all-Q Bayesian fitting with GUI integration
  - `fitting/bayesian_assembly.py` coordinator for per-Q NUTS sampling
  - Batch fitting dialog in G2 Fit tab with progress tracking
  - `assemble_fit_summary()` converts Bayesian posteriors to legacy format
- NLSQ 0.6.0 adoption with full feature delegation
  - Statistical metrics: R², adjusted R², RMSE, MAE, AIC, BIC
  - Confidence intervals (95% level) and prediction intervals
  - `workflow` parameter replacing presets ('auto', 'auto_global', 'hpc')
  - `stability`, `fallback`, `compute_diagnostics`, `show_progress` parameters
  - NLSQResult property delegation to native CurveFitResult
- SVG icon system with theme-aware `currentColor` replacement and caching
- Category tab bar with separator lines between tab groups
- 12-tab layout: G2 View, G2 Fit, G2 Map as separate tabs
- Bayesian inference worker with NUTS diagnosis dialog
- Q-bin spinbox with per-Q navigation and subplot highlight in G2 Fit tab
- ROI handles and Q-map colormap selector in SimpleMask
- `Q_UNIT_DISPLAY` constant and unit normalization helpers for consistent Q-unit rendering
- Comprehensive structured logging (rate limiting, JSON aggregation, `@log_timing`)
- Type-safe schema dataclasses for HDF5 data validation
- HDF5 facade with connection pooling and batch reading
- HDF5 connection pool (`max_size=25`, health checks)
- JAX GPU installation targets for system CUDA
- 91+ new tests (Phase 1 test suite update)
- Performance profiling and bottleneck analysis test suite

### Changed

- G2 tab split into three tabs (G2 View, G2 Fit, G2 Map)
- Compact layout for G2 Fit (2-column grid), Diffusion (3-column controls, 80px tauq_msg), and G2 Map tabs
- NLSQ `preset` parameter renamed to `workflow`
- Fitting backend migrated from scipy to NLSQ 0.6.0 library
- NLSQ sampler falls back to `init_to_uniform` with 1.5x warmup when health score is low
- `param_names` field added to NLSQResult and FitResult for canonical parameter ordering
- Exception-based validation replaces tuple-return pattern (`XPCSValidationError`)
- Lambda slots replaced with named functions in async kernel
- `NoFontMerging` font strategy restricted to macOS only (fixes Unicode glyph rendering on Linux)
- `scripts/` reorganized into `build/`, `screenshots/`, `validation/` subdirectories
- `examples/` directory removed (content covered by notebooks)
- `validation/` directory consolidated into `scripts/validation/`

### Fixed

- Bayesian fit results now stored on `bayesian_fit_summary` (no longer overwritten by NLSQ)
- G2 Fit tab auto-refreshes after "Fit All Q" batch completes
- Diffusion tab prefers Bayesian fit results when available
- Sampler kwargs (warmup, samples, chains) now forwarded to batch workers
- NLSQResult delegation: handle `confidence_intervals` as method or property
- NLSQResult delegation: handle `prediction_interval` returning `(n, 2)` array
- NLSQ fallback to uniform initialization when warm-start params are unreliable
- `param_names` ordering: prevent alphabetical dict-key scrambling in posterior predictive plots
- Q-unit double-encoded UTF-8 Angstrom bytes from HDF5 (Latin-1 fallback decode)
- JAX `TracerArrayConversionError` in model functions (use `jnp.exp` not `np.exp`)
- MCMC warm-start: use `init_to_value` instead of partial `init_params`
- Power-law fitting: correct `tau_err` and `bounds` API
- Legacy fitter: use initial guess for fixed parameters
- HDF5 fork-safety and float64 type coercions
- Qt main thread safety and resource leak prevention
- 38 P0/P1/P2 GUI issues from deep RCA analysis
- Batch coordinator initialization order (before `load_path`)
- Deterministic GPU fallback tests and PyQt6 compatibility coverage
- Type safety improvements across widgets and threading modules
- Concurrency deadlocks and signal safety in threading layer

### Performance

- G2, SAXS 1D, and twotime hot paths optimized (reduced array copies)
- Mask handling and file I/O: fewer intermediate array allocations
- `@jax.jit` applied to NLSQ model functions
- `vmap`/`scan`/`fori_loop` added to JAX backend
- HDF5 connection pooling and batch reading
- Tighter container margins and reduced splitter handles across all tabs

### Documentation

- Tutorial notebooks rewritten as self-contained with synthetic data
- Gallery screenshots refreshed for 12-tab compact layout
- API docs added for icons, category tab bar, Bayesian worker, exceptions
- Fitting and threading Sphinx references updated
- Fitting module Sphinx docs with NLSQResult metric docstrings
- Architecture docs: dependency analysis, integration catalog
- Installation docs updated to `xpcsviewer-gui` package name
- Dark-theme gallery screenshots
- Furo toctree cleanup (hidden sidebar navigation)
- Legacy redirect pages removed
- AGENTS.md and CONTRIBUTING.md modernized

## [0.1.2] - 2026-01-06

### Added

- SimpleMask module for interactive mask editing and Q-map generation
  - Drawing tools: Rectangle, Circle, Polygon, Line, Ellipse, Eraser
  - Q-map computation from detector geometry
  - Partition (Q-binning) utilities with dynamic/static partition support
  - Undo/redo history for mask operations
  - Export masks and partitions to XPCS Viewer via signals

### Documentation

- Comprehensive SimpleMask documentation added to Sphinx
- AST-based documentation analysis and updates

## [0.1.1] - 2026-01-06

### Added

- G2 Map tab with dynamic UI creation for visualizing g2 values across Q-space
- G2 stability visualization ported from upstream
- XPCS Viewer logo with Qt icon resources

### Changed

- Window title changed from "XPCS Toolkit" to "XPCS Viewer"
- G2 Map tab positioned after G2 tab for logical workflow
- Python target version updated to 3.13
- Session manager updated to support 11 tabs
- Sphinx documentation theme switched to Furo

### Documentation

- RST documentation files converted to Markdown
- Logo added to Sphinx header

### Fixed

- Session manager properly handles increased tab count
- Qt error regression test fixture added
- Method renamed from `plot_g2map` to `plot_g2_map` for consistency
- Bandit false positive B608 suppressed with nosec comment
- Ruff linting issues resolved across codebase
- CI breakages fixed: UI generation and missing assets repaired
- Invalid file scanning paths guarded
- Performance test stabilized

## [0.1.0] - 2026-01-05

### Initial Release - xpcsviewer package

This is the first release under the new package name `xpcsviewer`.

### Features

- G2 correlation analysis with single/double exponential fitting
- SAXS 1D/2D visualization and analysis
- Two-time correlation analysis with batch processing
- HDF5/NeXus data support (APS-8IDI beamline format)
- Sample stability monitoring
- File averaging with parallel processing

### GUI Modernization

- Light/dark theme support with automatic system detection
- Session persistence - resume work where you left off
- Command palette (Ctrl+Shift+P) for quick access to all commands
- Toast notifications for non-intrusive status updates
- Keyboard shortcut manager with customizable keybindings
- Drag-and-drop file handling with visual feedback
- Theme-aware PyQtGraph and Matplotlib plot backends

### Technical

- Python 3.12+ with PySide6 GUI framework
- Centralized constants module for configuration
- Comprehensive test coverage
- CI/CD with GitHub Actions
- Sphinx documentation
