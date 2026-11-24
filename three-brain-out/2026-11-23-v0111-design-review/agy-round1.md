## Verdict
NOT READY

Stage 4's architecture and task ordering are sound, and Task 6/7's ROI rasterisation design and numeric assertions are verified against pyqtgraph and upstream `pySimpleMask`. However, six blocking issues will break tests or force implementer guesswork across Tasks 2, 3, 4, 5, 6, and 8 (swung Q-map math/JIT gap, nonexistent fit UI widget and missed callers/tests, modal mask validation pre-empting `load_mask`, layer-history invariants vs existing tests and `max_history=3`, cancelled averaging overwrite, and invalid `NOTICE` packaging config).

## Blocking findings

1. **B1** | High | Plan `L180-196`, Spec `L112-116`
   - **Evidence:** `xpcsviewer/simplemask/qmap.py:154,401,423` computes `q = sin(alpha) * k0` in `float64` across two paths (`_get_transmission_qmap_jit` L121-168, `_compute_transmission_qmap_backend` L348-439), and `compute_qmap` (L90-97) omits swing angles. Upstream `/tmp/xpcs-parity.UVfRoj/pySimpleMask/src/pysimplemask/core/qmap.py:133,139-143` computes `q = 2*k0*sin(alpha/2)` and casts `qx`/`qy` to `float32`. At 20° swing, `sin(alpha)*k0` vs `2*k0*sin(alpha/2)` differ by 1.52% (`0.0258 Å⁻¹`), and `float32` `qx`/`qy` differ from `float64` by `5.3e-8`.
   - **Why it matters:** Porting rotation only into `_compute_transmission_qmap_cached` leaves the JAX JIT path un-swung and fails `test_swung_matches_upstream` at `rtol=1e-12` on both `q` and `qx`/`qy`.
   - **Suggested fix:** Adopt `q = 2*k0*sin(alpha/2)` in transmission Q-maps (noting small-angle helpers at `qmap.py:226,272,342` stay unchanged and `test_qmap.py:105,251` remain within `1e-6`), thread swing angles through `_compute_transmission_qmap_backend`, `_get_transmission_qmap_jit`, and `compute_qmap` (`metadata.get("swing_angle_horizontal", 0.0)` / `swing_angle_vertical`), and test `qx`/`qy` at float32 tolerance (`rtol=1e-6`) or match dtype.

2. **B2** | High | Spec `L95-102`, Plan `L52-70`
   - **Evidence:** `xpcsviewer/xpcs_viewer.py:4203-4253` (`check_g2_fitting_number`) has only min/max bound spinboxes (`g2_amin..g2_fmax`) and checkboxes (`g2_afit..g2_ffit`)—no initial-value UI widget; `XpcsFile.fit_g2` (`xpcsviewer/xpcs_file.py:1733-1745`) synthesizes `p0` from bounds midpoint. `XpcsFile.fit_g2_high_performance` (`xpcsviewer/xpcs_file.py:2080,2093`, outside `~L1706-1917`) omits `p0` entirely (`fit_tauq` at L2395 passes `p0`). Existing `tests/unit/fitting/test_nlsq_bayesian_integration.py:117-140` (`test_fixed_param_without_p0_uses_midpoint`) asserts the old `p0=None` fallback.
   - **Why it matters:** Implementers will hunt for a nonexistent UI entry, break `fit_g2_high_performance` with fixed parameters, and fail `pytest tests/unit/fitting -q` at Step 4.
   - **Suggested fix:** State that the UI supplies bounds (`fit_g2` computes `p0`) while `legacy.py` enforces explicit `p0` at the solver API; update `xpcs_file.py:2074-2095` to compute/pass `p0`; update `test_fixed_param_without_p0_uses_midpoint` to expect `ValueError`.

3. **B3** | High | Spec `L85-88,184-185`, Plan `L20,90-110`
   - **Evidence:** `SimpleMaskWindow._on_load_mask` (`xpcsviewer/simplemask/simplemask_window.py:445-466`) calls `_validate_mask_dimensions` (L468-514) before `kernel.load_mask`. That helper pops modal dialogs (`QMessageBox.warning` at L483 for missing `"mask"` key and L494 for any `mask_shape != detector_shape`—including transposed `(8,6)` masks, falsely claiming "cropped or padded"; `QMessageBox.critical` at L507 for unreadable/non-HDF5 files) and returns `False` without updating `status_bar`.
   - **Why it matters:** `test_window_reports_failure` (patching only `QMessageBox.warning`) fails if the stub returns `No` or hits `QMessageBox.critical`, and real users still get misleading modal prompts. Plan L20 ("Load Failed `QMessageBox` stays") also conflicts with Spec L184-185 ("non-modal status message plus a log entry").
   - **Suggested fix:** Remove `_validate_mask_dimensions` (letting `load_mask` validate and accept transposed shapes) or update it to allow transposed shapes and report failures non-modally; align Spec L184-185 and Plan L20 on the `"Load Failed"` dialog.

4. **B4** | High | Plan `L112-132`
   - **Evidence:** `tests/unit/simplemask/test_area_mask.py:211,233,246-247,262` mutates `ma.mask_record` directly; `test_reset_clears_history` (L240-253) appends two arrays only to `mask_record` and calls `redo_undo("reset")`. Also `xpcsviewer/simplemask/area_mask.py:437-439` caps `len(mask_record)` at `max_history + mask_ptr_min`: with `max_history=3, mask_ptr_min=0` and 6 draws, `mask_record` retains only `[d4, d5, d6]` (`mask_ptr=2`).
   - **Why it matters:** Popping `layer_record` inside the `mask_record` loop raises `IndexError` on `test_area_mask.py` (required unchanged by Plan L116), and Plan L128's assertion ("undo three times returns 5th, 4th, 3rd") fails because `d3` was evicted.
   - **Suggested fix:** Guard `layer_record` truncation with its own length check (`while len(self.layer_record) > 1 + self.mask_ptr_min:`) on reset/discard/trim, and change Plan L128 to `max_history=4` (or assert two undos returning 5th and 4th).

5. **B5** | Medium | Plan `L72-88`
   - **Evidence:** `xpcsviewer/module/average_toolbox.py:183-185` (plan cites `~L174`) runs after `_process_files_sequential` (L263-267) and `_process_files_parallel` (L382-386), which set `self.status = "killed"` and return early when `self.is_killed.is_set()`. Also `do_average` logs `"average job %d ..."` at L124, L239, whereas `__init__` defaults `jid` to `uuid.uuid4()` (L64).
   - **Why it matters:** Calling `_finish(0)` unconditionally on `np.sum(mask) == 0` overwrites `"killed"` with `"finished"` and emits `progress=100`. Tests omitting `jid` hit `TypeError` at L124.
   - **Suggested fix:** Check `if self.is_killed.is_set(): return` before the `np.sum(mask) == 0` branch; pass an int `jid` in Task 3 tests or change `%d` to `%s` at L124/L239.

6. **B6** | Medium | Spec `L168-172`, Plan `L150-151`
   - **Evidence:** `pyproject.toml:10` uses legacy `license = {file = "LICENSE"}` with no `license-files` key and no `MANIFEST.in`; `[tool.setuptools.package-data]` (L187-196) covers only `xpcsviewer/`.
   - **Why it matters:** Plan L150's instruction ("as the build config already does for `LICENSE`") does not match `pyproject.toml`. Mixing PEP 639 `license-files` with `license = {file = ...}` fails setuptools validation; adding root `NOTICE` to `package-data` silently omits it.
   - **Suggested fix:** Specify exact packaging—PEP 639 (`setuptools>=77.0.3`, `license = "MIT"`, `license-files = ["LICENSE", "NOTICE"]`) or `MANIFEST.in`—and verify `NOTICE` lands in `.dist-info/licenses/` of the built wheel.

## Non-blocking findings

1. **N1** | Low | Plan `L33-50`
   - **Evidence:** `xpcsviewer/fitting/results.py:943` queries `safe_version("xpcs-toolkit")` for Bayesian reproducibility metadata. `LoggingConfig.__init__` (`xpcsviewer/utils/logging_config.py:52-55`) creates `~/.xpcsviewer/logs` and clears root handlers.
   - **Why it matters:** Reproducibility exports still miss `"xpcsviewer-gui"`, and instantiating `LoggingConfig()` directly in Task 1 tests mutates filesystem/logging state.
   - **Suggested fix:** Update `results.py:943` to `"xpcsviewer-gui"` (or `_package_version()`), and specify `LoggingConfig.__new__(LoggingConfig)._get_app_version()` in Task 1 Step 1.

2. **N2** | Low | Plan `L75,157,162,177`
   - **Evidence:** Drifted line refs (`average_toolbox.py:183` vs `~L174`; `simplemask_kernel.py:367-416` vs `~L332-380`). `probe_mask_gui_parity.py:12-15` takes only `side` and `output` (no extra args, not in `reproduce.md`).
   - **Why it matters:** Minor implementer friction in Task 7 Step 5.
   - **Suggested fix:** Fix line refs, drop `<args as in reproduce.md>`, and note `getLocalHandlePositions()` yields `(name, QPointF)` tuples.

3. **N3** | Low | Spec `L112-116,156`, Plan `L204`
   - **Evidence:** `parity-audit.md:45` already maps `swung transmission` to **M16**. Plan L204 directs re-running `probe_core_parity.py` ("same commands as `reproduce.md`") to verify M18, M20, M21, V14a, V19, and swung transmission, but `reproduce.md` invokes `probe.py`; `probe_core_parity.py` omits M18 (`probe_mask_gui_parity.py`), V14a, and M16 swung transmission (`probe.py`), requires `--multitau-fixture` for V19, and its `M20 import .h5 wrong` checks `get_mask()` (unchanged all-ones).
   - **Why it matters:** Task 9 Step 2's gate misses M16, M18, V14a, and V19 unless exact scripts/flags are specified.
   - **Suggested fix:** Map M16 in Spec L112-116 and list exact Task 9 Step 2 commands for `probe.py`, `probe_core_parity.py --multitau-fixture ...`, and `probe_mask_gui_parity.py`.

## Verified claims

- `xpcsviewer/__init__.py:80-83` and `logging_config.py:140-147` look up `"xpcsviewer"` instead of `"xpcsviewer-gui"` (`pyproject.toml:6`); `cli_main.py:154-158,182,266` reads `xpcsviewer.__version__`.
- `fitting/legacy.py:262,391,596` fall back to `np.mean(bounds, axis=0)`; failure placeholders sit at L313, L436, L547-551, L657.
- `average_toolbox.py:183-185` returns early in `"running"` state; `xpcs_viewer.py:3274-3328` and `viewer_kernel.py:900-959` do not open outputs on `finished`.
- `area_mask.py:172-218` and `simplemask_kernel.py:246-261` return `True` on shape mismatch; `MaskAssemble.apply` (L415-420) ANDs onto the combined mask.
- `simplemask_kernel.py:367-416` pastes `getArrayRegion` blocks, reproducing the angled-line (`rows 2-4, cols 4-19`) and vertical-line (`rows 3-6, cols 8-22`) defects.
- Task 7 ROI expectations are empirically exact with `polygon2mask` + `mapToItem` (offscreen, with or without a view): `RectROI((7,5),(10,8))` → rows 5-12, cols 7-16 (80 px); `LineROI((10,3),(10,18),4.)` → rows 3-17, cols 8-11 (60 px); angled line, `CircleROI` (subclass of `EllipseROI`), `EllipseROI`, `PolyLineROI`, rotated `RectROI`, and zero-length `LineROI` (0 px, no crash) all match.
- Task ordering (1→9) has no dependency inversions (Task 6 precedes Task 7; mask-layer and I/O work are cleanly separated).
