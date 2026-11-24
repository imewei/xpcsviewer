## Verdict
NOT READY

Most Round-4 additions fit cleanly, but two blocking defects were introduced: Task 7's `test_rotated_ellipse_matches_upstream` omits pyqtgraph's corner-pivot shift and upstream `ellipse_vertices`'s opposite angle sign (46–78 pixel mismatches), and Task 9 Step 2's intended-change list mislabels `V19` as version and flags legitimate `q` shifts (`M15`, `unrotated transmission`) as regressions.

## Blocking findings

1. **B1** | High | Plan `L198` (Task 7 `test_rotated_ellipse_matches_upstream`)
   - **Evidence:** `pg.EllipseROI((6,7),(12,6)).setAngle(40)` pivots around local `(0,0)` (item `(6,7)`), shifting its center from `(row=9.5, col=11.5)` to `(12.655, 8.168)`. Upstream `ellipse_vertices` (`src/pysimplemask/core/rasterize.py:35-43`) also rotates `(row, col)` with opposite sign to pyqtgraph's `(x, y)` `setAngle`. Empirically against `rasterize((24, 28), ...)`: `ellipse_vertices((9.5,11.5),(3.0,6.0), 40)` differs by **78 pixels**; at the rotated center `(12.655, 8.168)`, `angle_deg=+40` still differs by **46 pixels**; only `angle_deg=-40` at the rotated center matches (**0 diff**).
   - **Why it matters:** A literal test fails against a correct `roi_to_polygon`, forcing guesswork on rotation conventions.
   - **Suggested fix:** Specify `roi.setAngle(40, center=[0.5, 0.5])` against `ellipse_vertices((9.5, 11.5), (3.0, 6.0), angle_deg=-40.0)` (or compute the rotated center around `(6, 7)` with `angle_deg=-40.0`), noting the handedness flip.

2. **B2** | Medium | Plan `L240-241` (Task 9 Step 2 intended-change list)
   - **Evidence:** Plan `L240` writes `V19 (version)` (V19 is **averaging** in `parity-audit.md:94`; version is **V30**), says `M16 swung-transmission rows (q)` (`swung transmission` previously raised `TypeError`, so all maps appear, while `unrotated transmission` shifts in `q` by `2.9e-9`), and requires every other key to be identical. Empirically, `M15 geometry update` (`probe_core_parity.py:138-144`) returns `tool.qmap["q"][20:23, 24:27]`, which shifts by `1.05e-12` under `2*k0*sin(alpha/2)` (`array_equal == False`). Conversely, `M20 import .h5 wrong` (`probe_core_parity.py:58-64`) and `fixed beta and baseline` (`probe.py:123-131`) do not change in those JSON diffs.
   - **Why it matters:** The gate falsely flags `M15` and `unrotated transmission` as regressions and confuses V19 with V30.
   - **Suggested fix:** List the JSON diffs that actually move—`swung transmission` (replaces `TypeError`), `unrotated transmission` (`q` within `1e-8`), `M15 geometry update` (`q` within `1e-11`), `M18` (`gui-local-mask.json`), `M21 M22` (`core-local-mask.json`), `V19 averaging excluded` (`core-local-viewer.json`)—and note M20/V14a/V30 are checked via unit/wheel tests.

## Non-blocking findings

1. **N1** | Low | Plan `L101-102` (Task 3 `_poll_avg_job`)
   - **Evidence:** `XpcsViewer.remove_avg_job` (`xpcs_viewer.py:3166-3169`) clears `vk.avg_worker = None` without stopping `self.timer`.
   - **Why it matters:** `_poll_avg_job` could raise `AttributeError` if a job is removed while polling, or read `"finished"` before `n_accepted` is set.
   - **Suggested fix:** Guard `if self.vk is None or self.vk.avg_worker is None: self.timer.stop(); return`; initialize `n_accepted = 0` in `__init__` and write `n_accepted` before `status = "finished"`.

2. **N2** | Low | Plan `L226` (Task 8 JAX zero-swing branch)
   - **Evidence:** A Python `if swing_h == 0.0 and swing_v == 0.0:` inside `@jax.jit` raises `TracerBoolConversionError` on traced args.
   - **Why it matters:** Minor implementer trap on the JAX path.
   - **Suggested fix:** Dispatch the zero-swing identity branch outside `@jax.jit`.

3. **N3** | Low | Plan `L50,148,171` (minor pointers)
   - **Evidence:** `git grep -n "cli_main" tests/unit` has 0 matches (`test_cli_version_reports_package_version` also passes before Task 1 since `cli_main.py:154-158` already reads `__version__`); Plan `L148` says today's trim lands on "draw 4" (actual: draw 3); Plan `L171` should run `"${TMPDIR:-/tmp}/wv/bin/python"`.
   - **Why it matters:** Minor implementer friction.
   - **Suggested fix:** Drop the empty grep, fix "draw 4" → "draw 3", qualify the venv `python` path.

## Verified claims

- **Task 1 (`L49`):** `test_logging_banner_falls_back_when_metadata_missing` fails before (`"unknown"`, `logging_config.py:147`) and passes after.
- **Task 2 (`L70-71,79`):** `fit_val[0, 0, 2]` matches `legacy.py:267`; `test_failed_fit_keeps_fixed_slots` fails before (`L313,436,657` overwrite fixed slots) and passes after.
- **Task 3 (`L98-99,102`, Spec `L206`):** `tests/conftest.py:132-134,220-222` sets `WARNING`; `_poll_avg_job` test fails before/passes after; empty-`flist` note matches `average_toolbox.py:80`.
- **Task 4 (`L120,125`):** `MaskBase.evaluate -> bool | None` (`area_mask.py:82`) keeps all subclass overrides mypy-clean; `test_scalar_and_1d_datasets_return_false` fails before (`np.swapaxes` `AxisError`, `area_mask.py:206`) and passes after.
- **Task 5 (`L145-151`, Spec `L208`):** Popping `mask_ptr_min + 1` (`max_history=4`, 6 draws) retains `[base, d4, d5, d6]` (3 undos → d5, d4, base; 4th no-op; reset → `[base]`); both new tests fail before/pass after.
- **Task 6 (`L171`):** `uv build --wheel --out-dir` flags verified via `uv build --help`.
- **Task 7 (`L184,187,195-202`, Spec `L207`):** 1-point `PolyLineROI` and 0-width `RectROI` construct cleanly (area `0.0`); `pending_roi_count` + `_on_apply_drawing` guard (`simplemask_window.py:332-352`) matches Spec `L207`.
- **Task 8 (`L218-219,226`):** Pre-change golden, rotated-vector component test, and `NOTICE` entry align with Task 6 and `qmap.py`.
