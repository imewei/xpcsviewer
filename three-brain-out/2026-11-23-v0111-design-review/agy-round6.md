## Verdict
READY

Every Round-6 edit across Tasks 2, 3, 5, 6, 7, 8, 9, and the spec is sound, consistent with the codebase at `e3dab2b`, and free of blocking defects. All new tests fail before their respective changes and pass after, and Stage 4 is ready to implement.

## Blocking findings

none

## Non-blocking findings

1. **N1** | Low | Plan `L173` (Task 6 Step 5 commit list)
   - **Evidence:** Plan `L172` creates `LICENSE-APACHE-2.0` and adds it to `[tool.setuptools] license-files`, but `L173` lists only `(include pyproject.toml, uv.lock, NOTICE)`.
   - **Why it matters:** Leaving `LICENSE-APACHE-2.0` uncommitted breaks wheel builds in clean checkouts (`license-files` enforces matches).
   - **Suggested fix:** Add `LICENSE-APACHE-2.0` to `L173`'s commit list.

2. **N2** | Low | Plan `L101,185` (minor guards)
   - **Evidence:** In Task 7 (`L185`), calling `v.mean(axis=0)` on a 0-vertex `(0, 2)` array emits a `RuntimeWarning`/NaN; checking `len(v) < 3` first avoids it cleanly. In Task 3 (`L101`), `average_toolbox.py:154-155` also checks `multiprocessing.cpu_count() > 1` for the parallel branch.
   - **Why it matters:** Tiny edge-case hygiene (single-core containers / empty `PolyLineROI`).
   - **Suggested fix:** Guard `len(v) < 3` before `matrix_rank` and patch `cpu_count` if needed.

## Verified claims

- **Task 2 (`L60,75`):** `_convert_optimized_results_to_xpcs_format` (`xpcs_file.py:2147-2160`) and `fit_g2` cache/memory-manager pattern (`xpcs_file.py:1685-1699`; same `XpcsFile.__new__` idiom as `probe_core_parity.py:328-344`) match the code.
- **Task 3 (`L98,101`):** `test_poll_stops_timer_when_worker_removed` and `test_killed_parallel_job_is_marked_killed` fail before and pass after; `n_accepted` stays `0` on kill without clobbering `"killed"`.
- **Task 5 (`L148,152`):** Effective cap `max(self.max_history, 2) + mask_ptr_min` with `max_history=1` keeps `[base, drawn]` (undo → `base`), whereas current `area_mask.py:437-439` evicts `base`; test fails before and passes after.
- **Task 6 & Spec (`Spec L185-193`, `Plan L172`):** `LICENSE-APACHE-2.0` + `NOTICE` + `LICENSE` under `[tool.setuptools] license-files` match between spec and plan.
- **Task 7 (`L185,201-202,204`):** Empirically verified: 0-length line and 0-width rect have `matrix_rank == 1` (skipped); bow-tie `[(4.5,4.5),(14.5,14.5),(14.5,4.5),(4.5,14.5)]` has `matrix_rank == 2` and fills 60 pixels (kept); `pending_roi_count` returning `len(_drawable_polygons())` ensures off-image/degenerate ROIs do not clear the prior draw layer.
- **Task 8 (`L220,223`):** Same-backend pre-change goldens avoid cross-backend rounding; corner `alpha` is `0.209 rad` (`~0.21 rad`).
- **Task 9 (`L243,245-246`):** Common-key JSON diffing, wheel-version Step 3, and `alpha^2/8` changelog wording align with Spec `L126-132`.
