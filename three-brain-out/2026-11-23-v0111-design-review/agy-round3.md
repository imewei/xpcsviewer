## Verdict
READY

All Round-2 blocking findings (B1-B3) and non-blocking findings (N1-N2) are resolved cleanly across both documents, with no new blocking defects introduced. Both the spec and the Stage 4 implementation plan are internally consistent, aligned with the codebase at `e3dab2b`, and actionable without guesswork.

## Blocking findings

none

## Non-blocking findings

1. **N1** | Low | Plan `L162,189`
   - **Evidence:** Task 6 Step 4 (`output under $TMPDIR`) and Task 7 Step 5 (`$TMPDIR/gui-local-mask.json`) still use bare `$TMPDIR`, whereas `$TMPDIR` is unset on this host (fixed to `${TMPDIR:-/tmp}` in Task 9 Step 2, Plan `L219`).
   - **Why it matters:** Minor shell friction if copied verbatim.
   - **Suggested fix:** Use `${TMPDIR:-/tmp}` at Plan `L162` and `L189`.

2. **N2** | Low | Plan `L73`
   - **Evidence:** `git grep -n "fit_g2(" tests` matches `tests/unit/core/test_xpcs_file_fitting.py`, which uses a `MagicMock` rather than a real `XpcsFile.fit_g2` call.
   - **Why it matters:** Minor implementer detour when looking for a real `XpcsFile` fixture; `XpcsFile.__new__(XpcsFile)` (as in `probe_core_parity.py:333`) works directly.
   - **Suggested fix:** Optional note to stub `get_g2_data` + `_memory_manager` on `XpcsFile.__new__(XpcsFile)` if no real fixture is handy.

## Verified claims

- **Round-2 B1: RESOLVED** — Spec `L106-108` and Plan `L59,65,73` explicitly leave dead/broken `fit_g2_high_performance` (`xpcsviewer/xpcs_file.py:1980-2160`) untouched, restrict `initial_fit_guess` (`[1]` / `[1, 4]`) to `fit_g2` (`single_exp_all`/`double_exp_all`), and redirect Task 2's test to `fit_g2`.
- **Round-2 B2: RESOLVED** — Plan `L204` uses the exact module constant `E2KCONST = 12.39841984` (`xpcsviewer/simplemask/qmap.py:26`), making `2*k0*sin(TTH_rad/2)` match to `< 1e-12`; Plan `L206` mandates `compute_transmission_qmap.cache_clear()` before each backend run so the JAX-JIT path is genuinely exercised.
- **Round-2 B3: RESOLVED** — Plan `L219-222` sets `out="${TMPDIR:-/tmp}/stage4-probes"; mkdir -p "$out"` and exports `QT_QPA_PLATFORM=offscreen JAX_PLATFORMS=cpu MPLBACKEND=Agg PYTHONPATH=<this worktree>`. Empirically verified: all three probe scripts' `--help` CLI signatures match the exact arguments in Plan `L220-222`, `$scratch/browser-fixtures/S0001_synthetic_a0001_f000400_r00001_multitau.hdf` exists, and exported `PYTHONPATH=<this worktree>` resolves `xpcsviewer` to `<this worktree>/xpcsviewer/__init__.py` even when invoking a script under `$audit`.
- **Round-2 N1: RESOLVED** — Plan `L130` pads `layer_record` up to `mask_ptr + 1` via `_layers_at` before appending in `apply`, preventing index drift after direct `mask_record` mutation.
- **Round-2 N2: RESOLVED** — Plan `L96` sets `if self.is_killed.is_set(): self.status = "killed"; return` (covering `_process_files_parallel` L369) and adds `test_killed_parallel_job_is_marked_killed`; Plan `L169` cites `apply_drawing` at `~L367-416`; Plan `L199` corrects the audit-geometry relative `q` shift to `about 3e-9` (`alpha^2/8 = 2.87e-9`).
