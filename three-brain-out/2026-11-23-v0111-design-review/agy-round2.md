## Verdict
NOT READY

Round-1 items B3, B4, B5, B6, N1, and N2 are resolved, and B1's core design is fixed, but three blocking defects remain in the revised plan: Task 2's new `fit_g2_high_performance` change uses wrong `tau_indices` and calls dead broken code, Task 8's new tests fail on a wrong `E2KCONST` literal and skip `cache_clear()`, and Task 9 Step 2's probe commands fail on unset `$TMPDIR`/missing `$out` and omit `PYTHONPATH` on command 3.

## Blocking findings

1. **B1** | High | Plan `L65,73,76` (Round-1 **B2: NOT RESOLVED**)
   - **Evidence:** Spec `L99-111` and Plan `L59-60,74` fix the UI premise and `test_fixed_param_without_p0_uses_midpoint`, but `fit_g2_high_performance` (`xpcsviewer/xpcs_file.py:2045-2052`) selects `legacy.single_exp(x, tau, bkg, cts)` (`legacy.py:40`: 3 params, `tau` at index **0**, `bkg` at **1**) and `legacy.double_exp(x, tau1, bkg, cts1, tau2, cts2)` (`legacy.py:56`: 5 params, `tau` at **[0, 3]**)—not `single_exp_all`/`double_exp_all` (`[1]` / `[1, 4]`). Worse, `fit_g2_high_performance` passes `fit_val` (a 3D `np.ndarray`) to `_convert_optimized_results_to_xpcs_format`, which crashes unconditionally at `xpcs_file.py:2160` (`fit_result["status"]` → `IndexError`). Also `git grep -n "fit_g2_high_performance" tests` returns zero matches (contradicting Plan L73).
   - **Why it matters:** Passing `tau_indices=[1]` takes the geometric mean of `bkg` (failing Plan L73's midpoint assertion), and calling `fit_g2_high_performance` crashes at L2160 even with valid `p0`.
   - **Suggested fix:** Map `tau_indices` by actual model (`[0]` for `single_exp`, `[0, 3]` for `double_exp`, `[1]`/`[1, 4]` for `*_all`). Either fix/remove the dead `_convert_optimized_results_to_xpcs_format` path in scope or test `p0` forwarding by stubbing `fit_with_fixed*` and `_convert_optimized_results_to_xpcs_format`; drop the nonexistent `git grep` hint.

2. **B2** | High | Plan `L204,206` (New in Task 8 tests; Round-1 **B1: RESOLVED** in design)
   - **Evidence:** `xpcsviewer/simplemask/qmap.py:26` and upstream `qmap.py:9` define `E2KCONST = 12.39841984`, not `12.398419843320026`. At Plan L204's geometry (`det_dist=1.0`, `pix_dim=0.075`, corner `(0,0)` has `alpha=0.209 rad`), `2*k0*sin(alpha/2)` differs by `2.73e-10`—270× larger than the `1e-12` tolerance. Separately, `_compute_transmission_qmap_cached` (`qmap.py:442-451`) does not include the backend in its `lru_cache` key, so Plan L206's NumPy-then-JAX comparison without `compute_transmission_qmap.cache_clear()` returns the cached NumPy result without invoking `_get_transmission_qmap_jit`.
   - **Why it matters:** `test_q_is_momentum_transfer_magnitude` fails on a correct implementation, and `test_jit_and_numpy_backends_agree_with_swing` is vacuous.
   - **Suggested fix:** Use `E2KCONST` (`12.39841984`) in Plan L204 and require `compute_transmission_qmap.cache_clear()` between backend switches in Plan L206.

3. **B3** | Medium | Plan `L219-222` (Round-1 **N3: NOT RESOLVED**)
   - **Evidence:** On this host `$TMPDIR` is unset, so `out=$TMPDIR/stage4-probes` expands to `/stage4-probes`, and none of `run_probes.py:34`, `probe_core_parity.py:471`, or `probe_mask_gui_parity.py:167` creates `$out` (`FileNotFoundError`). Moreover, command 3 (`<venv python> $audit/probe_mask_gui_parity.py ...`) omits `PYTHONPATH=<this worktree>`; empirically, running `<venv python>` with `sys.path[0]=$audit` imports `/home/wei/Documents/GitHub/xpcsviewer/xpcsviewer/__init__.py` (main checkout), not the worktree.
   - **Why it matters:** The Task 9 Step 2 commands fail as written and command 3 silently tests the unedited main repo.
   - **Suggested fix:** Set `out="${TMPDIR:-/tmp}/stage4-probes"; mkdir -p "$out"`, export the env vars, and add `PYTHONPATH=<this worktree>` to command 3.

## Non-blocking findings

1. **N1** | Low | Plan `L130,140` (Task 5 `_layers_at` index drift)
   - **Evidence:** If a caller mutates `mask_record` directly (`len=2`, `mask_ptr=1`, `len(layer_record)=1`) and calls `apply("mask_draw")`, appending one entry to `layer_record` yields `len(layer_record)=2` while `mask_ptr=2`, shifting subsequent lookups by 1.
   - **Why it matters:** A second draw after direct mutation would read the wrong slot.
   - **Suggested fix:** Before appending in `apply()`, pad `layer_record` to `mask_ptr + 1` using `{"accumulated": mask_record[i]}`.

2. **N2** | Low | Plan `L93-96,169,199`
   - **Evidence:** `average_toolbox.py:369-370` (`_process_files_parallel`, `tot_num >= 8`) returns on `is_killed.is_set()` without setting `self.status = "killed"` (unlike L266, L385). Plan L199 states relative `q` shift is `3e-6` at the audit probe geometry (`alpha=1.52e-4 rad`), whereas `alpha^2/8 = 2.87e-9`. Plan L169 still cites `apply_drawing` as `~L332-380` (actual `367-416`).
   - **Why it matters:** Pre-killed parallel jobs (`>=8` files) stay `"running"`; minor doc inaccuracies.
   - **Suggested fix:** Set `if self.is_killed.is_set(): self.status = "killed"; return` before `np.sum(mask) == 0`; update `2.9e-9` and `~L367-416`.

## Verified claims

- **Round-1 status:** **B1** RESOLVED (Spec L121-133, Plan L192-211; new test bugs tracked in Round-2 B2); **B2** NOT RESOLVED (see B1 above); **B3** RESOLVED (Spec L85-92,203-205, Plan L20,105,116-119; no tests reference `_validate_mask_dimensions` or `"Load Failed"`); **B4** RESOLVED (Plan L130,139-142: independent length guards + `max_history=4` gives `[d3,d4,d5,d6]`, 3 undos → 5th, 4th, 3rd, 4th undo no-op); **B5** RESOLVED (Spec L206-209, Plan L83,93-96: sequential killed guard + `%s` for `jid`); **B6** RESOLVED (Spec L187-190, Plan L162: `[tool.setuptools] license-files` verified via setuptools 84.2.1 `_apply` + `_finalize_license_files` → `['LICENSE', 'NOTICE']`, emits deprecation warning only); **N1** RESOLVED (Plan L39,48-51); **N2** RESOLVED (Plan L21,83,174,189); **N3** NOT RESOLVED (M16 mapped at Spec L121, but Task 9 Step 2 commands fail—see B3).
- **New claims verified:** `initial_fit_guess([[0.1,1e-3,0.5],[1.0,1e1,1.5]], [1]) == [0.55, 0.1, 1.0]` is exact; `q = 2*k0*sin(alpha/2)` matches upstream `qmap.py:133`; `h5py` in `simplemask_window.py:12` becomes unused after removing `_validate_mask_dimensions` (L481 was its only use, matching Plan L119's cleanup note); `$scratch/browser-fixtures/S0001_synthetic_a0001_f000400_r00001_multitau.hdf` exists and `run_probes.py` runs `probe.py` with 6 args once `$out` exists.
