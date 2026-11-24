## Verdict
READY
All Round 7 edits are accurate and consistent with the codebase at `e3dab2b` and across tasks; no blocking issues remain.

## Blocking findings
none

## Non-blocking findings
1. **N1** (Low) — Plan L243 (`Task 9 Step 2`).
   - **Evidence:** `xpcsviewer/simplemask/simplemask_kernel.py:39,303` sets `group_handle.attrs["version"] = __version__` from module-local `__version__ = "1.0.0"`, not `xpcsviewer.__version__`.
   - **Why it matters:** Task 1 will not change `M30`'s `root_attrs.version` (it stays `"1.0.0"`). Because L243 says "may differ", it is harmless, just factually inaccurate.
   - **Suggested fix:** Drop the `root_attrs.version` caveat (or note it stays `"1.0.0"`).

## Verified claims
- **Task 2 (L75):** Fixture wording accurately references `probe_core_parity.py ~L328-344` (`fit_tauq`) adapted for `fit_g2` and `xpcs_file.py ~L1690-1699`.
- **Task 3 (L101):** Patching `multiprocessing.cpu_count` to `>= 2` matches `average_toolbox.py:154-155`; resetting `self.n_accepted = 0` at `do_average` entry avoids stale values on reuse; assigning `n_accepted` and `etime` before `status = "finished"` and `signals.status.emit` fixes the `average_toolbox.py:231-233` poll race.
- **Task 5 (L152):** Conflicting old cap parenthetical removed; only `max(self.max_history, 2) + mask_ptr_min` remains.
- **Task 6 (L173):** Commit list includes `LICENSE-APACHE-2.0`, matching Step 4 (L172).
- **Task 7 (L185):** `fewer than 3 vertices` guard before `v.mean(axis=0)` prevents empty-array `RuntimeWarning`/NaN.
- **Task 9 (L240, L242-243):** `core-local-mask-qmap.hdf` matches the recorded M30 key basename; required-keys rule matches skipped APS/second-M30 fixtures; `1e-11` absolute tolerance for `M15` covers the measured `1.05e-12` shift.
