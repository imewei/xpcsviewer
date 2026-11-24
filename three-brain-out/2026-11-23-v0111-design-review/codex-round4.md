## Verdict

READY

The round-three blockers are resolved. The revised plan and spec are ready for implementation.

## Blocking findings

None.

## Non-blocking findings

1. **Wheel verification · Plan:172,238,245.** Use the throwaway interpreter with `-I` for both wheel checks. The checkout working directory and the exported `PYTHONPATH` can otherwise make Python import the source tree while reading the installed wheel's distribution metadata.

## Verified claims

- **Task 9 · RESOLVED.** Plan:240–245 restores the recorded basename, requires all recorded keys except the named omissions, allows M30's version attribute to move, and sets an absolute `1e-11` tolerance for M15. The cross-import key is basename-derived, and the export writes the package version (`probe_core_parity.py:201–205,446–447`, `core-local-viewer.json:2,22`, `simplemask_kernel.py:302–303`).
- **Task 2 · RESOLVED.** Plan:75 cites the `fit_tauq` fixture as a starting point and explicitly adapts it for `fit_g2` with cache setup and a `fit_val` assertion (`probe_core_parity.py:328–344`, `xpcs_file.py:1690–1699,1822–1858`).
- **Task 3 · RESOLVED.** Plan:101 forces the CPU, file-count, and memory conditions, resets `n_accepted` at entry, and explicitly requires `n_accepted` and `etime` before terminal status and its signal (`average_toolbox.py:153–171,231–238`).
- **Task 5 · RESOLVED.** Plan:152 names only the clamped effective cap and preserves the blank reset baseline required by Spec:210.
- **Task 6 · RESOLVED.** Plan:172–173 includes `LICENSE-APACHE-2.0` in both the wheel file list and the commit, consistent with Spec:185–192.

This was a document review. Tests, wheel builds, and probe reruns were skipped.
