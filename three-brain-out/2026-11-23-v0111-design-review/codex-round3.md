## Verdict

NOT READY

The history, poll, and ROI revisions address the core issues. The audit gate still skips the regenerated M30 cross-import and treats an expected version change as a regression.

## Blocking findings

1. **N2 · NOT RESOLVED · High · Plan:240–245.** Step 3 now verifies wheel metadata separately, and the expected-move list correctly includes both M16 cases, M15, and V19. However, the cross-import key uses the input basename, so passing `q-loc.hdf` drops the regenerated artifact from the intersection with the recorded `core-local-mask-qmap.hdf` row (`probe_core_parity.py:429–447`, `core-local-viewer.json:2,22`). Keep the recorded basename and require all expected keys except explicitly skipped fixtures. Also allow M30's `root_attrs.version` to change with package metadata independently of `q` (`simplemask_kernel.py:302–303`, `probe_core_parity.py:201–205`). Specify absolute tolerance for M15's `1e-11`.

## Non-blocking findings

1. **N4 · NOT RESOLVED · Low · Plan:75,199,220,228.** The centre-pivot ellipse with a negated row/column angle, same-backend goldens, and pre-JIT or static branch are specified correctly (`pyqtgraph ROI.py:342–369`, upstream `core/rasterize.py:35–43`, `qmap.py:138–159`). The fit fixture now names cache setup and the summary's `fit_val`, but its reference calls `fit_tauq` (`probe_core_parity.py:328–344`). Say to adapt that fixture for `fit_g2` (`xpcs_file.py:1690–1699,1822–1858`).

2. **N6 · Medium · Plan:101,152,173.** Force `multiprocessing.cpu_count() >= 2` as well as file count and memory conditions (`average_toolbox.py:153–171`). Reset `n_accepted` at run entry if supporting repeated direct calls, and set `etime` before status emission (110–127,231–238). Drop the conflicting unclamped cap in the first sentence of Plan:152. Include `LICENSE-APACHE-2.0` explicitly in the Task 6 commit list.

## Verified claims

- **B1 · RESOLVED.** The effective cap and cap-one test preserve baseline and current. Cap four yields `[base,d4,d5,d6]`. The old implementation retained cumulative states 3–6 and reset to state 3 (Plan:147–152, `area_mask.py:427–439,473–477`).
- **B4 · RESOLVED.** The poll guards both missing objects and tests removal. The fresh-job kill guard preserves zero accepted and suppresses successful completion. The parallel fixture now explicitly forces the branch, subject to N6 (Plan:96–102, `xpcs_viewer.py:3166–3169`, `viewer_kernel.py:910–915`, `average_toolbox.py:368–370`).
- **B6 · RESOLVED.** Shared drawable filtering protects the existing layer when every candidate is ignored. Centred rank admits the bow-tie while rejecting collinear vertices. Both cases have tests (Plan:185,197–204, Spec:209, `simplemask_window.py:340–346`, upstream `core/rasterize.py:78–84`).
- **N1 · RESOLVED.** Spec:185–192 and Plan:172 require the full Apache text, attribution and wheel verification alongside MIT `LICENSE` (`pyproject.toml:10`).
- **N5 · RESOLVED.** The corner reaches approximately 0.209 rad. The changelog now calls out saved maps and q-based partitions (Plan:223,246, `qmap.py:389–398`). Tests, builds, and audit reruns were skipped for this read-only review.
