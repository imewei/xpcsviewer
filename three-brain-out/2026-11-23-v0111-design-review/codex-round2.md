## Verdict

NOT READY

The revision fixes the main fit and import issues and specifies the swing calculation. The averaging poll can dereference a removed job, the history cap has a one-entry edge case, and the audit comparison rejects expected changes.

## Blocking findings

1. **B1 · High · NOT RESOLVED · Plan:136–151.** Hidden edits now create history entries. The new trim correctly yields `[base,d4,d5,d6]` at cap 4, with undo reaching d5, d4, then base. At cap 1, popping index 1 removes the just-applied state, so `apply` returns a mask that history immediately loses (`xpcsviewer/simplemask/area_mask.py:352–392,427–440`). Define and test a minimum capacity that retains base and current. Correct the old-reset expectation at Plan:148: the old cap 4 retains cumulative states 3–6, so reset lands on state 3.

2. **B4 · High · NOT RESOLVED · Plan:91–102.** The poll is a valid GUI-thread delivery path, but removing a job clears `avg_worker` without stopping the timer (`xpcsviewer/xpcs_viewer.py:3166–3169,3253–3272`, `viewer_kernel.py:910–915`). Guard absent kernel/worker and test removal while polling. The new kill guard covers the parallel early return at `module/average_toolbox.py:368–370`, but exits before setting `n_accepted`, contradicting “every call.” Define and test the killed value and publish completion fields before terminal status. Force the parallel branch in the kill test because memory pressure can override the file-count threshold (153–171).

3. **B6 · High · NOT RESOLVED · Spec:207, Plan:184–202.** Zero-length and no-pixel filtering fixes the original rasterizer problem. However, `pending_roi_count` still counts candidates the rasterizer rejects. With only off-image or degenerate candidates, the window applies an all-True mask and clears the previous draw layer (`simplemask_window.py:340–346`), contrary to “ignored.” Specify and test that window case with an existing layer. Also avoid treating signed shoelace cancellation as proof of degeneracy. A self-crossing polyline can have zero signed area and fill pixels (`pyqtgraph_mod.py:87–106`, pinned `core/rasterize.py:78–84`).

4. **N2 · High · NOT RESOLVED · Plan:236–241.** The GUI and swing invocations are now identified. The allowlist misses unrotated M16 and M15, which returns raw transmission `q` (`probe.py:84–91`, `probe_core_parity.py:138–144`). V19 is averaging and V30 is version (`parity-audit.md:94,105`), which the runner does not test (`run_probes.py:38–47`). M20 reads `MaskFile.get_mask()` and V14a already supplies explicit starts, so those probe values can stay identical (`probe_core_parity.py:58–64`, `probe.py:123–131`). Compare like-for-like inputs: the recorded JSON has APS keys and two M30 cross-import files, while these commands supply no APS inputs and one differently named file (`probe-results.json:1518`, `core-local-viewer.json:2,22`, `probe_core_parity.py:446–447`). Define expected differences by field and fixture, including affected downstream partitions, and verify the wheel version separately.

## Non-blocking findings

1. **N1 · Medium · NOT RESOLVED · Spec:185–190, Plan:171,226.** Both ports now get attribution and the wheel checks are explicit. Task 6 still calls for an Apache text *reference*, while the spec requires the text itself. Include the pinned Apache license text and verify the wheel contains it alongside MIT `LICENSE` (`pyproject.toml:10`, pinned `core/qmap.py:1–2`).

2. **N4 · Medium · Plan:61,76,79,198,218–226.** The converter raises on a nonempty solver ndarray, not “any input” (`xpcs_file.py:2147–2160`). The cited stub probes `fit_tauq`. A `fit_g2` fixture also needs cache configuration and asserts the returned summary’s `fit_val` (`probe_core_parity.py:328–344`, `xpcs_file.py:1690–1699,1822–1858`). For rotated ellipse goldens, rotate the centre about the ROI origin and reverse the angle when converting x/y to row/col (pinned `core/rasterize.py:35–43`). Capture exact zero-swing goldens on the same backend. For JAX, branch outside the JIT or use JAX control flow for traced angles (`qmap.py:138–159`).

3. **N5 · Low · Plan:221,233 · Wide-angle claim and release note.** The fixture reaches `atan2(sqrt(2)*0.15,1) ≈ 0.209` rad at its corners. The 0.15 estimate applies along a detector axis (`qmap.py:389–398`). The test still distinguishes the formulas. Make the changelog instruction explicit about saved zero-swing Q-maps and derived partitions, as Spec:130–132 requires.

## Verified claims

- **B2 · RESOLVED.** Your correction is right. Pinned `core/qmap.py:129–155` publishes float32 `qx` and `qy`; only `qr` and `qz` are commented out. Plan:214–226 correctly separates the magnitude formula, component goldens, tolerances and zero-swing path.
- **B3 · RESOLVED.** Plan:66–79 specifies the actual bounds-derived starts, fixes indexing and preserves fixed slots on failure (`xpcs_file.py:1732–1745`, `legacy.py:267,308–314,431–437,651–658`).
- **B5 · RESOLVED.** Plan:114–125 removes the modal preflight, handles transposes and rejects non-2D datasets before swapping axes. The base return union permits the worker override (`area_mask.py:82,191–213`, `simplemask_window.py:444–512`).
- **N3 · RESOLVED.** Plan:45–51,186–201 adds version preservation checks and routes ROI regressions through the existing kernel entry point. The explicit-version CLI test is expected to pass before the changes (`cli_main.py:154–158,182`). The offscreen one-point polyline fixture constructed successfully.
- The default UUID and two incompatible `%d` log calls are confirmed (`average_toolbox.py:63–66,124,239`). Matching and transposed imports, explicit-start fitting, and the rectangle ROI are preservation cases. Full suites, builds and audit reruns were skipped.
