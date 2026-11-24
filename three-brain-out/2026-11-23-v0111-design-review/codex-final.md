## Verdict

READY

No blockers in the four frozen changes.

- **Audit.** M30 records the kernel-local `1.0.0`. The corrected probe labels match the scripts (Plan:243–247, `simplemask_kernel.py:39,303`, `probe.py:84–91,116–118`, `probe_core_parity.py:89,144,447`).
- **Wheels.** Both checks use `-I`. Only Task 9 explicitly asserts venv provenance. Task 6 uses an isolated import from the throwaway venv (Plan:175,249).
- **Averaging.** The interfaces match the numbered steps, including the kill guard, output ordering and guarded GUI poll (Plan:90,101–105, `average_toolbox.py:183–185,213–238`).
- **Files and ROI.** Task 6 creates both license artifacts, Task 8 updates `NOTICE`, and Task 7 shares its drawable filter with the window guard (Plan:162–163,207,216, Spec:185–192,209).

Tests, builds and probes were reviewed by inspection.
