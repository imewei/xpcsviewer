"""Regression tests for review round 2: R2-05, R2-16, R2-17, R2-21 (kernel),
R2-26, R2-32, R2-33, R2-34."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.unit.test_review_r4_cli_average import fake_qmap_from_labels

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]


def _load_script(name):
    path = REPO / "scripts" / "validation" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"r2_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- twotime CLI
class _StubTwotimeFile:
    labels: list[str] = []

    def __init__(self, path):
        self.atype = ["Twotime"]

    def get_twotime_qbin_labels(self):
        return self.labels

    # numeric axes the CLI reads (it no longer parses coordinates from labels)
    @property
    def c2_processed_bins(self):
        return fake_qmap_from_labels(self.labels)[0]

    @property
    def qmap(self):
        return fake_qmap_from_labels(self.labels)[1]

    def get_twotime_c2(self, selection=None, correct_diag=True):
        return {"c2_mat": np.eye(8), "delta_t": 1.0}

    def clear_cache(self):
        pass


def _args(root, out, **kw):
    base = {
        "input": str(root),
        "output": str(out),
        "q": 0.05,
        "phi": None,
        "q_phi": None,
        "format": "png",
        "dpi": 40,
    }
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def tb(monkeypatch):
    from xpcsviewer.cli import twotime_batch

    monkeypatch.setattr(twotime_batch, "XpcsFile", _StubTwotimeFile)
    return twotime_batch


def test_r2_26_a_slash_b_and_a_dunder_b_do_not_collide(tb, tmp_path):
    root, out = tmp_path / "in", tmp_path / "out"
    (root / "a").mkdir(parents=True)
    (root / "a" / "b.h5").write_bytes(b"")
    (root / "a__b.h5").write_bytes(b"")
    _StubTwotimeFile.labels = ["qbin=1, q=0.0500, phi=10.0"]

    n = tb.process_directory(str(root), _args(root, out))

    assert n == 2
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.png")) == [
        "a/b.h5_bin0000_q0.0500_phi10.0.png",
        "a__b.h5_bin0000_q0.0500_phi10.0.png",
    ]


def test_r2_26_name_keeps_extension_so_stem_and_ext_variants_differ(tb, tmp_path):
    root, out = tmp_path / "in", tmp_path / "out"
    root.mkdir()
    (root / "run.h5").write_bytes(b"")
    (root / "run.hdf").write_bytes(b"")
    _StubTwotimeFile.labels = ["qbin=1, q=0.0500, phi=10.0"]

    assert tb.process_directory(str(root), _args(root, out)) == 2
    assert len(list(out.rglob("*.png"))) == 2


def test_r2_26_repeated_output_path_is_refused_not_counted(tb, tmp_path, monkeypatch):
    root, out = tmp_path / "in", tmp_path / "out"
    root.mkdir()
    (root / "x.h5").write_bytes(b"")
    (root / "y.h5").write_bytes(b"")
    _StubTwotimeFile.labels = ["qbin=1, q=0.0500, phi=10.0"]
    monkeypatch.setattr(tb, "generate_output_filename", lambda *a, **k: "same.png")

    n = tb.process_directory(str(root), _args(root, out))

    assert n == 1
    assert [p.name for p in out.rglob("*.png")] == ["same.png"]


# ---------------------------------------------------------------------- R2-05
def test_r2_05_q_selection_works_without_phi_in_labels(tb):
    _StubTwotimeFile.labels = ["q=0.05000 A^-1", "q=0.10000 A^-1"]
    got = tb.find_qbins_for_q(_StubTwotimeFile("x"), 0.05)
    assert [(i, q, phi) for i, _lbl, q, phi in got] == [(0, 0.05, None)]


def test_r2_05_radial_only_batch_writes_image(tb, tmp_path):
    root, out = tmp_path / "in", tmp_path / "out"
    root.mkdir()
    (root / "run.h5").write_bytes(b"")
    _StubTwotimeFile.labels = ["q=0.05000 A^-1", "q=0.10000 A^-1"]

    assert tb.process_directory(str(root), _args(root, out)) == 1
    assert [p.name for p in out.rglob("*.png")] == ["run.h5_bin0000_q0.0500.png"]


def test_r2_05_phi_stays_mandatory_for_angle_modes(tb):
    _StubTwotimeFile.labels = ["q=0.05000 A^-1"]
    xf = _StubTwotimeFile("x")
    assert tb.find_qbins_for_phi(xf, 10.0) == []
    assert tb.find_qbin_for_qphi(xf, 0.05, 10.0) is None


# ---------------------------------------------------------------------- R2-32
@pytest.mark.parametrize(
    "kw",
    [
        {"q": float("nan")},
        {"q": float("inf")},
        {"q": None, "phi": float("nan")},
        {"q": None, "phi": float("inf")},
        {"q": None, "q_phi": "nan,10"},
        {"q": None, "q_phi": "0.05,nan"},
        {"q": None, "q_phi": "0.05,inf"},
    ],
)
def test_r2_32_non_finite_selector_is_rejected(tb, tmp_path, kw):
    root, out = tmp_path / "in", tmp_path / "out"
    root.mkdir()
    (root / "run.h5").write_bytes(b"")
    _StubTwotimeFile.labels = ["qbin=1, q=0.0500, phi=10.0"]

    assert tb.run_twotime_batch(_args(root, out, **kw)) != 0
    assert not out.exists()


def test_r2_32_finite_selector_still_runs(tb, tmp_path):
    root, out = tmp_path / "in", tmp_path / "out"
    root.mkdir()
    (root / "run.h5").write_bytes(b"")
    _StubTwotimeFile.labels = ["qbin=1, q=0.0500, phi=10.0"]

    assert tb.run_twotime_batch(_args(root, out)) == 0
    assert len(list(out.rglob("*.png"))) == 1


# ------------------------------------------------------------ viewer_kernel
def _g2_xf(label, fname, value, fit=False):
    fit_summary = None
    if fit:
        fit_summary = {
            "fit_func": "single",
            "q_range": "None",
            "t_range": "None",
            "bounds": None,
            "fit_flag": "[True]",
            "q_val": np.array([0.01]),
            "fit_val": np.full((1, 2, 4), value),
            "fit_line": np.full((1, 3), value),
            "fit_x": np.array([0.1, 0.2, 0.3]),
        }
    return SimpleNamespace(
        label=label,
        fname=fname,
        atype="Multitau",
        get_g2_data=lambda: (
            np.array([0.01]),
            np.array([0.1, 0.2, 0.3]),
            np.full((3, 1), value),
            np.full((3, 1), 0.01),
            ["a"],
        ),
        fit_summary=fit_summary,
    )


def _kernel(monkeypatch, xfs):
    from xpcsviewer.viewer_kernel import ViewerKernel

    vk = ViewerKernel.__new__(ViewerKernel)
    monkeypatch.setattr(vk, "get_xf_list", lambda **k: xfs, raising=False)
    return vk


def test_r2_34_same_display_label_does_not_overwrite_g2_export(tmp_path, monkeypatch):
    xfs = [
        _g2_xf("scan", "/d/scan.hdf", 1.5, fit=True),
        _g2_xf("scan", "/d/scan_results.hdf", 1.9, fit=True),
    ]
    _kernel(monkeypatch, xfs).export_g2(str(tmp_path))

    for stem, value in (("scan", 1.5), ("scan_results", 1.9)):
        data = np.loadtxt(tmp_path / f"{stem}_g2_data.txt")
        np.testing.assert_allclose(data[:, 1], value)
        assert (tmp_path / f"{stem}_g2_fitting.txt").exists()
        curves = np.loadtxt(tmp_path / f"{stem}_g2_fitted_curves.txt")
        np.testing.assert_allclose(curves[:, 1], value)
    assert len(list(tmp_path.glob("*.txt"))) == 6


def test_r2_34_identical_source_names_are_suffixed_deterministically(
    tmp_path, monkeypatch
):
    xfs = [
        _g2_xf("scan", "/d1/scan.hdf", 1.0),
        _g2_xf("scan", "/d2/scan.hdf", 2.0),
        _g2_xf("scan", "/d3/scan.hdf", 3.0),
    ]
    _kernel(monkeypatch, xfs).export_g2(str(tmp_path))

    for stem, value in (("scan", 1.0), ("scan_2", 2.0), ("scan_3", 3.0)):
        np.testing.assert_allclose(
            np.loadtxt(tmp_path / f"{stem}_g2_data.txt")[:, 1], value
        )


def _tauq_xf(label, fname, success, fit_val):
    return SimpleNamespace(
        label=label,
        fname=fname,
        atype="Multitau",
        fit_summary={
            "tauq_success": success,
            "tauq_fit_val": fit_val,
        },
    )


def test_r2_21_failed_tauq_fit_is_not_exported(tmp_path, monkeypatch):
    good = _tauq_xf("good", "/d/good.hdf", True, np.array([[2.0, -1.0], [0.1, 0.2]]))
    bad_zero = _tauq_xf("badz", "/d/badz.hdf", False, np.zeros((2, 2)))
    bad_nan = _tauq_xf("badn", "/d/badn.hdf", False, np.full((2, 2), np.nan))
    _kernel(monkeypatch, [bad_zero, good, bad_nan]).export_diffusion(str(tmp_path))

    assert (tmp_path / "good_diffusion_fitting.txt").exists()
    assert not (tmp_path / "badz_diffusion_fitting.txt").exists()
    assert not (tmp_path / "badn_diffusion_fitting.txt").exists()
    summary = (tmp_path / "diffusion_summary.txt").read_text()
    assert "good" in summary
    assert "badz" not in summary
    assert "badn" not in summary


def test_r2_21_export_fails_when_every_tauq_fit_failed(tmp_path, monkeypatch):
    bad = _tauq_xf("bad", "/d/bad.hdf", False, np.zeros((2, 2)))
    with pytest.raises(ValueError, match="fit failed"):
        _kernel(monkeypatch, [bad]).export_diffusion(str(tmp_path))
    assert not list(tmp_path.glob("*.txt"))


def test_r2_34_diffusion_export_does_not_overwrite_same_label(tmp_path, monkeypatch):
    a = _tauq_xf("scan", "/d/scan.hdf", True, np.array([[2.0, -1.0], [0.1, 0.2]]))
    b = _tauq_xf("scan", "/d/scan_results.hdf", True, np.array([[3.0, -1.0], [0, 0]]))
    _kernel(monkeypatch, [a, b]).export_diffusion(str(tmp_path))

    assert "2.000000e+00" in (tmp_path / "scan_diffusion_fitting.txt").read_text()
    text = (tmp_path / "scan_results_diffusion_fitting.txt").read_text()
    assert "3.000000e+00" in text


def test_r2_21_plot_tauq_does_not_display_failed_fit(monkeypatch):
    from xpcsviewer import viewer_kernel

    info_calls = []

    def _failed_xf():
        xf = _tauq_xf("bad", "/d/bad.hdf", False, np.zeros((2, 2)))
        xf.fit_tauq = lambda *a, **k: None
        xf.get_fitting_info = lambda mode: info_calls.append(mode) or "a = 0; b = 0"
        return xf

    xf = _failed_xf()
    vk = _kernel(monkeypatch, [xf])
    monkeypatch.setattr(
        viewer_kernel,
        "_get_module",
        lambda name: SimpleNamespace(plot=lambda *a, **k: None),
    )
    result = vk.plot_tauq(rows=[0])

    assert info_calls == []
    assert "failed" in result["bad"]


# ------------------------------------------------------------ validation scripts
@pytest.fixture
def runner(tmp_path, monkeypatch):
    mod = _load_script("comprehensive_validation_runner")
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    return mod


def _fixed_accuracy(orch):
    def _run():
        orch.results["accuracy"] = SimpleNamespace(
            overall_accuracy_score=100.0, tests_passed=9, tests_failed=0
        )
        return True

    return _run


def test_r2_16_performance_is_unavailable_not_scored(runner, tmp_path):
    orch = runner.ValidationOrchestrator(output_dir=str(tmp_path / "o"))
    orch.run_performance_benchmarks()
    orch.results["accuracy"] = SimpleNamespace(overall_accuracy_score=100.0)
    orch.results["integration"] = {"success": True, "stdout": ""}
    orch.results["stability"] = SimpleNamespace(overall_stability_score=100.0)

    overall, scores = orch.calculate_overall_scores()

    assert orch.results["performance"] is None
    assert "performance" not in scores
    assert overall == pytest.approx(100.0)  # weights renormalised without it


def test_r2_33_failing_required_subprocess_fails_phase_and_exit(
    runner, tmp_path, monkeypatch
):
    proc = SimpleNamespace(returncode=1, stdout="1 FAILED", stderr="")
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: proc)
    orch = runner.ValidationOrchestrator(output_dir=str(tmp_path / "o"))

    assert orch.run_integration_tests() is False
    assert orch.run_stability_tests() is False

    monkeypatch.setattr(runner, "ValidationOrchestrator", lambda **kw: orch)
    monkeypatch.setattr(orch, "run_accuracy_validation", _fixed_accuracy(orch))
    monkeypatch.setattr("sys.argv", ["runner"])
    assert runner.main() is False
    assert orch.failed_steps == ["Integration Tests", "Stability Tests"]


def test_r2_33_passing_subprocesses_still_pass(runner, tmp_path, monkeypatch):
    proc = SimpleNamespace(returncode=0, stdout="ok", stderr="")
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: proc)
    orch = runner.ValidationOrchestrator(output_dir=str(tmp_path / "o"))
    monkeypatch.setattr(runner, "ValidationOrchestrator", lambda **kw: orch)
    monkeypatch.setattr(orch, "run_accuracy_validation", _fixed_accuracy(orch))
    monkeypatch.setattr("sys.argv", ["runner"])

    assert runner.main() is True
    assert orch.failed_steps == []


def _fake_validator(report=None, exc=None):
    class _V:
        def __init__(self, *a, **k):
            pass

        def run_all_validations(self):
            if exc:
                raise exc
            return report

    return _V


def _report(score, passed, failed):
    return SimpleNamespace(
        overall_accuracy_score=score,
        tests_passed=passed,
        tests_failed=failed,
        recommendations=[],
    )


def test_r2_17_main_returns_nonzero_on_critical_failure(monkeypatch):
    sa = _load_script("scientific_accuracy_validator")
    fake = _fake_validator(_report(40.0, 6, 3))
    monkeypatch.setattr(sa, "ScientificAccuracyValidationFramework", fake)
    assert sa.main() == 1


def test_r2_17_main_returns_nonzero_on_exception(monkeypatch):
    sa = _load_script("scientific_accuracy_validator")
    fake = _fake_validator(exc=RuntimeError("boom"))
    monkeypatch.setattr(sa, "ScientificAccuracyValidationFramework", fake)
    assert sa.main() == 1


def test_r2_17_main_returns_zero_on_clean_pass(monkeypatch):
    sa = _load_script("scientific_accuracy_validator")
    fake = _fake_validator(_report(100.0, 9, 0))
    monkeypatch.setattr(sa, "ScientificAccuracyValidationFramework", fake)
    assert sa.main() == 0


def test_r2_17_entry_point_passes_status_to_sys_exit():
    src = (REPO / "scripts/validation/scientific_accuracy_validator.py").read_text()
    tree = ast.parse(src)
    guard = next(
        n for n in tree.body if isinstance(n, ast.If) and "__main__" in ast.dump(n.test)
    )
    assert "sys.exit(main())" in ast.unparse(guard)
