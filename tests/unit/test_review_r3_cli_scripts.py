"""Regression tests for review round 3: R3-04, R3-08, R3-09, R3-10, R3-11."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from xpcsviewer.fileIO.hdf_reader import HDF5ConnectionPool
from xpcsviewer.io.hdf5_facade import HDF5Facade, HDF5ValidationError

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]


def _load_script(name):
    path = REPO / "scripts" / "validation" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"r3_{name}", path)
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


@pytest.fixture
def errors(tb, monkeypatch):
    msgs: list[str] = []
    monkeypatch.setattr(tb.logger, "error", lambda m, *a, **k: msgs.append(str(m)))
    return msgs


def _two_sources(tmp_path):
    root = tmp_path / "in"
    for d in ("a", "b"):
        (root / d).mkdir(parents=True)
        (root / d / "sample.h5").write_bytes(b"")
    _StubTwotimeFile.labels = ["qbin=1, q=0.0500, phi=10.0"]
    return root


def test_r3_08_symlinked_output_subdirs_do_not_overwrite(
    tb, tmp_path, errors, monkeypatch
):
    root = _two_sources(tmp_path)
    out = tmp_path / "out"
    (out / "a").mkdir(parents=True)
    try:
        os.symlink(out / "a", out / "b", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available")

    rc = tb.run_twotime_batch(_args(root, out))

    assert rc != 0
    assert len(list(out.rglob("*.png"))) == 1
    refusal = [m for m in errors if "already written" in m]
    assert len(refusal) == 1
    assert "a" in refusal[0] and "b" in refusal[0]
    assert str(root / "a" / "sample.h5") in refusal[0]
    assert str(root / "b" / "sample.h5") in refusal[0]


def test_r3_08_refused_collision_makes_exit_nonzero(tb, tmp_path, monkeypatch):
    root = _two_sources(tmp_path)
    out = tmp_path / "out"
    monkeypatch.setattr(tb, "generate_output_filename", lambda *a, **k: "same.png")

    assert tb.run_twotime_batch(_args(root, out)) != 0
    assert [p.name for p in out.rglob("*.png")] == ["same.png"]


def test_r3_08_distinct_outputs_still_exit_zero(tb, tmp_path):
    root = _two_sources(tmp_path)
    out = tmp_path / "out"

    assert tb.run_twotime_batch(_args(root, out)) == 0
    assert len(list(out.rglob("*.png"))) == 2


def test_r3_08_separate_runs_overwrite_by_design(tb, tmp_path):
    """Scope decision: a later run into the same folder replaces earlier images."""
    root = tmp_path / "in"
    root.mkdir()
    (root / "sample.h5").write_bytes(b"")
    _StubTwotimeFile.labels = ["qbin=1, q=0.0500, phi=10.0"]
    out = tmp_path / "out"

    assert tb.run_twotime_batch(_args(root, out)) == 0
    assert tb.run_twotime_batch(_args(root, out)) == 0
    assert len(list(out.rglob("*.png"))) == 1


# ---------------------------------------------------------------------- R3-11
def _labels(*phis):
    return [f"qbin={i}, q=0.0500 A^-1, φ={p:.1f} degree" for i, p in enumerate(phis)]


def test_r3_11_phi_selection_uses_wrapped_distance(tb, monkeypatch):
    _StubTwotimeFile.labels = _labels(2.0, 340.0)
    msgs: list[str] = []
    monkeypatch.setattr(tb.logger, "info", lambda m, *a, **k: msgs.append(str(m)))

    got = tb.find_qbins_for_phi(_StubTwotimeFile("x"), 359.0)

    assert [g[0] for g in got] == [0]
    assert any("diff=3.00" in m for m in msgs)


def test_r3_11_phi_selection_wraps_in_signed_range(tb):
    _StubTwotimeFile.labels = _labels(-170.0, 10.0)
    got = tb.find_qbins_for_phi(_StubTwotimeFile("x"), 175.0)
    assert [g[0] for g in got] == [0]  # 15 deg away, not 165


def test_r3_11_qphi_selection_uses_wrapped_distance(tb, monkeypatch):
    _StubTwotimeFile.labels = _labels(2.0, 340.0)
    msgs: list[str] = []
    monkeypatch.setattr(tb.logger, "info", lambda m, *a, **k: msgs.append(str(m)))

    got = tb.find_qbin_for_qphi(_StubTwotimeFile("x"), 0.05, 359.0)

    assert got[0] == 0
    assert any("diff=3.00" in m for m in msgs)


def test_r3_11_non_wrapping_selection_unchanged(tb):
    _StubTwotimeFile.labels = _labels(10.0, 100.0, 200.0)
    assert [g[0] for g in tb.find_qbins_for_phi(_StubTwotimeFile("x"), 110.0)] == [1]
    assert tb.find_qbin_for_qphi(_StubTwotimeFile("x"), 0.05, 190.0)[0] == 2


# ---------------------------------------------------------------------- R3-04
def test_r3_04_rank_one_g2_raises_documented_validation_error(tmp_path):
    path = str(tmp_path / "g2.h5")
    with h5py.File(path, "w") as f:
        g = f.create_group("xpcs/g2")
        g["g2"] = np.ones(3)
        g["g2_err"] = np.full(3, 0.01)
        g["delay_times"] = np.array([1.0, 2.0, 3.0])
        g["q_values"] = np.array([0.1, 0.2, 0.3])
    facade = HDF5Facade(pool=HDF5ConnectionPool(max_pool_size=2))

    with pytest.raises(HDF5ValidationError):
        facade.read_g2_data(path)
    with pytest.raises(HDF5ValidationError):
        facade.read_g2_data(path, q_idx=0)


# ------------------------------------------------------------ validation scripts
@pytest.fixture
def runner(tmp_path, monkeypatch):
    mod = _load_script("comprehensive_validation_runner")
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    return mod


def _fake_validator(report):
    class _V:
        def __init__(self, *a, **k):
            pass

        def run_all_validations(self):
            return report

    return _V


def _report(score, passed, failed):
    return SimpleNamespace(
        overall_accuracy_score=score,
        tests_passed=passed,
        tests_failed=failed,
        recommendations=[],
    )


def test_r3_09_failed_scientific_case_fails_phase(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner,
        "ScientificAccuracyValidationFramework",
        _fake_validator(_report(98.0, 49, 1)),
    )
    monkeypatch.setattr(runner, "VALIDATION_COMPONENTS_AVAILABLE", True)
    orch = runner.ValidationOrchestrator(output_dir=str(tmp_path / "o"))

    assert orch.run_accuracy_validation() is False
    assert orch.phase_validation_status["Phase3_Vectorization"] == "failed"


def test_r3_09_clean_report_passes_phase(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner,
        "ScientificAccuracyValidationFramework",
        _fake_validator(_report(100.0, 50, 0)),
    )
    monkeypatch.setattr(runner, "VALIDATION_COMPONENTS_AVAILABLE", True)
    orch = runner.ValidationOrchestrator(output_dir=str(tmp_path / "o"))

    assert orch.run_accuracy_validation() is True
    assert orch.phase_validation_status["Phase3_Vectorization"] == "passed"


def test_r3_09_runner_and_validator_share_the_predicate(runner):
    sa = _load_script("scientific_accuracy_validator")
    for rep in (_report(98.0, 49, 1), _report(79.0, 10, 0), _report(100.0, 5, 0)):
        assert runner.report_passed(rep) == sa.report_passed(rep)
    assert runner.report_passed(_report(100.0, 5, 0)) is True
    assert runner.report_passed(_report(79.9, 5, 0)) is False


def test_r3_09_standalone_main_uses_same_predicate(monkeypatch):
    sa = _load_script("scientific_accuracy_validator")
    monkeypatch.setattr(
        sa,
        "ScientificAccuracyValidationFramework",
        _fake_validator(_report(98.0, 49, 1)),
    )
    assert sa.main() == 1


def _run_all(runner, orch, tmp_path, monkeypatch, stability_rc, integration_rc=0):
    def fake_run(cmd, **k):
        rc = stability_rc if "test_memory_limits" in " ".join(cmd) else integration_rc
        return SimpleNamespace(returncode=rc, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    def acc():
        orch.results["accuracy"] = _report(100.0, 9, 0)
        return True

    monkeypatch.setattr(orch, "run_accuracy_validation", acc)
    return orch.run_comprehensive_validation()


def test_r3_10_stability_score_follows_subprocess_outcome(
    runner, tmp_path, monkeypatch
):
    orch = runner.ValidationOrchestrator(output_dir=str(tmp_path / "o"))
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr=""),
    )
    assert orch.run_stability_tests() is False
    res = orch.results["stability"]
    assert res.overall_stability_score == 0.0
    assert all(t["system_stability_score"] == 0.0 for t in res.stress_test_results)

    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    assert orch.run_stability_tests() is True
    assert orch.results["stability"].overall_stability_score == 100.0


def test_r3_10_failed_stability_never_recommends_deploying(
    runner, tmp_path, monkeypatch
):
    orch = runner.ValidationOrchestrator(output_dir=str(tmp_path / "o"))
    report = _run_all(runner, orch, tmp_path, monkeypatch, stability_rc=1)

    ready = report.deployment_readiness
    assert orch.failed_steps == ["Stability Tests"]
    assert ready["deployment_blockers"]
    assert not (
        ready["ready_for_production"]
        or ready["ready_for_beta"]
        or ready["ready_for_alpha"]
    )
    assert "Deploy to" not in ready["recommended_action"]


def test_r3_10_clean_run_still_recommends_production(runner, tmp_path, monkeypatch):
    orch = runner.ValidationOrchestrator(output_dir=str(tmp_path / "o"))
    report = _run_all(runner, orch, tmp_path, monkeypatch, stability_rc=0)

    assert report.deployment_readiness["ready_for_production"] is True


def test_r3_10_blockers_gate_before_score(runner, tmp_path):
    orch = runner.ValidationOrchestrator(output_dir=str(tmp_path / "o"))
    got = orch.assess_deployment_readiness(99.0, ["something is broken"])
    assert not (
        got["ready_for_production"] or got["ready_for_beta"] or got["ready_for_alpha"]
    )

    orch.failed_steps.append("Integration Tests")
    got = orch.assess_deployment_readiness(99.0, [])
    assert not (
        got["ready_for_production"] or got["ready_for_beta"] or got["ready_for_alpha"]
    )
