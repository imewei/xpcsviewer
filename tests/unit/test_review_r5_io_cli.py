"""Regression tests for review round 5: R5-01 (CLI), R5-03, R5-15.

R5-01  the two-time CLI writes every image through the run's ``ExportGuard``
       (inputs = every input HDF file of the run); a source scan proves every
       file-writing call in ``xpcsviewer/cli`` is guarded or listed with a reason.
R5-03  the shared file-revision identity includes ctime, so a timestamp- and
       size-preserving in-place rewrite is seen by every cache; a source scan
       proves nothing else decides "same revision" from its own stat call.
R5-15  BFMI is extracted from the ArviZ DataTree (minimum over chains), and a
       BFMI that cannot be computed is not evidence of convergence.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from tests.unit.test_review_r3_maps_roi import K, _write_nexus
from tests.unit.test_review_r4_cli_average import _args, _write_twotime
from xpcsviewer.cli import twotime_batch as tb
from xpcsviewer.file_locator import FileLocator
from xpcsviewer.fileIO.hdf_reader import _file_identity, clear_connection_pool

pytestmark = pytest.mark.unit

PKG = Path(__file__).resolve().parents[2] / "xpcsviewer"


@pytest.fixture(autouse=True)
def _fresh_pool():
    clear_connection_pool()
    yield
    clear_connection_pool()


def _symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available")


# ------------------------------------------------------------------- R5-01
def test_r5_01_output_symlink_to_the_input_is_not_written_through(tmp_path):
    path = tmp_path / "scan.h5"
    out = tmp_path / "out"
    out.mkdir()
    _write_twotime(path, [0.01])
    before = path.read_bytes()
    name = tb.generate_output_filename(str(path), 0.01, None, qbin_index=0)
    _symlink(out / name, path)

    code = tb.run_twotime_batch(_args(path, out, q=0.01))
    clear_connection_pool()

    assert code == 1
    assert path.read_bytes() == before
    assert h5py.is_hdf5(path)


def test_r5_01_directory_run_guards_every_input_not_only_the_current_one(tmp_path):
    root = tmp_path / "in"
    root.mkdir()
    victim = root / "b.h5"
    _write_twotime(root / "a.h5", [0.01])
    _write_twotime(victim, [0.01])
    before = victim.read_bytes()
    out = tmp_path / "out"
    out.mkdir()
    # a.h5's own image name is a stable link to the OTHER input b.h5
    name = tb.generate_output_filename(
        str(root / "a.h5"), 0.01, None, qbin_index=0, root=str(root)
    )
    _symlink(out / name, victim)

    code = tb.run_twotime_batch(_args(root, out, q=0.01))
    clear_connection_pool()

    assert code == 1
    assert victim.read_bytes() == before
    assert h5py.is_hdf5(victim)
    # b.h5 itself was still exported to its own, unrelated destination
    assert any(p.name.startswith("b.h5") for p in out.iterdir() if not p.is_symlink())


def test_r5_01_refused_input_collision_is_logged_not_counted(tmp_path, monkeypatch):
    path = tmp_path / "scan.h5"
    out = tmp_path / "out"
    out.mkdir()
    _write_twotime(path, [0.01])
    name = tb.generate_output_filename(str(path), 0.01, None, qbin_index=0)
    _symlink(out / name, path)
    errors: list[str] = []
    monkeypatch.setattr(tb.logger, "error", lambda m, *a, **k: errors.append(str(m)))
    refused: list[str] = []

    n = tb.process_single_file(str(path), _args(path, out, q=0.01), refused=refused)

    assert n == 0
    assert len(refused) == 1
    assert str(path) in refused[0] and "input file" in refused[0]
    assert any("input file" in m for m in errors)


def test_r5_01_separate_runs_into_one_folder_still_overwrite(tmp_path):
    path = tmp_path / "scan.h5"
    out = tmp_path / "out"
    _write_twotime(path, [0.01])
    assert tb.run_twotime_batch(_args(path, out, q=0.01)) == 0
    assert tb.run_twotime_batch(_args(path, out, q=0.01)) == 0
    assert len(list(out.glob("*.png"))) == 1


# call names that publish bytes to a path
_WRITE_ATTRS = {
    "savefig",
    "savetxt",
    "to_csv",
    "to_netcdf",
    "write_text",
    "write_bytes",
    "savez",
    "savez_compressed",
}
_WRITE_BARE = {"copyfile", "copy2", "copyfileobj", "put"}
_WRITE_MODES = "wax+"

# (file relative to xpcsviewer/, function, call) -> why no ExportGuard.claim
_UNGUARDED = {
    (
        "module/average_toolbox.py",
        "_fresh_file_mode",
        "os.open",
    ): "temporary mode probe in the output directory, removed immediately",
    (
        "module/average_toolbox.py",
        "_publish_average",
        "os.open",
    ): "private staging file with a unique random name beside the output",
    (
        "module/average_toolbox.py",
        "_publish_average",
        "copyfile",
    ): "copies the template into the staging file only",
    (
        "module/average_toolbox.py",
        "_publish_average",
        "put",
    ): "writes the staging file only; the output is published by os.replace",
    (
        "module/average_toolbox.py",
        "_publish_average",
        "os.replace",
    ): "single destination; _check_output_not_input runs right before it",
}


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        base = func.value.id if isinstance(func.value, ast.Name) else "?"
        return f"{base}.{func.attr}"
    return func.id if isinstance(func, ast.Name) else "?"


def _mode_of(node: ast.Call, position: int) -> ast.expr | None:
    for kw in node.keywords:
        if kw.arg == "mode":
            return kw.value
    return node.args[position] if len(node.args) > position else None


def _is_write_call(node: ast.Call) -> bool:
    name = _call_name(node)
    func = node.func
    attr = func.attr if isinstance(func, ast.Attribute) else None
    if attr in _WRITE_ATTRS or name in _WRITE_BARE:
        return True
    if name in {"np.save", "numpy.save", "os.replace", "os.rename", "os.open"}:
        return True
    if name.startswith("shutil."):
        return attr in {"copy", "copy2", "copyfile", "copyfileobj", "move"}
    if name in {"open", "h5py.File", "io.open"} or attr == "open":
        mode = _mode_of(node, 0 if attr == "open" else 1)
        if mode is None:
            return False  # default read mode
        if not (isinstance(mode, ast.Constant) and isinstance(mode.value, str)):
            return True  # a mode we cannot read is treated as a write
        return any(c in mode.value for c in _WRITE_MODES)
    return False


def _write_calls(relpath: str):
    """Yield (function, call name, line, claimed_before) for each write call.

    Calls outside any function belong to ``<module>``; a nested function's calls
    are attributed to the innermost enclosing function.
    """
    tree = ast.parse((PKG / relpath).read_text())
    functions = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
    ]

    def scope(line: int) -> str:
        inner = [f for f in functions if f.lineno <= line <= (f.end_lineno or f.lineno)]
        return max(inner, key=lambda f: f.lineno).name if inner else "<module>"

    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    claims: dict[str, list[int]] = {}
    for c in calls:
        if isinstance(c.func, ast.Attribute) and c.func.attr == "claim":
            claims.setdefault(scope(c.lineno), []).append(c.lineno)
    for node in calls:
        if _is_write_call(node):
            name = scope(node.lineno)
            yield (
                name,
                _call_name(node),
                node.lineno,
                any(c < node.lineno for c in claims.get(name, [])),
            )


def test_r5_01_every_file_write_in_cli_is_guarded_or_listed():
    files = sorted(str(p.relative_to(PKG)) for p in (PKG / "cli").glob("*.py")) + [
        "module/average_toolbox.py"
    ]
    seen: set[tuple[str, str, str]] = set()
    unguarded = []
    for rel in files:
        for func, call, line, claimed in _write_calls(rel):
            key = (rel, func, call)
            if key in _UNGUARDED:
                seen.add(key)
            elif not claimed:
                unguarded.append(f"{rel}:{line} {func} calls {call}")
    assert not unguarded, "write without a preceding guard.claim: " + "; ".join(
        unguarded
    )
    assert seen == set(_UNGUARDED), f"stale allow-list: {set(_UNGUARDED) - seen}"
    # the scan itself must still find the CLI's real write
    assert any(
        call.endswith("savefig")
        for func, call, _, _ in _write_calls("cli/twotime_batch.py")
    )


def test_r5_01_the_scan_flags_an_unguarded_write(tmp_path, monkeypatch):
    module = tmp_path / "bad.py"
    module.write_text(
        "def f(fig, p):\n    fig.savefig(p)\n\n"
        "def g(guard, fig, p):\n    guard.claim(p)\n    fig.savefig(p)\n\n"
        "def h(p):\n    open(p, 'w')\n    open(p)\n\nopen('x', 'a')\n"
    )
    monkeypatch.setattr(sys.modules[__name__], "PKG", tmp_path)
    got = {(f, c, ok) for f, c, _, ok in _write_calls("bad.py")}
    assert got == {
        ("f", "fig.savefig", False),
        ("g", "fig.savefig", True),
        ("h", "open", False),
        ("<module>", "open", False),
    }


# ------------------------------------------------------------------- R5-03
_NO_CTIME = pytest.mark.skipif(
    os.name == "nt", reason="st_ctime is the creation time on Windows"
)


@_NO_CTIME
def test_r5_03_identity_changes_when_content_is_rewritten_with_same_mtime_and_size(
    tmp_path,
):
    path = tmp_path / "scan.h5"
    mirror = tmp_path / "corrected.h5"
    _write_nexus(path)
    st = path.stat()
    shutil.copyfile(path, mirror)
    with h5py.File(mirror, "a") as f:
        f[K["g2"]][:] = 2.5
    os.utime(mirror, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert mirror.stat().st_size == st.st_size
    before = _file_identity(str(path))

    shutil.copy2(mirror, path)  # same inode, mtime and size; ctime moves

    now = path.stat()
    assert (now.st_ino, now.st_mtime_ns, now.st_size) == (
        st.st_ino,
        st.st_mtime_ns,
        st.st_size,
    )
    assert _file_identity(str(path)) != before


@_NO_CTIME
def test_r5_03_reload_after_a_timestamp_preserving_rewrite_serves_new_data(tmp_path):
    """The reviewer's probe: FileLocator must drop the old entry and re-read."""
    path = tmp_path / "scan.h5"
    mirror = tmp_path / "corrected.h5"
    _write_nexus(path)
    st = path.stat()
    shutil.copyfile(path, mirror)
    with h5py.File(mirror, "a") as f:
        f[K["g2"]][:] = 2.5
    os.utime(mirror, ns=(st.st_atime_ns, st.st_mtime_ns))
    locator = FileLocator(str(tmp_path))
    locator.build()
    locator.add_target([path.name])
    [before] = locator.get_xf_list()
    assert before.g2[0, 0] != 2.5

    shutil.copy2(mirror, path)  # no pool clearing: the identity must do it
    locator.build()
    [after] = locator.get_xf_list()

    assert str(path) in {os.fspath(p) for p in locator.replaced_paths}
    assert after is not before
    assert after.g2[0, 0] == 2.5


def test_r5_03_docstring_states_the_ctime_consequence():
    from xpcsviewer.fileIO import hdf_reader

    doc = hdf_reader._file_identity.__doc__ or ""
    assert "ctime" in doc and "chmod" in doc and "rename" in doc


def test_r5_03_no_other_code_stats_a_file_to_decide_the_revision():
    """Revision = ``_file_identity`` only. Any other stat-like call is listed."""
    allowed = {
        ("fileIO/hdf_reader.py", "stat"): "inside _file_identity itself",
        ("fileIO/hdf_reader.py", "st_ino"): "inside _file_identity itself",
        ("fileIO/hdf_reader.py", "st_dev"): "inside _file_identity itself",
        ("fileIO/hdf_reader.py", "st_mtime_ns"): "inside _file_identity itself",
        ("fileIO/hdf_reader.py", "st_ctime_ns"): "inside _file_identity itself",
        ("fileIO/hdf_reader.py", "st_size"): "inside _file_identity itself",
        ("fileIO/hdf_reader.py", "getsize"): "file_size_mb in a metadata report",
        ("file_locator.py", "getsize"): "size shown in the file list",
        ("file_locator.py", "getmtime"): "sort key of the 'Time' file ordering",
        ("simplemask/reader/formats/base.py", "getsize"): "size shown only",
    }
    pattern = re.compile(
        r"\b(stat)\(|\b(getmtime|getctime|getsize|st_ino|st_dev|st_size|"
        r"st_[mca]time\w*)\b"
    )
    found = set()
    for path in PKG.rglob("*.py"):
        rel = str(path.relative_to(PKG))
        if rel.endswith("_ui.py"):
            continue
        for line in path.read_text().splitlines():
            for m in pattern.finditer(line.split("#", 1)[0]):
                found.add((rel, m.group(1) or m.group(2)))
    assert not found - set(allowed), (
        f"revision decided outside _file_identity: {found - set(allowed)}"
    )


# ------------------------------------------------------------------- R5-15
@pytest.fixture(scope="module")
def numpyro_runs():
    """Real two-chain NumPyro runs with an explicit seed, with/without energy."""
    jax = pytest.importorskip("jax")
    numpyro = pytest.importorskip("numpyro")
    az = pytest.importorskip("arviz")
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    def model(y=None):
        mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
        numpyro.sample("y", dist.Normal(mu, sigma), obs=y)

    y = np.random.default_rng(0).normal(0.3, 0.5, 30)
    runs = {}
    for label, extra in (("with", ("num_steps", "energy")), ("without", ())):
        mcmc = MCMC(
            NUTS(model),
            num_warmup=150,
            num_samples=150,
            num_chains=2,
            progress_bar=False,
        )
        mcmc.run(jax.random.PRNGKey(0), y=y, extra_fields=extra)
        runs[label] = az.from_numpyro(mcmc)
        runs[label + "_mcmc"] = mcmc
    return az, runs


def test_r5_15_bfmi_is_extracted_from_the_datatree_as_min_over_chains(numpyro_runs):
    from xpcsviewer.fitting.sampler import compute_bfmi

    az, runs = numpyro_runs
    per_chain = np.asarray(az.bfmi(runs["with"]).to_dataset()["energy"].values)
    assert per_chain.shape == (2,)

    got = compute_bfmi(runs["with"])

    assert isinstance(got, float)
    assert got == pytest.approx(float(per_chain.min()))
    assert got < float(per_chain.mean())  # the worst chain, not the mean


def test_r5_15_low_bfmi_still_warns(numpyro_runs, monkeypatch):
    from xpcsviewer.fitting import sampler

    az, runs = numpyro_runs
    messages: list[str] = []
    monkeypatch.setattr(
        sampler.logger, "warning", lambda m, *a, **k: messages.append(str(m))
    )
    monkeypatch.setattr(sampler.az, "bfmi", lambda d: np.array([0.9, 0.1]))

    assert sampler.compute_bfmi(runs["with"]) == pytest.approx(0.1)
    assert any("Low BFMI" in m for m in messages)


def test_r5_15_unavailable_bfmi_is_none_and_logged_at_warning(
    numpyro_runs, monkeypatch
):
    from xpcsviewer.fitting import sampler

    _, runs = numpyro_runs
    messages: list[str] = []
    monkeypatch.setattr(
        sampler.logger, "warning", lambda m, *a, **k: messages.append(str(m))
    )

    assert sampler.compute_bfmi(runs["without"]) is None  # no energy recorded
    assert any("BFMI" in m for m in messages)


def test_r5_15_bfmi_is_computed_in_exactly_one_place():
    hits = []
    for path in PKG.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "bfmi"
            ):
                hits.append(f"{path.relative_to(PKG)}:{node.lineno}")
    assert len(hits) == 1 and hits[0].startswith("fitting/sampler.py"), hits


def _good_diagnostics(**kw):
    from xpcsviewer.fitting.results import FitDiagnostics

    base = {
        "r_hat": {"a": 1.0},
        "ess_bulk": {"a": 1000},
        "ess_tail": {"a": 1000},
        "divergences": 0,
        "bfmi": 0.9,
    }
    base.update(kw)
    return FitDiagnostics(**base)


def test_r5_15_missing_bfmi_is_not_convergence():
    assert _good_diagnostics().converged
    assert not _good_diagnostics(bfmi=None).converged
    assert not _good_diagnostics(bfmi=float("nan")).converged
    assert not _good_diagnostics(bfmi=0.19).converged


def test_r5_15_fit_result_carries_numeric_bfmi_and_missing_bfmi_fails_it(
    numpyro_runs,
):
    """Through the sampler's own ``_build_fit_result`` on a real NumPyro run."""
    from xpcsviewer.fitting.sampler import SamplerConfig, _build_fit_result

    _, runs = numpyro_runs
    config = SamplerConfig(num_warmup=150, num_samples=150, num_chains=2)

    def build(label):
        mcmc = runs[label + "_mcmc"]
        return _build_fit_result(
            mcmc,
            mcmc.get_samples(),
            nlsq_init={"mu": 0.3, "sigma": 0.5},
            param_names=["mu", "sigma"],
            config=config,
        )

    with_energy = build("with").diagnostics
    assert with_energy.bfmi is not None and np.isfinite(with_energy.bfmi)
    assert build("without").diagnostics.bfmi is None
    assert not build("without").diagnostics.converged
