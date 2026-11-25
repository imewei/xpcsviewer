"""Regression tests for review round 1: R1-01, R1-02, R1-08, R1-19."""

import multiprocessing
import os
from types import SimpleNamespace

import numpy as np
import pytest

from xpcsviewer.module import average_toolbox as at
from xpcsviewer.module.average_toolbox import AverageToolbox

TARGET = "xpcsviewer.module.average_toolbox"


# ---------------------------------------------------------------- R1-01 / R1-02
def _fake_xf_factory(coords_by_index):
    """XF stub: file ``f<i>_*.h5`` has g2 ~ 1 and the given (t_el, dqlist)."""

    def _fake_xf(path, fields=None):
        i = int(os.path.basename(path).split("_")[0][1:])
        t_el, dq = coords_by_index[i]
        return SimpleNamespace(
            g2=np.full((3, 1), 1.0 + 0.005 * i),
            g2_err=np.full((3, 1), 0.01),
            t_el=np.asarray(t_el, dtype=float),
            dqlist=np.asarray(dq, dtype=float),
            clear_cache=lambda: None,
        )

    return _fake_xf


@pytest.fixture
def patched(monkeypatch):
    written = {}
    monkeypatch.setattr(f"{TARGET}.put", lambda p, result, **k: written.update(result))
    monkeypatch.setattr(
        f"{TARGET}.MemoryMonitor.is_memory_pressure_high",
        staticmethod(lambda *a, **k: False),
    )
    return written


def _toolbox(tmp_path, n):
    return AverageToolbox(
        work_dir=str(tmp_path), flist=[f"f{i}_0001.h5" for i in range(n)], jid=1
    )


@pytest.mark.unit
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize(
    "other",
    [
        ([1.0, 10.0], [0.1]),  # different delay axis
        ([0.01, 0.1], [0.2]),  # different Q grid
    ],
)
def test_r1_02_class_skips_files_with_different_coordinates(
    tmp_path, monkeypatch, patched, parallel, other
):
    n = 8 if parallel else 2
    ref = ([0.01, 0.1], [0.1])
    coords = {i: ref for i in range(n)}
    coords[1] = other
    monkeypatch.setattr(f"{TARGET}.XF", _fake_xf_factory(coords))
    monkeypatch.setattr(f"{TARGET}.copyfile", lambda *a: None)
    monkeypatch.setattr(multiprocessing, "cpu_count", lambda: 4 if parallel else 1)
    tb = _toolbox(tmp_path, n)
    tb.do_average(save_path=str(tmp_path / "out.h5"), fields=["g2", "g2_err"])

    assert tb.n_accepted == n - 1
    expected = (sum(1.0 + 0.005 * i for i in range(n) if i != 1)) / (n - 1)
    np.testing.assert_allclose(patched["g2"], expected)


@pytest.mark.unit
def test_r1_02_class_accepts_matching_coordinates(tmp_path, monkeypatch, patched):
    coords = {i: ([0.01, 0.1], [0.1]) for i in range(3)}
    monkeypatch.setattr(f"{TARGET}.XF", _fake_xf_factory(coords))
    monkeypatch.setattr(f"{TARGET}.copyfile", lambda *a: None)
    monkeypatch.setattr(multiprocessing, "cpu_count", lambda: 1)
    tb = _toolbox(tmp_path, 3)
    tb.do_average(save_path=str(tmp_path / "out.h5"), fields=["g2"])
    assert tb.n_accepted == 3


@pytest.mark.unit
def test_r1_02_standalone_skips_files_with_different_coordinates(
    tmp_path, monkeypatch, patched
):
    coords = {0: ([0.01, 0.1], [0.1]), 1: ([1.0, 10.0], [0.1])}
    monkeypatch.setattr(f"{TARGET}.XF", _fake_xf_factory(coords))
    monkeypatch.setattr(f"{TARGET}.copyfile", lambda *a: None)
    at.do_average(
        ["f0_0001.h5", "f1_0001.h5"],
        work_dir=str(tmp_path),
        save_path=str(tmp_path / "out.h5"),
        fields=["g2"],
        n_jobs=1,
    )
    np.testing.assert_allclose(patched["g2"], 1.0)  # file 1 (1.005) excluded


def _two_files(tmp_path):
    a, b = tmp_path / "f0_0001.h5", tmp_path / "f1_0001.h5"
    a.write_bytes(b"AAAA")
    b.write_bytes(b"BBBB")
    return a, b


@pytest.mark.unit
@pytest.mark.parametrize("link", ["same", "symlink", "hardlink"])
def test_r1_01_class_refuses_output_that_is_an_input(
    tmp_path, monkeypatch, patched, link
):
    coords = {i: ([0.01, 0.1], [0.1]) for i in range(2)}
    monkeypatch.setattr(f"{TARGET}.XF", _fake_xf_factory(coords))
    monkeypatch.setattr(multiprocessing, "cpu_count", lambda: 1)
    a, b = _two_files(tmp_path)
    out = b
    if link != "same":
        out = tmp_path / "alias.h5"
        (out.symlink_to if link == "symlink" else lambda p: os.link(b, out))(b)
    tb = _toolbox(tmp_path, 2)
    tb.do_average(save_path=str(out), fields=["g2"])

    assert b.read_bytes() == b"BBBB"
    assert tb.n_accepted == 0
    assert patched == {}


@pytest.mark.unit
def test_r1_01_standalone_refuses_output_that_is_an_input(
    tmp_path, monkeypatch, patched
):
    coords = {i: ([0.01, 0.1], [0.1]) for i in range(2)}
    monkeypatch.setattr(f"{TARGET}.XF", _fake_xf_factory(coords))
    a, b = _two_files(tmp_path)
    with pytest.raises(ValueError, match="input"):
        at.do_average(
            [a.name, b.name],
            work_dir=str(tmp_path),
            save_path=str(b),
            fields=["g2"],
            n_jobs=1,
        )
    assert b.read_bytes() == b"BBBB"


# ---------------------------------------------------------------------- R1-19
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


def _batch_args(root, out):
    return SimpleNamespace(
        input=str(root),
        output=str(out),
        q=0.05,
        phi=None,
        q_phi=None,
        format="png",
        dpi=40,
    )


@pytest.mark.unit
def test_r1_19_recursive_batch_does_not_overwrite_same_named_files(
    tmp_path, monkeypatch
):
    from xpcsviewer.cli import twotime_batch as tb

    root, out = tmp_path / "in", tmp_path / "out"
    for sub in ("a", "b"):
        (root / sub).mkdir(parents=True)
        (root / sub / "run.h5").write_bytes(b"")
    _StubTwotimeFile.labels = ["qbin=1, q=0.0500, phi=10.0"]
    monkeypatch.setattr(tb, "XpcsFile", _StubTwotimeFile)

    n = tb.process_directory(str(root), _batch_args(root, out))

    assert n == 2
    assert len(list(out.glob("*.png"))) == 2


@pytest.mark.unit
def test_r1_19_rounded_q_phi_of_distinct_bins_do_not_collide(tmp_path, monkeypatch):
    from xpcsviewer.cli import twotime_batch as tb

    root, out = tmp_path / "in", tmp_path / "out"
    root.mkdir()
    (root / "run.h5").write_bytes(b"")
    _StubTwotimeFile.labels = [
        "qbin=1, q=0.0500, phi=10.01",
        "qbin=2, q=0.0500, phi=10.04",
    ]
    monkeypatch.setattr(tb, "XpcsFile", _StubTwotimeFile)

    n = tb.process_directory(str(root), _batch_args(root, out))

    assert n == 2
    assert len(list(out.glob("*.png"))) == 2


# ---------------------------------------------------------------------- R1-08
@pytest.mark.unit
def test_r1_08_export_g2_labels_cropped_fit_with_its_own_q(tmp_path, monkeypatch):
    from xpcsviewer.viewer_kernel import ViewerKernel

    q_src = np.array([0.01, 0.02, 0.03])
    t_el = np.logspace(-2, 0, 5)
    fit_x = np.logspace(-2, 0, 4)
    fit_val = np.zeros((2, 2, 4))
    fit_val[0, 0, 0], fit_val[1, 0, 0] = 200.0, 300.0
    fit_line = np.stack([np.full(4, 2.0), np.full(4, 3.0)])
    xf = SimpleNamespace(
        label="runA",
        atype="Multitau",
        get_g2_data=lambda: (
            q_src,
            t_el,
            np.ones((5, 3)),
            np.full((5, 3), 0.01),
            ["a", "b", "c"],
        ),
        fit_summary={
            "fit_func": "single",
            "q_range": "(0.02, 0.03)",
            "t_range": "None",
            "bounds": None,
            "fit_flag": "[True]",
            "q_val": np.array([0.02, 0.03]),
            "fit_val": fit_val,
            "fit_line": fit_line,
            "fit_x": fit_x,
        },
    )
    vk = ViewerKernel.__new__(ViewerKernel)
    monkeypatch.setattr(vk, "get_xf_list", lambda **k: [xf], raising=False)
    vk.export_g2(str(tmp_path))

    rows = [
        ln.split("\t")
        for ln in (tmp_path / "runA_g2_fitting.txt").read_text().splitlines()
        if ln and not ln.startswith("#") and not ln.startswith("q_value")
    ]
    got = {float(r[0]): float(r[1]) for r in rows}
    assert got == {0.02: 200.0, 0.03: 300.0}

    header = (tmp_path / "runA_g2_fitted_curves.txt").read_text().splitlines()[0]
    assert "q=0.020000" in header
    assert "q=0.030000" in header
    assert "q=0.010000" not in header
    curves = np.loadtxt(tmp_path / "runA_g2_fitted_curves.txt")
    assert curves.shape == (4, 3)
    np.testing.assert_allclose(curves[:, 1], 2.0)
    np.testing.assert_allclose(curves[:, 2], 3.0)
