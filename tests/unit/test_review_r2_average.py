"""Regression tests for review round 2: R2-09, R2-10, R2-18, R2-19, R2-29."""

import multiprocessing
import os
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from xpcsviewer.fileIO.aps_8idi import key as hdf_key
from xpcsviewer.fileIO.hdf_reader import clear_connection_pool, get
from xpcsviewer.module import average_toolbox as at
from xpcsviewer.module.average_toolbox import AverageToolbox

pytestmark = pytest.mark.unit

TARGET = "xpcsviewer.module.average_toolbox"
GRID = {"t_el": [0.01, 0.1], "dqlist": [0.1, 0.2]}


def _fake_xf_factory(per_file=None, g2_err=None, on_open=None):
    """XF stub for files ``f<i>_*.h5``; ``per_file[i]`` overrides attributes."""

    def _fake_xf(path, fields=None):
        i = int(os.path.basename(path).split("_")[0][1:])
        # the real loader resolves every requested field through the alias map
        for f in fields or []:
            hdf_key["nexus"][f]
        if on_open is not None:
            on_open(i)
        attrs = {**GRID, **(per_file or {}).get(i, {})}
        err = 0.01 if g2_err is None else g2_err(i)
        return SimpleNamespace(
            g2=np.full((3, 1), 1.0 + 0.005 * i),
            g2_err=np.full((3, 1), err),
            G2=np.full((3, 3, 2), float(i)),
            saxs_1d={"data_raw": np.full(2, 1.0 + i)},
            abs_cross_section_scale=1.0,
            clear_cache=lambda: None,
            **{k: np.asarray(v, dtype=float) for k, v in attrs.items()},
        )

    return _fake_xf


@pytest.fixture
def written(monkeypatch):
    out = {}

    def _put(p, result, ftype="nexus", mode="raw"):
        for k in result:  # real put resolves aliases the same way
            hdf_key[ftype][k]
        out.update(result)

    monkeypatch.setattr(f"{TARGET}.put", _put)
    monkeypatch.setattr(
        f"{TARGET}.MemoryMonitor.is_memory_pressure_high",
        staticmethod(lambda *a, **k: False),
    )
    monkeypatch.setattr(multiprocessing, "cpu_count", lambda: 1)
    return out


@pytest.fixture(autouse=True)
def _fresh_pool():
    clear_connection_pool()
    yield
    clear_connection_pool()


def _names(n):
    return [f"f{i}_0001.h5" for i in range(n)]


def _run(kind, tmp_path, n, fields, save):
    """Run the class (``kind='class'``) or standalone averaging."""
    if kind == "class":
        tb = AverageToolbox(work_dir=str(tmp_path), flist=_names(n), jid=1)
        tb.do_average(save_path=str(save), fields=fields)
        return tb.n_accepted
    at.do_average(
        _names(n), work_dir=str(tmp_path), save_path=str(save), fields=fields, n_jobs=1
    )
    return None


# ------------------------------------------------------------------- R2-09
@pytest.mark.parametrize("kind", ["class", "standalone"])
def test_r2_09_g2_ip_if_request_is_averaged_through_g2(
    tmp_path, monkeypatch, written, kind
):
    monkeypatch.setattr(f"{TARGET}.XF", _fake_xf_factory())
    monkeypatch.setattr(f"{TARGET}.copyfile", lambda *a: None)
    n_acc = _run(kind, tmp_path, 2, ["G2", "IP", "IF"], tmp_path / "out.h5")

    if kind == "class":
        assert n_acc == 2
    assert set(written) == {"G2"}
    np.testing.assert_allclose(written["G2"], 0.5)  # mean of 0 and 1


# ------------------------------------------------------------------- R2-10
@pytest.mark.parametrize("kind", ["class", "standalone"])
def test_r2_10_g2_err_of_mean_is_quadrature_over_n(
    tmp_path, monkeypatch, written, kind
):
    errs = {0: 0.01, 1: 0.09}
    monkeypatch.setattr(f"{TARGET}.XF", _fake_xf_factory(g2_err=lambda i: errs[i]))
    monkeypatch.setattr(f"{TARGET}.copyfile", lambda *a: None)
    _run(kind, tmp_path, 2, ["g2", "g2_err"], tmp_path / "out.h5")

    # sigma(mean of N independent values) = sqrt(sum sigma_i^2) / N
    np.testing.assert_allclose(written["g2_err"], np.sqrt(0.01**2 + 0.09**2) / 2)
    np.testing.assert_allclose(written["g2"], 1.0025)  # g2 itself is a plain mean


def test_r2_10_equal_errors_keep_sigma_over_sqrt_n(tmp_path, monkeypatch, written):
    monkeypatch.setattr(f"{TARGET}.XF", _fake_xf_factory())
    monkeypatch.setattr(f"{TARGET}.copyfile", lambda *a: None)
    _run("standalone", tmp_path, 4, ["g2", "g2_err"], tmp_path / "out.h5")
    np.testing.assert_allclose(written["g2_err"], 0.01 / np.sqrt(4))


# ------------------------------------------------------------------- R2-19
@pytest.mark.parametrize("kind", ["class", "standalone"])
@pytest.mark.parametrize(
    ("fields", "attr", "ref", "other"),
    [
        (["g2", "g2_err"], "dynamic_index_mapping", [0], [1]),
        (["g2", "g2_err"], "dplist", [0.0, 90.0], [0.0, 180.0]),
        (["saxs_1d"], "static_index_mapping", [0], [1]),
        (["saxs_1d"], "splist", [0.0, 90.0], [0.0, 180.0]),
    ],
)
def test_r2_19_different_packed_bins_are_not_averaged(
    tmp_path, monkeypatch, written, kind, fields, attr, ref, other
):
    per_file = {0: {attr: ref}, 1: {attr: ref}, 2: {attr: other}}
    monkeypatch.setattr(f"{TARGET}.XF", _fake_xf_factory(per_file))
    monkeypatch.setattr(f"{TARGET}.copyfile", lambda *a: None)
    n_acc = _run(kind, tmp_path, 3, fields, tmp_path / "out.h5")

    if kind == "class":
        assert n_acc == 2
    key = fields[0]
    expected = (1.0 + 1.005) / 2 if key == "g2" else (1.0 + 2.0) / 2
    np.testing.assert_allclose(written[key], expected)  # file 2 excluded


def test_r2_19_index_mapping_found_on_qmap_object(tmp_path, monkeypatch, written):
    # XpcsFile exposes the index mappings only through ``xf.qmap``
    base = _fake_xf_factory()

    def _xf(path, fields=None):
        xf = base(path, fields)
        i = int(os.path.basename(path).split("_")[0][1:])
        xf.qmap = SimpleNamespace(dynamic_index_mapping=np.array([i]))
        return xf

    monkeypatch.setattr(f"{TARGET}.XF", _xf)
    monkeypatch.setattr(f"{TARGET}.copyfile", lambda *a: None)
    n_acc = _run("class", tmp_path, 2, ["g2"], tmp_path / "out.h5")
    assert n_acc == 1


def test_r2_19_matching_bins_are_still_averaged(tmp_path, monkeypatch, written):
    same = {i: {"dynamic_index_mapping": [0], "dplist": [0.0, 90.0]} for i in range(3)}
    monkeypatch.setattr(f"{TARGET}.XF", _fake_xf_factory(same))
    monkeypatch.setattr(f"{TARGET}.copyfile", lambda *a: None)
    assert _run("class", tmp_path, 3, ["g2"], tmp_path / "out.h5") == 3


# ------------------------------------------------------------------- R2-18
@pytest.mark.parametrize("kind", ["class", "standalone"])
def test_r2_18_output_aliased_to_input_during_run_does_not_clobber_input(
    tmp_path, monkeypatch, written, kind
):
    a, b, out = tmp_path / "f0_0001.h5", tmp_path / "f1_0001.h5", tmp_path / "out.h5"
    a.write_bytes(b"AAAA")
    b.write_bytes(b"BBBB")

    def alias_after_first_open(i):
        if i == 0 and not out.is_symlink():
            out.symlink_to(b)  # concurrent change after the start-up check

    monkeypatch.setattr(
        f"{TARGET}.XF", _fake_xf_factory(on_open=alias_after_first_open)
    )
    if kind == "standalone":
        with pytest.raises(ValueError, match="input"):
            _run(kind, tmp_path, 2, ["g2"], out)
    else:
        assert _run(kind, tmp_path, 2, ["g2"], out) == 0

    assert b.read_bytes() == b"BBBB"
    assert a.read_bytes() == b"AAAA"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [a.name, b.name, out.name]
    )


# ------------------------------------------------------------------- R2-29
def _h5(path, ident, g2=None):
    with h5py.File(path, "w") as f:
        f["/entry/identity"] = ident
        if g2 is not None:
            f[hdf_key["nexus"]["g2"]] = g2


def _identity(path):
    with h5py.File(path, "r") as f:
        return int(f["/entry/identity"][()]), f[hdf_key["nexus"]["g2"]][()]


@pytest.mark.parametrize("kind", ["class", "standalone"])
def test_r2_29_failed_average_keeps_previous_output(
    tmp_path, monkeypatch, written, kind
):
    _h5(tmp_path / "f0_0001.h5", 42)
    _h5(tmp_path / "f1_0001.h5", 42)
    out = tmp_path / "out.h5"
    _h5(out, 99, g2=np.full((3, 1), 5.0))
    old = out.read_bytes()

    def failing_put(p, result, **kw):
        raise OSError("boom")

    monkeypatch.setattr(f"{TARGET}.XF", _fake_xf_factory())
    monkeypatch.setattr(f"{TARGET}.put", failing_put)
    if kind == "standalone":
        with pytest.raises(OSError, match="boom"):
            _run(kind, tmp_path, 2, ["g2"], out)
    else:
        assert _run(kind, tmp_path, 2, ["g2"], out) == 0

    assert out.read_bytes() == old
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "f0_0001.h5",
        "f1_0001.h5",
        "out.h5",
    ]


@pytest.mark.parametrize("kind", ["class", "standalone"])
def test_r2_29_replaces_output_that_the_viewer_has_open(
    tmp_path, monkeypatch, written, kind
):
    _h5(tmp_path / "f0_0001.h5", 42)
    _h5(tmp_path / "f1_0001.h5", 42)
    out = tmp_path / "out.h5"
    _h5(out, 99, g2=np.full((3, 1), 5.0))
    # pooled read-only handle, as the viewer holds after reading the old output
    np.testing.assert_allclose(get(str(out), ["g2"], "alias")["g2"], 5.0)

    from xpcsviewer.fileIO.hdf_reader import put as real_put

    monkeypatch.setattr(f"{TARGET}.XF", _fake_xf_factory())
    monkeypatch.setattr(f"{TARGET}.put", real_put)
    _run(kind, tmp_path, 2, ["g2"], out)

    ident, g2 = _identity(out)
    assert ident == 42  # template of the new run
    np.testing.assert_allclose(g2, 1.0025)
    # the pooled handle must not keep serving the replaced file
    np.testing.assert_allclose(get(str(out), ["g2"], "alias")["g2"], 1.0025)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "f0_0001.h5",
        "f1_0001.h5",
        "out.h5",
    ]
