"""Regression tests for review round 4: R4-03, R4-04, R4-07.

Invariant: bins and outputs are identified by their real numeric or filesystem
identity, never by rounded display text or a partially resolved path.

R4-03  the two-time CLI selects bins by the numeric (q, phi) of the file's own
       arrays, not by numbers parsed back out of the rounded display label.
R4-04  averaging also requires identical axis names and units: the same bin
       centres in (q, phi) and in (x, y) pixels are different quantities.
R4-07  the same-run output collision key follows the final file name too, so
       two symlinks to one image are one destination.

All tests write real NeXus-layout HDF5 files with h5py and load them through
the real ``XpcsFile`` / ``QMap``.
"""

from __future__ import annotations

import os
import re
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from tests.unit.test_review_r3_average import K, _average, _read
from tests.unit.test_review_r3_average import _write_nexus as _write_avg_nexus
from tests.unit.test_review_r3_maps_roi import _write_nexus as _write_base_nexus
from xpcsviewer.cli import twotime_batch as tb
from xpcsviewer.fileIO.hdf_reader import clear_connection_pool
from xpcsviewer.xpcs_file import XpcsFile

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh_pool(monkeypatch):
    monkeypatch.setattr(
        "xpcsviewer.module.average_toolbox.MemoryMonitor.is_memory_pressure_high",
        staticmethod(lambda *a, **k: False),
    )
    clear_connection_pool()
    yield
    clear_connection_pool()


def fake_qmap_from_labels(labels):
    """Numeric ``(c2_processed_bins, qmap)`` for label-only stub files.

    Lets stubs that only know their display labels expose the numeric axes the
    CLI now reads (the production code never parses the label text).
    """
    pairs = []
    for lab in labels:
        q = re.search(r"q=([0-9.eE+-]+)", lab)
        p = re.search(r"(?:phi|φ)=([0-9.eE+-]+)", lab)
        pairs.append((float(q.group(1)), float(p.group(1)) if p else None))
    if all(p is None for _, p in pairs):
        dq, dp = [q for q, _ in pairs], [0.0]
        mapping, num_pts = list(range(len(pairs))), [len(pairs), 1]
    else:
        dq = sorted({q for q, _ in pairs})
        dp = sorted({p for _, p in pairs})
        if len(dp) == 1:  # a one-point angle axis would be a radial-only map
            dp.append(dp[0] + 180.0)  # unused second azimuth keeps phi
        mapping = [dq.index(q) * len(dp) + dp.index(p) for q, p in pairs]
        num_pts = [len(dq), len(dp)]
    qmap = SimpleNamespace(
        map_names=["q", "phi"],
        dqlist=np.array(dq),
        dplist=np.array(dp),
        dynamic_index_mapping=np.array(mapping),
        dynamic_num_pts=np.array(num_pts),
    )
    return np.arange(1, len(pairs) + 1), qmap


def _write_twotime(path, dqlist, dplist=None, value=1.0):
    """Two-time NeXus file whose processed bins are every dynamic bin."""
    n_q = len(dqlist)
    n_phi = 1 if dplist is None else len(dplist)
    n_bins = n_q * n_phi
    _write_base_nexus(path)
    with h5py.File(path, "a") as f:
        for name in ("dqlist", "dplist", "dynamic_index_mapping", "dynamic_num_pts"):
            del f[K[name]]
        f[K["dqlist"]] = np.asarray(dqlist, dtype=np.float64)
        f[K["dplist"]] = np.array([0.0] if dplist is None else dplist)
        f[K["dynamic_index_mapping"]] = np.arange(n_bins)
        f[K["dynamic_num_pts"]] = np.array([n_q, n_phi])
        f[K["c2_processed_bins"]] = np.arange(1, n_bins + 1)
        f[K["c2_stride_frame"]] = 1
        f[K["c2_avg_frame"]] = 1
        f[K["c2_g2"]] = np.ones((5, n_bins))
        f[K["c2_g2_segments"]] = np.ones((5, n_bins, 2))
        for index in range(1, n_bins + 1):
            f[f"{K['c2_prefix']}/c2_{index:05d}"] = np.triu(
                np.full((4, 4), value * index, dtype=np.float32)
            )


def _args(root, out, **kw):
    base = {
        "input": str(root),
        "output": str(out),
        "q": None,
        "phi": None,
        "q_phi": None,
        "format": "png",
        "dpi": 30,
    }
    base.update(kw)
    return SimpleNamespace(**base)


# ------------------------------------------------------------------- R4-03
def test_r4_03_q_selection_uses_numeric_q_not_the_rounded_label(tmp_path):
    path = tmp_path / "scan.h5"
    _write_twotime(path, [0.010001, 0.010004])
    xf = XpcsFile(str(path))
    # the display labels really are identical, the numbers really are not
    assert len(set(xf.get_twotime_qbin_labels())) == 2  # carries the bin number
    assert all("q=0.01000" in lab for lab in xf.get_twotime_qbin_labels())

    got = tb.find_qbins_for_q(xf, 0.010004)

    assert [row[0] for row in got] == [1]
    assert got[0][2] == 0.010004  # full precision, not the label's 0.01000
    assert got[0][3] is None  # radial-only map: no phi, Q-only still works
    assert [row[0] for row in tb.find_qbins_for_q(xf, 0.010001)] == [0]


def test_r4_03_angle_selection_uses_numeric_phi_not_the_rounded_label(tmp_path):
    path = tmp_path / "scan.h5"
    _write_twotime(path, [0.01, 0.02], dplist=[10.01, 10.04])
    xf = XpcsFile(str(path))
    assert xf.qmap.dynamic_num_pts.tolist() == [2, 2]

    # bins 0..3 = (q0,phi0) (q0,phi1) (q1,phi0) (q1,phi1)
    assert [r[0] for r in tb.find_qbins_for_phi(xf, 10.04)] == [1, 3]
    assert [r[0] for r in tb.find_qbins_for_phi(xf, 10.01)] == [0, 2]
    assert [r[0] for r in tb.find_qbins_for_q(xf, 0.02)] == [2, 3]
    best = tb.find_qbin_for_qphi(xf, 0.02, 10.04)
    assert best[0] == 3
    assert (best[2], best[3]) == (0.02, 10.04)


def test_r4_03_cli_exports_only_the_exactly_requested_bin(tmp_path):
    root, out = tmp_path / "in", tmp_path / "out"
    root.mkdir()
    _write_twotime(root / "scan.h5", [0.010001, 0.010004])

    assert tb.run_twotime_batch(_args(root, out, q=0.010004)) == 0

    assert [p.name for p in out.rglob("*.png")] == ["scan.h5_bin0001_q0.0100.png"]


def test_r4_03_non_finite_selector_is_still_rejected(tmp_path):
    root, out = tmp_path / "in", tmp_path / "out"
    root.mkdir()
    _write_twotime(root / "scan.h5", [0.010001, 0.010004])
    for kw in ({"q": float("nan")}, {"q": None, "q_phi": "0.01,inf"}):
        assert tb.run_twotime_batch(_args(root, out, **kw)) != 0
    assert not out.exists()


# ------------------------------------------------------------------- R4-04
def _axes_pair(tmp_path, *, names=(b"x", b"y"), units=(b"px", b"px")):
    """Two files with identical numeric axes; the second declares other axes."""
    a, b = tmp_path / "f0_0001.h5", tmp_path / "f1_0001.h5"
    _write_avg_nexus(a, g2=1.0, saxs=1.0)
    _write_avg_nexus(b, g2=1.04, saxs=3.0)
    with h5py.File(b, "a") as h:
        h[K["map_names"]][:] = list(names)
        h[K["map_units"]][:] = list(units)
    return [a.name, b.name]


@pytest.mark.parametrize("kind", ["class", "standalone"])
@pytest.mark.parametrize(
    ("field", "names", "units"),
    [
        ("g2", (b"x", b"y"), (b"px", b"px")),
        ("saxs_1d", (b"x", b"y"), (b"px", b"px")),
        ("G2", (b"x", b"y"), (b"px", b"px")),
        ("g2", (b"q", b"phi"), (b"1/nm", b"deg")),  # same names, other unit
    ],
    ids=["g2", "saxs_1d", "G2", "units-only"],
)
def test_r4_04_unlike_axis_meaning_is_not_averaged(tmp_path, kind, field, names, units):
    flist = _axes_pair(tmp_path, names=names, units=units)
    out = tmp_path / "o.h5"
    a, b = (XpcsFile(str(tmp_path / n)).qmap for n in flist)
    assert a.dqlist.tolist() == b.dqlist.tolist()  # same numbers, other meaning

    n_accepted = _average(kind, tmp_path, flist, [field], out)

    if kind == "class":
        assert n_accepted == 1
    stored = _read(out, field)
    # only the first file's values: no mix with the second file's 1.04 / 3.0
    expected = {"g2": 1.0, "G2": 1.0, "saxs_1d": 1.0}[field]
    np.testing.assert_allclose(stored, expected)
    assert [s.decode() for s in _read(out, "map_names")] == ["q", "phi"]


@pytest.mark.parametrize("kind", ["class", "standalone"])
def test_r4_04_same_axes_and_axis_independent_fields_still_average(tmp_path, kind):
    flist = _axes_pair(tmp_path, names=(b"q", b"phi"), units=(b"1/A", b"deg"))
    out = tmp_path / "o.h5"
    _average(kind, tmp_path, flist, ["g2", "saxs_1d"], out)
    np.testing.assert_allclose(_read(out, "g2"), 1.02)
    np.testing.assert_allclose(_read(out, "saxs_1d"), 2.0)

    # the pixel image does not depend on the bin axes' meaning
    flist = _axes_pair(tmp_path, names=(b"x", b"y"), units=(b"px", b"px"))
    _average(kind, tmp_path, flist, ["saxs_2d"], tmp_path / "o2.h5")
    np.testing.assert_allclose(_read(tmp_path / "o2.h5", "saxs_2d"), 2.0)


# ------------------------------------------------------------------- R4-07
def _symlink_or_skip(link, target):
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available")


def _alias_tree(tmp_path):
    """in/{a,b}/scan.h5 whose output PNG names are symlinks to one image."""
    root, out, backing = tmp_path / "in", tmp_path / "out", tmp_path / "shared.png"
    for idx, sub in enumerate(("a", "b"), 1):
        (root / sub).mkdir(parents=True)
        _write_twotime(root / sub / "scan.h5", [0.01], value=float(idx))
        (out / sub).mkdir(parents=True)
        name = tb.generate_output_filename(
            str(root / sub / "scan.h5"), 0.01, None, qbin_index=0, root=str(root)
        )
        _symlink_or_skip(out / name, backing)
    return root, out, backing


def test_r4_07_two_symlinks_to_one_image_are_one_destination(tmp_path, monkeypatch):
    root, out, backing = _alias_tree(tmp_path)
    errors: list[str] = []
    monkeypatch.setattr(tb.logger, "error", lambda m, *a, **k: errors.append(str(m)))
    saved: list[str] = []
    real_savefig = tb.plt.Figure.savefig
    monkeypatch.setattr(
        tb.plt.Figure,
        "savefig",
        lambda self, path, *a, **k: (
            saved.append(str(path)),
            real_savefig(self, path, *a, **k),
        )[1],
    )

    rc = tb.run_twotime_batch(_args(root, out, q=0.01))

    assert rc == 1
    assert len(saved) == 1  # the second source was refused, not written through
    assert backing.stat().st_size > 0
    refusal = [m for m in errors if "already written" in m]
    assert len(refusal) == 1
    assert str(root / "a" / "scan.h5") in refusal[0]
    assert str(root / "b" / "scan.h5") in refusal[0]


def test_r4_07_distinct_final_names_still_both_written(tmp_path):
    root, out = tmp_path / "in", tmp_path / "out"
    for sub in ("a", "b"):
        (root / sub).mkdir(parents=True)
        _write_twotime(root / sub / "scan.h5", [0.01])
    assert tb.run_twotime_batch(_args(root, out, q=0.01)) == 0
    assert len(list(out.rglob("*.png"))) == 2


def test_r4_07_separate_runs_into_one_folder_still_overwrite(tmp_path):
    root, out = tmp_path / "in", tmp_path / "out"
    root.mkdir()
    _write_twotime(root / "scan.h5", [0.01])
    assert tb.run_twotime_batch(_args(root, out, q=0.01)) == 0
    assert tb.run_twotime_batch(_args(root, out, q=0.01)) == 0
    assert len(list(out.rglob("*.png"))) == 1
