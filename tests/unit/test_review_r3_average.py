"""Regression tests for review round 3: R3-01, R3-06, R3-07, R3-32.

Invariant: averaging only combines arrays whose elements mean the same
physical thing, never changes its inputs, and never changes anything about an
existing output except its contents.

The end-to-end tests write real NeXus-layout HDF5 files with h5py and load
them through the real ``XpcsFile`` (no stand-in objects).
"""

import os
import stat

import h5py
import numpy as np
import pytest

from xpcsviewer.fileIO.aps_8idi import key as hdf_key
from xpcsviewer.fileIO.hdf_reader import clear_connection_pool
from xpcsviewer.module import average_toolbox as at
from xpcsviewer.module.average_toolbox import AverageToolbox
from xpcsviewer.xpcs_file import XpcsFile

pytestmark = pytest.mark.unit

K = hdf_key["nexus"]
TARGET = "xpcsviewer.module.average_toolbox"
H = W = 4


def _write_nexus(
    path,
    *,
    g2=1.0,
    saxs=1.0,
    bcx=2.0,
    bcy=2.0,
    det_dist=5.0,
    pixel_size=75e-6,
    energy=8.0,
    mask=None,
    dqmap=None,
    dyn_map=(0, 1),
    t_el_scale=1.0,
):
    """Smallest multitau NeXus file the real ``XpcsFile`` can load."""
    cols = np.tile(np.array([1, 1, 2, 2], dtype=np.int32), (H, 1))
    with h5py.File(path, "w") as f:
        f[K["g2"]] = np.full((3, 2), g2)
        f[K["g2_err"]] = np.full((3, 2), 0.01)
        f[K["G2"]] = np.full((3, 2, 3), g2)
        f[K["tau"]] = np.array([1.0, 2.0, 4.0]) * t_el_scale
        f[K["stride_frame"]] = 1
        f[K["avg_frame"]] = 1
        f[K["t0"]] = 0.1
        f[K["t1"]] = 0.1
        f[K["start_time"]] = "2024-01-01"
        f[K["Int_t"]] = np.ones((2, 5))
        f[K["saxs_1d"]] = np.full((1, 2), saxs)
        f[K["Iqp"]] = np.full((1, 2), saxs)
        f[K["saxs_2d"]] = np.full((1, H, W), saxs)
        f[K["mask"]] = np.ones((H, W), np.int32) if mask is None else mask
        f[K["dqmap"]] = cols if dqmap is None else dqmap
        f[K["sqmap"]] = cols
        f[K["dqlist"]] = np.array([0.01, 0.02])
        f[K["sqlist"]] = np.array([0.01, 0.02])
        f[K["dplist"]] = np.array([0.0])
        f[K["splist"]] = np.array([0.0])
        f[K["bcx"]] = bcx
        f[K["bcy"]] = bcy
        f[K["X_energy"]] = energy
        f[K["pixel_size"]] = pixel_size
        f[K["det_dist"]] = det_dist
        f[K["static_index_mapping"]] = np.array([0, 1])
        f[K["dynamic_index_mapping"]] = np.array(dyn_map)
        f[K["static_num_pts"]] = np.array([2, 1])
        f[K["dynamic_num_pts"]] = np.array([2, 1])
        f[K["map_names"]] = np.array([b"q", b"phi"])
        f[K["map_units"]] = np.array([b"1/A", b"deg"])


@pytest.fixture(autouse=True)
def _fresh_pool(monkeypatch):
    monkeypatch.setattr(
        f"{TARGET}.MemoryMonitor.is_memory_pressure_high",
        staticmethod(lambda *a, **k: False),
    )
    clear_connection_pool()
    yield
    clear_connection_pool()


def _average(kind, tmp_path, names, fields, out):
    """Run the GUI worker (``class``) or the standalone function; return the output."""
    if kind == "class":
        tb = AverageToolbox(work_dir=str(tmp_path), flist=names, jid=1)
        tb.do_average(save_path=str(out), fields=fields)
        n_accepted = tb.n_accepted
    else:
        at.do_average(
            names, work_dir=str(tmp_path), save_path=str(out), fields=fields, n_jobs=1
        )
        n_accepted = None
    return n_accepted


def _read(path, name):
    with h5py.File(path, "r") as f:
        return f[K[name]][()]


ALL_FIELDS = ["saxs_2d", "saxs_1d", "g2", "g2_err", "G2"]


# ------------------------------------------------------------------- R3-32
def test_r3_32_real_files_standalone_average_does_not_mutate_open_input(tmp_path):
    a, b = tmp_path / "f0_0001.h5", tmp_path / "f1_0001.h5"
    _write_nexus(a, g2=1.0, saxs=1.0)
    _write_nexus(b, g2=1.04, saxs=3.0)
    opened = XpcsFile(str(a), fields=["saxs_2d", "G2"])
    before = {k: np.array(getattr(opened, k)) for k in ("g2", "g2_err", "saxs_2d")}
    before["G2"] = np.array(opened.G2)

    _average("standalone", tmp_path, [a.name, b.name], ALL_FIELDS, tmp_path / "o.h5")

    np.testing.assert_allclose(_read(tmp_path / "o.h5", "g2"), 1.02)  # it did average
    for k, v in before.items():
        np.testing.assert_array_equal(getattr(opened, k), v, err_msg=k)
    again = XpcsFile(str(a), fields=["saxs_2d", "G2"])  # via the shared read cache
    np.testing.assert_array_equal(again.g2, before["g2"])
    np.testing.assert_array_equal(again.saxs_2d, before["saxs_2d"])


def test_r3_32_accumulate_never_aliases_first_contribution():
    first = {
        "g2": np.ones((2, 2)),
        "g2_err": np.full((2, 2), 0.1),
        "saxs_1d": np.ones(2, dtype=np.int32),
        "_coords": {},
    }
    saved = {k: v.copy() for k, v in first.items() if k != "_coords"}
    result = at._accumulate_into({}, ["g2", "g2_err", "saxs_1d"], first)
    for k, v in saved.items():
        np.testing.assert_array_equal(first[k], v)
        assert not np.shares_memory(result[k], first[k])
        assert result[k].dtype == np.float64
    result["g2"] += 1  # an in-place update must not reach the input either
    np.testing.assert_array_equal(first["g2"], saved["g2"])


# ------------------------------------------------------------------- R3-07
@pytest.mark.parametrize("kind", ["class", "standalone"])
@pytest.mark.parametrize(
    "geometry",
    [
        {"bcx": 3.0},
        {"bcy": 3.0},
        {"det_dist": 6.0},
        {"pixel_size": 172e-6},
        {"energy": 12.0},
        {"mask": np.triu(np.ones((H, W), np.int32))},
        {"dqmap": np.ones((H, W), np.int32)},
    ],
    ids=["bcx", "bcy", "det_dist", "pixel_size", "energy", "mask", "dqmap"],
)
def test_r3_07_real_files_with_different_detector_geometry_are_not_averaged(
    tmp_path, kind, geometry
):
    names = ["f0_0001.h5", "f1_0001.h5", "f2_0001.h5"]
    _write_nexus(tmp_path / names[0], saxs=1.0)
    _write_nexus(tmp_path / names[1], saxs=9.0, **geometry)
    _write_nexus(tmp_path / names[2], saxs=5.0)
    out = tmp_path / "o.h5"

    n_accepted = _average(kind, tmp_path, names, ["saxs_2d"], out)

    if kind == "class":
        assert n_accepted == 2
    np.testing.assert_allclose(_read(out, "saxs_2d"), 3.0)  # mean of 1 and 5
    assert _read(out, "saxs_2d").shape[-2:] == (H, W)


def test_r3_07_threaded_class_path_checks_geometry_too(tmp_path):
    names = [f"f{i}_0001.h5" for i in range(8)]
    for i, name in enumerate(names):
        _write_nexus(tmp_path / name, saxs=float(i), bcx=3.0 if i == 5 else 2.0)
    n_accepted = _average("class", tmp_path, names, ["saxs_2d"], tmp_path / "o.h5")
    assert n_accepted == 7
    np.testing.assert_allclose(_read(tmp_path / "o.h5", "saxs_2d"), (28 - 5) / 7)


@pytest.mark.parametrize("kind", ["class", "standalone"])
def test_r3_07_unnormalized_g2_also_requires_matching_geometry(tmp_path, kind):
    names = ["f0_0001.h5", "f1_0001.h5"]
    _write_nexus(tmp_path / names[0], g2=1.0)
    _write_nexus(tmp_path / names[1], g2=1.04, bcx=3.0)
    out = tmp_path / "o.h5"
    _average(kind, tmp_path, names, ["G2"], out)
    np.testing.assert_allclose(_read(out, "G2"), 1.0)


@pytest.mark.parametrize("kind", ["class", "standalone"])
def test_r3_07_real_files_with_equal_geometry_are_still_averaged(tmp_path, kind):
    names = ["f0_0001.h5", "f1_0001.h5"]
    _write_nexus(tmp_path / names[0], saxs=1.0, g2=1.0)
    _write_nexus(tmp_path / names[1], saxs=3.0, g2=1.04)
    out = tmp_path / "o.h5"
    n_accepted = _average(kind, tmp_path, names, ALL_FIELDS, out)
    if kind == "class":
        assert n_accepted == 2
    np.testing.assert_allclose(_read(out, "saxs_2d"), 2.0)
    np.testing.assert_allclose(_read(out, "g2"), 1.02)
    np.testing.assert_allclose(_read(out, "G2"), 1.02)


def _ns(**attrs):
    from types import SimpleNamespace

    return SimpleNamespace(**attrs)


def test_r3_07_missing_detector_attribute_is_a_mismatch():
    full = _ns(
        bcx=1.0, bcy=1.0, det_dist=5.0, pixel_size=1e-4, X_energy=8.0, mask=np.ones(2)
    )
    partial = _ns(bcx=1.0, bcy=1.0, det_dist=5.0, pixel_size=1e-4, X_energy=8.0)
    ref: dict = {}
    assert at._accept_coords(ref, "a", at._coords_of(full, ["saxs_2d"]))
    assert not at._accept_coords(ref, "b", at._coords_of(partial, ["saxs_2d"]))
    assert at._accept_coords(ref, "c", at._coords_of(full, ["saxs_2d"]))


def test_r3_07_every_averageable_field_has_declared_coordinates():
    # every field the GUI checkboxes and the standalone default can request
    xf = _ns()
    for field in ("G2", "IP", "IF", "g2", "g2_err", "saxs_1d", "saxs_2d"):
        assert at._coords_of(xf, [field]), field
    assert {"bcx", "mask"} <= set(at._coords_of(xf, ["saxs_2d"]))
    assert {"bcx", "t_el", "dynamic_index_mapping"} <= set(at._coords_of(xf, ["G2"]))
    assert "bcx" not in at._coords_of(xf, ["g2", "g2_err"])
    assert {"sqlist", "static_index_mapping"} <= set(at._coords_of(xf, ["saxs_1d"]))
    # a field with no declared coordinates must not escape the check
    assert {"bcx", "t_el", "sqlist"} <= set(at._coords_of(xf, ["Iqp"]))


# ------------------------------------------------------------------- R3-06
@pytest.mark.parametrize("kind", ["class", "standalone"])
def test_r3_06_real_files_with_unequal_large_bin_ids_are_not_averaged(tmp_path, kind):
    names = ["f0_0001.h5", "f1_0001.h5", "f2_0001.h5"]
    _write_nexus(tmp_path / names[0], g2=1.0, dyn_map=(0, 100000))
    _write_nexus(tmp_path / names[1], g2=1.04, dyn_map=(0, 100001))
    _write_nexus(tmp_path / names[2], g2=1.02, dyn_map=(0, 100000))
    out = tmp_path / "o.h5"
    n_accepted = _average(kind, tmp_path, names, ["g2", "g2_err"], out)
    if kind == "class":
        assert n_accepted == 2
    np.testing.assert_allclose(_read(out, "g2"), 1.01)


@pytest.mark.parametrize("dtype", [np.int64, np.uint32, np.float64])
def test_r3_06_index_mappings_compare_exactly(dtype):
    a = {"dynamic_index_mapping": np.array([0, 100000]).astype(dtype)}
    b = {"dynamic_index_mapping": np.array([0, 100001]).astype(dtype)}
    ref: dict = {}
    assert at._accept_coords(ref, "a", a)
    assert not at._accept_coords(ref, "b", b)


def test_r3_06_any_integer_array_compares_exactly():
    a = {"anything": np.array([100000], dtype=np.int64)}
    ref: dict = {}
    assert at._accept_coords(ref, "a", a)
    assert not at._accept_coords(ref, "b", {"anything": np.array([100001])})


def test_r3_06_continuous_coordinates_use_the_explicit_tolerance():
    ref: dict = {}
    base = np.array([1e-6, 1.0, 100.0])
    assert at._accept_coords(ref, "a", {"t_el": base})
    assert at._accept_coords(ref, "b", {"t_el": base * (1 + 1e-9)})
    assert not at._accept_coords(ref, "c", {"t_el": base * (1 + 1e-3)})
    # no absolute floor: a 1% error on a microsecond delay is still a mismatch
    assert not at._accept_coords(ref, "d", {"t_el": base + [1e-8, 0, 0]})


# ------------------------------------------------------------------- R3-01
def _template_and_inputs(tmp_path):
    template = tmp_path / "f0_0001.h5"
    _write_nexus(template)
    return str(template), [str(template)]


@pytest.mark.parametrize("mode", [0o600, 0o640, 0o664, 0o444])
def test_r3_01_existing_output_keeps_its_permission_bits(tmp_path, mode):
    template, inputs = _template_and_inputs(tmp_path)
    out = tmp_path / "out.h5"
    out.write_bytes(b"old")
    os.chmod(out, mode)
    old_umask = os.umask(0o002)  # would make a fresh file 0664
    try:
        at._publish_average(template, str(out), {"g2": np.full((3, 2), 7.0)}, inputs)
    finally:
        os.umask(old_umask)
    assert stat.S_IMODE(out.stat().st_mode) == mode
    np.testing.assert_allclose(_read(out, "g2"), 7.0)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["f0_0001.h5", "out.h5"]


@pytest.mark.parametrize("umask", [0o022, 0o077, 0o002])
def test_r3_01_new_output_gets_umask_derived_mode(tmp_path, umask):
    template, inputs = _template_and_inputs(tmp_path)
    out = tmp_path / "out.h5"
    old_umask = os.umask(umask)
    try:
        at._publish_average(template, str(out), {"g2": np.full((3, 2), 7.0)}, inputs)
    finally:
        os.umask(old_umask)
    assert stat.S_IMODE(out.stat().st_mode) == 0o666 & ~umask
    assert sorted(p.name for p in tmp_path.iterdir()) == ["f0_0001.h5", "out.h5"]


def test_r3_01_staging_file_is_private_while_it_is_written(tmp_path, monkeypatch):
    template, inputs = _template_and_inputs(tmp_path)
    seen = []
    real_put = at.put

    def spying_put(path, *a, **k):
        seen.append(stat.S_IMODE(os.stat(path).st_mode))
        return real_put(path, *a, **k)

    monkeypatch.setattr(f"{TARGET}.put", spying_put)
    old_umask = os.umask(0o002)
    try:
        at._publish_average(
            template, str(tmp_path / "out.h5"), {"g2": np.full((3, 2), 7.0)}, inputs
        )
    finally:
        os.umask(old_umask)
    assert seen == [0o600]


def test_r3_01_failed_write_leaves_private_output_untouched(tmp_path, monkeypatch):
    template, inputs = _template_and_inputs(tmp_path)
    out = tmp_path / "out.h5"
    out.write_bytes(b"old")
    os.chmod(out, 0o600)

    def boom(*a, **k):
        raise OSError("boom")

    monkeypatch.setattr(f"{TARGET}.put", boom)
    with pytest.raises(OSError, match="boom"):
        at._publish_average(template, str(out), {}, inputs)
    assert out.read_bytes() == b"old"
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    assert sorted(p.name for p in tmp_path.iterdir()) == ["f0_0001.h5", "out.h5"]


def test_r3_01_symlink_at_output_is_replaced_not_written_through(tmp_path):
    template, inputs = _template_and_inputs(tmp_path)
    target = tmp_path / "target.h5"
    target.write_bytes(b"target")
    os.chmod(target, 0o600)
    out = tmp_path / "out.h5"
    out.symlink_to(target)
    at._publish_average(template, str(out), {"g2": np.full((3, 2), 7.0)}, inputs)
    assert not out.is_symlink()
    assert target.read_bytes() == b"target"
    np.testing.assert_allclose(_read(out, "g2"), 7.0)


def test_r3_01_unretainable_mode_fails_clearly_and_keeps_output(tmp_path, monkeypatch):
    template, inputs = _template_and_inputs(tmp_path)
    out = tmp_path / "out.h5"
    out.write_bytes(b"old")

    def refuse(*a, **k):
        raise PermissionError("cannot copy mode")

    monkeypatch.setattr(f"{TARGET}.copymode", refuse)
    with pytest.raises(PermissionError):
        at._publish_average(template, str(out), {"g2": np.full((3, 2), 7.0)}, inputs)
    assert out.read_bytes() == b"old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["f0_0001.h5", "out.h5"]
