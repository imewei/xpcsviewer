"""Regression tests for review round 3 (second wave): R3-13, R3-20, R3-24.

R3-13  an imported Q-map partition is DISPLAY-ONLY: every quantitative or
       selection path reads the file's own maps (``xf.dqmap`` / ``xf.sqmap`` /
       ``xf.qmap``); only the designated Q-map displays show the import.
R3-20  Pie / Ring ROI export works on a real loaded file, with calibrated
       q / phi from the real ``QMap`` (not attributes only a stub has).
R3-24  the diffusion plot and the diffusion export select the same fit and the
       export records the fit's method and provenance.

The R3-13 and R3-20 tests write real NeXus-layout HDF5 files with h5py and
load them through the real ``XpcsFile`` / ``QMap`` (no stand-in objects).
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest

from xpcsviewer.fileIO.aps_8idi import key as hdf_key
from xpcsviewer.fileIO.hdf_reader import clear_connection_pool
from xpcsviewer.utils.memory_manager import MemoryPressure
from xpcsviewer.xpcs_file import XpcsFile

pytestmark = pytest.mark.unit

K = hdf_key["nexus"]
H = W = 8
BC = 3.5  # beam centre (pixels): half-integer so no pixel sits on an axis
PIXEL = 75e-6  # m
DIST = 5.0  # m
ENERGY = 8.0  # keV
K0 = 2 * math.pi / (12.398 / ENERGY)  # 1/Angstrom

# static bins are annuli in pixel radius: id 1: r < 2, id 2: 2 <= r < 3.5, id 3: rest
R_EDGES = (2.0, 3.5)
R_CENTRES = (1.5, 2.75, 4.25)  # pixels; the file's sqlist is q at these radii


def q_at(r_pixel: float) -> float:
    """q [1/Angstrom] at a radius in pixels: k0 * sin(arctan(r * pixel / L))."""
    return K0 * math.sin(math.atan(r_pixel * PIXEL / DIST))


def _pixel_polar():
    """Independent oracle: (r_pixel, phi_deg in [0, 360)) per pixel."""
    r = np.zeros((H, W))
    phi = np.zeros((H, W))
    for y in range(H):
        for x in range(W):
            dy, dx = y - BC, x - BC
            r[y, x] = math.hypot(dx, dy)
            phi[y, x] = (-math.degrees(math.atan2(dy, dx))) % 360.0
    return r, phi


def _static_map():
    r, _ = _pixel_polar()
    return np.where(r < R_EDGES[0], 1, np.where(r < R_EDGES[1], 2, 3)).astype(np.int32)


def _intensity():
    yy, xx = np.mgrid[0:H, 0:W]
    return 1.0 + yy + 10.0 * xx  # all positive, every pixel distinct


def _write_nexus(path, *, dqmap=None, saxs=None, mask=None, azimuthal=False):
    """Smallest multitau NeXus file the real ``XpcsFile`` can load."""
    cols = np.tile(np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32), (H, 1))
    with h5py.File(path, "w") as f:
        f[K["g2"]] = np.full((3, 2), 1.2)
        f[K["g2_err"]] = np.full((3, 2), 0.01)
        f[K["G2"]] = np.full((3, 2, 3), 1.2)
        f[K["tau"]] = np.array([1.0, 2.0, 4.0])
        f[K["stride_frame"]] = 1
        f[K["avg_frame"]] = 1
        f[K["t0"]] = 0.1
        f[K["t1"]] = 0.1
        f[K["start_time"]] = "2024-01-01"
        f[K["Int_t"]] = np.ones((2, 5))
        n_static = 4 if azimuthal else 3  # compressed static bins
        f[K["saxs_1d"]] = np.full((1, n_static), 1.0)
        f[K["Iqp"]] = np.full((1, n_static), 1.0)
        f[K["saxs_2d"]] = (_intensity() if saxs is None else saxs)[None]
        f[K["mask"]] = np.ones((H, W), np.int32) if mask is None else mask
        f[K["dqmap"]] = cols if dqmap is None else dqmap
        sq = _static_map()
        if azimuthal:
            # 3 q bins x 2 azimuth bins, sparse: only flat grid cells 1, 2, 4, 5
            # (= q0/phi1, q1/phi0, q2/phi0, q2/phi1) hold pixels, as bins 1..4:
            # the outer annulus is split into a left (bin 3) and right (bin 4) half
            right = np.arange(W)[None, :] >= W // 2
            sq = np.where((sq == 3) & right, 4, sq)
        f[K["sqmap"]] = sq
        f[K["dqlist"]] = np.array([0.01, 0.02])
        f[K["sqlist"]] = np.array([q_at(r) for r in R_CENTRES])
        f[K["dplist"]] = np.array([0.0])
        f[K["splist"]] = np.array([0.0, 180.0]) if azimuthal else np.array([0.0])
        f[K["bcx"]] = BC
        f[K["bcy"]] = BC
        f[K["X_energy"]] = ENERGY
        f[K["pixel_size"]] = PIXEL
        f[K["det_dist"]] = DIST
        f[K["static_index_mapping"]] = (
            np.array([1, 2, 4, 5]) if azimuthal else np.array([0, 1, 2])
        )
        f[K["dynamic_index_mapping"]] = np.array([0, 1])
        f[K["static_num_pts"]] = np.array([3, 2] if azimuthal else [3, 1])
        f[K["dynamic_num_pts"]] = np.array([2, 1])
        f[K["map_names"]] = np.array([b"q", b"phi"])
        f[K["map_units"]] = np.array([b"1/A", b"deg"])


@pytest.fixture(autouse=True)
def _fresh_pool():
    clear_connection_pool()
    yield
    clear_connection_pool()


@pytest.fixture
def real_xf(tmp_path):
    path = tmp_path / "roi_0001.h5"
    _write_nexus(path)
    return XpcsFile(str(path))


# ------------------------------------------------------------------ R3-20
def _pie_expected(angle_range, dist_pixel):
    """Mean intensity per static q bin inside the wedge, NaN beyond the cut-off."""
    # angle_range is in the PieROI frame: atan2(row, column) - 90 degrees
    yy, xx = np.mgrid[0:H, 0:W]
    phi = (np.degrees(np.arctan2(yy - BC, xx - BC)) - 90.0) % 360.0
    sq, img = _static_map(), _intensity()
    pmin, pmax = angle_range
    if pmax < pmin:
        pmax, phi = pmax + 360.0, np.where(phi < pmin, phi + 360.0, phi)
    inside = (phi >= pmin) & (phi < pmax)
    qlist = np.array([q_at(r) for r in R_CENTRES])
    y = np.full(3, np.nan)
    for k in (1, 2, 3):
        sel = inside & (sq == k)
        if sel.any():
            y[k - 1] = img[sel].mean()
    y[qlist >= q_at(dist_pixel)] = np.nan
    return qlist, y


def test_r3_20_real_file_pie_export_uses_calibrated_q_and_phi(real_xf):
    roi = {"sl_type": "Pie", "angle_range": (0.0, 90.0), "dist": 3.0}
    x, y = real_xf.get_roi_data(roi)
    x_exp, y_exp = _pie_expected(roi["angle_range"], roi["dist"])
    np.testing.assert_allclose(x, x_exp, rtol=1e-6)
    np.testing.assert_allclose(y, y_exp, rtol=1e-6)
    assert np.isnan(y[2]) and np.isfinite(y[:2]).all()  # bin 3 lies beyond dist


def test_r3_20_real_file_pie_streaming_matches_eager(real_xf):
    roi = {"sl_type": "Pie", "angle_range": (270.0, 90.0), "dist": 50.0}  # wraps 0
    x_e, y_e = real_xf.get_roi_data(roi)
    out = real_xf._get_roi_data_streaming(roi)
    assert out is not None
    np.testing.assert_allclose(out[0], x_e)
    np.testing.assert_allclose(out[1], y_e, equal_nan=True)
    assert np.isfinite(y_e).all()


def test_r3_20_real_file_ring_export_bins_phi(real_xf):
    roi = {"sl_type": "Ring", "radius": (1.0, 3.0)}
    x, y = real_xf.get_roi_data(roi, phi_num=4)
    r, phi = _pixel_polar()
    sel = (r >= 1.0) & (r < 3.0)
    lo, hi = phi[sel].min(), phi[sel].max()
    img = _intensity()
    np.testing.assert_allclose(x, np.linspace(lo, hi, 4))
    idx = np.clip(((phi - lo) / ((hi - lo) / 4)).astype(int), 0, 3)
    expected = [img[sel & (idx == i)].mean() for i in range(4)]
    np.testing.assert_allclose(y, expected)


def test_r3_20_real_file_ring_swapped_radii_and_empty_annulus(real_xf, caplog):
    x1, y1 = real_xf.get_roi_data({"sl_type": "Ring", "radius": (1.0, 3.0)}, 4)
    x2, y2 = real_xf.get_roi_data({"sl_type": "Ring", "radius": (3.0, 1.0)}, 4)
    np.testing.assert_allclose(x1, x2)
    np.testing.assert_allclose(y1, y2)
    with caplog.at_level("WARNING"):
        xe, ye = real_xf.get_roi_data({"sl_type": "Ring", "radius": (50.0, 60.0)})
    assert xe.size == 0 and ye.size == 0
    assert "no pixels" in caplog.text


def test_r3_20_real_file_one_angle_annulus_has_no_zero_division(tmp_path):
    # the file's own mask keeps a single pixel: the annulus holds one angle
    mask = np.zeros((H, W), np.int32)
    mask[2, 5] = 1
    path = tmp_path / "one_0001.h5"
    _write_nexus(path, mask=mask)
    xf = XpcsFile(str(path))
    with np.errstate(all="raise"):
        x, y = xf.get_roi_data({"sl_type": "Ring", "radius": (0.0, 5.0)})
    np.testing.assert_allclose(x, [_pixel_polar()[1][2, 5]])
    np.testing.assert_allclose(y, [_intensity()[2, 5]])


def test_r3_20_real_file_export_saxs1d_writes_pie_and_ring(real_xf, tmp_path):
    out = tmp_path / "export"
    out.mkdir()
    rois = [
        {"sl_type": "Pie", "angle_range": (0.0, 90.0), "dist": 3.0},
        {"sl_type": "Ring", "radius": (1.0, 3.0)},
    ]
    real_xf.export_saxs1d(rois, str(out))
    names = sorted(p.name for p in out.iterdir())
    assert any("_Pie_000.txt" in n for n in names)
    assert any("_Ring_001.txt" in n for n in names)
    pie = next(p for p in out.iterdir() if "_Pie_000" in p.name)
    q, i = np.loadtxt(pie, unpack=True)
    np.testing.assert_allclose(q, _pie_expected((0.0, 90.0), 3.0)[0], rtol=1e-6)
    assert "q(1/Angstrom)" in pie.read_text().splitlines()[0]


def test_r3_20_real_file_streaming_dispatch_and_ring_fallback(real_xf, monkeypatch):
    big = real_xf.saxs_2d.view(type("Big", (np.ndarray,), {"nbytes": 600 * 2**20}))
    real_xf.saxs_2d = big
    monkeypatch.setattr(
        real_xf._memory_manager, "get_memory_pressure", lambda: MemoryPressure.HIGH
    )
    pie = {"sl_type": "Pie", "angle_range": (0.0, 360.0), "dist": 50.0}
    ring = {"sl_type": "Ring", "radius": (1.0, 3.0)}
    x_s, y_s = real_xf.get_roi_data(pie)  # streaming dispatch
    monkeypatch.setattr(
        real_xf._memory_manager, "get_memory_pressure", lambda: MemoryPressure.LOW
    )
    x_e, y_e = real_xf.get_roi_data(pie)
    np.testing.assert_allclose(y_s, y_e, equal_nan=True)
    monkeypatch.setattr(
        real_xf._memory_manager, "get_memory_pressure", lambda: MemoryPressure.HIGH
    )
    xr, yr = real_xf.get_roi_data(ring, phi_num=4)
    xe, ye = real_xf._roi_profile(ring, 4)  # chunked and eager Ring agree
    np.testing.assert_allclose(xr, xe)
    np.testing.assert_allclose(yr, ye, equal_nan=True)
    assert xr.size == 4 and np.isfinite(yr).all()


def test_r3_20_real_file_vectorized_entry_points_use_the_same_result(real_xf):
    roi = {"sl_type": "Pie", "angle_range": (0.0, 90.0), "dist": 3.0}
    x, y = real_xf.get_roi_data(roi)
    xv, yv = real_xf.get_roi_data_vectorized(roi)
    np.testing.assert_allclose(xv, x)
    np.testing.assert_allclose(yv, y, equal_nan=True)
    (xp, yp), (xq, yq) = real_xf.get_multiple_roi_data_parallel(
        [roi, {"sl_type": "Ring", "radius": (1.0, 3.0)}]
    )
    np.testing.assert_allclose(xp, x)
    assert xq.size > 0


def test_r3_20_ring_bins_without_pixels_are_nan_not_zero(real_xf):
    _x, y = real_xf.get_roi_data({"sl_type": "Ring", "radius": (1.0, 3.0)}, 180)
    assert np.isnan(y).any() and np.nanmin(y) > 0


def test_r3_20_streaming_has_no_other_roi_types(real_xf):
    assert real_xf._get_roi_data_streaming({"sl_type": "Phi_ring"}) is None
    assert real_xf.get_roi_data({"sl_type": "Phi_ring"}) is None


def test_r3_20_unsupported_roi_type_is_skipped_by_export(real_xf, tmp_path, caplog):
    out = tmp_path / "e2"
    out.mkdir()
    with caplog.at_level("WARNING"):
        real_xf.export_saxs1d([{"sl_type": "Mystery"}], str(out))
    assert "Mystery" in caplog.text
    assert not [p for p in out.iterdir() if "Mystery" in p.name]


# ------------------------------------------------------------------ R3-13
def _swap_labels(arr, a, b):
    out = np.array(arr, copy=True)
    out[arr == a], out[arr == b] = b, a
    return out


def _import_partition(xf, partition):
    """Run the real ``XpcsViewer.import_partition`` on a real XpcsFile."""
    from xpcsviewer.xpcs_viewer import XpcsViewer

    viewer = SimpleNamespace(
        vk=SimpleNamespace(target=["f"], get_xf_list=lambda: [xf]),
        statusbar=MagicMock(),
        update_plot=MagicMock(),
        _display_partition_version=0,
        _invalidate_plots=MagicMock(),
    )
    XpcsViewer.import_partition(viewer, partition)
    viewer.update_plot.assert_called_once()


def _canonical_maps(xf):
    return np.array(xf.qmap.dqmap, copy=True), np.array(xf.qmap.sqmap, copy=True)


@pytest.fixture
def imported(real_xf):
    """(xf, canonical dqmap, canonical sqmap, imported dqmap, imported sqmap)."""
    dq0, sq0 = _canonical_maps(real_xf)
    dq1, sq1 = _swap_labels(dq0, 1, 2), _swap_labels(sq0, 1, 2)
    _import_partition(real_xf, {"dynamic_roi_map": dq1, "static_roi_map": sq1})
    return real_xf, dq0, sq0, dq1, sq1


def test_r3_13_import_keeps_canonical_maps_on_the_file(imported):
    xf, dq0, sq0, dq1, sq1 = imported
    np.testing.assert_array_equal(xf.dqmap, dq0)
    np.testing.assert_array_equal(xf.sqmap, sq0)
    np.testing.assert_array_equal(xf.get_cropped_qmap("dqmap", enabled=False), dq0)
    np.testing.assert_array_equal(xf.get_cropped_qmap("sqmap", enabled=False), sq0)
    np.testing.assert_array_equal(xf.get_display_qmap("dqmap"), dq1)
    np.testing.assert_array_equal(xf.get_display_qmap("sqmap"), sq1)


def test_r3_13_roi_export_is_unchanged_by_a_display_only_import(real_xf):
    pie = {"sl_type": "Pie", "angle_range": (0.0, 90.0), "dist": 50.0}
    before = real_xf.get_roi_data(pie)
    dq0, sq0 = _canonical_maps(real_xf)
    _import_partition(
        real_xf,
        {
            "dynamic_roi_map": _swap_labels(dq0, 1, 2),
            "static_roi_map": _swap_labels(sq0, 1, 3),
        },
    )
    after = real_xf.get_roi_data(pie)
    np.testing.assert_allclose(after[0], before[0])
    np.testing.assert_allclose(after[1], before[1], equal_nan=True)
    assert np.isfinite(after[1]).all()


def test_r3_13_twotime_maps_keep_the_files_own_partition(real_xf):
    before = real_xf.get_twotime_maps(scale="linear", auto_crop=False)[0]
    dq0, _sq0 = _canonical_maps(real_xf)
    _import_partition(real_xf, {"dynamic_roi_map": _swap_labels(dq0, 1, 2)})
    after = real_xf.get_twotime_maps(scale="linear", auto_crop=False)[0]
    np.testing.assert_array_equal(after, before)


def test_r3_13_g2_map_draws_the_files_own_detector_map(imported, qapp):
    from xpcsviewer.viewer_kernel import ViewerKernel

    xf, dq0, _sq0, dq1, _sq1 = imported
    qmap_hdl = MagicMock()
    kernel = SimpleNamespace(get_xf_list=lambda rows=None: [xf])
    ViewerKernel.plot_g2map(kernel, MagicMock(), qmap_hdl, MagicMock(), qbin=0)
    drawn = qmap_hdl.setImage.call_args.args[0]
    np.testing.assert_array_equal(drawn, dq0)  # whole 8x8 map: nothing to crop
    assert not np.array_equal(drawn, dq1)


@pytest.mark.parametrize(
    ("target", "which"),
    [("dynamic_roi_map", 3), ("static_roi_map", 4)],
)
def test_r3_13_designated_qmap_display_shows_the_import(imported, target, which):
    from xpcsviewer.viewer_kernel import ViewerKernel

    xf = imported[0]
    hdl = MagicMock()
    kernel = SimpleNamespace(get_xf_list=lambda rows=None: [xf])
    ViewerKernel.plot_qmap(kernel, hdl, target=target)
    np.testing.assert_array_equal(hdl.setImage.call_args.args[0], imported[which])


def test_r3_13_qmap_display_without_import_shows_the_files_map(real_xf):
    from xpcsviewer.viewer_kernel import ViewerKernel

    hdl = MagicMock()
    kernel = SimpleNamespace(get_xf_list=lambda rows=None: [real_xf])
    ViewerKernel.plot_qmap(kernel, hdl, target="dynamic_roi_map")
    np.testing.assert_array_equal(hdl.setImage.call_args.args[0], real_xf.qmap.dqmap)


def test_r3_13_no_code_assigns_dqmap_or_sqmap_on_a_file_object():
    """The bug class: assigning xf.dqmap / xf.sqmap shadows the file's own map."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[2] / "xpcsviewer"
    pattern = re.compile(r"\b(?!self\b)\w+\.(dqmap|sqmap)\s*=(?!=)")
    skip = {"qmap_utils.py", "validators.py", "hdf5_facade.py"}
    offenders = [
        f"{p.relative_to(root)}:{n}"
        for p in root.rglob("*.py")
        if p.name not in skip
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if pattern.search(line) and not line.lstrip().startswith("#")
    ]
    assert offenders == []


# ------------------------------------------------------------------ R3-24
TAUQ_BOUNDS = np.array([[1e-3, -3.0], [1e3, 3.0]])
Q_FIT = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])


def _g2_summary(a, b, **extra):
    """Legacy-format G2 fit summary whose tau(q) is exactly a * q**b."""
    tau = a * Q_FIT**b
    fit_val = np.zeros((len(Q_FIT), 2, 4))
    fit_val[:, 0, 0] = 0.5  # contrast
    fit_val[:, 0, 1] = tau
    fit_val[:, 1, 1] = 0.01 * tau
    fit_val[:, 0, 2] = 1.0  # stretch
    fit_val[:, 0, 3] = 1.0  # baseline
    return {
        "fit_func": "single_exp",
        "fit_val": fit_val,
        "q_val": Q_FIT.copy(),
        "failed_mask": np.zeros(len(Q_FIT), dtype=bool),
        **extra,
    }


def _diffusion_kernel(monkeypatch, xf):
    from xpcsviewer import viewer_kernel
    from xpcsviewer.viewer_kernel import ViewerKernel

    shown = []
    kernel = ViewerKernel.__new__(ViewerKernel)
    monkeypatch.setattr(kernel, "get_xf_list", lambda **k: [xf], raising=False)
    monkeypatch.setattr(
        viewer_kernel,
        "_get_module",
        lambda name: SimpleNamespace(
            plot=lambda xfs, **k: shown.append(xfs[0].fit_summary),
            plot_pre=lambda xfs, hdl: shown.append(xfs[0].fit_summary),
        ),
    )
    return kernel, shown


def _plot_and_export(kernel, tmp_path):
    out = kernel.plot_tauq(
        hdl=None,
        bounds=TAUQ_BOUNDS,
        rows=[0],
        fit_flag=np.array([True, True]),
        q_range=(0.0, 1.0),
    )
    kernel.export_diffusion(str(tmp_path), rows=[0])
    return out


def _exported_params(tmp_path, stem):
    text = (tmp_path / f"{stem}_diffusion_fitting.txt").read_text()
    rows = {
        line.split("\t")[0]: float(line.split("\t")[1])
        for line in text.splitlines()
        if line[:2] in ("a\t", "b\t")
    }
    return rows, text


def test_r3_24_bayesian_only_fit_is_plotted_and_exported(
    real_xf, monkeypatch, tmp_path
):
    real_xf.fit_summary = None
    real_xf.bayesian_fit_summary = _g2_summary(4.0, -1.5, fname=real_xf.fname)
    kernel, shown = _diffusion_kernel(monkeypatch, real_xf)
    result = _plot_and_export(kernel, tmp_path)
    assert shown and shown[0] is real_xf.bayesian_fit_summary
    assert "a = " in result[real_xf.label]
    params, text = _exported_params(tmp_path, "roi_0001")
    assert params["a"] == pytest.approx(4.0, rel=1e-3)
    assert params["b"] == pytest.approx(-1.5, abs=1e-3)
    assert "# Fit method: bayesian" in text
    assert "NOT credible intervals" in text
    assert f"# G2 fit source file: {real_xf.fname}" in text
    assert real_xf.fit_summary is None  # the displayed summary is restored


def test_r3_24_export_matches_plot_when_both_fits_exist(real_xf, monkeypatch, tmp_path):
    real_xf.fit_summary = _g2_summary(1.0, -1.0)
    stale = {  # an older deterministic tau-q fit that must not be exported
        "tauq_success": True,
        "tauq_fit_val": np.array([[9.0, -2.0], [0.1, 0.1]]),
    }
    real_xf.fit_summary.update(stale)
    nlsq = real_xf.fit_summary
    real_xf.bayesian_fit_summary = _g2_summary(4.0, -1.5, fname=real_xf.fname)
    kernel, shown = _diffusion_kernel(monkeypatch, real_xf)
    _plot_and_export(kernel, tmp_path)
    assert shown[0] is real_xf.bayesian_fit_summary
    params, text = _exported_params(tmp_path, "roi_0001")
    assert params["a"] == pytest.approx(4.0, rel=1e-3)  # not the stale 9.0
    assert "# Fit method: bayesian" in text
    assert real_xf.fit_summary is nlsq
    assert nlsq["tauq_fit_val"][0, 0] == 9.0  # NLSQ summary untouched by the plot


def test_r3_24_nlsq_only_export_records_its_method(real_xf, monkeypatch, tmp_path):
    real_xf.fit_summary = _g2_summary(2.0, -1.0)
    kernel, shown = _diffusion_kernel(monkeypatch, real_xf)
    _plot_and_export(kernel, tmp_path)
    assert shown[0] is real_xf.fit_summary
    params, text = _exported_params(tmp_path, "roi_0001")
    assert params["a"] == pytest.approx(2.0, rel=1e-3)
    assert "# Fit method: nlsq" in text
    assert "credible" not in text.replace("NOT credible", "")
    summary = (tmp_path / "diffusion_summary.txt").read_text()
    assert "fit_method" in summary and "\tnlsq\t" in summary


def test_r3_24_failed_tauq_fit_is_still_not_displayed_or_exported(
    real_xf, monkeypatch, tmp_path
):
    real_xf.fit_summary = None
    bad = _g2_summary(4.0, -1.5)
    bad["failed_mask"][:] = True  # no usable G2 fit in any bin
    real_xf.bayesian_fit_summary = bad
    kernel, _shown = _diffusion_kernel(monkeypatch, real_xf)
    result = kernel.plot_tauq(
        hdl=None,
        bounds=TAUQ_BOUNDS,
        rows=[0],
        fit_flag=np.array([True, True]),
        q_range=(0.0, 1.0),
    )
    assert result[real_xf.label] == "tauq fitting failed"
    with pytest.raises(ValueError, match="failed"):
        kernel.export_diffusion(str(tmp_path), rows=[0])
    assert not list(tmp_path.glob("*diffusion*"))


# ------------------------------------------------------------ hand-over changes
def test_handover_c2_cache_is_keyed_by_file_revision(real_xf, tmp_path, monkeypatch):
    import os
    import sys

    module = sys.modules[XpcsFile.__module__]
    calls = []
    monkeypatch.setattr(
        module,
        "get_single_c2_from_hdf",
        lambda *a, **k: calls.append(1) or np.ones((2, 2)),
    )
    real_xf.__dict__.update(c2_processed_bins=np.array([1]), c2_t0=1.0)
    real_xf.get_twotime_c2()
    real_xf.get_twotime_c2()
    assert len(calls) == 1  # unchanged file: cached
    replacement = tmp_path / "replacement.h5"
    _write_nexus(replacement, saxs=np.full((H, W), 7.0))
    os.replace(replacement, real_xf.fname)  # new revision at the same path
    real_xf.get_twotime_c2()
    assert len(calls) == 2


def test_handover_plot_twotime_replaces_a_stale_cached_file_object(monkeypatch):
    import threading

    from xpcsviewer import viewer_kernel
    from xpcsviewer.viewer_kernel import ViewerKernel

    def _xf(tag):
        return SimpleNamespace(
            fname="/d/a.h5", tag=tag, get_twotime_qbin_labels=lambda: [tag]
        )

    old, new = _xf("old"), _xf("new")
    plotted = []
    kernel = SimpleNamespace(
        get_xf_list=lambda rows=None, filter_atype=None: [new],
        _get_cached_dataset=lambda fname, fallback: old,  # same path, older object
        _dset_cache_lock=threading.Lock(),
        _current_dset_cache={"/d/a.h5": old},
        _memory_cleanup_threshold=2.0,
        _cleanup_memory=lambda: None,
    )
    monkeypatch.setattr(
        viewer_kernel,
        "_get_module",
        lambda name: SimpleNamespace(
            plot_twotime=lambda dset, hdl, **k: plotted.append(dset)
        ),
    )
    monkeypatch.setattr(
        viewer_kernel.MemoryMonitor, "is_memory_pressure_high", lambda *a, **k: False
    )
    labels = ViewerKernel.plot_twotime(kernel, MagicMock(), rows=[0])
    assert plotted == [new]
    assert kernel._current_dset_cache["/d/a.h5"] is new
    assert labels == ["new"]


def test_handover_diffusion_bayesian_input_drops_failed_q_bins(real_xf):
    from xpcsviewer.xpcs_viewer import XpcsViewer

    summary = _g2_summary(4.0, -1.5)
    summary["fit_success"] = np.array([True, True, False, True, True, True])
    summary["fit_val"][2, 0, 1] = 123.0  # finite junk in the failed bin
    summary["fit_val"][2, 1, 1] = 1.0
    real_xf.fit_summary = summary
    viewer = SimpleNamespace(
        _guard_no_data=lambda name: True,
        get_selected_rows=lambda: [0],
        vk=SimpleNamespace(get_xf_list=lambda rows: [real_xf]),
        statusbar=MagicMock(),
        tauq_qmin=SimpleNamespace(text=lambda: "0"),
        tauq_qmax=SimpleNamespace(text=lambda: "1"),
    )
    bundle = XpcsViewer._extract_diffusion_for_bayesian(viewer)
    assert 123.0 not in bundle.y
    np.testing.assert_allclose(bundle.x, np.delete(Q_FIT, 2))


def _drawn_wedge_direction_deg(roi):
    """Central direction of the drawn wedge in image coordinates (column, row)."""
    from qtpy import QtCore

    apex = roi.mapToParent(QtCore.QPointF(0.0, roi.size().y() / 2))
    tip = roi.mapToParent(QtCore.QPointF(roi.size().x(), roi.size().y() / 2))
    return math.degrees(math.atan2(tip.y() - apex.y(), tip.x() - apex.x()))


@pytest.mark.parametrize("angle", [30.0, 120.0, 210.0, 300.0])
def test_r3_20_exported_wedge_is_the_wedge_the_pie_roi_draws(real_xf, qapp, angle):
    """Pie angles use the ROI's own frame (PieROI.get_parameter), not raw QMap phi."""
    from xpcsviewer.plothandler.pyqtgraph_handler import PieROI

    roi = PieROI((BC, BC), 100.0)
    roi.setAngle(angle)
    centre = _drawn_wedge_direction_deg(roi)
    assert (centre - angle + 180) % 360 - 180 == pytest.approx(0.0, abs=1e-6)
    param = roi.get_parameter()
    param["dist"] = 1e3  # no q cut-off
    # R4-08: with the drawn polygon the export uses the drawn region itself. This
    # test pins the angle frame of the legacy descriptor (wedge about the beam
    # centre), so it exercises that path; the polygon path is in test_review_r4.
    for key in ("polygon", "origin", "rotation"):
        param.pop(key)
    # pixels inside the drawn wedge: direction (column, row) from the beam centre
    # within the half angle of the central direction
    yy, xx = np.mgrid[0:H, 0:W]
    theta = np.degrees(np.arctan2(yy - BC, xx - BC))
    off = np.abs((theta - centre + 180) % 360 - 180)
    assert np.min(np.abs(off - roi.half_angle)) > 1e-6  # no pixel on the edge
    inside = off <= roi.half_angle
    assert inside.any() and not inside.all()
    sq, img = _static_map(), _intensity()
    expected = np.array(
        [
            img[inside & (sq == k)].mean() if (inside & (sq == k)).any() else np.nan
            for k in (1, 2, 3)
        ]
    )
    _x, y = real_xf.get_roi_data(param)
    np.testing.assert_allclose(y, expected, rtol=1e-6)


def test_r3_20_static_bins_with_azimuth_use_the_row_major_q_by_phi_layout(tmp_path):
    """sqmap bin k -> q bin static_index_mapping[k-1] // len(splist) (reshape_phi_analysis)."""
    path = tmp_path / "az_0001.h5"
    _write_nexus(path, azimuthal=True)
    xf = XpcsFile(str(path))
    # same layout the SAXS 1D reader uses: row-major (len(sqlist), len(splist)),
    # empty cells NaN
    flat = xf.qmap.reshape_phi_analysis(
        np.array([10.0, 20.0, 30.0, 40.0]), label="x", mode="saxs_1d"
    )["Iq"]
    # rows: [mean over phi, phi0 line, phi1 line]; columns: q bins
    np.testing.assert_allclose(flat[1], [np.nan, 20.0, 30.0], equal_nan=True)
    np.testing.assert_allclose(flat[2], [10.0, np.nan, 40.0], equal_nan=True)
    # the ROI profile over the whole detector averages per q band, whatever the
    # azimuth split of the band
    roi = {"sl_type": "Pie", "angle_range": (0.0, 360.0), "dist": 1e3}
    x, y = xf.get_roi_data(roi)
    r, _phi = _pixel_polar()
    img = _intensity()
    expected = [
        img[(r < R_EDGES[0])].mean(),
        img[(r >= R_EDGES[0]) & (r < R_EDGES[1])].mean(),
        img[r >= R_EDGES[1]].mean(),
    ]
    np.testing.assert_allclose(x, [q_at(c) for c in R_CENTRES], rtol=1e-6)
    np.testing.assert_allclose(y, expected, rtol=1e-6)


def test_r3_20_inconsistent_static_layout_is_an_error_not_a_truncation(real_xf):
    real_xf.qmap.static_index_mapping = np.array([0, 1, 9])  # q bin 9 of 3
    with pytest.raises(ValueError, match="static_index_mapping"):
        real_xf.get_roi_data({"sl_type": "Pie", "angle_range": (0, 360), "dist": 9})
