"""Round 5, second wave: R5-09, R5-08 and the follow-ups handed over.

R5-09  ONE function turns (file, display options) into the displayed image, its
       orientation and the beam centre (``module.saxs2d.render_frame``); the
       synchronous plot and the background worker both go through it and
       through the one ``saxs2d.draw``.
beam   the beam-centre marker (and the default ROI origin) is placed at the
       centre of the pixel whose INDEX is (bcx, bcy), through the image item's
       own mapping, rotated or not.
R5-08  a mask / partition import is applied per target, only where the detector
       shape matches; the others are skipped and named.
4a-4f  SAXS-1D request carries the background revision; Reload redraws and
       clears a vanished background; the G2 regroup worker guards its own write;
       the legacy Pie path refuses a static axis that is not inverse angstrom.

Real h5py-written NeXus files go through the real ``XpcsFile`` / ``QMap`` /
``ViewerKernel``; the view, image item and ROIs are the real offscreen
pyqtgraph objects and the viewer is the real offscreen ``XpcsViewer``.
"""

from __future__ import annotations

import ast
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest
from qtpy import QtCore, QtWidgets

from tests.unit import test_review_r3_maps_roi as r3
from tests.unit.test_review_r3_fit_status import _write_file
from tests.unit.test_review_r4_viewer_gui import (  # noqa: F401  (fixtures)
    _idle,
    _select,
    _tab,
    _write,
    viewer_env,
)
from xpcsviewer.fileIO.hdf_reader import clear_connection_pool
from xpcsviewer.gui.g2_regroup_dialog import G2RegroupExportWorker
from xpcsviewer.module import saxs2d
from xpcsviewer.plothandler.pyqtgraph_handler import ImageViewDev
from xpcsviewer.threading.plot_workers import SaxsPlotWorker
from xpcsviewer.utils.export_guard import ExportCollisionError
from xpcsviewer.xpcs_file import XpcsFile

pytestmark = pytest.mark.unit

PKG = Path(__file__).resolve().parents[2] / "xpcsviewer"
K = r3.K


@pytest.fixture(autouse=True)
def _fresh_pool():
    clear_connection_pool()
    yield
    clear_connection_pool()


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _real_file(path, bcx, bcy, scale=0):
    """Real 2x3 (non-square) NeXus file; pixel value = 1 + 3 * row + col + scale."""
    _write_file(path, np.logspace(-3, 0, 14))
    with h5py.File(path, "r+") as f:
        f["/xpcs/temporal_mean/scattering_2d"][:] = (
            np.arange(1, 7).reshape(1, 2, 3) + scale
        )
        f["/xpcs/qmap/beam_center_x"][()] = bcx
        f["/xpcs/qmap/beam_center_y"][()] = bcy
    return XpcsFile(str(path))


def _displayed(hdl):
    """The picture on screen: stored pixel values placed where the real image
    item draws them, so it is independent of how orientation is implemented."""
    img = np.asarray(hdl.image)
    cells = {}
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            x, y = hdl.pixel_centre(c, r)
            cells[int(np.floor(x)), int(np.floor(y))] = img[r, c]
    x0, y0 = min(k[0] for k in cells), min(k[1] for k in cells)
    out = np.full(
        (max(k[1] for k in cells) - y0 + 1, max(k[0] for k in cells) - x0 + 1), np.nan
    )
    for (x, y), v in cells.items():
        out[y - y0, x - x0] = v
    return out


def _marker(hdl):
    x, y = hdl.roi_record["Center"].getData()
    return float(x[0]), float(y[0])


def _item_transform(hdl):
    t = hdl.getImageItem().transform()
    return (t.m11(), t.m12(), t.m21(), t.m22(), t.dx(), t.dy())


def _state(hdl):
    return {
        "picture": _displayed(hdl),
        "transform": _item_transform(hdl),
        "marker": _marker(hdl),
        "levels": tuple(np.round(hdl.getImageItem().levels, 9)),
    }


def _same_state(a, b):
    np.testing.assert_array_equal(a["picture"], b["picture"])
    assert a["transform"] == b["transform"]
    np.testing.assert_allclose(a["marker"], b["marker"])
    assert a["levels"] == b["levels"]


# --------------------------------------------------------------------------
# R5-09: one rendering function
# --------------------------------------------------------------------------
def _function(tree, name, owner=None):
    for node in ast.walk(tree):
        if owner is not None and isinstance(node, ast.ClassDef) and node.name == owner:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
        if owner is None and isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{owner}.{name} not found")


def _called(node):
    names = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            names.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    return names


def _parse(rel):
    return ast.parse((PKG / rel).read_text())


def test_r5_09_every_saxs2d_entry_point_goes_through_the_one_function():
    """Enumerates the SAXS-2D drawing entry points and what each one calls."""
    # synchronous: ViewerKernel.plot_saxs_2d -> saxs2d.plot -> render_frame + draw
    kernel = _function(_parse("viewer_kernel.py"), "plot_saxs_2d", "ViewerKernel")
    assert "plot" in _called(kernel)
    plot = _function(_parse("module/saxs2d.py"), "plot")
    assert {"render_frame", "draw"} <= _called(plot)
    # background: the worker produces the frame ...
    worker = _function(_parse("threading/plot_workers.py"), "do_work", "SaxsPlotWorker")
    assert "render_frame" in _called(worker)
    # ... and the viewer applies exactly it
    apply = _function(_parse("xpcs_viewer.py"), "apply_saxs_2d_result", "XpcsViewer")
    assert "draw" in _called(apply)
    # none of the three recombines image, orientation or centre itself
    for fn in (worker, apply, plot):
        assert not {"setImage", "rot90", "log10", "add_roi", "setLevels"} & _called(fn)
        assert not {"bcx", "bcy"} & {
            n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)
        }


def test_r5_09_nothing_else_draws_into_the_saxs2d_view():
    """Whole-package scan: ``pg_saxs`` image / marker calls exist only in draw()."""
    offenders = []
    for path in PKG.rglob("*.py"):
        if path.name.endswith("_ui.py") or path.name == "icons_rc.py":
            continue
        tree = ast.parse(path.read_text())
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            for call in [c for c in ast.walk(fn) if isinstance(c, ast.Call)]:
                f = call.func
                if not (
                    isinstance(f, ast.Attribute)
                    and f.attr in ("setImage", "add_roi", "setLevels")
                ):
                    continue
                receiver = ast.unparse(f.value)
                if "pg_saxs" in receiver or (
                    f.attr == "add_roi"
                    and any(
                        isinstance(k.value, ast.Constant) and k.value.value == "Center"
                        for k in call.keywords
                    )
                ):
                    offenders.append(f"{path.relative_to(PKG)}:{fn.name}")
    assert sorted(set(offenders)) == ["module/saxs2d.py:draw"], offenders


@pytest.mark.parametrize("rotate", [False, True])
def test_r5_09_render_frame_orientation_is_np_rot90(tmp_path, qapp, rotate):
    xf = _real_file(tmp_path / "a_0001.h5", 1.0, 0.0)
    frame = saxs2d.render_frame(xf, plot_type="linear", rotate=rotate)
    assert frame.image.shape == (2, 3)  # stored orientation, never rotated in memory
    hdl = ImageViewDev()
    saxs2d.draw(hdl, frame, cmap="viridis", autolevel=True)
    shown = _displayed(hdl)
    expected = np.rot90(xf.saxs_2d) if rotate else np.asarray(xf.saxs_2d)
    np.testing.assert_array_equal(shown, expected)
    assert frame.display_shape == expected.shape
    # one pixel decides it for every orientation: the log-scale choice too
    log = saxs2d.render_frame(xf, plot_type="log", rotate=rotate)
    np.testing.assert_allclose(log.image, np.asarray(xf.saxs_2d_log))


BEAM_CENTRES = [(1.0, 0.0), (2.0, 1.0), (0.25, 0.75), (1.6, 0.4), (0.0, 1.0)]


@pytest.mark.parametrize("rotate", [False, True])
@pytest.mark.parametrize("bcx,bcy", BEAM_CENTRES)
def test_beam_centre_marker_is_the_centre_of_the_pixel_it_indexes(
    tmp_path, qapp, bcx, bcy, rotate
):
    """Calibrated (bcx, bcy): the centre of pixel i is at i. The marker sits at
    the centre of that pixel as drawn (display frame), integer or not."""
    xf = _real_file(tmp_path / "b_0001.h5", bcx, bcy)
    hdl = ImageViewDev()
    saxs2d.plot(xf, hdl, plot_type="linear", cmap="viridis", rotate=rotate)
    width = xf.saxs_2d.shape[1]
    # independent oracle (np.rot90: stored (row, col) is shown at (W-1-col, row)):
    # item point (bcx + .5, bcy + .5) -> view (x, y)
    want = (bcy + 0.5, width - (bcx + 0.5)) if rotate else (bcx + 0.5, bcy + 0.5)
    np.testing.assert_allclose(_marker(hdl), want)
    if float(bcx).is_integer() and float(bcy).is_integer():
        # on the drawn pixel whose stored value is that of pixel (bcx, bcy)
        shown = _displayed(hdl)
        mx, my = _marker(hdl)
        assert (
            shown[int(np.floor(my)), int(np.floor(mx))]
            == xf.saxs_2d[int(bcy), int(bcx)]
        )
        assert mx % 1 == pytest.approx(0.5) and my % 1 == pytest.approx(0.5)


def test_pixel_index_at_inverts_the_display(tmp_path, qapp):
    """Mouse read-out: the pixel under a display position, rotated or not."""
    xf = _real_file(tmp_path / "c_0001.h5", 1.0, 0.0)
    for rotate in (False, True):
        hdl = ImageViewDev()
        saxs2d.plot(xf, hdl, plot_type="linear", cmap="viridis", rotate=rotate)
        for r in range(2):
            for c in range(3):
                x, y = hdl.pixel_centre(c, r)
                assert hdl.pixel_index_at(x, y) == (c, r)
                assert hdl.pixel_index_at(x + 0.4, y - 0.4) == (c, r)


def _two_files(env, rotate_first=False):
    """Replace a.h5 / b.h5 by non-square files with different beam centres."""
    for idx, name in enumerate(("a.h5", "b.h5")):
        tmp = env.root / f"replacement_{name}"
        _write(tmp, 1)
        with h5py.File(tmp, "r+") as f:
            f["/xpcs/temporal_mean/scattering_2d"][:] = (
                np.arange(1, 7).reshape(1, 2, 3) + 10 * idx
            )
            f["/xpcs/qmap/beam_center_x"][()] = idx + 0.25
        os.replace(tmp, env.data / name)
    env.view.reload_source()


def _sync_draw(view):
    view.plot_saxs_2d()


def _async_draw(view):
    view._invalidate_plots()  # nothing counts as "already drawn"
    view.update_plot_async("saxs_2d")
    _idle(view)


@pytest.mark.parametrize("rotate", [False, True])
def test_r5_09_sync_and_async_paths_draw_the_same_thing(viewer_env, rotate):  # noqa: F811
    """Real viewer, real non-square files: either path, any file, same frame."""
    view = viewer_env.view
    _two_files(viewer_env)
    view.tabWidget.setCurrentIndex(_tab(viewer_env.xv, "saxs_2d"))
    view.saxs2d_rotate.blockSignals(True)  # no automatic plot: each path is explicit
    view.saxs2d_rotate.setChecked(rotate)
    view.saxs2d_rotate.blockSignals(False)
    for row, bcx in ((0, 0.25), (1, 1.25)):
        view.list_view_target.selectionModel().select(
            view.vk.target.index(row, 0),
            QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect,
        )
        _sync_draw(view)
        sync = _state(view.pg_saxs)
        _async_draw(view)
        asyn = _state(view.pg_saxs)
        _same_state(sync, asyn)
        # the image, the orientation and the centre are all of THIS file
        shown = sync["picture"]
        stored = np.log10(np.arange(1, 7).reshape(2, 3) + 10 * row + 1)
        if view.cb_saxs2D_type.currentText() == "linear":
            stored = np.arange(1, 7).reshape(2, 3) + 10 * row
        np.testing.assert_allclose(shown, np.rot90(stored) if rotate else stored)
        np.testing.assert_allclose(
            sync["marker"], view.pg_saxs.pixel_centre(bcx, 1.0), atol=1e-12
        )
        assert view.saxs2d_rotate.isChecked() == rotate


def test_r5_09_reviewer_scenario_rotate_then_switch_target(viewer_env):  # noqa: F811
    """Rotate checked, select the second target: async draw keeps ITS centre and
    the orientation; clicking Plot afterwards changes nothing."""
    view = viewer_env.view
    _two_files(viewer_env)
    view.tabWidget.setCurrentIndex(_tab(viewer_env.xv, "saxs_2d"))
    _select(view, 0)
    view.saxs2d_rotate.click()
    _idle(view)
    first = _state(view.pg_saxs)
    assert first["picture"].shape == (3, 2)
    _select(view, 1)  # automatic (async) request
    switched = _state(view.pg_saxs)
    assert switched["picture"].shape == (3, 2)
    np.testing.assert_allclose(
        switched["marker"], view.pg_saxs.pixel_centre(1.25, 1.0), atol=1e-12
    )
    assert switched["marker"] != first["marker"]
    view.pushButton_plot_saxs2d.click()  # explicit (sync) request
    _idle(view)
    _same_state(switched, _state(view.pg_saxs))
    assert view.saxs2d_rotate.isChecked()


def test_r5_09_worker_result_is_one_object_of_one_file(viewer_env):  # noqa: F811
    """The worker's result carries image, orientation and centre together."""
    view = viewer_env.view
    _two_files(viewer_env)
    kwargs = {
        "rows": [1],
        "plot_type": "linear",
        "cmap": "viridis",
        "rotate": True,
        "autolevel": True,
        "vmin": 0,
        "vmax": 1,
    }
    worker = SaxsPlotWorker(view.vk, view.pg_saxs, kwargs)
    result = worker.do_work()
    frame = result["frame"]
    np.testing.assert_array_equal(frame.image, np.arange(1, 7).reshape(2, 3) + 10)
    assert frame.rotate and frame.center == (1.25, 1.0)
    assert result["display"]["cmap"] == "viridis"
    assert frame.label == view.vk.get_xf_list([1])[0].label


# ---- the default ROI origin ------------------------------------------------
@pytest.mark.parametrize("rotate", [False, True])
@pytest.mark.parametrize("sl_type", ["Q-Wedge", "Phi-Ring"])
def test_default_roi_origin_is_the_beam_centre_pixel_of_the_shown_file(
    viewer_env,  # noqa: F811
    sl_type,
    rotate,
):
    view = viewer_env.view
    _two_files(viewer_env)
    hdl = ImageViewDev()
    xf = view.vk.get_xf_list([1])[0]
    saxs2d.plot(xf, hdl, plot_type="linear", cmap="viridis", rotate=rotate)
    assert view.vk.add_roi(hdl, rows=[1], sl_type=sl_type)  # file b: centre 1.25
    want = np.array(hdl.pixel_centre(1.25, 1.0))
    other = np.array(hdl.pixel_centre(0.25, 1.0))  # file a's: must not be used
    assert not np.allclose(want, other)
    roi = hdl.roi_record["RingA" if sl_type == "Phi-Ring" else "roi_000000"]
    centre = np.array(roi.pos()) + np.array(roi.size()) / 2.0
    np.testing.assert_allclose(centre, want, atol=1e-9)


def test_ring_profile_is_independent_of_the_display_rotation(tmp_path, qapp):
    """Ring phi is a detector-frame azimuth: rotating the display must not turn it."""
    path = tmp_path / "ring_0001.h5"
    r3._write_nexus(path)
    xf = XpcsFile(str(path))
    profiles = []
    for rotate in (False, True):
        hdl = ImageViewDev()
        saxs2d.plot(xf, hdl, plot_type="linear", cmap="viridis", rotate=rotate)
        cx, cy = hdl.pixel_centre(r3.BC, r3.BC)
        hdl.add_roi(cen=(cx, cy), radius=3.2, sl_type="Phi-Ring", label="RingA")
        hdl.add_roi(cen=(cx, cy), radius=1.1, sl_type="Phi-Ring", label="RingB")
        (param,) = hdl.get_roi_list()
        profiles.append(xf.get_roi_data(param, phi_num=12))
    np.testing.assert_allclose(profiles[0][0], profiles[1][0])
    np.testing.assert_allclose(profiles[0][1], profiles[1][1], equal_nan=True)


# --------------------------------------------------------------------------
# R5-08: per-target shape check on import
# --------------------------------------------------------------------------
@pytest.fixture
def shape_env(qtbot, monkeypatch, tmp_path):
    """Real viewer over two targets: a.h5 (8x8 detector) and b.h5 (8x6)."""
    from xpcsviewer.gui.state.recent_paths import RecentPathsManager
    from xpcsviewer.gui.state.session_manager import (
        AnalysisParameters,
        SessionManager,
    )

    xv = sys.modules["xpcsviewer.xpcs_viewer"]
    data = tmp_path / "data"
    data.mkdir()
    r3._write_nexus(data / "a.h5")
    r3._write_nexus(data / "b.h5")
    with h5py.File(data / "b.h5", "a") as f:
        for name in ("saxs_2d", "mask", "dqmap", "sqmap"):
            original = f[K[name]][()]
            del f[K[name]]
            f[K[name]] = original[..., :6]
    session = SimpleNamespace(
        data_path=str(data),
        target_files=[],
        active_tab=-1,
        window_geometry=SimpleNamespace(maximized=True),
        analysis_params=AnalysisParameters(),
    )
    monkeypatch.setattr(SessionManager, "load_session", lambda self: session)
    monkeypatch.setattr(SessionManager, "get_warnings", lambda self: [])
    monkeypatch.setattr(xv.XpcsViewer, "closeEvent", lambda self, ev: ev.accept())
    monkeypatch.setattr(xv.XpcsViewer, "_maybe_prompt_start_path", lambda self: None)
    monkeypatch.setattr(RecentPathsManager, "add_path", lambda self, path: None)
    dialog = MagicMock()  # no modal dialog may open
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", dialog)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", dialog)
    view = xv.XpcsViewer(path=str(data))
    qtbot.addWidget(view)
    view.vk.add_target(["a.h5", "b.h5"])
    view.update_box(view.vk.target, mode="target")
    files = view.vk.get_xf_list()
    assert [x.saxs_2d.shape for x in files] == [(8, 8), (8, 6)]
    return SimpleNamespace(
        view=view, files=files, root=tmp_path, dialog=dialog, xv=xv, data=data
    )


def test_r5_08_mask_import_applies_only_where_the_detector_matches(
    shape_env, caplog, tmp_path
):
    view, (xa, xb) = shape_env.view, shape_env.files
    before_b = xb.mask
    mask = np.ones((8, 8), dtype=bool)
    mask[0, 0] = False
    with caplog.at_level(logging.WARNING):
        view.import_mask(mask)

    assert xa.mask is mask
    assert xb.mask is before_b and np.shape(xb.mask) == (8, 6)  # never wrong-shaped
    shown = view.statusbar.currentMessage()
    assert "b.h5" in shown and "a.h5" not in shown.split("Skipped")[1]
    assert any("b.h5" in r.getMessage() for r in caplog.records)
    shape_env.dialog.assert_not_called()

    # the reviewer's failure: the export of ALL targets with a Pie profile
    out = tmp_path / "out"
    out.mkdir()
    image_view = ImageViewDev()
    assert view.vk.add_roi(image_view, sl_type="Pie")
    view.vk.export_saxs_1d(image_view, str(out))
    names = sorted(p.name for p in out.iterdir())
    assert "a_Pie_000.txt" in names and "b_Pie_000.txt" in names
    assert "a_saxs1d.txt" in names and "b_saxs1d.txt" in names


def test_r5_08_mask_matching_no_file_is_refused_and_changes_nothing(shape_env):
    view, files = shape_env.view, shape_env.files
    before = [x.mask for x in files]
    view.import_mask(np.ones((4, 4), dtype=bool))
    assert all(x.mask is b for x, b in zip(files, before, strict=True))
    shape_env.dialog.assert_called_once()  # the one existing warning dialog


def test_r5_08_partition_import_is_checked_per_target_too(shape_env):
    view, (xa, xb) = shape_env.view, shape_env.files
    own_b = np.array(xb.get_display_qmap("dqmap"))
    dq = np.ones((8, 8), dtype=np.int32)
    sq = np.full((8, 8), 2, dtype=np.int32)
    view.import_partition({"dynamic_roi_map": dq, "static_roi_map": sq})
    assert xa.display_dqmap is dq and xa.display_sqmap is sq
    assert xb.display_dqmap is None and xb.display_sqmap is None
    np.testing.assert_array_equal(xb.get_display_qmap("dqmap"), own_b)
    assert "b.h5" in view.statusbar.currentMessage()
    # one wrong-shaped map is enough to keep the whole partition off a file
    view.import_partition(
        {"dynamic_roi_map": np.ones((8, 6), np.int32), "static_roi_map": sq}
    )
    assert xb.display_dqmap is None and xa.display_dqmap is dq


def _pushes_onto_every_target(tree):
    """``(function, attribute)`` for each ``for f in ...: f.attr = <not f>`` (or
    ``setattr(f, name, <not f>)``): one loop-invariant value put on every file."""
    found = set()
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for loop in [n for n in ast.walk(fn) if isinstance(n, ast.For)]:
            if not isinstance(loop.target, ast.Name):
                continue
            var = loop.target.id
            # loops over loaded files (xf_list, accepted, ...): by name of the
            # variable or of what is iterated
            if "xf" not in var and "xf" not in ast.unparse(loop.iter):
                continue
            for node in [n for n in ast.walk(loop) if n is not loop]:
                if isinstance(node, ast.Assign):
                    value, names = node.value, node.targets
                    attrs = [
                        t.attr
                        for t in names
                        if isinstance(t, ast.Attribute)
                        and isinstance(t.value, ast.Name)
                        and t.value.id == var
                    ]
                elif (
                    isinstance(node, ast.Call)
                    and getattr(node.func, "id", "") == "setattr"
                    and len(node.args) == 3
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == var
                ):
                    value, attrs = node.args[2], ["<setattr>"]
                else:
                    continue
                if attrs and var not in {
                    n.id for n in ast.walk(value) if isinstance(n, ast.Name)
                }:
                    found.update((fn.name, a) for a in attrs)
    return found


def test_r5_08_every_push_of_one_array_onto_targets_is_shape_checked():
    """Whole-package scan: a value put on every loaded file in a loop (not
    derived from that file) may only happen in a function that first splits the
    targets by their own detector shape."""
    pushes = {}
    for path in PKG.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for fn_name, attr in _pushes_onto_every_target(tree):
            fn = _function(tree, fn_name) if fn_name != "<lambda>" else None
            pushes[f"{path.relative_to(PKG)}:{fn_name}.{attr}"] = (
                "_split_by_detector_shape" in _called(fn)
            )
    assert pushes == {
        "xpcs_viewer.py:import_mask.mask": True,
        "xpcs_viewer.py:import_partition.display_dqmap": True,
        "xpcs_viewer.py:import_partition.display_sqmap": True,
    }, pushes


# --------------------------------------------------------------------------
# 4a / 4b: the SAXS background
# --------------------------------------------------------------------------
def _saxs1d_lines(view):
    return [
        (p.name(), p.getData()[1].tolist())
        for p in view.mp_saxs.listDataItems()
        if p.name()
    ]


def _write_background(path, value):
    _write_file(path, np.logspace(-3, 0, 14), saxs=(value, value, value))


def test_saxs_background_replaced_on_disk_redraws_after_reload_without_plot(
    viewer_env,
    monkeypatch,  # noqa: F811
):
    view = viewer_env.view
    view.tabWidget.setCurrentIndex(_tab(viewer_env.xv, "saxs_1d"))
    view.cb_saxs_type.setCurrentIndex(0)
    view.cb_sub_bkg.setChecked(True)
    view.bkg_weight.setValue(1)
    _select(view, 0)
    bkg = viewer_env.data / "background.h5"
    _write_background(bkg, 1)
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getOpenFileName", lambda *a, **k: (str(bkg), "HDF5")
    )
    view.btn_select_bkgfile.click()
    view.pushButton_plot_saxs1d.click()
    _idle(view)
    first = _saxs1d_lines(view)
    assert first and first[0][1] == [9.0, 4.0, 1.0]
    request = view._plot_request({"rows": [0]})

    replacement = viewer_env.root / "replacement_background.h5"
    _write_background(replacement, 2)
    os.replace(replacement, bkg)
    # the request differs as soon as the background differs on disk (4a)
    assert view._plot_request({"rows": [0]}) != request
    view.reload_source()  # NO explicit Plot (4b)
    _idle(view)
    assert _saxs1d_lines(view)[0][1] == [8.0, 3.0, 0.0]

    # a vanished background is dropped, and the GUI stops showing its path
    assert view.le_bkg_fname.text()
    bkg.unlink()
    view.reload_source()
    _idle(view)
    assert view.vk.meta["saxs1d_bkg_path"] is None
    assert view.le_bkg_fname.text() == ""
    assert _saxs1d_lines(view)[0][1] == [10.0, 5.0, 2.0]  # no subtraction left


def test_plot_request_carries_the_background_revision(viewer_env):  # noqa: F811
    view = viewer_env.view
    assert view._plot_request({"rows": [0]})["saxs1d_background"] is None  # none chosen


# --------------------------------------------------------------------------
# 4d: the G2 regroup worker guards its own write
# --------------------------------------------------------------------------
def _regroup_result(source, qmap):
    from xpcsviewer.fileIO.g2_regroup import G2RegroupResult

    return G2RegroupResult(
        source_path=source,
        qmap_path=qmap,
        delay_unit=None,
        delay_list=np.arange(3.0),
        delay_seconds=np.arange(3.0),
        dynamic_bin_ids=np.array([1]),
        normalized_g2=np.ones((3, 1)),
        normalized_g2_err=np.ones((3, 1)),
        static_roi_map=np.ones((2, 2), dtype=np.int32),
        dynamic_roi_map=np.ones((2, 2), dtype=np.int32),
    )


@pytest.mark.parametrize("alias_of", ["source", "qmap"])
def test_g2_regroup_worker_refuses_to_write_through_an_alias_of_its_input(
    tmp_path, qapp, alias_of
):
    source, qmap = tmp_path / "source.h5", tmp_path / "qmap.h5"
    _write_file(source, np.logspace(-3, 0, 14))
    _write_file(qmap, np.logspace(-3, 0, 14))
    target = {"source": source, "qmap": qmap}[alias_of]
    alias = tmp_path / "alias.h5"
    alias.symlink_to(target)
    original = target.read_bytes()
    worker = G2RegroupExportWorker(_regroup_result(source, qmap), alias)
    with pytest.raises(ExportCollisionError):
        worker.do_work()  # called directly, without the Viewer's dispatch claim
    assert target.read_bytes() == original

    fresh = tmp_path / "regrouped.h5"
    assert (
        G2RegroupExportWorker(_regroup_result(source, qmap), fresh).do_work() == fresh
    )
    with h5py.File(fresh, "r") as f:
        assert f["/xpcs/multitau/normalized_g2"].shape == (3, 1)


# --------------------------------------------------------------------------
# 4f: the legacy Pie path needs a static axis in inverse angstrom
# --------------------------------------------------------------------------
def test_legacy_pie_refuses_a_static_axis_that_is_not_inverse_angstrom(tmp_path):
    ok, other = tmp_path / "ok_0001.h5", tmp_path / "nm_0001.h5"
    for path in (ok, other):
        r3._write_nexus(path)
    with h5py.File(other, "r+") as f:
        del f[K["map_units"]]
        f[K["map_units"]] = np.array([b"1/nm", b"deg"])
    legacy = {"sl_type": "Pie", "angle_range": (0.0, 360.0), "dist": 50.0}

    xf_ok = XpcsFile(str(ok))
    assert np.isfinite(xf_ok.get_roi_data(legacy)[1]).any()

    xf_nm = XpcsFile(str(other))
    assert xf_nm._static_axis()[1] != xf_ok._static_axis()[1]
    with pytest.raises(ValueError, match="static q axis"):
        xf_nm.get_roi_data(legacy)
    # a Pie ROI that carries its drawn polygon never compares against that cut-off
    drawn = {
        "sl_type": "Pie",
        "polygon": [(0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0)],
        "origin": (0.0, 0.0),
        "rotation": 0.0,
    }
    assert np.isfinite(xf_nm.get_roi_data(drawn)[1]).any()
