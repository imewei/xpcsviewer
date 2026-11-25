"""Round 6, SimpleMask editor: R6-01 (editor side), 02, 03, 04, 06, 07, 18, 19.

Real h5py-written raw files go through the real readers; the real offscreen
SimpleMaskWindow / kernel and real pyqtgraph ROIs supply the geometry. No modal
dialog is ever opened: the file dialog and the message boxes are patched.
"""

from __future__ import annotations

import ast
import inspect
import logging
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import pyqtgraph as pg
import pytest
from qtpy.QtCore import QPointF, QRectF
from qtpy.QtWidgets import QFileDialog, QMessageBox

import xpcsviewer.simplemask as simplemask_pkg
from xpcsviewer.helper.listmodel import ListDataModel
from xpcsviewer.simplemask import SimpleMaskWindow, roi_geometry
from xpcsviewer.simplemask.reader import io_utils
from xpcsviewer.simplemask.reader.beamlines.aps_8idi import METADATA_KEYMAPS
from xpcsviewer.simplemask.reader.formats import HdfDataset
from xpcsviewer.simplemask.simplemask_kernel import SimpleMaskKernel
from xpcsviewer.utils.export_guard import ExportCollisionError

pytestmark = pytest.mark.unit

SHAPE = (64, 80)
BC = (39.0, 29.0)  # (bcx, bcy): pixel-INDEX convention
GEOMETRY = {"det_dist": 5000.0, "pix_dim": 0.075, "energy": 10.0}
WAIT_MS = 20_000


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def write_raw(path, frames):
    """A real APS-layout raw file: ``/entry/data/data`` plus the metadata keys."""
    with h5py.File(path, "w") as f:
        f["/entry/data/data"] = frames
        for key, value in {
            "energy": 10.0,
            "detector_distance": 5.0,
            "x_pixel_size": 75e-6,
            "y_pixel_size": 75e-6,
            "ccdx": 0.0,
            "ccdy": 0.0,
            "ccdx0": 0.0,
            "ccdy0": 0.0,
            "bcx0": 1.5,
            "bcy0": 1.5,
        }.items():
            f[METADATA_KEYMAPS[key]] = value
    return str(path)


def make_window(qtbot, image=None, mask=None, parent_viewer=None, **meta):
    window = SimpleMaskWindow(parent_viewer=parent_viewer)
    qtbot.addWidget(window)
    image = np.full(SHAPE, 10.0) if image is None else image
    metadata = {"bcx": BC[0], "bcy": BC[1], **GEOMETRY, **meta}
    if mask is not None:
        metadata["mask"] = mask
    window.load_from_viewer(image, metadata)
    return window


def open_raw(qtbot, window, path):
    window.load_from_raw_file(path, "APS_8IDI")
    qtbot.waitUntil(lambda: window._raw_source_path == path, timeout=WAIT_MS)


def drawn_rois(window):
    return [
        roi
        for key, roi in window.image_view.roi_items.items()
        if key.startswith("roi_")
    ]


def draw_rect(window, tool_key_or_eraser, pos, size):
    if tool_key_or_eraser == "eraser":
        window.action_eraser.trigger()
    else:
        window.tool_actions[tool_key_or_eraser].trigger()
    (roi,) = drawn_rois(window)
    roi.setPos(*pos)
    roi.setSize(size)
    window.action_apply.trigger()


def masked_pixels(window):
    return {(int(r), int(c)) for r, c in zip(*np.nonzero(~window.kernel.mask))}


def block(r0, c0, n):
    return {(r, c) for r in range(r0, r0 + n) for c in range(c0, c0 + n)}


@pytest.fixture(autouse=True)
def _no_modal_dialogs():
    with (
        patch.object(QMessageBox, "warning") as warn,
        patch.object(QMessageBox, "critical") as crit,
        patch.object(QMessageBox, "question") as question,
    ):
        question.return_value = QMessageBox.StandardButton.No
        yield SimpleNamespace(warning=warn, critical=crit, question=question)


# --------------------------------------------------------------------------- #
# R6-02: unsigned raw counts
# --------------------------------------------------------------------------- #
class TestUnsignedCounts:
    COUNTS = np.array([[32767, 32768, 40000, 65533, 65534, 65535]], dtype=np.uint16)
    EXPECTED = [32767, 32768, 40000, 65533, -2, -1]  # only 2^N-1, 2^N-2 are flags

    def test_decode_keeps_ordinary_counts_and_maps_only_the_top_code(self):
        decoded = io_utils.decode_counts(self.COUNTS)
        np.testing.assert_array_equal(decoded, [self.EXPECTED])
        assert decoded.dtype.kind == "i" and decoded.dtype.itemsize > 2

    @pytest.mark.parametrize(
        ("dtype", "ordinary"),
        [(np.uint8, 200), (np.uint16, 40000), (np.uint32, 3_000_000_000)],
    )
    def test_every_unsigned_width_keeps_high_bit_counts(self, dtype, ordinary):
        top = np.iinfo(dtype).max
        decoded = io_utils.decode_counts(
            np.array([ordinary, top - 2, top - 1, top], dtype=dtype)
        )
        np.testing.assert_array_equal(decoded, [ordinary, top - 2, -2, -1])

    def test_signed_and_float_input_pass_through(self):
        arr = np.array([1, -2, 3], dtype=np.int32)
        assert io_utils.decode_counts(arr) is arr
        flt = np.array([1.5, 2.5])
        assert io_utils.decode_counts(flt) is flt

    def test_frame_average_of_a_real_stack_preserves_counts(self, tmp_path):
        stack = np.stack([self.COUNTS, self.COUNTS - 5000]).astype(np.uint16)
        stack[:, :, 4:] = 65535
        path = tmp_path / "scan.h5"
        write_raw(path, stack)
        mean = io_utils.average_frames_parallel(str(path), num_processes=1)
        np.testing.assert_allclose(mean, [[30267, 30268, 37500, 63033, -1, -1]])
        loader = HdfDataset(str(path))
        np.testing.assert_allclose(
            loader.get_frames_mean(0, 2), [[30267, 30268, 37500, 63033, -1, -1]]
        )

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
    def test_chunk_sum_equals_the_sum_of_decoded_frames(self, dtype):
        top = np.iinfo(dtype).max
        rng = np.random.default_rng(6)
        frames = rng.integers(0, top, size=(5, 3, 4), dtype=dtype, endpoint=True)
        frames[2, 1, 1] = top
        frames[:, 0, 0] = [top, top, 7, top - 1, 0]
        np.testing.assert_array_equal(
            io_utils.sum_counts(frames),
            io_utils.decode_counts(frames).sum(axis=0, dtype=np.float64),
        )

    def test_pool_worker_chunk_sum_preserves_counts(self, tmp_path):
        path = tmp_path / "scan.h5"
        write_raw(path, np.stack([self.COUNTS, self.COUNTS]))
        total = io_utils.process_chunk(str(path), "/entry/data/data", 0, 2)
        np.testing.assert_array_equal(total, 2 * np.array([self.EXPECTED]))

    def test_single_2d_image_preserves_counts(self, tmp_path):
        path = tmp_path / "single.h5"
        with h5py.File(path, "w") as f:
            f["/entry/data/data"] = self.COUNTS
        image = HdfDataset(str(path)).get_scattering()
        assert image.dtype == np.float32
        np.testing.assert_array_equal(image, [self.EXPECTED])

    def test_editor_shows_and_previews_ordinary_counts(self, qtbot, tmp_path):
        counts = np.array([[[32768, 40000, 65533, 65535]]] * 3, dtype=np.uint16)
        path = write_raw(tmp_path / "high-counts.h5", counts)
        window = SimpleMaskWindow()
        qtbot.addWidget(window)
        open_raw(qtbot, window, path)
        np.testing.assert_array_equal(
            window.kernel.detector_image, [[32768, 40000, 65533, -1]]
        )
        window.raw_frame_display.setCurrentIndex(1)  # log10 frame preview
        window.raw_frame_start.setValue(1)
        qtbot.waitUntil(
            lambda: window.raw_frame_range_label.text() == "Frame 1", timeout=WAIT_MS
        )
        np.testing.assert_allclose(
            window.image_view.image,
            [[np.log10(32769), np.log10(40001), np.log10(65534), 0.0]],
            rtol=1e-6,
        )


# --------------------------------------------------------------------------- #
# R6-03: eraser removes from the drawn exclusions only
# --------------------------------------------------------------------------- #
class TestEraser:
    def test_erase_subtracts_from_previous_drawn_exclusion(self, qtbot):
        window = make_window(qtbot)
        draw_rect(window, "rectangle", (10, 10), (8, 8))
        assert masked_pixels(window) == block(10, 10, 8)
        draw_rect(window, "eraser", (10, 10), (4, 4))
        assert masked_pixels(window) == block(10, 10, 8) - block(10, 10, 4)
        assert len(masked_pixels(window)) == 48

    def test_erase_undo_redo_reset(self, qtbot):
        window = make_window(qtbot)
        draw_rect(window, "rectangle", (10, 10), (8, 8))
        draw_rect(window, "eraser", (10, 10), (4, 4))
        after_erase = masked_pixels(window)
        window.action_undo.trigger()
        assert masked_pixels(window) == block(10, 10, 8)
        window.action_redo.trigger()
        assert masked_pixels(window) == after_erase
        window.action_undo.trigger()
        window.action_undo.trigger()
        assert not masked_pixels(window)
        window.action_redo.trigger()
        window.action_redo.trigger()
        window.action_reset.trigger()
        assert not masked_pixels(window)

    def test_eraser_with_nothing_drawn_masks_nothing(self, qtbot):
        window = make_window(qtbot)
        draw_rect(window, "eraser", (10, 10), (4, 4))
        assert not masked_pixels(window)

    def test_other_layers_are_untouched_by_the_eraser(self, qtbot):
        seed = np.ones(SHAPE, dtype=bool)
        seed[11, 11] = False  # the file's own exclusion, under the eraser
        window = make_window(qtbot, mask=seed)
        kernel = window.kernel
        kernel.mask_kernel.evaluate("mask_list", zero_loc=np.array([[12], [12]]))
        kernel.mask_apply("mask_list")  # a non-drawn layer, also under the eraser
        draw_rect(window, "rectangle", (10, 10), (8, 8))
        draw_rect(window, "eraser", (10, 10), (4, 4))
        expected = block(10, 10, 8) - block(10, 10, 4) | {(11, 11), (12, 12)}
        assert masked_pixels(window) == expected

    def test_successive_exclusions_accumulate(self, qtbot):
        window = make_window(qtbot)
        draw_rect(window, "rectangle", (10, 10), (8, 8))
        draw_rect(window, "rectangle", (30, 30), (8, 8))
        assert masked_pixels(window) == block(10, 10, 8) | block(30, 30, 8)
        window.action_undo.trigger()
        assert masked_pixels(window) == block(10, 10, 8)


# --------------------------------------------------------------------------- #
# R6-04: default shapes and the centre marker sit on the calibrated pixel
# --------------------------------------------------------------------------- #
class TestCalibratedCentre:
    def item_centre(self, window, roi, local):
        mapped = roi.mapToItem(window.image_view.imageItem, QPointF(*local))
        return mapped.x(), mapped.y()

    @pytest.mark.parametrize("shift", [(0.0, 0.0), (7.0, 3.0)])
    def test_default_circle_is_centred_on_the_pixel_the_centre_indexes(
        self, qtbot, shift
    ):
        window = make_window(qtbot)
        window.image_view.imageItem.setPos(*shift)  # a real, non-trivial mapping
        window.tool_actions["circle"].trigger()
        (circle,) = drawn_rois(window)
        r = circle.size().x() / 2
        centre = self.item_centre(window, circle, (r, r))
        assert centre == pytest.approx((BC[0] + 0.5, BC[1] + 0.5))
        # the pixels the circle selects do not change when it is re-centred
        before = window.kernel.drawable_masks()[0][1]
        want = roi_geometry.pixel_centre(window.image_view.imageItem, *BC)
        got = circle.mapToParent(QPointF(r, r))
        circle.setPos(circle.pos() + QPointF(want[0] - got.x(), want[1] - got.y()))
        assert np.array_equal(before, window.kernel.drawable_masks()[0][1])

    @pytest.mark.parametrize("tool", ["rectangle", "ellipse"])
    def test_default_rectangle_and_ellipse_origin(self, qtbot, tool):
        window = make_window(qtbot)
        window.tool_actions[tool].trigger()
        (roi,) = drawn_rois(window)
        assert self.item_centre(window, roi, (0, 0)) == pytest.approx(
            (BC[0] + 0.5, BC[1] + 0.5)
        )

    def test_default_polygon_is_centred_on_the_pixel(self, qtbot):
        window = make_window(qtbot)
        window.tool_actions["polygon"].trigger()
        (poly,) = drawn_rois(window)
        pts = np.array(
            [
                (p.x(), p.y())
                for p in (
                    poly.mapToItem(window.image_view.imageItem, h.pos())
                    for h in poly.getHandles()
                )
            ]
        )
        radii = np.hypot(pts[:, 0] - (BC[0] + 0.5), pts[:, 1] - (BC[1] + 0.5))
        assert radii == pytest.approx(60.0)

    def test_centre_marker_is_drawn_on_the_pixel_centre(self, qtbot):
        window = make_window(qtbot)
        window.image_view.imageItem.setPos(5.0, 2.0)
        window.kernel.show_saxs()
        marker = window.image_view.roi_items["center"]
        x, y = marker.getData()
        want = roi_geometry.pixel_centre(window.image_view.imageItem, *BC)
        assert (x[0], y[0]) == pytest.approx(want)
        assert want == pytest.approx((BC[0] + 5.5, BC[1] + 2.5))

    def test_pixel_index_at_is_the_inverse_of_pixel_centre(self, qtbot):
        item = pg.ImageItem()
        item.setPos(3.0, -2.0)
        for col, row in [(0, 0), (5, 9), (79, 63)]:
            x, y = roi_geometry.pixel_centre(item, col, row)
            assert roi_geometry.pixel_index_at(item, x, y) == (col, row)
        # just left of pixel 0 is column -1, not 0 (no truncation toward zero)
        x0, _ = roi_geometry.pixel_centre(item, 0, 0)
        assert roi_geometry.pixel_index_at(item, x0 - 0.9, -2.0 + 0.5)[0] == -1

    def test_cursor_readout_uses_the_pixel_under_the_cursor(self):
        messages = []
        infobar = SimpleNamespace(
            clearMessage=lambda: messages.clear(), showMessage=messages.append
        )
        kernel = SimpleMaskKernel(infobar=infobar)
        kernel.read_data(np.full(SHAPE, 10.0), {"bcx": BC[0], "bcy": BC[1], **GEOMETRY})
        kernel.hdl = SimpleNamespace(
            scene=SimpleNamespace(itemsBoundingRect=lambda: QRectF(-50, -50, 500, 500)),
            getView=lambda: SimpleNamespace(mapSceneToView=lambda p: p),
            imageItem=pg.ImageItem(),
        )
        kernel.show_location(QPointF(-0.4, 3.2))  # outside: left of pixel 0
        assert messages == []
        kernel.show_location(QPointF(39.9, 29.1))
        assert messages[0].startswith("x=39, y=29")

    def test_rois_select_pixels_by_their_centres(self, qtbot):
        window = make_window(qtbot)
        window.tool_actions["rectangle"].trigger()
        (roi,) = drawn_rois(window)
        roi.setPos(10.4, 10.4)  # pixel 10's centre (10.5) is inside, 18's is not
        roi.setSize((8, 8))
        (mode_filled,) = window.kernel.drawable_masks()
        rows, cols = np.nonzero(mode_filled[1])
        assert (rows.min(), rows.max(), cols.min(), cols.max()) == (10, 17, 10, 17)


# --------------------------------------------------------------------------- #
# R6-06: the Line tool creates an editable line that Apply rasterises
# --------------------------------------------------------------------------- #
class TestLineTool:
    def test_line_tool_creates_a_movable_line_with_two_distinct_ends(self, qtbot):
        window = make_window(qtbot)
        window.tool_actions["line"].trigger()
        (line,) = drawn_rois(window)
        assert line.translatable and len(line.getHandles()) >= 1
        item = window.image_view.imageItem
        w, h = line.size()
        first = line.mapToItem(item, QPointF(0, h / 2))
        second = line.mapToItem(item, QPointF(w, h / 2))
        assert (first.x(), first.y()) == pytest.approx((BC[0] + 0.5, BC[1] + 0.5))
        assert np.hypot(second.x() - first.x(), second.y() - first.y()) > 5

    def test_apply_masks_exactly_the_strip(self, qtbot):
        window = make_window(qtbot)
        window.tool_actions["line"].trigger()
        (line,) = drawn_rois(window)
        item = window.image_view.imageItem
        w, h = line.size()
        a = line.mapToItem(item, QPointF(0, h / 2))
        b = line.mapToItem(item, QPointF(w, h / 2))
        a, b = np.array([a.x(), a.y()]) - 0.5, np.array([b.x(), b.y()]) - 0.5
        axis = (b - a) / np.linalg.norm(b - a)
        window.action_apply.trigger()
        got = masked_pixels(window)
        assert got

        def inside(col, row, grow):
            d = np.array([col, row], float) - a
            t, perp = d @ axis, d @ np.array([-axis[1], axis[0]])
            return (
                -grow <= t <= np.linalg.norm(b - a) + grow and abs(perp) <= h / 2 + grow
            )

        every = [(r, c) for r in range(SHAPE[0]) for c in range(SHAPE[1])]
        sure = {(r, c) for r, c in every if inside(c, r, -1e-6)}
        possible = {(r, c) for r, c in every if inside(c, r, 1e-6)}
        assert sure <= got <= possible


# --------------------------------------------------------------------------- #
# R6-07: a non-finite pixel does not break beam-centre refinement
# --------------------------------------------------------------------------- #
def ring_image(center=(39.0, 29.0)):
    yy, xx = np.indices(SHAPE)
    radius = np.hypot(xx - center[0], yy - center[1])
    return 20 + 1000 * np.exp(-(((radius - 19) / 1.5) ** 2))


class TestBeamCentreFinite:
    @pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
    def test_one_nonfinite_pixel_still_refines(self, bad):
        image = ring_image()
        image[3, 60] = bad
        kernel = SimpleMaskKernel()
        kernel.read_data(image, {"bcx": 40.0, "bcy": 28.0, **GEOMETRY})
        cx, cy, _ = kernel.find_beam_center()
        assert (cx, cy) == pytest.approx(BC, abs=0.1)

    def test_nonfinite_pixel_is_not_a_ring_candidate_for_a_given_threshold(self):
        from xpcsviewer.simplemask.calibration import compute_center_from_ring

        image = ring_image()
        image[3, 60] = np.inf
        cx, cy, _ = compute_center_from_ring(
            image, intensity_threshold=300.0, initial_center=(40.0, 28.0)
        )
        assert (cx, cy) == pytest.approx(BC, abs=0.1)

    def test_all_nonfinite_image_is_a_clear_error(self):
        from xpcsviewer.simplemask.calibration import compute_center_from_ring

        with pytest.raises(ValueError, match="Insufficient ring pixels"):
            compute_center_from_ring(np.full(SHAPE, np.inf))


# --------------------------------------------------------------------------- #
# R6-18 / R6-19: Save Mask never writes over an input; failures are reported
# --------------------------------------------------------------------------- #
@pytest.fixture
def raw_editor(qtbot, tmp_path):
    """An editor whose input is a real raw file, with one unsaved edit."""
    path = write_raw(tmp_path / "high-counts.h5", np.full((2, 4, 4), 7, np.uint16))
    window = SimpleMaskWindow()
    qtbot.addWidget(window)
    open_raw(qtbot, window, path)
    window.kernel.mask_kernel.evaluate("mask_list", zero_loc=np.array([[0], [1]]))
    window.kernel.mask_apply("mask_list")
    window.mark_unsaved()
    return window, Path(path)


def save_to(window, destination):
    with patch.object(
        QFileDialog, "getSaveFileName", return_value=(str(destination), "All (*)")
    ):
        window.action_save_mask.trigger()


def root_keys(path):
    with h5py.File(path, "r") as f:
        return sorted(f)


class TestSaveMask:
    def test_extension_appended_name_that_is_the_input_is_refused(self, raw_editor):
        window, raw = raw_editor
        before = raw.read_bytes()
        save_to(window, raw.with_suffix(""))  # dialog confirmed "high-counts"
        assert raw.read_bytes() == before
        assert root_keys(raw) == ["entry"]
        assert window.has_unsaved_changes()
        assert raw.name in window.status_bar.currentMessage()

    def test_direct_choice_of_the_input_is_refused(self, raw_editor):
        window, raw = raw_editor
        before = raw.read_bytes()
        save_to(window, raw)
        assert raw.read_bytes() == before and window.has_unsaved_changes()

    def test_symlink_to_the_input_is_refused(self, raw_editor, tmp_path):
        window, raw = raw_editor
        link = tmp_path / "alias.h5"
        link.symlink_to(raw)
        before = raw.read_bytes()
        save_to(window, link)
        assert raw.read_bytes() == before and window.has_unsaved_changes()

    def test_input_of_the_viewer_is_refused(self, qtbot, tmp_path):
        loaded = tmp_path / "analysis.h5"
        loaded.write_bytes(b"viewer input")
        target = ListDataModel()
        target.append("analysis.h5")
        viewer = SimpleNamespace(
            vk=SimpleNamespace(path=str(tmp_path), target=target, cache={})
        )
        window = make_window(qtbot, parent_viewer=viewer)
        window.mark_unsaved()
        save_to(window, tmp_path / "analysis")  # becomes analysis.h5
        assert loaded.read_bytes() == b"viewer input"
        assert window.has_unsaved_changes()

    def test_kernel_writers_refuse_an_input_themselves(self, raw_editor, qtbot):
        window, raw = raw_editor
        before = raw.read_bytes()
        with pytest.raises(ExportCollisionError):
            window.kernel.save_mask(str(raw), input_paths=[str(raw)])
        window.kernel.new_partition = {"x": np.zeros(3)}
        with pytest.raises(ExportCollisionError):
            window.kernel.save_partition(str(raw), input_paths=[str(raw)])
        assert raw.read_bytes() == before

    def test_existing_file_under_an_appended_name_needs_confirmation(
        self, qtbot, tmp_path, _no_modal_dialogs
    ):
        window = make_window(qtbot)
        window.mark_unsaved()
        existing = tmp_path / "out.h5"
        existing.write_bytes(b"precious")
        question = _no_modal_dialogs.question

        question.return_value = QMessageBox.StandardButton.No
        save_to(window, tmp_path / "out")
        assert question.called
        assert existing.read_bytes() == b"precious" and window.has_unsaved_changes()

        question.return_value = QMessageBox.StandardButton.Yes
        save_to(window, tmp_path / "out")
        assert root_keys(existing) == ["mask"] and not window.has_unsaved_changes()

    def test_dialog_confirmed_overwrite_is_not_asked_twice(
        self, qtbot, tmp_path, _no_modal_dialogs
    ):
        window = make_window(qtbot)
        existing = tmp_path / "out.h5"
        existing.write_bytes(b"x")
        save_to(window, existing)  # the file dialog already asked
        assert not _no_modal_dialogs.question.called
        assert root_keys(existing) == ["mask"]

    def test_plain_save_still_works_and_clears_the_dirty_flag(self, qtbot, tmp_path):
        window = make_window(qtbot)
        window.mark_unsaved()
        save_to(window, tmp_path / "mask")
        assert root_keys(tmp_path / "mask.h5") == ["mask"]
        assert not window.has_unsaved_changes()

    def test_failed_write_is_reported_and_keeps_the_dirty_state(
        self, qtbot, tmp_path, _no_modal_dialogs
    ):
        window = make_window(qtbot)
        window.mark_unsaved()
        destination = tmp_path / "missing-folder" / "mask.h5"
        save_to(window, destination)  # h5py raises an OSError subclass
        assert window.has_unsaved_changes()
        assert "mask.h5" in window.status_bar.currentMessage()
        assert not destination.exists()


# --------------------------------------------------------------------------- #
# every file write in the package is behind the export gate
# --------------------------------------------------------------------------- #
WRITE_MODES = re.compile(r"[wax+]")
# (module path relative to the package, function) -> reason it is not gated
UNGATED_WRITES: dict[tuple[str, str], str] = {}
# the kernel's writers call its one helper, which builds the guard and claims
GATES = {"claim", "_claim_output"}
WRITERS = {
    "save",
    "savez",
    "savez_compressed",
    "savetxt",
    "imwrite",
    "imsave",
    "tofile",
    "to_hdf",
    "write_text",
    "write_bytes",
}
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _is_write(node):
    """A call that can create or modify a file."""
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    if name in WRITERS:
        return True
    if name not in {"File", "open"}:
        return False
    # builtin open(path, mode) / h5py.File(path, mode) vs Path.open(mode)
    index = 0 if name == "open" and isinstance(func, ast.Attribute) else 1
    given = node.args[index : index + 1] or [
        k.value for k in node.keywords if k.arg == "mode"
    ]
    if not given:
        return False  # the default mode reads
    mode = given[0]
    # a mode that is not a literal cannot be shown to be read-only
    return not isinstance(mode, ast.Constant) or bool(
        WRITE_MODES.search(str(mode.value))
    )


def test_every_file_write_in_the_package_is_behind_a_claim():
    root = Path(simplemask_pkg.__file__).parent
    found, ungated = [], []
    for source in sorted(root.rglob("*.py")):
        rel = source.relative_to(root).as_posix()
        tree = ast.parse(source.read_text())
        owner = {}  # call node -> innermost enclosing function (None: module level)

        def visit(node, current):
            for child in ast.iter_child_nodes(node):
                inner = child if isinstance(child, FUNCTIONS) else current
                if isinstance(child, ast.Call):
                    owner[child] = current
                visit(child, inner)

        visit(tree, None)
        for call, func in owner.items():
            if not _is_write(call):
                continue
            scope = func if func is not None else tree
            name = func.name if func is not None else "<module>"
            found.append((rel, name))
            gated = any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in GATES
                for n in ast.walk(scope)
            )
            if not gated and (rel, name) not in UNGATED_WRITES:
                ungated.append((rel, name, call.lineno))
    assert ("simplemask_kernel.py", "save_mask") in found
    assert ("simplemask_kernel.py", "save_partition") in found
    assert not ungated, f"file writes without ExportGuard.claim: {ungated}"
    helper = inspect.getsource(SimpleMaskKernel._claim_output)
    assert "ExportGuard(" in helper and ".claim(" in helper


# --------------------------------------------------------------------------- #
# R6-01 (editor side): the file's own mask seeds the editor
# --------------------------------------------------------------------------- #
class TestSeededMask:
    @staticmethod
    def file_mask():
        mask = np.ones(SHAPE, dtype=np.uint8)
        mask[0, 0] = 0
        mask[5, 6:9] = 0
        return mask

    def test_editor_starts_from_the_files_mask_and_is_not_dirty(self, qtbot):
        mask = self.file_mask()
        window = make_window(qtbot, mask=mask)
        np.testing.assert_array_equal(window.kernel.mask, mask.astype(bool))
        assert not window.has_unsaved_changes()
        assert not window.windowTitle().endswith("*")

    def test_seeded_exclusions_are_visible_and_stay_visible_after_reset(self, qtbot):
        mask = self.file_mask().astype(bool)
        image = 1.0 + np.add.outer(np.arange(SHAPE[0]), np.arange(SHAPE[1]))
        window = make_window(qtbot, image=image, mask=mask)
        log = np.log10(image + 1)
        normalised = (log - log.min()) / (log.max() - log.min())
        expected = np.where(mask, normalised, 0.3 * normalised)
        assert window.action_toggle_mask.isChecked()
        np.testing.assert_allclose(window.image_view.image, expected)
        draw_rect(window, "rectangle", (10, 10), (4, 4))
        window.action_reset.trigger()
        assert window.action_toggle_mask.isChecked()
        np.testing.assert_allclose(window.image_view.image, expected)

    def test_without_exclusions_the_overlay_stays_off(self, qtbot):
        window = make_window(qtbot, mask=np.ones(SHAPE, dtype=bool))
        assert not window.action_toggle_mask.isChecked()

    def test_seed_is_exported_and_history_starts_from_it(self, qtbot):
        mask = self.file_mask().astype(bool)
        window = make_window(qtbot, mask=mask)
        exported = []
        window.mask_exported.connect(exported.append)
        window.action_apply_viewer.trigger()
        np.testing.assert_array_equal(exported[0], mask)
        draw_rect(window, "rectangle", (10, 10), (4, 4))
        assert masked_pixels(window) == {(0, 0), (5, 6), (5, 7), (5, 8)} | block(
            10, 10, 4
        )
        window.action_undo.trigger()
        np.testing.assert_array_equal(window.kernel.mask, mask)
        window.action_undo.trigger()  # nothing before the seeded state
        np.testing.assert_array_equal(window.kernel.mask, mask)
        window.action_redo.trigger()
        window.action_reset.trigger()
        np.testing.assert_array_equal(window.kernel.mask, mask)

    def test_seed_is_saved(self, qtbot, tmp_path):
        mask = self.file_mask().astype(bool)
        window = make_window(qtbot, mask=mask)
        save_to(window, tmp_path / "seed.h5")
        with h5py.File(tmp_path / "seed.h5", "r") as f:
            np.testing.assert_array_equal(f["mask"][()], mask.astype(np.uint8))

    def test_geometry_updates_still_work_after_seeding(self, qtbot):
        window = make_window(qtbot, mask=self.file_mask())
        window.kernel.update_parameters({"bcx": 40.0})  # compares the metadata dicts
        assert window.kernel.get_center()[0] == 40.0
        assert "mask" not in window.kernel.metadata

    def test_absent_key_keeps_every_pixel_accepted(self, qtbot):
        window = make_window(qtbot)
        assert window.kernel.mask.all()

    def test_shape_mismatch_warns_and_keeps_every_pixel_accepted(self, qtbot, caplog):
        with caplog.at_level(logging.WARNING):
            window = make_window(qtbot, mask=np.zeros((10, 10), dtype=bool))
        assert window.kernel.mask.all()
        assert any("mask" in r.message and "shape" in r.message for r in caplog.records)

    def test_seed_is_a_copy(self, qtbot):
        mask = self.file_mask()
        window = make_window(qtbot, mask=mask)
        mask[:] = 0  # the viewer's array changes later; the editor's layer must not
        assert window.kernel.mask.sum() == mask.size - 4
