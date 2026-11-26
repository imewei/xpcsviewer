"""Regression tests for review round 8: state remembered across events keeps its
identity key.

R8-01  the Mask Editor's source file is (path, revision): a same-path file with
       another revision is an ordinary target, not "the source"
R8-04  the SAXS-2D view has ONE record of what it renders (path, revision,
       rotate); cursor readout, default ROI origin and the spinbox read it
R8-05  the saved two-time selection survives a save made before the tab drew;
       a save -> restore -> save cycle changes no session field

Identity key of every value remembered from one event to a later one:

=====================================  =========================================
value                                  key
=====================================  =========================================
editor hand-off (viewer, R8-01)        (path, ``_file_identity`` revision) + one
                                       token per editor session
per-target mask record (viewer)        the target's XpcsFile object (weak): a
                                       reload or re-add is a new object
SAXS-2D rendered frame (view, R8-04)   (path, revision, rotate) of the file drawn
SAXS-2D spinbox / slider               the row of the rendered frame's path
saved two-time selection (session)     (path, absolute processed-bin ID)
=====================================  =========================================

The viewer is the real offscreen ``XpcsViewer`` and the editor the real
``SimpleMaskWindow``, over real h5py-written files.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import pytest
from qtpy import QtCore, QtWidgets
from qtpy.QtWidgets import QFileDialog, QMessageBox

import xpcsviewer.xpcs_viewer  # noqa: F401  (registers the module for _make_env)
from tests.unit import test_review_r3_maps_roi as r3
from tests.unit.test_review_r4_cli_average import _write_twotime
from tests.unit.test_review_r4_viewer_gui import _idle, _tab
from tests.unit.test_review_r6_simplemask import draw_rect
from tests.unit.test_review_r6_viewer_gui import _make_env, _select, _write_detector
from xpcsviewer.fileIO.hdf_reader import _file_identity, clear_connection_pool

pytestmark = pytest.mark.unit

K = r3.K
SHAPE = (8, 10)


@pytest.fixture(autouse=True)
def _fresh_pool():
    clear_connection_pool()
    yield
    clear_connection_pool()


@pytest.fixture(autouse=True)
def _no_modal_dialogs():
    with (
        patch.object(QMessageBox, "warning") as warn,
        patch.object(QMessageBox, "critical") as crit,
        patch.object(QMessageBox, "question") as question,
    ):
        question.return_value = QMessageBox.StandardButton.Yes
        yield SimpleNamespace(warning=warn, critical=crit, question=question)


def _write_frame(path, value, centre, masked=()):
    """Real detector file whose SAXS image is ``value`` everywhere."""
    _write_detector(path, SHAPE, list(masked), centre=centre)
    with h5py.File(path, "r+") as f:
        del f[K["saxs_2d"]]
        f[K["saxs_2d"]] = np.full((1, *SHAPE), float(value))


def _replace_with(path, write):
    """Atomically replace ``path`` by a freshly written file."""
    staged = path.with_name("staged_" + path.name)
    write(staged)
    os.replace(staged, path)


# --------------------------------------------------------------------------
# R8-04 : SAXS-2D record of what is rendered
# --------------------------------------------------------------------------
@pytest.fixture
def saxs(qtbot, monkeypatch, tmp_path):
    env = _make_env(
        qtbot,
        monkeypatch,
        tmp_path,
        {
            "A.h5": lambda p: _write_frame(p, 4, (2.0, 3.0)),
            "B.h5": lambda p: _write_frame(p, 81, (6.0, 5.0)),
        },
    )
    view = env.view
    view.show()
    view.tabWidget.setCurrentIndex(_tab(env.xv, "saxs_2d"))
    view.cb_saxs2D_type.setCurrentText("linear")
    QtWidgets.QApplication.processEvents()
    env.a, env.b = env.data / "A.h5", env.data / "B.h5"
    return env


def _click(view, row):
    """Highlight ``row`` and let the list's click redraw the SAXS tab."""
    view.list_view_target.selectionModel().select(
        view.vk.target.index(row, 0),
        QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect,
    )
    view.update_plot()
    _idle(view)


def _spin(view, row):
    view.spinBox_saxs2d_selection.setValue(row)
    _idle(view)


def _image(view):
    return np.unique(np.asarray(view.pg_saxs.image))


def _hover(view, col=0, row=0):
    view.saxs2d_display.setText("")
    centre = view.pg_saxs.pixel_centre(col, row)
    view.saxs2d_mouseMoved(
        view.pg_saxs.getView().mapViewToScene(QtCore.QPointF(*centre))
    )
    return view.saxs2d_display.text()


def _ring_centre(view):
    view.cb_saxs2D_roi_type.setCurrentIndex(1)  # Phi-Ring
    view.saxs2d_roi_add()
    (ring,) = view.pg_saxs.get_roi_list()
    return ring["center"]


def test_r8_04_cursor_reads_the_file_the_spinbox_drew(saxs):
    view = saxs.view
    _click(view, 0)
    assert _hover(view).startswith("I=4.0000e+00")
    _spin(view, 1)
    np.testing.assert_array_equal(_image(view), [81])
    assert view.get_selected_rows() == [0]  # the list still highlights A
    assert _hover(view).startswith("I=8.1000e+01")


def test_r8_04_clicking_the_highlighted_row_redraws_it(saxs):
    view = saxs.view
    _click(view, 0)
    _spin(view, 1)
    np.testing.assert_array_equal(_image(view), [81])
    view.update_plot()  # a click on the row that is already highlighted
    _idle(view)
    np.testing.assert_array_equal(_image(view), [4])
    assert _hover(view).startswith("I=4.0000e+00")


def test_r8_04_default_roi_origin_is_the_drawn_files_beam_centre(saxs):
    view = saxs.view
    _click(view, 0)
    _spin(view, 1)
    expected = view.pg_saxs.pixel_centre(6.0, 5.0)  # B's beam centre, not A's (2, 3)
    np.testing.assert_allclose(_ring_centre(view), expected)


def test_r8_04_the_record_names_path_revision_and_rotate(saxs):
    view = saxs.view
    _click(view, 0)
    frame = view.pg_saxs.rendered_frame  # async path
    assert (frame.path, frame.revision, frame.rotate) == (
        str(saxs.a),
        _file_identity(str(saxs.a)),
        False,
    )
    view.saxs2d_rotate.setChecked(True)
    _idle(view)
    _spin(view, 1)  # sync path
    frame = view.pg_saxs.rendered_frame
    assert (frame.path, frame.revision, frame.rotate) == (
        str(saxs.b),
        _file_identity(str(saxs.b)),
        True,
    )
    assert frame.center == (6.0, 5.0)
    # the marker, the cursor and the ROI export all follow that same orientation
    assert _hover(view).startswith("I=8.1000e+01")
    np.testing.assert_allclose(_ring_centre(view), view.pg_saxs.pixel_centre(6.0, 5.0))
    (ring,) = view.pg_saxs.get_roi_list()
    tf = view.pg_saxs.getImageItem().transform()
    col_step = ring["pixel_frame"]["col_step"]
    np.testing.assert_allclose(col_step, (tf.m11(), tf.m12()), atol=1e-9)


def test_r8_04_a_replaced_file_is_not_read_through_the_old_frame(saxs):
    view = saxs.view
    _click(view, 0)
    _spin(view, 1)
    assert _hover(view).startswith("I=8.1000e+01")
    _replace_with(saxs.b, lambda p: _write_frame(p, 7, (6.0, 5.0)))
    # not reloaded yet: the frame on screen belongs to a revision that is gone,
    # so the readout must not pair it with the new file's numbers (it says why,
    # instead of freezing on its last value)
    assert _hover(view) == "file changed on disk"
    view.reload_source()
    _idle(view)
    # the reload redraws the highlighted row; whatever is on screen now, the
    # record names its CURRENT revision and the readout and spinbox agree with it
    frame = view.pg_saxs.rendered_frame
    assert frame.revision == _file_identity(frame.path)
    assert _hover(view).startswith(f"I={_image(view)[0]:.4e}")
    name = os.path.basename(frame.path)
    assert view.vk.target[view.spinBox_saxs2d_selection.value()] == name
    # B itself, drawn again, reads the replaced file's numbers
    _spin(view, 1)
    assert view.pg_saxs.rendered_frame.revision == _file_identity(str(saxs.b))
    assert _hover(view).startswith("I=7.0000e+00")


def test_r8_04_spinbox_follows_the_file_drawn_by_a_list_click(saxs):
    view = saxs.view
    _click(view, 1)
    assert view.spinBox_saxs2d_selection.value() == 1
    assert view.horizontalSlider_saxs2d_selection.value() == 1
    _click(view, 0)
    assert view.spinBox_saxs2d_selection.value() == 0


def test_r8_04_spinbox_after_a_reorder_names_the_file_it_shows(saxs):
    view = saxs.view
    _click(view, 0)
    view.btn_down.click()  # A moves to row 1
    _idle(view)
    assert list(view.vk.target) == ["B.h5", "A.h5"]
    np.testing.assert_array_equal(_image(view), [4])
    assert view.spinBox_saxs2d_selection.value() == 1
    assert _hover(view).startswith("I=4.0000e+00")


def test_r8_04_spinbox_after_remove_and_readd_is_clamped_and_truthful(saxs):
    view = saxs.view
    _click(view, 0)
    _spin(view, 1)
    np.testing.assert_array_equal(_image(view), [81])
    _click(view, 0)  # highlight A, remove it
    view.remove_target()
    _idle(view)
    assert list(view.vk.target) == ["B.h5"]
    assert view.spinBox_saxs2d_selection.maximum() == 0
    assert view.spinBox_saxs2d_selection.value() == 0
    np.testing.assert_array_equal(_image(view), [81])
    assert _hover(view).startswith("I=8.1000e+01")
    view.vk.add_target(["A.h5"])  # re-added behind B
    view.update_box(view.vk.target, mode="target")
    assert view.spinBox_saxs2d_selection.maximum() == 1
    assert view.spinBox_saxs2d_selection.value() == 0  # B is still the file drawn
    assert _hover(view).startswith("I=8.1000e+01")
    _spin(view, 1)
    np.testing.assert_array_equal(_image(view), [4])
    assert _hover(view).startswith("I=4.0000e+00")


def test_r8_04_an_emptied_target_list_leaves_no_readout(saxs):
    view = saxs.view
    _click(view, 0)
    _spin(view, 1)
    view.list_view_target.selectAll()
    view.remove_target()
    _idle(view)
    assert len(view.vk.target) == 0
    assert view.spinBox_saxs2d_selection.minimum() == 0
    assert view.spinBox_saxs2d_selection.value() == 0
    assert _hover(view) == ""


def test_r8_04_the_spinbox_is_not_a_session_field():
    xv = sys.modules["xpcsviewer.xpcs_viewer"]
    widgets = {row[1] for row in xv._SESSION_CONTROLS}
    assert not {"spinBox_saxs2d_selection", "horizontalSlider_saxs2d_selection"} & (
        widgets
    )


def test_r8_04_an_open_mask_editor_does_not_disturb_the_record(saxs):
    view = saxs.view
    _click(view, 0)
    view.open_simplemask()
    editor = view._simplemask_window
    try:
        _spin(view, 1)
        editor.action_apply_viewer.trigger()  # import_mask -> redraw
        _idle(view)
        # whichever file is on screen after the Apply, the readout, the record
        # and the spinbox all name that one file
        drawn = _image(view)[0]
        assert _hover(view).startswith(f"I={drawn:.4e}")
        name = os.path.basename(view.pg_saxs.rendered_frame.path)
        assert view.vk.target[view.spinBox_saxs2d_selection.value()] == name
    finally:
        editor.mark_saved()
        editor.close()


# --------------------------------------------------------------------------
# R8-01 : the editor's source is (path, revision)
# --------------------------------------------------------------------------
MASK_SHAPE = (16, 20)


@pytest.fixture
def mask_env(qtbot, monkeypatch, tmp_path):
    """Two real targets with DIFFERENT canonical masks: A rejects (0, 0), B (5, 5)."""
    env = _make_env(
        qtbot,
        monkeypatch,
        tmp_path,
        {
            "A.h5": lambda p: _write_detector(p, MASK_SHAPE, [(0, 0)]),
            "B.h5": lambda p: _write_detector(p, MASK_SHAPE, [(5, 5)]),
        },
    )
    env.xa, env.xb = env.view.vk.get_xf_list([0, 1])
    env.a_path, env.b_path = env.data / "A.h5", env.data / "B.h5"
    yield env
    window = env.view._simplemask_window
    if window is not None:
        window.mark_saved()
        window.close()


def _canonical(xf):
    return np.asarray(xf.qmap.mask) != 0


def _current(xf):
    return np.asarray(xf.mask) != 0


def _open_editor(env, row=0):
    _select(env.view, row)
    env.view.open_simplemask()
    return env.view._simplemask_window


def _user_edit(editor, origin=(10, 10)):
    before = editor.kernel.mask.copy()
    draw_rect(editor, "rectangle", origin, (3, 3))
    edited = before & ~editor.kernel.mask
    assert edited.sum() > 0
    return edited


def _xf(env, name):
    return env.view.vk.get_xf_list([list(env.view.vk.target).index(name)])[0]


def test_r8_01_a_replaced_source_file_gets_no_wholesale_mask(mask_env):
    env = mask_env
    editor = _open_editor(env)
    assert not editor.kernel.mask[0, 0]  # the OLD file's own exclusion
    _replace_with(env.a_path, lambda p: _write_detector(p, MASK_SHAPE, [(7, 7)]))
    env.view.reload_source()
    xa = _xf(env, "A.h5")
    assert xa is not env.xa and _current(xa)[0, 0] and not _current(xa)[7, 7]
    editor.action_apply_viewer.trigger()  # untouched editor
    assert _current(xa)[0, 0]  # the old file's exclusion did not travel
    assert not _current(xa)[7, 7]  # the new file keeps its own
    assert "changed on disk" in env.view.statusbar.currentMessage()
    np.testing.assert_array_equal(_current(_xf(env, "B.h5")), _canonical(env.xb))


def test_r8_01_a_replaced_source_file_still_loses_only_the_users_edits(mask_env):
    env = mask_env
    editor = _open_editor(env)
    edits = _user_edit(editor)
    _replace_with(env.a_path, lambda p: _write_detector(p, MASK_SHAPE, [(7, 7)]))
    env.view.reload_source()
    xa = _xf(env, "A.h5")
    editor.action_apply_viewer.trigger()
    np.testing.assert_array_equal(_current(xa), _canonical(xa) & ~edits)
    assert _current(xa)[0, 0]


def test_r8_01_an_unchanged_source_is_still_the_source_after_a_reload(mask_env):
    env = mask_env
    editor = _open_editor(env)
    edits = _user_edit(editor)
    env.view.reload_source()  # nothing changed on disk
    xa = _xf(env, "A.h5")
    editor.action_apply_viewer.trigger()
    np.testing.assert_array_equal(_current(xa), editor.kernel.mask & _canonical(xa))
    assert not _current(xa)[0, 0]  # the source keeps its own exclusion
    assert "changed on disk" not in env.view.statusbar.currentMessage()
    assert not _current(_xf(env, "B.h5"))[tuple(np.argwhere(edits)[0])]


def _save_mask_to(editor, destination):
    with patch.object(
        QFileDialog, "getSaveFileName", return_value=(str(destination), "All (*)")
    ):
        editor.action_save_mask.trigger()


def test_r8_01_save_mask_still_protects_the_path_whatever_the_revision(mask_env):
    env = mask_env
    editor = _open_editor(env)
    _replace_with(env.a_path, lambda p: _write_detector(p, MASK_SHAPE, [(7, 7)]))
    env.view.reload_source()
    assert str(env.a_path) in editor._protected_paths()
    before = env.a_path.read_bytes()
    _save_mask_to(editor, env.a_path)
    assert env.a_path.read_bytes() == before
    assert not editor.status_bar.currentMessage().startswith("Mask saved")


def test_r8_01_apply_undo_apply_without_a_seed_restores_the_other_targets(mask_env):
    env = mask_env
    env.xa.mask = np.ones((3, 3), dtype=bool)  # no usable seed: wrong shape
    editor = _open_editor(env)
    assert editor.kernel.seed_mask is None
    start_b = _current(env.xb).copy()
    edits = _user_edit(editor)
    editor.action_apply_viewer.trigger()
    np.testing.assert_array_equal(_current(env.xb), start_b & ~edits)
    editor.action_undo.trigger()
    assert editor.kernel.mask.all()
    editor.action_apply_viewer.trigger()
    np.testing.assert_array_equal(_current(env.xb), start_b)


def test_r8_01_a_target_removed_and_readded_starts_from_its_own_mask(mask_env):
    env = mask_env
    editor = _open_editor(env)
    edits = _user_edit(editor)
    editor.action_apply_viewer.trigger()
    np.testing.assert_array_equal(_current(env.xb), _canonical(env.xb) & ~edits)
    _select(env.view, 1)
    env.view.remove_target()
    env.view.vk.add_target(["B.h5"])
    env.view.update_box(env.view.vk.target, mode="target")
    xb = _xf(env, "B.h5")
    assert xb is not env.xb
    np.testing.assert_array_equal(_current(xb), _canonical(xb))  # a fresh file
    editor.action_undo.trigger()
    editor.action_apply_viewer.trigger()  # no edits left
    np.testing.assert_array_equal(_current(xb), _canonical(xb))


def test_r8_01_a_target_replaced_and_reloaded_does_not_reuse_the_old_base(mask_env):
    env = mask_env
    editor = _open_editor(env)
    edits = _user_edit(editor)
    editor.action_apply_viewer.trigger()
    _replace_with(env.b_path, lambda p: _write_detector(p, MASK_SHAPE, [(9, 9)]))
    env.view.reload_source()
    xb = _xf(env, "B.h5")
    editor.action_apply_viewer.trigger()  # same session, second Apply
    np.testing.assert_array_equal(_current(xb), _canonical(xb) & ~edits)
    assert not _current(xb)[9, 9] and _current(xb)[5, 5]


# --------------------------------------------------------------------------
# R8-05 : the saved two-time selection and the whole session round trip
# --------------------------------------------------------------------------
def _reopen(qtbot, env):
    view = env.xv.XpcsViewer(path=str(env.data))
    qtbot.addWidget(view)
    return view


def _identity(view):
    params = view._collect_session_state().analysis_params
    return params.twotime_source_file, params.twotime_selected_bin


def _saved_on_saxs(qtbot, monkeypatch, tmp_path):
    from tests.unit.test_review_r7_state import _twotime_session

    env = _twotime_session(qtbot, monkeypatch, tmp_path)  # bin 2 chosen
    env.view.tabWidget.setCurrentIndex(0)
    assert env.view.session_manager.save_session(env.view._collect_session_state())
    env.view.close()
    env.path = os.path.abspath(env.data / "scan.h5")
    return env


def test_r8_05_a_save_before_the_tab_was_drawn_keeps_the_selection(
    qtbot, monkeypatch, tmp_path
):
    from tests.unit.test_review_r7_state import _restored

    env = _saved_on_saxs(qtbot, monkeypatch, tmp_path)
    second = _reopen(qtbot, env)
    assert second.tabWidget.currentIndex() == 0
    assert _identity(second) == (env.path, 2)  # nothing drew Two-time
    assert second.session_manager.save_session(second._collect_session_state())
    second.close()
    third = _restored(qtbot, env)  # the third launch opens Two-time
    assert third.comboBox_twotime_selection.currentText().startswith("qbin=2")
    np.testing.assert_allclose(np.unique(third.mp_2t_hdls["tt"].image), 10.0)


def test_r8_05_a_real_selection_supersedes_the_restored_identity(
    qtbot, monkeypatch, tmp_path
):
    env = _saved_on_saxs(qtbot, monkeypatch, tmp_path)
    second = _reopen(qtbot, env)
    second.tabWidget.setCurrentIndex(_tab(env.xv, "twotime"))
    second.twotime_correct_diag.setChecked(False)
    second.plot_twotime()
    assert _identity(second) == (env.path, 2)  # drawn: the combo agrees
    second.comboBox_twotime_selection.setCurrentIndex(0)
    second.plot_twotime()
    assert _identity(second) == (env.path, 1)


def _event_remove_and_readd(env, view):
    _select(view, 0)
    view.remove_target()
    view.vk.add_target(["scan.h5"])
    view.update_box(view.vk.target, mode="target")


def _event_reload_with_other_bins(env, view):
    from tests.unit.test_review_r7_state import _reprocess

    _reprocess(env.data / "scan.h5", [2, 1], value=7)  # bin 2 moves to position 0
    view.reload_source()


def _event_reorder(env, view):
    view.vk.add_target(["other.h5"])
    view.update_box(view.vk.target, mode="target")
    _select(view, 0)
    view.btn_down.click()


def _event_open_mask_editor(env, view):
    view.open_simplemask()
    env.editor = view._simplemask_window


@pytest.mark.parametrize(
    "event",
    [
        _event_remove_and_readd,
        _event_reload_with_other_bins,
        _event_reorder,
        _event_open_mask_editor,
    ],
)
def test_r8_05_the_restored_identity_survives_what_happens_before_the_next_save(
    qtbot, monkeypatch, tmp_path, event
):
    env = _saved_on_saxs(qtbot, monkeypatch, tmp_path)
    _write_twotime(env.data / "other.h5", [0.01, 0.02], value=3)
    second = _reopen(qtbot, env)
    try:
        event(env, second)
        # the key is (path, ABSOLUTE bin ID): neither a re-added nor a reordered
        # nor a reprocessed file changes what was chosen
        assert _identity(second) == (env.path, 2)
    finally:
        editor = getattr(env, "editor", None)
        if editor is not None:
            editor.mark_saved()
            editor.close()


def _non_default(view):
    """Move every session control off its default."""
    for combo in (
        view.cb_saxs2D_cmap,
        view.cb_saxs2D_type,
        view.cb_saxs_type,
        view.cb_stab_type,
        view.g2_fitting_function,
        view.cb_twotime_cmap,
        view.cb_qmap_cmap,
    ):
        assert combo.count() > 1, combo.objectName()
        combo.setCurrentIndex(combo.count() - 1)
    view.saxs2d_autolevel.setChecked(False)
    view.g2_show_fit.setChecked(False)


def test_r8_05_save_restore_save_changes_no_session_field(qtbot, monkeypatch, tmp_path):
    from tests.unit.test_review_r7_state import _twotime_session

    env = _twotime_session(qtbot, monkeypatch, tmp_path)
    first = env.view
    _non_default(first)
    first.tabWidget.setCurrentIndex(_tab(env.xv, "saxs_1d"))
    saved = first._collect_session_state()
    assert first.session_manager.save_session(saved)
    first.close()

    second = _reopen(qtbot, env)  # no tab is visited after the restore
    again = second._collect_session_state()

    left, right = (dataclasses.asdict(s.analysis_params) for s in (saved, again))
    assert {k: (left[k], right[k]) for k in left if left[k] != right[k]} == {}
    assert again.active_tab == saved.active_tab
    assert again.data_path == saved.data_path
    assert [(f.path, f.order) for f in again.target_files] == [
        (f.path, f.order) for f in saved.target_files
    ]
    # and the values really are the non-default ones that were saved
    assert left["saxs2d_auto_level"] is False and left["g2_show_fit"] is False
    assert left["twotime_selected_bin"] == 2
    # a second cycle is the identity too
    assert second.session_manager.save_session(again)
    second.close()
    third = _reopen(qtbot, env)
    final = dataclasses.asdict(third._collect_session_state().analysis_params)
    assert final == right


# --------------------------------------------------------------------------
# Q-bin ordinals shown to the user: session g2_q_index / sb_g2_bayesian_qidx
# --------------------------------------------------------------------------
def test_r8_qbin_ordinal_names_the_displayed_bin_after_a_reload_with_other_bins(
    qtbot, monkeypatch, tmp_path
):
    from tests.unit.test_review_r3_fit_status import _write_file

    tau = np.logspace(-3, 0, 14)
    env = _make_env(
        qtbot,
        monkeypatch,
        tmp_path,
        {"g.h5": lambda p: _write_file(p, tau, q=(0.1, 0.2, 0.3))},
    )
    view = env.view
    _select(view, 0)
    view.tabWidget.setCurrentIndex(_tab(env.xv, "g2"))
    view.update_plot()
    _idle(view)
    view.g2_qauto.setChecked(False)  # keep the whole range through the reload
    view.g2_qmin.setValue(0.0)
    view.g2_qmax.setValue(1.0)
    view.sb_g2_bayesian_qidx.setMaximum(2)
    view.sb_g2_bayesian_qidx.setValue(1)
    assert view._g2_bin_table()[1][1][1] == pytest.approx(0.2)

    _replace_with(
        env.data / "g.h5", lambda p: _write_file(p, tau, q=(0.15, 0.25, 0.35))
    )
    view.reload_source()
    _idle(view)
    assert view.sb_g2_bayesian_qidx.value() == 1  # the ordinal is a position

    _fname, bins = view._g2_bin_table()
    bin_id, q_value, label = bins[1]
    assert q_value == pytest.approx(0.25)  # the bin shown now, not the old 0.2
    view._on_g2_qbin_changed(1)
    assert f"{q_value:.4g}" in view.statusbar.currentMessage()
    assert label in view.statusbar.currentMessage()
    bundle, q_idx = view._extract_g2_for_bayesian()
    assert q_idx == 1
    assert (bundle.bin_id, bundle.q_value) == (bin_id, pytest.approx(0.25))
    assert bundle.label == label


def test_r8_session_g2_q_index_is_not_a_saved_ordinal():
    # g2_q_index has no control (see the sweep-e test); the Bayesian Q-bin
    # spinbox is a position in the live bin table and is not a session field.
    xv = sys.modules["xpcsviewer.xpcs_viewer"]
    assert "sb_g2_bayesian_qidx" not in {row[1] for row in xv._SESSION_CONTROLS}
    assert "g2_q_index" not in {row[0] for row in xv._SESSION_CONTROLS}
