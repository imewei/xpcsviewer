"""Regression tests for review round 7: the Mask Editor hand-off.

R7-01  an Apply with no edits must not take pixels away from the OTHER targets:
       the source file gets the editor's mask, every other target loses only what
       the user excluded beyond the editor's seed, nobody gains a pixel its own
       canonical mask excludes
R7-02  the editor keeps protecting the analysis file it displays from Save Mask /
       Save Partition, independently of the viewer's target list and HDF5 pool

The viewer is the real offscreen ``XpcsViewer`` and the editor the real
``SimpleMaskWindow``, over real h5py-written files.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import pytest
from qtpy.QtWidgets import QFileDialog, QMessageBox

import xpcsviewer.xpcs_viewer  # noqa: F401  (registers the module for _make_env)
from tests.unit.test_review_r6_simplemask import draw_rect, open_raw, write_raw
from tests.unit.test_review_r6_viewer_gui import (
    _make_env,
    _select,
    _write_detector,
)
from xpcsviewer.fileIO.hdf_reader import clear_connection_pool
from xpcsviewer.utils.export_guard import ExportCollisionError

pytestmark = pytest.mark.unit

SHAPE = (16, 20)


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


@pytest.fixture
def env(qtbot, monkeypatch, tmp_path):
    """Two same-shaped real targets with DIFFERENT canonical masks:
    A.h5 rejects (0, 0); B.h5 rejects (5, 5)."""
    env = _make_env(
        qtbot,
        monkeypatch,
        tmp_path,
        {
            "A.h5": lambda p: _write_detector(p, SHAPE, [(0, 0)]),
            "B.h5": lambda p: _write_detector(p, SHAPE, [(5, 5)]),
        },
    )
    env.xa, env.xb = env.view.vk.get_xf_list([0, 1])
    env.a_path, env.b_path = env.data / "A.h5", env.data / "B.h5"
    yield env
    window = env.view._simplemask_window
    if window is not None:
        window.mark_saved()
        window.close()


def canonical(xf):
    return np.asarray(xf.qmap.mask) != 0


def current(xf):
    return np.asarray(xf.mask) != 0


def open_editor(env, row=0):
    _select(env.view, row)
    env.view.open_simplemask()
    return env.view._simplemask_window


def user_edit(editor):
    """Draw one rectangle; returns the pixels it excluded beyond the seed."""
    before = editor.kernel.mask.copy()
    draw_rect(editor, "rectangle", (10, 10), (3, 3))
    edited = before & ~editor.kernel.mask
    assert edited.sum() > 0
    return edited


# --------------------------------------------------------------------------
# R7-01
# --------------------------------------------------------------------------
def test_r7_01_untouched_apply_leaves_the_other_target_alone(env):
    editor = open_editor(env)
    own_b = current(env.xb).copy()
    assert not editor.kernel.mask[0, 0] and own_b[0, 0]
    editor.action_apply_viewer.trigger()  # no edits
    assert not current(env.xa)[0, 0]  # the source keeps its own exclusion
    np.testing.assert_array_equal(current(env.xb), own_b)
    assert current(env.xb)[0, 0] and not current(env.xb)[5, 5]


def test_r7_01_other_target_loses_only_the_users_edits(env):
    editor = open_editor(env)
    edits = user_edit(editor)
    # the source is the file the editor was opened on, not the row selected now
    _select(env.view, 1)
    editor.action_apply_viewer.trigger()
    np.testing.assert_array_equal(
        current(env.xa), editor.kernel.mask & canonical(env.xa)
    )
    np.testing.assert_array_equal(current(env.xb), canonical(env.xb) & ~edits)
    assert current(env.xb)[0, 0]  # A's own exclusion did not travel to B


def test_r7_01_no_target_gains_a_pixel_its_canonical_mask_excludes(env):
    seed = np.ones(SHAPE, dtype=bool)
    seed[0, 0] = False
    editor_mask = np.ones(SHAPE, dtype=bool)  # re-enables (0, 0) and (5, 5)
    env.view.import_mask(editor_mask, seed, str(env.a_path))
    assert not current(env.xa)[0, 0]  # A's canonical mask excludes it
    assert not current(env.xb)[5, 5] and current(env.xb)[0, 0]
    for xf in (env.xa, env.xb):
        assert not (current(xf) & ~canonical(xf)).any()


def test_r7_01_apply_undo_apply_returns_other_targets_to_their_start(env):
    editor = open_editor(env)
    start_b = current(env.xb).copy()
    edits = user_edit(editor)
    editor.action_apply_viewer.trigger()
    np.testing.assert_array_equal(current(env.xb), start_b & ~edits)
    editor.action_undo.trigger()
    assert editor.kernel.mask[tuple(np.argwhere(edits)[0])]
    editor.action_apply_viewer.trigger()
    np.testing.assert_array_equal(current(env.xb), start_b)


def test_r7_01_a_mask_imported_by_other_means_keeps_its_exclusions(env):
    imported = canonical(env.xb).copy()
    imported[9, 9] = False  # a mask this target already holds (not from the editor)
    env.xb.mask = imported
    editor = open_editor(env)
    edits = user_edit(editor)
    editor.action_apply_viewer.trigger()
    np.testing.assert_array_equal(current(env.xb), imported & ~edits)
    editor.action_undo.trigger()
    editor.action_apply_viewer.trigger()
    np.testing.assert_array_equal(current(env.xb), imported)  # still excluded


def test_r7_01_a_mask_set_after_an_apply_is_the_new_base(env):
    editor = open_editor(env)
    edits = user_edit(editor)
    editor.action_apply_viewer.trigger()
    replaced = canonical(env.xb).copy()
    replaced[9, 9] = False
    env.xb.mask = replaced  # imported after the first Apply
    editor.action_apply_viewer.trigger()
    np.testing.assert_array_equal(current(env.xb), replaced & ~edits)


def test_r7_01_a_new_editor_session_builds_on_what_the_target_holds(env):
    editor = open_editor(env)
    first = user_edit(editor)
    editor.action_apply_viewer.trigger()
    editor.mark_saved()
    editor.close()
    editor = open_editor(env)  # reopened: seeded with A's mask, including `first`
    assert not (editor.kernel.mask & first).any()
    draw_rect(editor, "rectangle", (2, 2), (3, 3))
    second = editor.kernel.seed_mask & ~editor.kernel.mask
    assert second.sum() > 0
    editor.action_apply_viewer.trigger()
    np.testing.assert_array_equal(current(env.xb), canonical(env.xb) & ~first & ~second)


def test_r7_01_callers_that_send_only_a_mask_keep_the_old_behaviour(env):
    editor = open_editor(env)
    sent = []
    editor.mask_exported.connect(lambda m: sent.append(m))
    applied = []
    editor.mask_applied.connect(lambda *a: applied.append(a))
    editor.action_apply_viewer.trigger()
    assert len(sent) == 1 and len(applied) == 1
    mask, seed, source = applied[0]
    assert mask.shape == SHAPE and seed is editor.kernel.seed_mask
    assert source == str(env.a_path)
    # a bare mask has no seed and no source: every target gets (mask AND canonical)
    bare = np.ones(SHAPE, dtype=bool)
    bare[3, 3] = False
    env.view.import_mask(bare)
    for xf in (env.xa, env.xb):
        np.testing.assert_array_equal(current(xf), bare & canonical(xf))


# --------------------------------------------------------------------------
# R7-02
# --------------------------------------------------------------------------
def test_r7_02_the_viewer_sends_the_file_path_as_source_file(env):
    _select(env.view, 1)
    _, meta = env.view._get_simplemask_data()
    assert meta["source_file"] == str(env.b_path)
    editor = open_editor(env, 0)
    assert editor.kernel.metadata["source_file"] == str(env.a_path)
    assert str(env.a_path) in editor._protected_paths()


def _save_mask_to(editor, destination):
    with patch.object(
        QFileDialog, "getSaveFileName", return_value=(str(destination), "All (*)")
    ):
        editor.action_save_mask.trigger()


def _leave_the_viewer(env, name):
    """Remove ``name`` from the target list and the viewer's file cache."""
    view = env.view
    _select(view, 0)
    view.remove_target()
    path = str(env.data / name)
    assert path not in view.vk.cache or view.vk.cache.pop(path) is not None
    clear_connection_pool()
    assert name not in list(view.vk.target)
    assert path not in view.vk.cache


def test_r7_02_editor_keeps_protecting_a_file_the_viewer_no_longer_holds(
    env, _no_modal_dialogs
):
    editor = open_editor(env, 0)
    _leave_the_viewer(env, "A.h5")
    assert editor.isVisible()
    before = env.a_path.read_bytes()
    _save_mask_to(editor, env.a_path.with_suffix(""))  # the generic-overwrite route
    assert env.a_path.read_bytes() == before
    with h5py.File(env.a_path) as f:
        assert "entry" in f
    assert not editor.status_bar.currentMessage().startswith("Mask saved")
    _no_modal_dialogs.question.assert_not_called()  # refused before any prompt
    # a different name still saves
    _save_mask_to(editor, env.root / "fine.h5")
    assert (env.root / "fine.h5").exists()


def test_r7_02_protection_survives_a_hand_off_attempt_of_another_file(env):
    editor = open_editor(env, 0)
    _leave_the_viewer(env, "A.h5")
    _select(env.view, 0)  # B is now row 0
    env.view.open_simplemask()  # visible editor: focused, not reloaded
    assert editor.kernel.metadata["source_file"] == str(env.a_path)
    assert str(env.a_path) in editor._protected_paths()


def test_r7_02_protection_moves_with_the_data_the_editor_displays(env):
    editor = open_editor(env, 0)
    editor.mark_saved()
    editor.close()
    editor = open_editor(env, 1)
    assert editor.kernel.metadata["source_file"] == str(env.b_path)
    _leave_the_viewer_row(env, 1)  # B goes away; the editor still shows B
    assert str(env.b_path) in editor._protected_paths()


def _leave_the_viewer_row(env, row):
    _select(env.view, row)
    env.view.remove_target()
    env.view.vk.cache.pop(str(env.b_path), None)
    clear_connection_pool()


def test_r7_02_raw_open_replaces_the_hand_off_protection(env, qtbot, tmp_path):
    editor = open_editor(env, 0)
    _leave_the_viewer(env, "A.h5")
    raw = Path(write_raw(tmp_path / "raw.h5", np.full((2, 4, 4), 7, np.uint16)))
    open_raw(qtbot, editor, str(raw))
    assert "source_file" not in editor.kernel.metadata
    assert str(raw) in editor._protected_paths()
    assert str(env.a_path) not in editor._protected_paths()
    before = raw.read_bytes()
    _save_mask_to(editor, raw.with_suffix(""))
    assert raw.read_bytes() == before


def test_r7_02_mask_file_may_be_overwritten_but_never_the_source(env, tmp_path):
    editor = open_editor(env, 0)
    saved = tmp_path / "mine.h5"
    _save_mask_to(editor, saved)
    assert saved.exists()
    with patch.object(
        QFileDialog, "getOpenFileName", return_value=(str(saved), "All (*)")
    ):
        editor.action_load_mask.trigger()
    editor.mark_unsaved()
    _save_mask_to(editor, saved)  # a mask file the user loaded: overwrite is allowed
    assert not editor.has_unsaved_changes()
    _leave_the_viewer(env, "A.h5")
    with patch.object(
        QFileDialog, "getOpenFileName", return_value=(str(env.a_path), "All (*)")
    ):
        editor.action_load_mask.trigger()  # loading A as a "mask" fails: no such key
    _save_mask_to(editor, env.a_path)
    with h5py.File(env.a_path) as f:
        assert "entry" in f


def test_r7_02_partition_save_refuses_the_source(env):
    editor = open_editor(env, 0)
    _leave_the_viewer(env, "A.h5")
    editor.kernel.new_partition = {"x": np.zeros(3)}
    before = env.a_path.read_bytes()
    with pytest.raises(ExportCollisionError):
        editor.kernel.save_partition(str(env.a_path))
    assert env.a_path.read_bytes() == before


def test_r7_02_probe_scenario_with_many_other_files(env):
    """The reviewer's sequence: 30 other files, literal name confirmed."""
    editor = open_editor(env, 0)
    _leave_the_viewer(env, "A.h5")
    names = []
    for i in range(30):
        copy = env.data / f"scan-{i}.h5"
        shutil.copyfile(env.b_path, copy)
        names.append(copy.name)
    env.view.vk.add_target(names)
    before = env.a_path.read_bytes()
    _save_mask_to(editor, env.a_path.with_suffix(""))
    assert env.a_path.read_bytes() == before
