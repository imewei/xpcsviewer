"""Draw and parameter edits replace their own layer in MaskAssemble."""

import numpy as np

from xpcsviewer.simplemask.area_mask import MaskAssemble

SHAPE = (6, 8)
# phi grows by 10 deg per column: 0, 10, ..., 70.
PHI = np.tile(np.arange(SHAPE[1], dtype=float) * 10.0, (SHAPE[0], 1))


def excluded(mask):
    return {(int(r), int(c)) for r, c in zip(*np.nonzero(~mask), strict=True)}


def draw(row, col):
    arr = np.zeros(SHAPE, dtype=bool)
    arr[row, col] = True
    return arr


def apply_draw(ma, row, col):
    ma.evaluate("mask_draw", arr=draw(row, col))
    return ma.apply("mask_draw")


def apply_list(ma, row, col):
    ma.evaluate("mask_list", zero_loc=np.array([[row], [col]]))
    return ma.apply("mask_list")


def test_second_draw_replaces_first():
    ma = MaskAssemble(SHAPE)
    apply_draw(ma, 1, 1)
    out = apply_draw(ma, 4, 4)
    assert excluded(out) == {(4, 4)}
    assert excluded(ma.get_mask()) == {(4, 4)}


def test_draw_edit_preserves_unrelated_layers():
    ma = MaskAssemble(SHAPE)
    apply_list(ma, 0, 0)
    apply_draw(ma, 1, 1)
    out = apply_draw(ma, 4, 4)
    assert excluded(out) == {(0, 0), (4, 4)}


def test_parameter_edit_replaces_parameter_only():
    ma = MaskAssemble(SHAPE, qmap={"phi": PHI})
    apply_draw(ma, 1, 1)
    # P1 keeps columns 0-3, P2 keeps columns 4-7: neither contains the other.
    ma.evaluate("mask_parameter", constraints=[("phi", "AND", "deg", 0.0, 30.0)])
    ma.apply("mask_parameter")
    ma.evaluate("mask_parameter", constraints=[("phi", "AND", "deg", 40.0, 70.0)])
    out = ma.apply("mask_parameter")
    p2 = {(r, c) for r in range(SHAPE[0]) for c in range(4)}
    assert excluded(out) == {(1, 1)} | p2


def test_non_replacing_targets_still_accumulate():
    ma = MaskAssemble(SHAPE)
    apply_list(ma, 0, 0)
    out = apply_list(ma, 2, 3)
    assert excluded(out) == {(0, 0), (2, 3)}


def test_undo_restores_previous_draw_layer():
    ma = MaskAssemble(SHAPE)
    apply_draw(ma, 1, 1)
    apply_draw(ma, 4, 4)
    ma.redo_undo("undo")
    assert excluded(ma.get_mask()) == {(1, 1)}
    ma.redo_undo("redo")
    assert excluded(ma.get_mask()) == {(4, 4)}
    ma.redo_undo("undo")
    apply_draw(ma, 2, 5)
    assert excluded(ma.get_mask()) == {(2, 5)}
    ptr = ma.mask_ptr
    ma.redo_undo("redo")
    assert ma.mask_ptr == ptr
    assert excluded(ma.get_mask()) == {(2, 5)}


def test_reset_clears_every_layer():
    ma = MaskAssemble(SHAPE, qmap={"phi": PHI})
    apply_list(ma, 0, 0)
    apply_draw(ma, 1, 1)
    ma.evaluate("mask_parameter", constraints=[("phi", "AND", "deg", 0.0, 30.0)])
    ma.apply("mask_parameter")
    ma.redo_undo("reset")
    assert len(ma.mask_record) == 1
    assert len(ma.layer_record) == 1
    assert ma.get_mask().all()
    assert set(ma._layers_at(ma.mask_ptr)) == {"accumulated"}


def test_hidden_draw_edit_is_recorded():
    ma = MaskAssemble(SHAPE)
    apply_list(ma, 1, 1)
    a = draw(1, 1)
    ma.evaluate("mask_draw", arr=a)
    before = len(ma.mask_record)
    ma.apply("mask_draw")
    assert len(ma.mask_record) == before + 1
    apply_draw(ma, 4, 4)
    ma.redo_undo("undo")
    np.testing.assert_array_equal(ma._layers_at(ma.mask_ptr)["mask_draw"], ~a)


def test_redo_branch_discarded_after_new_edit():
    ma = MaskAssemble(SHAPE)
    apply_draw(ma, 1, 1)
    apply_draw(ma, 4, 4)
    ma.redo_undo("undo")
    apply_draw(ma, 2, 5)
    assert len(ma.mask_record) == len(ma.layer_record) == 3
    ptr = ma.mask_ptr
    ma.redo_undo("redo")
    assert ma.mask_ptr == ptr


def test_history_trim_keeps_layer_record_aligned():
    ma = MaskAssemble(SHAPE, max_history=4)
    draws = [(0, i) for i in range(1, 7)]
    for row, col in draws:
        apply_draw(ma, row, col)
    assert len(ma.mask_record) == 4
    assert len(ma.layer_record) == len(ma.mask_record)
    assert ma.mask_record[0].all()
    for k, rec in zip((4, 5, 6), ma.mask_record[1:], strict=True):
        assert excluded(rec) == {draws[k - 1]}
    ma.redo_undo("undo")
    assert excluded(ma.get_mask()) == {draws[4]}
    ma.redo_undo("undo")
    assert excluded(ma.get_mask()) == {draws[3]}
    ma.redo_undo("undo")
    assert ma.get_mask().all()
    ma.redo_undo("undo")
    assert ma.mask_ptr == 0
    assert ma.get_mask().all()


def test_history_cap_below_two_keeps_baseline_and_current():
    ma = MaskAssemble(SHAPE, max_history=1)
    out = apply_draw(ma, 1, 1)
    assert excluded(out) == {(1, 1)}
    assert len(ma.mask_record) == 2
    assert ma.mask_record[0].all()
    assert excluded(ma.mask_record[1]) == {(1, 1)}
    ma.redo_undo("undo")
    assert ma.get_mask().all()


def test_reset_after_trim_restores_blank_baseline():
    ma = MaskAssemble(SHAPE, max_history=4)
    for col in range(1, 7):
        apply_draw(ma, 0, col)
    ma.redo_undo("reset")
    assert len(ma.mask_record) == 1
    assert ma.get_mask().all()


def test_direct_mask_record_mutation_does_not_break_apply():
    ma = MaskAssemble(SHAPE)
    extra = np.ones(SHAPE, dtype=bool)
    extra[2, 2] = False
    ma.mask_record.append(extra)
    ma.mask_ptr = 1
    out = apply_draw(ma, 1, 1)
    assert excluded(out) == {(2, 2), (1, 1)}
    assert len(ma.layer_record) == len(ma.mask_record) == 3
    np.testing.assert_array_equal(ma._layers_at(1)["accumulated"], extra)
    ma.redo_undo("reset")
    assert len(ma.mask_record) == 1
    assert ma.get_mask().all()
