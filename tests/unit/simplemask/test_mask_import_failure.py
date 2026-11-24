"""Mask import must report failure when the file is rejected."""

import h5py
import numpy as np
import pytest

from xpcsviewer.simplemask import SimpleMaskWindow, simplemask_window
from xpcsviewer.simplemask.simplemask_kernel import SimpleMaskKernel

SHAPE = (6, 8)
METADATA = {
    "bcx": 4.0,
    "bcy": 3.0,
    "det_dist": 5000.0,
    "pix_dim": 0.075,
    "energy": 10.0,
}


def _kernel():
    kernel = SimpleMaskKernel()
    kernel.read_data(np.ones(SHAPE), dict(METADATA))
    return kernel


def _write(path, data, key="mask"):
    with h5py.File(path, "w") as f:
        f[key] = data
    return str(path)


def _state(kernel):
    return (
        kernel.mask.copy(),
        kernel.mask_kernel.mask_ptr,
        len(kernel.mask_kernel.mask_record),
    )


def test_wrong_shape_returns_false_and_changes_nothing(tmp_path):
    kernel = _kernel()
    before = _state(kernel)
    assert kernel.load_mask(_write(tmp_path / "m.h5", np.zeros((5, 7)))) is False
    after = _state(kernel)
    np.testing.assert_array_equal(after[0], before[0])
    assert after[1:] == before[1:]


def test_missing_file_returns_false(tmp_path):
    assert _kernel().load_mask(str(tmp_path / "absent.h5")) is False


def test_wrong_key_returns_false(tmp_path):
    kernel = _kernel()
    assert (
        kernel.load_mask(_write(tmp_path / "m.h5", np.zeros(SHAPE), "other")) is False
    )


def test_non_hdf5_extension_returns_false(tmp_path):
    path = tmp_path / "mask.txt"
    path.write_text("not a mask")
    assert _kernel().load_mask(str(path)) is False


def test_transposed_mask_is_still_accepted(tmp_path):
    kernel = _kernel()
    data = np.ones((8, 6), dtype=np.uint8)
    data[7, 2] = 0
    assert kernel.load_mask(_write(tmp_path / "m.h5", data)) is True
    assert not kernel.mask[2, 7]
    assert kernel.mask.sum() == kernel.mask.size - 1


@pytest.mark.parametrize("data", [np.float64(0.0), np.zeros(48)])
def test_scalar_and_1d_datasets_return_false(tmp_path, data):
    kernel = _kernel()
    before = _state(kernel)
    assert kernel.load_mask(_write(tmp_path / "m.h5", data)) is False
    assert _state(kernel)[1:] == before[1:]


def test_matching_mask_returns_true(tmp_path):
    kernel = _kernel()
    data = np.ones(SHAPE, dtype=np.uint8)
    data[1, 1] = 0
    assert kernel.load_mask(_write(tmp_path / "m.h5", data)) is True
    assert not kernel.mask[1, 1]


@pytest.fixture
def window(qtbot, monkeypatch):
    win = SimpleMaskWindow()
    qtbot.addWidget(win)
    win.load_from_viewer(np.ones(SHAPE), dict(METADATA))
    calls = []
    monkeypatch.setattr(
        simplemask_window.QMessageBox,
        "warning",
        lambda *a, **k: calls.append("warning"),
    )
    monkeypatch.setattr(
        simplemask_window.QMessageBox,
        "critical",
        lambda *a, **k: calls.append("critical"),
    )
    win.dialog_calls = calls
    return win


def _pick(monkeypatch, path):
    monkeypatch.setattr(
        simplemask_window.QFileDialog,
        "getOpenFileName",
        lambda *a, **k: (str(path), ""),
    )


def test_window_reports_failure_without_dialogs(window, tmp_path, monkeypatch):
    bad_text = tmp_path / "notes.h5"
    bad_text.write_text("not hdf5")
    cases = [
        _write(tmp_path / "wrong_shape.h5", np.zeros((5, 7))),
        _write(tmp_path / "no_dataset.h5", np.zeros(SHAPE), "other"),
        str(bad_text),
    ]
    before = window.kernel.mask.copy()
    for path in cases:
        _pick(monkeypatch, path)
        window._on_load_mask()
        text = window.status_bar.currentMessage()
        assert tmp_path.joinpath(path).name in text
        assert "failed" in text.lower()
        np.testing.assert_array_equal(window.kernel.mask, before)
    assert window.dialog_calls == []


def test_window_loads_transposed_mask_without_prompt(window, tmp_path, monkeypatch):
    data = np.ones((8, 6), dtype=np.uint8)
    data[7, 2] = 0
    _pick(monkeypatch, _write(tmp_path / "t.h5", data))
    window._on_load_mask()
    assert window.dialog_calls == []
    assert "Loaded mask" in window.status_bar.currentMessage()
    assert not window.kernel.mask[2, 7]


def _write_typed(path, kind):
    with h5py.File(path, "w") as f:
        if kind == "bytes":
            f["mask"] = np.full(SHAPE, b"1", dtype="S1")
        elif kind == "vlen-str":
            f.create_dataset("mask", SHAPE, dtype=h5py.string_dtype())
        else:
            f["mask"] = np.ones(SHAPE, dtype=complex)
    return str(path)


@pytest.mark.parametrize("kind", ["bytes", "vlen-str", "complex"])
def test_unsupported_dtype_returns_false(tmp_path, kind):
    kernel = _kernel()
    before = _state(kernel)
    assert kernel.load_mask(_write_typed(tmp_path / "m.h5", kind)) is False
    after = _state(kernel)
    np.testing.assert_array_equal(after[0], before[0])
    assert after[1:] == before[1:]


@pytest.mark.parametrize("name", ["M.H5", "m.HDF5"])
def test_upper_case_extension_is_accepted(tmp_path, name):
    data = np.ones(SHAPE, dtype=np.uint8)
    data[1, 1] = 0
    kernel = _kernel()
    assert kernel.load_mask(_write(tmp_path / name, data)) is True
    assert not kernel.mask[1, 1]


def test_window_reports_string_mask_failure(window, tmp_path, monkeypatch):
    _pick(monkeypatch, _write_typed(tmp_path / "strings.h5", "vlen-str"))
    window._on_load_mask()
    assert "Failed to load mask from strings.h5" in window.status_bar.currentMessage()
    assert window.dialog_calls == []
