"""R5-01: the single write gate every exporter goes through."""

import os

import pytest

from xpcsviewer.utils.export_guard import ExportCollisionError, ExportGuard

pytestmark = pytest.mark.unit


def test_plain_destinations_are_allowed(tmp_path):
    guard = ExportGuard([tmp_path / "in.h5"])
    assert guard.claim(tmp_path / "a.txt") == str(tmp_path / "a.txt")
    assert guard.claim(tmp_path / "b.txt") == str(tmp_path / "b.txt")


def test_same_destination_twice_in_one_run_is_refused(tmp_path):
    guard = ExportGuard()
    guard.claim(tmp_path / "a.txt", "file A")
    with pytest.raises(ExportCollisionError, match="file A"):
        guard.claim(tmp_path / "a.txt", "file B")


def test_a_new_run_may_overwrite_an_earlier_runs_output(tmp_path):
    out = tmp_path / "a.txt"
    ExportGuard().claim(out)
    out.write_text("first run")
    assert ExportGuard().claim(out) == str(out)


def test_destination_that_is_an_input_is_refused(tmp_path):
    src = tmp_path / "scan.h5"
    src.write_bytes(b"data")
    with pytest.raises(ExportCollisionError, match="input file"):
        ExportGuard([src]).claim(src)


def test_symlink_to_an_input_is_refused(tmp_path):
    src = tmp_path / "scan.h5"
    src.write_bytes(b"data")
    link = tmp_path / "out.png"
    try:
        link.symlink_to(src)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(ExportCollisionError, match="input file"):
        ExportGuard([src]).claim(link)
    assert src.read_bytes() == b"data"


def test_two_symlinks_to_one_output_are_one_destination(tmp_path):
    shared = tmp_path / "shared.txt"
    shared.write_text("x")
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    try:
        a.symlink_to(shared)
        b.symlink_to(shared)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    guard = ExportGuard()
    guard.claim(a, "file A")
    with pytest.raises(ExportCollisionError):
        guard.claim(b, "file B")


def test_hard_link_to_an_input_is_refused(tmp_path):
    src = tmp_path / "scan.h5"
    src.write_bytes(b"data")
    link = tmp_path / "out.txt"
    try:
        os.link(src, link)
    except (OSError, NotImplementedError):
        pytest.skip("hard links unavailable")
    with pytest.raises(ExportCollisionError, match="input file"):
        ExportGuard([src]).claim(link)
