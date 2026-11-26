"""One write gate for every export run.

An export must never write through a path that is really one of its own input
files, or really a file it already wrote earlier in the same run (two names can
reach one file through symlinks or hard links). Every exporter creates one
``ExportGuard`` per run and calls ``claim`` before each write.

A separate run into the same folder overwrites earlier output by design: the
guard only remembers what *this* run wrote.
"""

from __future__ import annotations

import os
from collections.abc import Iterable


class ExportCollisionError(ValueError):
    """The destination is an input file or an earlier output of this run."""


def _real(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.realpath(os.fspath(path)))


def _same_file(a: str, b: str) -> bool:
    """True for one file reached by two names (also catches hard links)."""
    try:
        return os.path.samefile(a, b)
    except OSError:  # one of them does not exist yet
        return False


def same_destination(a: str | os.PathLike[str], b: str | os.PathLike[str]) -> bool:
    """True when two output paths name one file (links, hard links, ``..``)."""
    a, b = os.fspath(a), os.fspath(b)
    return _real(a) == _real(b) or _same_file(a, b)


class ExportGuard:
    """Refuses destinations that alias an input or an earlier output."""

    def __init__(self, input_paths: Iterable[str | os.PathLike[str]] = ()) -> None:
        self._inputs = {_real(p): os.fspath(p) for p in input_paths}
        self._written: dict[str, str] = {}

    def claim(self, path: str | os.PathLike[str], source: str = "") -> str:
        """Register ``path`` as a destination of this run and return it.

        Raises ``ExportCollisionError`` if it resolves to an input file or to a
        destination already claimed in this run. ``source`` names what is being
        written, for the error message.
        """
        path = os.fspath(path)
        key = _real(path)
        for real_input, given in self._inputs.items():
            if key == real_input or _same_file(path, given):
                raise ExportCollisionError(
                    f"refusing to write {path}: it is the input file {given}"
                )
        for real_out, earlier in self._written.items():
            if key == real_out or _same_file(path, real_out):
                raise ExportCollisionError(
                    f"refusing to write {path}: already written in this run "
                    f"by {earlier or 'an earlier export'}"
                    + (f", now requested by {source}" if source else "")
                )
        self._written[key] = source
        return path
