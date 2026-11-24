"""The wheel is published as ``xpcsviewer-gui``; version lookups must use that name."""

from __future__ import annotations

import importlib.metadata
from importlib.metadata import PackageNotFoundError

import numpy as np
import pytest

import xpcsviewer
from xpcsviewer.fitting import results
from xpcsviewer.utils.logging_config import LoggingConfig


def test_looks_up_gui_distribution():
    calls: list[str] = []

    def lookup(name: str) -> str:
        calls.append(name)
        return "9.8.7"

    assert xpcsviewer._package_version(lookup) == "9.8.7"
    assert calls == ["xpcsviewer-gui"]


def test_missing_distribution_does_not_claim_a_release():
    def lookup(name: str) -> str:
        raise PackageNotFoundError(name)

    assert xpcsviewer._package_version(lookup) == "0+unknown"


def test_logging_banner_uses_gui_distribution(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    def recorder(name: str) -> str:
        calls.append(name)
        return "1.2.3"

    monkeypatch.setattr(importlib.metadata, "version", recorder)
    assert LoggingConfig.__new__(LoggingConfig)._get_app_version() == "1.2.3"
    assert calls == ["xpcsviewer-gui"]


def test_logging_banner_falls_back_when_metadata_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    def recorder(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", recorder)
    assert LoggingConfig.__new__(LoggingConfig)._get_app_version() == "0+unknown"


def test_reproducibility_metadata_names_gui_distribution(
    monkeypatch: pytest.MonkeyPatch,
):
    names: list[str] = []

    def recorder(name: str) -> str:
        names.append(name)
        return "1.0"

    monkeypatch.setattr(results, "safe_version", recorder)
    out = results.FitResult(samples={"a": np.zeros(3)}).to_dict()
    assert "xpcsviewer-gui" in names
    assert "xpcs-toolkit" not in names
    assert out["versions"]["xpcsviewer"] == "1.0"
