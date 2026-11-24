"""fit_tauq must derive starting values from usable samples only."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from xpcsviewer.xpcs_file import XpcsFile

Q = np.linspace(0.01, 0.05, 6)


def _stub(tau: np.ndarray) -> SimpleNamespace:
    fit_val = np.zeros((tau.size, 2, 2))
    fit_val[:, 0, 1] = tau
    fit_val[:, 1, 1] = 0.01
    memory = SimpleNamespace(
        cache_get=lambda *a, **k: None, cache_put=lambda *a, **k: None
    )
    return SimpleNamespace(
        fit_summary={"q_val": Q, "fit_val": fit_val},
        _generate_cache_key=lambda *a, **k: "k",
        _cache_prefix="t",
        _memory_manager=memory,
    )


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_fit_tauq_ignores_nonpositive_tau_when_choosing_start(bad: float) -> None:
    tau = 1.0e-3 / Q**2
    tau[2] = bad
    stub = _stub(tau)
    bounds = np.array([[1e-8, -4.0], [1e-1, 0.0]])
    # b is held fixed, so it needs a finite starting value.
    out = XpcsFile.fit_tauq(stub, (0.0, 1.0), bounds, np.array([True, False]))
    assert out is stub.fit_summary
    assert "tauq_success" in out
