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


BOUNDS = np.array([[1e-8, -4.0], [1e-1, 0.0]])


def _assert_recovers_true_law(out: dict, q_kept: np.ndarray) -> None:
    # tau = 1e-3 * q**-2 on the kept samples, so a == 1e-3 with b fixed at -2.
    assert out["tauq_success"]  # np.bool_
    assert out["tauq_fit_val"][0, 0] == pytest.approx(1e-3, rel=1e-6)
    np.testing.assert_array_equal(out["tauq_q"], q_kept)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_fit_tauq_drops_nonpositive_tau(bad: float) -> None:
    tau = 1.0e-3 / Q**2
    tau[2] = bad
    stub = _stub(tau)
    # b is held fixed, so it needs a finite starting value.
    out = XpcsFile.fit_tauq(stub, (0.0, 1.0), BOUNDS, np.array([True, False]))
    assert out is stub.fit_summary
    _assert_recovers_true_law(out, np.delete(Q, 2))


@pytest.mark.parametrize("bad", [0.0, -0.01, np.inf])
def test_fit_tauq_drops_nonpositive_or_infinite_q(bad: float) -> None:
    q = Q.copy()
    tau = 1.0e-3 / q**2
    q[2], tau[2] = bad, 5.0  # finite, positive tau at an unusable q
    stub = _stub(tau)
    stub.fit_summary["q_val"] = q
    out = XpcsFile.fit_tauq(stub, (-1.0, np.inf), BOUNDS, np.array([True, False]))
    _assert_recovers_true_law(out, np.delete(Q, 2))


def test_fit_tauq_all_positive_input_unchanged() -> None:
    out = XpcsFile.fit_tauq(
        _stub(1.0e-3 / Q**2), (0.0, 1.0), BOUNDS, np.array([True, False])
    )
    _assert_recovers_true_law(out, Q)


@pytest.mark.parametrize(
    ("q", "tau"),
    [([0.5, 0.5, 2.0], [1.0, 2.0, 0.0]), ([1.0, 1.0], [1.0, 2.0])],
)
def test_fit_tauq_single_distinct_q_fails_when_both_parameters_free(q, tau) -> None:
    fit_val = np.zeros((len(q), 2, 2))
    fit_val[:, 0, 1], fit_val[:, 1, 1] = tau, 0.01
    stub = _stub(np.asarray(tau))
    stub.fit_summary = {"q_val": np.asarray(q), "fit_val": fit_val}
    out = XpcsFile.fit_tauq(stub, (0.0, 10.0), BOUNDS, np.array([True, True]))
    assert not out["tauq_success"]


def test_fit_tauq_three_distinct_q_with_both_free_still_succeeds() -> None:
    out = XpcsFile.fit_tauq(
        _stub(1.0e-3 / Q**2), (0.0, 1.0), BOUNDS, np.array([True, True])
    )
    assert out["tauq_success"]
    assert out["tauq_fit_val"][0, 1] == pytest.approx(-2.0, rel=1e-3)


def test_fit_tauq_single_distinct_q_with_one_parameter_free_still_fits() -> None:
    q = np.array([0.5, 0.5])
    fit_val = np.zeros((2, 2, 2))
    fit_val[:, 0, 1], fit_val[:, 1, 1] = 1.0e-3 / q**2, 0.01
    stub = _stub(np.zeros(2))
    stub.fit_summary = {"q_val": q, "fit_val": fit_val}
    out = XpcsFile.fit_tauq(stub, (0.0, 10.0), BOUNDS, np.array([True, False]))
    assert out["tauq_success"]
