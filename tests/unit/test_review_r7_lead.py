"""Round 7 leftovers closed by the lead: the placeholder-Q marker in Bayesian exports."""

import csv
import inspect

import numpy as np
import pytest

from xpcsviewer.fitting.viz import export_bayesian_csv

pytestmark = pytest.mark.unit


def _rows(tmp_path, **kw):
    path = tmp_path / "p.csv"
    fit_val = np.ones((2, 2, 4))
    export_bayesian_csv(path, fit_val, np.array([0.1, 0.2]), "single", **kw)
    with path.open() as f:
        return list(csv.reader(f))


def test_r7_10_bayesian_csv_marks_a_placeholder_q_map(tmp_path):
    plain = _rows(tmp_path)
    marked = _rows(tmp_path, placeholder_q=True)

    assert marked[0] == [*plain[0], "q_map_placeholder"]  # appended, nothing moved
    assert [r[:-1] for r in marked[1:]] == plain[1:]
    assert {r[-1] for r in marked[1:]} == {"1"}


def test_r7_10_viewer_and_cli_pass_the_placeholder_flag():
    from xpcsviewer import xpcs_viewer
    from xpcsviewer.cli import twotime_batch

    assert inspect.getsource(xpcs_viewer).count("placeholder_q=placeholder") == 2
    assert "has_placeholder_qmap(xfile)" in inspect.getsource(twotime_batch)
