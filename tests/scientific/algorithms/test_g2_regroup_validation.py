"""Scientific release check for representative multi-tau G2 delays."""

import re
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from tests.integration.test_g2_regroup_hdf import CHANNELS, CUBE_PATH, DELAY_PATH
from tests.integration.test_g2_regroup_hdf import qmap_path as qmap_path
from tests.integration.test_g2_regroup_hdf import source_path as source_path

pytestmark = [
    pytest.mark.scientific,
    pytest.mark.skipif(
        sys.platform not in {"linux", "darwin"}, reason="requires peak RSS support"
    ),
]

SCRIPT = (
    Path(__file__).resolve().parents[3] / "scripts/validation/validate_g2_regroup.py"
)


def test_validation_reports_agreement_and_peak_rss(source_path, qmap_path):
    command = [sys.executable, str(SCRIPT), str(source_path), str(qmap_path)]
    valid = subprocess.run(command, capture_output=True, text=True, check=False)

    assert valid.returncode == 0, valid.stderr
    assert "Detector shape: (1, 4)" in valid.stdout
    assert "Compared 3 delays and 2 bins" in valid.stdout
    for metric in (
        "Peak RSS",
        "Max absolute G2 difference",
        "Max absolute error difference",
    ):
        match = re.search(rf"^{metric}: (\S+)", valid.stdout, re.MULTILINE)
        assert match, valid.stdout
        value = float(match[1])
        assert np.isfinite(value) and value >= 0

    with h5py.File(source_path, "r+") as source:
        source[CUBE_PATH][2, 1, 0, 0] = 0

    invalid = subprocess.run(command, capture_output=True, text=True, check=False)
    assert invalid.returncode != 0
    assert "delay index 2" in invalid.stderr


def test_validation_compares_longest_delay_when_storage_is_unsorted(
    source_path, qmap_path
):
    with h5py.File(source_path, "r+") as source:
        del source[DELAY_PATH]
        source[DELAY_PATH] = [0, 8, 2, 1, 4]
        del source[CUBE_PATH]
        source[CUBE_PATH] = np.stack([CHANNELS * scale for scale in range(1, 6)])

    run = subprocess.run(
        [sys.executable, str(SCRIPT), str(source_path), str(qmap_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert run.returncode == 0, run.stderr
    assert "Compared 4 delays and 2 bins" in run.stdout
    assert "[0, 1, 2, 4]" in run.stdout


@pytest.mark.parametrize("metric", [0, 1], ids=["g2", "error"])
def test_validation_rejects_mismatched_reference(
    source_path, qmap_path, monkeypatch, capsys, metric
):
    from scripts.validation import validate_g2_regroup

    reference_delay = validate_g2_regroup._reference_delay

    def mismatched_reference(*args):
        reference = reference_delay(*args)
        reference[metric][0] += 0.25
        return reference

    monkeypatch.setattr(validate_g2_regroup, "_reference_delay", mismatched_reference)
    assert validate_g2_regroup.main([str(source_path), str(qmap_path)]) != 0
    assert "Numerical mismatch" in capsys.readouterr().err
