"""The saved partition records where the beam lands on the detector."""

import numpy as np
import pytest

from xpcsviewer.simplemask.simplemask_kernel import SimpleMaskKernel

BASE = {
    "bcx": 100,
    "bcy": 50,
    "det_dist": 5000.0,
    "pix_dim": 0.075,
    "energy": 10.0,
}


def _partition(**extra):
    kernel = SimpleMaskKernel()
    kernel.read_data(np.ones((100, 200), dtype=np.float64), BASE | extra)
    return kernel.compute_partition()


def test_unswung_partition_keeps_the_metadata_centre():
    partition = _partition()
    assert (partition["beam_center_x"], partition["beam_center_y"]) == (100, 50)


def test_swung_partition_saves_the_swing_shifted_centre():
    partition = _partition(swing_angle_horizontal=0.5, swing_angle_vertical=-0.25)
    shift_x = 5000.0 * np.tan(np.deg2rad(0.5)) / 0.075
    shift_y = 5000.0 * np.tan(np.deg2rad(-0.25)) / 0.075
    assert partition["beam_center_x"] == pytest.approx(100 + shift_x, abs=1e-9)
    assert partition["beam_center_y"] == pytest.approx(50 + shift_y, abs=1e-9)
