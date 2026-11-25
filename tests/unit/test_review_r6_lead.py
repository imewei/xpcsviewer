"""Round 6 leftovers closed by the lead: units and axes at the viewer boundary."""

import inspect

import numpy as np
import pytest

from tests.unit.test_review_r4_viewer_gui import (  # noqa: F401  (fixtures)
    g2_env,
    viewer_env,
)
from tests.unit.test_review_r5_bayes_gui import _set_q, diff_env  # noqa: F401
from xpcsviewer.threading import plot_workers

pytestmark = pytest.mark.unit


def test_r6_10_bayesian_diffusion_rows_are_in_inverse_angstrom(diff_env):  # noqa: F811
    view, xf = diff_env.view, diff_env.xf
    xf.qmap.map_units = ["nm^-1", "deg"]  # Q = (0.2, 0.3, 1.0) nm^-1
    _set_q(view, 0.025, 0.11)  # the boxes are in inverse angstrom

    bundle = view._extract_diffusion_for_bayesian()

    np.testing.assert_allclose(bundle.x, [0.03, 0.1])


def test_r6_10_bayesian_diffusion_refuses_a_pixel_axis(diff_env):  # noqa: F811
    view, xf = diff_env.view, diff_env.xf
    xf.qmap.map_units = ["px", "px"]
    _set_q(view, 0.1, 1.1)

    assert view._extract_diffusion_for_bayesian() is None
    assert "refused" in view.statusbar.currentMessage()


def test_r6_10_async_g2_fit_converts_the_q_range_per_file():
    src = inspect.getsource(plot_workers.G2PlotWorker._perform_g2_fitting)
    assert "q_range_in_file_unit" in src
    assert "fit_g2(q_range" not in src


def test_r6_17_viewer_passes_the_bin_axes_to_every_bayesian_writer():
    from xpcsviewer import xpcs_viewer

    src = inspect.getsource(xpcs_viewer)
    assert src.count('.get("bin_phi")') == src.count('.get("bin_axes")') == 3
    assert '"bin_axes": resolve_bin_axes(' in src
