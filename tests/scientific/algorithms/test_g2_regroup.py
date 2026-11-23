"""Scientific checks for ROI-based G2 regrouping."""

import numpy as np
import pytest

from xpcsviewer.module.g2_regroup import G2Regrouping

pytestmark = pytest.mark.scientific


def test_regroup_delay_uses_static_bin_weighting_and_pixel_error():
    static = np.array([[1, 1, 2, 3]])
    dynamic = np.array([[1, 1, 1, 2]])
    channels = np.array([[[4, 12, 24, 10]], [[1, 2, 2, 1]], [[2, 3, 2, 5]]])
    grouping = G2Regrouping(static, dynamic)
    g2, error = grouping.compute_delay(channels)
    np.testing.assert_array_equal(grouping.dynamic_bin_ids, [1, 2])
    np.testing.assert_allclose(g2, [61 / 15, 2], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(error, [4 * np.sqrt(6) / 9, 0], rtol=1e-12, atol=1e-12)
    assert g2.dtype == error.dtype == np.float64


def test_regroup_sparse_unsigned_ids_preserve_order():
    static = np.array([[1, 1, 2, 3]])
    dynamic = np.array([[17, 17, 17, 2**63 + 1]], dtype=np.uint64)
    channels = np.array([[[4, 12, 24, 10]], [[1, 2, 2, 1]], [[2, 3, 2, 5]]])
    grouping = G2Regrouping(static, dynamic)
    g2, error = grouping.compute_delay(channels)
    np.testing.assert_array_equal(
        grouping.dynamic_bin_ids, np.array([17, 2**63 + 1], dtype=np.uint64)
    )
    assert grouping.dynamic_bin_ids.dtype == dynamic.dtype
    assert g2.shape == error.shape == (2,)
    np.testing.assert_allclose(g2, [61 / 15, 2], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(error, [4 * np.sqrt(6) / 9, 0], rtol=1e-12, atol=1e-12)

    masked_static = np.pad(static, ((0, 0), (0, 1)))
    masked_dynamic = np.pad(dynamic, ((0, 0), (0, 1)))
    masked_channels = np.concatenate(
        (channels, np.array([[[np.nan]], [[np.inf]], [[0]]])), axis=2
    )
    grouping = G2Regrouping(masked_static, masked_dynamic)
    masked_g2, masked_error = grouping.compute_delay(masked_channels)
    np.testing.assert_array_equal(masked_g2, g2)
    np.testing.assert_array_equal(masked_error, error)


@pytest.mark.parametrize(
    ("static", "dynamic"),
    [
        ([[1, 1]], [[1], [1]]),
        ([[1.0, 1.0]], [[1, 1]]),
        ([[1, -1]], [[1, 0]]),
        ([[1, 0]], [[1, 1]]),
        ([[0, 0]], [[0, 0]]),
        ([[1, 1]], [[1, 2]]),
    ],
    ids=["shape", "dtype", "negative", "support", "empty", "split"],
)
def test_regroup_rejects_invalid_roi_maps(static, dynamic):
    with pytest.raises(ValueError):
        G2Regrouping(np.array(static), np.array(dynamic))


@pytest.mark.parametrize(
    ("ip", "if_"),
    [
        ([0, 2], [2, 3]),
        ([-1, 2], [2, 3]),
        ([np.nan, 2], [2, 3]),
        ([1, 2], [np.inf, 3]),
        ([1e308, 2], [1e308, 3]),
        ([1, -3], [3, -1]),
    ],
    ids=["zero", "negative", "nan", "infinity", "overflow", "static-mean-product"],
)
def test_regroup_rejects_invalid_active_channels(ip, if_):
    grouping = G2Regrouping(np.array([[1, 1]]), np.array([[1, 1]]))
    channels = np.array([[[4, 12]], [ip], [if_]])
    with pytest.raises(ValueError):
        grouping.compute_delay(channels)


def test_regroup_rejects_wrong_channel_count():
    grouping = G2Regrouping(np.array([[1, 1]]), np.array([[1, 1]]))
    with pytest.raises(ValueError):
        grouping.compute_delay(np.ones((2, 1, 2)))


def test_regroup_rejects_nonfinite_normalized_results():
    grouping = G2Regrouping(np.array([[1]]), np.array([[1]]))
    with pytest.raises(ValueError):
        grouping.compute_delay(np.array([[[1e308]], [[0.1]], [[1]]]))


@pytest.mark.parametrize("amplitude", [1e155, 1e-200])
def test_regroup_standard_error_preserves_float64_range(amplitude):
    grouping = G2Regrouping(np.array([[1, 2]]), np.array([[1, 1]]))
    channels = np.array([[[0, amplitude]], [[1, 1]], [[1, 1]]])

    g2, error = grouping.compute_delay(channels)

    np.testing.assert_allclose(g2, [amplitude / 2], rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        error, [amplitude / (2 * np.sqrt(2))], rtol=1e-12, atol=0
    )


def test_regroup_constant_pixel_ratios_have_exactly_zero_error():
    roi = np.ones((32, 32), dtype=int)
    channels = np.ones((3, 32, 32))
    channels[0] = 1e20

    _, error = G2Regrouping(roi, roi).compute_delay(channels)

    np.testing.assert_array_equal(error, [0])


def test_regroup_nearly_equal_pixel_ratios_preserve_small_error():
    roi = np.ones((32, 32), dtype=int)
    channels = np.ones((3, 32, 32))
    lower = 10000.1
    upper = lower + 2 * np.spacing(lower)
    channels[0, :16] = lower
    channels[0, 16:] = upper

    _, error = G2Regrouping(roi, roi).compute_delay(channels)

    expected = (upper - lower) / (2 * np.sqrt(roi.size))
    np.testing.assert_allclose(error, [expected], rtol=1e-12, atol=0)


def test_regroup_static_mean_survives_sum_overflow():
    grouping = G2Regrouping(np.array([[1, 1]]), np.array([[1, 1]]))
    channels = np.array([[[1, 1]], [[1e308, 1e308]], [[1e-308, 1e-308]]])

    g2, error = grouping.compute_delay(channels)

    np.testing.assert_allclose(g2, [1], rtol=1e-12, atol=0)
    np.testing.assert_array_equal(error, [0])


@pytest.mark.parametrize("sign", [1, -1])
def test_regroup_dynamic_mean_and_error_survive_intermediate_overflow(sign):
    grouping = G2Regrouping(np.array([[1, 2, 3, 4]]), np.array([[1, 1, 1, 1]]))
    amplitude = sign * 1.7e308
    channels = np.array(
        [
            [[amplitude, amplitude, amplitude, -amplitude]],
            [[1, 1, 1, 1]],
            [[1, 1, 1, 1]],
        ]
    )

    g2, error = grouping.compute_delay(channels)

    np.testing.assert_allclose(g2, [amplitude / 2], rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        error, [abs(amplitude / 4) * np.sqrt(3)], rtol=1e-12, atol=0
    )


def test_regroup_matches_upstream_positive_domain():
    static = np.array([[1, 1], [2, 3]])
    dynamic = np.array([[1, 1], [1, 2]])
    channels = np.array([[[4, 12], [24, 10]], [[1, 2], [2, 1]], [[2, 3], [2, 5]]])
    second_delay = channels.copy()
    second_delay[0] += 2

    grouping = G2Regrouping(static, dynamic)
    g2, error = zip(
        *(grouping.compute_delay(delay) for delay in (channels, second_delay))
    )

    # Pinned upstream compute_g2() at 952a0bf4daaf84e6c8b718c3c7f5620d28381c49.
    np.testing.assert_allclose(
        g2, [[4.066666666666666, 2.0], [4.583333333333333, 2.4]], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        error,
        [[1.0886621079036347, 0.0], [1.0550681135736384, 0.0]],
        rtol=1e-12,
        atol=1e-12,
    )
