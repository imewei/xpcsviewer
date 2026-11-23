"""Regroup unnormalized multi-tau G2 channels by ROI membership."""

import numpy as np


def _grouped_mean(
    indices: np.ndarray, values: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    means = np.bincount(indices, weights=values) / counts
    overflow = ~np.isfinite(means)
    if np.any(overflow):
        # Scale by a power of two larger than the group count to keep sums finite.
        shifts = np.frexp(counts)[1]
        scaled = np.ldexp(values, -shifts[indices])
        scaled_means = np.bincount(indices, weights=scaled)[overflow] / counts[overflow]
        shifts = shifts[overflow]
        limit = np.ldexp(np.finfo(np.float64).max, -shifts)
        means[overflow] = np.ldexp(np.clip(scaled_means, -limit, limit), shifts)
    return means


class G2Regrouping:
    """Cache the ROI membership and counts shared by every delay."""

    def __init__(self, static_map: np.ndarray, dynamic_map: np.ndarray) -> None:
        static_map = np.asarray(static_map)
        dynamic_map = np.asarray(dynamic_map)
        if static_map.ndim != 2 or static_map.shape != dynamic_map.shape:
            raise ValueError(
                "Static and dynamic ROI maps must have matching 2-D shapes"
            )
        for roi_map in (static_map, dynamic_map):
            if roi_map.dtype.kind not in "iu" or np.any(roi_map < 0):
                raise ValueError("ROI maps must contain nonnegative integer labels")

        self._included = static_map > 0
        if not np.array_equal(self._included, dynamic_map > 0):
            raise ValueError("Static and dynamic ROI maps must include the same pixels")
        if not np.any(self._included):
            raise ValueError("ROI maps must include at least one pixel")

        _, static_first, self._static_index = np.unique(
            static_map[self._included], return_index=True, return_inverse=True
        )
        self.dynamic_bin_ids, self._dynamic_index = np.unique(
            dynamic_map[self._included], return_inverse=True
        )
        self._static_to_dynamic = self._dynamic_index[static_first]
        if not np.array_equal(
            self._static_to_dynamic[self._static_index], self._dynamic_index
        ):
            raise ValueError("Each static ROI bin must belong to one dynamic ROI bin")

        self._static_pixel_counts = np.bincount(self._static_index)
        self._dynamic_pixel_counts = np.bincount(self._dynamic_index)
        self._dynamic_static_counts = np.bincount(self._static_to_dynamic)

    def compute_delay(self, channels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return dynamic-bin G2 and standard error for a single G2, IP, IF slice."""
        channels = np.asarray(channels)
        expected_shape = (3, *self._included.shape)
        if channels.shape != expected_shape:
            raise ValueError(f"Delay channels must have shape {expected_shape}")
        if channels.dtype.kind not in "iuf":
            raise ValueError("Delay channels must contain real numbers")

        with np.errstate(all="ignore"):
            active = channels[:, self._included].astype(np.float64, copy=False)
            if not np.isfinite(active).all():
                raise ValueError(
                    "Included pixels must have finite G2, IP and IF channels"
                )
            pixel_product = active[1] * active[2]
            if not np.isfinite(pixel_product).all() or np.any(pixel_product <= 0):
                raise ValueError("Included pixels must have finite, positive IP * IF")

            static_means = np.array(
                [
                    _grouped_mean(
                        self._static_index, channel, self._static_pixel_counts
                    )
                    for channel in active
                ]
            )
            static_product = static_means[1] * static_means[2]
            if not np.isfinite(static_product).all() or np.any(static_product <= 0):
                raise ValueError(
                    "Static bins must have finite, positive mean(IP) * mean(IF)"
                )
            static_g2 = static_means[0] / static_product
            pixel_g2 = active[0] / pixel_product
            if not np.isfinite(static_g2).all() or not np.isfinite(pixel_g2).all():
                raise ValueError("Normalized static-bin and pixel G2 must be finite")

            g2 = _grouped_mean(
                self._static_to_dynamic, static_g2, self._dynamic_static_counts
            )

            # Center and square in a bounded range before rescaling the error.
            max_magnitude = np.zeros(self._dynamic_pixel_counts.size)
            np.maximum.at(max_magnitude, self._dynamic_index, np.abs(pixel_g2))
            shifts = np.frexp(max_magnitude)[1]
            scaled_g2 = np.ldexp(pixel_g2, -shifts[self._dynamic_index])
            pixel_means = (
                np.bincount(self._dynamic_index, weights=scaled_g2)
                / self._dynamic_pixel_counts
            )
            squared_deviations = np.square(scaled_g2 - pixel_means[self._dynamic_index])
            error = np.ldexp(
                np.sqrt(np.bincount(self._dynamic_index, weights=squared_deviations))
                / self._dynamic_pixel_counts,
                shifts,
            )

        if not np.isfinite(g2).all() or not np.isfinite(error).all():
            raise ValueError("Regrouped G2 and standard error must be finite")
        return g2, error
