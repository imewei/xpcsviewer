"""Check regrouped multi-tau G2 against an independent NumPy reference."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from xpcsviewer.fileIO.aps_8idi import key
from xpcsviewer.fileIO.g2_regroup import read_g2_regroup

RTOL = 1e-10
ATOL = 1e-12


def _reference_delay(
    channels: np.ndarray,
    static_map: np.ndarray,
    dynamic_map: np.ndarray,
    bin_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    g2, error = [], []
    for bin_id in bin_ids:
        selected = dynamic_map == bin_id
        active = channels[:, selected].astype(np.float64, copy=False)
        static_ids = static_map[selected]
        normalized = []
        for static_id in np.unique(static_ids):
            mean_g2, mean_ip, mean_if = active[:, static_ids == static_id].mean(axis=1)
            normalized.append(mean_g2 / (mean_ip * mean_if))
        g2.append(np.mean(normalized))
        pixel_g2 = active[0] / (active[1] * active[2])
        error.append(np.std(pixel_g2, ddof=0) / np.sqrt(pixel_g2.size))
    return np.asarray(g2), np.asarray(error)


def _peak_rss_mib() -> float:
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes and Linux reports KiB.
    return rss / (1024**2 if sys.platform == "darwin" else 1024)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a source against a registered Q-map and report process peak RSS."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, metavar="SOURCE", help="APS multi-tau HDF5 file"
    )
    parser.add_argument(
        "qmap", type=Path, metavar="QMAP", help="registered Q-map HDF5 file"
    )
    args = parser.parse_args(argv)
    if sys.platform not in {"linux", "darwin"}:
        print("Peak RSS reporting requires Linux or macOS", file=sys.stderr)
        return 1

    try:
        result = read_g2_regroup(args.source, args.qmap)
        delay_count = result.delay_seconds.size
        delay_indices = sorted(
            {0, delay_count // 2, delay_count - 1, int(result.delay_seconds.argmax())}
        )
        bin_ids = np.unique(result.dynamic_roi_map[result.dynamic_roi_map > 0])
        if not np.array_equal(result.dynamic_bin_ids, bin_ids):
            raise ValueError(
                "Regrouped dynamic-bin IDs disagree with the selected Q-map"
            )
        expected_shape = (delay_count, bin_ids.size)
        if (
            result.normalized_g2.shape != expected_shape
            or result.normalized_g2_err.shape != expected_shape
        ):
            raise ValueError(
                f"Expected regrouped curves and errors with shape {expected_shape}"
            )

        reference_g2 = np.empty((len(delay_indices), bin_ids.size))
        reference_error = np.empty_like(reference_g2)
        with h5py.File(args.source, "r") as source, np.errstate(all="ignore"):
            cube = source[key["nexus"]["G2"]]
            for row, delay_index in enumerate(delay_indices):
                reference_g2[row], reference_error[row] = _reference_delay(
                    cube[delay_index],
                    result.static_roi_map,
                    result.dynamic_roi_map,
                    bin_ids,
                )
    except (KeyError, OSError, ValueError) as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1

    actual_g2 = result.normalized_g2[delay_indices]
    actual_error = result.normalized_g2_err[delay_indices]
    print(f"Detector shape: {result.dynamic_roi_map.shape}")
    print(f"Regrouped shape: {result.normalized_g2.shape}")
    print(
        f"Compared {len(delay_indices)} delays and {bin_ids.size} bins at delay indices {delay_indices}"
    )
    with np.errstate(over="ignore", invalid="ignore"):
        print(
            f"Max absolute G2 difference: {np.max(np.abs(actual_g2 - reference_g2)):.12g}"
        )
        print(
            f"Max absolute error difference: {np.max(np.abs(actual_error - reference_error)):.12g}"
        )
        agrees = np.allclose(
            actual_g2, reference_g2, rtol=RTOL, atol=ATOL
        ) and np.allclose(actual_error, reference_error, rtol=RTOL, atol=ATOL)
    print(f"Peak RSS: {_peak_rss_mib():.3f} MiB")

    if not agrees:
        print("Numerical mismatch against the independent reference", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
