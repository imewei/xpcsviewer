"""Translate FileReader's (upstream-schema) metadata into SimpleMaskKernel's schema.

Two independent conversion rules -- do not conflate them:

1. Beam center (``beam_center_x``/``beam_center_y`` -> ``bcx``/``bcy``):
   rename only. Both schemas store these in pixels.
2. Length fields (``pixel_size`` -> ``pix_dim``, ``detector_distance`` ->
   ``det_dist``): rename AND convert meters -> millimeters (``x * 1000``).

See docs/superpowers/specs/2026-08-10-multi-format-reader-stack-design.md
for the full rationale, including why this was wrong in the spec's first
draft.
"""

from __future__ import annotations

import math
from typing import Any

_REQUIRED_READER_FIELDS = (
    "energy",
    "beam_center_x",
    "beam_center_y",
    "detector_shape_x",
    "detector_shape_y",
    "pixel_size",
    "detector_distance",
)


def _swing_degrees(reader_metadata: dict[str, Any], key: str) -> float:
    """Return a swing angle in degrees; absent means 0.0, present must be finite."""
    value = reader_metadata.get(key)
    if value is None:
        return 0.0
    try:
        angle = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a finite scalar, got {value!r}") from exc
    if not math.isfinite(angle):
        raise ValueError(f"{key} must be a finite scalar, got {value!r}")
    return angle


def to_kernel_metadata(reader_metadata: dict[str, Any], stype: str) -> dict[str, Any]:
    """Convert a :class:`~xpcsviewer.simplemask.reader.base_reader.FileReader`
    metadata dict into the schema :meth:`SimpleMaskKernel.read_data` expects.

    Args:
        reader_metadata: ``reader.metadata`` after ``prepare_data()``.
        stype: ``reader.stype`` ("Transmission" or "Reflection").

    Raises:
        KeyError: A field this function needs is missing (a reader bug --
            :func:`~xpcsviewer.simplemask.reader.base_reader.get_fake_metadata`
            guarantees these fields exist at runtime, so this should never
            fire against real reader output).
        ValueError: A swing angle is present but not a finite scalar. Callers
            in the raw-file worker wrap this in ``RawDataReadError``.
        NotImplementedError: ``stype == "Reflection"`` -- the 9-ID-D
            orientation mapping (incident_angle/orientation) lands in PR2.
            Refusing beats silently defaulting to a wrong incidence angle.
    """
    missing = [f for f in _REQUIRED_READER_FIELDS if f not in reader_metadata]
    if missing:
        raise KeyError(f"reader metadata missing required field(s): {missing}")

    if stype == "Reflection":
        raise NotImplementedError(
            "Reflection geometry adaptation (9-ID-D incident_angle/orientation "
            "mapping) is not implemented until PR2."
        )

    return {
        "bcx": reader_metadata["beam_center_x"],
        "bcy": reader_metadata["beam_center_y"],
        "pix_dim": reader_metadata["pixel_size"] * 1000.0,
        "det_dist": reader_metadata["detector_distance"] * 1000.0,
        "energy": reader_metadata["energy"],
        "shape": (
            reader_metadata["detector_shape_y"],
            reader_metadata["detector_shape_x"],
        ),
        "stype": stype,
        "swing_angle_horizontal": _swing_degrees(
            reader_metadata, "swing_angle_horizontal"
        ),
        "swing_angle_vertical": _swing_degrees(reader_metadata, "swing_angle_vertical"),
    }
