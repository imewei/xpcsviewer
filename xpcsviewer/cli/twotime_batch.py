"""Batch processing module for twotime correlation data."""

import contextlib
import gc
import math
import os
import re
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from xpcsviewer.fileIO.hdf_reader import clear_connection_pool
from xpcsviewer.fileIO.qmap_utils import (
    Q_UNIT_DISPLAY,
    convert_q,
    map_axis,
    normalize_unit,
)
from xpcsviewer.utils.export_guard import ExportCollisionError, ExportGuard
from xpcsviewer.utils.logging_config import get_logger
from xpcsviewer.utils.qmap_notice import PLACEHOLDER_QMAP_WARNING, has_placeholder_qmap
from xpcsviewer.xpcs_file import XpcsFile

# Use non-interactive backend for batch processing
matplotlib.use("Agg")

logger = get_logger(__name__)


def parse_q_phi_pair(q_phi_str: str) -> tuple[float, float]:
    """
    Parse q-phi pair string into q and phi values.

    Args:
        q_phi_str: String in format "q,phi" (e.g., "0.05,45")

    Returns:
        Tuple of (q_value, phi_value)

    Raises:
        ValueError: If string format is invalid
    """
    try:
        parts = q_phi_str.split(",")
        num_parts = 2
        if len(parts) != num_parts:
            raise ValueError(f"Q-phi pair must be in format 'q,phi', got: {q_phi_str}")

        q_value = float(parts[0].strip())
        phi_value = float(parts[1].strip())
        if not (math.isfinite(q_value) and math.isfinite(phi_value)):
            raise ValueError("q and phi must be finite")

        return q_value, phi_value
    except ValueError as e:
        raise ValueError(f"Invalid q-phi pair format '{q_phi_str}': {e}") from e


def _angular_diff(a: float, b: float) -> float:
    """Absolute angular difference in degrees on the circle, in [0, 180]."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _bin_coords(qmap, absolute: int) -> tuple[float | None, float | None]:
    """Full-precision (q, phi) of dynamic bin ``absolute`` from the QMap arrays.

    Mirrors ``QMap.create_qbin_labels``: axis 0 is ``dqlist``; with a second
    axis the bins form a ``dqlist`` x ``dplist`` grid, axis 0 major. An axis
    contributes only when it is named ``q`` / ``phi``; a radial-only map (one
    point on axis 1) has no phi. A map with neither name (e.g. Cartesian ``x`` /
    ``y``) returns its first and second axis values positionally: they are named
    by ``_plain_axes`` / ``map_axis``, not by these variable names.
    """
    if absolute < 0:
        raise IndexError(absolute)
    dq = np.asarray(qmap.dqlist, dtype=np.float64)
    names = list(qmap.map_names)
    if int(qmap.dynamic_num_pts[1]) == 1:
        coords = {names[0]: dq[absolute]}
    else:
        dp = np.asarray(qmap.dplist, dtype=np.float64)
        i0, i1 = divmod(absolute, len(dp))
        coords = {names[0]: dq[i0], names[1]: dp[i1]}
    if not {"q", "phi"} & set(names[:2]):
        first, second = [*coords.values(), None][:2]
        return float(first), None if second is None else float(second)
    q, phi = coords.get("q"), coords.get("phi")
    return (None if q is None else float(q)), (None if phi is None else float(phi))


def _plain_axes(qmap) -> bool:
    """True for a ``q`` / ``(q, phi)`` map, whose bins keep the q / phi wording."""
    names = {str(n) for n in list(getattr(qmap, "map_names", ["q", "phi"]))[:2]}
    return bool({"q", "phi"} & names)


def _axis_names(qmap) -> tuple[tuple[str, str], tuple[str, str] | None] | None:
    """File-name words of a map that is not (q, phi): its two ``(name, unit)`` axes.

    ``None`` for a plain map; the second axis is ``None`` for a one-axis map.
    """
    if _plain_axes(qmap):
        return None
    second = int(qmap.dynamic_num_pts[1]) != 1
    return map_axis(qmap, 0), map_axis(qmap, 1) if second else None


def _unit_word(unit: str) -> str:
    """ASCII file-name token of a unit (``Å⁻¹`` -> ``invA``, ``px`` -> ``px``)."""
    unit = normalize_unit(unit)
    return {Q_UNIT_DISPLAY: "invA", "nm⁻¹": "invnm"}.get(unit) or re.sub(
        r"[^0-9A-Za-z]", "", unit
    )


def target_q_in_file_unit(xfile: XpcsFile, value: float, q_unit: str) -> float:
    """``--q`` / ``--q-phi`` value ``value``, typed in ``q_unit``, in this file's unit.

    The one conversion of a CLI Q selector: the selector is a physical value in
    the unit the user declared (``--q-unit``, inverse angstrom by default); each
    file is queried in ITS OWN axis unit. Raises ``ValueError`` for a file whose
    first axis cannot hold that value: not a reciprocal length (e.g. pixels, for
    which the user must declare ``--q-unit px``) or a radial ``q`` that is not
    positive. A signed value is valid on a signed axis (``qx``, ``qy``).
    """
    name, unit = map_axis(getattr(xfile, "qmap", None), 0)
    try:
        own = float(convert_q(value, q_unit, unit))
    except ValueError as exc:
        raise ValueError(
            f"its first axis is {name} in {unit}; the selector is in {q_unit} "
            f"(set --q-unit): {exc}"
        ) from exc
    if name == "q" and own <= 0:
        raise ValueError(f"the radial q must be positive, got {value} {q_unit}")
    return own


def _collect_valid_qbins(
    xfile: XpcsFile, require_phi: bool = True
) -> list[tuple[int, str, float, float | None]]:
    """Return (index, label, q, phi) for the processed bins that have a q value.

    The index is the position in ``c2_processed_bins``. q and phi are the
    numbers in the file's own arrays; the label is display text only. phi is
    None for a radial-only map; such bins are kept only when ``require_phi`` is
    False. Bins with a missing or non-finite coordinate are skipped.
    """
    qmap = xfile.qmap
    mapping = np.asarray(qmap.dynamic_index_mapping)
    labels = xfile.get_twotime_qbin_labels()
    valid_qbins = []
    for i, (label, bin_id) in enumerate(
        zip(labels, np.ravel(xfile.c2_processed_bins).tolist(), strict=True)
    ):
        q_value = phi_value = None
        if 1 <= bin_id <= len(mapping):
            with contextlib.suppress(IndexError):  # bin outside the q/phi axes
                q_value, phi_value = _bin_coords(qmap, int(mapping[int(bin_id) - 1]))
        if (
            q_value is not None
            and math.isfinite(q_value)
            and (phi_value is None or math.isfinite(phi_value))
            and (phi_value is not None or not require_phi)
        ):
            valid_qbins.append((i, label, q_value, phi_value))
        else:
            logger.warning(f"No usable q/phi for processed qbin {i}: {label}")
    return valid_qbins


def _qbins_with_phi(xfile: XpcsFile) -> list[tuple[int, str, float, float]]:
    """Qbins that have both q and phi (the angle-based modes need phi).

    A map whose second axis is not an angle (Cartesian ``y`` in pixels) has none:
    matching it on the circle would pair unrelated pixel rows.
    """
    qmap = xfile.qmap
    if int(qmap.dynamic_num_pts[1]) != 1 and map_axis(qmap, 1)[0] != "phi":
        logger.warning(
            "Angle selection needs a phi axis; this file's map is "
            f"{map_axis(qmap, 0)[0]} / {map_axis(qmap, 1)[0]}"
        )
        return []
    return [
        (i, label, q, phi)
        for i, label, q, phi in _collect_valid_qbins(xfile)
        if phi is not None
    ]


def find_qbins_for_q(
    xfile: XpcsFile, target_q: float
) -> list[tuple[int, str, float, float | None]]:
    """
    Find all qbins with the closest q-value(s) across all phi angles.

    Args:
        xfile: XpcsFile instance
        target_q: Target q-value to match

    Returns:
        List of tuples (qbin_index, label, q_value, phi_value) for closest
        matching qbins
    """
    # phi is optional here: a radial-only map has no angle in its labels
    valid_qbins = _collect_valid_qbins(xfile, require_phi=False)

    if not valid_qbins:
        logger.warning("No valid qbins found for q selection")
        return []

    # Find the closest q-value(s)
    q_values = [qbin[2] for qbin in valid_qbins]
    closest_q = min(q_values, key=lambda q: abs(q - target_q))
    closest_q_diff = abs(closest_q - target_q)

    # Get all qbins with the closest q-value
    tolerance = 1e-10  # Use tight tolerance for exact match
    matching_qbins = [
        (i, label, q_val, phi_val)
        for i, label, q_val, phi_val in valid_qbins
        if abs(q_val - closest_q) < tolerance
    ]

    logger.info(
        f"Found closest q={closest_q:.6f} "
        f"(diff={closest_q_diff:.6f} from target {target_q:.6f})"
    )
    logger.info(
        f"Selected {len(matching_qbins)} qbins with q={closest_q:.6f} "
        f"across different phi angles"
    )

    # Log the phi angles found
    phi_angles = sorted(p for _, _, _, p in matching_qbins if p is not None)
    logger.info(f"Phi angles found: {phi_angles}")

    return matching_qbins


def find_qbins_for_phi(
    xfile: XpcsFile, target_phi: float
) -> list[tuple[int, str, float, float]]:
    """
    Find all qbins with the closest phi-value(s) across all q values.

    Args:
        xfile: XpcsFile instance
        target_phi: Target phi-value to match

    Returns:
        List of tuples (qbin_index, label, q_value, phi_value) for closest
        matching qbins
    """
    valid_qbins = _qbins_with_phi(xfile)

    if not valid_qbins:
        logger.warning("No valid qbins found for phi selection")
        return []

    # Find the closest phi-value(s)
    phi_values = [qbin[3] for qbin in valid_qbins]
    closest_phi = min(phi_values, key=lambda phi: _angular_diff(phi, target_phi))
    closest_phi_diff = _angular_diff(closest_phi, target_phi)

    # Get all qbins with the closest phi-value
    phi_tolerance = 1e-10  # Use tight tolerance for exact match
    matching_qbins = [
        (i, label, q_val, phi_val)
        for i, label, q_val, phi_val in valid_qbins
        if _angular_diff(phi_val, closest_phi) < phi_tolerance
    ]

    logger.info(
        f"Found closest phi={closest_phi:.2f}° "
        f"(diff={closest_phi_diff:.2f}° from target {target_phi:.2f}°)"
    )
    logger.info(
        f"Selected {len(matching_qbins)} qbins with phi={closest_phi:.2f}° "
        f"across different q values"
    )

    # Log the q values found
    q_values = sorted([q_val for _, _, q_val, _ in matching_qbins])
    logger.info(f"Q values found: {q_values}")

    return matching_qbins


def find_qbin_for_qphi(
    xfile: XpcsFile,
    target_q: float,
    target_phi: float,
) -> tuple[int, str, float, float] | None:
    """
    Find single qbin closest to specific q-phi pair.

    Args:
        xfile: XpcsFile instance
        target_q: Target q-value to match
        target_phi: Target phi-value to match

    Returns:
        Tuple (qbin_index, label, q_value, phi_value) for closest matching qbin or None
    """
    valid_qbins = _qbins_with_phi(xfile)

    if not valid_qbins:
        logger.warning("No valid qbins found for q-phi selection")
        return None

    # Find qbin with minimum combined distance
    # Normalize distances to make them comparable
    q_values = [qbin[2] for qbin in valid_qbins]
    phi_values = [qbin[3] for qbin in valid_qbins]
    q_range = max(q_values) - min(q_values) if len(set(q_values)) > 1 else 1.0
    phi_range = max(phi_values) - min(phi_values) if len(set(phi_values)) > 1 else 1.0

    best_match = None
    best_distance = float("inf")

    for i, label, q_value, phi_value in valid_qbins:
        # Normalized euclidean distance
        q_norm = abs(q_value - target_q) / q_range
        phi_norm = _angular_diff(phi_value, target_phi) / phi_range
        distance = (q_norm**2 + phi_norm**2) ** 0.5

        if distance < best_distance:
            best_distance = distance
            best_match = (i, label, q_value, phi_value)

    if best_match:
        q_diff = abs(best_match[2] - target_q)
        phi_diff = _angular_diff(best_match[3], target_phi)
        logger.info(
            f"Found closest qbin: q={best_match[2]:.6f} (diff={q_diff:.6f}), "
            f"phi={best_match[3]:.2f}° (diff={phi_diff:.2f}°)"
        )
        logger.info(f"Selected qbin: {best_match[1]}")
    else:
        logger.warning(f"No qbin found for q={target_q:.4f}, phi={target_phi:.1f}°")

    return best_match


def create_twotime_plot_matplotlib(
    c2_matrix: np.ndarray, delta_t: float, title: str, dpi: int = 300
) -> plt.Figure:
    """
    Create twotime correlation plot using matplotlib.

    Args:
        c2_matrix: 2D correlation matrix
        delta_t: Time step for axes scaling
        title: Plot title
        dpi: Image resolution

    Returns:
        Matplotlib Figure object
    """
    # Clean C2 data to remove NaN/inf values
    finite_mask = np.isfinite(c2_matrix)
    if np.any(finite_mask):
        finite_values = c2_matrix[finite_mask]
        if len(finite_values) > 0:
            vmin, vmax = np.percentile(finite_values, [0.5, 99.5])
            if vmin >= vmax:
                vmax = vmin + 1e-6
        else:
            vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = 0.0, 1.0
        logger.warning("All C2 values are non-finite, using default levels")

    # Replace non-finite values
    c2_clean = np.nan_to_num(c2_matrix, nan=vmin, posinf=vmax, neginf=vmin)

    # Create figure with high DPI
    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)

    # Create time axes
    size = c2_clean.shape[0]
    extent = [0, size * delta_t, 0, size * delta_t]

    # Plot correlation matrix
    im = ax.imshow(
        c2_clean,
        cmap="jet",
        aspect="equal",
        origin="lower",
        extent=tuple(extent),
        vmin=vmin,
        vmax=vmax,
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("C₂(t₁, t₂)", rotation=270, labelpad=20)

    # Set labels and title
    ax.set_xlabel("t₁ (s)")
    ax.set_ylabel("t₂ (s)")
    ax.set_title(title)

    # Adjust layout
    plt.tight_layout()

    return fig


def generate_output_filename(
    input_path: str,
    q_value: float,
    phi_value: float | None,
    output_format: str = "png",
    qbin_index: int | None = None,
    root: str | None = None,
    axes: tuple[tuple[str, str], tuple[str, str] | None] | None = None,
) -> str:
    """
    Generate the output path of one image, relative to the output directory.

    The input file's directory (relative to ``root``) is kept as real
    subdirectories and the input file's full name, including its extension, is
    kept unmodified, so distinct input files can never share an output path.
    The Q-bin index is part of the name because the rounded q/phi values alone
    can coincide for different bins.

    Args:
        input_path: Input HDF file path
        q_value: Q-value
        phi_value: Phi-value, or None for a radial-only map (omitted from the name)
        output_format: Image format
        qbin_index: Q-bin index within the file; part of the name when given
        root: Batch root directory; the input path is made relative to it
        axes: ``(name, unit)`` of the two axes of a map that is not (q, phi)
            (``_axis_names``); the bin is then named by BOTH coordinates and
            their units, ``_x40.0000px_y120.0000px``, instead of ``_q..._phi...``
            (``q_value`` / ``phi_value`` are then the first- and second-axis
            values)

    Returns:
        Output path relative to the output directory
    """
    rel = os.path.relpath(input_path, root) if root else Path(input_path).name
    qbin = f"_bin{qbin_index:04d}" if qbin_index is not None else ""
    if axes is not None:
        words = [
            (re.sub(r"\W", "", str(a[0])) or "axis", _unit_word(a[1]))
            for a in axes
            if a is not None
        ]
        coords = "".join(
            f"_{w}{v:.4f}{u}"
            for (w, u), v in zip(words, (q_value, phi_value), strict=False)
            if v is not None
        )
        return f"{rel}{qbin}{coords}.{output_format}"
    phi = f"_phi{phi_value:.1f}" if phi_value is not None else ""
    return f"{rel}{qbin}_q{q_value:.4f}{phi}.{output_format}"


def process_single_file(
    file_path: str,
    args,
    guard: ExportGuard | None = None,
    refused: list[str] | None = None,
) -> int:
    """
    Process a single HDF file and generate twotime images.

    Every image goes through ``guard.claim`` before it is written, so within one
    run no image is written onto an input HDF file or onto an image already
    written by another (input file, Q-bin) pair, however the paths are spelled
    (symlinked directories, symlinked file names, hard links, case).
    Re-running the CLI into the same output folder in a separate run overwrites
    the earlier images by design.

    Args:
        file_path: Path to HDF file
        args: Command line arguments
        guard: The run's write gate; its inputs must be every input HDF file of
            the run. Defaults to a gate whose only input is ``file_path``.
        refused: Descriptions of images refused by the guard. Appended to in
            place.

    Returns:
        Number of images generated
    """
    logger.info(f"Processing file: {file_path}")

    if guard is None:
        guard = ExportGuard([file_path])
    if refused is None:
        refused = []
    xfile = None  # Initialize outside try block for finally cleanup
    try:
        # Load XpcsFile
        xfile = XpcsFile(file_path)
        if has_placeholder_qmap(xfile):
            logger.warning(f"{file_path}: {PLACEHOLDER_QMAP_WARNING}")

        # Verify this is a twotime file
        if "Twotime" not in xfile.atype:
            logger.warning(
                f"File {file_path} is not a twotime file "
                f"(type: {xfile.atype}), skipping"
            )
            return 0

        # Get all available qbins for debugging and summary
        qbin_labels = xfile.get_twotime_qbin_labels()
        logger.info(f"File contains {len(qbin_labels)} total qbins")

        # Log all qbin labels for debugging
        logger.debug("Available qbins:")
        for i, label in enumerate(qbin_labels):
            logger.debug(f"  Qbin {i}: {label}")

        # Determine qbins to process based on selection mode
        qbins_to_process: Sequence[tuple[int, str, float, float | None]] = []

        # The selector is a physical Q in the unit the user declared; every file
        # is queried in its own unit (``target_q_in_file_unit``).
        q_unit = getattr(args, "q_unit", None) or Q_UNIT_DISPLAY
        try:
            if args.q is not None:
                # Mode 1: All phi angles at specific q
                qbins_to_process = find_qbins_for_q(
                    xfile, target_q_in_file_unit(xfile, args.q, q_unit)
                )
            elif args.phi is not None:
                # Mode 2: All q values at specific phi
                qbins_to_process = find_qbins_for_phi(xfile, args.phi)
            elif args.q_phi is not None:
                # Mode 3: Specific q-phi pair
                target_q, target_phi = parse_q_phi_pair(args.q_phi)
                qbin_match = find_qbin_for_qphi(
                    xfile, target_q_in_file_unit(xfile, target_q, q_unit), target_phi
                )
                if qbin_match:
                    qbins_to_process = [qbin_match]
        except ValueError as exc:
            logger.error(f"{file_path}: the Q selection cannot be applied: {exc}")
            return 0

        if not qbins_to_process:
            logger.warning(f"No matching qbins found for file {file_path}")
            return 0

        # Create output directory if it doesn't exist
        os.makedirs(args.output, exist_ok=True)

        images_generated = 0

        # Process each qbin
        for qbin_index, label, q_value, phi_value in qbins_to_process:
            logger.info(f"Processing qbin {qbin_index}: {label}")

            try:
                # Get C2 data for this qbin
                c2_result = xfile.get_twotime_c2(
                    selection=qbin_index, correct_diag=True
                )
                if c2_result is None:
                    logger.warning(f"Failed to get C2 data for qbin {qbin_index}")
                    continue

                c2_matrix = c2_result["c2_mat"]
                delta_t = c2_result["delta_t"]

                # Generate plot
                title = f"{Path(file_path).name} - {label}"
                fig = None
                try:
                    fig = create_twotime_plot_matplotlib(
                        c2_matrix, delta_t, title, args.dpi
                    )

                    # Save image
                    output_filename = generate_output_filename(
                        file_path,
                        q_value,
                        phi_value,
                        args.format,
                        qbin_index=qbin_index,
                        root=args.input if os.path.isdir(args.input) else None,
                        axes=_axis_names(xfile.qmap),
                    )
                    output_path = os.path.join(args.output, output_filename)
                    out_dir = os.path.dirname(output_path)
                    os.makedirs(out_dir, exist_ok=True)
                    try:
                        output_path = guard.claim(
                            output_path, f"{file_path} (qbin {qbin_index})"
                        )
                    except ExportCollisionError as collision:
                        logger.error(str(collision))
                        refused.append(str(collision))
                        continue

                    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
                    logger.info(f"Saved image: {output_path}")
                    images_generated += 1
                finally:
                    if fig is not None:
                        plt.close(fig)  # Always free memory

            except Exception as e:
                logger.error(f"Error processing qbin {qbin_index}: {e}")
                continue

        logger.info(f"\n--- PROCESSING SUMMARY FOR {Path(file_path).name} ---")
        logger.info(f"Total qbins available: {len(qbin_labels)}")
        logger.info(f"Qbins selected for processing: {len(qbins_to_process)}")
        logger.info(f"Images successfully generated: {images_generated}")
        logger.info(f"Processing completed for {file_path}")
        return images_generated

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return 0
    finally:
        # CRITICAL: Clean up XpcsFile resources to prevent memory accumulation
        if xfile is not None:
            try:
                logger.debug(f"Cleaning up resources for {file_path}")
                xfile.clear_cache()  # Clear internal caches and cached data
            except Exception as cleanup_error:
                logger.warning(f"Error during cache cleanup: {cleanup_error}")
            finally:
                del xfile  # Explicitly delete reference to allow garbage collection


def find_hdf_files(directory: str) -> list[str]:
    """
    Find all HDF files in directory recursively.

    Args:
        directory: Directory to search

    Returns:
        List of HDF file paths
    """
    hdf_extensions = [".h5", ".hdf5", ".hdf"]
    hdf_files = []

    for root, _dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in hdf_extensions):
                hdf_files.append(os.path.join(root, file))

    return sorted(hdf_files)


def process_directory(directory: str, args, refused: list[str] | None = None) -> int:
    """
    Process all HDF files in directory.

    Args:
        directory: Directory containing HDF files
        args: Command line arguments
        refused: Appended with one description per image refused because its
            destination is an input file or was already written in this run

    Returns:
        Total number of images generated
    """
    logger.info(f"Processing directory: {directory}")

    hdf_files = find_hdf_files(directory)
    if not hdf_files:
        logger.warning(f"No HDF files found in directory: {directory}")
        return 0

    logger.info(f"Found {len(hdf_files)} HDF files to process")

    total_images = 0
    successful_files = 0
    guard = ExportGuard(hdf_files)

    for i, file_path in enumerate(hdf_files, 1):
        logger.info(f"Processing file {i}/{len(hdf_files)}: {file_path}")

        try:
            images_count = process_single_file(file_path, args, guard, refused)
            total_images += images_count
            if images_count > 0:
                successful_files += 1
        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")

        # Periodic cleanup to prevent memory accumulation in long batch jobs
        if i % 10 == 0:
            logger.debug(f"Performing periodic cleanup after {i} files")
            gc.collect()  # Force garbage collection
            clear_connection_pool()  # Clear HDF5 connection pool cache

    logger.info(
        f"Directory processing complete: {successful_files}/{len(hdf_files)} "
        f"files processed successfully"
    )
    logger.info(f"Total images generated: {total_images}")

    return total_images


def run_twotime_batch(args) -> int:
    """
    Main entry point for twotime batch processing.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger.info("Starting twotime batch processing")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Selection mode: q={args.q}, phi={args.phi}, q_phi={args.q_phi}")
    logger.info(f"Image settings: format={args.format}, DPI={args.dpi}")

    # Validate arguments
    selection_count = sum(x is not None for x in [args.q, args.phi, args.q_phi])
    if selection_count != 1:
        logger.error(f"Exactly one selection argument required, got {selection_count}")
        logger.error(f"Arguments: q={args.q}, phi={args.phi}, q_phi={args.q_phi}")
        return 1

    # Validate argument values
    for name, value in (("Q", args.q), ("Phi", args.phi)):
        if value is not None and not math.isfinite(value):
            logger.error(f"{name}-value must be finite, got: {value}")
            return 1

    # The sign of a Q is not judged here: a radial ``q`` must be positive, but
    # ``qx`` / ``qy`` of a Cartesian map are signed. Each file checks its own axis
    # (``target_q_in_file_unit``); a batch where no file accepts it generates
    # no image and fails.
    max_phi = 360
    if args.phi is not None and not (0 <= args.phi <= max_phi):
        logger.warning(f"Phi-value {args.phi}° is outside typical range [0, 360]°")

    if args.q_phi is not None:
        try:
            q_val, phi_val = parse_q_phi_pair(args.q_phi)
            logger.info(f"Validated q-phi pair: q={q_val}, phi={phi_val}")
        except ValueError as e:
            logger.error(f"Invalid q-phi pair format: {e}")
            return 1

    logger.info("Argument validation passed")

    try:
        # Validate input path
        if not os.path.exists(args.input):
            logger.error(f"Input path does not exist: {args.input}")
            return 1

        # Process input (file or directory)
        refused: list[str] = []
        if os.path.isfile(args.input):
            images_generated = process_single_file(
                args.input, args, ExportGuard([args.input]), refused
            )
        elif os.path.isdir(args.input):
            images_generated = process_directory(args.input, args, refused)
        else:
            logger.error(f"Input path is neither file nor directory: {args.input}")
            return 1

        if refused:
            logger.error(
                f"{len(refused)} image(s) were refused: the output file is an "
                f"input file or another source's output in this run"
            )
            return 1

        if images_generated == 0:
            logger.warning("No images were generated")
            return 1

        logger.info(
            f"Batch processing completed successfully. "
            f"Generated {images_generated} images."
        )
        return 0

    except Exception as e:
        logger.error(f"Batch processing failed: {e}", exc_info=True)
        return 1
