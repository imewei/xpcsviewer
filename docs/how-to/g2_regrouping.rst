Regroup multi-tau G2 with an external Q-map
===========================================

1. Add the APS result to the Target list, select its row, and choose
   :menuselection:`File --> Regroup G2…`.
2. Click **Choose Q-map…** and select an HDF5 Q-map file. Confirm that its
   pixels match the source detector's coordinates and orientation. The Viewer
   checks shape and bin membership. Confirm the registration yourself, even
   when the detector and Q-map dimensions match.
3. Click **Preview** and select a **Dynamic bin** by its ID. The plot shows
   positive delays in seconds on a logarithmic axis, with G2 and its error for
   that bin. A new Q-map needs its own preview before it can be saved. If
   validation fails, the previous plot remains visible.
4. Click **Save as…** and choose a new HDF5 filename. Canceling the picker
   creates no file. When the picker returns a filename, the Viewer accepts
   the save only if the preview still matches the registered source and Q-map.
   An accepted save finishes even if you close the dialog or switch datasets.
   The Viewer reports success after the complete file appears at the destination.
   If the destination exists or its filesystem cannot publish atomically,
   choose another destination.

Input files and normalization
-----------------------------

- The APS result must contain ``/xpcs/multitau/unnormalized_G2`` with shape
  ``(delays, 3, y, x)`` and channels ``G2``, ``IP``, and ``IF`` in that order.
  Combined multi-tau and two-time results are eligible when the unnormalized
  cube was saved. Normalized curves alone do not contain the pixel data needed
  for regrouping.
- ``/xpcs/multitau/delay_list`` can have shape ``(n,)``, ``(1, n)``, or ``(n, 1)``.
  Its finite, nonnegative frame delays must match the cube and include at least
  one positive delay. Elapsed seconds are the frame delays times the positive
  detector frame time at ``/entry/instrument/detector_1/frame_time``. If timing
  factors are present, both ``stride_frame`` and ``avg_frame`` must be positive
  integers under ``/xpcs/multitau/config`` and are also multiplied in.
- The Q-map must contain ``static_roi_map`` and ``dynamic_roi_map`` together
  under ``/qmap`` or ``/xpcs/qmap``. The Viewer uses ``/qmap`` when both groups
  contain a pair. Both maps must match the detector shape, use nonnegative
  integer IDs, and cover the same nonempty set of pixels with positive IDs.
  Each positive static bin must belong to one dynamic bin. Calibration metadata
  in the external Q-map is optional.

For each delay, the Viewer averages the three channels within each static bin
and divides the mean G2 by the product of the mean IP and mean IF. Each dynamic
bin is the unweighted mean of its normalized static bins. Its error is the
population standard deviation of the included pixel ratios ``G2 / (IP * IF)``,
divided by the square root of the number of included pixels. A single-pixel
bin has zero error. The Viewer rejects the preview at the first delay with
nonfinite included channels or normalization values, or a nonpositive
normalization product. Pixels outside the shared map support are ignored.

Export file layout
------------------

The standalone HDF5 file has the root attribute ``xpcsviewer_artifact`` set to
``g2_regroup``. When the Target loader reads the file, it logs that the file
is a derived G2 export for downstream analysis. Under
``/xpcs/multitau`` it contains these datasets.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Dataset
     - Contents
   * - ``delay_list``
     - One-dimensional native frame delays, with the source ``units``
       attribute when supplied.
   * - ``delay_seconds``
     - One-dimensional elapsed delays, with the ``units`` attribute ``s``.
   * - ``dynamic_bin_ids``
     - Sorted positive IDs present in the dynamic ROI map.
   * - ``normalized_g2``
     - G2 as a float64 array with shape ``(delay_count, dynamic_bin_count)``.
   * - ``normalized_g2_err``
     - G2 error as a float64 array with the same shape.

Columns in both G2 arrays follow ``dynamic_bin_ids``. All delays are exported,
including zero. The correlation group has ``source_filename`` and
``qmap_filename`` provenance attributes containing the input filenames.
``/qmap/static_roi_map`` and ``/qmap/dynamic_roi_map`` contain the validated
ROI maps in their original integer dtypes. The source and Q-map files are
read-only inputs.
