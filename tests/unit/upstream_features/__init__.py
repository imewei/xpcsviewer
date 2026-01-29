"""Unit tests for upstream feature integration.

This module contains tests for features integrated from upstream repositories
(pyXpcsViewer and pySimpleMask):

1. test_roi_scale_handles - Enhanced ROI scale handles (Rectangle 8, Ellipse 8, Circle 2)
2. test_qmap_colormap - Qmap colormap selector with tab20b default
3. test_g2_partial_safety - g2_partial availability check before stability plotting
4. test_gixpcs_precision - GIXPCS display precision (6 decimals for qx/qr)
5. test_blemish_export - Blemish map tracking and export in partition
"""
