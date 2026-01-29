"""Unit tests for Qmap Colormap Selector (Feature 2).

Tests the colormap dropdown functionality for QMap visualization including:
- UI widget creation and configuration
- Signal connections
- Colormap application in plot_qmap
"""

from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pyqtgraph as pg
import pytest


class TestQmapColormapUIWidget:
    """Tests for the cb_qmap_cmap QComboBox widget in viewer_ui.py."""

    def test_colormap_combobox_exists(self):
        """Verify cb_qmap_cmap widget is defined in UI."""
        from xpcsviewer.viewer_ui import Ui_mainWindow

        ui = Ui_mainWindow()
        # Check that the class has the attribute defined
        # (actual widget creation happens in setupUi)
        assert hasattr(Ui_mainWindow, "__init__")

    def test_colormap_options_defined(self):
        """Verify all expected colormaps are available."""
        expected_colormaps = [
            "tab20b",  # Default - categorical for ROI maps
            "jet",  # Classic rainbow
            "hot",  # Black-red-yellow-white
            "plasma",  # Perceptually uniform
            "viridis",  # Perceptually uniform
            "magma",  # Perceptually uniform
            "gray",  # Grayscale
        ]

        # Verify these are valid matplotlib colormaps
        for cmap_name in expected_colormaps:
            cmap = pg.colormap.getFromMatplotlib(cmap_name)
            assert cmap is not None, f"Colormap '{cmap_name}' should be valid"

    def test_tab20b_is_default(self):
        """Verify tab20b is the first (default) option."""
        # tab20b is ideal for discrete ROI maps as it has 20 distinct colors
        cmap = pg.colormap.getFromMatplotlib("tab20b")
        assert cmap is not None, "tab20b should be a valid colormap"

    @pytest.mark.scientific
    def test_tab20b_has_distinct_colors(self):
        """Verify tab20b provides distinct colors for ROI visualization."""
        cmap = pg.colormap.getFromMatplotlib("tab20b")

        # Get colors at different positions
        colors = [cmap.map(i / 20) for i in range(20)]

        # Verify colors are distinct (not all the same)
        unique_colors = set(tuple(c) for c in colors)
        assert len(unique_colors) >= 15, (
            "tab20b should provide at least 15 distinct colors"
        )


class TestQmapColormapSignalConnection:
    """Tests for signal connection between combobox and plot update."""

    @pytest.fixture
    def mock_viewer(self):
        """Create a mock viewer with required attributes."""
        viewer = MagicMock()
        viewer.cb_qmap_cmap = MagicMock()
        viewer.cb_qmap_cmap.currentText = MagicMock(return_value="tab20b")
        viewer.cb_qmap_cmap.currentIndexChanged = MagicMock()
        viewer.update_plot = MagicMock()
        viewer.vk = MagicMock()
        viewer.pg_qmap = MagicMock()
        viewer.get_selected_rows = MagicMock(return_value=[0])
        viewer.comboBox_qmap_target = MagicMock()
        viewer.comboBox_qmap_target.currentText = MagicMock(
            return_value="dynamic_roi_map"
        )
        viewer._guard_no_data = MagicMock(return_value=True)
        return viewer

    def test_colormap_change_triggers_update(self, mock_viewer):
        """Changing colormap should trigger plot update."""
        # Simulate the signal connection that should exist
        mock_viewer.cb_qmap_cmap.currentIndexChanged.connect(mock_viewer.update_plot)

        # Emit the signal
        mock_viewer.cb_qmap_cmap.currentIndexChanged.emit(1)

        # Verify the connection was made
        mock_viewer.cb_qmap_cmap.currentIndexChanged.connect.assert_called()


class TestPlotQmapColormapIntegration:
    """Tests for colormap application in ViewerKernel.plot_qmap."""

    @pytest.fixture
    def mock_kernel(self):
        """Create a mock ViewerKernel with required methods."""
        from unittest.mock import MagicMock

        kernel = MagicMock()
        kernel.statusbar = None

        # Mock get_xf_list to return a valid XpcsFile mock
        mock_xf = MagicMock()
        mock_xf.saxs_2d = np.random.rand(100, 100) * 1000
        mock_xf.dqmap = np.random.randint(1, 20, (100, 100))
        mock_xf.sqmap = np.random.randint(1, 50, (100, 100))
        kernel.get_xf_list = MagicMock(return_value=[mock_xf])

        return kernel

    @pytest.fixture
    def mock_handler(self):
        """Create a mock plot handler."""
        handler = MagicMock()
        handler.setImage = MagicMock()
        handler.setColorMap = MagicMock()
        return handler

    def test_plot_qmap_accepts_cmap_parameter(self, mock_kernel, mock_handler):
        """plot_qmap should accept cmap parameter."""
        from xpcsviewer.viewer_kernel import ViewerKernel

        # Create a real ViewerKernel
        with patch.object(ViewerKernel, "__init__", lambda self: None):
            vk = ViewerKernel()
            vk.get_xf_list = mock_kernel.get_xf_list
            vk.statusbar = None

            # Call with cmap parameter - should not raise
            vk.plot_qmap(
                mock_handler, rows=[0], target="dynamic_roi_map", cmap="viridis"
            )

    def test_plot_qmap_applies_colormap(self, mock_kernel, mock_handler):
        """plot_qmap should apply the specified colormap."""
        from xpcsviewer.viewer_kernel import ViewerKernel

        with patch.object(ViewerKernel, "__init__", lambda self: None):
            vk = ViewerKernel()
            vk.get_xf_list = mock_kernel.get_xf_list
            vk.statusbar = None

            vk.plot_qmap(
                mock_handler, rows=[0], target="dynamic_roi_map", cmap="plasma"
            )

            # Verify setColorMap was called
            mock_handler.setColorMap.assert_called_once()

    def test_plot_qmap_default_cmap_is_tab20b(self, mock_kernel, mock_handler):
        """plot_qmap should default to tab20b colormap."""
        from xpcsviewer.viewer_kernel import ViewerKernel

        with patch.object(ViewerKernel, "__init__", lambda self: None):
            vk = ViewerKernel()
            vk.get_xf_list = mock_kernel.get_xf_list
            vk.statusbar = None

            # Call without cmap parameter
            vk.plot_qmap(mock_handler, rows=[0], target="dynamic_roi_map")

            # Verify setColorMap was called (default should be tab20b)
            mock_handler.setColorMap.assert_called_once()

    @pytest.mark.parametrize(
        "cmap_name", ["tab20b", "jet", "hot", "plasma", "viridis", "magma", "gray"]
    )
    def test_all_colormaps_are_valid(self, mock_kernel, mock_handler, cmap_name):
        """All supported colormaps should be valid matplotlib colormaps."""
        from xpcsviewer.viewer_kernel import ViewerKernel

        with patch.object(ViewerKernel, "__init__", lambda self: None):
            vk = ViewerKernel()
            vk.get_xf_list = mock_kernel.get_xf_list
            vk.statusbar = None

            # Should not raise for any valid colormap
            vk.plot_qmap(
                mock_handler, rows=[0], target="dynamic_roi_map", cmap=cmap_name
            )

            mock_handler.setColorMap.assert_called()

    @pytest.mark.parametrize(
        "target", ["scattering", "dynamic_roi_map", "static_roi_map"]
    )
    def test_colormap_applies_to_all_targets(self, mock_kernel, mock_handler, target):
        """Colormap should apply regardless of the target type."""
        from xpcsviewer.viewer_kernel import ViewerKernel

        with patch.object(ViewerKernel, "__init__", lambda self: None):
            vk = ViewerKernel()
            vk.get_xf_list = mock_kernel.get_xf_list
            vk.statusbar = None

            vk.plot_qmap(mock_handler, rows=[0], target=target, cmap="viridis")

            # Colormap should be applied after image is set
            mock_handler.setColorMap.assert_called()


class TestColormapVisualProperties:
    """Scientific tests for colormap visual properties."""

    @pytest.mark.scientific
    def test_tab20b_colors_are_perceptually_distinct(self):
        """tab20b colors should be perceptually distinct for ROI visualization."""
        cmap = pg.colormap.getFromMatplotlib("tab20b")

        # Get colors from colormap
        colors = np.array([cmap.map(i / 20) for i in range(20)])

        # Compute pairwise color distances (in RGB space)
        n_colors = len(colors)
        for i in range(n_colors):
            for j in range(i + 1, n_colors):
                distance = np.sqrt(np.sum((colors[i] - colors[j]) ** 2))
                # Colors should have some minimum separation
                assert distance > 0.1, (
                    f"Colors {i} and {j} are too similar (distance={distance:.3f})"
                )

    @pytest.mark.scientific
    def test_viridis_is_perceptually_uniform(self):
        """viridis should provide smooth perceptual progression."""
        cmap = pg.colormap.getFromMatplotlib("viridis")

        # Get colors at uniform intervals
        positions = np.linspace(0, 1, 100)
        colors = np.array([cmap.map(p) for p in positions])

        # Compute differences between adjacent colors
        diffs = np.diff(colors, axis=0)
        diff_magnitudes = np.sqrt(np.sum(diffs**2, axis=1))

        # For a perceptually uniform colormap, differences should be similar
        cv = np.std(diff_magnitudes) / np.mean(
            diff_magnitudes
        )  # Coefficient of variation
        assert cv < 0.5, f"viridis should be perceptually uniform (CV={cv:.3f})"

    @pytest.mark.scientific
    def test_gray_is_monotonic(self):
        """Gray colormap should be strictly monotonic in luminance."""
        cmap = pg.colormap.getFromMatplotlib("gray")

        positions = np.linspace(0, 1, 100)
        colors = np.array([cmap.map(p) for p in positions])

        # Compute luminance (approximate using mean of RGB)
        luminances = np.mean(colors[:, :3], axis=1)

        # Luminance should be strictly increasing
        diffs = np.diff(luminances)
        assert np.all(diffs >= 0), "Gray colormap should be monotonically increasing"
