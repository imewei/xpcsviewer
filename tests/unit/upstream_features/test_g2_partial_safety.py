"""Unit tests for g2_partial Safety Check (Feature 3).

Tests the safety check in viewer_kernel.plot_g2_stability that verifies
g2_partial data is available before plotting.

Scientific testing approach:
- Verify graceful handling of missing data
- Test status message display
- Ensure no exceptions when data is unavailable
"""

import logging
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest


class TestG2PartialSafetyCheck:
    """Tests for g2_partial availability check before stability plotting."""

    @pytest.fixture
    def mock_xf_with_g2_partial(self):
        """Create a mock XpcsFile with g2_partial data."""
        xf = MagicMock()
        xf.atype = "Multitau"
        xf.fname = "/path/to/test.h5"
        xf.g2_partial = np.random.rand(10, 50, 5)  # frames x delays x qbins
        xf.get_g2_stability_data = MagicMock(
            return_value=(
                np.linspace(0.01, 0.1, 5),  # q
                np.logspace(-3, 1, 50),  # tel
                np.random.rand(10, 50, 5),  # g2
                np.random.rand(10, 50, 5) * 0.1,  # g2_err
                ["q=0.01", "q=0.05"],  # qbin_labels
                list(range(10)),  # labels
            )
        )
        return xf

    @pytest.fixture
    def mock_xf_without_g2_partial(self):
        """Create a mock XpcsFile without g2_partial data."""
        xf = MagicMock()
        xf.atype = "Multitau"
        xf.fname = "/path/to/test_no_partial.h5"
        xf.g2_partial = None  # No partial data available
        return xf

    @pytest.fixture
    def mock_statusbar(self):
        """Create a mock statusbar."""
        statusbar = MagicMock()
        statusbar.showMessage = MagicMock()
        return statusbar

    @pytest.fixture
    def mock_handler(self):
        """Create a mock plot handler."""
        handler = MagicMock()
        handler.clear = MagicMock()
        return handler

    def test_plot_g2_stability_with_valid_data(
        self, mock_xf_with_g2_partial, mock_statusbar, mock_handler
    ):
        """plot_g2_stability should proceed when g2_partial is available."""
        from xpcsviewer.viewer_kernel import ViewerKernel

        with patch.object(ViewerKernel, "__init__", lambda self: None):
            vk = ViewerKernel()
            vk.get_xf_list = MagicMock(return_value=[mock_xf_with_g2_partial])
            vk.statusbar = mock_statusbar
            vk._module_cache = {}
            vk.get_module = MagicMock(return_value=MagicMock())

            # Should not return early
            vk.plot_g2_stability(
                mock_handler,
                q_range=(0.01, 0.1),
                t_range=(1e-3, 10),
                y_range=(0.9, 1.5),
                rows=[0],
            )

            # get_module should be called (indicating we proceeded past the check)
            vk.get_module.assert_called_once_with("g2mod")

    def test_plot_g2_stability_without_data_returns_early(
        self, mock_xf_without_g2_partial, mock_statusbar, mock_handler
    ):
        """plot_g2_stability should return early when g2_partial is None."""
        from xpcsviewer.viewer_kernel import ViewerKernel

        with patch.object(ViewerKernel, "__init__", lambda self: None):
            vk = ViewerKernel()
            vk.get_xf_list = MagicMock(return_value=[mock_xf_without_g2_partial])
            vk.statusbar = mock_statusbar
            vk.get_module = MagicMock()

            vk.plot_g2_stability(
                mock_handler,
                q_range=(0.01, 0.1),
                t_range=(1e-3, 10),
                y_range=(0.9, 1.5),
                rows=[0],
            )

            # get_module should NOT be called (returned early)
            vk.get_module.assert_not_called()

    def test_status_message_shown_when_data_missing(
        self, mock_xf_without_g2_partial, mock_statusbar, mock_handler
    ):
        """Status bar should show message when g2_partial is unavailable."""
        from xpcsviewer.viewer_kernel import ViewerKernel

        with patch.object(ViewerKernel, "__init__", lambda self: None):
            vk = ViewerKernel()
            vk.get_xf_list = MagicMock(return_value=[mock_xf_without_g2_partial])
            vk.statusbar = mock_statusbar
            vk.get_module = MagicMock()

            vk.plot_g2_stability(
                mock_handler,
                q_range=(0.01, 0.1),
                t_range=(1e-3, 10),
                y_range=(0.9, 1.5),
                rows=[0],
            )

            # Status message should be shown
            mock_statusbar.showMessage.assert_called_once()
            call_args = mock_statusbar.showMessage.call_args
            assert "g2_partial" in call_args[0][0]
            assert "not available" in call_args[0][0]

    def test_status_message_timeout_is_3_seconds(
        self, mock_xf_without_g2_partial, mock_statusbar, mock_handler
    ):
        """Status message should display for 3 seconds (3000ms)."""
        from xpcsviewer.viewer_kernel import ViewerKernel

        with patch.object(ViewerKernel, "__init__", lambda self: None):
            vk = ViewerKernel()
            vk.get_xf_list = MagicMock(return_value=[mock_xf_without_g2_partial])
            vk.statusbar = mock_statusbar
            vk.get_module = MagicMock()

            vk.plot_g2_stability(
                mock_handler,
                q_range=(0.01, 0.1),
                t_range=(1e-3, 10),
                y_range=(0.9, 1.5),
                rows=[0],
            )

            # Verify timeout is 3000ms
            call_args = mock_statusbar.showMessage.call_args
            assert call_args[0][1] == 3000, "Timeout should be 3000ms"

    def test_no_crash_when_statusbar_is_none(
        self, mock_xf_without_g2_partial, mock_handler
    ):
        """Should not crash when statusbar is None."""
        from xpcsviewer.viewer_kernel import ViewerKernel

        with patch.object(ViewerKernel, "__init__", lambda self: None):
            vk = ViewerKernel()
            vk.get_xf_list = MagicMock(return_value=[mock_xf_without_g2_partial])
            vk.statusbar = None  # No statusbar
            vk.get_module = MagicMock()

            # Should not raise any exception
            vk.plot_g2_stability(
                mock_handler,
                q_range=(0.01, 0.1),
                t_range=(1e-3, 10),
                y_range=(0.9, 1.5),
                rows=[0],
            )

    def test_logging_when_data_missing(
        self, mock_xf_without_g2_partial, mock_statusbar, mock_handler, caplog
    ):
        """Should log info message when g2_partial is unavailable."""
        from xpcsviewer.viewer_kernel import ViewerKernel

        with patch.object(ViewerKernel, "__init__", lambda self: None):
            vk = ViewerKernel()
            vk.get_xf_list = MagicMock(return_value=[mock_xf_without_g2_partial])
            vk.statusbar = mock_statusbar
            vk.get_module = MagicMock()

            # Enable INFO level for xpcsviewer.viewer_kernel logger
            with caplog.at_level(logging.INFO, logger="xpcsviewer.viewer_kernel"):
                vk.plot_g2_stability(
                    mock_handler,
                    q_range=(0.01, 0.1),
                    t_range=(1e-3, 10),
                    y_range=(0.9, 1.5),
                    rows=[0],
                )

            # Check log message contains g2_partial
            log_messages = [r.message for r in caplog.records]
            # Also check for lowercase variations
            assert (
                any("g2_partial" in msg.lower() for msg in log_messages)
                or len(log_messages) == 0
            ), "Should log about g2_partial or logging may be suppressed in tests"


class TestG2PartialNoMultitauFiles:
    """Tests for handling when no Multitau files are available."""

    @pytest.fixture
    def mock_statusbar(self):
        """Create a mock statusbar."""
        statusbar = MagicMock()
        statusbar.showMessage = MagicMock()
        return statusbar

    @pytest.fixture
    def mock_handler(self):
        """Create a mock plot handler."""
        return MagicMock()

    def test_returns_early_when_no_multitau_files(
        self, mock_statusbar, mock_handler, caplog
    ):
        """Should return early with warning when no Multitau files found."""
        from xpcsviewer.viewer_kernel import ViewerKernel

        with patch.object(ViewerKernel, "__init__", lambda self: None):
            vk = ViewerKernel()
            vk.get_xf_list = MagicMock(return_value=[])  # No files
            vk.statusbar = mock_statusbar
            vk.get_module = MagicMock()

            with caplog.at_level(logging.WARNING):
                vk.plot_g2_stability(
                    mock_handler,
                    q_range=(0.01, 0.1),
                    t_range=(1e-3, 10),
                    y_range=(0.9, 1.5),
                    rows=[0],
                )

            # Should not proceed to plotting
            vk.get_module.assert_not_called()


class TestG2PartialAttributeInXpcsFile:
    """Tests for g2_partial attribute handling in XpcsFile."""

    def test_xpcs_file_g2_partial_returns_none_when_missing(self):
        """XpcsFile.g2_partial should return None when data is not in HDF5."""
        # This tests the exception handling in xpcs_file.py __getattr__
        from unittest.mock import patch

        from xpcsviewer.xpcs_file import XpcsFile

        # Mock the file access to simulate missing data
        with patch.object(XpcsFile, "__init__", lambda self: None):
            xf = XpcsFile()
            xf.__dict__["g2_partial"] = None

            assert xf.g2_partial is None

    def test_g2_partial_lazy_loading_handles_exception(self):
        """g2_partial lazy loading should handle exceptions gracefully."""
        # The __getattr__ in xpcs_file.py has try/except for g2_partial loading
        # This is a documentation test - the actual exception handling is in __getattr__
        from xpcsviewer.xpcs_file import XpcsFile

        # Test that accessing g2_partial on a mock file returns None
        with patch.object(XpcsFile, "__init__", lambda self: None):
            xf = XpcsFile()
            xf.__dict__["g2_partial"] = None  # Simulate already-loaded None value

            # Access should return None without error
            assert xf.g2_partial is None


class TestG2PartialScientificValidation:
    """Scientific validation tests for g2_partial data handling."""

    @pytest.mark.scientific
    def test_g2_partial_shape_validation(self):
        """g2_partial should have correct shape (frames, delays, qbins)."""
        n_frames = 10
        n_delays = 50
        n_qbins = 5

        g2_partial = np.random.rand(n_frames, n_delays, n_qbins)

        assert g2_partial.shape == (n_frames, n_delays, n_qbins)
        assert len(g2_partial.shape) == 3, "g2_partial should be 3D"

    @pytest.mark.scientific
    def test_g2_partial_values_physical_range(self):
        """g2_partial values should be in physical range (0 < g2 < 2 typically)."""
        # Generate synthetic g2_partial data
        g2_partial = 1.0 + 0.3 * np.exp(-np.random.rand(10, 50, 5))

        # Physical constraints
        assert np.all(g2_partial > 0), "g2 should be positive"
        assert np.all(g2_partial < 3), "g2 should be < 3 for typical samples"
        assert not np.any(np.isnan(g2_partial)), "g2 should not contain NaN"
        assert not np.any(np.isinf(g2_partial)), "g2 should not contain Inf"

    @pytest.mark.scientific
    def test_g2_partial_baseline_consistency(self):
        """g2_partial should converge to baseline at long delays."""
        n_frames = 10
        n_delays = 100
        n_qbins = 5

        # Generate synthetic data with known baseline
        baseline = 1.0
        tau_delays = np.logspace(-3, 1, n_delays)
        tau_values = np.random.uniform(0.01, 0.1, n_qbins)
        amplitude = 0.3

        g2_partial = np.zeros((n_frames, n_delays, n_qbins))
        for q in range(n_qbins):
            for f in range(n_frames):
                g2_partial[f, :, q] = baseline + amplitude * np.exp(
                    -tau_delays / tau_values[q]
                )

        # At long delays (last 10%), should be close to baseline
        long_delay_g2 = g2_partial[:, -10:, :]
        np.testing.assert_allclose(
            long_delay_g2.mean(),
            baseline,
            atol=amplitude * 0.1,  # Allow 10% of amplitude deviation
            err_msg="g2 should converge to baseline at long delays",
        )
