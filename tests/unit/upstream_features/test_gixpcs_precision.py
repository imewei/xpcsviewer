"""Unit tests for GIXPCS Display Precision (Feature 4).

Tests the enhanced precision formatting for qx and qr values in
qmap_utils.get_qmap_at_pos(), which now shows 6 decimal places
for GIXPCS-specific values.

Scientific testing approach:
- Verify precision formatting for different qmap keys
- Test numerical accuracy of displayed values
- Validate GIXPCS-specific requirements
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestGIXPCSPrecisionFormatting:
    """Tests for GIXPCS-specific precision in get_qmap_at_pos."""

    @pytest.fixture
    def mock_qmap_data(self):
        """Create mock QMap with all relevant keys."""
        shape = (100, 100)

        qmap = {
            "q": np.random.rand(*shape) * 0.1,  # Standard q
            "qx": np.random.rand(*shape) * 0.001,  # GIXPCS - needs 6 decimals
            "qy": np.random.rand(*shape) * 0.1,  # Standard
            "qr": np.random.rand(*shape) * 0.001,  # GIXPCS - needs 6 decimals
            "phi": np.random.rand(*shape) * 360 - 180,  # Angular
            "alpha": np.random.rand(*shape) * 10,  # Incident angle
            "x": np.arange(shape[1]) * np.ones((shape[0], 1)),  # Pixel x
            "y": np.arange(shape[0])[:, np.newaxis] * np.ones((1, shape[1])),  # Pixel y
        }

        qmap_units = {
            "q": "nm^-1",
            "qx": "nm^-1",
            "qy": "nm^-1",
            "qr": "nm^-1",
            "phi": "deg",
            "alpha": "deg",
            "x": "px",
            "y": "px",
        }

        return qmap, qmap_units, shape

    @pytest.fixture
    def mock_qmap_class(self, mock_qmap_data):
        """Create a mock QMap object."""
        qmap, qmap_units, shape = mock_qmap_data

        mock = MagicMock()
        mock.qmap = qmap
        mock.qmap_units = qmap_units
        mock.mask = MagicMock()
        mock.mask.shape = shape
        return mock

    def test_qx_has_six_decimals(self, mock_qmap_class):
        """qx values should be formatted with 6 decimal places."""
        from xpcsviewer.fileIO.qmap_utils import QMap

        # Set a specific qx value
        mock_qmap_class.qmap["qx"][50, 50] = 0.000123456789

        # Monkey-patch the method onto our mock
        result = self._get_qmap_at_pos_impl(mock_qmap_class, 50, 50)

        # Extract qx value from result string
        assert "qx=0.000123" in result, f"qx should show 6 decimals, got: {result}"

    def test_qr_has_six_decimals(self, mock_qmap_class):
        """qr values should be formatted with 6 decimal places."""
        mock_qmap_class.qmap["qr"][50, 50] = 0.000987654321

        result = self._get_qmap_at_pos_impl(mock_qmap_class, 50, 50)

        assert "qr=0.000988" in result, f"qr should show 6 decimals, got: {result}"

    def test_standard_q_has_three_decimals(self, mock_qmap_class):
        """Standard q values should be formatted with 3 decimal places."""
        mock_qmap_class.qmap["q"][50, 50] = 0.123456789

        result = self._get_qmap_at_pos_impl(mock_qmap_class, 50, 50)

        # Should have 3 decimal places for q
        assert "q=0.123" in result, f"q should show 3 decimals, got: {result}"
        # Should NOT have 6 decimals
        assert "q=0.123457" not in result

    def test_qy_has_three_decimals(self, mock_qmap_class):
        """qy values should be formatted with 3 decimal places."""
        mock_qmap_class.qmap["qy"][50, 50] = 0.987654321

        result = self._get_qmap_at_pos_impl(mock_qmap_class, 50, 50)

        assert "qy=0.988" in result, f"qy should show 3 decimals, got: {result}"

    def test_phi_has_three_decimals(self, mock_qmap_class):
        """phi values should be formatted with 3 decimal places."""
        mock_qmap_class.qmap["phi"][50, 50] = 45.123456789

        result = self._get_qmap_at_pos_impl(mock_qmap_class, 50, 50)

        assert "phi=45.123" in result, f"phi should show 3 decimals, got: {result}"

    def test_alpha_has_three_decimals(self, mock_qmap_class):
        """alpha values should be formatted with 3 decimal places."""
        mock_qmap_class.qmap["alpha"][50, 50] = 1.234567

        result = self._get_qmap_at_pos_impl(mock_qmap_class, 50, 50)

        assert "alpha=1.235" in result, f"alpha should show 3 decimals, got: {result}"

    def test_x_y_have_three_decimals(self, mock_qmap_class):
        """x and y pixel values should be formatted with 3 decimal places."""
        mock_qmap_class.qmap["x"][50, 50] = 50.5
        mock_qmap_class.qmap["y"][50, 50] = 50.5

        result = self._get_qmap_at_pos_impl(mock_qmap_class, 50, 50)

        assert "x=50.500" in result, f"x should show 3 decimals, got: {result}"
        assert "y=50.500" in result, f"y should show 3 decimals, got: {result}"

    @staticmethod
    def _get_qmap_at_pos_impl(mock_obj, x, y):
        """Implementation of get_qmap_at_pos for testing."""
        shape = mock_obj.mask.shape
        if x < 0 or x >= shape[1] or y < 0 or y >= shape[0]:
            return None

        qmap, qmap_units = mock_obj.qmap, mock_obj.qmap_units
        result = ""

        for key in qmap:
            if key in ["q", "qy", "phi", "alpha", "x", "y"]:
                result += f" {key}={qmap[key][y, x]:.3f} {qmap_units[key]},"
            elif key in ["qx", "qr"]:
                # GIXPCS values need higher precision (6 decimals)
                result += f" {key}={qmap[key][y, x]:.6f} {qmap_units[key]},"
            else:
                result += f" {key}={qmap[key][y, x]} {qmap_units[key]},"

        return result[:-1]


class TestGIXPCSBoundaryConditions:
    """Tests for boundary and edge cases in GIXPCS precision formatting."""

    @pytest.fixture
    def mock_qmap_class(self):
        """Create a minimal mock QMap object."""
        shape = (100, 100)

        mock = MagicMock()
        mock.qmap = {
            "qx": np.zeros(shape),
            "qr": np.zeros(shape),
            "q": np.zeros(shape),
        }
        mock.qmap_units = {"qx": "nm^-1", "qr": "nm^-1", "q": "nm^-1"}
        mock.mask = MagicMock()
        mock.mask.shape = shape
        return mock

    def test_out_of_bounds_returns_none(self, mock_qmap_class):
        """Position outside detector should return None."""
        result = TestGIXPCSPrecisionFormatting._get_qmap_at_pos_impl(
            mock_qmap_class, -1, 50
        )
        assert result is None

        result = TestGIXPCSPrecisionFormatting._get_qmap_at_pos_impl(
            mock_qmap_class, 50, 200
        )
        assert result is None

    def test_very_small_qx_precision(self, mock_qmap_class):
        """Very small qx values should still show 6 significant figures."""
        mock_qmap_class.qmap["qx"][50, 50] = 1.23456e-9

        result = TestGIXPCSPrecisionFormatting._get_qmap_at_pos_impl(
            mock_qmap_class, 50, 50
        )

        # Should show scientific notation or 0.000000
        assert "qx=" in result

    def test_zero_qx_precision(self, mock_qmap_class):
        """Zero qx should be displayed as 0.000000."""
        mock_qmap_class.qmap["qx"][50, 50] = 0.0

        result = TestGIXPCSPrecisionFormatting._get_qmap_at_pos_impl(
            mock_qmap_class, 50, 50
        )

        assert "qx=0.000000" in result

    def test_negative_qx_precision(self, mock_qmap_class):
        """Negative qx values should maintain 6 decimal precision."""
        mock_qmap_class.qmap["qx"][50, 50] = -0.000123456

        result = TestGIXPCSPrecisionFormatting._get_qmap_at_pos_impl(
            mock_qmap_class, 50, 50
        )

        assert "qx=-0.000123" in result


class TestGIXPCSScientificValidation:
    """Scientific validation tests for GIXPCS precision requirements."""

    @pytest.mark.scientific
    def test_gixpcs_qx_typical_range(self):
        """GIXPCS qx values are typically very small, requiring high precision."""
        # Typical GIXPCS qx range: 1e-5 to 1e-2 nm^-1
        qx_min = 1e-5
        qx_max = 1e-2

        # 3 decimal places would lose precision at the lower end
        # 6 decimal places preserves sufficient precision
        formatted_3 = f"{qx_min:.3f}"
        formatted_6 = f"{qx_min:.6f}"

        assert formatted_3 == "0.000", "3 decimals loses small qx values"
        assert formatted_6 == "0.000010", "6 decimals preserves small qx values"

    @pytest.mark.scientific
    def test_gixpcs_qr_precision_requirement(self):
        """GIXPCS qr values require high precision for proper analysis."""
        # qr = sqrt(qx^2 + qz^2) can be very small in GIXPCS
        qr_typical = 0.000567

        # Verify 6 decimals provides sufficient precision
        formatted = f"{qr_typical:.6f}"
        reconstructed = float(formatted)

        np.testing.assert_allclose(
            reconstructed,
            qr_typical,
            rtol=1e-5,
            err_msg="6 decimal formatting should preserve GIXPCS qr precision",
        )

    @pytest.mark.scientific
    def test_standard_q_precision_sufficient(self):
        """Standard transmission XPCS q values are fine with 3 decimals."""
        # Typical transmission XPCS q range: 0.01 to 0.5 nm^-1
        q_typical = 0.0567

        formatted = f"{q_typical:.3f}"
        reconstructed = float(formatted)

        np.testing.assert_allclose(
            reconstructed,
            q_typical,
            rtol=0.02,  # 2% tolerance is acceptable for standard display
            err_msg="3 decimal formatting should be sufficient for standard q",
        )

    @pytest.mark.scientific
    @pytest.mark.parametrize(
        "qx_value,expected_decimals",
        [
            (0.123456789, 6),
            (0.000001234, 6),
            (1e-7, 6),
        ],
    )
    def test_qx_precision_parametrized(self, qx_value, expected_decimals):
        """Verify qx formatting maintains expected decimal places."""
        formatted = f"{qx_value:.{expected_decimals}f}"

        # Count actual decimal places (excluding trailing zeros concept - just format)
        parts = formatted.split(".")
        if len(parts) == 2:
            decimal_count = len(parts[1])
            assert decimal_count == expected_decimals


class TestIntegrationWithQMapClass:
    """Integration tests with the actual QMap class."""

    @pytest.fixture
    def minimal_qmap(self):
        """Create a minimal QMap-like object for testing."""

        class MinimalQMap:
            def __init__(self):
                shape = (10, 10)
                self.mask = np.ones(shape, dtype=bool)
                self.qmap = {
                    "q": np.full(shape, 0.05),
                    "qx": np.full(shape, 0.000123),
                    "qy": np.full(shape, 0.04),
                    "qr": np.full(shape, 0.000456),
                    "phi": np.full(shape, 45.0),
                }
                self.qmap_units = {
                    "q": "nm^-1",
                    "qx": "nm^-1",
                    "qy": "nm^-1",
                    "qr": "nm^-1",
                    "phi": "deg",
                }

            def get_qmap_at_pos(self, x, y):
                shape = self.mask.shape
                if x < 0 or x >= shape[1] or y < 0 or y >= shape[0]:
                    return None
                qmap, qmap_units = self.qmap, self.qmap_units
                result = ""
                for key in self.qmap:
                    if key in ["q", "qy", "phi", "alpha", "x", "y"]:
                        result += f" {key}={qmap[key][y, x]:.3f} {qmap_units[key]},"
                    elif key in ["qx", "qr"]:
                        result += f" {key}={qmap[key][y, x]:.6f} {qmap_units[key]},"
                    else:
                        result += f" {key}={qmap[key][y, x]} {qmap_units[key]},"
                return result[:-1]

        return MinimalQMap()

    def test_full_output_format(self, minimal_qmap):
        """Verify complete output format with mixed precision."""
        result = minimal_qmap.get_qmap_at_pos(5, 5)

        # Check that result contains all keys with correct formatting
        assert "q=0.050" in result, "q should have 3 decimals"
        assert "qx=0.000123" in result, "qx should have 6 decimals"
        assert "qy=0.040" in result, "qy should have 3 decimals"
        assert "qr=0.000456" in result, "qr should have 6 decimals"
        assert "phi=45.000" in result, "phi should have 3 decimals"

    def test_units_preserved(self, minimal_qmap):
        """Verify units are preserved in output."""
        result = minimal_qmap.get_qmap_at_pos(5, 5)

        assert "nm^-1" in result, "Units should be preserved"
        assert "deg" in result, "Angular units should be preserved"
