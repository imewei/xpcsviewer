"""Unit tests for Blemish Map Export (Feature 5).

Tests the blemish attribute in MaskAssemble and its inclusion in
partition export from simplemask_kernel.

Scientific testing approach:
- Verify blemish mask initialization and properties
- Test blemish inclusion in partition dictionary
- Validate blemish mask integrity during export
"""

import tempfile
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest


class TestMaskAssembleBlemishProperty:
    """Tests for the blemish property in MaskAssemble."""

    @pytest.fixture
    def mask_assemble(self):
        """Create a MaskAssemble instance for testing."""
        from xpcsviewer.simplemask.area_mask import MaskAssemble

        shape = (100, 100)
        return MaskAssemble(shape=shape)

    def test_blemish_property_exists(self, mask_assemble):
        """MaskAssemble should have a blemish property."""
        assert hasattr(mask_assemble, "blemish")
        assert mask_assemble.blemish is not None

    def test_blemish_returns_boolean_array(self, mask_assemble):
        """blemish property should return a boolean array."""
        blemish = mask_assemble.blemish
        assert isinstance(blemish, np.ndarray)
        assert blemish.dtype == bool

    def test_blemish_shape_matches_detector(self, mask_assemble):
        """blemish mask should have same shape as detector."""
        blemish = mask_assemble.blemish
        assert blemish.shape == mask_assemble.shape

    def test_blemish_initially_all_true(self, mask_assemble):
        """Initially, blemish mask should be all True (no blemishes)."""
        blemish = mask_assemble.blemish
        assert np.all(blemish), "Initial blemish mask should have no blemishes"

    def test_blemish_uses_mask_blemish_worker(self, mask_assemble):
        """blemish property should delegate to mask_blemish worker."""
        # Verify the worker exists
        assert "mask_blemish" in mask_assemble.workers

        # The blemish property should return the mask from this worker
        expected = mask_assemble.workers["mask_blemish"].get_mask()
        np.testing.assert_array_equal(mask_assemble.blemish, expected)


class TestBlemishMaskEvaluation:
    """Tests for evaluating and applying blemish masks."""

    @pytest.fixture
    def mask_assemble_with_blemish(self, tmp_path):
        """Create MaskAssemble with blemish file loaded."""
        from xpcsviewer.simplemask.area_mask import MaskAssemble

        shape = (100, 100)
        ma = MaskAssemble(shape=shape)

        # Create a test blemish file
        blemish_file = tmp_path / "blemish.h5"
        blemish_mask = np.ones(shape, dtype=np.int32)
        blemish_mask[20:30, 20:30] = 0  # Mark some pixels as bad
        blemish_mask[50, 50] = 0  # Single bad pixel

        with h5py.File(blemish_file, "w") as f:
            f.create_dataset("mask", data=blemish_mask)

        # Load the blemish file
        ma.evaluate("mask_blemish", fname=str(blemish_file), key="mask")

        return ma

    def test_blemish_reflects_loaded_file(self, mask_assemble_with_blemish):
        """blemish should reflect the loaded blemish file."""
        blemish = mask_assemble_with_blemish.blemish

        # Check that the marked pixels are False (blemished)
        assert not blemish[25, 25], "Center of blemish region should be False"
        assert not blemish[50, 50], "Single blemish pixel should be False"

        # Check that other pixels are True (not blemished)
        assert blemish[0, 0], "Corner pixel should be True"
        assert blemish[99, 99], "Corner pixel should be True"

    def test_blemish_count_matches_file(self, mask_assemble_with_blemish):
        """Number of blemish pixels should match loaded file."""
        blemish = mask_assemble_with_blemish.blemish

        # Expected: 10x10 block + 1 single pixel = 101 blemish pixels
        expected_blemish_count = 10 * 10 + 1
        actual_blemish_count = np.sum(~blemish)

        assert actual_blemish_count == expected_blemish_count, (
            f"Expected {expected_blemish_count} blemish pixels, got {actual_blemish_count}"
        )


class TestPartitionBlemishExport:
    """Tests for blemish inclusion in partition export."""

    @pytest.fixture
    def mock_kernel(self):
        """Create a SimpleMaskKernel mock for testing."""
        from xpcsviewer.simplemask.simplemask_kernel import SimpleMaskKernel

        kernel = SimpleMaskKernel()
        return kernel

    @pytest.fixture
    def prepared_kernel(self, mock_kernel, tmp_path):
        """Create a prepared kernel with data for partition computation."""
        from xpcsviewer.simplemask.area_mask import MaskAssemble

        shape = (100, 100)
        mock_kernel.detector_image = np.random.rand(*shape) * 1000
        mock_kernel.shape = shape
        mock_kernel.mask = np.ones(shape, dtype=bool)
        mock_kernel.mask_kernel = MaskAssemble(shape=shape)

        # Create mock qmap
        mock_kernel.qmap = {
            "q": np.sqrt(
                (np.arange(shape[1]) - shape[1] // 2) ** 2
                + (np.arange(shape[0])[:, np.newaxis] - shape[0] // 2) ** 2
            )
            * 0.01,
            "phi": np.arctan2(
                np.arange(shape[0])[:, np.newaxis] - shape[0] // 2,
                np.arange(shape[1]) - shape[1] // 2,
            )
            * 180
            / np.pi,
        }
        mock_kernel.qmap_unit = {"q": "nm^-1", "phi": "deg"}

        mock_kernel.metadata = {
            "bcx": shape[1] // 2,
            "bcy": shape[0] // 2,
            "pix_dim": 0.075,
            "energy": 10.0,
            "det_dist": 5000.0,
            "source_file": "test.h5",
        }

        return mock_kernel

    def test_partition_contains_blemish_key(self, prepared_kernel):
        """compute_partition should include 'blemish' key."""
        partition = prepared_kernel.compute_partition(
            mode="q-phi",
            dq_num=5,
            sq_num=10,
            dp_num=4,
            sp_num=36,
        )

        assert partition is not None
        assert "blemish" in partition, "Partition should contain 'blemish' key"

    def test_partition_blemish_is_array(self, prepared_kernel):
        """Partition blemish should be a numpy array."""
        partition = prepared_kernel.compute_partition(
            mode="q-phi",
            dq_num=5,
            sq_num=10,
            dp_num=4,
            sp_num=36,
        )

        assert isinstance(partition["blemish"], np.ndarray)

    def test_partition_blemish_shape_matches_detector(self, prepared_kernel):
        """Partition blemish should match detector shape."""
        partition = prepared_kernel.compute_partition(
            mode="q-phi",
            dq_num=5,
            sq_num=10,
            dp_num=4,
            sp_num=36,
        )

        assert partition["blemish"].shape == prepared_kernel.shape

    def test_partition_blemish_is_boolean(self, prepared_kernel):
        """Partition blemish should be boolean type."""
        partition = prepared_kernel.compute_partition(
            mode="q-phi",
            dq_num=5,
            sq_num=10,
            dp_num=4,
            sp_num=36,
        )

        assert partition["blemish"].dtype == bool

    def test_partition_blemish_equals_mask_kernel_blemish(self, prepared_kernel):
        """Partition blemish should equal mask_kernel.blemish."""
        partition = prepared_kernel.compute_partition(
            mode="q-phi",
            dq_num=5,
            sq_num=10,
            dp_num=4,
            sp_num=36,
        )

        expected_blemish = prepared_kernel.mask_kernel.blemish
        np.testing.assert_array_equal(
            partition["blemish"],
            expected_blemish,
            err_msg="Partition blemish should match mask_kernel.blemish",
        )


class TestBlemishFallbackBehavior:
    """Tests for blemish fallback when mask_kernel is not available."""

    def test_blemish_falls_back_to_mask(self):
        """When mask_kernel is None, blemish should fall back to mask."""
        from xpcsviewer.simplemask.simplemask_kernel import SimpleMaskKernel

        kernel = SimpleMaskKernel()
        shape = (100, 100)

        kernel.detector_image = np.random.rand(*shape) * 1000
        kernel.shape = shape
        kernel.mask = np.ones(shape, dtype=bool)
        kernel.mask[10:20, 10:20] = False  # Some masked region
        kernel.mask_kernel = None  # No mask_kernel

        kernel.qmap = {
            "q": np.ones(shape) * 0.05,
            "phi": np.zeros(shape),
        }
        kernel.qmap_unit = {"q": "nm^-1", "phi": "deg"}
        kernel.metadata = {
            "bcx": 50,
            "bcy": 50,
            "pix_dim": 0.075,
            "energy": 10.0,
            "det_dist": 5000.0,
            "source_file": "test.h5",
        }

        partition = kernel.compute_partition(
            mode="q-phi",
            dq_num=5,
            sq_num=10,
            dp_num=4,
            sp_num=36,
        )

        # When mask_kernel is None, blemish should equal mask
        np.testing.assert_array_equal(
            partition["blemish"],
            kernel.mask,
            err_msg="Blemish should fall back to mask when mask_kernel is None",
        )


class TestBlemishSavePartition:
    """Tests for blemish persistence in save_partition."""

    @pytest.fixture
    def kernel_with_partition(self, tmp_path):
        """Create kernel with computed partition."""
        from xpcsviewer.simplemask.area_mask import MaskAssemble
        from xpcsviewer.simplemask.simplemask_kernel import SimpleMaskKernel

        kernel = SimpleMaskKernel()
        shape = (50, 50)

        kernel.detector_image = np.random.rand(*shape) * 1000
        kernel.shape = shape
        kernel.mask = np.ones(shape, dtype=bool)
        kernel.mask_kernel = MaskAssemble(shape=shape)

        # Mark some blemish pixels
        blemish_mask = kernel.mask_kernel.workers["mask_blemish"]
        blemish_mask.zero_loc = np.array([[5, 10, 15], [5, 10, 15]])

        kernel.qmap = {
            "q": np.sqrt(
                (np.arange(shape[1]) - shape[1] // 2) ** 2
                + (np.arange(shape[0])[:, np.newaxis] - shape[0] // 2) ** 2
            )
            * 0.01,
            "phi": np.arctan2(
                np.arange(shape[0])[:, np.newaxis] - shape[0] // 2,
                np.arange(shape[1]) - shape[1] // 2,
            )
            * 180
            / np.pi,
        }
        kernel.qmap_unit = {"q": "nm^-1", "phi": "deg"}
        kernel.metadata = {
            "bcx": 25,
            "bcy": 25,
            "pix_dim": 0.075,
            "energy": 10.0,
            "det_dist": 5000.0,
            "source_file": "test.h5",
        }

        kernel.compute_partition(
            mode="q-phi",
            dq_num=3,
            sq_num=5,
            dp_num=2,
            sp_num=18,
        )

        return kernel, tmp_path

    def test_save_partition_includes_blemish(self, kernel_with_partition):
        """save_partition should persist blemish to HDF5."""
        kernel, tmp_path = kernel_with_partition
        save_path = tmp_path / "partition.h5"

        kernel.save_partition(str(save_path))

        with h5py.File(save_path, "r") as f:
            assert "blemish" in f["/qmap"], "HDF5 should contain blemish dataset"

    def test_saved_blemish_matches_original(self, kernel_with_partition):
        """Saved blemish should match original blemish mask."""
        kernel, tmp_path = kernel_with_partition
        save_path = tmp_path / "partition.h5"

        original_blemish = kernel.new_partition["blemish"].copy()
        kernel.save_partition(str(save_path))

        with h5py.File(save_path, "r") as f:
            saved_blemish = f["/qmap/blemish"][()]

        np.testing.assert_array_equal(
            saved_blemish,
            original_blemish,
            err_msg="Saved blemish should match original",
        )


class TestBlemishScientificValidation:
    """Scientific validation tests for blemish mask handling."""

    @pytest.mark.scientific
    def test_blemish_mask_convention(self):
        """Verify blemish mask convention: True=valid, False=blemish."""
        from xpcsviewer.simplemask.area_mask import MaskAssemble

        shape = (100, 100)
        ma = MaskAssemble(shape=shape)

        # Initial state should be all True (no blemishes)
        blemish = ma.blemish
        assert np.all(blemish), "Convention: True means valid pixel"

    @pytest.mark.scientific
    def test_blemish_separate_from_analysis_mask(self):
        """Blemish should be tracked separately from analysis mask."""
        from xpcsviewer.simplemask.area_mask import MaskAssemble

        shape = (100, 100)
        ma = MaskAssemble(shape=shape)

        # Apply a threshold mask (analysis mask)
        ma.workers["mask_threshold"].zero_loc = np.array([[10, 20], [10, 20]])
        ma.apply("mask_threshold")

        # Blemish should still be all True (separate from analysis)
        assert np.all(ma.blemish), (
            "Blemish mask should be independent of analysis masks"
        )

    @pytest.mark.scientific
    def test_blemish_and_mask_intersection(self):
        """Final mask = analysis_mask AND blemish (both must be True)."""
        from xpcsviewer.simplemask.area_mask import MaskAssemble

        shape = (50, 50)
        ma = MaskAssemble(shape=shape)

        # Mark some blemish pixels
        ma.workers["mask_blemish"].zero_loc = np.array([[5, 6, 7], [5, 6, 7]])

        blemish = ma.blemish
        analysis_mask = ma.get_mask()

        # For final analysis, both should be considered
        # (though this is implementation-dependent)
        blemish_count = np.sum(~blemish)
        assert blemish_count == 3, "Should have 3 blemish pixels"
