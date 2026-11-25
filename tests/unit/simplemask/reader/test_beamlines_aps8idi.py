"""Unit tests for reader/beamlines/aps_8idi.py and reader/__init__.get_reader."""

import h5py
import numpy as np
import pytest

from xpcsviewer.simplemask.qmap import compute_qmap
from xpcsviewer.simplemask.reader import get_reader
from xpcsviewer.simplemask.reader.adapter import to_kernel_metadata
from xpcsviewer.simplemask.reader.beamlines.aps_8idi import (
    APS8IDIReader,
    get_nexus_metadata,
)


def _write_8idi_file(
    path,
    n_frames=3,
    shape=(4, 5),
    energy=11.0,
    distance=6.0,
    x_pixel_size=0.000075,
    y_pixel_size=0.000075,
    ccdx=0.0,
    ccdy=0.0,
    ccdx0=0.0,
    ccdy0=0.0,
    bcx0=100.0,
    bcy0=50.0,
):
    with h5py.File(path, "w") as f:
        data = np.full((n_frames, *shape), 8, dtype=np.uint16)
        f.create_dataset("/entry/data/data", data=data)
        f.create_dataset("/entry/instrument/incident_beam/incident_energy", data=energy)
        f.create_dataset("/entry/instrument/detector_1/distance", data=distance)
        f.create_dataset("/entry/instrument/detector_1/x_pixel_size", data=x_pixel_size)
        f.create_dataset("/entry/instrument/detector_1/y_pixel_size", data=y_pixel_size)
        f.create_dataset("/entry/instrument/detector_1/position_x", data=ccdx)
        f.create_dataset("/entry/instrument/detector_1/position_y", data=ccdy)
        f.create_dataset(
            "/entry/instrument/detector_1/beam_center_position_x", data=ccdx0
        )
        f.create_dataset(
            "/entry/instrument/detector_1/beam_center_position_y", data=ccdy0
        )
        f.create_dataset("/entry/instrument/detector_1/beam_center_x", data=bcx0)
        f.create_dataset("/entry/instrument/detector_1/beam_center_y", data=bcy0)


class TestGetNexusMetadata:
    def test_computes_beam_center_from_translation(self, tmp_path):
        fname = str(tmp_path / "scan.h5")
        _write_8idi_file(fname)
        meta = get_nexus_metadata(fname)
        # ccdx == ccdx0 and ccdy == ccdy0 in this fixture, so beam center
        # equals bcx0/bcy0 directly (no translation offset).
        assert meta["energy"] == pytest.approx(11.0)
        assert meta["detector_distance"] == pytest.approx(6.0)
        assert meta["pixel_size"] == pytest.approx(0.000075)
        assert "bcx0" not in meta
        assert "x_pixel_size" not in meta

    def test_computes_beam_center_with_nonzero_translation(self, tmp_path):
        # ccdx != ccdx0 and ccdy != ccdy0 here, so the translation term in
        # get_nexus_metadata()'s formula is nonzero -- a sign flip or
        # swapped x/y axis in that formula must fail this test.
        fname = str(tmp_path / "translated.h5")
        _write_8idi_file(
            fname,
            x_pixel_size=0.0001,
            y_pixel_size=0.0002,
            ccdx=0.0005,
            ccdy=0.0006,
            ccdx0=0.0,
            ccdy0=0.0,
            bcx0=100.0,
            bcy0=50.0,
        )
        meta = get_nexus_metadata(fname)
        # beam_center_x = bcx0 + (ccdx - ccdx0) / x_pixel_size
        #               = 100.0 + 0.0005 / 0.0001 = 105.0
        assert meta["beam_center_x"] == pytest.approx(105.0)
        # beam_center_y = bcy0 + (ccdy - ccdy0) / y_pixel_size
        #               = 50.0 + 0.0006 / 0.0002 = 53.0
        assert meta["beam_center_y"] == pytest.approx(53.0)

    def test_missing_required_field_raises(self, tmp_path):
        fname = str(tmp_path / "incomplete.h5")
        with h5py.File(fname, "w") as f:
            f.create_dataset("/entry/data/data", data=np.zeros((2, 2, 2)))
        with pytest.raises((KeyError, FileNotFoundError)):
            get_nexus_metadata(fname)


class TestAPS8IDIReader:
    def test_prepare_data_success(self, tmp_path):
        fname = str(tmp_path / "scan.h5")
        _write_8idi_file(fname, n_frames=3, shape=(4, 5))
        reader = APS8IDIReader(fname)
        assert reader.ftype == "APS_8IDI"
        assert reader.stype == "Transmission"
        reader.prepare_data()
        assert reader.scat.shape == (4, 5)
        assert reader.metadata_is_placeholder is False
        assert reader.metadata["energy"] == pytest.approx(11.0)

    def test_forwards_kwargs_to_format_loader(self, tmp_path):
        # get_reader(beamline, fname, **kwargs) forwards kwargs all the way
        # to the format loader; a custom data_path must actually work.
        fname = str(tmp_path / "custom_path.h5")
        with h5py.File(fname, "w") as f:
            f.create_dataset(
                "/my/custom/path", data=np.full((2, 3, 4), 9, dtype=np.uint16)
            )
        reader = APS8IDIReader(fname, data_path="/my/custom/path")
        assert reader.shape == (3, 4)

    def test_prepare_data_falls_back_on_bad_metadata(self, tmp_path):
        fname = str(tmp_path / "no_metadata.h5")
        with h5py.File(fname, "w") as f:
            f.create_dataset(
                "/entry/data/data", data=np.full((2, 3, 4), 1, dtype=np.uint16)
            )
        reader = APS8IDIReader(fname)
        reader.prepare_data()
        assert reader.metadata_is_placeholder is True
        # Placeholder metadata still lets prepare_data() complete.
        assert reader.scat.shape == (3, 4)


class TestGetReader:
    def test_aps_8idi_returns_correct_reader_type(self, tmp_path):
        fname = str(tmp_path / "scan.h5")
        _write_8idi_file(fname)
        reader = get_reader("APS_8IDI", fname)
        assert isinstance(reader, APS8IDIReader)

    def test_unsupported_beamline_raises(self, tmp_path):
        fname = str(tmp_path / "scan.h5")
        _write_8idi_file(fname)
        with pytest.raises(ValueError, match="unsupported beamline"):
            get_reader("APS_9IDD", fname)


SWING_PATHS = {
    "swing_angle_horizontal": "/entry/instrument/detector_1/flightpath_swing",
    "swing_angle_vertical": "/entry/instrument/detector_1/flightpath_swing_vertical",
}


def _add_swing(fname, horizontal, vertical):
    with h5py.File(fname, "a") as f:
        f.create_dataset(SWING_PATHS["swing_angle_horizontal"], data=horizontal)
        f.create_dataset(SWING_PATHS["swing_angle_vertical"], data=vertical)


class TestSwingAngles:
    def _kernel_metadata(self, fname):
        reader = APS8IDIReader(fname)
        reader.prepare_data()
        assert reader.metadata_is_placeholder is False
        return to_kernel_metadata(reader.metadata, reader.stype)

    def test_swing_reaches_kernel_metadata_and_changes_qmap(self, tmp_path):
        plain = str(tmp_path / "plain.h5")
        swung = str(tmp_path / "swung.h5")
        _write_8idi_file(plain, shape=(8, 9), bcx0=4.0, bcy0=3.0)
        _write_8idi_file(swung, shape=(8, 9), bcx0=4.0, bcy0=3.0)
        _add_swing(swung, 20.0, 5.0)

        plain_meta = self._kernel_metadata(plain)
        swung_meta = self._kernel_metadata(swung)

        assert swung_meta["swing_angle_horizontal"] == 20.0
        assert swung_meta["swing_angle_vertical"] == 5.0
        assert plain_meta["swing_angle_horizontal"] == 0.0
        assert plain_meta["swing_angle_vertical"] == 0.0
        # Non-swung file: the kernel metadata is otherwise unchanged.
        assert {k: v for k, v in plain_meta.items() if not k.startswith("swing_")} == {
            "bcx": 4.0,
            "bcy": 3.0,
            "pix_dim": pytest.approx(0.075),
            "det_dist": 6000.0,
            "energy": 11.0,
            "shape": (8, 9),
            "stype": "Transmission",
        }

        q_plain = compute_qmap("Transmission", plain_meta)[0]["q"]
        q_swung = compute_qmap("Transmission", swung_meta)[0]["q"]
        assert q_plain[3, 4] == pytest.approx(0.0, abs=1e-12)
        assert q_swung[3, 4] > 0.1

    @pytest.mark.parametrize(
        ("horizontal", "vertical", "field"),
        [
            (float("nan"), 0.0, "swing_angle_horizontal"),
            (float("inf"), 0.0, "swing_angle_horizontal"),
            (0.0, float("nan"), "swing_angle_vertical"),
            (0.0, float("-inf"), "swing_angle_vertical"),
        ],
    )
    def test_non_finite_swing_is_rejected_naming_the_field(
        self, tmp_path, horizontal, vertical, field
    ):
        fname = str(tmp_path / "bad.h5")
        _write_8idi_file(fname)
        _add_swing(fname, horizontal, vertical)
        with pytest.raises(ValueError, match=field):
            self._kernel_metadata(fname)
