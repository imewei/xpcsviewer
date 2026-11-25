"""Integration checks for regrouping G2 from APS HDF5 files."""

import errno
import os

import h5py
import numpy as np
import pytest

from xpcsviewer.fileIO import g2_regroup, hdf_reader
from xpcsviewer.fileIO.g2_regroup import read_g2_regroup
from xpcsviewer.fileIO.hdf_reader import get_analysis_type

pytestmark = pytest.mark.integration

CUBE_PATH = "/xpcs/multitau/unnormalized_G2"
DELAY_PATH = "/xpcs/multitau/delay_list"
FRAME_TIME_PATH = "/entry/instrument/detector_1/frame_time"
STRIDE_PATH = "/xpcs/multitau/config/stride_frame"
AVERAGE_PATH = "/xpcs/multitau/config/avg_frame"
CHANNELS = np.array([[[4, 12, 24, 10]], [[1, 2, 2, 1]], [[2, 3, 2, 5]]])
STATIC_MAP = np.array([[1, 1, 2, 3]], dtype=np.uint16)
DYNAMIC_MAP = np.array([[1, 1, 1, 2]], dtype=np.uint32)


@pytest.fixture
def source_path(tmp_path):
    path = tmp_path / "source.h5"
    with h5py.File(path, "w") as source:
        source.create_dataset(
            CUBE_PATH,
            data=np.stack([CHANNELS] * 3),
            chunks=(1, *CHANNELS.shape),
            compression="gzip",
        )
        source[DELAY_PATH] = np.array([0, 2, 4], dtype=np.int32)
        source[DELAY_PATH].attrs["units"] = "frames"
        source[FRAME_TIME_PATH] = 0.25
        source[STRIDE_PATH] = 2
        source[AVERAGE_PATH] = 3
    return path


@pytest.fixture
def qmap_path(tmp_path):
    path = tmp_path / "qmap.h5"
    with h5py.File(path, "w") as qmap:
        qmap["/qmap/static_roi_map"] = STATIC_MAP
        qmap["/qmap/dynamic_roi_map"] = DYNAMIC_MAP
    return path


@pytest.fixture
def delay_slices(monkeypatch):
    reads = []
    getitem = h5py.Dataset.__getitem__

    def record_slice(dataset, selection):
        if dataset.name == CUBE_PATH:
            assert isinstance(selection, slice)
            assert selection.start is not None and selection.stop is not None
            assert selection.step in (None, 1)
            reads.append((selection.start, selection.stop))
        return getitem(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", record_slice)
    return reads


@pytest.mark.parametrize(
    ("delays_per_chunk", "expected_slices"),
    [(1, [(0, 1), (1, 2), (2, 3)]), (2, [(0, 2), (2, 3)]), (8, [(0, 3)])],
)
def test_regroup_reads_bounded_delay_slices(
    source_path, qmap_path, delay_slices, delays_per_chunk, expected_slices
):
    result = read_g2_regroup(source_path, qmap_path, delays_per_chunk=delays_per_chunk)

    assert delay_slices == expected_slices
    assert result.source_path == source_path
    assert result.qmap_path == qmap_path
    assert result.delay_unit == "frames"
    assert result.delay_list.dtype == np.int32
    assert result.delay_seconds.dtype == np.float64
    assert result.dynamic_bin_ids.dtype == DYNAMIC_MAP.dtype
    assert result.static_roi_map.dtype == STATIC_MAP.dtype
    assert result.dynamic_roi_map.dtype == DYNAMIC_MAP.dtype
    np.testing.assert_array_equal(result.delay_list, [0, 2, 4])
    np.testing.assert_array_equal(result.delay_seconds, [0, 3, 6])
    np.testing.assert_array_equal(result.dynamic_bin_ids, [1, 2])
    np.testing.assert_array_equal(result.static_roi_map, STATIC_MAP)
    np.testing.assert_array_equal(result.dynamic_roi_map, DYNAMIC_MAP)
    np.testing.assert_allclose(result.normalized_g2, [[61 / 15, 2]] * 3)
    np.testing.assert_allclose(result.normalized_g2_err, [[4 * np.sqrt(6) / 9, 0]] * 3)


def test_regroup_uses_frame_time_without_optional_factors(source_path, qmap_path):
    with h5py.File(source_path, "r+") as source:
        del source[STRIDE_PATH]
        del source[AVERAGE_PATH]
        del source[DELAY_PATH].attrs["units"]

    result = read_g2_regroup(source_path, qmap_path)

    assert result.delay_unit is None
    np.testing.assert_array_equal(result.delay_seconds, [0, 0.5, 1.0])


@pytest.mark.parametrize(
    ("root_complete", "nested_complete", "expected"),
    [(True, True, [1, 2]), (False, True, [7, 9]), (False, False, None)],
    ids=["root-precedence", "nested-fallback", "split-pair"],
)
def test_regroup_selects_one_complete_qmap_group(
    source_path, qmap_path, root_complete, nested_complete, expected
):
    with h5py.File(qmap_path, "r+") as qmap:
        if not root_complete:
            del qmap["/qmap/dynamic_roi_map"]
        if nested_complete:
            qmap["/xpcs/qmap/static_roi_map"] = STATIC_MAP
        qmap["/xpcs/qmap/dynamic_roi_map"] = [[7, 7, 7, 9]]

    if expected is None:
        with pytest.raises(ValueError, match="ROI maps"):
            read_g2_regroup(source_path, qmap_path)
    else:
        result = read_g2_regroup(source_path, qmap_path)
        np.testing.assert_array_equal(result.dynamic_bin_ids, expected)


@pytest.mark.parametrize(
    ("delay_shape", "expected", "seconds"),
    [
        ((3,), [0, 2, 4], [0, 3, 6]),
        ((1, 3), [0, 2, 4], [0, 3, 6]),
        ((3, 1), [0, 2, 4], [0, 3, 6]),
        ((1, 1), [2], [3]),
    ],
)
def test_regroup_accepts_viewer_delay_and_timing_shapes(
    source_path, qmap_path, delay_shape, expected, seconds
):
    with h5py.File(source_path, "r+") as source:
        del source[DELAY_PATH]
        source[DELAY_PATH] = np.array(expected).reshape(delay_shape)
        source[DELAY_PATH].attrs["units"] = np.bytes_("frames")
        del source[STRIDE_PATH]
        source[STRIDE_PATH] = [2.0]
        del source[AVERAGE_PATH]
        source[AVERAGE_PATH] = [[3.0]]
        if len(expected) == 1:
            del source[CUBE_PATH]
            source[CUBE_PATH] = CHANNELS[np.newaxis]

    result = read_g2_regroup(source_path, qmap_path)

    assert result.delay_list.shape == (len(expected),)
    assert result.delay_unit == b"frames"
    np.testing.assert_array_equal(result.delay_list, expected)
    np.testing.assert_array_equal(result.delay_seconds, seconds)


def test_regroup_applies_timing_factors_before_converting_delays(
    source_path, qmap_path
):
    with h5py.File(source_path, "r+") as source:
        del source[CUBE_PATH]
        source[CUBE_PATH] = CHANNELS[np.newaxis]
        del source[DELAY_PATH]
        source[DELAY_PATH] = [1e-100]
        source[FRAME_TIME_PATH][()] = 1e-250
        del source[STRIDE_PATH]
        source[STRIDE_PATH] = 1e150
        source[AVERAGE_PATH][()] = 1

    result = read_g2_regroup(source_path, qmap_path)

    np.testing.assert_allclose(result.delay_seconds, [1e-200], rtol=1e-12, atol=0)


@pytest.mark.skipif(
    np.finfo(np.longdouble).minexp >= np.finfo(np.float64).minexp,
    reason="requires a wider exponent range than float64",
)
def test_regroup_preserves_wider_delays_and_timing(source_path, qmap_path):
    delays = np.array(["1e350"], dtype=np.longdouble)
    with h5py.File(source_path, "r+") as source:
        del source[CUBE_PATH]
        source[CUBE_PATH] = CHANNELS[np.newaxis]
        del source[DELAY_PATH]
        source[DELAY_PATH] = delays
        del source[FRAME_TIME_PATH]
        source[FRAME_TIME_PATH] = np.longdouble("1e-500")
        del source[STRIDE_PATH]
        del source[AVERAGE_PATH]

    result = read_g2_regroup(source_path, qmap_path)

    assert result.delay_list.dtype == result.delay_seconds.dtype == np.longdouble
    np.testing.assert_array_equal(result.delay_list, delays)
    np.testing.assert_allclose(result.delay_seconds, [1e-150], rtol=1e-12, atol=0)


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (CUBE_PATH, None, "unnormalized_G2"),
        (CUBE_PATH, np.ones((3, 3, 4)), "unnormalized_G2"),
        (CUBE_PATH, np.ones((3, 2, 1, 4)), "unnormalized_G2"),
        (CUBE_PATH, np.ones((3, 3, 1, 4), dtype=complex), "unnormalized_G2"),
        (DELAY_PATH, [], "delay_list"),
        (DELAY_PATH, [0, 2], "delay_list"),
        (DELAY_PATH, [[0, 2], [4, 6]], "delay_list"),
        (DELAY_PATH, 2, "delay_list"),
        (DELAY_PATH, [0, 2, -4], "delay_list"),
        (DELAY_PATH, [0, 2, np.inf], "delay_list"),
        (DELAY_PATH, [0, 2, np.nan], "delay_list"),
        (DELAY_PATH, [0, 0, 0], "delay_list"),
        (DELAY_PATH, [b"0", b"2", b"4"], "delay_list"),
        (FRAME_TIME_PATH, None, "frame_time"),
        (FRAME_TIME_PATH, [0.25, 0.25], "frame_time"),
        (FRAME_TIME_PATH, 0, "frame_time"),
        (FRAME_TIME_PATH, np.nan, "frame_time"),
        (FRAME_TIME_PATH, b"0.25", "frame_time"),
        (FRAME_TIME_PATH, 1e308, "finite"),
        (STRIDE_PATH, None, "timing"),
        (AVERAGE_PATH, None, "timing"),
        (STRIDE_PATH, 0, "stride_frame"),
        (AVERAGE_PATH, 1.5, "avg_frame"),
    ],
)
def test_regroup_rejects_incompatible_source(
    source_path, qmap_path, path, replacement, message
):
    with h5py.File(source_path, "r+") as source:
        del source[path]
        if replacement is not None:
            source[path] = replacement

    with pytest.raises(ValueError, match=message):
        read_g2_regroup(source_path, qmap_path)


def test_regroup_rejects_qmap_that_does_not_match_detector(source_path, qmap_path):
    with h5py.File(qmap_path, "w") as qmap:
        qmap["/qmap/static_roi_map"] = STATIC_MAP.T
        qmap["/qmap/dynamic_roi_map"] = DYNAMIC_MAP.T

    with pytest.raises(ValueError, match="detector shape"):
        read_g2_regroup(source_path, qmap_path)


def test_regroup_reports_bad_delay_index_and_honors_cancel(
    source_path, qmap_path, delay_slices
):
    with h5py.File(source_path, "r+") as source:
        source[CUBE_PATH][1, 1, 0, 0] = 0

    with pytest.raises(ValueError, match="delay index 1"):
        read_g2_regroup(source_path, qmap_path, delays_per_chunk=2)

    delay_slices.clear()

    def check_cancelled():
        if delay_slices:
            raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError, match="cancelled"):
        read_g2_regroup(source_path, qmap_path, check_cancelled=check_cancelled)

    assert delay_slices == [(0, 1)]


def test_regroup_export_round_trips_data_and_provenance(
    tmp_path, source_path, qmap_path
):
    input_bytes = source_path.read_bytes(), qmap_path.read_bytes()
    result = read_g2_regroup(source_path, qmap_path)
    output = tmp_path / "derived.h5"

    g2_regroup.export_g2_regroup(result, output)

    with h5py.File(output, "r") as derived:
        multitau = derived["/xpcs/multitau"]
        for name in (
            "delay_list",
            "delay_seconds",
            "dynamic_bin_ids",
            "normalized_g2",
            "normalized_g2_err",
        ):
            np.testing.assert_array_equal(multitau[name][()], getattr(result, name))
        assert multitau["delay_list"].attrs["units"] == "frames"
        assert multitau["delay_seconds"].attrs["units"] == "s"
        assert multitau.attrs["source_filename"] == source_path.name
        assert multitau.attrs["qmap_filename"] == qmap_path.name
        for name in ("static_roi_map", "dynamic_roi_map"):
            np.testing.assert_array_equal(
                derived["/qmap"][name][()], getattr(result, name)
            )

    assert (source_path.read_bytes(), qmap_path.read_bytes()) == input_bytes
    assert set(tmp_path.iterdir()) == {source_path, qmap_path, output}


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX byte filenames")
def test_regroup_export_round_trips_native_filename_bytes(
    tmp_path, source_path, qmap_path
):
    source_bytes, qmap_bytes = b"source-\xff.h5", b"qmap-\xfe.h5"
    source_name, qmap_name = os.fsdecode(source_bytes), os.fsdecode(qmap_bytes)
    if "\udcff" not in source_name or "\udcfe" not in qmap_name:
        pytest.skip(
            "filesystem encoding decodes the filenames without surrogate escape"
        )
    try:
        source_path = source_path.rename(tmp_path / source_name)
        qmap_path = qmap_path.rename(tmp_path / qmap_name)
    except OSError as exc:
        if exc.errno != errno.EILSEQ:
            raise
        pytest.skip("filesystem rejects non-UTF-8 filenames (e.g. macOS APFS)")
    output = tmp_path / "derived.h5"

    g2_regroup.export_g2_regroup(read_g2_regroup(source_path, qmap_path), output)

    with h5py.File(output, "r") as derived:
        attrs = derived["/xpcs/multitau"].attrs
        assert os.fsencode(attrs["source_filename"]) == source_bytes
        assert os.fsencode(attrs["qmap_filename"]) == qmap_bytes
    assert set(tmp_path.iterdir()) == {source_path, qmap_path, output}


def test_regroup_export_is_not_a_viewer_result(tmp_path, source_path, qmap_path):
    output = tmp_path / "derived.h5"
    g2_regroup.export_g2_regroup(read_g2_regroup(source_path, qmap_path), output)

    with h5py.File(output, "r") as derived:
        assert derived.attrs["xpcsviewer_artifact"] == "g2_regroup"
    with pytest.raises(ValueError, match="derived G2 export"):
        get_analysis_type(str(output), use_pool=False)
    assert get_analysis_type(str(source_path), use_pool=False) == ("Multitau",)


def test_regroup_export_detection_keeps_shared_handle_open(tmp_path, monkeypatch):
    path = str(tmp_path / "derived.h5")
    with h5py.File(path, "w") as derived:
        derived.attrs["xpcsviewer_artifact"] = "g2_regroup"

    pool = hdf_reader.HDF5ConnectionPool()
    monkeypatch.setattr(hdf_reader, "_connection_pool", pool)
    try:
        with pool.get_connection(path) as open_file:
            with pytest.raises(ValueError, match="derived G2 export"):
                get_analysis_type(path)
            assert open_file.attrs["xpcsviewer_artifact"] == "g2_regroup"
    finally:
        pool.clear_pool()


@pytest.mark.parametrize("marker", ["g2_regroup_preview", [b"g2_regroup", b"other"]])
def test_regroup_export_detection_keeps_other_marker_values(source_path, marker):
    with h5py.File(source_path, "r+") as source:
        source.attrs["xpcsviewer_artifact"] = marker
        source.create_group("/xpcs/twotime")

    assert get_analysis_type(str(source_path), use_pool=False) == (
        "Multitau",
        "Twotime",
    )


@pytest.mark.parametrize(
    "existing",
    [
        "output",
        "source",
        "qmap",
        "source_symlink",
        "qmap_symlink",
        "source_hardlink",
        "qmap_hardlink",
    ],
)
def test_regroup_export_rejects_existing_path_and_aliases(
    tmp_path, source_path, qmap_path, existing
):
    destination = tmp_path / "derived.h5"
    if existing == "output":
        destination.write_bytes(b"existing")
    elif existing in ("source", "qmap"):
        destination = source_path if existing == "source" else qmap_path
    else:
        target = source_path if existing.startswith("source") else qmap_path
        try:
            if existing.endswith("symlink"):
                destination.symlink_to(target)
            else:
                os.link(target, destination)
        except OSError as exc:
            pytest.skip(f"Filesystem does not support {existing}: {exc}")

    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(FileExistsError):
        g2_regroup.export_g2_regroup(
            read_g2_regroup(source_path, qmap_path), destination
        )

    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_regroup_export_refuses_publication_race(
    tmp_path, source_path, qmap_path, monkeypatch
):
    result = read_g2_regroup(source_path, qmap_path)
    output = tmp_path / "derived.h5"
    link = os.link

    def competing_writer(temporary, destination):
        output.write_bytes(b"winner")
        link(temporary, destination)

    monkeypatch.setattr(os, "link", competing_writer)
    with pytest.raises(FileExistsError):
        g2_regroup.export_g2_regroup(result, output)

    assert output.read_bytes() == b"winner"
    assert set(tmp_path.iterdir()) == {source_path, qmap_path, output}


@pytest.mark.parametrize("failure", ["write", "publication"])
def test_regroup_export_cleans_up_failed_write_or_publication(
    tmp_path, source_path, qmap_path, monkeypatch, failure
):
    result = read_g2_regroup(source_path, qmap_path)
    output = tmp_path / "derived.h5"

    def fail_write(*args, **kwargs):
        raise RuntimeError("simulated dataset write failure")

    def fail_link(*args, **kwargs):
        raise OSError(errno.ENOTSUP, "hard links unavailable")

    if failure == "write":
        monkeypatch.setattr(h5py.Group, "create_dataset", fail_write)
        expected = pytest.raises(RuntimeError, match="simulated dataset write failure")
    else:
        monkeypatch.setattr(os, "link", fail_link)
        expected = pytest.raises(OSError, match="choose another destination")

    with expected:
        g2_regroup.export_g2_regroup(result, output)

    assert set(tmp_path.iterdir()) == {source_path, qmap_path}


def test_regroup_rejects_positive_delay_that_underflows_to_zero_seconds(
    source_path, qmap_path
):
    with h5py.File(source_path, "r+") as source:
        del source[DELAY_PATH], source[FRAME_TIME_PATH]
        del source[STRIDE_PATH], source[AVERAGE_PATH]
        source[DELAY_PATH] = [0, 1e-200, 2e-200]
        source[FRAME_TIME_PATH] = 1e-200

    with pytest.raises(ValueError, match="underflow"):
        read_g2_regroup(source_path, qmap_path)
