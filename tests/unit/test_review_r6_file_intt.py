"""Round 6 regressions in xpcs_file.py and module/intt.py: R6-08, R6-09, R6-20.

R6-08  the Intensity-vs-time FFT frequency axis uses the same sample interval
       (``xf.t0``) as the tab's time axis, so a known sinusoid peaks at its true
       frequency in Hz.
R6-09  metadata stored as one-element arrays (real NeXus files store scalars both
       ways) neither breaks ``get_pg_tree`` nor leaks into timing arithmetic.
R6-20  loading a file does not call the enhanced reader with options it does not
       declare, so nothing is logged as an error and nothing is read twice.

Every test writes a real NeXus-layout HDF5 file with h5py and loads it through
the real ``XpcsFile``.
"""

from __future__ import annotations

import logging
import sys

import h5py
import numpy as np
import pyqtgraph as pg
import pytest
from qtpy.QtWidgets import QApplication

from tests.unit.test_review_r3_maps_roi import K, _write_nexus
from xpcsviewer.fileIO import hdf_reader
from xpcsviewer.fileIO.hdf_reader import clear_connection_pool
from xpcsviewer.module import intt
from xpcsviewer.xpcs_file import XpcsFile

pytestmark = pytest.mark.unit

N = 32
PERIOD_FRAMES = 8


@pytest.fixture(autouse=True)
def _fresh_pool():
    clear_connection_pool()
    yield
    clear_connection_pool()


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _sinusoid_file(path, frame_time, *, period=PERIOD_FRAMES, n=N, as_array=False):
    """Real file whose intensity is 10 + 2 cos(2 pi frame / period)."""
    _write_nexus(path)
    frames = np.arange(n)
    signal = 10 + 2 * np.cos(2 * np.pi * frames / period)
    with h5py.File(path, "r+") as f:
        del f[K["t0"]]
        f[K["t0"]] = [frame_time] if as_array else frame_time
        del f[K["Int_t"]]
        f[K["Int_t"]] = np.stack([frames, signal])
    return path


def _module():
    # xpcsviewer.xpcs_file is a package; the class lives in the sibling module
    return sys.modules[XpcsFile.__module__]


def _peak(xf):
    f, y = xf.Int_t_fft
    return float(f[np.argmax(y)]), float(np.max(y)), f


# ------------------------------------------------------------------ R6-08
@pytest.mark.parametrize("frame_time", [0.25, 1.0, 0.004, 2.5])
def test_r6_08_fft_peak_is_in_hz_for_any_frame_time(tmp_path, frame_time):
    xf = XpcsFile(str(_sinusoid_file(tmp_path / "s.h5", frame_time)))
    peak, height, freqs = _peak(xf)
    assert peak == pytest.approx(1.0 / (PERIOD_FRAMES * frame_time), rel=1e-12)
    # amplitude convention unchanged: |sum| = N * A / 2 for a pure cosine
    assert height == pytest.approx(N * 2 / 2)
    assert freqs[0] > 0  # DC stays excluded


def test_r6_08_fft_reaches_nyquist_of_the_real_sample_rate(tmp_path):
    # period 2 frames at 0.25 s: 2 Hz, exactly the Nyquist frequency of 4 Hz sampling
    xf = XpcsFile(str(_sinusoid_file(tmp_path / "n.h5", 0.25, period=2)))
    peak, _, freqs = _peak(xf)
    assert peak == pytest.approx(2.0)
    assert freqs[-1] == pytest.approx(2.0)


def test_r6_08_one_element_frame_time_array_gives_same_axis(tmp_path):
    scalar = XpcsFile(str(_sinusoid_file(tmp_path / "a.h5", 0.25)))
    array = XpcsFile(str(_sinusoid_file(tmp_path / "b.h5", 0.25, as_array=True)))
    np.testing.assert_allclose(array.Int_t_fft[0], scalar.Int_t_fft[0])
    np.testing.assert_allclose(array.Int_t_fft[1], scalar.Int_t_fft[1])


def test_r6_08_tab_time_axis_and_fft_axis_share_one_interval(tmp_path, qapp):
    xf = XpcsFile(str(_sinusoid_file(tmp_path / "t.h5", 0.25)))
    widget = pg.GraphicsLayoutWidget()
    intt.plot([xf], widget, xlabel="Time (second)")
    t, _ = widget.getItem(0, 0).listDataItems()[0].getData()
    freq, amp = widget.getItem(1, 0).listDataItems()[0].getData()
    dt_plot = float(t[1] - t[0])
    assert dt_plot == pytest.approx(0.25)
    # the Fourier bin spacing is 1 / (N * dt) of the very samples that were drawn
    assert float(freq[1] - freq[0]) == pytest.approx(1.0 / (len(t) * dt_plot))
    assert float(freq[np.argmax(amp)]) == pytest.approx(0.5)


def test_r6_08_fft_ignores_window_and_sampling_of_the_plot(tmp_path, qapp):
    # the FFT is always of the full raw series, whose spacing is t0 whatever the
    # time-series smoothing does; the peak stays at 0.5 Hz
    xf = XpcsFile(str(_sinusoid_file(tmp_path / "w.h5", 0.25)))
    widget = pg.GraphicsLayoutWidget()
    intt.plot([xf], widget, xlabel="Time (second)", window=3, sampling=2)
    freq, amp = widget.getItem(1, 0).listDataItems()[0].getData()
    assert float(freq[np.argmax(amp)]) == pytest.approx(0.5)


# ------------------------------------------------------------------ R6-09
def _set_scalar(path, key, value):
    with h5py.File(path, "r+") as f:
        del f[K[key]]
        f[K[key]] = value


@pytest.mark.parametrize("wrap", [lambda v: [v], lambda v: [[v]]], ids=["1d", "2d"])
def test_r6_09_one_element_timing_arrays_become_scalars(tmp_path, wrap):
    path = tmp_path / "m.h5"
    _write_nexus(path)
    _set_scalar(path, "t0", wrap(0.1))
    _set_scalar(path, "t1", wrap(0.2))
    _set_scalar(path, "stride_frame", wrap(2))
    _set_scalar(path, "avg_frame", wrap(3))
    xf = XpcsFile(str(path))
    for name in ("t0", "t1", "g2_t0"):
        assert np.ndim(getattr(xf, name)) == 0, name
    assert float(xf.t0) == pytest.approx(0.1 * 2 * 3)
    assert float(xf.t1) == pytest.approx(0.2)
    np.testing.assert_allclose(xf.t_el, xf.tau * 0.6)


def test_r6_09_twotime_one_element_timing_gives_scalar_c2_t0(tmp_path):
    from tests.unit.test_review_r3_fit_status import _write_file

    path = tmp_path / "tt.h5"
    _write_file(path, [0.1, 0.2, 0.4], processed=[1])
    with h5py.File(path, "r+") as f:
        for key in (
            "/xpcs/twotime/config/stride_frame",
            "/xpcs/twotime/config/avg_frame",
        ):
            del f[key]
            f[key] = [2]
        del f["/entry/instrument/detector_1/frame_time"]
        f["/entry/instrument/detector_1/frame_time"] = [0.5]
    xf = XpcsFile(str(path))
    assert np.ndim(xf.c2_t0) == 0
    assert float(xf.c2_t0) == pytest.approx(0.5 * 2 * 2)


def test_r6_09_get_pg_tree_opens_with_one_element_arrays(tmp_path, qapp):
    path = tmp_path / "m.h5"
    _write_nexus(path)
    _set_scalar(path, "t0", [0.1])
    xf = XpcsFile(str(path))
    tree = xf.get_pg_tree()  # raised TypeError "only 0-dimensional arrays ..."
    assert tree.windowTitle() == str(path)
    tree.close()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (np.array([0.1]), 0.1),
        (np.array([[7]]), 7),
        (np.array(3.5), 3.5),
        (np.float64(2.0), 2.0),
        (b"abc", "abc"),
        (np.bytes_(b"abc"), "abc"),
        (np.array([b"abc"]), "abc"),
        ("text", "text"),
        (4, 4),
        (None, None),
    ],
)
def test_r6_09_metadata_value_formatter(value, expected):
    _format_metadata_value = _module()._format_metadata_value

    out = _format_metadata_value(value)
    assert out == expected
    assert not isinstance(out, np.ndarray)


def test_r6_09_metadata_value_formatter_summarises_large_arrays():
    _format_metadata_value = _module()._format_metadata_value

    big = np.zeros((100, 100), dtype=np.float32)
    summary = _format_metadata_value(big)
    assert isinstance(summary, str)
    assert "100, 100" in summary
    assert "float32" in summary
    small = np.arange(6.0).reshape(2, 3)
    assert _format_metadata_value(small) is small


# ------------------------------------------------------------------ R6-20
# Opening a file never reached the enhanced call (the reader is attached after
# load_data runs); the failing re-read is the one Dataset Details makes through
# get_pg_tree -> load_data on the open file.
def test_r6_20_reloading_metadata_logs_no_error_and_reads_once(
    tmp_path, caplog, monkeypatch
):
    from xpcsviewer.fileIO.hdf_reader_enhanced import EnhancedHDF5Reader

    path = tmp_path / "c.h5"
    _write_nexus(path)
    xf = XpcsFile(str(path))
    calls = []
    real = EnhancedHDF5Reader.read_multiple_datasets

    def spy(self, *a, **kw):
        calls.append((a, kw))
        return real(self, *a, **kw)

    monkeypatch.setattr(EnhancedHDF5Reader, "read_multiple_datasets", spy)
    reads = []
    real_batch = hdf_reader._connection_pool.batch_read_datasets

    def count(*a, **kw):
        reads.append(a)
        return real_batch(*a, **kw)

    monkeypatch.setattr(hdf_reader._connection_pool, "batch_read_datasets", count)
    with caplog.at_level(logging.DEBUG):
        payload = xf.load_data()
    bad = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "nhanced" in r.getMessage()
    ]
    assert not bad, [r.getMessage() for r in bad]
    assert not calls
    assert len(reads) == 1  # the metadata is read once
    # same data as the load that built the instance
    assert payload["t0"] == xf.t0
    np.testing.assert_array_equal(payload["tau"], xf.tau)
    np.testing.assert_array_equal(payload["g2"], xf.g2)


def test_r6_20_dataset_details_logs_no_error(tmp_path, caplog, qapp):
    path = tmp_path / "d.h5"
    _write_nexus(path)
    xf = XpcsFile(str(path))
    with caplog.at_level(logging.DEBUG):
        tree = xf.get_pg_tree()
    tree.close()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_r6_20_missing_metadata_field_still_raises_on_open(tmp_path):
    path = tmp_path / "x.h5"
    _write_nexus(path)
    with h5py.File(path, "r+") as f:
        del f[K["t1"]]
    with pytest.raises(ValueError, match="t1"):  # unchanged: batch_read_fields' error
        XpcsFile(str(path))
