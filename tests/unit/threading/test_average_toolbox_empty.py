"""AverageToolbox must reach a terminal state when no input is accepted."""

import logging
import multiprocessing
import os
from unittest.mock import MagicMock

import numpy as np
import pytest

from xpcsviewer.module.average_toolbox import AverageToolbox
from xpcsviewer.xpcs_viewer import XpcsViewer

JID = 7


def _make(tmp_path, n_files=2):
    toolbox = AverageToolbox(
        work_dir=str(tmp_path),
        flist=[f"f{i}_0001.h5" for i in range(n_files)],
        jid=JID,
    )
    events = {"status": [], "progress": [], "finished": 0}
    toolbox.signals.status.connect(events["status"].append)
    toolbox.signals.progress.connect(events["progress"].append)
    toolbox.signals.finished.connect(
        lambda: events.__setitem__("finished", events["finished"] + 1)
    )
    return toolbox, events


@pytest.fixture
def all_rejected(monkeypatch):
    """Sequential helper that accepts nothing (mask stays zero)."""
    monkeypatch.setattr(
        AverageToolbox, "_process_files_sequential", lambda self, *a, **k: None
    )


def test_all_rejected_reaches_finished_state(tmp_path, all_rejected):
    toolbox, events = _make(tmp_path)
    out = tmp_path / "out.h5"
    toolbox.do_average(save_path=str(out))

    assert toolbox.status == "finished"
    assert toolbox.n_accepted == 0
    assert events["status"] == [(JID, "finished")]
    assert events["finished"] == 1
    assert (JID, 100) in events["progress"]
    assert toolbox.etime != "--:--:--"
    assert not out.exists()


def test_all_rejected_logs_user_visible_reason(tmp_path, all_rejected, caplog):
    caplog.set_level(logging.INFO, logger="xpcsviewer.module.average_toolbox")
    toolbox, _ = _make(tmp_path)
    toolbox.do_average(save_path=str(tmp_path / "out.h5"))
    assert "no dataset is valid" in caplog.text


def test_zero_input_files_does_not_divide_by_zero(tmp_path):
    toolbox, events = _make(tmp_path)
    toolbox.model.clear()
    toolbox.do_average(save_path=str(tmp_path / "out.h5"))

    assert toolbox.status == "finished"
    assert toolbox.n_accepted == 0
    assert events["finished"] == 1


def test_killed_job_stays_killed(tmp_path, monkeypatch):
    def fake_sequential(self, *args, **kwargs):
        self.status = "killed"

    monkeypatch.setattr(AverageToolbox, "_process_files_sequential", fake_sequential)
    toolbox, events = _make(tmp_path)
    toolbox.kill()
    toolbox.do_average(save_path=str(tmp_path / "out.h5"))

    assert toolbox.status == "killed"
    assert toolbox.n_accepted == 0
    assert events["finished"] == 0
    assert (JID, 100) not in events["progress"]
    assert events["status"] == []


def test_killed_parallel_job_is_marked_killed(tmp_path, monkeypatch):
    # Force the parallel branch: >= 8 files, > 1 worker, no memory pressure.
    monkeypatch.setattr(multiprocessing, "cpu_count", lambda: 4)
    monkeypatch.setattr(
        "xpcsviewer.module.average_toolbox.MemoryMonitor.is_memory_pressure_high",
        staticmethod(lambda *a, **k: False),
    )
    called = []

    def fake_parallel(self, *args, **kwargs):
        called.append(True)
        self.kill()  # returns without touching status, like the real kill path

    monkeypatch.setattr(AverageToolbox, "_process_files_parallel", fake_parallel)
    toolbox, events = _make(tmp_path, n_files=8)
    toolbox.do_average(save_path=str(tmp_path / "out.h5"))

    assert called
    assert toolbox.status == "killed"
    assert toolbox.n_accepted == 0
    assert events["finished"] == 0


def _poll_self(status, n_accepted):
    viewer = MagicMock()
    viewer.vk.avg_worker.status = status
    viewer.vk.avg_worker.n_accepted = n_accepted
    return viewer


def _empty_messages(viewer):
    return [
        c
        for c in viewer.statusbar.showMessage.call_args_list
        if "no valid dataset" in c.args[0]
    ]


def test_poll_stops_timer_when_worker_removed():
    viewer = MagicMock()
    viewer.vk.avg_worker = None  # as remove_avg_job leaves it
    XpcsViewer._poll_avg_job(viewer)
    viewer.timer.stop.assert_called_once()
    viewer = MagicMock()
    viewer.vk = None
    XpcsViewer._poll_avg_job(viewer)
    viewer.timer.stop.assert_called_once()


def test_poll_reports_empty_result_and_stops_timer():
    viewer = _poll_self("finished", 0)
    XpcsViewer._poll_avg_job(viewer)
    viewer.vk.update_avg_info.assert_called_once()
    viewer.timer.stop.assert_called_once()
    assert len(_empty_messages(viewer)) == 1
    assert viewer.statusbar.showMessage.call_count == 1


def test_poll_finished_with_accepted_inputs_is_silent():
    viewer = _poll_self("finished", 3)
    XpcsViewer._poll_avg_job(viewer)
    viewer.timer.stop.assert_called_once()
    assert not _empty_messages(viewer)


def test_poll_killed_stops_timer_without_empty_message():
    viewer = _poll_self("killed", 0)
    XpcsViewer._poll_avg_job(viewer)
    viewer.timer.stop.assert_called_once()
    assert not _empty_messages(viewer)


def test_poll_running_keeps_timer_running():
    viewer = _poll_self("running", 0)
    XpcsViewer._poll_avg_job(viewer)
    viewer.timer.stop.assert_not_called()
    assert not _empty_messages(viewer)


class _FakeXF:
    """Stand-in XpcsFile: g2 == 1, g2_err == 10; odd files get one extra lag."""

    def __init__(self, path, fields=None):
        index = int(os.path.basename(path)[1:].split("_")[0])
        n_lag = 5 + (index % 2 if _FakeXF.mixed else 0)
        self.g2 = np.ones((n_lag, 3))
        self.g2_err = np.full((n_lag, 3), 10.0)
        self.saxs_2d = np.full((4, 5), float(index))

    def clear_cache(self):
        pass


def _average(tmp_path, monkeypatch, parallel, mixed):
    monkeypatch.setattr(_FakeXF, "mixed", mixed, raising=False)
    monkeypatch.setattr("xpcsviewer.module.average_toolbox.XF", _FakeXF)
    monkeypatch.setattr("xpcsviewer.module.average_toolbox.copyfile", lambda *a: None)
    written = {}
    monkeypatch.setattr(
        "xpcsviewer.module.average_toolbox.put",
        lambda path, result, **k: written.update(result),
    )
    monkeypatch.setattr(multiprocessing, "cpu_count", lambda: 4 if parallel else 1)
    monkeypatch.setattr(
        "xpcsviewer.module.average_toolbox.MemoryMonitor.is_memory_pressure_high",
        staticmethod(lambda *a, **k: False),
    )
    calls = []
    for name in ("_process_files_parallel", "_process_files_sequential"):
        original = getattr(AverageToolbox, name)

        def spy(self, *a, _orig=original, _name=name, **k):
            calls.append(_name)
            return _orig(self, *a, **k)

        monkeypatch.setattr(AverageToolbox, name, spy)
    toolbox, _ = _make(tmp_path, n_files=8)
    toolbox.do_average(
        save_path=str(tmp_path / "out.h5"), fields=["g2", "g2_err", "saxs_2d"]
    )
    expected = "_process_files_parallel" if parallel else "_process_files_sequential"
    assert calls == [expected]
    return toolbox, written


@pytest.mark.parametrize("mixed", [False, True])
def test_parallel_merge_matches_sequential(tmp_path, monkeypatch, mixed):
    par, par_out = _average(tmp_path, monkeypatch, parallel=True, mixed=mixed)
    seq, seq_out = _average(tmp_path, monkeypatch, parallel=False, mixed=mixed)
    assert par.status == seq.status == "finished"
    assert par.n_accepted == seq.n_accepted == (4 if mixed else 8)
    assert par_out.keys() == seq_out.keys()
    for key in seq_out:
        np.testing.assert_array_equal(par_out[key], seq_out[key])
    np.testing.assert_array_equal(par_out["g2"], 1.0)
    np.testing.assert_allclose(par_out["g2_err"], 10.0 / np.sqrt(par.n_accepted))


def _start_self(status):
    viewer = MagicMock()
    viewer.vk.avg_worker.status = status
    return viewer


def test_second_start_of_queued_job_is_refused():
    viewer = _start_self("wait")
    XpcsViewer.start_avg_job(viewer)
    XpcsViewer.start_avg_job(viewer)  # still queued: a second start crashed Qt
    viewer.thread_pool.start.assert_called_once_with(viewer.vk.avg_worker)


@pytest.mark.parametrize("status", ["running", "finished", "killed"])
def test_non_waiting_job_is_not_restarted(status):
    viewer = _start_self(status)
    XpcsViewer.start_avg_job(viewer)
    viewer.thread_pool.start.assert_not_called()
    assert viewer.vk.avg_worker.status == status
    assert status in viewer.statusbar.showMessage.call_args.args[0]
