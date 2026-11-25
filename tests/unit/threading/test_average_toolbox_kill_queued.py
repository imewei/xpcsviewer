"""A queued averaging job can be killed before it ever opens a file."""

from unittest.mock import MagicMock

import pytest

from xpcsviewer.module.average_toolbox import AverageToolbox
from xpcsviewer.xpcs_viewer import XpcsViewer


def _viewer(worker):
    viewer = MagicMock()
    viewer.vk.avg_worker = worker
    return viewer


def test_kill_queued_job_then_run_reads_nothing(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(
        "xpcsviewer.module.average_toolbox.XF", lambda *a, **k: opened.append(a)
    )
    worker = AverageToolbox(str(tmp_path), ["f0_0001.h5", "f1_0001.h5"], jid=3)
    worker.status = "queued"
    viewer = _viewer(worker)

    XpcsViewer.avg_kill_job(viewer)
    assert worker.is_killed.is_set()
    viewer.timer.stop.assert_called_once()

    worker.do_average(save_path=str(tmp_path / "out.h5"))
    assert worker.status == "killed"
    assert opened == []


@pytest.mark.parametrize("status", ["running", "queued"])
def test_kill_accepts_running_and_queued(tmp_path, status):
    worker = AverageToolbox(str(tmp_path), ["f0_0001.h5"], jid=4)
    worker.status = status
    XpcsViewer.avg_kill_job(_viewer(worker))
    assert worker.is_killed.is_set()


@pytest.mark.parametrize("status", ["finished", "killed", "wait"])
def test_kill_ignores_other_states(tmp_path, status):
    worker = AverageToolbox(str(tmp_path), ["f0_0001.h5"], jid=5)
    worker.status = status
    viewer = _viewer(worker)
    XpcsViewer.avg_kill_job(viewer)
    assert not worker.is_killed.is_set()
    viewer.timer.stop.assert_not_called()
