"""Averaging accumulates in float64 whatever dtype the first file stores."""

import multiprocessing
import os
from types import SimpleNamespace

import numpy as np
import pytest

from xpcsviewer.module.average_toolbox import AverageToolbox

N_FILES = 8


def _fake_xf(path, fields=None):
    index = int(os.path.basename(path).split("_")[0][1:])
    row = np.array([10], dtype=np.uint16) if index == 0 else np.array([20.5])
    return SimpleNamespace(
        saxs_1d={"data_raw": row}, g2=np.ones((3, 1)), clear_cache=lambda: None
    )


@pytest.mark.parametrize("parallel", [True, False])
def test_integer_first_file_does_not_truncate_or_inflate_count(
    tmp_path, monkeypatch, parallel
):
    target = "xpcsviewer.module.average_toolbox"
    monkeypatch.setattr(f"{target}.XF", _fake_xf)
    monkeypatch.setattr(f"{target}.copyfile", lambda *a: None)
    written = {}
    monkeypatch.setattr(
        f"{target}.put", lambda path, result, **k: written.update(result)
    )
    monkeypatch.setattr(multiprocessing, "cpu_count", lambda: 4 if parallel else 1)
    monkeypatch.setattr(
        f"{target}.MemoryMonitor.is_memory_pressure_high",
        staticmethod(lambda *a, **k: False),
    )
    toolbox = AverageToolbox(
        work_dir=str(tmp_path),
        flist=[f"f{i}_0001.h5" for i in range(N_FILES)],
        jid=1,
    )
    toolbox.do_average(save_path=str(tmp_path / "out.h5"), fields=["saxs_1d"])

    assert toolbox.status == "finished"
    assert toolbox.n_accepted == N_FILES
    np.testing.assert_allclose(written["saxs_1d"], [(10 + 7 * 20.5) / N_FILES])
