"""scripts/qpc_driver.level_crossing_stats against a hand-built iteration table."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from qpc_driver import level_crossing_stats  # noqa: E402


def test_swings_coincide_with_count_changes():
    # M_loc alternates 0.6 / 1.0 every 10 iterations; the up-spin level count near mu changes exactly then
    it = np.arange(1, 121)
    phase = (it // 10) % 2
    m = 0.6 + 0.4 * phase + 1e-3 * np.sin(it)
    rows = np.column_stack([it, m, 2 + phase, np.full(it.size, 3), 0.01 + 0.02 * phase, np.full(it.size, -0.05)])
    s = level_crossing_stats(rows)
    last = m[-100:]
    assert s["n"] == 100
    assert np.isclose(s["mean"], last.mean()) and np.isclose(s["std"], last.std())
    assert s["swings"] == 10 and s["swings_with_count_change"] == 10 and s["count_changes"] == 10
    assert s["corr_dM_dcount"] > 0.99
    assert np.isclose(s["min_abs_dmin_meV"], 0.01)


def test_no_crossings():
    it = np.arange(1, 51)
    rows = np.column_stack([it, 0.5 + 0.1 * (it % 2), np.full(50, 2), np.full(50, 2), np.full(50, 0.03),
                            np.full(50, 0.04)])
    s = level_crossing_stats(rows)
    assert s["count_changes"] == 0 and s["swings_with_count_change"] == 0 and np.isnan(s["corr_dM_dcount"])
