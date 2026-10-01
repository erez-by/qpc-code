import numpy as np
from qpc.units import Units
from qpc.grid import grid_from_cutoff


def test_nyquist_resolves_2Gcut():
    u = Units()
    e = u.meV_to_au(15.0)
    g = grid_from_cutoff(u.nm_to_au(5000.0), u.nm_to_au(320.0), e)
    G_cut = np.sqrt(2 * e)
    assert np.pi / g.dx >= 2 * G_cut
    assert np.pi / g.dy >= 2 * G_cut


def test_plane_wave_count_matches_area_estimate():
    u = Units()
    e = u.meV_to_au(15.0)
    Lx, Ly = u.nm_to_au(5000.0), u.nm_to_au(320.0)
    g = grid_from_cutoff(Lx, Ly, e)
    n_pw = int(g.cutoff_mask(e).sum())
    estimate = 2 * e * Lx * Ly / (4 * np.pi)      # pi G_cut^2 / ((2 pi)^2 / (Lx Ly))
    assert abs(n_pw - estimate) / estimate < 0.03
