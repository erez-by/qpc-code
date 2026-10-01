import numpy as np
import pytest

from qpc.grid import grid_from_cutoff
from qpc.hamiltonian import Hamiltonian
from qpc.occupation import fermi, find_mu, total_number, density, check_band_margin
from qpc.solver import lowest_states


# ---------------------------------------------------------------- fermi
def test_fermi_half_and_symmetry():
    mu, kT = 0.3, 0.01
    assert fermi(mu, mu, kT) == 0.5
    x = np.linspace(-1, 1, 101)
    np.testing.assert_allclose(fermi(mu + x, mu, kT) + fermi(mu - x, mu, kT), 1.0, atol=1e-15)


def test_fermi_no_overflow():
    kT = 1e-3
    with np.errstate(all="raise"):
        hi = fermi(1e4 * kT, 0.0, kT)
        lo = fermi(-1e4 * kT, 0.0, kT)
    assert hi == 0.0 and lo == 1.0


# ---------------------------------------------------------------- find_mu
@pytest.mark.parametrize("seed", range(5))
def test_find_mu_random_spectra(seed):
    rng = np.random.default_rng(seed)
    eigs = [np.sort(rng.uniform(0, 1, 200)), np.sort(rng.uniform(0.02, 1, 200))]
    kT = 0.005
    for N in (1.0, 37.0, 140.0, 255.5):
        mu = find_mu(eigs, N, kT)
        assert abs(total_number(eigs, mu, kT) - N) < 1e-10


def test_find_mu_degenerate_level_at_fermi_energy():
    # 2 spins x levels 0, 0.1, 0.2(x4 degenerate), 0.3, ... ; N chosen so mu sits on the degenerate level
    e = np.array([0.0, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5])
    kT = 1e-3
    N = 2 * (2 + 2)            # half-filled degenerate level in each spin
    mu = find_mu([e, e], N, kT)
    assert abs(total_number([e, e], mu, kT) - N) < 1e-10
    assert abs(mu - 0.2) < 1e-6
    # single channel input
    mu1 = find_mu([e], 4.0, kT)
    assert abs(total_number([e], mu1, kT) - 4.0) < 1e-10


# ---------------------------------------------------------------- density
def test_density_parseval():
    grid = grid_from_cutoff(60.0, 24.0, 0.8)
    H = Hamiltonian(grid, 0.8)
    x, y = grid.real_axes()
    X_, Y_ = np.meshgrid(x, y, indexing="ij")
    H.set_potential(0.5 * 0.05 * (Y_ - grid.Ly / 2) ** 2
                    + 0.1 * np.sin(2 * np.pi * X_ / grid.Lx))
    res = lowest_states(H, 40, method="dense")
    f = fermi(res.eigvals, res.eigvals[20], 0.01)
    n = density(H, res.X, f, chunk=7)              # several chunks
    assert n.shape == (grid.Nx, grid.Ny)
    assert n.min() >= 0
    assert abs(n.sum() * grid.dx * grid.dy - f.sum()) < 1e-10
    # chunk size does not matter
    np.testing.assert_allclose(density(H, res.X, f, chunk=64), n, rtol=0, atol=1e-14)


def test_check_band_margin():
    kT = 0.01
    assert check_band_margin(np.array([0.0, 0.5]), 0.3, kT)          # 0.2 > 0.15
    assert not check_band_margin(np.array([0.0, 0.4]), 0.3, kT)      # 0.1 < 0.15
