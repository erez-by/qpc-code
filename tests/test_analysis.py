import numpy as np

from qpc.analysis import transverse_levels, wire_subbands
from qpc.grid import grid_from_cutoff
from qpc.hamiltonian import Hamiltonian
from qpc.occupation import find_mu
from qpc.potential import QPCParams
from qpc.scf import external_potential, wire_states
from qpc.units import Units

U = Units()


def test_transverse_levels_harmonic():
    """transverse_levels on the bare parabola: (n + 1/2) hbar w_y at every x (analytic)."""
    grid = grid_from_cutoff(U.nm_to_au(1000.0), U.nm_to_au(320.0), U.meV_to_au(15.0))
    ham = Hamiltonian(grid, U.meV_to_au(15.0))
    lev = U.au_to_meV(transverse_levels(external_potential(grid, QPCParams(), include_qpc=False), ham, 3))
    np.testing.assert_allclose(lev, np.array([1.0, 3.0, 5.0])[None, :] * np.ones((grid.Nx, 1)), atol=1e-3)


def test_wire_subbands_harmonic():
    """Non-interacting harmonic wire, N = 140: bottoms (n + 1/2) hbar w_y, all electrons in n = 0,
    mu - e_0 = k_F^2/2 (k_F = pi n_1D/2) within 2 kT; occupation sum equals N."""
    grid = grid_from_cutoff(U.nm_to_au(5000.0), U.nm_to_au(320.0), U.meV_to_au(15.0))
    ham = Hamiltonian(grid, U.meV_to_au(15.0))
    V = external_potential(grid, QPCParams(), include_qpc=False)
    ham.set_potential(V)
    kT = U.meV_to_au(0.05)
    e, _ = wire_states(ham, 150)
    mu = find_mu([e, e], 140.0, kT)
    bottoms, N_sub = wire_subbands(ham, V, mu, kT)
    np.testing.assert_allclose(U.au_to_meV(bottoms[:3]), [1.0, 3.0, 5.0], atol=1e-3)
    assert abs(2 * N_sub.sum() - 140.0) < 1e-6
    assert 2 * N_sub[1:].sum() < 1e-5
    kF = np.pi * 2.8e-2 * U.length_nm / 2
    assert abs(mu - bottoms[0] - kF ** 2 / 2) < 2 * kT


def test_transverse_levels_bare_qpc_top():
    """Bare QPC at x = 0: -(1/2)d_y^2 + V0/2 + (1/2)(w_y + V0)^2 y^2 -> V0/2 + (w_y + V0)/2 = 4 meV."""
    grid = grid_from_cutoff(U.nm_to_au(5000.0), U.nm_to_au(320.0), U.meV_to_au(15.0))
    ham = Hamiltonian(grid, U.meV_to_au(15.0))
    V = external_potential(grid, QPCParams(hbar_wx_meV=1.5))
    e0 = U.au_to_meV(transverse_levels(V, ham)[:, 0])
    assert abs(e0[0] - 4.0) < 1e-4
    assert abs(e0[grid.Nx // 2] - 1.0) < 1e-4
    # the KS cutoff basis is coarser at the stiff x = 0 point
    assert U.au_to_meV(transverse_levels(V, ham, basis="cutoff")[0, 0]) > e0[0] + 5e-3
