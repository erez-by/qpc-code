"""Delta (reference-subtracted) formulation: tests (a)-(c) of the M5.1 review."""
import numpy as np
import pytest

from qpc.analysis import barrier_profile, n1d, transverse_levels
from qpc.fourier import min_image
from qpc.grid import grid_from_cutoff
from qpc.hamiltonian import Hamiltonian
from qpc.hartree import Hartree
from qpc.potential import QPCParams
from qpc.scf import (SCFParams, build_external, clean_wire_reference, external_potential,
                     ks_potentials, physical_xy, run_scf)
from qpc.units import Units

U = Units()
N1D_NM = 2.8e-2


@pytest.fixture(scope="module")
def prod_wire():
    grid = grid_from_cutoff(U.nm_to_au(5000.0), U.nm_to_au(320.0), U.meV_to_au(15.0))
    ham = Hamiltonian(grid, U.meV_to_au(15.0))
    p = SCFParams(spin_polarized=False, method="wire", tol=1e-9)
    hart = Hartree(grid, p.a_metal_au(U), verbose=False)
    ref = clean_wire_reference(ham, QPCParams(), N1D_NM * 5000.0, p.kT_au(U))
    return grid, ham, hart, p, ref


def test_reference_wire_values(prod_wire):
    """Non-interacting harmonic wire: mu = 2.1019 meV, single subband, mu - e_0 ~ k_F^2/2."""
    grid, ham, hart, p, ref = prod_wire
    assert abs(U.au_to_meV(ref.mu) - 2.1019) < 1e-3
    assert ref.N_sub[1:].sum() < 1e-5
    kF = np.pi * N1D_NM * U.length_nm / 2
    assert abs(ref.mu - ref.subbands[0] - kF ** 2 / 2) < 2 * p.kT_au(U)
    assert abs(ref.n_ref.sum() * grid.dx * grid.dy - 140.0) < 1e-8


def test_V_ext_eff_is_parabola_at_reference(prod_wire):
    """(b) V_ext_eff + V_H[n_ref] + v_xc[n_ref/2, n_ref/2] == (1/2) w_y^2 y^2, both spins."""
    grid, ham, hart, p, ref = prod_wire
    q = QPCParams()
    V_eff = build_external(grid, hart, q, p, ref, include_qpc=False)
    Vs = ks_potentials(V_eff, hart.potential(ref.n_ref), ref.n_ref / 2, ref.n_ref / 2, 0.0, p.interp)
    bare = external_potential(grid, q, include_qpc=False)
    for V in Vs:
        np.testing.assert_allclose(V, bare, rtol=0, atol=1e-12)


def test_reference_is_fixed_point(prod_wire):
    """(a) V_QPC = 0, fixed mu_wire, perturbed start -> SCF returns n_ref (|dn| < 1e-6 electrons)."""
    grid, ham, hart, p, ref = prod_wire
    V_eff = build_external(grid, hart, QPCParams(), p, ref, include_qpc=False)
    _, Y = physical_xy(grid)
    n0 = ref.n_ref * (1 + 0.2 * np.tanh(Y / U.nm_to_au(20.0))) * 1.03
    r = run_scf(ham, hart, V_eff, p, mu=ref.mu, n_init=(n0 / 2, n0 / 2), verbose=False)
    assert r.converged
    dA = grid.dx * grid.dy
    assert np.abs(r.n_up + r.n_dn - ref.n_ref).sum() * dA < 1e-6
    assert abs(r.N - 140.0) < 1e-6


def test_qpc_far_field_density():
    """(c) QPC (hbar w_x = 1.5 meV, unpolarised, fixed mu_wire): far-field n_1D within 0.5 % of
    2.8e-2 nm^-1. Reduced cell (Lx = 2000 nm, E_cut = 10 meV) to keep the test fast; the
    production value is checked by scripts/run_qpc.py."""
    Lx_nm = 2000.0
    grid = grid_from_cutoff(U.nm_to_au(Lx_nm), U.nm_to_au(320.0), U.meV_to_au(10.0))
    ham = Hamiltonian(grid, U.meV_to_au(10.0))
    p = SCFParams(spin_polarized=False, method="dense")
    hart = Hartree(grid, p.a_metal_au(U), verbose=False)
    q = QPCParams(hbar_wx_meV=1.5)
    ref = clean_wire_reference(ham, q, N1D_NM * Lx_nm, p.kT_au(U))
    V_eff = build_external(grid, hart, q, p, ref)
    r = run_scf(ham, hart, V_eff, p, mu=ref.mu, n_init=(ref.n_ref / 2, ref.n_ref / 2),
                verbose=False)
    assert r.converged
    x_nm = U.au_to_nm(min_image(grid.real_axes()[0], grid.Lx))
    line = n1d(r.n_up + r.n_dn, grid) / U.length_nm                 # nm^-1
    far = np.abs(x_nm) > Lx_nm / 2 - 300.0
    assert np.abs(line[far] / N1D_NM - 1).max() < 5e-3
    assert line[0] < 0.8 * N1D_NM                                    # depleted at the QPC
    # barrier profile: far field is the bare subband bottom (1 meV), centre raised
    e0 = U.au_to_meV(barrier_profile(r, ham))
    assert abs(e0[grid.Nx // 2, 0] - 1.0) < 0.02
    assert e0[0, 0] > e0[grid.Nx // 2, 0] + 0.2
    np.testing.assert_allclose(e0[:, 0], e0[:, 1])                   # unpolarised


def test_transverse_levels_harmonic():
    """transverse_levels on the bare parabola: (n + 1/2) hbar w_y at every x (independent check)."""
    grid = grid_from_cutoff(U.nm_to_au(1000.0), U.nm_to_au(320.0), U.meV_to_au(15.0))
    ham = Hamiltonian(grid, U.meV_to_au(15.0))
    q = QPCParams()
    lev = U.au_to_meV(transverse_levels(external_potential(grid, q, include_qpc=False), ham, 3))
    np.testing.assert_allclose(lev, np.array([1.0, 3.0, 5.0])[None, :] * np.ones((grid.Nx, 1)),
                               atol=1e-3)
