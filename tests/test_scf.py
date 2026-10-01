import numpy as np
import pytest

from qpc.grid import grid_from_cutoff
from qpc.hamiltonian import Hamiltonian
from qpc.hartree import Hartree
from qpc.potential import QPCParams
from qpc.scf import (SCFParams, SCFResult, external_potential, ks_potentials, run_scf,
                     wire_states)
from qpc.occupation import density, fermi
from qpc.units import Units

U = Units()
E_CUT = 0.8
A_NM = 50.0


@pytest.fixture(scope="module")
def small():
    grid = grid_from_cutoff(60.0, 24.0, E_CUT)
    ham = Hamiltonian(grid, E_CUT)
    hart = Hartree(grid, U.nm_to_au(A_NM), verbose=False)
    return grid, ham, hart


def params(**kw):
    base = dict(kT=0.5, a_nm=A_NM, nb_init=30, tol=1e-7)
    base.update(kw)
    return SCFParams(**base)


def test_wire_states_equal_dense(small):
    """x-independent potential: block eigenpairs == dense eigenpairs (independent method)."""
    grid, ham, _ = small
    V = external_potential(grid, QPCParams(), include_qpc=False)
    ham.set_potential(V)
    e_w, X_w = wire_states(ham, 25)
    e_d, X_d = ham.eigh_dense(25)
    np.testing.assert_allclose(e_w, e_d, atol=1e-11)
    np.testing.assert_allclose(np.linalg.norm(X_w, axis=0), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(ham.apply(X_w) - X_w * e_w).max(), 0, atol=1e-10)
    # same density from both (degenerate subspaces may rotate, the density does not)
    f = fermi(e_w, 0.5 * (e_w[9] + e_w[10]), 0.01)
    np.testing.assert_allclose(density(ham, X_w, f), density(ham, X_d, f), atol=1e-10)


def test_clean_wire_unpolarised(small):
    """Wire SCF: converges, conserves N, x-independent, wire == dense route, self-consistent."""
    grid, ham, hart = small
    V = external_potential(grid, QPCParams(), include_qpc=False)
    N = 4.0
    r_w = run_scf(ham, hart, V, params(spin_polarized=False, method="wire"), N=N, verbose=False)
    r_d = run_scf(ham, hart, V, params(spin_polarized=False, method="dense"), N=N, verbose=False)
    assert r_w.converged and r_d.converged
    dA = grid.dx * grid.dy
    assert abs((r_w.n_up + r_w.n_dn).sum() * dA - N) < 1e-6
    np.testing.assert_allclose(r_w.n_up, r_w.n_dn)
    assert np.ptp(r_w.n_up, axis=0).max() < 1e-8 * r_w.n_up.max()       # uniform in x
    np.testing.assert_allclose(r_w.mu, r_d.mu, atol=1e-6)
    np.testing.assert_allclose(r_w.n_up, r_d.n_up, atol=1e-6 * r_w.n_up.max())
    # one more KS step from the converged density changes it by < tol
    Vs = ks_potentials(V, hart.potential(r_w.n_up + r_w.n_dn), r_w.n_up, r_w.n_dn, 0.0, "exchange")
    ham.set_potential(Vs[0])
    e, X = wire_states(ham, len(r_w.eigs[0]))
    n_new = density(ham, X, fermi(e, r_w.mu, r_w.params.kT_au()))
    assert np.abs(n_new - r_w.n_up).sum() * dA < 1e-6


def test_fixed_mu_reproduces_N(small):
    grid, ham, hart = small
    V = external_potential(grid, QPCParams(), include_qpc=False)
    r_N = run_scf(ham, hart, V, params(spin_polarized=False, method="wire"), N=4.0, verbose=False)
    r_mu = run_scf(ham, hart, V, params(spin_polarized=False, method="wire"), mu=r_N.mu,
                   n_init=(r_N.n_up * 0.9, r_N.n_dn * 0.9), verbose=False)
    assert r_mu.converged
    assert abs(r_mu.N - 4.0) < 1e-5


def test_zeeman_polarises_and_B0_stays_unpolarised(small):
    grid, ham, hart = small
    V = external_potential(grid, QPCParams(), include_qpc=False)
    r0 = run_scf(ham, hart, V, params(method="wire"), N=4.0, verbose=False)
    assert abs(r0.M) < 1e-6
    rB = run_scf(ham, hart, V, params(method="wire", B_T=6.0), N=4.0, verbose=False)
    assert rB.converged and rB.M > 1e-3
    # Zeeman shift of the eigenvalues: up lowered relative to down
    assert rB.eigs[0][0] < rB.eigs[1][0]


def test_save_load_roundtrip(small, tmp_path):
    grid, ham, hart = small
    V = external_potential(grid, QPCParams(), include_qpc=False)
    r = run_scf(ham, hart, V, params(spin_polarized=False, method="wire"), N=4.0, verbose=False)
    path = tmp_path / "r.npz"
    r.save_npz(path)
    q = SCFResult.load_npz(path)
    np.testing.assert_array_equal(q.n_up, r.n_up)
    np.testing.assert_array_equal(q.X[1], r.X[1])
    assert q.mu == r.mu and q.iterations == r.iterations and q.converged == r.converged
    assert q.params == r.params
    assert len(q.history) == len(r.history)
