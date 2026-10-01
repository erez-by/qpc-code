"""LOBPCG against the dense reference (small cells, so the whole file runs in seconds)."""
import numpy as np
import pytest

from qpc.grid import grid_from_cutoff
from qpc.hamiltonian import Hamiltonian
from qpc.solver import (lobpcg, lowest_states, svqb, tpa_preconditioner, random_start,
                        XAveragedPreconditioner, dense_lowest)

E_CUT = 0.8      # Ha*


def make(Lx=60.0, Ly=24.0, e_cut=E_CUT):
    grid = grid_from_cutoff(Lx, Ly, e_cut)
    return grid, Hamiltonian(grid, e_cut)


def xy(grid):
    x, y = grid.real_axes()
    return np.meshgrid(x, y, indexing="ij")


def asym_potential(grid, shift=0.0):
    """Smooth, NOT even -> complex Hermitian H; plus a y-confinement like the wire."""
    X, Y = xy(grid)
    kx, ky = 2 * np.pi / grid.Lx, 2 * np.pi / grid.Ly
    return (0.3 * np.cos(kx * X + 0.4) + 0.2 * np.sin(2 * kx * X) * np.cos(ky * Y)
            + 0.5 * (1 - np.cos(ky * Y)) + shift * np.cos(3 * kx * X))


def subspace_gap(A, B):
    """|| A A^H - B B^H ||_2 for orthonormal blocks: 0 iff they span the same space."""
    return np.linalg.norm(A @ A.conj().T - B @ B.conj().T, 2)


@pytest.fixture
def ham():
    grid, H = make()
    H.set_potential(asym_potential(grid))
    return H


def test_svqb_orthonormal_and_drops_dependent_columns():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((200, 6)) + 1j * rng.standard_normal((200, 6))
    A = np.hstack([A, A[:, :2] + 1e-14 * rng.standard_normal((200, 2))])   # 2 dependent cols
    Q = svqb(A)
    assert Q.shape[1] == 6
    assert np.allclose(Q.conj().T @ Q, np.eye(6), atol=1e-12)


def test_tpa_limits():
    kin = np.array([0.0, 1.0, 1e4])
    X = np.array([[0.0], [1.0], [0.0]]) + 0j            # E_kin = 1 -> s = T_G
    K = tpa_preconditioner(kin, X)[:, 0]
    assert np.all((K > 0) & (K <= 1))
    assert np.isclose(K[0], 1.0)                         # s = 0: low G untouched
    assert np.isclose(K[2], 1 / (2 * 1e4), rtol=1e-3)    # s >> 1: K ~ 1/(2 s)


@pytest.mark.parametrize("precond", ["block", "tpa", None])
def test_lobpcg_matches_dense(ham, precond):
    nb = 12
    w_ref, v_ref = ham.eigh_dense(nb)
    res = lowest_states(ham, nb, tol=1e-8, precond=precond)
    assert res.converged
    assert np.allclose(res.eigvals, w_ref, atol=1e-10, rtol=0)        # error ~ tol^2/gap
    assert np.all(res.resnorms < 1e-8)
    assert np.allclose(res.X.conj().T @ res.X, np.eye(nb), atol=1e-10)
    # eigenvectors: same subspace (phases / rotations inside degenerate pairs are free)
    assert subspace_gap(res.X, v_ref.astype(complex)) < 1e-6


def test_residual_is_true_residual(ham):
    """The reported residual must be ||H x - theta x|| with the REAL operator."""
    res = lowest_states(ham, 8, tol=1e-7)
    R = ham.apply(res.X) - res.X * res.eigvals
    assert np.allclose(np.linalg.norm(R, axis=0), res.resnorms, atol=1e-12)


def test_degenerate_free_particle():
    """Constant potential: eigenvalues are T_G + V0, with many exact degeneracies (+-G)."""
    grid, H = make()
    V0 = 0.37
    H.set_potential(np.full((grid.Nx, grid.Ny), V0))
    nb = 15
    exact = np.sort(H.kinetic)[:nb] + V0
    res = lowest_states(H, nb, tol=1e-8)
    assert res.converged
    assert np.allclose(res.eigvals, exact, atol=1e-10)


def test_warm_start_needs_fewer_iterations():
    """SCF situation: potential changes a little; restarting from the old vectors is cheaper."""
    grid, H = make()
    nb = 12
    H.set_potential(asym_potential(grid))
    first = lowest_states(H, nb, tol=1e-7)
    H.set_potential(asym_potential(grid, shift=0.01))
    cold = lowest_states(H, nb, tol=1e-7)
    warm = lowest_states(H, nb, tol=1e-7, X0=first.X_full)
    w_ref, _ = H.eigh_dense(nb)
    assert warm.converged and np.allclose(warm.eigvals, w_ref, atol=1e-10)
    assert warm.n_iter < cold.n_iter


def test_buffer_bands_need_not_converge(ham):
    X0 = random_start(ham.n_pw, 15)
    res = lobpcg(ham.apply, X0, n_wanted=10, kinetic=ham.kinetic, tol=1e-8)
    assert res.converged and np.all(res.resnorms[:10] < 1e-8)


def test_dense_front_end_same_interface(ham):
    a = lowest_states(ham, 6, method="dense")
    b = lowest_states(ham, 6, tol=1e-8)
    assert np.allclose(a.eigvals, b.eigvals, atol=1e-10)


def wire_potential(grid, wy=0.18):
    """x-independent harmonic confinement, folded to the cell (as in the production runs)."""
    _, Y = xy(grid)
    Yf = Y - grid.Ly * np.round(Y / grid.Ly)
    return 0.5 * wy ** 2 * Yf ** 2


def test_block_preconditioner_is_exact_for_x_independent_V():
    """For V = V(y): H_0 = H, so the block eigenvalues ARE the spectrum of H."""
    grid, H = make()
    H.set_potential(wire_potential(grid))
    K = XAveragedPreconditioner(H)
    w_blocks = np.sort(K.d.ravel())[:20]
    w_ref, _ = H.eigh_dense(20)
    assert np.allclose(w_blocks, w_ref, atol=1e-11)


def test_block_preconditioner_hermitian_positive():
    grid, H = make()
    H.set_potential(asym_potential(grid))
    K = XAveragedPreconditioner(H)
    n = H.n_pw
    E = np.eye(n, dtype=complex)[:, :40]
    M = E.conj().T @ K(E, np.array([0.0]))                       # 40 x 40 corner of K
    assert np.allclose(M, M.conj().T, atol=1e-12)
    assert np.linalg.eigvalsh(M).min() > 0


def test_block_preconditioner_beats_tpa_on_the_wire():
    """Confinement-dominated H: the stiff part is V(y), not T_G, so TPA is the wrong model.
    (Even with H_0 = H the count is not 1: K = (H - sigma)^-1 with fixed sigma is
    shift-and-invert, converging like (theta_i - sigma)/(theta_k+1 - sigma) per step.)"""
    grid, H = make()
    H.set_potential(wire_potential(grid))
    blk = lowest_states(H, 12, tol=1e-8, precond="block")
    tpa = lowest_states(H, 12, tol=1e-8, precond="tpa")
    assert blk.converged and tpa.converged
    assert blk.n_iter < 0.6 * tpa.n_iter


def test_dense_lowest_matches_full_eigh(ham):
    w, v = dense_lowest(ham, 7)
    w_all, _ = ham.eigh_dense()
    assert np.allclose(w, w_all[:7], atol=1e-12)
