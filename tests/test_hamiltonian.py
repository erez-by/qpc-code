import numpy as np
import pytest
from qpc.grid import grid_from_cutoff, Grid
from qpc.hamiltonian import Hamiltonian

E_CUT = 0.8                       # Ha*  (~8.8 meV in GaAs units)


def make(Lx=60.0, Ly=24.0, e_cut=E_CUT):
    grid = grid_from_cutoff(Lx, Ly, e_cut)
    return grid, Hamiltonian(grid, e_cut)


def xy(grid):
    x, y = grid.real_axes()
    return np.meshgrid(x, y, indexing="ij")


def rough_potential(grid):
    """Smooth periodic potential, deliberately NOT even, so H is complex Hermitian."""
    X, Y = xy(grid)
    return (0.20 * np.cos(2 * np.pi * X / grid.Lx)
            + 0.10 * np.sin(2 * np.pi * (X / grid.Lx + 2 * Y / grid.Ly))
            + 0.05 * np.cos(2 * np.pi * 3 * Y / grid.Ly))


def test_basis_list_matches_cutoff():
    grid, H = make()
    assert H.n_pw == int(grid.cutoff_mask(E_CUT).sum())
    assert H.ix.shape == H.iy.shape == (H.n_pw,)
    assert np.all(H.kinetic <= E_CUT + 1e-12)


def test_constant_potential():
    grid, H = make()
    H.set_potential(np.full((grid.Nx, grid.Ny), 0.3))
    assert abs(H.V_hat[0, 0] - 0.3) < 1e-14                   # normalisation of V_hat
    mask = np.ones_like(H.V_hat, bool); mask[0, 0] = False
    assert np.abs(H.V_hat[mask]).max() < 1e-14
    w, _ = H.eigh_dense()
    assert np.allclose(w, np.sort(H.kinetic) + 0.3, atol=1e-12)


def test_zero_potential_gives_kinetic_energies():
    grid, H = make()
    H.set_potential(np.zeros((grid.Nx, grid.Ny)))
    w, _ = H.eigh_dense()
    assert np.allclose(w, np.sort(H.kinetic), atol=1e-12)


def test_dense_is_hermitian():
    grid, H = make()
    H.set_potential(rough_potential(grid))
    Hd = H.dense()
    assert np.allclose(Hd, Hd.conj().T, atol=1e-12)
    assert np.abs(Hd.imag).max() > 1e-3                        # the test potential really is complex


def test_dense_matches_matrix_free():
    grid, H = make()
    H.set_potential(rough_potential(grid))
    rng = np.random.default_rng(0)
    c = rng.normal(size=H.n_pw) + 1j * rng.normal(size=H.n_pw)
    assert np.allclose(H.dense() @ c, H.apply(c), atol=1e-11)
    block = rng.normal(size=(H.n_pw, 3)) + 1j * rng.normal(size=(H.n_pw, 3))   # several vectors at once
    assert np.allclose(H.dense() @ block, H.apply(block), atol=1e-11)


def test_set_potential_rejects_wrong_shape():
    grid, H = make()
    with pytest.raises(ValueError):
        H.set_potential(np.zeros((grid.Nx + 1, grid.Ny)))


def test_clean_wire_levels():
    """V = (1/2) w^2 y^2 (folded), no x dependence: E = w(n+1/2) + G_x^2/2."""
    w = 0.1
    Lx, Ly = 16.0, 40.0
    grid, H = make(Lx, Ly, e_cut=2.0)             # larger cutoff: the n >= 2 states need more G_y
    X, Y = xy(grid)
    Yf = Y - Ly * np.round(Y / Ly)                             # minimum-image fold
    H.set_potential(0.5 * w ** 2 * Yf ** 2)
    ev, _ = H.eigh_dense(n_bands=12)
    expected = sorted(w * (n + 0.5) + 0.5 * (2 * np.pi * m / Lx) ** 2
                      for n in range(8) for m in range(-8, 9))[:12]
    assert np.allclose(ev, expected, atol=1e-9)


def test_to_real_space_parseval():
    """For sum|c|^2 = 1 the cell average of |psi|^2 is exactly 1 (Parseval)."""
    grid, H = make()
    rng = np.random.default_rng(3)
    c = rng.normal(size=H.n_pw) + 1j * rng.normal(size=H.n_pw)
    c /= np.linalg.norm(c)
    psi = H.to_real_space(c)
    assert psi.shape == (grid.Nx, grid.Ny)
    assert np.isclose(np.mean(np.abs(psi) ** 2), 1.0, atol=1e-12)
