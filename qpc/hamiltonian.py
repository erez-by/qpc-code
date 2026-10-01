"""Plane-wave Hamiltonian H = T + V on the periodic Lx x Ly cell (atomic units).

Design: the SCF loop changes only the real array V_eff(x, y) on the grid. So the basis
list and the kinetic diagonal are built ONCE in __init__, and the potential is swapped
with set_potential().

Conventions (keep them identical everywhere in the code):
  * Basis: plane waves exp(i G.r) with |G|^2 / 2 <= e_cut. A coefficient vector c has
    length n_pw; element p belongs to the plane wave at FFT-array indices (ix[p], iy[p]).
    The ordering comes from np.nonzero(mask) ONCE, and is stored on the object.
  * psi(r) = sum_p c[p] exp(i G_p . r)   (no 1/sqrt(Area) here). Normalised coefficients
    sum|c|^2 = 1 mean the physical wavefunction is psi / sqrt(Lx * Ly).
  * V_hat = fft2(V_grid) / (Nx * Ny) holds the Fourier-series coefficients V_Q, so that
    V(r_j) = sum_Q V_hat[Q] exp(i Q . r_j).
"""
import numpy as np
import scipy.linalg as la

from .grid import Grid


class Hamiltonian:
    """H = T + V on the plane waves with |G|^2/2 <= e_cut.

    Attributes
    ----------
    grid     : Grid
    e_cut    : float, cutoff energy (Ha*)
    ix, iy   : int arrays (n_pw,), FFT-array indices of the basis plane waves
    n_pw     : int, number of basis functions
    kinetic  : float array (n_pw,), T = G^2 / 2 on the basis
    V_grid   : float array (Nx, Ny), the local potential (set by set_potential)
    V_hat    : complex array (Nx, Ny), its Fourier-series coefficients
    """

    def __init__(self, grid: Grid, e_cut: float):
        self.grid = grid
        self.e_cut = float(e_cut)
        self.ix = self.iy = self.kinetic = None
        self.n_pw = 0
        self.V_grid = None
        self.V_hat = None
        self._build_basis()

    # ------------------------------------------------------------------ static part
    def _build_basis(self):
        """Part A, step 3 (+ the kinetic diagonal of step 5).

        TODO:
          1. mask = self.grid.cutoff_mask(self.e_cut)
          2. self.ix, self.iy = indices of the True entries (np.nonzero). Do this ONCE:
             this ordering defines the coefficient vector everywhere.
          3. self.n_pw = number of basis functions
          4. self.kinetic = G^2/2 evaluated on the basis (pick the basis entries out of the
             full (Nx, Ny) array with the index lists)
        """

        mask = self.grid.cutoff_mask(self.e_cut)
        self.ix , self.iy = np.nonzero(mask) # returns the indexes that are inside the energyt of the mask
        self.n_pw = len(self.ix)
        self.kinetic = self.grid.kinetic()[self.ix, self.iy]

    # ------------------------------------------------------------------ the part the SCF loop changes
    def set_potential(self, V_grid):
        """Part A, steps 1-2: store the potential and its Fourier coefficients.

        The shape check is already done. TODO: compute V_hat with the normalisation
        1 / (Nx * Ny). Test: a constant V0 must give V_hat[0, 0] == V0 and zeros elsewhere.
        """
        V_grid = np.asarray(V_grid, dtype=float)
        if V_grid.shape != (self.grid.Nx, self.grid.Ny):
            raise ValueError(
                f"V_grid has shape {V_grid.shape}, expected {(self.grid.Nx, self.grid.Ny)}")
        self.V_grid = V_grid
        self.V_hat = np.fft.fft2(V_grid)/(self.grid.Nx * self.grid.Ny)
        # Test: a constant V0 must give V_hat[0, 0] == V0 and zeros elsewhere.
        # if np.allclose(self.V_hat[1:, 1:], 0):
        #     print("Test passed: constant potential gives correct Fourier coefficients.")
        # else:
        #     raise ValueError("Test failed: constant potential does not give correct Fourier coefficients.")

    # ------------------------------------------------------------------ dense route (tests, small problems)
    def dense(self):
        """Part A, steps 4-5: the explicit n_pw x n_pw matrix.

        TODO:
          * H_pq = V_hat[(ix_p - ix_q) % Nx, (iy_p - iy_q) % Ny]. Build the whole matrix
            with fancy indexing and broadcasting (ix[:, None] - ix[None, :]), no Python loop.
          * add the kinetic energy on the diagonal
          * return a complex array (the potential may not be even, so do not drop the
            imaginary part here; eigh_dense decides)
        """
        # this is the hamiltonian enterias cebuse we calcualte V by the diffrance 
        H_pq = self.V_hat[(self.ix[:, None] - self.ix[None, :]) % self.grid.Nx,
                          (self.iy[:, None] - self.iy[None, :]) % self.grid.Ny] 
        H_pq += np.diag(self.kinetic)
        return H_pq

    def eigh_dense(self, n_bands=None):
        """Part A, step 6: eigenvalues and eigenvectors from the dense matrix.

        TODO:
          * H = self.dense()
          * if the imaginary part is negligible (< 1e-12 relative to the largest entry), use the
            real part and la.eigh (faster); otherwise keep it complex. Never discard it unchecked.
          * return (w, v) sorted ascending, restricted to the lowest n_bands if given;
            v[:, i] is the coefficient vector of state i (normalised: sum|c|^2 = 1)
        """
        H = self.dense()
        if np.max(np.abs(H.imag)) < 1e-12 * np.max(np.abs(H)):
            w, v = la.eigh(H.real)
        else:
            w, v = la.eigh(H)
            print("Warning: Hamiltonian has significant imaginary part; using complex diagonalisation.")
        if n_bands is not None:
            w = w[:n_bands]
            v = v[:, :n_bands]
        return w, v

    def _scatter(self, c):
        """Place coefficients c (n_pw,) or (n_pw, k) on the zero-filled grid (Nx, Ny[, k])."""
        c = np.asarray(c)
        C = np.zeros((self.grid.Nx, self.grid.Ny) + c.shape[1:], dtype=complex)
        C[self.ix, self.iy] = c
        return C

    def to_real_space(self, c):
        return np.fft.ifft2(self._scatter(c), axes=(0, 1)) * (self.grid.Nx * self.grid.Ny)

    def apply(self, c):
        self._require_potential()
        c = np.asarray(c)
        N = self.grid.Nx * self.grid.Ny
        psi_r = np.fft.ifft2(self._scatter(c), axes=(0, 1)) * N
        V = self.V_grid.reshape(self.V_grid.shape + (1,) * (c.ndim - 1))      # extra trailing axis if c is a block
        V_c = np.fft.fft2(V * psi_r, axes=(0, 1)) / N
        kin = self.kinetic.reshape(self.kinetic.shape + (1,) * (c.ndim - 1))
        return V_c[self.ix, self.iy] + kin * c

# ------------------------------------------------------------------ helpers
    def _require_potential(self):
        if self.V_hat is None:
            raise RuntimeError("call set_potential() first")
