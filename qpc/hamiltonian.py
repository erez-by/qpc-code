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
        """Basis list (FFT-array indices inside the cutoff disc) and kinetic diagonal G^2/2."""
        mask = self.grid.cutoff_mask(self.e_cut)
        self.ix, self.iy = np.nonzero(mask)      # indices of the plane waves inside the cutoff
        self.n_pw = len(self.ix)
        self.kinetic = self.grid.kinetic()[self.ix, self.iy]

    # ------------------------------------------------------------------ the part the SCF loop changes
    def set_potential(self, V_grid):
        """Store the potential and its Fourier coefficients V_hat = fft2(V) / (Nx * Ny)."""
        V_grid = np.asarray(V_grid, dtype=float)
        if V_grid.shape != (self.grid.Nx, self.grid.Ny):
            raise ValueError(
                f"V_grid has shape {V_grid.shape}, expected {(self.grid.Nx, self.grid.Ny)}")
        self.V_grid = V_grid
        self.V_hat = np.fft.fft2(V_grid) / (self.grid.Nx * self.grid.Ny)

    # ------------------------------------------------------------------ dense route (tests, small problems)
    def dense(self):
        """Explicit n_pw x n_pw matrix: H_pq = V_hat[(ix_p - ix_q) % Nx, (iy_p - iy_q) % Ny] + G_p^2/2 delta_pq.

        The potential entry depends only on the difference of the two plane waves (Toeplitz
        structure). Returns a complex array; eigh_dense decides whether the imaginary part matters.
        """
        self._require_potential()
        H_pq = self.V_hat[(self.ix[:, None] - self.ix[None, :]) % self.grid.Nx,
                          (self.iy[:, None] - self.iy[None, :]) % self.grid.Ny]
        H_pq += np.diag(self.kinetic)
        return H_pq

    def eigh_dense(self, n_bands=None):
        """Eigenvalues (ascending) and eigenvectors (columns, sum|c|^2 = 1) of the dense matrix."""
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

    # ------------------------------------------------------------------ matrix-free route
    def _scatter(self, c):
        """Place coefficients c (n_pw,) or (n_pw, k) on a zero-filled grid array (Nx, Ny[, k])."""
        c = np.asarray(c)
        C = np.zeros((self.grid.Nx, self.grid.Ny) + c.shape[1:], dtype=complex)
        C[self.ix, self.iy] = c
        return C

    def apply(self, c):
        """H c without building the matrix. c: (n_pw,) or (n_pw, k); output has the same shape.

        Steps: scatter -> inverse FFT (times Nx*Ny) -> multiply by V(r) -> FFT (divide by Nx*Ny)
        -> gather the basis entries -> add the kinetic part. FFTs act on axes (0, 1) only, so a
        block of k vectors is handled in one call.
        """
        self._require_potential()
        c = np.asarray(c)
        N = self.grid.Nx * self.grid.Ny
        psi_r = np.fft.ifft2(self._scatter(c), axes=(0, 1)) * N
        V = self.V_grid.reshape(self.V_grid.shape + (1,) * (c.ndim - 1))      # extra axis for a block
        V_c = np.fft.fft2(V * psi_r, axes=(0, 1)) / N
        kin = self.kinetic.reshape(self.kinetic.shape + (1,) * (c.ndim - 1))
        return V_c[self.ix, self.iy] + kin * c

    def to_real_space(self, c):
        """psi(r_j) = sum_p c[p] exp(i G_p . r_j) on the grid, shape (Nx, Ny[, k]).

        Same scatter + inverse FFT as the first half of apply. For sum|c|^2 = 1 the cell
        average of |psi|^2 is 1 (Parseval). Densities later: n(r) = sum_i f_i |psi_i|^2 / (Lx*Ly).
        """
        return np.fft.ifft2(self._scatter(c), axes=(0, 1)) * (self.grid.Nx * self.grid.Ny)

    # ------------------------------------------------------------------ helpers
    def _require_potential(self):
        if self.V_hat is None:
            raise RuntimeError("call set_potential() first")
