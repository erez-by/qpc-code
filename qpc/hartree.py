"""Gate-screened Hartree potential: Fourier series in x (periodic), direct convolution in y (open).

Interaction of two electrons in the 2DEG with a metal gate plane at distance a (image charge at
distance 2a, opposite sign), atomic units (e^2/eps = 1):

    v(rho) = 1/rho - 1/sqrt(rho^2 + 4 a^2)

PHYSICS-CHOICE: single metal plane at distance a (default a = 100 nm in SCFParams); HMW do not
state their electrostatics (docs/OPEN_QUESTIONS.md #3).

A 2D FFT would add periodic image wires at y = +-Ly, ... (~14 % error for Ly = 320 nm, see
docs/SPEC.md M2), so y is treated with an open boundary: physical y coordinates are
min_image(j dy, Ly) and the kernel is evaluated at the true distance y_i - y_j (no wrap-around).

1D Fourier transform in x of v at fixed y (k = G_x), using int dx cos(kx)/sqrt(x^2+c^2) = 2 K0(|k|c):

    w(k, y) = 2 [ K0(|k||y|) - K0(|k| sqrt(y^2 + 4 a^2)) ]     (k != 0)
    w(0, y) = ln( (y^2 + 4 a^2) / y^2 )                         (k = 0)

Cell-averaged kernel (removes the integrable log singularity at y = 0):

    W[k, i, j] = (1/dy) int_{D - dy/2}^{D + dy/2} w(k, y') dy',   D = y_i - y_j

Hartree potential:  n_k(y) = rfft_x(n)/Nx,  V_k(y_i) = sum_j W[k,i,j] n_k(y_j) dy,
V_H = irfft_x(V_k) * Nx.
"""
import time

import numpy as np
from scipy.special import iti0k0, k0

from .fourier import min_image

# Gauss-Legendre rule on [-1/2, 1/2] for the smooth cell integrals. The nearest singularity of
# w (y' = 0) lies at least dy/2 outside every cell with |D| >= dy, so 24 nodes converge to
# machine precision (Bernstein ellipse rho = 2 + sqrt(3)).
_GL_T, _GL_W = np.polynomial.legendre.leggauss(24)
_GL_T, _GL_W = 0.5 * _GL_T, 0.5 * _GL_W


def kernel_w(k, y, a):
    """Mixed (k, y) kernel w(k, y) of v(rho) = 1/rho - 1/sqrt(rho^2 + 4a^2)   [dimensionless].

    w(k, y) = 2 [K0(|k||y|) - K0(|k| sqrt(y^2 + 4a^2))] for k != 0, ln((y^2+4a^2)/y^2) for k = 0.
    k in a*^-1, y and a in a*. Broadcasts over k and y. y = 0 gives +inf.
    """
    k = np.abs(np.asarray(k, dtype=float))
    y = np.abs(np.asarray(y, dtype=float))
    k, y = np.broadcast_arrays(k, y)
    out = np.empty(k.shape)
    z = k == 0
    with np.errstate(divide="ignore"):
        out[z] = np.log((y[z] ** 2 + 4 * a * a) / y[z] ** 2)
        nz = ~z
        out[nz] = 2 * (k0(k[nz] * y[nz]) - k0(k[nz] * np.sqrt(y[nz] ** 2 + 4 * a * a)))
    return out


def _int0_singular(k, h):
    """int_0^h of the singular part of w(k, y'): 2 K0(k y') (k > 0) or -2 ln y' (k = 0).

    Uses int_0^x K0(t) dt = iti0k0(x)[1] and int_0^h ln y dy = h ln h - h.  k: array, h > 0.
    """
    k = np.asarray(k, dtype=float)
    out = np.empty(k.shape)
    z = k == 0
    out[z] = -2 * (h * np.log(h) - h)
    out[~z] = 2 * iti0k0(k[~z] * h)[1] / k[~z]
    return out


def _int0_regular(k, h, a):
    """int_0^h of the regular part: -2 K0(k sqrt(y^2+4a^2)) (k > 0) or ln(y^2+4a^2) (k = 0); Gauss."""
    k = np.asarray(k, dtype=float)[:, None]
    y = h * (_GL_T + 0.5)[None, :]
    r = np.sqrt(y ** 2 + 4 * a * a)
    with np.errstate(divide="ignore"):
        f = np.where(k == 0, np.log(r ** 2), -2 * k0(np.where(k == 0, 1.0, k) * r))
    return h * (f * _GL_W).sum(axis=1)


def cell_moment_kernel(k, m, dy, a, p):
    """M_p(k, m) = (1/dy) int_{-dy/2}^{dy/2} w(k, m dy - t) (t/dy)^p dt,  p = 0, 1, 2, integer m.

    p = 0 is the plain cell average. M_0, M_2 are even in m, M_1 is odd (w is even in y).
    m = 0, p = 2: adaptive quad (log endpoint singularity at t = 0); m = 0, p = 1: 0 by symmetry.
    |m| >= 1: Gauss-Legendre (24 nodes).  Returns array (len(k), len(m)).
    """
    from scipy.integrate import quad
    if p == 0:
        return cell_averaged_kernel(k, m, dy, a)
    k = np.atleast_1d(np.asarray(k, dtype=float))
    m = np.atleast_1d(np.asarray(m))
    W = np.empty((k.size, m.size))
    for c, mm in enumerate(m):
        if mm == 0:
            if p == 1:
                W[:, c] = 0.0
            else:
                h = 0.5 * dy
                W[:, c] = [2 * quad(lambda t: kernel_w(kk, t, a) * (t / dy) ** 2, 0, h,
                                    epsabs=0, epsrel=1e-12, limit=200)[0] / dy for kk in k]
        else:
            t = _GL_T * dy                                       # t in [-dy/2, dy/2]
            y = mm * dy - t
            W[:, c] = (kernel_w(k[:, None], y[None, :], a) * (t / dy) ** p) @ _GL_W
    return W


def quadratic_product_kernel(k, d, dy, a):
    """Convolution weights for int w(k, y_i - y') n(y') dy' / dy with n quadratic in each cell.

    On cell c (centre y_c, t = y' - y_c): n ~ n_c + t n'_c + t^2 n''_c / 2 with central differences
    n'_c = (n_{c+1} - n_{c-1})/(2 dy), n''_c = (n_{c+1} - 2 n_c + n_{c-1})/dy^2. Collecting the
    coefficient of n_j (d = m_i - m_j):

        K(d) = M0(d) - M2(d) + [M1(d+1) + M2(d+1)]/2 + [M2(d-1) - M1(d-1)]/2

    with the moments M_p of cell_moment_kernel. Error O(dy^4 n'''') instead of O(dy^2 n'') for
    the plain cell average. Cells outside the y range (density 0) are included implicitly,
    so the result equals that of a larger zero-padded cell exactly.  Returns (len(k), len(d)).
    """
    d = np.atleast_1d(np.asarray(d))
    mm = np.arange(np.abs(d).max() + 2)                          # |m| values needed
    M = [cell_moment_kernel(k, mm, dy, a, p) for p in range(3)]
    def get(p, m):                                               # parity: M1 odd, M0/M2 even
        val = M[p][:, np.abs(m)]
        return val * np.sign(m) if p == 1 else val
    return (get(0, d) - get(2, d) + 0.5 * (get(1, d + 1) + get(2, d + 1))
            + 0.5 * (get(2, d - 1) - get(1, d - 1)))


def cell_averaged_kernel(k, m, dy, a):
    """W(k, m dy) = (1/dy) int_{(m-1/2)dy}^{(m+1/2)dy} w(k, y') dy'   for integer m (any sign).

    m = 0: 2 * int_0^{dy/2} w, singular part analytic (iti0k0 / log), regular part Gauss-Legendre.
    |m| >= 1: Gauss-Legendre (24 nodes) on the cell.  Returns array (len(k), len(m)).
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    m = np.abs(np.atleast_1d(np.asarray(m)))
    W = np.empty((k.size, m.size))
    for c, mm in enumerate(m):
        if mm == 0:
            h = 0.5 * dy
            W[:, c] = 2 * (_int0_singular(k, h) + _int0_regular(k, h, a)) / dy
        else:
            y = (mm + _GL_T) * dy
            W[:, c] = kernel_w(k[:, None], y[None, :], a) @ _GL_W
    return W


class Hartree:
    """Gate-screened Hartree potential on a Grid, periodic in x and open in y (docs/SPEC.md M2).

    Attributes: grid, a (a*), k (Nx//2+1,) |G_x| values, W (Nx//2+1, Ny, Ny) y-convolution kernel.

    y_rule: "quadratic" (default): product integration with n quadratic per cell
            (quadratic_product_kernel), O(dy^4); "cell": the plain cell average of SPEC M2, O(dy^2).
    """

    def __init__(self, grid, a, y_rule="quadratic", verbose=True):
        t0 = time.perf_counter()
        self.grid = grid
        self.a = float(a)
        self.y_rule = y_rule
        Nx, Ny, dy = grid.Nx, grid.Ny, grid.dy
        self.k = 2 * np.pi * np.arange(Nx // 2 + 1) / grid.Lx
        # physical y index (open boundary): y_j = min_image(j dy) = m_j dy
        m_j = np.rint(min_image(np.arange(Ny) * dy, grid.Ly) / dy).astype(int)
        diff = m_j[:, None] - m_j[None, :]                       # (Ny, Ny), true separation / dy
        # NUMERICS-CHOICE: SPEC M2 prescribes the cell average (piecewise-constant n), whose
        # O(dy^2) error is 1.7e-3 at production dy for a 25 nm Gaussian (> the 1e-3 test
        # tolerance). The quadratic product rule keeps the kernel exact and only improves the
        # treatment of n inside a cell. See docs/OPEN_QUESTIONS.md.
        if y_rule == "quadratic":
            d_vals = np.arange(-np.abs(diff).max(), np.abs(diff).max() + 1)
            Kd = quadratic_product_kernel(self.k, d_vals, dy, self.a)
            self.W = Kd[:, diff - d_vals[0]]                     # (Nk, Ny, Ny)
        elif y_rule == "cell":
            m_vals = np.arange(np.abs(diff).max() + 1)
            Wm = cell_averaged_kernel(self.k, m_vals, dy, self.a)
            self.W = Wm[:, np.abs(diff)]
        else:
            raise ValueError(y_rule)
        self.setup_time = time.perf_counter() - t0
        if verbose:
            print(f"Hartree: kernel W {self.W.shape} built in {self.setup_time:.2f} s")

    def potential(self, n):
        """V_H(x, y) [Ha*] of the total density n (Nx, Ny) [a*^-2].

        V_H(r) = int d^2r' v(r - r') n(r'), periodic in x, open in y, evaluated as
        V_k(y_i) = sum_j W[k,i,j] n_k(y_j) dy with n_k = rfft_x(n)/Nx, V_H = irfft_x(V_k) Nx.
        """
        g = self.grid
        n = np.asarray(n, dtype=float)
        n_k = np.fft.rfft(n, axis=0) / g.Nx                       # (Nk, Ny)
        V_k = np.einsum("kij,kj->ki", self.W, n_k) * g.dy
        return np.fft.irfft(V_k, n=g.Nx, axis=0) * g.Nx

    def energy(self, n, V_H=None):
        """E_H = (1/2) int n V_H dx dy   [Ha*]."""
        if V_H is None:
            V_H = self.potential(n)
        return 0.5 * float(np.sum(n * V_H)) * self.grid.dx * self.grid.dy
