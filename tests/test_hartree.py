import numpy as np
import pytest
from scipy.integrate import quad

from qpc.fourier import min_image
from qpc.grid import Grid, grid_from_cutoff
from qpc.hartree import Hartree, cell_averaged_kernel, kernel_w
from qpc.units import Units

U = Units()
A = U.nm_to_au(100.0)


@pytest.fixture(scope="module")
def prod():
    grid = grid_from_cutoff(U.nm_to_au(5000.0), U.nm_to_au(320.0), U.meV_to_au(15.0))
    assert (grid.Nx, grid.Ny) == (525, 35)
    return grid, Hartree(grid, A, verbose=False)


def phys_xy(grid):
    x, y = grid.real_axes()
    return np.meshgrid(min_image(x, grid.Lx), min_image(y, grid.Ly), indexing="ij")


def test_cell_kernel_vs_quad():
    """Independent check of the cell average (analytic + Gauss) against adaptive quadrature."""
    dy = U.nm_to_au(320.0) / 35
    for k in (0.0, 0.05, 1.0, 8.0):
        W = cell_averaged_kernel([k], [0, 1, 2, 7], dy, A)[0]
        for c, m in enumerate((0, 1, 2, 7)):
            f = lambda y: kernel_w(k, y, A)
            if m == 0:
                ref = 2 * quad(f, 0, dy / 2, limit=200)[0] / dy
            else:
                ref = quad(f, (m - 0.5) * dy, (m + 0.5) * dy, limit=200)[0] / dy
            assert abs(W[c] - ref) <= 1e-9 * max(1.0, abs(ref)), (k, m, W[c], ref)


def test_cell_moments_vs_quad():
    """Moments M_1, M_2 of the quadratic product rule against adaptive quadrature."""
    from qpc.hartree import cell_moment_kernel
    dy = U.nm_to_au(320.0) / 35
    for k in (0.0, 0.3, 5.0):
        for p in (1, 2):
            ms = [-3, -1, 0, 1, 4]
            M = cell_moment_kernel([k], ms, dy, A, p)[0]
            for c, m in enumerate(ms):
                f = lambda t: kernel_w(k, m * dy - t, A) * (t / dy) ** p
                pts = [m * dy] if m == 0 else None
                ref = quad(f, -dy / 2, dy / 2, points=pts, limit=200)[0] / dy
                assert abs(M[c] - ref) <= 1e-9 * max(1.0, abs(ref)), (k, p, m, M[c], ref)


def test_y_rule_convergence_order():
    """Line charge error vs dy: 'cell' rule O(dy^2), 'quadratic' rule O(dy^4)."""
    s = U.nm_to_au(25.0)
    nfun = lambda y: np.exp(-y ** 2 / (2 * s * s)) / (np.sqrt(2 * np.pi) * s)
    ref = 2 * quad(lambda y: nfun(y) * np.log(1 + 4 * A * A / y ** 2), 0, 12 * s, limit=200)[0]
    err = {}
    for rule in ("cell", "quadratic"):
        for Ny in (35, 70):
            g = Grid(U.nm_to_au(5000.0), U.nm_to_au(320.0), 15, Ny)
            H = Hartree(g, A, y_rule=rule, verbose=False)
            y = min_image(np.arange(Ny) * g.dy, g.Ly)
            err[rule, Ny] = abs(H.potential(np.tile(nfun(y), (15, 1)))[0, 0] - ref) / ref
    assert 3.3 < err["cell", 35] / err["cell", 70] < 4.7
    assert err["quadratic", 35] / err["quadratic", 70] > 10
    assert err["quadratic", 35] < 2e-4


def test_line_charge(prod):
    """(a) x-uniform Gaussian line charge: V_H(y=0) = int dy' n(y') ln(1 + 4a^2/y'^2)."""
    grid, H = prod
    s = U.nm_to_au(25.0)
    lam = 1.0
    X, Y = phys_xy(grid)
    nfun = lambda y: lam / (np.sqrt(2 * np.pi) * s) * np.exp(-y ** 2 / (2 * s ** 2))
    V = H.potential(nfun(Y))
    ref = 2 * quad(lambda y: nfun(y) * np.log(1 + 4 * A * A / y ** 2), 0, 12 * s, limit=200)[0]
    assert np.ptp(V[:, 0]) < 1e-12 * abs(ref)                 # x-independent
    assert abs(V[0, 0] - ref) < 1e-3 * abs(ref), (V[0, 0], ref)


def test_gaussian_blob(prod):
    """(b) 2D Gaussian blob at the origin: V_H(0) = int 2 pi rho n(rho) v(rho) d rho."""
    grid, H = prod
    s = U.nm_to_au(30.0)
    Q = 1.0
    X, Y = phys_xy(grid)
    n = Q / (2 * np.pi * s * s) * np.exp(-(X ** 2 + Y ** 2) / (2 * s * s))
    V = H.potential(n)
    ref = quad(lambda r: 2 * np.pi * Q / (2 * np.pi * s * s) * np.exp(-r * r / (2 * s * s))
               * (1 - r / np.sqrt(r * r + 4 * A * A)), 0, 15 * s, limit=200)[0]
    assert abs(V[0, 0] - ref) < 1e-3 * abs(ref), (V[0, 0], ref)


def test_open_boundary_in_y():
    """(c) Ly = 320 nm vs 640 nm (same dy, zero-padded density) agree on the common y points."""
    Lx = U.nm_to_au(5000.0)
    g1 = Grid(Lx, U.nm_to_au(320.0), 525, 35)
    g2 = Grid(Lx, U.nm_to_au(640.0), 525, 70)
    assert abs(g1.dy - g2.dy) < 1e-14
    H1, H2 = Hartree(g1, A, verbose=False), Hartree(g2, A, verbose=False)
    s = U.nm_to_au(30.0)
    X1, Y1 = phys_xy(g1)
    X2, Y2 = phys_xy(g2)
    prof = lambda X, Y: (1 + 0.5 * np.exp(-X ** 2 / (2 * U.nm_to_au(200.0) ** 2))) \
        * np.exp(-Y ** 2 / (2 * s * s))
    n1 = prof(X1, Y1)
    m1 = np.rint(Y1[0] / g1.dy).astype(int)
    m2 = np.rint(Y2[0] / g2.dy).astype(int)
    common = np.isin(m2, m1)
    n2 = np.where(common[None, :], prof(X2, Y2), 0.0)          # zero padding outside the small cell
    V1, V2 = H1.potential(n1), H2.potential(n2)
    order1 = np.argsort(m1)
    order2 = np.argsort(np.where(common, m2, 10 ** 6))[:common.sum()]
    np.testing.assert_array_equal(m1[order1], m2[order2])
    diff = np.abs(V1[:, order1] - V2[:, order2]).max()
    assert diff < 1e-8 * np.abs(V1).max(), diff


def test_positivity_linearity_real(prod):
    """(d) E_H > 0 for random smooth densities; V_H linear and real."""
    grid, H = prod
    rng = np.random.default_rng(1)
    X, Y = phys_xy(grid)
    def smooth_random():
        n = np.zeros_like(X)
        for _ in range(6):
            x0, y0 = rng.uniform(-50, 50), rng.uniform(-8, 8)
            sx, sy = rng.uniform(2, 20), rng.uniform(2, 6)
            n += rng.uniform(-1, 1) * np.exp(-(X - x0) ** 2 / (2 * sx * sx) - (Y - y0) ** 2 / (2 * sy * sy))
        return n
    for _ in range(5):
        n = smooth_random()                    # signed: positive definiteness, not just n > 0
        assert H.energy(n) > 0
    n1, n2 = smooth_random(), smooth_random()
    V12 = H.potential(2.0 * n1 - 3.0 * n2)
    np.testing.assert_allclose(V12, 2.0 * H.potential(n1) - 3.0 * H.potential(n2),
                               atol=1e-12 * np.abs(V12).max())
    assert np.isrealobj(V12)


def test_k0_limit_smooth(prod):
    """(e) the k = 0 and k = k_1 = 2 pi/Lx kernels join smoothly."""
    grid, H = prod
    assert abs(H.k[1] - 2 * np.pi / grid.Lx) < 1e-15
    d01 = np.abs(H.W[1] - H.W[0]).max()
    d12 = np.abs(H.W[2] - H.W[1]).max()
    assert d01 < 1e-2 * np.abs(H.W[0]).max(), d01
    assert d01 < 2 * d12 + 1e-12                    # no jump at k = 0 beyond the k-dependence itself
