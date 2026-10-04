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


def test_net_spin_window():
    """Window moment: localised spin blob inside the window counted fully; a uniform lead
    polarisation identical to the wire's cancels; a blob outside is ignored."""
    from types import SimpleNamespace
    from qpc.analysis import net_spin_window
    from qpc.scf import physical_xy
    grid = grid_from_cutoff(U.nm_to_au(5000.0), U.nm_to_au(320.0), U.meV_to_au(15.0))
    X, Y = physical_xy(grid)
    dA = grid.dx * grid.dy
    lead = 1e-3 * np.exp(-Y ** 2 / 4.0)
    blob = np.exp(-(X ** 2 + Y ** 2) / (2 * 3.0 ** 2))
    blob /= blob.sum() * dA                                           # one electron at x = 0
    far = np.roll(blob, grid.Nx // 2, axis=0)                         # same blob at x = Lx/2
    wire = SimpleNamespace(n_up=lead * 1.2, n_dn=lead * 0.8)
    res = SimpleNamespace(n_up=lead * 1.2 + 0.5 * blob + 0.5 * far, n_dn=lead * 0.8 - 0.5 * blob)
    assert abs(net_spin_window(res, wire, grid) - 1.0) < 1e-6


def test_barrier_features_two_peaks():
    """Synthetic potential with a double-humped barrier: both peaks, the dip and the far level found."""
    from types import SimpleNamespace
    from qpc.analysis import barrier_features
    from qpc.scf import physical_xy
    grid = grid_from_cutoff(U.nm_to_au(5000.0), U.nm_to_au(320.0), U.meV_to_au(15.0))
    ham = Hamiltonian(grid, U.meV_to_au(15.0))
    X, Y = physical_xy(grid)
    x_nm = U.au_to_nm(X)
    bump = lambda x0: np.exp(-(x_nm - x0) ** 2 / (2 * 30.0 ** 2))
    V = external_potential(grid, QPCParams(), include_qpc=False) + U.meV_to_au(0.6) * (bump(80) + bump(-80))
    res = SimpleNamespace(V=[V, V], n_up=np.zeros_like(V), n_dn=np.zeros_like(V), mu=U.meV_to_au(2.1))
    f = barrier_features(res, ham)
    for s in range(2):
        assert len(f["peaks"][s]) == 2
        (x1, h1), (x2, h2) = f["peaks"][s]
        assert abs(x1 + 80) < 10 and abs(x2 - 80) < 10
        assert abs(h1 - 0.6) < 0.03 and abs(h2 - 0.6) < 0.03
        assert 0.0 < f["centre"][s] < 0.2
        assert abs(f["e0_far"][s] - 1.0) < 1e-3 and abs(f["mu_e0far"][s] - 1.1) < 1e-3


def test_ldos_clean_wire():
    """Non-interacting harmonic clean wire: (i) the LDOS at x0 integrates over e to
    sum_i int dy |psi_i(x0, y)|^2 (Gaussian broadening, wide window); (ii) in the first subband
    rho(e) ~ 1/(pi hbar v(e)) per spin = 1/(pi sqrt(2 (e - e_0))) in a.u., within the discreteness."""
    from types import SimpleNamespace
    from qpc.analysis import ldos_1d
    grid = grid_from_cutoff(U.nm_to_au(5000.0), U.nm_to_au(320.0), U.meV_to_au(15.0))
    ham = Hamiltonian(grid, U.meV_to_au(15.0))
    ham.set_potential(external_potential(grid, QPCParams(), include_qpc=False))
    e, X = wire_states(ham, 120)                 # all subband-0 states up to ~2.9 meV (< e_1 = 3)
    res = SimpleNamespace(eigs=[e, e], X=[X, X])
    E = np.linspace(0.0, 4.5, 4501)
    _, rho = ldos_1d(res, ham, 0.0, E, eta_meV=0.05, shape="gauss")
    psi = ham.to_real_space(X)[0]
    total = (np.abs(psi) ** 2).sum() * grid.dy / (grid.Lx * grid.Ly) / U.length_nm   # nm^-1
    assert abs(rho[0].sum() * (E[1] - E[0]) / total - 1) < 1e-6
    # free 1D band: rho = 1/(pi v), v = sqrt(2 (e - e0)); convert 1/(Ha* a*) -> 1/(meV nm)
    _, rho_b = ldos_1d(res, ham, 0.0, E, eta_meV=0.1, shape="gauss")
    for e_rel in (0.5, 1.0, 1.5):
        k = np.argmin(np.abs(E - (1.0 + e_rel)))
        exact = 1 / (np.pi * np.sqrt(2 * U.meV_to_au(e_rel))) / (U.energy_meV * U.length_nm)
        assert abs(rho_b[0][k] / exact - 1) < 0.05, (e_rel, rho_b[0][k], exact)
