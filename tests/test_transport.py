"""qpc.transport: 1D scattering vs analytic results and independent solvers, 2D Kwant vs Buttiker and the
clean-wire staircase, 1D vs 2D in the adiabatic limit, wavefunctions, thermal conductance, energy zero."""
import json

import numpy as np
import pytest

from qpc.analysis_moments import eps0_fd
from qpc.grid import grid_from_cutoff
from qpc.potential import QPCParams, V_qpc
from qpc.scf import external_potential, physical_xy
from qpc.transport import (HB2_2M, TransportState, _numerov_q, check_energy_zero, mode_profile, scatter_1d,
                           smooth_coeffs, thermal_conductance, transmission_1d_ivp, transmission_1d_steps,
                           transverse_modes, veff_on_grid, y_points)
from qpc.units import Units

U = Units()


def poschl_teller(E, U0, al):
    """Landau & Lifshitz QM sec. 25 problem 4, U0 / cosh^2(x/alpha), q = 8 m U0 alpha^2 / hbar^2."""
    k = np.sqrt(E / HB2_2M)
    q = 4 * U0 * al ** 2 / HB2_2M
    s = np.sinh(np.pi * k * al) ** 2
    c = np.cosh(0.5 * np.pi * np.sqrt(q - 1)) ** 2 if q > 1 else np.cos(0.5 * np.pi * np.sqrt(1 - q)) ** 2
    return s / (s + c)


def harmonic_y(ys, hw=2.0):
    return hw ** 2 * ys ** 2 / (4 * HB2_2M)          # (1/2) m w^2 y^2 in meV, y in nm


# ----------------------------------------------------------------------------- potential and modes
@pytest.fixture(scope="module")
def cell():
    grid = grid_from_cutoff(U.nm_to_au(1000.0), U.nm_to_au(320.0), U.meV_to_au(15.0))
    q = QPCParams(hbar_wx_meV=0.5, V0_meV=3.0)
    X, Y = physical_xy(grid)
    W = U.meV_to_au(0.8) * np.exp(-(Y / U.nm_to_au(60.0)) ** 2) / np.cosh(X / U.nm_to_au(150.0))
    return grid, q, external_potential(grid, q) + W


def test_veff_reproduces_grid_and_modes_match_eps0_fd(cell):
    grid, q, V = cell
    F = smooth_coeffs(V, grid, q)
    x, y = U.au_to_nm(physical_xy(grid)[0][:, 0]), U.au_to_nm(physical_xy(grid)[1][0])
    np.testing.assert_allclose(veff_on_grid(F, grid, q, x, y), U.au_to_meV(V), atol=1e-10)
    ys = y_points(160.0, 0.5)
    e = transverse_modes(veff_on_grid(F, grid, q, x, ys), ys, 2)
    ref = U.au_to_meV(eps0_fd(V, grid, q, dy_nm=0.5, n_levels=2))
    assert np.abs(e - ref).max() < 1e-6


def test_bare_levels_analytic(cell):
    """Bare V_ext: eps_0(x) = V(x)/2 + (hbar w_y + V(x))/2 (V = V0 sech^2(x/d))."""
    grid, q, _ = cell
    F = np.zeros((grid.Nx, grid.Ny), complex)
    x = np.linspace(-400, 400, 41)
    ys = y_points(160.0, 0.5)
    e0 = transverse_modes(veff_on_grid(F, grid, q, x, ys), ys, 1)[:, 0]
    Vb = q.V0_meV / np.cosh(U.nm_to_au(x) / q.d) ** 2
    assert np.abs(e0 - (0.5 * Vb + 0.5 * (q.hbar_wy_meV + Vb))).max() < 2e-4


# ----------------------------------------------------------------------------- 1D
@pytest.mark.parametrize("U0,al", [(1.0, 40.0), (0.05, 30.0), (-1.0, 40.0)])
def test_poschl_teller(U0, al):
    """Numerov vs the analytic formula (both branches and a well); independent solvers agree."""
    x = np.arange(-1500.0, 1500.01, 1.0)
    e = U0 / np.cosh(x / al) ** 2
    for E in (0.3, 0.9, 1.5):
        T = scatter_1d(x, e, E)["T"][0]
        assert abs(T - poschl_teller(E, U0, al)) < 1e-6
        assert abs(transmission_1d_ivp(x, e, E)[0] - T) < 1e-6
        assert abs(transmission_1d_steps(x, e, E)[0] - T) < 2e-4


def test_flux_reciprocity_random_barriers():
    rng = np.random.default_rng(3)
    x = np.arange(-800.0, 800.01, 1.0)
    for _ in range(5):
        e = sum(rng.uniform(-1.0, 1.5) * np.exp(-((x - rng.uniform(-300, 300)) / rng.uniform(20, 120)) ** 2)
                for _ in range(3))
        E = np.linspace(0.05, 2.0, 7)
        o = scatter_1d(x, e, E)
        assert np.abs(o["T"] + o["R"] - 1).max() < 1e-8
        assert np.abs(o["T"] - o["T_rev"]).max() < 1e-8
        assert abs(transmission_1d_ivp(x, e, 0.8)[0] - scatter_1d(x, e, 0.8)["T"][0]) < 1e-6


def test_clean_channel_and_closed():
    x = np.arange(-300.0, 300.01, 1.0)
    o = scatter_1d(x, np.zeros_like(x), np.array([-0.1, 0.5, 1.0]))
    assert o["T"][0] == 0 and abs(o["T"][1] - 1) < 1e-12 and abs(o["T"][2] - 1) < 1e-12


def test_wavefunctions_1d():
    """Far side |psi_L|^2 = T; incident side 1 + |r|^2 + 2 Re(r e^{-2iqx}); symmetric barrier psi_R(x) = psi_L(-x);
    asymmetric barrier: same T, different |psi_L|^2 and |psi_R(-x)|^2 inside."""
    x = np.arange(-600.0, 600.01, 1.0)
    E = 0.6
    asym = 0.8 * np.exp(-((x + 80) / 30) ** 2) + 0.4 * np.exp(-((x - 60) / 60) ** 2)
    o = scatter_1d(x, asym, E, want_psi=True)
    xf, pL = o["x_full"], o["psi_L"][0]
    right, left = xf > 400, xf < -400
    assert np.abs(np.abs(pL[right]) ** 2 - o["T"][0]).max() < 1e-8
    w = 1 + E / HB2_2M / 12.0
    q = _numerov_q(np.array([w]), 1.0)[0]
    pred = 1 + abs(o["r"][0]) ** 2 + 2 * np.real(o["r"][0] * np.exp(-2j * q * xf[left]))
    assert np.abs(np.abs(pL[left]) ** 2 - pred).max() < 1e-8
    assert abs(o["T"][0] - o["T_rev"][0]) < 1e-10
    mid = np.abs(xf) < 150
    assert np.abs(np.abs(pL[mid]) ** 2 - np.abs(o["psi_R"][0][::-1][mid]) ** 2).max() > 1e-2
    sym = 0.8 * np.exp(-(x / 40) ** 2)
    s = scatter_1d(x, sym, E, want_psi=True)
    assert np.abs(s["psi_L"][0] - s["psi_R"][0][::-1]).max() < 1e-8


def test_thermal_conductance():
    mu, kT = 7.0, 0.05
    E = np.linspace(mu - 8 * kT, mu + 8 * kT, 801)
    one = thermal_conductance(E, np.ones_like(E), np.ones_like(E), mu, kT)
    assert abs(one["G_e2h_kT"] - 2) < 1e-10 and abs(one["g_2e2h_kT"] - 1) < 1e-10
    step = np.heaviside(E - mu, 0.5)                      # theta(0) = 1/2 at the sample E = mu
    st = thermal_conductance(E, step, np.zeros_like(E), mu, kT)
    assert abs(st["T_up_kT"] - 0.5) < 1e-3 and st["T_dn_kT"] == 0


# ----------------------------------------------------------------------------- 2D (Kwant)
def _qpc_lattice(a, X=800.0, W=100.0, wx=0.35, V0=1.0):
    """Analytic QPC potential (HMW Eq. 1, no interaction) on the lattice x = -X..X, y = -W+a..W-a [meV]."""
    q = QPCParams(hbar_wx_meV=wx, V0_meV=V0)
    xs, ys = np.arange(-X, X + 1e-9, a), y_points(W, a)
    xa, ya = U.nm_to_au(xs)[:, None], U.nm_to_au(ys)[None, :]
    V = U.au_to_meV(0.5 * q.wy ** 2 * ya ** 2 + V_qpc(xa, ya, q))
    return V, xs, ys


def test_clean_wire_staircase_and_normalisation():
    kwant = pytest.importorskip("kwant")                                    # noqa: F841
    from qpc.transport import kwant_system, transmission_2d, wavefunctions_2d
    a = 4.0
    ys = y_points(100.0, a)
    xs = np.arange(-200.0, 200.01, a)
    f = kwant_system(np.tile(harmonic_y(ys), (xs.size, 1)), a)
    for E, n in ((1.5, 1), (3.5, 2), (5.5, 3)):
        o = transmission_2d(f, E)
        assert abs(o["T"] - n) < 1e-6 and o["n_open"] == n
    wf = wavefunctions_2d(f, 1.5, xs.size, ys.size)
    for k in ("psi_L", "psi_R"):
        rho = (np.abs(wf[k][0]) ** 2).sum(axis=1)
        assert np.abs(rho - 1).max() < 1e-6


def test_saddle_point_buttiker():
    """V0 - (1/2) m w_x^2 x^2 + (1/2) m w_y^2 y^2 (x parabola frozen beyond |x| = X): Kwant vs Buttiker
    T_n = 1/(1 + exp(-2 pi eps_n)), and vs the 1D solver summed over the (separable) modes."""
    pytest.importorskip("kwant")
    from qpc.transport import kwant_system, transmission_2d
    a, X, wx, V0 = 4.0, 350.0, 0.5, 3.0
    xs, ys = np.arange(-X, X + 1e-9, a), y_points(100.0, a)
    Vx = V0 - wx ** 2 * xs ** 2 / (4 * HB2_2M)
    Vy = harmonic_y(ys)
    f = kwant_system(Vx[:, None] + Vy[None, :], a)
    lev = transverse_modes(Vy[None, :], ys, 3)[0]
    for E in (3.6, 4.0, 4.2, 4.5, 5.0):
        T2 = transmission_2d(f, E)["T"]
        Tb = sum(1 / (1 + np.exp(-2 * np.pi * (E - lev[n] - V0) / wx)) for n in range(3))
        T1 = sum(scatter_1d(xs, Vx + lev[n], E)["T"][0] for n in range(3))
        assert abs(T2 - Tb) < 2e-2
        assert abs(T2 - T1) < 5e-3


def test_adiabatic_long_constriction_1d_vs_2d():
    """Long smooth QPC (wx 0.35 meV): 1D lowest-mode result (dy = a) vs full 2D, < 1e-2 through the step;
    the y-integrated 2D density of the left-incident state follows the 1D |psi_L|^2."""
    pytest.importorskip("kwant")
    from qpc.transport import kwant_system, transmission_2d, wavefunctions_2d
    a = 4.0
    V, xs, ys = _qpc_lattice(a)
    f = kwant_system(V, a)
    e0 = transverse_modes(V, ys, 1)[:, 0]
    lead_e1 = transverse_modes(V[:1], ys, 2)[0, 1]
    for E in np.linspace(e0.max() - 0.15, e0.max() + 0.25, 5):
        assert E < lead_e1
        assert abs(transmission_2d(f, E)["T"] - scatter_1d(xs, e0, E)["T"][0]) < 1e-2
    E = e0.max() + 0.02
    rho2 = (np.abs(wavefunctions_2d(f, E, xs.size, ys.size)["psi_L"][0]) ** 2).sum(axis=1)
    o = scatter_1d(xs, e0, E, want_psi=True)
    rho1 = np.abs(o["psi_L"][0][4:-4]) ** 2
    assert np.abs(rho2 - rho1).max() < 2e-2 * max(rho1.max(), 1)


def test_lattice_convergence_proxy():
    """a = 4 vs 2 nm on the long constriction: |dT| < 1e-2 near the step."""
    pytest.importorskip("kwant")
    from qpc.transport import kwant_system, transmission_2d
    T = []
    for a in (4.0, 2.0):
        V, xs, ys = _qpc_lattice(a, X=600.0)
        e0max = transverse_modes(V[V.shape[0] // 2:V.shape[0] // 2 + 1], ys, 1)[0, 0]
        T.append(transmission_2d(kwant_system(V, a), e0max + 0.05)["T"])
    assert abs(T[0] - T[1]) < 1e-2


# ----------------------------------------------------------------------------- energy zero
def test_check_energy_zero(tmp_path, cell):
    grid, q, _ = cell
    q = QPCParams(hbar_wx_meV=1.5, V0_meV=3.0)                           # short barrier: flat at the edge
    st = TransportState(F=[np.zeros((grid.Nx, grid.Ny), complex)] * 2, grid=grid, qpc=q, mu_meV=1.5,
                        converged=True)
    ys = y_points(150.0, 0.25)
    lev = transverse_modes(harmonic_y(ys, q.hbar_wy_meV)[None, :], ys, 2)[0].tolist()
    p = tmp_path / "wire.json"
    json.dump(dict(mu_meV=1.5, subbands_up_meV=lev, subbands_dn_meV=lev), open(p, "w"))
    out = check_energy_zero(st, p)
    assert abs(out["mu_minus_e0_edge"] - (1.5 - lev[0])) < 1e-6
    st.mu_meV = 1.51
    with pytest.raises(ValueError):
        check_energy_zero(st, p)


def test_mode_profile_symmetric_state(cell):
    """mode_profile on a symmetric synthetic state is even in x."""
    grid, q, V = cell
    st = TransportState(F=[smooth_coeffs(V, grid, q)] * 2, grid=grid, qpc=q, mu_meV=0.0, converged=True)
    x = np.linspace(-300, 300, 61)
    e = mode_profile(st, 0, x, n_modes=1)[:, 0]
    assert np.abs(e - e[::-1]).max() < 1e-8
