"""qpc.ring_transmission: even/odd phase-shift transmission from the ring eigenstates (Meir's method).
1D Poschl-Teller ring vs the exact T (spec tests 1-4), parity handling, and a small 2D cell vs 1D Numerov."""
import numpy as np
import pytest

from qpc.fourier import min_image
from qpc.ring_transmission import (check_mirror, level_phase, ring_T_1d, split_parity, transmission,
                                   ring_transmission_2d)

D, V0 = 2.0, 0.6


def exact_pt(k, d=D, v0=V0):
    """hbar = m = 1, V = v0 / cosh^2(x/d): T = sinh^2(pi k d) / (sinh^2(pi k d) + cosh^2(pi sqrt(2 v0 d^2 - 1/4)))."""
    s = np.sinh(np.pi * k * d) ** 2
    return s / (s + np.cosh(np.pi * np.sqrt(2 * v0 * d * d - 0.25)) ** 2)


def pt_ring(L, v0=V0):
    N = int(round(1024 * L / 200.0))
    x = min_image(np.arange(N) * L / N, L)
    return v0 / np.cosh(x / D) ** 2


@pytest.mark.parametrize("method", ["M1", "M2"])
def test_poschl_teller_ring(method):
    """L = 200, N = 1024, d = 2, V0 = 0.6: |T - T_exact| < 5e-3 for k in [0.45, 1.5] (spec), achieved ~1e-4."""
    k = np.linspace(0.45, 1.5, 150)
    T, _ = ring_T_1d(pt_ring(200.0), 200.0, k ** 2 / 2, x1=20.0, x2=90.0, method=method)
    assert np.all(np.isfinite(T))
    assert np.max(np.abs(T - exact_pt(k))) < 1e-3


def test_cell_size_independence():
    k = np.linspace(0.45, 1.5, 150)
    T2, _ = ring_T_1d(pt_ring(200.0), 200.0, k ** 2 / 2, x1=20.0, x2=90.0)
    T3, _ = ring_T_1d(pt_ring(300.0), 300.0, k ** 2 / 2, x1=20.0, x2=135.0)
    assert np.max(np.abs(T2 - T3)) < 5e-3
    assert np.max(np.abs(T3 - exact_pt(k))) < 1e-3


def test_asymmetric_potential_raises():
    L = 200.0
    N = 1024
    x = min_image(np.arange(N) * L / N, L)
    V = V0 / np.cosh((x - 1.0) / D) ** 2                       # shifted: no mirror symmetry about x = 0
    with pytest.raises(ValueError):
        ring_T_1d(V, L, np.array([0.5]), x1=20.0, x2=90.0)
    with pytest.raises(ValueError):
        check_mirror(np.add.outer(V, np.zeros(5)), axis=0, tol=1e-9)


def test_bounds_and_free_limit():
    """0 <= T <= 1; V0 -> 0 gives T -> 1 (d_e = d_o mod pi)."""
    k = np.linspace(0.45, 1.5, 60)
    T, ph = ring_T_1d(pt_ring(200.0), 200.0, k ** 2 / 2, x1=20.0, x2=90.0)
    assert np.all((T >= -1e-12) & (T <= 1 + 1e-12))
    T0, ph0 = ring_T_1d(pt_ring(200.0, v0=1e-6), 200.0, k ** 2 / 2, x1=20.0, x2=90.0)
    assert np.max(np.abs(T0 - 1)) < 1e-6


def test_clean_ring_phases_zero_and_degenerate_pairs():
    """Clean ring: levels come in degenerate (cos, sin) pairs; split_parity makes them parity eigenstates and
    M1 gives d = 0 mod pi for both parities, so T = 1."""
    L, N = 100.0, 256
    x = min_image(np.arange(N) * L / N, L)
    k = 2 * np.pi * np.arange(1, 6) / L
    Psi = np.stack([np.exp(1j * kk * x) for kk in k] + [np.exp(-1j * kk * x) for kk in k], axis=1)
    E = np.concatenate([k ** 2 / 2, k ** 2 / 2])
    o = np.argsort(E, kind="stable")
    E, Psi, par = split_parity(E[o], Psi[:, o], (-np.arange(N)) % N)
    assert np.all(np.abs(np.abs(par) - 1) < 1e-12) and np.sum(par > 0) == 5
    kfun = lambda e: np.sqrt(2 * np.asarray(e))                  # noqa: E731
    de, do = level_phase(E[par > 0], kfun, L), level_phase(E[par < 0], kfun, L)
    assert np.max(np.abs(np.sin(de))) < 1e-9 and np.max(np.abs(np.sin(do))) < 1e-9
    Eg = np.linspace(E[0], E[-1], 20)
    np.testing.assert_allclose(transmission(Eg, E[par > 0], de, E[par < 0], do), 1.0, atol=1e-12)


def test_2d_small_cell_vs_numerov():
    """Bare HMW QPC (hbar w_x = 2 meV, V0 = 3 meV) in a 1500 nm cell: ring T(E) vs 1D Numerov on eps_0(x)
    of the plane-wave cutoff basis (the transverse operator the KS Hamiltonian contains), |dT| < 2e-2."""
    from qpc.analysis import transverse_levels
    from qpc.grid import grid_from_cutoff
    from qpc.hamiltonian import Hamiltonian
    from qpc.potential import QPCParams
    from qpc.scf import external_potential
    from qpc.transport import scatter_1d
    from qpc.units import Units
    U = Units()
    g = grid_from_cutoff(U.nm_to_au(1500.0), U.nm_to_au(320.0), U.meV_to_au(15.0))
    ham = Hamiltonian(g, U.meV_to_au(15.0))
    q = QPCParams(hbar_wx_meV=2.0, V0_meV=3.0)
    V = external_potential(g, q)
    E = np.linspace(3.2, 4.8, 81)
    r = ring_transmission_2d(V, ham, E, lead_win_nm=(300.0, 700.0), windows_nm=[(300.0, 700.0)])
    ham.set_potential(V)
    F = np.fft.fft(U.au_to_meV(transverse_levels(V, ham, 1, basis="cutoff")[:, 0])) / g.Nx
    xf = np.arange(-700.0, 700.01, 1.0)
    e0 = np.real(np.exp(1j * U.nm_to_au(xf)[:, None] * g.G_axes()[0][None, :]) @ F)
    T1 = scatter_1d(xf, e0, E)["T"]
    m = np.isfinite(r["T_M1"])
    assert m.sum() > 60
    assert np.max(np.abs(r["T_M1"][m] - T1[m])) < 2e-2
    assert r["E2"] < 4.0 < r["E3"]                              # barrier top between E2 and E3 (y-parity sector)
