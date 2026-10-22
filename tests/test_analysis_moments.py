"""qpc.analysis_moments: eps_0 by finite differences vs analytic / plane-wave transverse levels, lobe
detection and classification on synthetic profiles, full analyse() on a small SCF state."""
import numpy as np
import pytest

from qpc.analysis import transverse_levels
from qpc.analysis_moments import (analyse, classify, depleted_region, eps0_fd, find_lobes,
                                  friedel_spacing, _fourier_y)
from qpc.fourier import min_image
from qpc.grid import grid_from_cutoff
from qpc.hamiltonian import Hamiltonian
from qpc.hartree import Hartree
from qpc.potential import QPCParams, V_barrier
from qpc.scf import SCFParams, external_potential, physical_xy, run_scf
from qpc.units import Units

U = Units()


@pytest.fixture(scope="module")
def cell():
    grid = grid_from_cutoff(U.nm_to_au(1000.0), U.nm_to_au(320.0), U.meV_to_au(15.0))
    return grid, Hamiltonian(grid, U.meV_to_au(15.0)), QPCParams(hbar_wx_meV=0.5, V0_meV=3.0)


def test_eps0_bare_analytic(cell):
    """Bare V_ext: eps_0(x) = V(x)/2 + (w_y + V(x))/2 (hbar = m* = 1), FD with hard walls at +-Ly/2."""
    grid, ham, q = cell
    e = eps0_fd(None, grid, q, dy_nm=0.5)[:, 0]
    x = min_image(grid.real_axes()[0], grid.Lx)
    Vb = V_barrier(x, q)
    exact = 0.5 * Vb + 0.5 * (q.wy + Vb)
    assert U.au_to_meV(np.abs(e - exact).max()) < 2e-4


def test_eps0_matches_plane_wave_levels(cell):
    """V_ext + a smooth potential: FD (V_ext analytic + Fourier-interpolated rest) == the plane-wave
    transverse_levels on the full y grid (an independent method), to 5e-4 meV."""
    grid, ham, q = cell
    X, Y = physical_xy(grid)
    W = U.meV_to_au(0.8) * np.exp(-(Y / U.nm_to_au(60.0)) ** 2) / np.cosh(X / U.nm_to_au(150.0)) \
        + U.meV_to_au(0.3) * np.cos(2 * np.pi * Y / grid.Ly)
    V = external_potential(grid, q) + W
    e_fd = eps0_fd(V, grid, q, dy_nm=0.5)[:, 0]
    e_pw = transverse_levels(V, ham)[:, 0]
    assert U.au_to_meV(np.abs(e_fd - e_pw).max()) < 5e-4


def test_fourier_y_exact_on_grid(cell):
    grid, _, _ = cell
    rng = np.random.default_rng(1)
    Vs = rng.normal(size=(3, grid.Ny))
    np.testing.assert_allclose(_fourier_y(Vs, grid, grid.real_axes()[1]), Vs, atol=1e-12)


def synthetic(signs, friedel=True):
    x = np.arange(-2000.0, 2000.0, 5.0)
    n_lead = 2.8e-2
    n = n_lead * (1 - 0.8 * np.exp(-(x / 300.0) ** 2))
    s = np.zeros_like(x)
    pos = [0.0] if len(signs) == 1 else [-120.0, 120.0]
    for sg, x0 in zip(signs, pos):
        s += sg * 0.8e-2 * np.exp(-((x - x0) / 40.0) ** 2)
    s[np.abs(s) < 1e-12] = 0.0                            # lobes end at a zero of s, as in real profiles
    lam = 2 / n_lead                                      # pi / k_F
    if friedel:
        far = np.abs(x) > 700
        s += np.where(far, 0.5e-2 * np.cos(2 * np.pi * x / lam) * np.exp(-(np.abs(x) - 700) / 2000), 0.0)
    return x, n, s, n_lead, lam


@pytest.mark.parametrize("signs,cls", [((), "none"), ((1,), "single"), ((1, 1), "P"), ((1, -1), "AP")])
def test_lobes_and_classes(signs, cls):
    x, n, s, n_lead, lam = synthetic(signs)
    D, L_D, (xl, xr) = depleted_region(x, n, n_lead)
    assert 0 < L_D < 1000 and xl < 0 < xr
    lobes = find_lobes(x, s, D)
    in_D = [l for l in lobes if l["in_D"]]
    assert classify(in_D) == cls
    for l, sg in zip(in_D, signs):
        assert l["sign"] == sg and abs(abs(l["spin"]) - 0.8e-2 * 40.0 * np.sqrt(np.pi)) < 0.02
    out = [l for l in lobes if not l["in_D"]]
    assert len(out) > 4                                    # Friedel lobes found, but not counted
    assert abs(friedel_spacing(out) - lam) < 0.05 * lam


def test_no_depletion():
    x = np.linspace(-1000, 1000, 401)
    D, L_D, _ = depleted_region(x, np.full_like(x, 2.8e-2), 2.8e-2)
    assert L_D == 0 and not D.any()


def test_analyse_small_state():
    """Runs end to end on a small polarised SCF state: net spin = M, N_D <= N, eps_0 and LDOS present."""
    grid = grid_from_cutoff(60.0, 24.0, 0.8)
    ham = Hamiltonian(grid, 0.8)
    hart = Hartree(grid, U.nm_to_au(50.0), verbose=False)
    q = QPCParams(hbar_wx_meV=1.5)
    V = external_potential(grid, q)
    p = SCFParams(kT=0.5, nb_init=30, tol=1e-8, B_T=2.0, interp="exchange", hartree_scale=0.8)
    r = run_scf(ham, hart, V, p, N=4.0, verbose=False)
    n_lead = 4.0 / U.au_to_nm(grid.Lx)
    info, arr = analyse(r, ham, q, n_lead, units=U)
    assert abs(info["net_spin"] - r.M) < 1e-8
    assert 0 <= info["N_D"] <= r.N + 1e-9
    assert arr["eps0"].shape == (2, grid.Nx) and "ldos_e_meV" in arr and info["ldos"]


def test_single_moment_with_dip_and_satellites():
    """PRL-like single moment: two same-sign peaks with a shallow central dip (90 %) and weak opposite
    satellites at the edges of D -> raw rule says multi4, merged rule says single with spin ~ total."""
    x = np.arange(-1000.0, 1000.0, 2.5)
    n_lead = 2.8e-2
    n = n_lead * (1 - 0.6 * np.exp(-(x / 120.0) ** 2))
    s = 1.25e-2 * (np.exp(-((x - 30) / 35.0) ** 2) + np.exp(-((x + 30) / 35.0) ** 2)) / 1.05   # dip ~ 0.91 x peak
    s -= 0.55e-2 * (np.exp(-((x - 90) / 10.0) ** 2) + np.exp(-((x + 90) / 10.0) ** 2))
    s[np.abs(s) < 1e-12] = 0.0
    D, L_D, _ = depleted_region(x, n, n_lead)
    from qpc.analysis_moments import moments
    raw = [l for l in find_lobes(x, s, D) if l["in_D"]]
    assert classify(raw) == "multi4"
    mom, sats = moments(raw, x, s)
    assert classify(mom) == "single" and mom[0]["n_merged"] == 2 and abs(mom[0]["x"]) < 1
    pos = s.clip(min=0).sum() * (x[1] - x[0])
    assert abs(mom[0]["spin"] - pos) < 0.02 * pos
    assert len(sats) == 2 and all(l["sign"] == -1 for l in sats)


def test_two_separated_moments_stay_P():
    x = np.arange(-1000.0, 1000.0, 2.5)
    s = 0.8e-2 * (np.exp(-((x - 120) / 40.0) ** 2) + np.exp(-((x + 120) / 40.0) ** 2))
    D = np.abs(x) < 400
    from qpc.analysis_moments import moments
    mom, _ = moments([l for l in find_lobes(x, s, D) if l["in_D"]], x, s)
    assert classify(mom) == "P" and len(mom) == 2
    assert all(abs(l["spin"] - 0.8e-2 * 40 * np.sqrt(np.pi)) < 0.01 for l in mom)


def test_density_peaks_in_D_only():
    """Three bumps of a Wigner-like chain inside D are found with their spacing; a stronger bump outside D
    (Friedel / lead) is ignored."""
    from qpc.analysis_moments import density_peaks
    x = np.arange(-1000.0, 1000.0, 2.5)
    n = 2.8e-2 * (1 - 0.5 * np.exp(-(x / 250.0) ** 2))
    for x0 in (-70.0, 0.0, 70.0):
        n += 0.3e-2 * np.exp(-((x - x0) / 20.0) ** 2)
    n += 0.5e-2 * np.exp(-((x - 600.0) / 20.0) ** 2)
    D = np.abs(x) < 300
    pos, prom, sp = density_peaks(x, n, D)
    assert len(pos) == 3 and np.allclose(pos, [-70, 0, 70], atol=3) and np.allclose(sp, [70, 70], atol=5)
    assert all(p > 0.1e-2 for p in prom)
