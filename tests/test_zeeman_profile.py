"""Optional staggered Zeeman field (two-electron spec): uniform default unchanged, smooth periodic profile,
antisymmetric staggered solution, energy forms with E_Z(x), and d Omega / d mu = -N."""
import dataclasses
import os

import numpy as np
import pytest

from qpc.energy import total_energy
from qpc.grid import grid_from_cutoff
from qpc.hamiltonian import Hamiltonian
from qpc.hartree import Hartree
from qpc.potential import QPCParams
from qpc.scf import SCFParams, external_potential, run_scf, staggered_profile, zeeman_field
from qpc.units import Units

U = Units()
DATA = os.path.join(os.path.dirname(__file__), "data", "zeeman_uniform_regression.npz")


@pytest.fixture(scope="module")
def qpc_small():
    grid = grid_from_cutoff(60.0, 24.0, 0.8)
    ham = Hamiltonian(grid, 0.8)
    hart = Hartree(grid, U.nm_to_au(50.0), verbose=False)
    V = external_potential(grid, QPCParams(hbar_wx_meV=1.5), include_qpc=True)
    return grid, ham, hart, V


def base(**kw):
    d = dict(kT=0.5, a_image_nm=100.0, nb_init=30, tol=1e-9, B_T=2.0, interp="exchange", hartree_scale=0.8)
    d.update(kw)
    return SCFParams(**d)


def test_uniform_default_unchanged(qpc_small):
    """Regression: the default (uniform) field reproduces the state recorded with the code before the
    staggered option existed (tests/data/zeeman_uniform_regression.npz)."""
    grid, ham, hart, V = qpc_small
    ref = np.load(DATA)
    p = base()
    assert p.zeeman_profile == "uniform" and np.isscalar(zeeman_field(grid, p))
    r = run_scf(ham, hart, V, p, N=4.0, verbose=False)
    assert r.iterations == int(ref["iterations"])
    for a, b in ((r.n_up, ref["n_up"]), (r.n_dn, ref["n_dn"]), (r.V[0], ref["V_up"]), (r.V[1], ref["V_dn"])):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-14)
    assert abs(r.mu - float(ref["mu"])) < 1e-12 and abs(r.M - float(ref["M"])) < 1e-10
    E = total_energy(r, ham, hart, V, p)
    assert abs(E["Omega"] - float(ref["Omega"])) < 1e-11


def test_profile_smooth_and_periodic():
    """f is odd, -> +-1 away from the walls, ~0 at +-Lx/2, and its Fourier coefficients decay fast
    (unlike the minimum-image tanh, whose jump at the cell edge gives |c_k| ~ 1/k)."""
    Lx, w, we, N = 5000.0, 20.0, 200.0, 8192
    x = (np.arange(N) * Lx / N + 0.5 * Lx) % Lx - 0.5 * Lx
    f = staggered_profile(x, Lx, w, we)
    np.testing.assert_allclose(staggered_profile(-x, Lx, w, we), -f, atol=1e-14)
    assert abs(staggered_profile(0.5 * Lx, Lx, w, we)) < 1e-9
    assert abs(staggered_profile(500.0, Lx, w, we) - 1) < 1e-6 and abs(staggered_profile(-500.0, Lx, w, we) + 1) < 1e-6
    c = np.abs(np.fft.rfft(f)) / N
    c_mi = np.abs(np.fft.rfft(np.tanh(x / w))) / N
    k_hi = slice(1500, None)                       # wavelengths < 3.3 nm
    assert c[k_hi].max() < 1e-10 * c.max()
    assert c_mi[k_hi].max() > 1e-5 * c_mi.max()    # the min-image tanh rings


def test_staggered_B0_identical(qpc_small):
    grid, ham, hart, V = qpc_small
    r_u = run_scf(ham, hart, V, base(B_T=0.0), N=4.0, verbose=False)
    r_s = run_scf(ham, hart, V, base(B_T=0.0, zeeman_profile="staggered"), N=4.0, verbose=False)
    np.testing.assert_array_equal(r_u.n_up, r_s.n_up)
    np.testing.assert_array_equal(r_u.n_dn, r_s.n_dn)


def test_staggered_antisymmetric(qpc_small):
    """Symmetric QPC + odd field: n_up(x) = n_dn(-x), net spin 0, local spin nonzero; energy forms agree."""
    grid, ham, hart, V = qpc_small
    p = base(B_T=6.0, zeeman_profile="staggered")
    r = run_scf(ham, hart, V, p, N=4.0, verbose=False)
    assert r.converged
    mirror = (-np.arange(grid.Nx)) % grid.Nx
    np.testing.assert_allclose(r.n_up, r.n_dn[mirror], atol=1e-7 * r.n_up.max())
    assert abs(r.M) < 1e-6
    s = (r.n_up - r.n_dn).sum(axis=1) * grid.dy           # s(x) [a*^-1]
    assert np.abs(s).sum() * grid.dx > 1e-3                # local spin on each side, opposite signs
    E = total_energy(r, ham, hart, V, p)
    assert abs(E["E_eig"] - E["E_direct"]) <= 1e-8 * abs(E["E_direct"])


def test_dOmega_dmu_equals_minus_N(qpc_small):
    """Grand potential at fixed mu: d Omega / d mu = -N (central difference, delta mu = 1e-3 meV),
    spin-polarised QPC state, to 1e-3 relative."""
    grid, ham, hart, V = qpc_small
    p = base(tol=1e-11, maxiter=600)
    r0 = run_scf(ham, hart, V, p, N=4.0, verbose=False)
    dmu = U.meV_to_au(1e-3)
    Om = []
    for sgn in (1, -1):
        r = run_scf(ham, hart, V, p, mu=r0.mu + sgn * dmu, n_init=(r0.n_up, r0.n_dn), verbose=False)
        assert r.converged
        Om.append(total_energy(r, ham, hart, V, p)["Omega"])
    deriv = (Om[0] - Om[1]) / (2 * dmu)
    N0 = total_energy(r0, ham, hart, V, p)["N"]
    assert abs(deriv + N0) < 1e-3 * N0


@pytest.mark.parametrize("peaks", [(-60.0, 60.0), (-150.0, 0.0, 150.0), (-200.0, -70.0, 60.0, 190.0)])
def test_alternating_profile(peaks):
    """Sign (-1)^j at each peak x_j, ~0 outside D, odd/even symmetry follows the peaks, smooth FFT decay."""
    from qpc.scf import alternating_profile
    Lx, N = 5000.0, 8192
    x = (np.arange(N) * Lx / N + 0.5 * Lx) % Lx - 0.5 * Lx
    window = (min(peaks) - 80.0, max(peaks) + 80.0)
    f = alternating_profile(x, peaks, window, 20.0, 25.0)
    for j, xj in enumerate(sorted(peaks)):
        v = alternating_profile(np.array([xj]), peaks, window, 20.0, 25.0)[0]
        assert np.sign(v) == (-1) ** j and abs(v) > 0.9
    far = np.abs(x) > max(abs(window[0]), abs(window[1])) + 300
    assert np.abs(f[far]).max() < 1e-9
    c = np.abs(np.fft.rfft(f)) / N
    assert c[1500:].max() < 1e-10 * c.max()


def test_alternating_params_roundtrip(tmp_path):
    """Tuple-valued Zeeman parameters survive save_npz / load_npz (JSON lists -> tuples)."""
    grid = grid_from_cutoff(60.0, 24.0, 0.8)
    ham = Hamiltonian(grid, 0.8)
    hart = Hartree(grid, U.nm_to_au(50.0), verbose=False)
    V = external_potential(grid, QPCParams(hbar_wx_meV=1.5))
    p = base(B_T=4.0, zeeman_profile="alternating", zeeman_peaks_nm=(-80.0, 80.0), zeeman_window_nm=(-200.0, 200.0),
             maxiter=5)
    r = run_scf(ham, hart, V, p, N=4.0, verbose=False)
    r.save_npz(tmp_path / "a.npz")
    from qpc.scf import SCFResult
    q = SCFResult.load_npz(tmp_path / "a.npz")
    assert q.params == p
    EZ = zeeman_field(grid, p)
    assert EZ.shape == (grid.Nx, 1)
