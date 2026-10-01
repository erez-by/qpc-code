import numpy as np
import pytest

from qpc.xc import (exc_vxc, exc_energy, eps_c_pade, eps_x, TC_UNPOL, TC_POL, RY, N_FLOOR)

DA = 0.37        # arbitrary area element


def smooth_densities(seed, shape=(40, 30)):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi, shape[0])[:, None]
    y = np.linspace(0, 2 * np.pi, shape[1])[None, :]
    def field():
        a, b, c = rng.uniform(0.2, 1.0, 3)
        return 0.05 + 0.04 * (np.sin(a * x + c) * np.cos(b * y)) ** 2 + 0.02 * rng.uniform()
    return field(), 0.6 * field()


@pytest.mark.parametrize("interp", ["exchange", "quadratic"])
@pytest.mark.parametrize("spin", [0, 1])
def test_functional_derivative(interp, spin):
    """(i) [E(n + h dn_s) - E(n - h dn_s)]/2h == sum v_s dn_s dA, rel 1e-6."""
    n_up, n_dn = smooth_densities(3)
    rng = np.random.default_rng(11)
    dn = 0.01 * np.cos(np.linspace(0, 3, n_up.size)).reshape(n_up.shape) * rng.uniform(0.5, 1)
    h = 1e-4
    if spin == 0:
        Ep = exc_energy(n_up + h * dn, n_dn, DA, interp)
        Em = exc_energy(n_up - h * dn, n_dn, DA, interp)
    else:
        Ep = exc_energy(n_up, n_dn + h * dn, DA, interp)
        Em = exc_energy(n_up, n_dn - h * dn, DA, interp)
    _, v_up, v_dn = exc_vxc(n_up, n_dn, interp)
    v = v_up if spin == 0 else v_dn
    lhs = (Ep - Em) / (2 * h)
    rhs = np.sum(v * dn) * DA
    assert abs(lhs - rhs) <= 1e-6 * abs(rhs), (lhs, rhs)


@pytest.mark.parametrize("interp", ["exchange", "quadratic"])
def test_pade_limits(interp):
    """(ii) zeta = 0 and zeta = 1 reproduce the two Pade fits exactly (Ry* -> Ha*)."""
    rs = np.array([0.5, 1.0, 2.0, 3.7, 10.0])
    n = 1.0 / (np.pi * rs ** 2)
    pade = lambda r, c: RY * c[0] * (1 + c[1] * np.sqrt(r)) / (
        1 + c[1] * np.sqrt(r) + c[2] * r + c[3] * r ** 1.5)
    eps0, _, _ = exc_vxc(n / 2, n / 2, interp)
    np.testing.assert_allclose(eps0 - eps_x(rs, 0.0)[0], pade(rs, TC_UNPOL), rtol=1e-13)
    eps1, _, _ = exc_vxc(n, np.zeros_like(n), interp)
    np.testing.assert_allclose(eps1 - eps_x(rs, 1.0)[0], pade(rs, TC_POL), rtol=1e-13)
    # high-density limit of the unpolarised fit: a0 = -0.3568 Ry*
    assert abs(eps_c_pade(1e-12, TC_UNPOL)[0] / RY - (-0.3568)) < 1e-4


def test_exchange_unpolarised():
    """(iii) eps_x(zeta = 0) = -0.6002/rs Ha*; fully polarised is sqrt(2) larger."""
    rs = np.array([0.3, 1.0, 4.0])
    np.testing.assert_allclose(eps_x(rs, 0.0)[0] * rs, -0.6002, atol=1e-4)
    np.testing.assert_allclose(eps_x(rs, 1.0)[0] / eps_x(rs, 0.0)[0], np.sqrt(2), rtol=1e-14)


@pytest.mark.parametrize("interp", ["exchange", "quadratic"])
def test_unpolarised_potentials_equal(interp):
    """(iv) v_up == v_dn when n_up == n_dn."""
    n, _ = smooth_densities(5)
    _, v_up, v_dn = exc_vxc(n, n.copy(), interp)
    np.testing.assert_allclose(v_up, v_dn, rtol=0, atol=1e-15)


def test_floor_and_no_nan():
    """(v) zeros below the floor, no NaN anywhere (incl. n = 0, full polarisation, tiny negatives)."""
    n_up = np.array([0.0, 0.3 * N_FLOOR, 0.1, 0.1, 0.05, -1e-14])
    n_dn = np.array([0.0, 0.3 * N_FLOOR, 0.0, 0.1, 0.07, 0.02])
    for interp in ("exchange", "quadratic"):
        eps, v_up, v_dn = exc_vxc(n_up, n_dn, interp)
        for a in (eps, v_up, v_dn):
            assert np.all(np.isfinite(a))
            assert np.all(a[:2] == 0)
        assert np.all(eps[2:] < 0)


def test_vbh_alias():
    """'vbh' is a deprecated alias of 'exchange' (identical output)."""
    n_up, n_dn = smooth_densities(7)
    for a, b in zip(exc_vxc(n_up, n_dn, "vbh"), exc_vxc(n_up, n_dn, "exchange")):
        np.testing.assert_array_equal(a, b)


def test_polarisation_crossing_regression():
    """Regression on the TC coefficients: the fully polarised 2D liquid becomes lower in total
    energy per electron than the unpolarised one between rs = 30 and 40 (35.9 with the current
    coefficients). E(rs, zeta) = (1 + zeta^2)/(2 rs^2) + eps_x + eps_c  [Ha*]. If a coefficient is
    corrected and this fails, the change must be reviewed."""
    from scipy.optimize import brentq
    def dE(rs):
        e0 = 0.5 / rs ** 2 + eps_x(rs, 0.0)[0] + eps_c_pade(rs, TC_UNPOL)[0]
        e1 = 1.0 / rs ** 2 + eps_x(rs, 1.0)[0] + eps_c_pade(rs, TC_POL)[0]
        return e1 - e0
    assert dE(30.0) > 0 and dE(40.0) < 0
    rs_c = brentq(dE, 30.0, 40.0)
    assert 30.0 < rs_c < 40.0
