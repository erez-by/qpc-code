"""M5.1 (rewritten acceptance): interacting clean wire, a_image = 100 nm, production grid."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from run_wire import SCAN_N, acceptance, clean_wire_unpolarised, setup   # noqa: E402

from qpc.scf import SCFParams                                              # noqa: E402
from qpc.units import Units                                                # noqa: E402

U = Units()
LX_NM = 5000.0

# reference table (user's independent run), kT = 0.0086 meV:
#            N : (mu-e0 [meV], e1-e0 [meV], e1-mu [ueV], electrons in subband 1)
REF = {100: (0.5613, 1.244, +682.5, 0.000),
       110: (0.6793, 1.147, +468.1, 0.000),
       120: (0.8082, 1.061, +252.5, 0.000),
       130: (0.9447, 0.983, +38.7, 0.120),
       140: (0.9543, 0.948, -6.1, 9.828),
       150: (0.9475, 0.922, -25.2, 20.038),
       160: (0.9520, 0.900, -51.6, 29.903),
       170: (0.9671, 0.880, -86.7, 39.138),
       185: (0.9978, 0.850, -148.3, 51.325)}


@pytest.fixture(scope="module")
def scan():
    p = SCFParams(spin_polarized=False, method="wire", kT=0.0086, tol=1e-6)
    grid, ham, hart, V_ext = setup(p, U)
    out, n_init = {}, None
    for N in SCAN_N:
        res, b, Ns = clean_wire_unpolarised(ham, hart, V_ext, p, float(N), n_init, verbose=False)
        assert res.converged
        out[N] = (res, b, Ns, acceptance(grid, res, b, Ns, p, float(N), U))
        n_init = (res.n_up * (N + 10) / N, res.n_dn * (N + 10) / N)
    return out


def test_scan_table(scan):
    """Table values within 2 % (electron numbers within 0.3)."""
    meV = U.au_to_meV
    for N, (mu_e0, e10, e1mu, N1) in REF.items():
        res, b, Ns, _ = scan[N]
        assert abs(meV(res.mu - b[0]) / mu_e0 - 1) < 0.02, N
        assert abs(meV(b[1] - b[0]) / e10 - 1) < 0.02, N
        assert abs(1e3 * meV(b[1] - res.mu) - e1mu) <= 0.02 * abs(e1mu) + 0.5, N   # ueV
        assert abs(Ns[1] - N1) < 0.3, N


def test_scan_physics(scan):
    """mu - e0 in [0.90, 1.00] meV for n_1D >= 2.6e-2; subband 1 < 0.5 electron for n_1D <= 2.4e-2;
    and the subband-0 relation mu - e0 = (pi n0/2)^2/2 within 2 kT + level spacing."""
    meV = U.au_to_meV
    for N, (res, b, Ns, (ok, a)) in scan.items():
        n1d = N / LX_NM
        if n1d >= 2.6e-2 - 1e-12:
            assert 0.90 <= meV(res.mu - b[0]) <= 1.00, N
        if n1d <= 2.4e-2 + 1e-12:
            assert Ns[1] < 0.5, N
        assert ok, (N, meV(a["found"]), meV(a["pred"]))


def test_acceptance_kT005():
    """Default kT = 0.05 meV, N = 140: acceptance PASS, subband 1 holds 11.5 electrons (+-10 %)."""
    p = SCFParams(spin_polarized=False, method="wire")
    grid, ham, hart, V_ext = setup(p, U)
    res, b, Ns = clean_wire_unpolarised(ham, hart, V_ext, p, 140.0, verbose=False)
    ok, _ = acceptance(grid, res, b, Ns, p, 140.0, U)
    assert ok
    assert abs(Ns[1] / 11.5 - 1) < 0.10
    assert abs(U.au_to_meV(res.mu) - 8.9565) < 2e-3
