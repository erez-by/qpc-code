"""Step 5: spin-polarised clean wire at field B (lead state of HMW Eq. 2) and the fixed-mu protocol."""
import dataclasses

import numpy as np
import pytest

from qpc.analysis import n1d, net_spin, net_spin_local
from qpc.grid import grid_from_cutoff
from qpc.hamiltonian import Hamiltonian
from qpc.hartree import Hartree
from qpc.potential import QPCParams
from qpc.scf import SCFParams, clean_wire, external_potential, run_scf
from qpc.units import Units

U = Units()
LX_NM = 2000.0          # reduced cell (E_cut = 10 meV) so the dense QPC route stays fast


@pytest.fixture(scope="module")
def small():
    grid = grid_from_cutoff(U.nm_to_au(LX_NM), U.nm_to_au(320.0), U.meV_to_au(10.0))
    ham = Hamiltonian(grid, U.meV_to_au(10.0))
    p = SCFParams(tol=1e-9, method="dense")
    hart = Hartree(grid, p.a_metal_au(U), verbose=False)
    return grid, ham, hart, p


def test_qpc_solver_reproduces_clean_wire_at_6T(small):
    """V_QPC = 0, B = 6 T: the QPC SCF (dense) at fixed mu_wire(B) started from the clean-wire spin
    densities stays there; far-field n_up, n_dn equal the clean wire within 1e-6 nm^-1."""
    grid, ham, hart, p = small
    pB = dataclasses.replace(p, B_T=6.0)
    q = QPCParams(hbar_wx_meV=1.5)
    w = clean_wire(ham, hart, q, pB, 2.8e-2 * LX_NM)
    assert w.res.converged and w.M > 0.1                      # Zeeman-polarised leads, up favoured
    r = run_scf(ham, hart, external_potential(grid, q, include_qpc=False), pB, mu=w.mu,
                n_init=(w.n_up, w.n_dn), verbose=False)
    assert r.converged
    for a, b in ((r.n_up, w.n_up), (r.n_dn, w.n_dn)):
        diff = np.abs(n1d(a, grid) - n1d(b, grid)).max() / U.length_nm      # nm^-1
        assert diff < 1e-6, diff
    assert abs(net_spin_local(r, w, grid)) < 1e-6
    assert abs(r.N - w.res.N) < 1e-6


def test_clean_wire_B0_unpolarised(small):
    """At B = 0 the spin-polarised clean wire stays unpolarised and equals the unpolarised solution."""
    grid, ham, hart, p = small
    q = QPCParams()
    w = clean_wire(ham, hart, q, p, 2.8e-2 * LX_NM)
    wu = clean_wire(ham, hart, q, dataclasses.replace(p, spin_polarized=False), 2.8e-2 * LX_NM)
    assert abs(w.M) < 1e-8
    assert abs(w.mu - wu.mu) < 1e-9
    np.testing.assert_allclose(w.n_up, wu.n_up, atol=1e-9 * w.n_up.max())
    np.testing.assert_allclose(w.subbands[0], w.subbands[1], atol=1e-10)
    assert abs(net_spin(w.res, grid)) < 1e-8


def test_zeeman_sign_convention(small):
    """sigma = +-1/2 Zeeman: spin up lowered by E_Z/2 -> subband bottoms split by ~E_Z (plus xc
    enhancement, never less), up below down; E_Z = g* mu_B B."""
    grid, ham, hart, p = small
    pB = dataclasses.replace(p, B_T=6.0)
    w = clean_wire(ham, hart, QPCParams(), pB, 2.8e-2 * LX_NM)
    EZ = pB.EZ_au(U)
    assert abs(U.au_to_meV(EZ) - 0.44 * 5.788e-2 * 6.0) < 1e-12
    split = w.subbands[1][0] - w.subbands[0][0]
    assert split > 0.99 * EZ
