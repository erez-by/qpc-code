"""Model B (lead_reference = "bare"): HMW delta formulation with a NON-interacting clean-wire reference."""
import dataclasses

import numpy as np
import pytest

from qpc.grid import grid_from_cutoff
from qpc.hamiltonian import Hamiltonian
from qpc.hartree import Hartree
from qpc.potential import QPCParams
from qpc.scf import SCFParams, external_potential, ks_potentials, lead_state, run_scf
from qpc.units import Units

U = Units()


@pytest.fixture(scope="module")
def prod():
    grid = grid_from_cutoff(U.nm_to_au(5000.0), U.nm_to_au(320.0), U.meV_to_au(15.0))
    ham = Hamiltonian(grid, U.meV_to_au(15.0))
    p = SCFParams(lead_reference="bare", method="wire", tol=1e-9)
    return grid, ham, Hartree(grid, p.a_metal_au(U), verbose=False), p


@pytest.mark.parametrize("B", [0.0, 6.0])
def test_reference_is_fixed_point(prod, B):
    """V_QPC = 0: SCF at fixed mu_ref(B) from a perturbed start returns n_ref,s (< 1e-6 electrons)."""
    grid, ham, hart, p = prod
    pB = dataclasses.replace(p, B_T=B)
    ref, mu, V_ext = lead_state(ham, hart, QPCParams(), pB, 140.0, include_qpc=False)
    # at n = n_ref the KS potential is the bare parabola - sigma E_Z/2
    Vs = ks_potentials(V_ext, hart.potential(ref.n_up + ref.n_dn), ref.n_up, ref.n_dn,
                       pB.EZ_au(U), pB.interp)
    bare = external_potential(grid, QPCParams(), include_qpc=False)
    np.testing.assert_allclose(Vs[0], bare - 0.5 * pB.EZ_au(U), atol=1e-12)
    np.testing.assert_allclose(Vs[1], bare + 0.5 * pB.EZ_au(U), atol=1e-12)
    y = np.linspace(-1, 1, grid.Ny)[None, :]
    r = run_scf(ham, hart, V_ext, pB, mu=mu, n_init=(ref.n_up * (1 + 0.05 * y), ref.n_dn * (1 - 0.03 * y)),
                verbose=False)
    assert r.converged
    dA = grid.dx * grid.dy
    assert np.abs(r.n_up - ref.n_up).sum() * dA < 1e-6
    assert np.abs(r.n_dn - ref.n_dn).sum() * dA < 1e-6
    if B == 0.0:
        assert abs(U.au_to_meV(mu) - 2.1019) < 1e-3 and abs(ref.M) < 1e-10
    else:
        assert ref.M > 0


def test_model_a_unchanged_default():
    assert SCFParams().lead_reference == "interacting"
