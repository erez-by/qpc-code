"""qpc.energy: eigenvalue form == direct form of the KS energy (clean wire, converged)."""
import dataclasses

import numpy as np
import pytest

from qpc.energy import total_energy
from qpc.grid import grid_from_cutoff
from qpc.hamiltonian import Hamiltonian
from qpc.hartree import Hartree
from qpc.potential import QPCParams
from qpc.scf import SCFParams, external_potential, run_scf
from qpc.units import Units

U = Units()


@pytest.mark.parametrize("B,pol,scale", [(0.0, False, 1.0), (6.0, True, 1.0), (6.0, True, 0.8)])
def test_energy_forms_agree(B, pol, scale):
    grid = grid_from_cutoff(U.nm_to_au(2000.0), U.nm_to_au(320.0), U.meV_to_au(10.0))
    ham = Hamiltonian(grid, U.meV_to_au(10.0))
    p = SCFParams(method="wire", tol=1e-10, B_T=B, spin_polarized=pol, hartree_scale=scale)
    hart = Hartree(grid, p.a_metal_au(U), verbose=False)
    V = external_potential(grid, QPCParams(), include_qpc=False)
    r = run_scf(ham, hart, V, p, N=56.0, verbose=False)
    assert r.converged
    E = total_energy(r, ham, hart, V, p)
    assert abs(E["E_eig"] - E["E_direct"]) <= 1e-8 * abs(E["E_direct"])
    assert abs(E["N"] - 56.0) < 1e-6
    assert E["TS"] > 0
    # at self-consistency also the standard form: sum f e - E_H - int n v_xc + E_xc (via E_eig) is finite
    assert np.isfinite(E["Omega"])
