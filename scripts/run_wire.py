"""M5.1: clean-wire reference (V_QPC = 0, B = 0, unpolarised, fixed N = n_1D Lx = 140).

The potential is x-independent, so the exact KS states are the G_x blocks of
XAveragedPreconditioner (method="wire"). Saves results/wire.npz and prints mu_wire and the
acceptance check mu - e_0(k_x = 0) vs k_F^2/2, k_F = pi n_1D / 2 (docs/SPEC.md M5.1).
Usage: python scripts/run_wire.py [--force]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qpc.grid import grid_from_cutoff                     # noqa: E402
from qpc.hamiltonian import Hamiltonian                   # noqa: E402
from qpc.hartree import Hartree                           # noqa: E402
from qpc.occupation import fermi                          # noqa: E402
from qpc.potential import QPCParams                       # noqa: E402
from qpc.scf import (SCFParams, SCFResult, external_potential, ks_potentials,  # noqa: E402
                     run_scf)
from qpc.solver import XAveragedPreconditioner            # noqa: E402
from qpc.units import Units                               # noqa: E402

LX_NM, LY_NM, ECUT_MEV = 5000.0, 320.0, 15.0
N1D_NM = 2.8e-2
OUT = "results/wire.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recompute even if results exist")
    args = ap.parse_args()
    U = Units()
    grid = grid_from_cutoff(U.nm_to_au(LX_NM), U.nm_to_au(LY_NM), U.meV_to_au(ECUT_MEV))
    ham = Hamiltonian(grid, U.meV_to_au(ECUT_MEV))
    p = SCFParams(spin_polarized=False, method="wire", B_T=0.0)
    hart = Hartree(grid, U.nm_to_au(p.a_nm))
    V_ext = external_potential(grid, QPCParams(), include_qpc=False)
    N = N1D_NM * LX_NM

    os.makedirs("results", exist_ok=True)
    if os.path.exists(OUT) and not args.force and SCFResult.load_npz(OUT).converged:
        print(f"{OUT} exists and is converged; loading (use --force to recompute)")
        res = SCFResult.load_npz(OUT)
    else:
        res = run_scf(ham, hart, V_ext, p, N=N, checkpoint="results/wire_checkpoint.npz")
        res.save_npz(OUT)
        print(f"saved {OUT}")

    # analysis: subband bottoms of the self-consistent transverse problem at k_x = 0
    n = res.n_up + res.n_dn
    V = ks_potentials(V_ext, hart.potential(n), res.n_up, res.n_dn, 0.0, p.interp)[0]
    ham.set_potential(V)
    P = XAveragedPreconditioner(ham)
    g0 = int(np.nonzero(np.unique(ham.ix) == 0)[0][0])
    sub = np.sort(P.d[g0])[:4]
    kT = p.kT_au(U)
    # occupation per transverse subband: block levels P.d[g, s] (ascending in s for each G_x)
    d = np.where(P.d > 1e5, np.inf, P.d)
    N_sub = 2 * fermi(d, res.mu, kT).sum(axis=0)
    N_higher = N_sub[1:].sum()
    kF = np.pi * U.nm_to_au(1.0) ** -1 * N1D_NM / 2
    meV = U.au_to_meV
    dE_level = kF * 2 * np.pi / grid.Lx

    print("\n=== clean wire (M5.1) ===")
    print(f"converged: {res.converged} in {res.iterations} iterations, N = {res.N:.6f}")
    print(f"mu_wire                       = {meV(res.mu):.5f} meV")
    print(f"subband bottoms e_n(k_x=0)    = " + ", ".join(f"{meV(x):.5f}" for x in sub) + " meV")
    print(f"subband spacing e_1 - e_0     = {meV(sub[1] - sub[0]):.5f} meV (bare 2.0 meV)")
    print(f"electrons per subband n=0,1,2 = " + ", ".join(f"{x:.4f}" for x in N_sub[:3]))
    print(f"electrons in subbands n >= 1  = {N_higher:.4f}  (of {N:.1f})")
    print(f"mu - e_0                      = {meV(res.mu - sub[0]):.5f} meV")
    print(f"k_F^2/2 (k_F = pi n_1D/2)     = {meV(kF ** 2 / 2):.5f} meV")
    print(f"tolerance ~ 2 kT + level sp.  = {meV(2 * kT + dE_level):.5f} meV")
    ok = N_higher < 0.01 and abs(res.mu - sub[0] - kF ** 2 / 2) < 2 * kT + dE_level
    print(f"ACCEPTANCE (single subband, mu - e_0 = k_F^2/2): {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
