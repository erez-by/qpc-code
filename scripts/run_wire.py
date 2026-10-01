"""M5.1: clean-wire reference (docs/SPEC.md M5.1, delta formulation).

Default: the NON-interacting harmonic clean wire (hbar w_y = 2 meV, N = n_1D Lx = 140, kT of
SCFParams), the lead reference n_ref / mu_wire of the delta formulation (qpc/scf.py).
Acceptance: only subband 0 occupied and mu - e_0 = k_F^2/2 (k_F = pi n_1D/2) within
2 kT + level spacing. Saves results/wire_ref.npz.

--full: the interacting clean wire with full Hartree + xc (reference = "full"), kept as a
diagnostic (2nd subband at mu; docs/OPEN_QUESTIONS.md #7). Saves results/wire_full.npz.
Usage: python scripts/run_wire.py [--full] [--force]
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
from qpc.scf import (SCFParams, SCFResult, clean_wire_reference, external_potential,  # noqa: E402
                     ks_potentials, run_scf)
from qpc.solver import XAveragedPreconditioner            # noqa: E402
from qpc.units import Units                               # noqa: E402

LX_NM, LY_NM, ECUT_MEV = 5000.0, 320.0, 15.0
N1D_NM = 2.8e-2
OUT_FULL = "results/wire_full.npz"
OUT_REF = "results/wire_ref.npz"


def setup():
    U = Units()
    grid = grid_from_cutoff(U.nm_to_au(LX_NM), U.nm_to_au(LY_NM), U.meV_to_au(ECUT_MEV))
    ham = Hamiltonian(grid, U.meV_to_au(ECUT_MEV))
    return U, grid, ham


def report(U, grid, mu, subbands, N_sub, kT, N, label):
    """Print mu, subbands and the M5.1 acceptance check; return True on PASS."""
    meV = U.au_to_meV
    kF = np.pi * N1D_NM * U.length_nm / 2
    tol = 2 * kT + kF * 2 * np.pi / grid.Lx
    print(f"\n=== clean wire (M5.1): {label} ===")
    print(f"mu_wire                       = {meV(mu):.5f} meV")
    print(f"subband bottoms e_n(k_x=0)    = " + ", ".join(f"{meV(x):.5f}" for x in subbands[:4]) + " meV")
    print(f"subband spacing e_1 - e_0     = {meV(subbands[1] - subbands[0]):.5f} meV")
    print(f"electrons per subband n=0,1,2 = " + ", ".join(f"{x:.4f}" for x in N_sub[:3]) + f"  (of {N:.1f})")
    print(f"mu - e_0                      = {meV(mu - subbands[0]):.5f} meV")
    print(f"k_F^2/2 (k_F = pi n_1D/2)     = {meV(kF ** 2 / 2):.5f} meV")
    print(f"tolerance 2 kT + level sp.    = {meV(tol):.5f} meV")
    ok = N_sub[1:].sum() < 0.01 and abs(mu - subbands[0] - kF ** 2 / 2) < tol
    print(f"ACCEPTANCE (single subband, mu - e_0 = k_F^2/2): {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="interacting wire (diagnostic)")
    ap.add_argument("--force", action="store_true", help="recompute even if results exist")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    U, grid, ham = setup()
    N = N1D_NM * LX_NM
    if not args.full:
        p = SCFParams()
        ref = clean_wire_reference(ham, QPCParams(), N, p.kT_au(U))
        np.savez_compressed(OUT_REF, n_ref=ref.n_ref, mu=ref.mu, eigs=ref.eigs,
                            subbands=ref.subbands, N_sub=ref.N_sub, N=ref.N, kT=ref.kT)
        print(f"saved {OUT_REF}")
        ok = report(U, grid, ref.mu, ref.subbands, ref.N_sub, ref.kT, N,
                    "non-interacting reference (delta formulation)")
        sys.exit(0 if ok else 1)

    p = SCFParams(spin_polarized=False, method="wire", B_T=0.0, reference="full")
    hart = Hartree(grid, U.nm_to_au(p.a_nm))
    V_ext = external_potential(grid, QPCParams(), include_qpc=False)
    if os.path.exists(OUT_FULL) and not args.force and SCFResult.load_npz(OUT_FULL).converged:
        print(f"{OUT_FULL} exists and is converged; loading (use --force to recompute)")
        res = SCFResult.load_npz(OUT_FULL)
    else:
        res = run_scf(ham, hart, V_ext, p, N=N, checkpoint="results/wire_full_checkpoint.npz")
        res.save_npz(OUT_FULL)
        print(f"saved {OUT_FULL}")
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
    print(f"converged: {res.converged} in {res.iterations} iterations, N = {res.N:.6f}")
    report(U, grid, res.mu, sub, N_sub, kT, N, "INTERACTING, reference='full' (diagnostic)")

if __name__ == "__main__":
    main()
