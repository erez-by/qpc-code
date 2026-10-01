"""M5.1: interacting clean-wire reference (Hirose & Wingreen, cond-mat/0106581, Eq. 1: bare
parabola + gated Hartree + TC xc; V_QPC = 0, B = 0, unpolarised, fixed N = n_1D Lx).

The potential is x-independent, so the exact KS states are the G_x blocks of
XAveragedPreconditioner (method="wire"). Acceptance (docs/SPEC.md M5.1):
  * SCF converged, N = 140;
  * mu - e_0 = (pi n0 / 2)^2 / 2 with n0 = N0 / Lx from subband 0 ONLY (both spins), within
    2 kT + level spacing;
  * electrons in subband 1 reported (reference 11.5 at kT = 0.05 meV, 9.8 at kT = 0.0086 meV).
Saves results/wire.npz (or results/wire_kT{kT}.npz for a non-default kT).

Usage:
    python scripts/run_wire.py [--kT 0.05] [--force]
    python scripts/run_wire.py --scan [--kT 0.0086]     # N = 100, 110, ..., 170, 185
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qpc.analysis import wire_subbands                     # noqa: E402
from qpc.grid import grid_from_cutoff                      # noqa: E402
from qpc.hamiltonian import Hamiltonian                    # noqa: E402
from qpc.hartree import Hartree                            # noqa: E402
from qpc.potential import QPCParams                        # noqa: E402
from qpc.scf import SCFParams, SCFResult, external_potential, run_scf  # noqa: E402
from qpc.units import Units                                # noqa: E402

LX_NM, LY_NM, ECUT_MEV = 5000.0, 320.0, 15.0
N1D_NM = 2.8e-2
SCAN_N = (100, 110, 120, 130, 140, 150, 160, 170, 185)
KT_SCAN_DEFAULT = 0.0086


def setup(p, U=Units()):
    grid = grid_from_cutoff(U.nm_to_au(LX_NM), U.nm_to_au(LY_NM), U.meV_to_au(ECUT_MEV))
    ham = Hamiltonian(grid, U.meV_to_au(ECUT_MEV))
    hart = Hartree(grid, p.a_metal_au(U), verbose=False)
    V_ext = external_potential(grid, QPCParams(), include_qpc=False)
    return grid, ham, hart, V_ext


def clean_wire_unpolarised(ham, hart, V_ext, p, N, n_init=None, verbose=True, checkpoint=None):
    """Interacting unpolarised clean wire at fixed N; returns (res, bottoms, N_sub) with
    N_sub = electrons per subband, both spins."""
    res = run_scf(ham, hart, V_ext, p, N=N, n_init=n_init, verbose=verbose, checkpoint=checkpoint)
    bottoms, N_sub = wire_subbands(ham, res.V[0], res.mu, p.kT_au())
    return res, bottoms, 2.0 * N_sub


def acceptance(grid, res, bottoms, N_sub, p, N, U=Units()):
    """Returns (ok, dict of the acceptance quantities, Ha*)."""
    kT = p.kT_au(U)
    n0 = N_sub[0] / grid.Lx
    k0 = np.pi * n0 / 2
    pred = k0 ** 2 / 2
    tol = 2 * kT + k0 * 2 * np.pi / grid.Lx
    found = res.mu - bottoms[0]
    ok = res.converged and abs(res.N - N) < 1e-6 and abs(found - pred) < tol
    return ok, dict(pred=pred, found=found, tol=tol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true", help="scan N = 100..185")
    ap.add_argument("--kT", type=float, default=None, help="kT in meV")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    U = Units()
    meV = U.au_to_meV
    os.makedirs("results", exist_ok=True)

    if args.scan:
        kT = KT_SCAN_DEFAULT if args.kT is None else args.kT
        p = SCFParams(spin_polarized=False, method="wire", kT=kT, tol=1e-6)
        grid, ham, hart, V_ext = setup(p, U)
        print(f"clean-wire scan, kT = {kT} meV, a_image = {p.a_image_nm} nm")
        print(f"{'N':>5} {'mu-e0[meV]':>11} {'e1-e0[meV]':>11} {'e1-mu[ueV]':>11} {'N(subband 1)':>13}"
              f" {'pred[meV]':>10} {'it':>4}")
        n_init = None
        for N in SCAN_N:
            res, b, Ns = clean_wire_unpolarised(ham, hart, V_ext, p, float(N), n_init, verbose=False)
            ok, a = acceptance(grid, res, b, Ns, p, float(N), U)
            print(f"{N:5d} {meV(res.mu - b[0]):11.4f} {meV(b[1] - b[0]):11.3f} "
                  f"{1e3 * meV(b[1] - res.mu):+11.1f} {Ns[1]:13.3f} {meV(a['pred']):10.4f} "
                  f"{res.iterations:4d}", flush=True)
            n_init = (res.n_up * (N + 10) / N, res.n_dn * (N + 10) / N)
        return

    kT = 0.05 if args.kT is None else args.kT
    p = SCFParams(spin_polarized=False, method="wire", kT=kT)
    grid, ham, hart, V_ext = setup(p, U)
    N = N1D_NM * LX_NM
    out = "results/wire.npz" if kT == 0.05 else f"results/wire_kT{kT}.npz"
    if os.path.exists(out) and not args.force and SCFResult.load_npz(out).converged:
        print(f"{out} exists and is converged; loading (use --force to recompute)")
        res = SCFResult.load_npz(out)
        b, Ns = wire_subbands(ham, res.V[0], res.mu, p.kT_au(U))
        Ns = 2 * Ns
    else:
        res, b, Ns = clean_wire_unpolarised(ham, hart, V_ext, p, N,
                                            checkpoint=out.replace(".npz", "_ckpt.npz"))
        res.save_npz(out)
        print(f"saved {out}")
    ok, a = acceptance(grid, res, b, Ns, p, N, U)
    print(f"\n=== interacting clean wire (M5.1), kT = {kT} meV, a_image = {p.a_image_nm} nm ===")
    print(f"converged                      : {res.converged} in {res.iterations} iterations, N = {res.N:.6f}")
    print(f"mu_wire                        = {meV(res.mu):.5f} meV")
    print(f"subband bottoms e_n(k_x=0)     = " + ", ".join(f"{meV(x):.5f}" for x in b[:4]) + " meV")
    print(f"subband spacing e_1 - e_0      = {meV(b[1] - b[0]):.5f} meV")
    print(f"e_1 - mu                       = {1e3 * meV(b[1] - res.mu):+.2f} ueV")
    print(f"electrons per subband n=0,1,2  = " + ", ".join(f"{x:.4f}" for x in Ns[:3]))
    print(f"mu - e_0 (found)               = {meV(a['found']):.5f} meV")
    print(f"(pi n0/2)^2/2, n0 = N0/Lx      = {meV(a['pred']):.5f} meV")
    print(f"tolerance 2 kT + level sp.     = {meV(a['tol']):.5f} meV")
    print(f"ACCEPTANCE M5.1: {'PASS' if ok else 'FAIL'}   (subband-1 electrons: {Ns[1]:.3f})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
