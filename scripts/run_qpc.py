"""M5.2: QPC, unpolarised, B = 0, fixed mu = mu_wire, delta formulation (docs/SPEC.md M5.2).

Starts from the clean-wire reference density n_ref. Saves results/qpc_wx{wx}_unpol.npz
(resumable: skipped if a converged file exists; a mid-run checkpoint is reused as start).
Prints iterations, far-field n_1D (|x| > 1.5 um), n_1D(0), N, the barrier height
e_0(0) - e_0(far) of the self-consistent transverse problem, and mu - e_0(far).
Usage: python scripts/run_qpc.py --wx 1.5 --unpolarised [--force]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qpc.analysis import barrier_profile, n1d                     # noqa: E402
from qpc.fourier import min_image                                 # noqa: E402
from qpc.grid import grid_from_cutoff                             # noqa: E402
from qpc.hamiltonian import Hamiltonian                           # noqa: E402
from qpc.hartree import Hartree                                   # noqa: E402
from qpc.potential import QPCParams                               # noqa: E402
from qpc.scf import SCFParams, SCFResult, build_external, clean_wire_reference, run_scf  # noqa: E402
from qpc.units import Units                                       # noqa: E402

LX_NM, LY_NM, ECUT_MEV = 5000.0, 320.0, 15.0
N1D_NM = 2.8e-2
FAR_NM = 1500.0


def summarise(res, ham, U, ref, label):
    g = ham.grid
    meV = U.au_to_meV
    x_nm = U.au_to_nm(min_image(g.real_axes()[0], g.Lx))
    line = n1d(res.n_up + res.n_dn, g) / U.length_nm
    far = np.abs(x_nm) > FAR_NM
    e0 = barrier_profile(res, ham)                       # (Nx, 2) Ha*
    e0_far = e0[far].mean(axis=0)
    print(f"\n=== {label} ===")
    print(f"converged                 : {res.converged} in {res.iterations} iterations")
    print(f"mu (= mu_wire)            : {meV(res.mu):.5f} meV")
    print(f"N                         : {res.N:.4f}   (reference wire {ref.N:.1f})")
    print(f"M = N_up - N_dn           : {res.M:.5f}")
    print(f"far-field n_1D (|x|>1.5um): {line[far].mean():.6f} nm^-1  "
          f"(max dev {np.abs(line[far] / N1D_NM - 1).max() * 100:.3f} %)")
    print(f"n_1D(x = 0)               : {line[0]:.6f} nm^-1")
    for s, name in enumerate(("up", "dn")):
        print(f"e_0,{name}(0) - e_0(far)     : {meV(e0[0, s] - e0_far[s]):.5f} meV   "
              f"(bare V0 = 3.0; e_0(0) = {meV(e0[0, s]):.5f}, e_0(far) = {meV(e0_far[s]):.5f})")
        print(f"mu - e_0,{name}(far)        : {meV(res.mu - e0_far[s]):.5f} meV")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wx", type=float, default=1.5, help="hbar w_x in meV")
    ap.add_argument("--unpolarised", action="store_true", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    U = Units()
    os.makedirs("results", exist_ok=True)
    grid = grid_from_cutoff(U.nm_to_au(LX_NM), U.nm_to_au(LY_NM), U.meV_to_au(ECUT_MEV))
    ham = Hamiltonian(grid, U.meV_to_au(ECUT_MEV))
    p = SCFParams(spin_polarized=False, B_T=0.0)
    hart = Hartree(grid, U.nm_to_au(p.a_nm))
    q = QPCParams(hbar_wx_meV=args.wx)
    ref = clean_wire_reference(ham, q, N1D_NM * LX_NM, p.kT_au(U))
    print(f"reference wire: mu_wire = {U.au_to_meV(ref.mu):.5f} meV, N = {ref.N}")
    V_ext = build_external(grid, hart, q, p, ref)

    out = f"results/qpc_wx{args.wx}_unpol.npz"
    ckpt = f"results/qpc_wx{args.wx}_unpol_ckpt.npz"
    if os.path.exists(out) and not args.force and SCFResult.load_npz(out).converged:
        print(f"{out} exists and is converged; loading (use --force to recompute)")
        res = SCFResult.load_npz(out)
    else:
        n_init = (ref.n_ref / 2, ref.n_ref / 2)
        if os.path.exists(ckpt) and not args.force:
            z = np.load(ckpt)
            n_init = (z["n_up"], z["n_dn"])
            print(f"resuming from checkpoint {ckpt} (iteration {int(z['it'])})")
        t0 = time.perf_counter()
        res = run_scf(ham, hart, V_ext, p, mu=ref.mu, n_init=n_init, checkpoint=ckpt,
                      checkpoint_every=1)
        res.save_npz(out)
        print(f"saved {out}  ({time.perf_counter() - t0:.1f} s)")
    summarise(res, ham, U, ref, f"QPC hbar w_x = {args.wx} meV, unpolarised, B = 0, delta formulation")


if __name__ == "__main__":
    main()
