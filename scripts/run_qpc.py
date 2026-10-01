"""M5.2: QPC, unpolarised, B = 0, fixed mu = mu_wire (docs/SPEC.md M5.2).

Model: bare parabola + V_QPC + V_H[n] + v_xc[n] (a_image = 100 nm). mu_wire comes from the
interacting clean wire at N = n_1D Lx = 140 (method="wire"), whose density is also the start.
Saves results/qpc_wx{wx}_unpol.npz (resumable: skipped if a converged file exists; the per-
iteration checkpoint is reused as the start after an interruption).
Prints mu_wire, iterations, timing, far-field n_1D (|x| > 1.5 um), n_1D(0), N, the barrier
profile e_0(x) (lowest transverse level of V_eff, PRL Fig. 1(a)-(c)): e_0(far), barrier top
- far band bottom, mu - e_0(far), mu - barrier top.
Usage: python scripts/run_qpc.py --wx 1.5 --unpolarised [--force]
"""
import argparse
import dataclasses
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
from qpc.scf import SCFParams, SCFResult, external_potential, run_scf  # noqa: E402
from qpc.units import Units                                       # noqa: E402

LX_NM, LY_NM, ECUT_MEV = 5000.0, 320.0, 15.0
N1D_NM = 2.8e-2
FAR_NM = 1500.0


def summarise(res, ham, U, mu_wire, label, t_run=None):
    g = ham.grid
    meV = U.au_to_meV
    x_nm = U.au_to_nm(min_image(g.real_axes()[0], g.Lx))
    line = n1d(res.n_up + res.n_dn, g) / U.length_nm
    far = np.abs(x_nm) > FAR_NM
    e0 = barrier_profile(res, ham)                       # (Nx, 2) Ha*
    print(f"\n=== {label} ===")
    print(f"mu_wire (clean wire N=140)    : {meV(mu_wire):.5f} meV")
    print(f"converged                     : {res.converged} in {res.iterations} iterations"
          + (f", {t_run:.1f} s" if t_run is not None else ""))
    if res.history:
        dts = [h[5] for h in res.history]
        print(f"time per iteration            : mean {np.mean(dts):.2f} s, max {np.max(dts):.2f} s")
    print(f"N                             : {res.N:.4f}")
    print(f"M = N_up - N_dn               : {res.M:.5f}")
    print(f"far-field n_1D (|x|>1.5um)    : {line[far].mean():.6f} nm^-1  "
          f"(max dev {np.abs(line[far] / N1D_NM - 1).max() * 100:.3f} %)")
    print(f"n_1D(x = 0)                   : {line[0]:.6f} nm^-1")
    for s, name in enumerate(("up", "dn")):
        far_s = e0[far, s].mean()
        top = e0[:, s].max()
        x_top = x_nm[np.argmax(e0[:, s])]
        print(f"[{name}] e_0(far)                 : {meV(far_s):.5f} meV")
        print(f"[{name}] barrier top - e_0(far)   : {meV(top - far_s):.5f} meV  (top at x = {x_top:.1f} nm;"
              f" e_0(0) - e_0(far) = {meV(e0[0, s] - far_s):.5f}; bare 3.0)")
        print(f"[{name}] mu - e_0(far)            : {meV(res.mu - far_s):.5f} meV")
        print(f"[{name}] mu - barrier top         : {meV(res.mu - top):+.5f} meV")


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
    hart = Hartree(grid, p.a_metal_au(U))
    q = QPCParams(hbar_wx_meV=args.wx)

    # interacting clean wire (exact block states), fixed N = 140 -> mu_wire, start density
    t0 = time.perf_counter()
    pw = dataclasses.replace(p, method="wire")
    wire = run_scf(ham, hart, external_potential(grid, q, include_qpc=False), pw,
                   N=N1D_NM * LX_NM, verbose=False)
    print(f"clean wire: converged {wire.converged} in {wire.iterations} it, "
          f"mu_wire = {U.au_to_meV(wire.mu):.5f} meV ({time.perf_counter() - t0:.1f} s)")

    out = f"results/qpc_wx{args.wx}_unpol.npz"
    ckpt = f"results/qpc_wx{args.wx}_unpol_ckpt.npz"
    t_run = None
    if os.path.exists(out) and not args.force and SCFResult.load_npz(out).converged:
        print(f"{out} exists and is converged; loading (use --force to recompute)")
        res = SCFResult.load_npz(out)
    else:
        n_init = (wire.n_up, wire.n_dn)
        if os.path.exists(ckpt) and not args.force:
            z = np.load(ckpt)
            n_init = (z["n_up"], z["n_dn"])
            print(f"resuming from checkpoint {ckpt} (iteration {int(z['it'])})")
        t0 = time.perf_counter()
        res = run_scf(ham, hart, external_potential(grid, q), p, mu=wire.mu, n_init=n_init,
                      checkpoint=ckpt, checkpoint_every=1)
        t_run = time.perf_counter() - t0
        res.save_npz(out)
        print(f"saved {out}")
    summarise(res, ham, U, wire.mu,
              f"QPC hbar w_x = {args.wx} meV, unpolarised, B = 0, fixed mu = mu_wire", t_run)


if __name__ == "__main__":
    main()
