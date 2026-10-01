"""M5.2 / M5.3: QPC runs at fixed mu = mu_wire (docs/SPEC.md M5).

Model: bare parabola + V_QPC + V_H[n] + v_xc,s[n] - sigma_s E_Z/2 (a_image = 100 nm).

--unpolarised (M5.2): B = 0; mu_wire from the interacting clean wire (N = 140), whose density is
    the start. Saves results/qpc_wx{wx}_unpol.npz.
--B B1 B2 ... (M5.3, Janak ramp; HMW: "first solve in a polarizing field, then reduce the field"):
    for each B in the given order: (1) clean wire spin-polarised at B, fixed N = 140 (Zeeman in
    the leads, HMW Eq. 2) -> mu_wire(B), saved to results/wire_B{B}.npz; (2) QPC at fixed
    mu = mu_wire(B), starting from the previous converged QPC density (the first B: from the
    clean-wire spin densities of that field). Saves results/qpc_wx{wx}_B{B}.npz; converged files
    are skipped (resumable) and used as the start of the next B.
--seed-b0: B = 0 from the UNPOLARISED solution (results/qpc_wx{wx}_unpol.npz) with a 1 % seed
    polarisation for |x| < 300 nm (n_up *= 1.01, n_dn *= 0.99). Saves results/qpc_wx{wx}_B0_seed.npz.
Prints per B: iterations, time, M, M_loc (= M - M_wire(B)), far-field n_up/n_dn, barrier top
and mu - top per spin, N; a table is appended to results/ramp_wx{wx}.txt.
Usage: python scripts/run_qpc.py --wx 1.5 --unpolarised
       python scripts/run_qpc.py --wx 1.5 --B 10 9 8 7 6 5 4 3 2 1 0.5 0.25 0
       python scripts/run_qpc.py --wx 1.5 --seed-b0
"""
import argparse
import dataclasses
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qpc.analysis import barrier_profile, n1d, net_spin_local     # noqa: E402
from qpc.fourier import min_image                                 # noqa: E402
from qpc.grid import grid_from_cutoff                             # noqa: E402
from qpc.hamiltonian import Hamiltonian                           # noqa: E402
from qpc.hartree import Hartree                                   # noqa: E402
from qpc.potential import QPCParams                               # noqa: E402
from qpc.scf import (SCFParams, SCFResult, clean_wire, external_potential,  # noqa: E402
                     physical_xy, run_scf)
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


def setup(wx):
    U = Units()
    grid = grid_from_cutoff(U.nm_to_au(LX_NM), U.nm_to_au(LY_NM), U.meV_to_au(ECUT_MEV))
    ham = Hamiltonian(grid, U.meV_to_au(ECUT_MEV))
    hart = Hartree(grid, SCFParams().a_metal_au(U))
    return U, grid, ham, hart, QPCParams(hbar_wx_meV=wx)


def bstr(B):
    return f"{B:g}"


def spin_summary(res, wire, ham, U):
    """Dict of the per-B quantities (meV / nm^-1)."""
    g = ham.grid
    meV = U.au_to_meV
    x_nm = U.au_to_nm(min_image(g.real_axes()[0], g.Lx))
    far = np.abs(x_nm) > FAR_NM
    e0 = barrier_profile(res, ham)
    top = e0.max(axis=0)
    return dict(M=res.M, M_loc=net_spin_local(res, wire, g), N=res.N, M_wire=wire.M,
                nup_far=(n1d(res.n_up, g) / U.length_nm)[far].mean(),
                ndn_far=(n1d(res.n_dn, g) / U.length_nm)[far].mean(),
                top_up=meV(top[0] - e0[far, 0].mean()), top_dn=meV(top[1] - e0[far, 1].mean()),
                mu_top_up=meV(res.mu - top[0]), mu_top_dn=meV(res.mu - top[1]),
                mu=meV(res.mu))


HEADER = (f"{'B[T]':>5} {'M_loc':>8} {'M':>9} {'M_wire':>8} {'it':>4} {'t[s]':>7} {'N':>9} "
          f"{'nup_far':>9} {'ndn_far':>9} {'top_up':>7} {'top_dn':>7} {'mu-top_up':>9} {'mu-top_dn':>9} "
          f"{'conv':>5}")


def row(B, d, it, t, conv):
    return (f"{B:5g} {d['M_loc']:8.4f} {d['M']:9.4f} {d['M_wire']:8.4f} {it:4d} {t:7.1f} {d['N']:9.4f} "
            f"{d['nup_far']:9.6f} {d['ndn_far']:9.6f} {d['top_up']:7.4f} {d['top_dn']:7.4f} "
            f"{d['mu_top_up']:+9.4f} {d['mu_top_dn']:+9.4f} {str(conv):>5}")


def run_ramp(wx, B_list, table_path):
    U, grid, ham, hart, q = setup(wx)
    N = N1D_NM * LX_NM
    prev, wire_prev = None, None
    lines = []
    for B in B_list:
        p = SCFParams(B_T=B)
        t0 = time.perf_counter()
        wire_init = None if wire_prev is None else (wire_prev.n_up, wire_prev.n_dn)
        wire = clean_wire(ham, hart, q, p, N, n_init=wire_init)
        wire.res.save_npz(f"results/wire_B{bstr(B)}.npz")
        print(f"\n##### B = {B:g} T: clean wire converged {wire.res.converged} in {wire.res.iterations} it, "
              f"mu_wire = {U.au_to_meV(wire.mu):.5f} meV, M_wire = {wire.M:.4f} "
              f"({time.perf_counter() - t0:.1f} s)", flush=True)
        wire_prev = wire
        out = f"results/qpc_wx{wx}_B{bstr(B)}.npz"
        ckpt = out.replace(".npz", "_ckpt.npz")
        if os.path.exists(out) and SCFResult.load_npz(out).converged:
            res = SCFResult.load_npz(out)
            print(f"{out} exists and is converged; skipped", flush=True)
            t_run = float("nan")
        else:
            if os.path.exists(ckpt):
                z = np.load(ckpt)
                n_init = (z["n_up"], z["n_dn"])
                print(f"resuming from checkpoint {ckpt} (iteration {int(z['it'])})", flush=True)
            elif prev is None:
                n_init = (wire.n_up, wire.n_dn)
            else:
                n_init = (prev.n_up, prev.n_dn)
            t1 = time.perf_counter()
            res = run_scf(ham, hart, external_potential(grid, q), p, mu=wire.mu, n_init=n_init,
                          X_init=None if prev is None else prev.X, checkpoint=ckpt, checkpoint_every=1)
            t_run = time.perf_counter() - t1
            res.save_npz(out)
            print(f"saved {out}", flush=True)
        d = spin_summary(res, wire, ham, U)
        print(HEADER + "\n" + row(B, d, res.iterations, t_run, res.converged), flush=True)
        lines.append(row(B, d, res.iterations, t_run, res.converged))
        if not res.converged:
            print(f"WARNING: B = {B:g} not converged; continuing from it", flush=True)
        prev = res
    with open(table_path, "a") as fh:
        fh.write(f"# ramp wx = {wx} meV, B = {list(B_list)}  ({time.ctime()})\n{HEADER}\n"
                 + "\n".join(lines) + "\n")
    print("\n" + HEADER + "\n" + "\n".join(lines))


def run_seed_b0(wx, table_path):
    U, grid, ham, hart, q = setup(wx)
    src = f"results/qpc_wx{wx}_unpol.npz"
    unpol = SCFResult.load_npz(src)
    p = SCFParams(B_T=0.0)
    wire = clean_wire(ham, hart, q, p, N1D_NM * LX_NM)
    X, _ = physical_xy(grid)
    near = np.abs(U.au_to_nm(X)) < 300.0
    n_up = np.where(near, unpol.n_up * 1.01, unpol.n_up)
    n_dn = np.where(near, unpol.n_dn * 0.99, unpol.n_dn)
    print(f"B = 0 from {src} with 1 % seed for |x| < 300 nm: initial M = "
          f"{(n_up - n_dn).sum() * grid.dx * grid.dy:.4f}", flush=True)
    out = f"results/qpc_wx{wx}_B0_seed.npz"
    t1 = time.perf_counter()
    res = run_scf(ham, hart, external_potential(grid, q), p, mu=wire.mu, n_init=(n_up, n_dn),
                  checkpoint=out.replace(".npz", "_ckpt.npz"), checkpoint_every=1)
    t_run = time.perf_counter() - t1
    res.save_npz(out)
    d = spin_summary(res, wire, ham, U)
    line = row(0.0, d, res.iterations, t_run, res.converged)
    print("\n" + HEADER + "\n" + line)
    with open(table_path, "a") as fh:
        fh.write(f"# B = 0 from unpolarised + 1 % seed, wx = {wx} meV ({time.ctime()})\n{HEADER}\n{line}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wx", type=float, default=1.5, help="hbar w_x in meV")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--unpolarised", action="store_true")
    mode.add_argument("--B", type=float, nargs="+", help="Janak ramp, fields in T (in order)")
    mode.add_argument("--seed-b0", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    table = f"results/ramp_wx{args.wx}.txt"
    if args.B is not None:
        return run_ramp(args.wx, args.B, table)
    if args.seed_b0:
        return run_seed_b0(args.wx, table)
    run_unpolarised(args)


def run_unpolarised(args):
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
