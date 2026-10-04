"""B = 0 rerun from a converged finite-field state, with per-iteration level diagnostics.

Fixed mu = mu_wire(B = 0) of the chosen model; Pulay (alpha, history), tol 1e-4. Every iteration
records (from the KS OUTPUT of that iteration): M_loc, M_win300, N_up, N_dn, n_up(0), n_dn(0)
[nm^-1], and per spin the number of KS levels within +-0.1 meV of mu and the distance of the
closest level to mu [meV]; printed every 10 iterations, full table saved to notes/. If not
converged: mean / std of M_loc over the last 100 iterations, and whether the M_loc swings coincide
with a change of the level count near mu by one.

Usage: python scripts/b0_rerun.py --wx 1.0 --interp quadratic --start results/qpc_wx1.0_A_quad_B0.25.npz
           [--alpha 0.1 --history 12 --maxiter 400 --kT 0.05]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_qpc import LX_NM, N1D_NM, model_tag, setup, spin_summary, print_features, HEADER, row  # noqa: E402

from qpc.analysis import n1d                                              # noqa: E402
from qpc.fourier import min_image                                         # noqa: E402
from qpc.scf import SCFParams, SCFResult, lead_state, run_scf             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wx", type=float, default=1.0)
    ap.add_argument("--lead", default="interacting")
    ap.add_argument("--interp", default="quadratic")
    ap.add_argument("--start", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--history", type=int, default=12)
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--kT", type=float, default=0.05)
    args = ap.parse_args()
    U, grid, ham, hart, q = setup(args.wx)
    meV = U.au_to_meV
    p = SCFParams(B_T=0.0, kT=args.kT, interp=args.interp, lead_reference=args.lead,
                  alpha=args.alpha, history=args.history, maxiter=args.maxiter, tol=1e-4)
    # leads with default SCF settings (only model, interp and kT matter); the special mixing and
    # maxiter apply to the QPC run only
    p_lead = SCFParams(B_T=0.0, kT=args.kT, interp=args.interp, lead_reference=args.lead)
    lead, mu, V_ext = lead_state(ham, hart, q, p_lead, N1D_NM * LX_NM)
    if hasattr(lead, "res") and not lead.res.converged:
        raise RuntimeError("clean-wire lead state not converged")
    w = getattr(lead, "res", lead)
    start = SCFResult.load_npz(args.start)
    dA = grid.dx * grid.dy
    x_nm = U.au_to_nm(min_image(grid.real_axes()[0], grid.Lx))
    win = np.abs(x_nm) < 300.0
    Mw_tot = float((w.n_up - w.n_dn).sum() * dA)
    Mw_win = float((w.n_up - w.n_dn)[win].sum() * dA)
    tag = f"wx{args.wx}_{model_tag(args.lead, args.interp)}_B0_rerun" + ("" if args.kT == 0.05 else f"_kT{args.kT}")
    rows = []
    hdr = (f"{'it':>4} {'M_loc':>8} {'M_win300':>8} {'N_up':>9} {'N_dn':>9} {'nup(0)':>9} {'ndn(0)':>9} "
           f"{'cnt_up':>6} {'cnt_dn':>6} {'dmin_up':>8} {'dmin_dn':>8} {'resid':>9}")

    def cb(it, m_in, m_out, n_in, n_out, eigs, mu_it):
        nu, nd = n_out
        mloc = float((nu - nd).sum() * dA) - Mw_tot
        mwin = float((nu - nd)[win].sum() * dA) - Mw_win
        cnt, dmin = [], []
        for e in eigs:
            d = meV(np.asarray(e)) - meV(mu_it)
            cnt.append(int(np.sum(np.abs(d) <= 0.1)))
            dmin.append(float(d[np.argmin(np.abs(d))]))
        resid = sum(np.abs(n_out[s] - n_in[s]).sum() for s in range(2)) * dA
        r = (it, mloc, mwin, float(nu.sum() * dA), float(nd.sum() * dA),
             n1d(nu, grid)[0] / U.length_nm, n1d(nd, grid)[0] / U.length_nm,
             cnt[0], cnt[1], dmin[0], dmin[1], resid)
        rows.append(r)
        if it % 10 == 0 or it == 1:
            if it == 1:
                print(hdr, flush=True)
            print(f"{r[0]:4d} {r[1]:8.4f} {r[2]:8.4f} {r[3]:9.4f} {r[4]:9.4f} {r[5]:9.6f} {r[6]:9.6f} "
                  f"{r[7]:6d} {r[8]:6d} {r[9]:+8.4f} {r[10]:+8.4f} {r[11]:9.2e}", flush=True)

    print(f"B = 0 rerun: wx = {args.wx}, model {model_tag(args.lead, args.interp)}, start {args.start}, "
          f"alpha = {args.alpha}, history = {args.history}, maxiter = {args.maxiter}, kT = {args.kT} meV, "
          f"mu = {meV(mu):.5f} meV", flush=True)
    t0 = time.perf_counter()
    res = run_scf(ham, hart, V_ext, p, mu=mu, n_init=(start.n_up, start.n_dn), verbose=False,
                  callback=cb, checkpoint=f"results/qpc_{tag}_ckpt.npz", checkpoint_every=1)
    t_run = time.perf_counter() - t0
    res.save_npz(f"results/qpc_{tag}.npz")
    tab = np.array(rows, dtype=float)
    os.makedirs("notes", exist_ok=True)
    np.savetxt(f"notes/{tag}_table.txt", tab, fmt="%.6g", header=hdr.replace("  ", " "))
    print(f"\nconverged: {res.converged} after {res.iterations} iterations ({t_run:.0f} s); "
          f"table: notes/{tag}_table.txt", flush=True)
    d = spin_summary(res, lead, ham, U, args.wx, args.lead)
    print(HEADER + "\n" + row(0.0, d, res.iterations, t_run, res.converged), flush=True)
    print_features(res, ham, U, d, args.wx, 0.0)
    if not res.converged:
        last = tab[-100:]
        m = last[:, 1]
        print(f"NOT converged: M_loc over the last {len(last)} iterations: mean = {m.mean():.4f}, "
              f"std = {m.std():.4f}, min = {m.min():.4f}, max = {m.max():.4f}", flush=True)
        dM = np.abs(np.diff(m))
        dcnt = np.abs(np.diff(last[:, 7])) + np.abs(np.diff(last[:, 8]))
        big = dM > m.std()
        print(f"swings |dM_loc| > std: {big.sum()} of {len(dM)} steps; with a level-count change near mu: "
              f"{np.sum(big & (dcnt >= 1))}; count changes overall: {np.sum(dcnt >= 1)}; "
              f"corr(|dM_loc|, count change) = {np.corrcoef(dM, dcnt)[0, 1] if dcnt.std() > 0 else float('nan'):.3f}",
              flush=True)


if __name__ == "__main__":
    main()
