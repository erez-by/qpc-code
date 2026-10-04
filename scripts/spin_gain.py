"""Spin linear-response gain of the unpolarised QPC at B = 0 (stability of the M = 0 state).

Start from the converged UNPOLARISED state of the chosen model (computed if missing), apply the
seed n_up *= 1.01, n_dn *= 0.99 for |x| < 300 nm, and run the spin-polarised SCF at fixed mu
with LINEAR mixing (Pulay history = 1, alpha = 0.2) for exactly n_it iterations (no stop).

With M_n = int (n_up - n_dn) of the n-th INPUT density, linear mixing gives
    M_{n+1} = M_n + alpha (M_out,n - M_n)  ->  r = M_{n+1}/M_n = 1 + alpha (lambda - 1)
for the slowest spin mode, so lambda = 1 + (r - 1)/alpha is the spin gain dM_out/dM_in of that
mode: lambda < 1 stable (M -> 0), lambda > 1 unstable (a polarised branch grows). The ratios
drift upward while the slowest mode takes over, so lambda is estimated from the last 5 ratios
and r_40 - r_20 shows whether r is still rising.

Usage: python scripts/spin_gain.py --wx 1.5 --interp quadratic --lead interacting [--kT 0.05] [--n-it 40]
Output: notes/spin_gain_wx{wx}_{lead}_{interp}[_kT{kT}].txt
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from run_qpc import LX_NM, N1D_NM, unpolarised_state       # noqa: E402

from qpc.scf import SCFParams, lead_state, physical_xy, run_scf   # noqa: E402
from qpc.units import Units                                       # noqa: E402

ALPHA = 0.2


def spin_gain(wx, interp, lead, kT=0.05, n_it=40, out=print):
    U = Units()
    unpol, _, _, ham, hart, grid, q, _ = unpolarised_state(wx, lead, kT, verbose=False)
    p = SCFParams(B_T=0.0, kT=kT, interp=interp, lead_reference=lead, alpha=ALPHA, history=1,
                  maxiter=n_it)
    lead_obj, mu, V_ext = lead_state(ham, hart, q, p, N1D_NM * LX_NM)
    X, _ = physical_xy(grid)
    near = np.abs(U.au_to_nm(X)) < 300.0
    n_up = np.where(near, unpol.n_up * 1.01, unpol.n_up)
    n_dn = np.where(near, unpol.n_dn * 0.99, unpol.n_dn)
    dA = grid.dx * grid.dy
    M_in, M_out = [], []

    def cb(it, m_in, m_out, n_in, n_out):
        M_in.append(m_in)
        M_out.append(m_out)
        out(f"{it:4d}  M_in = {m_in:.6e}  M_out = {m_out:.6e}"
            + (f"  r = {m_in / M_in[-2]:.5f}" if len(M_in) > 1 else ""))

    out(f"spin gain: wx = {wx} meV, lead = {lead}, interp = {interp}, kT = {kT} meV, "
        f"mu = {U.au_to_meV(mu):.5f} meV, linear mixing alpha = {ALPHA}, {n_it} iterations")
    t0 = time.perf_counter()
    res = run_scf(ham, hart, V_ext, p, mu=mu, n_init=(n_up, n_dn), verbose=False, stop=False,
                  callback=cb)
    M = np.array(M_in + [float((res.n_up - res.n_dn).sum() * dA)])     # M_1 .. M_{n_it+1}
    r = M[1:] / M[:-1]                                                  # r_1 .. r_{n_it}
    r5 = r[-5:].mean()
    lam = 1 + (r5 - 1) / ALPHA
    summary = dict(wx=wx, lead=lead, interp=interp, kT=kT, M0=M[0], M_end=M[-1], r_last5=r5,
                   lam=lam, r20=r[19] if len(r) >= 20 else np.nan, r40=r[-1],
                   lam_out_last=M_out[-1] / M_in[-1], time=time.perf_counter() - t0)
    out(f"\nratios r_n = M_(n+1)/M_n: " + " ".join(f"{x:.4f}" for x in r))
    out(f"M_1 = {M[0]:.5e}, M_{len(M)} = {M[-1]:.5e}")
    out(f"r (mean of last 5) = {r5:.5f}  ->  lambda = 1 + (r - 1)/alpha = {lam:.4f}")
    out(f"r_20 = {summary['r20']:.5f}, r_{len(r)} = {r[-1]:.5f}, r_{len(r)} - r_20 = {r[-1] - summary['r20']:+.5f}")
    # two-step ratio removes a period-2 (alternating) component: rho2_n = sqrt(M_{n+2}/M_n)
    rho2 = np.sqrt(M[2:] / M[:-2])
    lam2 = 1 + (rho2[-1] - 1) / ALPHA
    summary.update(rho2_last=rho2[-1], rho2_20=rho2[17], lam2=lam2)
    out(f"two-step sqrt(M_(n+2)/M_n): last = {rho2[-1]:.5f} (lambda = {lam2:.4f}), at n=18: {rho2[17]:.5f}, "
        f"change = {rho2[-1] - rho2[17]:+.5f}")
    out(f"M_out/M_in at the last iteration = {summary['lam_out_last']:.4f}   ({summary['time']:.0f} s)")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wx", type=float, default=1.5)
    ap.add_argument("--interp", choices=["quadratic", "exchange"], default="quadratic")
    ap.add_argument("--lead", choices=["interacting", "bare"], default="interacting")
    ap.add_argument("--kT", type=float, default=0.05)
    ap.add_argument("--n-it", type=int, default=40)
    args = ap.parse_args()
    os.makedirs("notes", exist_ok=True)
    path = (f"notes/spin_gain_wx{args.wx}_{args.lead}_{args.interp}"
            + ("" if args.kT == 0.05 else f"_kT{args.kT}") + ".txt")
    with open(path, "w") as fh:
        def out(line):
            print(line, flush=True)
            fh.write(line + "\n")
            fh.flush()
        spin_gain(args.wx, args.interp, args.lead, args.kT, args.n_it, out)
    print(f"written {path}")


if __name__ == "__main__":
    main()
