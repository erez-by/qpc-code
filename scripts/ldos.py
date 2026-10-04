"""Local 1D DOS at the QPC centre for saved states (no SCF): HMW Fig. 2 / Fig. 1 insets.

For every given npz: rho_s(x0 = 0, e) for eta = 0.05 and 0.1 meV (Lorentzian), energies relative
to mu. For states with M_loc > 0.3 also the spin-up resonance (position rel. to mu, FWHM) and the
energy of max d rho_dn/de; U = their difference (HMW). The clean-wire state of the same B is
needed for M_loc: --wire <npz> (model A) or recomputed for model B (bare reference).
Usage: python scripts/ldos.py --wx 1.0 --files results/qpc_wx1.0_A_B0.npz ... [--wire-dir results]
Output: printed + notes/ldos_<name>.txt (table of e - mu, rho_up, rho_dn).
"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_qpc import HMW_TARGETS, LX_NM, N1D_NM, setup        # noqa: E402

from qpc.analysis import barrier_features, ldos_1d, ldos_features, net_spin_local  # noqa: E402
from qpc.scf import SCFResult, lead_state                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wx", type=float, default=1.0)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--etas", type=float, nargs="+", default=[0.05, 0.1])
    args = ap.parse_args()
    U, grid, ham, hart, q = setup(args.wx)
    for path in args.files:
        res = SCFResult.load_npz(path)
        p = res.params
        lead, mu, _ = lead_state(ham, hart, q, p, N1D_NM * LX_NM)
        M_loc = net_spin_local(res, lead, grid)
        f = barrier_features(res, ham, units=U)
        mu_meV = U.au_to_meV(res.mu)
        e_max = min(U.au_to_meV(np.max(res.eigs[s])) for s in range(2))
        print(f"\n=== {path}: B = {p.B_T} T, lead = {p.lead_reference}, interp = {p.interp}, "
              f"M_loc = {M_loc:.4f}; highest stored level mu + {e_max - mu_meV:.3f} meV", flush=True)
        E = np.arange(f["e0_far"][0] - 0.3, e_max - 0.15, 0.002)
        name = re.sub(r"[^A-Za-z0-9_.]", "_", os.path.basename(path).replace(".npz", ""))
        cols = [E - mu_meV]
        for eta in args.etas:
            _, rho = ldos_1d(res, ham, 0.0, E, eta_meV=eta, units=U)
            cols += [rho[0], rho[1]]
            if M_loc > 0.3:
                lf = ldos_features(E, rho, mu_meV, f["e0_far"][0])
                print(f"  eta = {eta}: spin-up resonance at e - mu = {lf['res_e']:+.4f} meV "
                      f"(e - e_0,up(far) = {lf['res_e'] + f['mu_e0far'][0]:.4f}), FWHM = {lf['fwhm']:.4f} meV; "
                      f"max d rho_dn/de at e - mu = {lf['onset_dn']:+.4f} meV; U = {lf['U']:.4f} meV"
                      + (f"   # HMW: {HMW_TARGETS[args.wx]['ldos']}" if args.wx in HMW_TARGETS else ""),
                      flush=True)
            else:
                print(f"  eta = {eta}: M_loc <= 0.3, no resonance analysis; rho_up/dn at mu = "
                      f"{np.interp(mu_meV, E, rho[0]):.4f} / {np.interp(mu_meV, E, rho[1]):.4f} 1/(meV nm)")
        hdr = "e-mu[meV] " + " ".join(f"rho_up(eta={x}) rho_dn(eta={x})" for x in args.etas)
        np.savetxt(f"notes/ldos_{name}.txt", np.column_stack(cols), fmt="%.6e",
                   header=f"{path}: LDOS at x = 0, 1/(meV nm), Lorentzian; M_loc = {M_loc:.4f}\n{hdr}")
        print(f"  written notes/ldos_{name}.txt")


if __name__ == "__main__":
    main()
