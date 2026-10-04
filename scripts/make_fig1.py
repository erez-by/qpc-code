"""PRL-Fig.-1-style panels from saved states (HMW 2003, Fig. 1): one figure per state.

Top:    e_0,s(x) - e_0(far) (lowest transverse level of V_eff,s; solid up, dashed down), mu arrow
        on the left; inset: LDOS at x = 0 vs e - e_0(far) (eta = 0.05 meV Lorentzian; solid up,
        dashed down).
Bottom: n_1D,up - n_1D,dn (solid) and (n_1D,up + n_1D,dn)/2 (dashed), units 1e-2 nm^-1.
Same axes as the paper: x in [-300, 300] nm, vertical 0 ... 1.5 (meV and 1e-2 nm^-1).
Usage: python scripts/make_fig1.py --wx 1.0 --state quad=results/...npz --state exch=results/...npz
Output: figures/fig1_wx{wx}_{label}.png
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from run_qpc import FAR_NM, LX_NM, N1D_NM, setup                  # noqa: E402

from qpc.analysis import barrier_profile, ldos_1d, n1d, net_spin_local   # noqa: E402
from qpc.fourier import min_image                                 # noqa: E402
from qpc.scf import SCFResult, lead_state                         # noqa: E402

INK, MUTED = "#1a1a1a", "#6b6b6b"


def make(wx, label, path, U, grid, ham, hart, q):
    res = SCFResult.load_npz(path)
    lead, _, _ = lead_state(ham, hart, q, res.params, N1D_NM * LX_NM)
    M_loc = net_spin_local(res, lead, grid)
    order = np.argsort(min_image(grid.real_axes()[0], grid.Lx))
    x = U.au_to_nm(min_image(grid.real_axes()[0], grid.Lx))[order]
    far = np.abs(x) > FAR_NM
    e0 = U.au_to_meV(barrier_profile(res, ham))[order]
    e0_far = e0[far].mean(axis=0)
    h = e0 - e0_far[None, :]
    mu_rel = U.au_to_meV(res.mu) - e0_far[0]
    nu = (n1d(res.n_up, grid) / U.length_nm)[order] * 100      # 1e-2 nm^-1
    nd = (n1d(res.n_dn, grid) / U.length_nm)[order] * 100
    E = np.arange(0.0, 1.5, 0.002)
    _, rho = ldos_1d(res, ham, 0.0, E + e0_far[0], eta_meV=0.05, units=U)

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.2, 6.0), sharex=True,
                                 gridspec_kw=dict(hspace=0.08))
    a1.plot(x, h[:, 0], color=INK, lw=2, label="spin up")
    a1.plot(x, h[:, 1], color=INK, lw=2, ls="--", label="spin down")
    a1.annotate("", xy=(-300, mu_rel), xytext=(-255, mu_rel),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    a1.text(-250, mu_rel, r" $\mu$", va="center", color=INK)
    a1.set_ylim(0, 1.5)
    a1.set_ylabel(r"$\epsilon_0(x) - \epsilon_0(\infty)$ (meV)")
    a1.legend(loc="center left", bbox_to_anchor=(0.0, 0.36), frameon=False, fontsize=8)
    a1.set_title(f"$\\hbar\\omega_x$ = {wx} meV, {label}: net spin {M_loc:.2f}", fontsize=10, color=INK)
    ins = a1.inset_axes([0.66, 0.58, 0.32, 0.36])
    ins.plot(E, rho[0], color=INK, lw=1.2)
    ins.plot(E, rho[1], color=INK, lw=1.2, ls="--")
    ins.axvline(mu_rel, color=MUTED, lw=0.8, ls=":")
    ins.set_xlim(0, 1.5)
    ins.set_xlabel(r"$\epsilon$ (meV)", fontsize=7, labelpad=1)
    ins.set_ylabel("LDOS", fontsize=7, labelpad=1)
    ins.tick_params(labelsize=6)
    a2.plot(x, nu - nd, color=INK, lw=2, label=r"$n_\uparrow - n_\downarrow$")
    a2.plot(x, 0.5 * (nu + nd), color=INK, lw=2, ls="--", label=r"$(n_\uparrow + n_\downarrow)/2$")
    a2.set_ylim(0, 1.5)
    a2.set_xlim(-300, 300)
    a2.set_xlabel("x (nm)")
    a2.set_ylabel(r"$n_{1D}$ ($10^{-2}$ nm$^{-1}$)")
    a2.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), frameon=False, fontsize=8)
    for a in (a1, a2):
        a.grid(True, color="#e6e6e6", lw=0.6)
        a.spines[["top", "right"]].set_visible(False)
    os.makedirs("figures", exist_ok=True)
    out = f"figures/fig1_wx{wx}_{label}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"{out}: {path}, M_loc = {M_loc:.3f}, mu - e0(far) = {mu_rel:.3f} meV")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wx", type=float, default=1.0)
    ap.add_argument("--state", action="append", required=True, help="label=path.npz")
    args = ap.parse_args()
    U, grid, ham, hart, q = setup(args.wx)
    for st in args.state:
        label, path = st.split("=", 1)
        make(args.wx, label, path, U, grid, ham, hart, q)


if __name__ == "__main__":
    main()
