"""P vs AP line densities at B = 0 (PRL Fig. 1 bottom-row style), from saved results only (no SCF).

Rows: (wx, V0) = (0.5, 3.0), (0.35, 3.0), (0.5, 3.5); columns: parallel (Phase 1) | antiparallel (Phase 2).
Solid: n_up - n_dn; dashed: (n_up + n_dn)/2, in 1e-2 nm^-1; grey band: depleted region D.
Reads results/two_e/C_exch_wx<W>_V<V0>[_AP]/analysis.npz / analysis.json and _AP/phase3.json.
Output: figures/two_e/fig6_density_P_AP.png

Usage: python scripts/plot_p_ap_density.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402

INK, MUTED, SHADE, GRID = "#1a1a1a", "#6b6b6b", "#ececec", "#e6e6e6"
CASES = [(0.5, 3.0), (0.35, 3.0), (0.5, 3.5)]
OUT = "results/two_e"


def main():
    fig, axs = plt.subplots(len(CASES), 2, figsize=(9.0, 3.0 * len(CASES)), sharey=True, squeeze=False,
                            gridspec_kw=dict(hspace=0.42, wspace=0.08))
    for r, (wx, V0) in enumerate(CASES):
        base = os.path.join(OUT, f"C_exch_wx{wx}_V{V0}")
        p3 = json.load(open(base + "_AP/phase3.json"))
        half = 0.5 * json.load(open(base + "/analysis.json"))["L_D"] + 250
        for c, (sfx, lab) in enumerate((("", "parallel (P)"), ("_AP", "antiparallel (AP)"))):
            a = axs[r, c]
            j = json.load(open(base + sfx + "/analysis.json"))
            z = np.load(base + sfx + "/analysis.npz")
            x, n, s = z["x_nm"], 100 * z["n_x"], 100 * z["s_x"]
            lo, hi = j["D_edges"]
            a.axvspan(lo, hi, color=SHADE, lw=0, zorder=0)
            a.plot(x, s, color=INK, lw=2, label=r"$n_\uparrow - n_\downarrow$")
            a.plot(x, 0.5 * n, color=INK, lw=2, ls="--", label=r"$(n_\uparrow + n_\downarrow)/2$")
            a.axhline(0, color=MUTED, lw=0.6)
            a.set_xlim(-half, half)
            a.set_ylim(-1.4, 1.9)
            a.grid(True, color=GRID, lw=0.6)
            a.spines[["top", "right"]].set_visible(False)
            a.tick_params(colors=MUTED, labelsize=8)
            a.set_title(f"$\\hbar\\omega_x$ = {wx:g} meV, $V_0$ = {V0:g} meV: {lab}, net spin {j['net_spin']:.2f}"
                        + ("" if j["converged"] else " (NOT CONV.)"), fontsize=8.5, color=INK)
            if c == 0:
                a.set_ylabel(r"$n_{1D}$ ($10^{-2}$ nm$^{-1}$)", fontsize=9)
            else:
                a.text(0.98, 0.04, f"$\\Omega_{{AP}}-\\Omega_P$ = {1000 * p3['Delta_meV']:+.0f} $\\mu$eV",
                       transform=a.transAxes, ha="right", fontsize=8, color=INK)
            if r == len(CASES) - 1:
                a.set_xlabel("x (nm)", fontsize=9)
    axs[0, 0].legend(loc="lower left", frameon=False, fontsize=8)
    fig.suptitle("B = 0, model C, exchange: parallel vs antiparallel (grey = depleted region D)",
                 fontsize=10, color=INK, y=0.935)
    os.makedirs("figures/two_e", exist_ok=True)
    fig.savefig("figures/two_e/fig6_density_P_AP.png", dpi=180, bbox_inches="tight")
    print("figures/two_e/fig6_density_P_AP.png")


if __name__ == "__main__":
    main()
