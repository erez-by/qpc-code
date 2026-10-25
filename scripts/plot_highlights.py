"""Two summary figures of the two-electron study, from saved results only (no SCF).

figures/two_e/highlight1_trapped_spins.png
  (a) s(x) = n_up - n_dn at B = 0 for V0 = 3.5 meV and decreasing hbar w_x (longer QPC), offset vertically,
      the depleted region D shaded, each curve labelled with its trapped spin int s dx.
  (b) trapped spin vs the length L_D of D for all 15 Phase 1 cases (marker shape = V0, filled = converged,
      hollow = not converged), guide lines at whole numbers and the linear fit of the converged states.
figures/two_e/highlight2_P_vs_AP.png
  main case hbar w_x = 0.5, V0 = 3.0: (top) n_up(x), n_dn(x) for P and AP; (bottom) eps_0,s(x) - mu;
  (right) Omega_AP - Omega_P for the three P/AP pairs.

Usage: python scripts/plot_highlights.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402
from matplotlib.gridspec import GridSpec                         # noqa: E402

INK, MUTED, SHADE, GRID = "#1a1a1a", "#6b6b6b", "#ececec", "#e6e6e6"
UP, DN = "#1f5fbf", "#e8710a"                  # spin up / down (also solid / dashed)
OUT = "results/two_e"
WXS, V0S = [1.0, 0.7, 0.5, 0.35, 0.25], [2.5, 3.0, 3.5]
MARK = {2.5: "o", 3.0: "s", 3.5: "^"}


def folder(wx, V0, ap=False):
    return os.path.join(OUT, f"C_exch_wx{float(wx)}_V{float(V0)}" + ("_AP" if ap else ""), "")


def load(wx, V0, ap=False):
    d = folder(wx, V0, ap)
    return json.load(open(d + "analysis.json")), np.load(d + "analysis.npz")


def style(a):
    a.grid(True, color=GRID, lw=0.6)
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(colors=MUTED, labelsize=8)


def highlight1():
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 5.4), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.3))
    style(a)
    series = [1.0, 0.7, 0.5, 0.35]
    step = 2.3
    for k, wx in enumerate(series):
        j, z = load(wx, 3.5)
        x, s = z["x_nm"], 100 * z["s_x"]
        off = step * k
        lo, hi = j["D_edges"]
        a.fill_between([lo, hi], off - 1.0, off + 1.75, color=SHADE, lw=0, zorder=0)
        a.fill_between(x, off, off + s, where=s > 0, color=UP, alpha=0.25, lw=0)
        a.fill_between(x, off, off + s, where=s < 0, color=DN, alpha=0.25, lw=0)
        a.plot(x, off + s, color=INK, lw=1.6)
        a.axhline(off, color=MUTED, lw=0.6)
        a.text(640, off + 0.35, f"$\\hbar\\omega_x$ = {wx:g} meV  (d = {j['d_nm']:.0f} nm)\n"
                                f"trapped spin = {j['net_spin']:.2f}", fontsize=8.5, color=INK, va="center")
    a.set_xlim(-620, 1150)
    a.set_xticks([-600, -400, -200, 0, 200, 400, 600])
    a.set_yticks([])
    a.set_xlabel("x (nm)", fontsize=9)
    a.set_ylabel(r"$n_\uparrow - n_\downarrow$ (offset; $10^{-2}$ nm$^{-1}$ per unit)", fontsize=9)
    a.set_title(r"(a) Spin density, $V_0$ = 3.5 meV, B = 0: longer QPC $\rightarrow$ more trapped spins",
                fontsize=9.5, color=INK, loc="left")
    a.annotate("", xy=(-560, step * 3 + 1.2), xytext=(-560, 0.2),
               arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    a.text(-545, 1.5 * step, "longer QPC", rotation=90, fontsize=8, color=MUTED, va="center")

    style(b)
    conv_pts = []
    for V0 in V0S:
        for wx in WXS:
            try:
                j, _ = load(wx, V0)
            except FileNotFoundError:
                continue
            filled = j["converged"]
            b.plot(j["L_D"], j["net_spin"], MARK[V0], ms=8, mfc=INK if filled else "white", mec=INK, mew=1.3,
                   ls="none")
            if filled:
                conv_pts.append((j["L_D"], j["net_spin"]))
    for k in range(1, 6):
        b.axhline(k, color=MUTED, lw=0.7, ls=":")
    P = np.array(conv_pts)
    kfit, cfit = np.polyfit(P[:, 0], P[:, 1], 1)
    xs = np.linspace(180, 1200, 50)
    b.plot(xs, kfit * xs + cfit, color=MUTED, lw=1.2, ls="--")
    b.text(1190, 0.35, f"dashed: linear fit (converged),\none spin per {1 / kfit:.0f} nm", fontsize=8, color=MUTED,
           ha="right")
    for V0 in V0S:
        b.plot([], [], MARK[V0], ms=7, mfc=INK, mec=INK, ls="none", label=f"$V_0$ = {V0:g} meV")
    b.plot([], [], "o", ms=7, mfc="white", mec=INK, ls="none", label="not converged")
    b.legend(frameon=False, fontsize=8, loc="upper left")
    b.set_xlabel(r"length of the depleted region $L_D$ (nm)", fontsize=9)
    b.set_ylabel(r"trapped spin $\int (n_\uparrow - n_\downarrow)\,dx$", fontsize=9)
    b.set_ylim(0, 6)
    b.set_title("(b) Trapped spin vs QPC length, all 15 cases", fontsize=9.5, color=INK, loc="left")
    fig.savefig("figures/two_e/highlight1_trapped_spins.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def highlight2(wx=0.5, V0=3.0):
    fig = plt.figure(figsize=(12.5, 6.4))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.55], hspace=0.32, wspace=0.28)
    jP, zP = load(wx, V0)
    jA, zA = load(wx, V0, ap=True)
    mu = jP["mu_meV"]
    half = 0.5 * jP["L_D"] + 200
    axes = {}
    for c, (j, z, lab) in enumerate(((jP, zP, "parallel (P)"), (jA, zA, "antiparallel (AP)"))):
        x, n, s = z["x_nm"], 100 * z["n_x"], 100 * z["s_x"]
        nu, nd = 0.5 * (n + s), 0.5 * (n - s)
        lo, hi = j["D_edges"]
        a = fig.add_subplot(gs[0, c], sharey=axes.get("top"))
        axes.setdefault("top", a)
        style(a)
        a.axvspan(lo, hi, color=SHADE, lw=0, zorder=0)
        a.plot(x, nu, color=UP, lw=2, label=r"spin up $n_\uparrow$")
        a.plot(x, nd, color=DN, lw=2, ls="--", label=r"spin down $n_\downarrow$")
        a.set_xlim(-half, half)
        a.set_ylim(0, 2.6)
        dx = x[1] - x[0]
        sr = z["s_x"]                                        # nm^-1, unscaled
        left, right = sr[(x < 0) & (x >= lo)].sum() * dx, sr[(x > 0) & (x <= hi)].sum() * dx
        a.set_title(f"{lab}: spin left {left:+.2f}, right {right:+.2f}, total {j['net_spin']:+.2f}",
                    fontsize=9, color=INK, loc="left")
        if c == 0:
            a.set_ylabel(r"$n_\sigma(x)$ ($10^{-2}$ nm$^{-1}$)", fontsize=9)
            a.legend(frameon=False, fontsize=8, loc="lower left")
        b = fig.add_subplot(gs[1, c], sharey=axes.get("bot"))
        axes.setdefault("bot", b)
        style(b)
        b.axvspan(lo, hi, color=SHADE, lw=0, zorder=0)
        b.plot(x, z["eps0"][0] - mu, color=UP, lw=2, label=r"$\epsilon_{0,\uparrow}(x)-\mu$")
        b.plot(x, z["eps0"][1] - mu, color=DN, lw=2, ls="--", label=r"$\epsilon_{0,\downarrow}(x)-\mu$")
        b.axhline(0, color=INK, lw=0.9, ls=":")
        b.text(-half + 15, 0.04, r"$\mu$", fontsize=9, color=INK)
        b.set_xlim(-half, half)
        b.set_xlabel("x (nm)", fontsize=9)
        if c == 0:
            b.set_ylabel("effective 1D barrier (meV)", fontsize=9)
            b.legend(frameon=False, fontsize=8, loc="upper left")
    axes["top"].figure.text(0.125, 0.955, f"$\\hbar\\omega_x$ = {wx:g} meV, $V_0$ = {V0:g} meV, B = 0 "
                            "(grey = depleted region)", fontsize=10, color=INK)

    d = fig.add_subplot(gs[:, 2])
    style(d)
    pairs = [(0.5, 3.0), (0.35, 3.0), (0.5, 3.5)]
    vals = [1000 * json.load(open(folder(w, v, True) + "phase3.json"))["Delta_meV"] for w, v in pairs]
    ys = np.arange(len(pairs))[::-1]
    d.barh(ys, vals, height=0.45, color=MUTED)
    for y, v in zip(ys, vals):
        d.text(v + 3, y, f"+{v:.0f} $\\mu$eV", va="center", fontsize=8.5, color=INK)
    d.set_yticks(ys)
    d.set_yticklabels([f"$\\hbar\\omega_x$ {w:g}\n$V_0$ {v:g}" for w, v in pairs], fontsize=8)
    d.axvline(0, color=INK, lw=0.9)
    d.axvline(50, color=MUTED, lw=0.8, ls=":")
    d.text(52, -0.55, "kT = 50 $\\mu$eV", fontsize=7.5, color=MUTED)
    d.set_xlim(0, 160)
    d.set_ylim(-0.7, len(pairs) - 0.4)
    d.set_xlabel(r"$\Omega_{AP}-\Omega_P$ ($\mu$eV)", fontsize=9)
    d.set_title("P is lower in all\nthree cases", fontsize=9, color=INK, loc="left")
    fig.savefig("figures/two_e/highlight2_P_vs_AP.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs("figures/two_e", exist_ok=True)
    highlight1()
    highlight2()
    print("figures/two_e/highlight1_trapped_spins.png, figures/two_e/highlight2_P_vs_AP.png")


if __name__ == "__main__":
    main()
