"""Two-electron scan, Phase 1 analysis additions (existing results only, no SCF).

1. For every B = 0 state: peaks of n(x) inside D (qpc.analysis_moments.density_peaks; strong: prominence
   >= 0.1e-2 nm^-1, weak >= 0.05e-2), positions and spacings, vs 1/n_center.
2. Net spin vs L_D and vs N_D, linear fits on the CONVERGED states only (slopes; nm per electron).
3. LDOS at x = 0 (eta 0.05 meV, one full dense diagonalisation per spin) for wx 1.0 / 0.5 / 0.35 at V0 3.0:
   spin-up quasi-bound peaks below mu, spin-down peaks above mu.
Non-converged states are marked everywhere (hollow markers, 'NOT CONV.'); their dOmega is never used.
Output: notes/TWO_E_ANALYSIS.md, figures/two_e/fig1_profiles.png, fig2_netspin.png, fig3_ldos_x0.png.

Usage: python scripts/two_e_analysis.py
"""
import glob
import json
import os
import re
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qpc.analysis_moments import N_PROMINENCE_WEAK_NM, density_peaks, ldos_at   # noqa: E402
from qpc.grid import grid_from_cutoff                            # noqa: E402
from qpc.hamiltonian import Hamiltonian                          # noqa: E402
from qpc.scf import SCFResult                                    # noqa: E402
from qpc.units import Units                                      # noqa: E402

U = Units()
OUT = "results/two_e"
PAT = re.compile(r"^C_exch_wx([0-9.]+)_V([0-9.]+)$")            # Phase 1, Lx 5000, kT 0.05 only
INK, MUTED, SHADE, GRID = "#1a1a1a", "#6b6b6b", "#ececec", "#e6e6e6"
WXS, V0S = [1.0, 0.7, 0.5, 0.35, 0.25], [2.5, 3.0, 3.5]
LDOS_CASES = [(1.0, 3.0), (0.5, 3.0), (0.35, 3.0)]


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def style(a):
    a.grid(True, color=GRID, lw=0.6)
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(colors=MUTED, labelsize=7)


def cases():
    out = {}
    for d in sorted(glob.glob(os.path.join(OUT, "*/"))):
        m = PAT.match(os.path.basename(d.rstrip("/")))
        a = load(d + "analysis.json")
        if not (m and a):
            continue
        z = np.load(d + "analysis.npz")
        un = load(d + "unpol.json") or {}
        a["dir"] = d
        a["omega_ok"] = bool(a["converged"] and un.get("converged", False))
        pos, prom, sp = density_peaks(z["x_nm"], z["n_x"], z["D_mask"])
        wpos, _, _ = density_peaks(z["x_nm"], z["n_x"], z["D_mask"], prominence=N_PROMINENCE_WEAK_NM)
        a.update(npk=pos, npk_prom=prom, npk_sp=sp, npk_weak=wpos)
        out[(float(m.group(1)), float(m.group(2)))] = (a, z)
    return out


def f(x, fmt="{:.3f}"):
    try:
        return fmt.format(x)
    except Exception:
        return "-"


def fit(xs, ys):
    if len(xs) < 2:
        return float("nan"), float("nan"), float("nan")
    (k, c), res, *_ = np.polyfit(xs, ys, 1, full=True)
    r = np.corrcoef(xs, ys)[0, 1] if len(xs) > 2 else float("nan")
    return float(k), float(c), float(r)


def fig_profiles(C):
    fig, axs = plt.subplots(len(WXS), len(V0S), figsize=(10.5, 12), sharey=True, squeeze=False,
                            gridspec_kw=dict(hspace=0.45, wspace=0.12))
    for r, wx in enumerate(WXS):
        for c, V0 in enumerate(V0S):
            a = axs[r, c]
            style(a)
            if (wx, V0) not in C:
                a.set_visible(False)
                continue
            j, z = C[(wx, V0)]
            x, n, s, D = z["x_nm"], 100 * z["n_x"], 100 * z["s_x"], z["D_mask"]
            lo, hi = j["D_edges"]
            a.axvspan(lo, hi, color=SHADE, lw=0, zorder=0)
            a.plot(x, s, color=INK, lw=1.6, label=r"$s(x)=n_\uparrow-n_\downarrow$")
            a.plot(x, 0.5 * n, color=INK, lw=1.2, ls="--", label=r"$n(x)/2$")
            for xp in j["npk"]:
                a.plot([xp], [0.5 * np.interp(xp, x, n)], marker="v", ms=5, color=INK, ls="none")
            a.axhline(0, color=MUTED, lw=0.6)
            half = max(400.0, 0.5 * j["L_D"] + 250)
            a.set_xlim(-half, half)
            a.set_ylim(-1.0, 2.6)
            tag = "" if j["converged"] else "  NOT CONV."
            a.set_title(f"$\\hbar\\omega_x$={wx:g}, $V_0$={V0:g}: net {j['net_spin']:.2f}, "
                        f"{len(j['npk'])} n-peaks{tag}", fontsize=7.5, color=INK)
            if r == len(WXS) - 1:
                a.set_xlabel("x (nm)", fontsize=8)
            if c == 0:
                a.set_ylabel(r"$10^{-2}$ nm$^{-1}$", fontsize=8)
    axs[0, 0].legend(loc="upper left", frameon=False, fontsize=6.5)
    fig.suptitle("Phase 1, B = 0 (model C, exchange): grey = depleted region D, triangles = n(x) peaks in D",
                 fontsize=9, color=INK, y=0.915)
    fig.savefig("figures/two_e/fig1_profiles.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_netspin(C, fits):
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.4), gridspec_kw=dict(wspace=0.28))
    for a, key, lab, (k, c, r) in ((axs[0], "L_D", r"$L_D$ (nm)", fits["L_D"]),
                                   (axs[1], "N_D", r"$N_D$ (electrons in D)", fits["N_D"])):
        style(a)
        for conv, mfc, name in ((True, INK, "converged"), (False, "white", "not converged")):
            pts = [(j[key], j["net_spin"]) for j, _ in C.values() if j["converged"] == conv]
            if pts:
                P = np.array(pts)
                a.plot(P[:, 0], P[:, 1], "o", ms=7, mfc=mfc, mec=INK, mew=1.4, ls="none", label=name)
        if np.isfinite(k):
            xs = np.linspace(min(j[key] for j, _ in C.values()), max(j[key] for j, _ in C.values()), 50)
            a.plot(xs, k * xs + c, color=MUTED, lw=1.2, ls="--", label=f"fit (converged): slope {k:.4f}")
        a.set_xlabel(lab, fontsize=8)
        a.set_ylabel("net spin", fontsize=8)
        a.legend(frameon=False, fontsize=7)
    fig.suptitle("Net spin at B = 0 vs the size of the depleted region", fontsize=9, color=INK)
    fig.savefig("figures/two_e/fig2_netspin.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def ldos_x0(C):
    out = {}
    for wx, V0 in LDOS_CASES:
        if (wx, V0) not in C:
            continue
        j, _ = C[(wx, V0)]
        res = SCFResult.load_npz(j["dir"] + "B0.npz")
        g = grid_from_cutoff(U.nm_to_au(j["lx"]), U.nm_to_au(320.0), U.meV_to_au(15.0))
        ham = Hamiltonian(g, U.meV_to_au(15.0))
        mu = j["mu_meV"]
        E, rhos, peaks = ldos_at(res, ham, [0.0], mu, j["eps0_far"] - 0.3, eta_meV=0.05, units=U)
        up, dn = peaks[0.0]
        out[(wx, V0)] = dict(E=E - mu, rho=rhos[0.0], up_below=[e for e in up if e < 0], dn_above=[e for e in dn if e > 0],
                             up_all=up, dn_all=dn, converged=j["converged"])
    return out


def fig_ldos(L):
    if not L:
        return
    fig, axs = plt.subplots(1, len(L), figsize=(3.4 * len(L), 3.2), sharey=False, squeeze=False,
                            gridspec_kw=dict(wspace=0.3))
    for a, ((wx, V0), d) in zip(axs[0], L.items()):
        style(a)
        a.plot(d["E"], d["rho"][0], color=INK, lw=1.4, label="spin up")
        a.plot(d["E"], d["rho"][1], color=INK, lw=1.2, ls="--", label="spin down")
        a.axvline(0, color=MUTED, lw=0.8, ls=":")
        a.text(0.01, 0.97, r"$\mu$", transform=a.get_xaxis_transform(), fontsize=8, color=MUTED, va="top")
        a.set_xlabel(r"$\epsilon-\mu$ (meV)", fontsize=8)
        a.set_title(f"$\\hbar\\omega_x$={wx:g}, $V_0$={V0:g}: {len(d['up_below'])} up < $\\mu$, "
                    f"{len(d['dn_above'])} dn > $\\mu$" + ("" if d["converged"] else " NOT CONV."), fontsize=7.5)
        a.set_ylim(0, None)
    axs[0, 0].set_ylabel(r"LDOS at $x=0$ (1/(meV nm))", fontsize=8)
    axs[0, 0].legend(frameon=False, fontsize=7)
    fig.savefig("figures/two_e/fig3_ldos_x0.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs("figures/two_e", exist_ok=True)
    C = cases()
    conv = [j for j, _ in C.values() if j["converged"]]
    fits = {k: fit([j[k] for j in conv], [j["net_spin"] for j in conv]) for k in ("L_D", "N_D")}
    inv_nc = np.mean([1.0 / j["n_center"] for j in conv]) if conv else float("nan")
    spac = [s for j in conv for s in j["npk_sp"]]
    fig_profiles(C)
    fig_netspin(C, fits)
    L = ldos_x0(C)
    fig_ldos(L)

    o = [f"# Two-electron scan: Phase 1 analysis ({time.strftime('%Y-%m-%d %H:%M')})", "",
         "Existing B = 0 states (Lx 5000 nm, kT 0.05 meV); no new SCF. NOT CONV. = maxiter (300) reached: shown, "
         "never fitted, and its dOmega is not used. n-peaks: maxima of n(x) inside D with prominence >= 0.1e-2 nm^-1 "
         "(weak: >= 0.05e-2). Figures: figures/two_e/fig1_profiles.png, fig2_netspin.png, fig3_ldos_x0.png.", "",
         "## 1. Peaks of n(x) inside D", "",
         "| wx | V0 | conv | net spin | L_D | N_D | 1/n_center | n-peaks (x, nm) | spacings | weak peaks | "
         "dOmega pol-unpol |", "|" + "---|" * 11]
    for (wx, V0), (j, _) in sorted(C.items(), key=lambda t: (-t[0][0], t[0][1])):
        dom = f(j["dOmega_pol_unpol_meV"], "{:+.5f}") if j["omega_ok"] else "excl. (not conv.)"
        o.append(f"| {wx:g} | {V0:g} | {'y' if j['converged'] else 'NO'} | {f(j['net_spin'], '{:.2f}')} | "
                 f"{f(j['L_D'], '{:.0f}')} | {f(j['N_D'], '{:.2f}')} | {f(1 / j['n_center'], '{:.0f}')} | "
                 f"{' '.join(f'{p:+.0f}' for p in j['npk']) or '-'} | {' '.join(f'{p:.0f}' for p in j['npk_sp']) or '-'} | "
                 f"{len(j['npk_weak'])} | {dom} |")
    o += ["", f"Mean n-peak spacing (converged): {f(np.mean(spac) if spac else float('nan'), '{:.0f}')} nm; "
          f"mean 1/n_center (converged): {f(inv_nc, '{:.0f}')} nm.", "",
          "## 2. Net spin vs L_D and N_D (linear fit on the converged states)", ""]
    for k, unit in (("L_D", "nm"), ("N_D", "electron in D")):
        sl, c, r = fits[k]
        o.append(f"- net spin vs {k}: slope {f(sl, '{:.5f}')} per {unit}, intercept {f(c, '{:+.2f}')}, r = {f(r)}"
                 + (f"; i.e. one spin per {f(1 / sl, '{:.0f}')} nm (compare 1/n_center = {f(inv_nc, '{:.0f}')} nm)"
                    if k == "L_D" and sl else ""))
    o.append(f"- converged states used: {len(conv)} of {len(C)}")
    o += ["", "## 3. LDOS at x = 0 (eta 0.05 meV), V0 = 3.0", "",
          "| wx | conv | spin-up peaks below mu (e - mu, meV) | spin-down peaks above mu (e - mu, meV) |", "|---|---|---|---|"]
    for (wx, V0), d in L.items():
        o.append(f"| {wx:g} | {'y' if d['converged'] else 'NO'} | {len(d['up_below'])}: "
                 f"{' '.join(f'{e:+.3f}' for e in d['up_below']) or '-'} | {len(d['dn_above'])}: "
                 f"{' '.join(f'{e:+.3f}' for e in d['dn_above']) or '-'} |")
    open("notes/TWO_E_ANALYSIS.md", "w").write("\n".join(o) + "\n")
    print("notes/TWO_E_ANALYSIS.md, figures/two_e/fig1-3 written")


if __name__ == "__main__":
    main()
