"""PRL-style figures from results/ only (works with whatever exists). Untagged folders only, unless
--tags T1,T2 is given: then only folders with those tags, files named ..._<tag>.png
(e.g. python scripts/make_figs.py --tags lx10000_kT0.02 -> fig1_C_exch_lx10000_kT0.02.png).

fig1_<model>_<interp>.png  PRL Fig. 1: one column per available wx (B = 0 states). Top: e_0,s(x) - e_0(far)
    (solid up, dashed down), x in [-380, 380] nm, -0.3 ... 1.6 meV, mu arrow on the left; LDOS(x = 0)
    side inset sharing the energy axis (eta = 0.05). Bottom: n_up - n_dn (solid), (n_up + n_dn)/2
    (dashed), 1e-2 nm^-1, -0.3 ... 1.8, no clipping.
fig2_<model>_<interp>_wx1.5.png  PRL Fig. 2: LDOS(x = 0) vs e - e_0(far) (0 ... 2 meV) for every
    available B, offset vertically (B = 0 at the bottom), up solid, down dashed, mu marked, eta = 0.05.
fig3_<model>_<interp>.png  PRL Fig. 3: U and Gamma vs hbar w_x (error bar = |eta 0.05 - eta 0.1|).
"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qpc.analysis import barrier_profile, n1d                    # noqa: E402
from qpc.fourier import min_image                                # noqa: E402
from qpc.grid import grid_from_cutoff                            # noqa: E402
from qpc.hamiltonian import Hamiltonian                          # noqa: E402
from qpc.scf import SCFResult                                    # noqa: E402
from qpc.units import Units                                      # noqa: E402

INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e6e6e6"
U = Units()
_HAM = {}
PAT = re.compile(r"^([ABC])_([a-z0-9.]+)_wx([0-9]+\.[0-9]+)(?:_(.+))?$")


def ham_for(ecut, lx):
    k = (ecut, lx)
    if k not in _HAM:
        g = grid_from_cutoff(U.nm_to_au(lx), U.nm_to_au(320.0), U.meV_to_au(ecut))
        _HAM[k] = Hamiltonian(g, U.meV_to_au(ecut))
    return _HAM[k]


def style(a):
    a.grid(True, color=GRID, lw=0.6)
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(colors=MUTED)


def folders(tags=None):
    """{(model, interp, tag): [(wx, dir)]}; tag None = untagged folders (the default set)."""
    out = {}
    for d in sorted(glob.glob("results/*/")):
        m = PAT.match(os.path.basename(d.rstrip("/")))
        if m and (m.group(4) is None if tags is None else m.group(4) in tags):
            out.setdefault((m.group(1), m.group(2), m.group(4)), []).append((float(m.group(3)), d))
    return out


def sfx(tag):
    return f"_{tag}" if tag else ""


def fig1(model, interp, items, tag=None):
    cols = [(wx, d) for wx, d in sorted(items) if os.path.exists(d + "B0.json") and os.path.exists(d + "B0.npz")]
    if not cols:
        return
    fig, axs = plt.subplots(2, len(cols), figsize=(3.6 * len(cols), 6.2), squeeze=False,
                            gridspec_kw=dict(hspace=0.12, wspace=0.3))
    for c, (wx, d) in enumerate(cols):
        j = json.load(open(d + "B0.json"))
        r = SCFResult.load_npz(d + "B0.npz")
        z = np.load(d + "B0.npz")
        ham = ham_for(j["ecut"], j["lx"])
        g = ham.grid
        x = U.au_to_nm(min_image(g.real_axes()[0], g.Lx))
        o = np.argsort(x)
        x = x[o]
        e0 = U.au_to_meV(barrier_profile(r, ham))[o]
        far = np.abs(x) > 0.3 * j["lx"]
        h = e0 - e0[far].mean(axis=0)[None, :]
        a1, a2 = axs[0, c], axs[1, c]
        a1.plot(x, h[:, 0], color=INK, lw=2, label="spin up")
        a1.plot(x, h[:, 1], color=INK, lw=2, ls="--", label="spin down")
        mu = j["mu_e0far"]
        a1.annotate("", xy=(-380, mu), xytext=(-330, mu), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
        a1.text(-325, mu, r" $\mu$", va="center", color=INK)
        a1.set_xlim(-380, 380)
        a1.set_ylim(-0.3, 1.6)
        a1.tick_params(labelbottom=False)
        a1.set_title(f"$\\hbar\\omega_x$ = {wx:g} meV: net spin {j['M_loc']:.2f}"
                     + ("" if j["converged"] else " (not conv.)"), fontsize=9, color=INK)
        if "ldos_up_eta0.05" in z:
            ins = a1.inset_axes([0.80, 0.0, 0.20, 1.0], sharey=a1)
            Er = z["ldos_e_minus_e0far"]
            ins.plot(z["ldos_up_eta0.05"], Er, color=INK, lw=1.0)
            ins.plot(z["ldos_dn_eta0.05"], Er, color=INK, lw=1.0, ls="--")
            ins.axhline(mu, color=MUTED, lw=0.7, ls=":")
            ins.set_xlim(0, None)
            ins.tick_params(labelleft=False, labelbottom=False, labeltop=True, top=True, bottom=False,
                            labelsize=6, colors=MUTED)
            ins.xaxis.set_label_position("top")
            ins.set_xlabel(r"LDOS $\nu$", fontsize=7)
            ins.patch.set_facecolor("white")
        nu = (n1d(r.n_up, g) / U.length_nm)[o] * 100
        nd = (n1d(r.n_dn, g) / U.length_nm)[o] * 100
        a2.plot(x, nu - nd, color=INK, lw=2, label=r"$n_\uparrow - n_\downarrow$")
        a2.plot(x, 0.5 * (nu + nd), color=INK, lw=2, ls="--", label=r"$(n_\uparrow + n_\downarrow)/2$")
        a2.set_xlim(-380, 380)
        lo = min(-0.3, float(np.min(nu - nd)) - 0.05)       # PRL axis, extended instead of clipping
        a2.set_ylim(lo, 1.8)
        a2.set_xlabel("x (nm)")
        for a in (a1, a2):
            style(a)
        if c == 0:
            a1.set_ylabel(r"$\epsilon_0(x) - \epsilon_0(\infty)$ (meV)")
            a2.set_ylabel(r"$n_{1D}$ ($10^{-2}$ nm$^{-1}$)")
            a1.legend(loc="upper left", frameon=False, fontsize=7)
            a2.legend(loc="upper left", ncol=2, frameon=False, fontsize=7)
    fig.suptitle(f"model {model}, interp {interp}{', ' + tag if tag else ''}, B = 0", fontsize=10, color=INK, y=1.03)
    os.makedirs("figures", exist_ok=True)
    fig.savefig(f"figures/fig1_{model}_{interp}{sfx(tag)}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig2(model, interp, d, tag=None):
    js = []
    for p in glob.glob(d + "B*.json"):
        try:
            js.append((float(os.path.basename(p)[1:-5]), p))
        except ValueError:
            pass
    js = [(B, p) for B, p in sorted(js) if os.path.exists(p[:-5] + ".npz")]
    if not js:
        return
    fig, a = plt.subplots(figsize=(4.0, 6.5))
    off, step = 0.0, None
    for B, p in js:
        z = np.load(p[:-5] + ".npz")
        if "ldos_up_eta0.05" not in z:
            continue
        j = json.load(open(p))
        E = z["ldos_e_minus_e0far"]
        up, dn = z["ldos_up_eta0.05"], z["ldos_dn_eta0.05"]
        if step is None:
            step = 0.6 * max(up.max(), dn.max())
        a.plot(E, up + off, color=INK, lw=1.2)
        a.plot(E, dn + off, color=INK, lw=1.2, ls="--")
        a.plot([j["mu_e0far"]] * 2, [off, off + 0.5 * step], color=MUTED, lw=0.8)
        a.text(2.02, off, f"{B:g} T", fontsize=7, color=MUTED, va="bottom")
        off += step
    a.set_xlim(0, 2)
    a.set_xlabel(r"$\epsilon - \epsilon_0(\infty)$ (meV)")
    a.set_ylabel(r"LDOS at $x = 0$ (offset)")
    a.set_yticks([])
    style(a)
    a.set_title(f"model {model}, interp {interp}{', ' + tag if tag else ''}, $\\hbar\\omega_x$ = 1.5 meV",
                fontsize=9, color=INK)
    fig.savefig(f"figures/fig2_{model}_{interp}_wx1.5{sfx(tag)}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig3(model, interp, items, tag=None):
    pts = []
    for wx, d in sorted(items):
        if os.path.exists(d + "B0.json"):
            L = json.load(open(d + "B0.json")).get("ldos", {})
            if "U_mean" in L:
                pts.append((wx, L["U_mean"], abs(L.get("U_diff", 0)), L["Gamma_mean"], abs(L.get("Gamma_diff", 0))))
    if not pts:
        return
    P = np.array(pts)
    fig, a = plt.subplots(figsize=(4.0, 3.2))
    a.errorbar(P[:, 0], P[:, 1], yerr=P[:, 2], color=INK, marker="o", ms=6, lw=1.5, capsize=3, label="U")
    a.errorbar(P[:, 0], P[:, 3], yerr=P[:, 4], color=INK, marker="s", ms=6, lw=1.5, ls="--", capsize=3,
               mfc="white", label=r"$\Gamma$")
    a.set_xlim(1.0, 2.0)
    a.set_ylim(0, 0.8)
    a.set_xlabel(r"$\hbar\omega_x$ (meV)")
    a.set_ylabel("energy (meV)")
    a.legend(frameon=False, fontsize=8)
    style(a)
    a.set_title(f"model {model}, interp {interp}{', ' + tag if tag else ''}, B = 0", fontsize=9, color=INK)
    fig.savefig(f"figures/fig3_{model}_{interp}{sfx(tag)}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default=None, help="comma-separated folder tags (default: untagged folders)")
    a = ap.parse_args()
    made = []
    for (model, interp, tag), items in folders(a.tags.split(",") if a.tags else None).items():
        for fn, args in ((fig1, (model, interp, items, tag)), (fig3, (model, interp, items, tag))):
            try:
                fn(*args)
            except Exception as exc:
                print(f"{fn.__name__} {model} {interp}: {exc!r}")
        for wx, d in items:
            if abs(wx - 1.5) < 1e-9:
                try:
                    fig2(model, interp, d, tag)
                except Exception as exc:
                    print(f"fig2 {model} {interp}: {exc!r}")
        made.append(f"{model}_{interp}{sfx(tag)}")
    print(f"figures for: {', '.join(made) or 'nothing yet'}")


if __name__ == "__main__":
    main()
