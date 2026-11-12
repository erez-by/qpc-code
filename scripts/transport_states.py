"""Landauer transmission of the saved B = 0 Kohn-Sham states (P, AP, unpolarised) of the two-electron study.

For each pair (wx, V0) in {(0.5, 3.0), (0.35, 3.0), (0.5, 3.5)} and each state P (results/two_e/<case>/B0.npz),
AP (<case>_AP/B0.npz), unpol (<case>/unpol.npz), per spin:
  * energy-zero check (mu and the edge-column transverse levels vs the clean wire);
  * eps_n,s(x), n = 0, 1 (hard walls |y| = 100 nm, dy = 0.5 nm) on |x| <= 2400 nm (1 nm);
  * 1D: lead cut X (eps_0 flat to 1e-4 meV beyond X), T_s(E) on mu - 0.6 ... mu + 0.4 meV (0.002 meV),
    psi_L, psi_R at E = mu, DOP853 cross-check, T(mu) vs X (lead-length scan), dy = 2 vs 0.5 nm;
  * 2D Kwant (a = 2 nm, |y| < 100 nm, Xc from the 1e-4 meV criterion): T at ~12 energies and the left / right
    scattering states at E = mu (y-integrated; full maps for the main case wx 0.5, V0 3.0);
  * conductance G = (e^2/h) sum_s T_s(mu) and its kT = 0.05 meV thermal average (n = 0 channel only).
Output: results/transport/<label>.json / .npz, notes/TRANSPORT_RESULTS.md, figures/transport/*.png.

Usage: python scripts/transport_states.py [--no-2d] [--figs-only] [--only LABEL]
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qpc.transport import (check_energy_zero, kwant_region, kwant_system, lead_cut, load_state,    # noqa: E402
                           mode_profile, scatter_1d, thermal_conductance, transmission_1d_ivp,
                           transmission_2d, wavefunctions_2d)

warnings.filterwarnings("ignore", message="MUMPS is not available")
PAIRS = [(0.5, 3.0), (0.35, 3.0), (0.5, 3.5)]
KINDS = ["P", "AP", "unpol"]
OUT, NOTES, FIGS = "results/transport", "notes/TRANSPORT_RESULTS.md", "figures/transport"
KT = 0.05
X_SCAN = [400, 600, 800, 1000, 1500, 2000, 2400]
MAIN = (0.5, 3.0)


def case_dir(wx, V0, ap=False):
    return os.path.join("results/two_e", f"C_exch_wx{float(wx)}_V{float(V0)}" + ("_AP" if ap else ""), "")


def label(wx, V0, kind):
    return f"wx{wx:g}_V{V0:g}_{kind}"


def jdump(path, d):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))
    with open(path + ".tmp", "w") as fh:
        json.dump(d, fh, indent=1, default=conv)
    os.replace(path + ".tmp", path)


def ldos_peaks(analysis_json):
    """All LDOS peak energies (e - mu, meV) of the state's analysis (lobe positions), per spin."""
    a = json.load(open(analysis_json))
    up = sorted({round(p, 4) for l in a.get("ldos", []) for p in l["peaks_up_minus_mu"]})
    dn = sorted({round(p, 4) for l in a.get("ldos", []) for p in l["peaks_dn_minus_mu"]})
    return up, dn, a


def features(E, T, mu):
    """Peaks (resonances) and dips of T(E) with prominence 0.02, as e - mu [meV]."""
    pk, _ = find_peaks(T, prominence=0.02)
    dp, _ = find_peaks(-T, prominence=0.02)
    return [float(E[i] - mu) for i in pk], [float(E[i] - mu) for i in dp]


def run_state(wx, V0, kind, do2d=True):
    lab = label(wx, V0, kind)
    jpath = os.path.join(OUT, lab + ".json")
    if os.path.exists(jpath) and json.load(open(jpath)).get("finished") and (not do2d or json.load(open(jpath)).get("has_2d")):
        print(f"[{lab}] done earlier")
        return
    t0 = time.time()
    P = case_dir(wx, V0)
    src = {"P": P + "B0.npz", "AP": case_dir(wx, V0, True) + "B0.npz", "unpol": P + "unpol.npz"}[kind]
    ajson = (case_dir(wx, V0, True) if kind == "AP" else P) + "analysis.json"
    st = load_state(src, wx, V0, label=lab)
    ez = check_energy_zero(st, P + "wire_B0.json", ajson)
    mu = st.mu_meV
    wire = json.load(open(P + "wire_B0.json"))
    e1_lead = wire["subbands_up_meV"][1] - mu
    ld_up, ld_dn, ana = ldos_peaks(ajson) if kind != "unpol" else ([], [], {})
    spins = [0] if kind == "unpol" else [0, 1]
    x = np.arange(-2400.0, 2400.01, 1.0)
    E = np.round(np.arange(mu - 0.6, mu + 0.4 + 1e-9, 0.002), 9)
    info = dict(label=lab, kind=kind, hbar_wx=wx, V0=V0, source=src, converged=st.converged, mu_meV=mu,
                energy_zero=ez, e1_lead_minus_mu=e1_lead, kT=KT, spins={})
    arr = dict(E=E, x=x)
    eps_all, T_all, pL_all, pR_all = [], [], [], []
    for s in spins:
        eps = mode_profile(st, s, x)                                        # (nx, 2)
        X, res = lead_cut(x, eps[:, 0])
        sel = np.abs(x) <= X
        o = scatter_1d(x[sel], eps[sel, 0], E)
        om = scatter_1d(x[sel], eps[sel, 0], mu, want_psi=True)
        Tmu = float(om["T"][0])
        ivp = transmission_1d_ivp(x[sel], eps[sel, 0], mu)[0]
        scan = {}
        for Xs in X_SCAN:
            ss = np.abs(x) <= Xs
            scan[Xs] = float(scatter_1d(x[ss], eps[ss, 0], mu)["T"][0])
        e2 = mode_profile(st, s, x[sel], dy=2.0, n_modes=1)[:, 0]
        T_dy2 = float(scatter_1d(x[sel], e2, mu)["T"][0])
        pk, dp = features(E, o["T"], mu)
        ldp = ld_up if s == 0 else ld_dn
        match = [dict(T_peak=p, ldos_peak=min(ldp, key=lambda q: abs(q - p)) if ldp else None) for p in pk]
        psiL = np.full(x.size, np.nan + 0j)
        psiR = np.full(x.size, np.nan + 0j)
        psiL[sel], psiR[sel] = om["psi_L"][0][4:-4], om["psi_R"][0][4:-4]
        info["spins"][s] = dict(X=X, tail_residual=res, e_L=float(eps[sel, 0][0]), e_R=float(eps[sel, 0][-1]),
                                T_mu=Tmu, T_mu_ivp=ivp, flux_err=float(np.abs(o["T"] + o["R"] - 1).max()),
                                recip_err=float(np.abs(o["T"] - o["T_rev"]).max()), T_mu_dy2=T_dy2,
                                X_scan=scan, barrier_top_minus_mu=float(eps[:, 0].max() - mu),
                                e1_min_minus_mu=float(eps[:, 1].min() - mu), T_peaks=pk, T_dips=dp,
                                peaks_vs_ldos=match, r_mu=[float(om["r"][0].real), float(om["r"][0].imag)])
        eps_all.append(eps.T)
        T_all.append(o["T"])
        pL_all.append(psiL)
        pR_all.append(psiR)
    if kind == "unpol":
        info["spins"][1] = info["spins"][0]
        eps_all, T_all, pL_all, pR_all = eps_all * 2, T_all * 2, pL_all * 2, pR_all * 2
    arr.update(eps=np.array(eps_all), T1d=np.array(T_all), psi_L=np.array(pL_all), psi_R=np.array(pR_all))
    info["G"] = thermal_conductance(E, arr["T1d"][0], arr["T1d"][1], mu, KT)
    if kind == "AP":
        info["AP_mirror_maxdiff"] = float(np.abs(arr["T1d"][0] - arr["T1d"][1]).max())
    if do2d:
        info["has_2d"] = True
        extra = sorted({round(p, 3) for s in spins for p in info["spins"][s]["T_peaks"] + info["spins"][s]["T_dips"]
                        if abs(p) < 0.3})[:4]
        E2 = np.array(sorted({0.0, -0.02, 0.02, -0.05, 0.05, -0.1, 0.1, 0.15, 0.25} | set(extra))) + mu
        T2, rhoL, rhoR, maps = [], [], [], {}
        two = {}
        for s in spins:
            V, xs, ys, Xc, res2 = kwant_region(st, s, a=2.0, W=100.0)
            f = kwant_system(V, 2.0)
            Ts = [transmission_2d(f, e) for e in E2]
            T2.append([d["T"] for d in Ts])
            wf = wavefunctions_2d(f, mu, xs.size, ys.size)
            rL = (np.abs(wf["psi_L"]) ** 2).sum(axis=(0, 2))
            rR = (np.abs(wf["psi_R"]) ** 2).sum(axis=(0, 2))
            rhoL.append(np.interp(x, xs, rL, left=np.nan, right=np.nan))
            rhoR.append(np.interp(x, xs, rR, left=np.nan, right=np.nan))
            if (wx, V0) == MAIN:
                win = np.abs(xs) <= 700
                maps[f"map_L_s{s}"] = (np.abs(wf["psi_L"][0]) ** 2)[win]
                maps[f"map_R_s{s}"] = (np.abs(wf["psi_R"][0]) ** 2)[win]
                maps[f"map_V_s{s}"] = V[win]
                maps["map_x"], maps["map_y"] = xs[win], ys
            e0a = mode_profile(st, s, xs, dy=2.0, n_modes=1)[:, 0]          # 1D at dy = a for the comparison
            T1a = float(scatter_1d(xs, e0a, mu)["T"][0])
            two[s] = dict(Xc=Xc, W=100.0, a=2.0, tail_residual=res2, sites=int(V.size),
                          T_mu=float(Ts[list(E2).index(mu)]["T"]), n_open=[d["n_open"] for d in Ts],
                          T_mu_1d_same_dy=T1a, flux_ok=True)
        if kind == "unpol":
            two[1] = two[0]
            T2, rhoL, rhoR = T2 * 2, rhoL * 2, rhoR * 2
        info["kwant"] = two
        info["E2_minus_mu"] = (E2 - mu).tolist()
        arr.update(E2=E2, T2d=np.array(T2), rho2d_L=np.array(rhoL), rho2d_R=np.array(rhoR), **maps)
    info["finished"] = True
    info["time_s"] = time.time() - t0
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, lab + ".npz"), **arr)
    jdump(jpath, info)
    s0, s1 = info["spins"][0], info["spins"][1]
    print(f"[{lab}] T_up(mu) {s0['T_mu']:.5f} T_dn(mu) {s1['T_mu']:.3e} | G T=0 {info['G']['G_e2h_T0']:.4f} "
          f"kT {info['G']['G_e2h_kT']:.4f} e^2/h"
          + (f" | 2D {info['kwant'][0]['T_mu']:.5f} / {info['kwant'][1]['T_mu']:.3e}" if do2d else "")
          + f" ({info['time_s']:.0f} s)", flush=True)


# ----------------------------------------------------------------------------- notes + figures
def load_all():
    out = {}
    for wx, V0 in PAIRS:
        for k in KINDS:
            lab = label(wx, V0, k)
            p = os.path.join(OUT, lab)
            if os.path.exists(p + ".json"):
                out[(wx, V0, k)] = (json.load(open(p + ".json")), np.load(p + ".npz"))
    return out


def f(x, fmt="{:.4f}"):
    try:
        return fmt.format(x)
    except Exception:
        return "-"


def notes(D):
    o = [f"# Transmission of the saved B = 0 Kohn-Sham states ({time.strftime('%Y-%m-%d %H:%M')})", "",
         "Landauer transmission of the frozen self-consistent potential V_eff,sigma(x, y) (model C, exchange, "
         "Lx 5000 nm, kT 0.05 meV). 1D: lowest transverse mode eps_0,sigma(x) (hard walls |y| = 100 nm, dy 0.5 nm, "
         "Numerov, leads frozen beyond X where eps_0 is flat to 1e-4 meV). 2D: Kwant, a = 2 nm, full V_eff. "
         "G in units of e^2/h (one spin channel = 1; 2e^2/h = both spins open). Thermal G: kT = 0.05 meV, n = 0 "
         "channel only. Code: qpc/transport.py, scripts/transport_states.py.", "",
         "## Transmission at mu and conductance", "",
         "| pair | state | conv | T_up(mu) 1D | T_dn(mu) 1D | T_up 2D | T_dn 2D | G(T=0) [e^2/h] | G(kT) [e^2/h] | "
         "barrier top - mu (up / dn) [meV] |", "|" + "---|" * 10]
    for (wx, V0, k), (j, _) in D.items():
        s0, s1 = j["spins"]["0"], j["spins"]["1"]
        kw = j.get("kwant", {})
        o.append(f"| wx {wx:g}, V0 {V0:g} | {k} | {'y' if j['converged'] else 'NO'} | {f(s0['T_mu'])} | "
                 f"{f(s1['T_mu'], '{:.2e}')} | {f(kw.get('0', {}).get('T_mu'))} | {f(kw.get('1', {}).get('T_mu'), '{:.2e}')} | "
                 f"{f(j['G']['G_e2h_T0'], '{:.3f}')} | {f(j['G']['G_e2h_kT'], '{:.3f}')} | "
                 f"{f(s0['barrier_top_minus_mu'], '{:+.3f}')} / {f(s1['barrier_top_minus_mu'], '{:+.3f}')} |")
    o += ["", "## Checks", "",
          "| state | spin | X [nm] | flux \\|T+R-1\\| | \\|T_LR - T_RL\\| | T(mu) Numerov vs DOP853 | 1D dy 2 nm | "
          "1D at dy = a vs 2D | Xc [nm] | 2D sites |", "|" + "---|" * 10]
    for (wx, V0, k), (j, _) in D.items():
        for s in ("0", "1") if k != "unpol" else ("0",):
            sp, kw = j["spins"][s], j.get("kwant", {}).get(s, {})
            o.append(f"| {label(wx, V0, k)} | {'up' if s == '0' else 'dn'} | {sp['X']:.0f} | {sp['flux_err']:.1e} | "
                     f"{sp['recip_err']:.1e} | {abs(sp['T_mu'] - sp['T_mu_ivp']):.1e} | {f(sp['T_mu_dy2'])} | "
                     f"{f(kw.get('T_mu_1d_same_dy'))} vs {f(kw.get('T_mu'))} | {f(kw.get('Xc'), '{:.0f}')} | "
                     f"{kw.get('sites', '-')} |")
    o += ["", "## Lead-length scan: T(mu) when the scatterer is cut at |x| = X (Friedel tail beyond X dropped)", "",
          "| state | spin | " + " | ".join(f"X={X}" for X in X_SCAN) + " |", "|" + "---|" * (len(X_SCAN) + 2)]
    for (wx, V0, k), (j, _) in D.items():
        for s in ("0", "1") if k != "unpol" else ("0",):
            sc = j["spins"][s]["X_scan"]
            o.append(f"| {label(wx, V0, k)} | {'up' if s == '0' else 'dn'} | "
                     + " | ".join(f(sc[str(X)], "{:.4g}") for X in X_SCAN) + " |")
    o += ["", "## Resonances of T(E) vs the LDOS quasi-bound peaks (e - mu, meV)", "",
          "| state | spin | T peaks | nearest LDOS peak | T dips |", "|---|---|---|---|---|"]
    for (wx, V0, k), (j, _) in D.items():
        if k == "unpol":
            continue
        for s in ("0", "1"):
            sp = j["spins"][s]
            o.append(f"| {label(wx, V0, k)} | {'up' if s == '0' else 'dn'} | "
                     f"{' '.join(f'{p:+.3f}' for p in sp['T_peaks']) or '-'} | "
                     f"{' '.join(f(m['ldos_peak'], '{:+.3f}') for m in sp['peaks_vs_ldos']) or '-'} | "
                     f"{' '.join(f'{p:+.3f}' for p in sp['T_dips']) or '-'} |")
    o += ["", "Notes: the second lead subband opens at e1(lead) - mu = "
          f"{f(next(iter(D.values()))[0]['e1_lead_minus_mu'], '{:+.3f}')} meV, inside the thermal window; the "
          "thermal G uses the n = 0 channel only. T is the transmission of the static LSDA (mean-field) potential, "
          "not a many-body conductance.", ""]
    os.makedirs("notes", exist_ok=True)
    open(NOTES, "w").write("\n".join(o))


INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e6e6e6"
UP, DN = "#1f5fbf", "#e8710a"


def style(a):
    a.grid(True, color=GRID, lw=0.6)
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(colors=MUTED, labelsize=7.5)


def figures(D):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIGS, exist_ok=True)
    for log in (False, True):
        fig, axs = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=True, squeeze=False,
                                gridspec_kw=dict(hspace=0.3, wspace=0.08))
        for c, (wx, V0) in enumerate(PAIRS):
            for r, k in enumerate(KINDS):
                a = axs[r, c]
                style(a)
                if (wx, V0, k) not in D:
                    continue
                j, z = D[(wx, V0, k)]
                mu = j["mu_meV"]
                Er = z["E"] - mu
                a.axvspan(-4 * KT, 4 * KT, color="#f2f2f2", lw=0, zorder=0)
                a.plot(Er, z["T1d"][0], color=UP, lw=1.6, label=r"$T_\uparrow$ (1D)")
                a.plot(Er, z["T1d"][1], color=DN, lw=1.4, ls="--", label=r"$T_\downarrow$ (1D)")
                if "T2d" in z:
                    a.plot(z["E2"] - mu, z["T2d"][0], "o", ms=4.5, mfc="white", mec=UP, mew=1.2, ls="none",
                           label=r"$T_\uparrow$ (2D Kwant)")
                    a.plot(z["E2"] - mu, z["T2d"][1], "s", ms=4, mfc="white", mec=DN, mew=1.2, ls="none",
                           label=r"$T_\downarrow$ (2D Kwant)")
                a.axvline(0, color=INK, lw=0.9, ls=":")
                a.axvline(j["e1_lead_minus_mu"], color=MUTED, lw=0.8, ls="-.")
                if log:
                    a.set_yscale("log")
                    a.set_ylim(1e-7, 3)
                else:
                    a.set_ylim(-0.03, 1.08)
                a.set_title(f"$\\hbar\\omega_x$={wx:g}, $V_0$={V0:g}: {k}, G = {j['G']['G_e2h_T0']:.2f} $e^2/h$",
                            fontsize=8.5, color=INK, loc="left")
                if r == 2:
                    a.set_xlabel(r"$E-\mu$ (meV)", fontsize=8.5)
                if c == 0:
                    a.set_ylabel("transmission", fontsize=8.5)
        axs[0, 0].legend(frameon=False, fontsize=7, loc="upper left" if not log else "lower right")
        fig.suptitle("Spin-resolved transmission of the B = 0 KS potentials (dotted: $\\mu$; dash-dot: 2nd lead "
                     "subband opens; grey: $\\pm 4kT$)", fontsize=10, color=INK, y=0.95)
        fig.savefig(os.path.join(FIGS, "fig_T_E_log.png" if log else "fig_T_E.png"), dpi=170, bbox_inches="tight")
        plt.close(fig)

    for wx, V0 in PAIRS:
        fig, axs = plt.subplots(3, 2, figsize=(11, 8.6), sharex=True, squeeze=False, gridspec_kw=dict(hspace=0.35,
                                                                                                        wspace=0.32))
        a_json = json.load(open(case_dir(wx, V0) + "analysis.json"))
        half = 0.5 * a_json["L_D"] + 400
        for r, k in enumerate(KINDS):
            if (wx, V0, k) not in D:
                continue
            j, z = D[(wx, V0, k)]
            mu = j["mu_meV"]
            for s, col in ((0, UP), (1, DN)):
                a = axs[r, s]
                style(a)
                x = z["x"]
                a.plot(x, np.abs(z["psi_L"][s]) ** 2, color=col, lw=1.7, label=r"$|\psi_L|^2$ (from left)")
                a.plot(x, np.abs(z["psi_R"][s]) ** 2, color=col, lw=1.0, alpha=0.55, label=r"$|\psi_R|^2$ (from right)")
                if "rho2d_L" in z:
                    a.plot(x, z["rho2d_L"][s], color=INK, lw=0.8, ls=":", label=r"2D $\int|\psi_L|^2dy$")
                a.set_xlim(-half, half)
                a.set_ylabel(r"$|\psi|^2$ (incident amplitude 1)", fontsize=8)
                b = a.twinx()
                b.plot(x, z["eps"][s][0] - mu, color=INK, lw=1.0, alpha=0.8)
                b.axhline(0, color=INK, lw=0.7, ls=":")
                b.set_ylabel(r"$\epsilon_{0,\sigma}(x)-\mu$ (meV)", fontsize=8, color=MUTED)
                b.tick_params(colors=MUTED, labelsize=7)
                b.spines[["top"]].set_visible(False)
                sp = j["spins"][str(s)]
                a.set_title(f"{k}, spin {'up' if s == 0 else 'down'}: T($\\mu$) = {sp['T_mu']:.3g}", fontsize=8.5,
                            color=INK, loc="left")
                if r == 2:
                    a.set_xlabel("x (nm)", fontsize=8.5)
                if r == 0 and s == 0:
                    fig.legend(*a.get_legend_handles_labels(), frameon=False, fontsize=8, ncol=3,
                               loc="upper center", bbox_to_anchor=(0.5, 0.935))
        fig.suptitle(f"Scattering states at E = $\\mu$, $\\hbar\\omega_x$ = {wx:g} meV, $V_0$ = {V0:g} meV "
                     "(black line, right axis: effective barrier)", fontsize=10, color=INK, y=0.975)
        fig.savefig(os.path.join(FIGS, f"fig_psi_wx{wx:g}_V{V0:g}.png"), dpi=170, bbox_inches="tight")
        plt.close(fig)

    if all((MAIN[0], MAIN[1], k) in D and "map_x" in D[(MAIN[0], MAIN[1], k)][1] for k in ("P", "AP")):
        fig, axs = plt.subplots(2, 4, figsize=(14, 4.6), sharex=True, sharey=True, squeeze=False,
                                gridspec_kw=dict(hspace=0.35, wspace=0.08))
        for r, k in enumerate(("P", "AP")):
            j, z = D[(MAIN[0], MAIN[1], k)]
            xs, ys = z["map_x"], z["map_y"]
            vmax = max(np.nanmax(z[f"map_{d}_s{s}"]) for d in ("L", "R") for s in (0, 1))
            for c, (d, s) in enumerate((("L", 0), ("R", 0), ("L", 1), ("R", 1))):
                a = axs[r, c]
                im = a.imshow(z[f"map_{d}_s{s}"].T, origin="lower", aspect="auto", cmap="Greys", vmin=0, vmax=vmax,
                              extent=[xs[0], xs[-1], ys[0], ys[-1]])
                a.contour(xs, ys, (z[f"map_V_s{s}"] - j["mu_meV"]).T, levels=[0.0], colors=[UP if s == 0 else DN],
                          linewidths=1.0)
                a.set_title(f"{k}: spin {'up' if s == 0 else 'down'}, from {'left' if d == 'L' else 'right'}, "
                            f"T = {j['kwant'][str(s)]['T_mu']:.3g}", fontsize=8, color=INK, loc="left")
                a.tick_params(colors=MUTED, labelsize=7)
                if r == 1:
                    a.set_xlabel("x (nm)", fontsize=8)
                if c == 0:
                    a.set_ylabel("y (nm)", fontsize=8)
            fig.colorbar(im, ax=axs[r, :].tolist(), fraction=0.015, pad=0.01).set_label(r"$|\psi|^2$", fontsize=8)
        fig.suptitle(f"2D scattering states at E = $\\mu$ (Kwant), $\\hbar\\omega_x$ = {MAIN[0]:g} meV, $V_0$ = "
                     f"{MAIN[1]:g} meV; coloured line: $V_{{eff,\\sigma}} = \\mu$", fontsize=10, color=INK, y=1.02)
        fig.savefig(os.path.join(FIGS, "fig_maps_wx0.5_V3.png"), dpi=170, bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-2d", action="store_true")
    ap.add_argument("--figs-only", action="store_true")
    ap.add_argument("--only", default=None, help="one label, e.g. wx0.5_V3_P")
    a = ap.parse_args()
    if not a.figs_only:
        for wx, V0 in PAIRS:
            for k in KINDS:
                if a.only and label(wx, V0, k) != a.only:
                    continue
                run_state(wx, V0, k, do2d=not a.no_2d)
    D = load_all()
    notes(D)
    figures(D)
    print(f"{NOTES}, {FIGS}/ ({len(D)} states)")


if __name__ == "__main__":
    main()
