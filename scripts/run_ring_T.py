"""Transmission from the ring eigenstates (even/odd phase shifts, docs/RING_TRANSMISSION_SPEC.md).

  python scripts/run_ring_T.py phase0     non-interacting validation (bare HMW Eq. 1 potential, no spin):
                                          hbar w_x in {1.0, 1.5, 2.0, 0.5} meV, V0 = 3.0 meV, Lx = 5000 nm,
                                          plus wx 1.5 at Lx = 7500 nm. Compared with (i) the adiabatic
                                          Poschl-Teller formula on E - hbar w_y/2, (ii) 1D Numerov on eps_0(x)
                                          of the plane-wave CUTOFF basis (the transverse operator contained in
                                          the KS Hamiltonian) and of the full y-grid basis.
Output: results/transmission/phase0/<tag>.json/.npz, figures/transmission/fig_phase0.png, notes/RING_T_SUMMARY.md.
The old Kwant/Numerov transport results are untouched (results/transport/, tag transport-v1, archive copy in
results/archive/transport_kwant_numerov_v1/).
"""
import argparse
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qpc.analysis import transverse_levels                       # noqa: E402
from qpc.fourier import min_image                                # noqa: E402
from qpc.grid import grid_from_cutoff                            # noqa: E402
from qpc.hamiltonian import Hamiltonian                          # noqa: E402
from qpc.potential import QPCParams                              # noqa: E402
from qpc.ring_transmission import ring_transmission_2d          # noqa: E402
from qpc.scf import external_potential                           # noqa: E402
from qpc.transport import HB2_2M, scatter_1d                     # noqa: E402
from qpc.units import Units                                      # noqa: E402

U = Units()
OUT = "results/transmission"
FIGS = "figures/transmission"
NOTES = "notes/RING_T_SUMMARY.md"
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e6e6e6"
PH0 = [(1.0, 3.0, 5000.0), (1.5, 3.0, 5000.0), (2.0, 3.0, 5000.0), (0.5, 3.0, 5000.0), (1.5, 3.0, 7500.0)]


def jdump(path, d):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(d, fh, indent=1, default=conv)
    os.replace(path + ".tmp", path)


def numerov_eps0(V, ham, basis, E, xmax):
    """1D Numerov T(E) on eps_0(x) of the given transverse basis, Fourier-interpolated to 1 nm."""
    g = ham.grid
    ham.set_potential(V)
    e0 = U.au_to_meV(transverse_levels(V, ham, 1, basis=basis)[:, 0])
    F = np.fft.fft(e0) / g.Nx
    xf = np.arange(-xmax, xmax + 0.01, 1.0)
    ef = np.real(np.exp(1j * U.nm_to_au(xf)[:, None] * g.G_axes()[0][None, :]) @ F)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return scatter_1d(xf, ef, E)["T"], float(e0.max())


def pt_formula(E1d, V0, d_nm):
    k = np.sqrt(np.clip(E1d, 1e-15, None) / HB2_2M)
    q = 4 * V0 * d_nm ** 2 / HB2_2M
    s = np.sinh(np.pi * k * d_nm) ** 2
    c = np.cosh(0.5 * np.pi * np.sqrt(q - 1)) ** 2 if q > 1 else np.cos(0.5 * np.pi * np.sqrt(1 - q)) ** 2
    return s / (s + c)


def dev(T, ref, E):
    m = np.isfinite(T) & np.isfinite(ref)
    if not m.any():
        return dict(max=None)
    e = np.abs(T[m] - ref[m])
    i = int(np.argmax(e))
    return dict(max=float(e[i]), at_E=float(E[m][i]), T=float(T[m][i]), ref=float(ref[m][i]))


def phase0():
    rows = []
    for wx, V0, lx in PH0:
        tag = f"wx{wx:g}_V{V0:g}_lx{lx:g}"
        t0 = time.time()
        g = grid_from_cutoff(U.nm_to_au(lx), U.nm_to_au(320.0), U.meV_to_au(15.0))
        ham = Hamiltonian(g, U.meV_to_au(15.0))
        q = QPCParams(hbar_wx_meV=wx, V0_meV=V0)
        V = external_potential(g, q)
        E = np.round(np.arange(1.0, 5.0, 0.002), 6)
        r = ring_transmission_2d(V, ham, E)
        d_nm = U.au_to_nm(q.d)
        Tpt = pt_formula(E - r["E1"], V0, d_nm)
        T_cut, top_cut = numerov_eps0(V, ham, "cutoff", E, 0.5 * lx - 100)
        T_grid, top_grid = numerov_eps0(V, ham, "grid", E, 0.5 * lx - 100)
        Tw = np.array(list(r["T_M2_windows"].values()))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scatter = float(np.nanmax(np.nanmax(Tw, axis=0) - np.nanmin(Tw, axis=0)))
        info = dict(tag=tag, hbar_wx=wx, V0=V0, lx=lx, d_nm=d_nm, E1=r["E1"], E2=r["E2"], E3=r["E3"],
                    E_top=r["E_top"], barrier_top_cutoff=top_cut, barrier_top_grid=top_grid,
                    n_states=r["n_states"], n_kept=r["n_kept"], n_y_odd=r["n_y_odd"], dropped=r["dropped"],
                    dM1_vs_M2_max=r["dM1_vs_M2_max"],
                    dM1_vs_M2_above_bottom=float(max(np.max(np.abs(np.angle(np.exp(2j * (ph["d_M1"] - ph["d_M2"]))[ph["E"] > r["E1"] + 0.02]) / 2))
                                                     for ph in r["phases"].values())),
                    dev_M1_vs_PT=dev(r["T_M1"], Tpt, E), dev_M1_vs_numerov_cutoff=dev(r["T_M1"], T_cut, E),
                    dev_M1_vs_numerov_grid=dev(r["T_M1"], T_grid, E), dev_M1_vs_M2=dev(r["T_M1"], r["T_M2"], E),
                    M2_window_scatter=scatter, time_s=time.time() - t0)
        lev = {k: dict(E=v["E"], d_M1=v["d_M1"], d_M2=v["d_M2"]) for k, v in r["phases"].items()}
        os.makedirs(os.path.join(OUT, "phase0"), exist_ok=True)
        np.savez_compressed(os.path.join(OUT, "phase0", tag + ".npz"), E=E, T_M1=r["T_M1"], T_M2=r["T_M2"], T_PT=Tpt,
                            T_num_cut=T_cut, T_num_grid=T_grid, T_M2_windows=Tw,
                            E_e=lev["e"]["E"], E_o=lev["o"]["E"])
        jdump(os.path.join(OUT, "phase0", tag + ".json"), info)
        rows.append(info)
        print(f"[{tag}] {info['n_kept']} channel-1 states, M1 vs PT {info['dev_M1_vs_PT']['max']:.2e}, vs Numerov(cutoff) "
              f"{info['dev_M1_vs_numerov_cutoff']['max']:.2e}, vs Numerov(grid) {info['dev_M1_vs_numerov_grid']['max']:.2e}, "
              f"M1-M2 {info['dev_M1_vs_M2']['max']:.1e}, window scatter {scatter:.1e} ({info['time_s']:.0f} s)", flush=True)
    fig_phase0()
    notes()


def fig_phase0():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tags = [f"wx{wx:g}_V{V0:g}_lx{lx:g}" for wx, V0, lx in PH0 if lx == 5000.0]
    fig, axs = plt.subplots(2, len(tags), figsize=(4.0 * len(tags), 6.2), sharex="col", squeeze=False,
                            gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.08, wspace=0.28))
    for c, tag in enumerate(tags):
        z = np.load(os.path.join(OUT, "phase0", tag + ".npz"))
        j = json.load(open(os.path.join(OUT, "phase0", tag + ".json")))
        E = z["E"]
        a, b = axs[0, c], axs[1, c]
        for ax in (a, b):
            ax.grid(True, color=GRID, lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=7.5)
        a.plot(E, z["T_PT"], color=MUTED, lw=2.2, label="adiabatic Poschl-Teller (exact 1D)")
        a.plot(E, z["T_num_cut"], color=INK, lw=1.0, ls="--", label="1D Numerov, cutoff-basis $\\epsilon_0$")
        a.plot(E, z["T_M1"], color="#1f5fbf", lw=1.4, label="ring method (interpolated)")
        Ep = np.concatenate([z["E_e"], z["E_o"]])
        fin = np.isfinite(z["T_M1"])
        ok = (Ep >= E[fin].min()) & (Ep <= E[fin].max())         # only levels inside the evaluated range
        Tp = np.interp(Ep, E[fin], z["T_M1"][fin])
        a.plot(Ep[ok], Tp[ok], "o", ms=3, mfc="white", mec="#1f5fbf", mew=0.9, ls="none", label="ring levels")
        for e, lab in ((j["E2"], "E2"), (j["E3"], "E3")):
            a.axvline(e, color=MUTED, lw=0.8, ls=":")
            a.text(e, 1.02, lab, fontsize=7, color=MUTED, ha="center")
        a.set_ylim(-0.03, 1.08)
        a.set_title(f"$\\hbar\\omega_x$ = {j['hbar_wx']:g} meV, $V_0$ = {j['V0']:g} meV (d = {j['d_nm']:.0f} nm)",
                    fontsize=8.5, color=INK, loc="left")
        b.semilogy(E, np.abs(z["T_M1"] - z["T_PT"]), color=MUTED, lw=1.2, label="|ring - PT formula|")
        b.semilogy(E, np.abs(z["T_M1"] - z["T_num_cut"]), color=INK, lw=1.0, label="|ring - Numerov (cutoff)|")
        b.set_ylim(1e-7, 1e-1)
        b.set_xlabel("E (meV)", fontsize=8.5)
        if c == 0:
            a.set_ylabel("T (channel 1)", fontsize=8.5)
            b.set_ylabel("|difference|", fontsize=8.5)
            a.legend(frameon=False, fontsize=6.8, loc="upper left")
            b.legend(frameon=False, fontsize=6.8, loc="lower right")
        a.set_xlim(j["E1"], j["E_top"])
    fig.suptitle("Phase 0: ring (even/odd phase-shift) transmission of the bare HMW potential, Lx = 5000 nm",
                 fontsize=10, color=INK, y=0.97)
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, "fig_phase0.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def f(x, fmt="{:.2e}"):
    try:
        return fmt.format(x)
    except Exception:
        return "-"


def notes():
    o = [f"# Ring (even/odd phase-shift) transmission ({time.strftime('%Y-%m-%d %H:%M')})", "",
         "Method (Meir): T(E) = cos^2(d_e(E) - d_o(E)) from the x-even / x-odd channel-1 eigenstates of the periodic "
         "cell; M1: d = n pi - k(E_n) Lx/2 with the exact lead dispersion of the plane-wave Hamiltonian; M2: fit of "
         "psi_1(x) = int phi_1 psi dy to cos(k|x| + d) / sign(x) sin(k|x| + d) in 600 <= |x| <= 1500 nm (free k). "
         "Phases interpolated with PCHIP; T is not evaluated for E < E1 + 0.05 meV (band bottom, k -> 0: ill-defined "
         "phases of the first levels gave errors up to 4e-2 there). Code: qpc/ring_transmission.py, scripts/run_ring_T.py. "
         "Old Kwant/Numerov results: git tag transport-v1, results/archive/transport_kwant_numerov_v1/.", "",
         "Range: channel 1 is the only y-EVEN channel up to E3 (V is exactly mirror symmetric in y, so the y-odd "
         "channel 2 above E2 does not mix); the spec's limit was E2. Both edges are listed; the curves go to E3 - 0.02 meV.",
         "", "## Phase 0: bare HMW potential (no Hartree, no xc, no spin), V0 = 3.0 meV", "",
         "| case | d [nm] | E1 / E2 / E3 [meV] | barrier top: cutoff / grid basis [meV] | ch-1 states | dropped | "
         "max\\|ring - PT\\| (at E) | max\\|ring - Numerov cutoff\\| | max\\|ring - Numerov grid\\| | max\\|T_M1 - T_M2\\| | "
         "M2 window scatter | M1-M2 phase (E > E1 + 0.02) [rad] |", "|" + "---|" * 12]
    for wx, V0, lx in PH0:
        tag = f"wx{wx:g}_V{V0:g}_lx{lx:g}"
        p = os.path.join(OUT, "phase0", tag + ".json")
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        o.append(f"| wx {wx:g}, Lx {lx:g} | {j['d_nm']:.0f} | {j['E1']:.4f} / {j['E2']:.4f} / {j['E3']:.4f} | "
                 f"{j['barrier_top_cutoff']:.4f} / {j['barrier_top_grid']:.4f} | {j['n_kept']} | {len(j['dropped'])} | "
                 f"{f(j['dev_M1_vs_PT']['max'])} ({f(j['dev_M1_vs_PT'].get('at_E'), '{:.3f}')}) | "
                 f"{f(j['dev_M1_vs_numerov_cutoff']['max'])} | {f(j['dev_M1_vs_numerov_grid']['max'])} | "
                 f"{f(j['dev_M1_vs_M2']['max'])} | {f(j['M2_window_scatter'])} | {f(j['dM1_vs_M2_above_bottom'])} |")
    pa = {k: os.path.join(OUT, "phaseA", f"C_exch_wx1.5_{k}.json") for k in ("P", "unpol")}
    if all(os.path.exists(p) for p in pa.values()):
        o += ["", "## Phase A: frozen B = 0 potential, model C, exchange, hbar w_x 1.5 meV, V0 3.0 meV, kT 0.05 meV", "",
              f"Headline: ring method M1 with the frozen leads padded to Lx = {PAD_LX:g} nm (the SCF cell is 5000 nm). "
              "Interpolation test on the P state, spin up: T(mu) = 0.7957 / 0.7806 / 0.7811 for Lx = 5000 / 10000 / "
              "20000 nm, i.e. converged at 10000 nm; the 5000 nm value carries ~+0.015 interpolation error because "
              "T(E) has structure on the scale of the level spacing (0.06 meV per parity).", "",
              "| state | spin | T(mu) ring Lx 10000 | T(mu) ring Lx 5000 | M2 (600-1500) | M2 window spread | <T> mu +- 2kT | "
              "1D Numerov cutoff basis | 1D Numerov old (continuum) | barrier top - mu: cutoff / continuum [meV] | ch-1 states | dropped |",
              "|" + "---|" * 12]
        for k, p in pa.items():
            j = json.load(open(p))
            for s_ in ("0", "1"):
                sp = j["spins"][s_]
                o.append(f"| {k} (conv {'y' if j['converged'] else 'NO'}, M = {j['M']:.3f}) | {'up' if s_ == '0' else 'dn'} | "
                         f"{sp['T_mu_M1']:.4f} | {sp['T_mu_M1_lx5000']:.4f} | {sp['T_mu_M2']:.4f} | {sp['T_mu_window_scatter']:.1e} | "
                         f"{sp['T_avg_2kT_M1']:.4f} | {sp['T_mu_numerov_cutoff']:.4f} | {sp['T_mu_numerov_old']:.4f} | "
                         f"{sp['barrier_top_cutoff'] - j['mu']:+.4f} / {sp['barrier_top_grid'] - j['mu']:+.4f} | {sp['n_kept']} | {len(sp['dropped'])} |")
            o.append(f"| {k} | G = T_up + T_dn | **{j['G_e2h_mu']:.4f} e^2/h** (= {j['G_e2h_mu'] / 2:.4f} x 2e^2/h); "
                     f"<G>_2kT {j['G_e2h_avg2kT']:.4f}; old Numerov {j['G_e2h_mu_numerov_old']:.4f} | | | | | | | | | |")
        o += ["", "Ring vs 1D Numerov: the converged ring value is 2D-exact in the frozen potential (it includes "
              "transverse mode mixing); for the P spin-up channel it lies 0.011 above 1D Numerov on the same "
              "Hamiltonian, a difference that does not shrink with Lx (non-adiabatic, not numerical). M2 fitted close to "
              "the QPC (400-1500 nm) is biased by the Friedel oscillations of the self-consistent potential (window "
              "spread listed); M1 uses only level energies and the clean lead at the cell edge.", ""]
    o += ["", "Reading: 'Numerov cutoff' isolates the ring method (same transverse operator as the KS Hamiltonian; the "
          "difference is non-adiabatic mixing + level interpolation). The PT formula and 'Numerov grid' use the exact "
          "continuum barrier; their difference to the ring result is dominated by the plane-wave cutoff, which raises "
          "the channel-1 barrier top (cutoff vs grid column).", "",
          "Caveats: KS-Landauer on LSDA is a static mean-field conductance (not the Kondo picture of HMW); the method "
          "needs a mirror-symmetric potential and a single channel per symmetry sector; the ring levels are discrete, "
          "so T(E) is interpolated between levels, not sampled; Friedel oscillations make k position dependent near "
          "the QPC (M2 window scan). Phase B: not run yet.", ""]
    os.makedirs("notes", exist_ok=True)
    open(NOTES, "w").write("\n".join(o))


# ----------------------------------------------------------------------------- Phase A
PAD_LX = 10000.0                         # frozen-potential leads padded to this cell length for T (Phase A test:
                                         # T_up(mu) 0.7957 / 0.7806 / 0.7811 at Lx 5000 / 10000 / 20000 nm)
PA_DIR = "results/C_exch_wx1.5"          # model C, exchange, hbar w_x 1.5, V0 3.0, Lx 5000, kT 0.05: B = 0 states
KT = 0.05


def phaseA():
    """Frozen B = 0 potentials (polarised P state and unpolarised reference), per spin: ring T(E) (M1, M2,
    window scatter), 1D Numerov on the same frozen potential (old qpc/transport.py pipeline, continuum barrier;
    and on the cutoff-basis eps_0), T(mu), Fermi-window average (mu +- 2 kT) and G = T_up + T_dn [e^2/h]."""
    from qpc.transport import lead_cut, load_state, mode_profile
    g = grid_from_cutoff(U.nm_to_au(5000.0), U.nm_to_au(320.0), U.meV_to_au(15.0))
    ham = Hamiltonian(g, U.meV_to_au(15.0))
    rows = {}
    for kind, fn in (("P", "B0.npz"), ("unpol", "unpol.npz")):
        z = np.load(os.path.join(PA_DIR, fn))
        mu = U.au_to_meV(float(z["mu"]))
        st = load_state(os.path.join(PA_DIR, fn), 1.5, 3.0)
        E = np.round(np.arange(mu - 1.05, mu + 1.84, 0.002), 6)
        info = dict(kind=kind, source=os.path.join(PA_DIR, fn), converged=bool(z["converged"]), mu=mu,
                    M=float(z["M"]), kT=KT, spins={})
        arr = dict(E=E)
        from qpc.ring_transmission import embed_potential
        g2, ham2 = None, None
        for s, key in ((0, "V_up"), (1, "V_dn")):
            t0 = time.time()
            V = z[key]
            r5 = ring_transmission_2d(V, ham, E)                        # spec cell, Lx = 5000 nm
            g2, V2 = embed_potential(st.F[s], st.grid, st.qpc, PAD_LX)
            ham2 = ham2 or Hamiltonian(g2, U.meV_to_au(15.0))
            r = ring_transmission_2d(V2, ham2, E)                       # headline: leads padded to PAD_LX
            T_cut, top_cut = numerov_eps0(V, ham, "cutoff", E, 2400.0)
            x = np.arange(-2400.0, 2400.01, 1.0)
            e0 = mode_profile(st, s, x, n_modes=1)[:, 0]
            X, _ = lead_cut(x, e0)
            sel = np.abs(x) <= X
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                T_old = scatter_1d(x[sel], e0[sel], E)["T"]
            Tw = {w: float(np.interp(mu, E, t)) for w, t in r["T_M2_windows"].items()}
            from qpc.ring_transmission import fermi_average
            sp = dict(E1=r["E1"], E2=r["E2"], E3=r["E3"], E_top=r["E_top"], n_kept=r["n_kept"], n_y_odd=r["n_y_odd"],
                      pad_lx=PAD_LX, T_mu_M1_lx5000=float(np.interp(mu, E, r5["T_M1"])),
                      T_mu_M2_lx5000=float(np.interp(mu, E, r5["T_M2"])), dropped_lx5000=r5["dropped"],
                      dropped=r["dropped"], barrier_top_cutoff=top_cut, barrier_top_grid=float(e0.max()),
                      T_mu_M1=float(np.interp(mu, E, r["T_M1"])), T_mu_M2=float(np.interp(mu, E, r["T_M2"])),
                      T_mu_M2_windows=Tw, T_mu_window_scatter=float(max(Tw.values()) - min(Tw.values())),
                      T_avg_2kT_M1=fermi_average(E, r["T_M1"], mu, KT, 2.0),
                      T_mu_numerov_cutoff=float(np.interp(mu, E, T_cut)), T_mu_numerov_old=float(np.interp(mu, E, T_old)),
                      dev_ring_vs_numerov_cutoff=dev(r["T_M1"], T_cut, E), dev_ring_vs_numerov_old=dev(r["T_M1"], T_old, E),
                      dev_M1_vs_M2=dev(r["T_M1"], r["T_M2"], E), dM1_vs_M2_phase=r["dM1_vs_M2_max"], time_s=time.time() - t0)
            info["spins"][s] = sp
            arr[f"T_M1_s{s}"], arr[f"T_M2_s{s}"] = r["T_M1"], r["T_M2"]
            arr[f"T_M1_lx5000_s{s}"] = r5["T_M1"]
            arr[f"T_num_cut_s{s}"], arr[f"T_num_old_s{s}"] = T_cut, T_old
            arr[f"E_e_s{s}"], arr[f"E_o_s{s}"] = r["phases"]["e"]["E"], r["phases"]["o"]["E"]
            print(f"[A {kind} s{s}] T(mu) ring Lx {PAD_LX:g}: {sp['T_mu_M1']:.5f}, Lx 5000: {sp['T_mu_M1_lx5000']:.5f} (M2 {sp['T_mu_M2']:.5f}, windows +-{sp['T_mu_window_scatter']:.1e}) "
                  f"<T>_2kT {sp['T_avg_2kT_M1']:.4f} | Numerov cutoff {sp['T_mu_numerov_cutoff']:.5f}, old {sp['T_mu_numerov_old']:.5f} "
                  f"| top cut/grid {top_cut - mu:+.4f}/{e0.max() - mu:+.4f} meV ({sp['time_s']:.0f} s)", flush=True)
        info["G_e2h_mu"] = info["spins"][0]["T_mu_M1"] + info["spins"][1]["T_mu_M1"]
        info["G_e2h_avg2kT"] = info["spins"][0]["T_avg_2kT_M1"] + info["spins"][1]["T_avg_2kT_M1"]
        info["G_e2h_mu_numerov_old"] = info["spins"][0]["T_mu_numerov_old"] + info["spins"][1]["T_mu_numerov_old"]
        os.makedirs(os.path.join(OUT, "phaseA"), exist_ok=True)
        np.savez_compressed(os.path.join(OUT, "phaseA", f"C_exch_wx1.5_{kind}.npz"), **arr)
        jdump(os.path.join(OUT, "phaseA", f"C_exch_wx1.5_{kind}.json"), info)
        rows[kind] = info
    fig_phaseA()
    notes()


def fig_phaseA():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    UP, DN = "#1f5fbf", "#e8710a"
    jP = json.load(open(os.path.join(OUT, "phaseA", "C_exch_wx1.5_P.json")))
    zP = np.load(os.path.join(OUT, "phaseA", "C_exch_wx1.5_P.npz"))
    zU = np.load(os.path.join(OUT, "phaseA", "C_exch_wx1.5_unpol.npz"))
    z0 = np.load(os.path.join(OUT, "phase0", "wx1.5_V3_lx5000.npz"))
    j0 = json.load(open(os.path.join(OUT, "phase0", "wx1.5_V3_lx5000.json")))
    mu = jP["mu"]
    E1 = jP["spins"]["0"]["E1"]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw=dict(wspace=0.22))
    for a, log in zip(axs, (False, True)):
        a.grid(True, color=GRID, lw=0.6)
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(colors=MUTED, labelsize=8)
        Er = zP["E"] - mu
        a.plot(Er, zP["T_M1_s0"], color=UP, lw=2, label=r"$T_\uparrow$ ring (P state)")
        a.plot(Er, zP["T_M1_s1"], color=DN, lw=2, ls="--", label=r"$T_\downarrow$ ring (P state)")
        a.plot(Er, zP["T_num_old_s0"], color=UP, lw=0.8, ls=":", label="1D Numerov, continuum barrier (old method)")
        a.plot(Er, zP["T_num_old_s1"], color=DN, lw=0.8, ls=":")
        a.plot(zU["E"] - mu, zU["T_M1_s0"], color=INK, lw=1.4, label="unpolarised state (ring), either spin")
        a.plot(z0["E"] - j0["E1"] + E1 - mu, z0["T_M1"], color=MUTED, lw=1.2, ls="-.",
               label="non-interacting (Phase 0), band bottoms aligned")
        a.axvline(0, color=INK, lw=0.9, ls=":")
        a.text(0.01, 0.02 if not log else 2e-7, r"$\mu$", fontsize=9, color=INK)
        e2 = jP["spins"]["0"]["E2"] - mu
        a.axvline(e2, color=MUTED, lw=0.8, ls=(0, (1, 3)))
        a.text(e2 + 0.02, 0.5 if not log else 1e-3, "E2 (y-odd ch. 2 opens)", fontsize=7, color=MUTED, rotation=90)
        a.set_xlim(E1 + 0.05 - mu, jP["spins"]["0"]["E_top"] - mu)
        if log:
            a.set_yscale("log")
            a.set_ylim(1e-8, 2)
        else:
            a.set_ylim(-0.02, 1.05)
        a.set_xlabel(r"$E-\mu$ (meV)", fontsize=9)
        a.set_ylabel("transmission (channel 1)", fontsize=9)
    s0, s1 = jP["spins"]["0"], jP["spins"]["1"]
    fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, fontsize=8, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Ring (even/odd phase-shift) transmission, model C, exchange, $\\hbar\\omega_x$ = 1.5 meV, B = 0 "
                 f"(net spin {jP['M']:.2f}): $T_\\uparrow(\\mu)$ = {s0['T_mu_M1']:.3f}, "
                 f"$T_\\downarrow(\\mu)$ = {s1['T_mu_M1']:.3f}, G = {jP['G_e2h_mu']:.3f} $e^2/h$", fontsize=9.5,
                 color=INK, y=1.0)
    axs[0].set_title("linear", fontsize=9, color=MUTED, loc="left")
    axs[1].set_title("log scale", fontsize=9, color=MUTED, loc="left")
    fig.text(0.125, -0.13, "T(E) in the potential self-consistent at mu (frozen), not a self-consistent function of mu. "
             "Static mean-field (KS-Landauer) transmission, not the Kondo picture.", fontsize=8, color=MUTED)
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, "fig_T_vs_E.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- Phase B
PB_ROOT = "results/transmission/phaseB"
PB_UP = [3.0, 3.5, 4.0, 5.0]               # 3.0 first (5 T start from the clean wire), then upward ...
PB_DOWN = [2.5, 2.0, 1.5, 1.0]             # ... and downward, each 5 T start seeded from the neighbouring V0
RAMP5 = [5.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.0]
PB_EST = "notes/ring_B_estimate.txt"


def pb_name(V0):
    return f"C_exch_wx1.5_V{float(V0)}"


def pb_neighbour(V0):
    for chain in (PB_UP, PB_DOWN):
        if V0 in chain:
            i = chain.index(V0)
            if chain is PB_UP:
                return None if i == 0 else PB_UP[i - 1]
            return 3.0 if i == 0 else PB_DOWN[i - 1]
    return None


def ring_T_state(npz, V0, kinds=(0, 1)):
    """Phase A procedure on one saved state: ring M1 (leads padded to PAD_LX) and M1 at Lx 5000, T at mu, per spin."""
    from qpc.ring_transmission import embed_potential, fermi_average
    from qpc.transport import load_state
    z = np.load(npz)
    mu = U.au_to_meV(float(z["mu"]))
    st = load_state(npz, 1.5, V0)
    E = np.round(np.arange(mu - 0.4, mu + 0.4 + 1e-9, 0.002), 6)
    out = dict(mu=mu, converged=bool(z["converged"]), M=float(z["M"]), spins={})
    g5 = st.grid
    ham5 = Hamiltonian(g5, U.meV_to_au(15.0))
    ham2 = None
    same = np.allclose(z["V_up"], z["V_dn"])
    for s in kinds:
        if s == 1 and same:
            out["spins"][1] = out["spins"][0]
            continue
        V = z["V_up" if s == 0 else "V_dn"]
        r5 = ring_transmission_2d(V, ham5, E, windows_nm=[(600.0, 1500.0), (1000.0, 2400.0)])
        g2, V2 = embed_potential(st.F[s], st.grid, st.qpc, PAD_LX)
        ham2 = ham2 or Hamiltonian(g2, U.meV_to_au(15.0))
        r = ring_transmission_2d(V2, ham2, E, windows_nm=[(600.0, 1500.0), (1000.0, 2400.0)])
        out["spins"][s] = dict(T_mu=float(np.interp(mu, E, r["T_M1"])), T_mu_lx5000=float(np.interp(mu, E, r5["T_M1"])),
                               T_mu_M2=float(np.interp(mu, E, r["T_M2"])),
                               T_avg_2kT=fermi_average(E, r["T_M1"], mu, KT, 2.0), E1=r["E1"], E2=r["E2"],
                               n_kept=r["n_kept"], dropped=len(r["dropped"]))
    out["G_e2h"] = out["spins"][0]["T_mu"] + out["spins"][1]["T_mu"]
    return out


def noninteracting_T(V0, e_above_bottom):
    """Bare HMW potential (wx 1.5, this V0), ring T at E = E1 + e_above_bottom (the interacting state's mu - E1)."""
    g = grid_from_cutoff(U.nm_to_au(5000.0), U.nm_to_au(320.0), U.meV_to_au(15.0))
    ham = Hamiltonian(g, U.meV_to_au(15.0))
    V = external_potential(g, QPCParams(hbar_wx_meV=1.5, V0_meV=V0))
    E = np.round(np.arange(1.06, 4.98, 0.002), 6)
    r = ring_transmission_2d(V, ham, E, windows_nm=[(600.0, 1500.0)])
    return float(np.interp(r["E1"] + e_above_bottom, E, r["T_M1"]))


def phaseB_case(V0, maxiter=300):
    """Janak ramp 5 -> 0 T at this V0 (Folder of scripts/qpc_driver.py), then the ring transmission of the
    B = 0 polarised state and of the unpolarised reference; ringT.json in the folder."""
    sys.path.insert(0, os.path.dirname(__file__))
    from qpc_driver import Folder
    t0 = time.time()
    F = Folder("C", "exchange", 1.5, maxiter=maxiter, log=lambda m: print(m, flush=True), level_diag=True, V0=V0,
               root=PB_ROOT, name=pb_name(V0))
    nb = pb_neighbour(V0)
    seed = os.path.join(PB_ROOT, pb_name(nb), "B5.npz") if nb is not None else None
    F.ramp(RAMP5, start_from=seed if seed and os.path.exists(seed) else None)
    t_scf = time.time() - t0
    b0 = json.load(open(F.path("B0.json")))
    P = ring_T_state(F.path("B0.npz"), V0)
    Un = ring_T_state(F.path("unpol.npz"), V0)
    w = json.load(open(F.path("wire_B0.json")))
    e_rel = P["mu"] - w["subbands_up_meV"][0]
    info = dict(V0=V0, hbar_wx=1.5, d_nm=U.au_to_nm(QPCParams(hbar_wx_meV=1.5, V0_meV=V0).d), seed=seed,
                converged=b0["converged"], iterations=b0["iterations"], M_loc=b0["M_loc"], zeta0=b0["zeta0"],
                n1d_0=b0["n1d_0"], P=P, unpol=Un, mu_minus_E1=e_rel, T_noninteracting=noninteracting_T(V0, e_rel),
                time_scf_s=t_scf, time_total_s=time.time() - t0)
    jdump(F.path("ringT.json"), info)
    print(f"[ringB V0 {V0}] conv {info['converged']} M_loc {info['M_loc']:.3f} | T_up {P['spins'][0]['T_mu']:.4f} "
          f"T_dn {P['spins'][1]['T_mu']:.4f} G {P['G_e2h']:.4f} e^2/h | unpol G {Un['G_e2h']:.4f} | "
          f"non-int T {info['T_noninteracting']:.4f} | SCF {t_scf / 60:.1f} min, total {info['time_total_s'] / 60:.1f} min",
          flush=True)


def phaseB_summary():
    rows = []
    for V0 in sorted(PB_UP + PB_DOWN):
        p = os.path.join(PB_ROOT, pb_name(V0), "ringT.json")
        if os.path.exists(p):
            rows.append(json.load(open(p)))
    if not rows:
        return
    cols = ["V0", "d_nm", "converged", "M_loc", "zeta0", "T_up", "T_dn", "G_e2h", "T_up_lx5000", "T_dn_lx5000",
            "G_unpol_e2h", "T_noninteracting", "mu_minus_E1"]
    lines = [",".join(cols)]
    for j in rows:
        P, Un = j["P"], j["unpol"]
        lines.append(",".join(str(v) for v in (j["V0"], f"{j['d_nm']:.1f}", j["converged"], f"{j['M_loc']:.4f}",
                     f"{j['zeta0']:.3f}", f"{P['spins']['0']['T_mu']:.5f}", f"{P['spins']['1']['T_mu']:.5f}",
                     f"{P['G_e2h']:.5f}", f"{P['spins']['0']['T_mu_lx5000']:.5f}", f"{P['spins']['1']['T_mu_lx5000']:.5f}",
                     f"{Un['G_e2h']:.5f}", f"{j['T_noninteracting']:.5f}", f"{j['mu_minus_E1']:.4f}")))
    os.makedirs(PB_ROOT, exist_ok=True)
    open(os.path.join(PB_ROOT, "summary.csv"), "w").write("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    UP, DN = "#1f5fbf", "#e8710a"
    V = np.array([j["V0"] for j in rows])
    conv = np.array([j["converged"] for j in rows])
    Tu = np.array([j["P"]["spins"]["0"]["T_mu"] for j in rows])
    Td = np.array([j["P"]["spins"]["1"]["T_mu"] for j in rows])
    Gu = np.array([j["unpol"]["G_e2h"] for j in rows])
    Tn = np.array([j["T_noninteracting"] for j in rows])
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1], hspace=0.1))
    for ax in (a, b):
        ax.grid(True, color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)

    def pts(ax, y, color, marker, label, ls="-"):
        ax.plot(V, y, color=color, lw=1.5, ls=ls)
        ax.plot(V[conv], y[conv], marker, color=color, ms=7, ls="none", label=label)
        ax.plot(V[~conv], y[~conv], marker, mfc="white", mec=color, ms=7, ls="none")
    pts(a, Tu + Td, INK, "o", r"$G/(e^2/h) = T_\uparrow + T_\downarrow$ (polarised B = 0 state)")
    pts(a, Tu, UP, "^", r"$T_\uparrow(\mu)$")
    pts(a, Td, DN, "v", r"$T_\downarrow(\mu)$", ls="--")
    a.plot(V, Gu, color=MUTED, lw=1.2, label="G/(e^2/h), unpolarised state")
    a.plot(V, 2 * Tn, color=MUTED, lw=1.2, ls="--", label="2 x T non-interacting (same E - band bottom)")
    a.set_ylabel(r"transmission / $G$ ($e^2/h$)", fontsize=9)
    a.legend(frameon=False, fontsize=7.5)
    a.set_title(r"Ring transmission at fixed $\mu$ vs barrier height, model C, exchange, $\hbar\omega_x$ = 1.5 meV "
                "(hollow: not converged)", fontsize=9, color=INK, loc="left")
    pts(b, np.array([j["M_loc"] for j in rows]), INK, "o", "net spin (M_loc)")
    pts(b, np.array([j["zeta0"] for j in rows]), MUTED, "s", r"$\zeta_0$")
    b.set_xlabel(r"$V_0$ (meV)  (d = $\sqrt{2V_0/m^*}/\omega_x$ grows with $V_0$)", fontsize=9)
    b.legend(frameon=False, fontsize=7.5)
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, "fig_T_vs_V0.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def phaseB(go):
    sys.path.insert(0, os.path.dirname(__file__))
    import overnight as ov
    n = ov.physical_cores()
    env = dict(os.environ, OMP_NUM_THREADS=str(n), OPENBLAS_NUM_THREADS=str(n), MKL_NUM_THREADS=str(n),
               PYTHONUNBUFFERED="1")
    order = PB_UP + PB_DOWN
    for i, V0 in enumerate(order if go else order[:1]):
        jid = f"RB_V{V0:g}"
        if ov.marker(jid) == "ok":
            ov.say(f"{jid}: ok earlier; skipped")
            continue
        ov.say(f"{jid} START V0 = {V0}")
        t0 = time.time()
        code = ov.run_job(jid, 240, [[ov.PY, "scripts/run_ring_T.py", "phaseB-case", "--V0", str(V0)]], env)
        ov.say(f"{jid} END {'ok' if code == 0 else 'FAILED (' + str(code) + ')'} after {(time.time() - t0) / 60:.1f} min")
        subprocess.run([ov.PY, "scripts/run_ring_T.py", "phaseB-summary"], cwd=ov.ROOT, env=env)
        import glob
        extra = (glob.glob(os.path.join(ov.ROOT, PB_ROOT, "*", "*.json")) + [os.path.join(ov.ROOT, PB_ROOT, "summary.csv")])
        ov.git_sync(jid, msg=f"ring transmission Phase B: V0 = {V0} meV", extra=[p for p in extra if os.path.exists(p)])
        if i == 0 and not go:
            j = json.load(open(os.path.join(PB_ROOT, pb_name(V0), "ringT.json")))
            m = j["time_total_s"] / 60
            txt = (f"V0 = {V0}: {m:.1f} min (SCF ramp {j['time_scf_s'] / 60:.1f} min + ring T). Remaining {len(order) - 1} "
                   f"V0 values: ~{(len(order) - 1) * m / 60:.1f} h (larger V0 may need more iterations).")
            open(PB_EST, "w").write(txt + "\n")
            ov.say(txt + " Stopping; run with --go for the rest.")
            return
    ov.say("Phase B queue finished")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["phase0", "phaseA", "figs", "phaseB", "phaseB-case", "phaseB-summary"])
    ap.add_argument("--V0", type=float, default=None)
    ap.add_argument("--go", action="store_true", help="phaseB: run all V0 (after the first-V0 estimate)")
    a = ap.parse_args()
    if a.phase == "phaseB":
        return phaseB(a.go)
    if a.phase == "phaseB-case":
        return phaseB_case(a.V0)
    if a.phase == "phaseB-summary":
        return phaseB_summary()
    if a.phase == "phase0":
        phase0()
    elif a.phase == "phaseA":
        phaseA()
    else:
        fig_phase0()
        if os.path.exists(os.path.join(OUT, "phaseA", "C_exch_wx1.5_P.json")):
            fig_phaseA()
        notes()


if __name__ == "__main__":
    main()
