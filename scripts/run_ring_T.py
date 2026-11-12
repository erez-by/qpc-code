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
    o += ["", "Reading: 'Numerov cutoff' isolates the ring method (same transverse operator as the KS Hamiltonian; the "
          "difference is non-adiabatic mixing + level interpolation). The PT formula and 'Numerov grid' use the exact "
          "continuum barrier; their difference to the ring result is dominated by the plane-wave cutoff, which raises "
          "the channel-1 barrier top (cutoff vs grid column).", "",
          "Caveats: KS-Landauer on LSDA is a static mean-field conductance (not the Kondo picture of HMW); the method "
          "needs a mirror-symmetric potential and a single channel per symmetry sector; the ring levels are discrete, "
          "so T(E) is interpolated between levels, not sampled; Friedel oscillations make k position dependent near "
          "the QPC (M2 window scan). Phases A and B: not run yet.", ""]
    os.makedirs("notes", exist_ok=True)
    open(NOTES, "w").write("\n".join(o))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["phase0", "figs"])
    a = ap.parse_args()
    if a.phase == "phase0":
        phase0()
    else:
        fig_phase0()
        notes()


if __name__ == "__main__":
    main()
