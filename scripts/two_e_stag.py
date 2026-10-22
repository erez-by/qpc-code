"""Two-electron spec, Phase 2 (alternating / staggered seed) and Phase 3 (Omega_AP - Omega_P).

Phase 2, per case (wx, V0), from the converged Phase 1 (P) state results/two_e/C_exch_wx<W>_V<V0>/:
  * field pattern (SCFParams.zeeman_profile = "alternating", qpc.scf.alternating_profile): sign (-1)^j on the
    segment of the j-th n(x) peak inside D (prominence >= 0.1e-2 nm^-1), tanh walls (w = 20 nm) at the
    midpoints, envelope -> 0 outside D (w_D = 25 nm), smooth and ~0 at the cell edge.
    FALLBACK (OPEN_QUESTIONS #14) when D holds < 2 n(x) peaks: N = round(net spin) equal segments across the
    spin moment (its interval between the zeros of s).
  * leads at zero field (the field vanishes outside D): mu(0) and the B = 0 wire at every B.
  * ramp B = 5, 4, 2, 1, 0.5, 0.25, 0 T; the 5 T start is the UNPOLARISED B = 0 reference of the P case, so the
    pattern alone chooses the branch. At B = 0 the Hamiltonian is the Phase 1 one.
  * folder results/two_e/C_exch_wx<W>_V<V0>_AP/ (same layout), analysis.json; outcome vs P: 'AP' (s(x)
    anti-correlated with the P state / net spin reduced), 'P' (converged back to P), 'none' (no moment).
Phase 3 (only if the P and AP B = 0 states are both converged and distinct): both re-converged at B = 0 with
  tol 1e-5, 1e-6, 1e-7, 1e-8 (each from the previous) until Omega changes < 1e-4 meV between levels;
  Delta = Omega_AP - Omega_P (qpc.energy.total_energy, same cell, grid, mu). Delta < 0: AP ground state.
Output notes/TWO_E_PHASE2.md, figures/two_e/fig4_P_vs_AP.png, fig5_dOmega.png; commit + push per case.

Usage (after the Phase 1 queue has stopped):
  nohup python scripts/two_e_stag.py >> logs/two_e_stag.log 2>&1 &
  python scripts/two_e_stag.py --status | --summary
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import overnight as ov                                       # noqa: E402

ROOT = ov.ROOT
OUT = os.environ.get("TWO_E_OUT", "results/two_e")     # override only for smoke tests
RAMP5 = [5.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.0]
CASES = [(0.5, 3.0), (0.35, 3.0), (0.5, 3.5)]          # main case first; the last one "if time permits"
TOLS = [1e-5, 1e-6, 1e-7, 1e-8]
OMEGA_STABLE_MEV = 1e-4


def pname(wx, V0):
    return f"C_exch_wx{float(wx)}_V{float(V0)}"


def log(m):
    print(m, flush=True)


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


# ---------------------------------------------------------------------------------------- pattern
def pattern(wx, V0):
    """(peaks_nm, window_nm, source) for the alternating field, from the P state's analysis."""
    import numpy as np
    from qpc.analysis_moments import density_peaks
    d = os.path.join(OUT, pname(wx, V0), "")
    a = load(d + "analysis.json")
    if not a or not a["converged"]:
        raise RuntimeError(f"{d}: no converged Phase 1 (P) state")
    z = np.load(d + "analysis.npz")
    pk, _, _ = density_peaks(z["x_nm"], z["n_x"], z["D_mask"])
    window = tuple(float(v) for v in a["D_edges"])
    if len(pk) >= 2:
        return tuple(pk), window, f"{len(pk)} n(x) peaks in D"
    mom = max(a["lobes"], key=lambda l: abs(l["spin"]))
    N = max(2, int(round(abs(a["net_spin"]))))
    lo, hi = mom["x_lo"], mom["x_hi"]
    pk = [lo + (j + 0.5) * (hi - lo) / N for j in range(N)]
    return tuple(float(p) for p in pk), window, (f"fallback: {len(pk)} equal segments of the spin moment "
                                                  f"[{lo:.0f}, {hi:.0f}] nm (only {len(density_peaks(z['x_nm'], z['n_x'], z['D_mask'])[0])} "
                                                  f"n(x) peak(s) in D; net spin {a['net_spin']:.2f})")


# ---------------------------------------------------------------------------------------- one case
def run_case(wx, V0, maxiter=300):
    import numpy as np
    from qpc_driver import Folder, jdump
    from qpc.analysis import n1d
    from qpc.analysis_moments import analyse
    from qpc.scf import SCFResult
    pk, window, source = pattern(wx, V0)
    zk = dict(zeeman_peaks_nm=pk, zeeman_window_nm=window)
    F = Folder("C", "exchange", wx, maxiter=maxiter, log=log, level_diag=True, V0=V0, root=OUT,
               name=pname(wx, V0) + "_AP", zeeman="alternating", zeeman_kw=zk)
    Pd = os.path.join(OUT, pname(wx, V0), "")
    jdump(F.path("pattern.json"), dict(peaks_nm=pk, window_nm=window, source=source, w_nm=20.0, wD_nm=25.0))
    log(f"[stag] {F.dir}: field pattern from {source}: peaks {np.round(pk, 0)}, D {np.round(window, 0)}")
    t0 = time.time()
    F.ramp(RAMP5, start_from=Pd + "unpol.npz")
    # analysis of the B = 0 state, and the comparison with P
    w = np.load(F.path("wire_B0.npz"))
    n_lead = float(n1d(w["n_up"] + w["n_dn"], F.grid).mean() / F.U.length_nm)
    res = SCFResult.load_npz(F.path("B0.npz"))
    info, arrays = analyse(res, F.ham, F.q, n_lead, units=F.U)
    zP, aP = np.load(Pd + "analysis.npz"), load(Pd + "analysis.json")
    sP, sA = zP["s_x"], arrays["s_x"]
    corr = float(np.dot(sP, sA) / np.sqrt(np.dot(sP, sP) * np.dot(sA, sA))) if np.dot(sA, sA) > 0 else 0.0
    if np.abs(sA).max() < 0.3e-2:
        outcome = "none (no moment)"
    elif corr > 0.9 and abs(info["net_spin"] - aP["net_spin"]) < 0.2:
        outcome = "P (converged back to the parallel state)"
    else:
        outcome = "AP-like (distinct from P)"
    info.update(hbar_wx=wx, V0=V0, pattern=dict(peaks_nm=pk, window_nm=window, source=source), corr_with_P=corr,
                net_spin_P=aP["net_spin"], outcome=outcome,
                ramp=[dict(B=B, **{k: (load(F.path(f"B{B:g}.json")) or {}).get(k) for k in ("M", "converged",
                                                                                            "iterations")})
                      for B in RAMP5])
    np.savez_compressed(F.path("analysis.npz"), **arrays)
    jdump(F.path("analysis.json"), info)
    log(f"[stag] B = 0: net spin {info['net_spin']:.3f} (P: {aP['net_spin']:.3f}), corr(s_AP, s_P) {corr:+.3f}, "
        f"{info['cls']} -> {outcome}; converged {info['converged']} ({info['iterations']} it)")
    if info["converged"] and aP["converged"] and outcome.startswith("AP"):
        phase3(F, Pd)
    else:
        jdump(F.path("phase3.json"), dict(skipped=True, reason=f"AP converged {info['converged']}, P converged "
                                                                 f"{aP['converged']}, outcome {outcome}"))
    jdump(F.path("timing.json"), dict(wall_s=time.time() - t0, finished=time.ctime()))


def tighten(F, res, label):
    """Re-converge a B = 0 state with tol 1e-5 ... 1e-8 until Omega changes < 1e-4 meV. Returns (history, last)."""
    from qpc.energy import total_energy
    from qpc.scf import run_scf
    lead, mu, V_ext = F.lead(0.0)
    hist, prev, r = [], None, res
    for tol in TOLS:
        p = F.params(0.0, tol=tol)
        r = run_scf(F.ham, F.hart, V_ext, p, mu=mu, n_init=(r.n_up, r.n_dn), verbose=False)
        Om = F.U.au_to_meV(total_energy(r, F.ham, F.hart, V_ext, p)["Omega"])
        h = dict(tol=tol, converged=bool(r.converged), iterations=int(r.iterations), Omega_meV=Om, M=float(r.M),
                 change_meV=None if prev is None else Om - prev)
        hist.append(h)
        log(f"[phase3] {label}: tol {tol:g}: Omega {Om:.6f} meV, change {h['change_meV']}, M {r.M:.4f}, "
            f"conv {r.converged} ({r.iterations} it)")
        if prev is not None and r.converged and abs(Om - prev) < OMEGA_STABLE_MEV:
            break
        prev = Om
    return hist, r


def phase3(F, Pd):
    """Delta = Omega_AP - Omega_P at B = 0 with tightened tolerance. Both states use F's grid, mu and
    Hamiltonian (at B = 0 the alternating field is zero, so H is the Phase 1 one)."""
    from qpc_driver import jdump
    from qpc.scf import SCFResult
    if os.path.exists(F.path("phase3.json")) and not (load(F.path("phase3.json")) or {}).get("skipped"):
        return
    hP, rP = tighten(F, SCFResult.load_npz(Pd + "B0.npz"), "P")
    hA, rA = tighten(F, SCFResult.load_npz(F.path("B0.npz")), "AP")
    ok = all(h[-1]["converged"] for h in (hP, hA)) and all(
        h[-1]["change_meV"] is not None and abs(h[-1]["change_meV"]) < OMEGA_STABLE_MEV for h in (hP, hA))
    dO = hA[-1]["Omega_meV"] - hP[-1]["Omega_meV"]
    rP.save_npz(F.path("P_tight.npz"))
    rA.save_npz(F.path("AP_tight.npz"))
    out = dict(P=hP, AP=hA, Delta_meV=dO, Omega_stable=ok, M_P=float(rP.M), M_AP=float(rA.M),
               verdict=("AP (antiparallel) ground state" if dO < 0 else "P (parallel) ground state")
               + ("" if ok else " -- NOT reliable: Omega not stable to 1e-4 meV"))
    jdump(F.path("phase3.json"), out)
    log(f"[phase3] Delta = Omega_AP - Omega_P = {dO:+.6f} meV -> {out['verdict']}")


# ---------------------------------------------------------------------------------------- summary + figures
def summary():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e6e6e6"

    def f(x, fmt="{:.3f}"):
        try:
            return fmt.format(x)
        except Exception:
            return "-"
    rows = [(wx, V0, load(os.path.join(ROOT, OUT, pname(wx, V0) + "_AP", "analysis.json")),
             load(os.path.join(ROOT, OUT, pname(wx, V0), "analysis.json")),
             load(os.path.join(ROOT, OUT, pname(wx, V0) + "_AP", "phase3.json"))) for wx, V0 in CASES]
    o = [f"# Two-electron spec: Phase 2 (alternating seed) and Phase 3 ({time.strftime('%Y-%m-%d %H:%M')})", "",
         "Model C, exchange, Lx 5000 nm, kT 0.05 meV. AP runs: ramp 5 -> 0 T with the alternating field (sign (-1)^j per "
         "n(x)-peak segment in D, zero outside D; leads at zero field), started from the unpolarised B = 0 state. "
         "B = 0 Hamiltonian = Phase 1. NOT CONV. states are marked and never used for Delta.", "",
         "| wx | V0 | field pattern | AP conv/it | net spin AP | net spin P | corr(s_AP, s_P) | class AP | outcome | "
         "Delta = Omega_AP - Omega_P (meV) |", "|" + "---|" * 10]
    for wx, V0, a, aP, p3 in rows:
        if not a:
            o.append(f"| {wx:g} | {V0:g} | (not run yet) | | | | | | | |")
            continue
        d = ("-" if not p3 or p3.get("skipped") else f"{p3['Delta_meV']:+.6f}" + ("" if p3["Omega_stable"] else " (NOT stable)"))
        o.append(f"| {wx:g} | {V0:g} | {a['pattern']['source']}: {' '.join(f'{p:+.0f}' for p in a['pattern']['peaks_nm'])} | "
                 f"{'y' if a['converged'] else 'NO'}/{a['iterations']} | {f(a['net_spin'])} | {f(aP['net_spin'])} | "
                 f"{f(a['corr_with_P'], '{:+.3f}')} | {a['cls']} | {a['outcome']} | {d} |")
    o += ["", "## Ramp (net spin M of the cell at each field; leads at zero field, so M = local moment)", "",
          "| case | " + " | ".join(f"{B:g} T" for B in RAMP5) + " |", "|" + "---|" * (len(RAMP5) + 1)]
    for wx, V0, a, _, _ in rows:
        if a:
            o.append(f"| wx {wx:g} V0 {V0:g} | " + " | ".join(
                "-" if r.get("M") is None else f(r["M"]) + ("" if r.get("converged") else " NO") for r in a["ramp"]) + " |")
    o += ["", "## Phase 3: Omega convergence with the SCF tolerance (meV)", ""]
    for wx, V0, a, _, p3 in rows:
        if not p3:
            continue
        if p3.get("skipped"):
            o.append(f"- wx {wx:g} V0 {V0:g}: skipped ({p3['reason']})")
            continue
        o.append(f"- wx {wx:g} V0 {V0:g}: {p3['verdict']}; Delta = {p3['Delta_meV']:+.6f} meV; M_P {p3['M_P']:.4f}, "
                 f"M_AP {p3['M_AP']:.4f}")
        for lab in ("P", "AP"):
            o.append(f"  - {lab}: " + "; ".join(f"tol {h['tol']:g}: {h['Omega_meV']:.6f} ({h['iterations']} it"
                                                  f"{'' if h['converged'] else ', NOT CONV.'})" for h in p3[lab]))
    open(os.path.join(ROOT, "notes", "TWO_E_PHASE2.md"), "w").write("\n".join(o) + "\n")

    done = [(wx, V0) for wx, V0, a, _, _ in rows if a]
    os.makedirs(os.path.join(ROOT, "figures", "two_e"), exist_ok=True)
    if done:
        from qpc.units import Units
        U = Units()
        fig, axs = plt.subplots(2, len(done), figsize=(4.0 * len(done), 5.6), squeeze=False,
                                gridspec_kw=dict(hspace=0.3, wspace=0.25))
        for c, (wx, V0) in enumerate(done):
            zP = np.load(os.path.join(ROOT, OUT, pname(wx, V0), "analysis.npz"))
            zA = np.load(os.path.join(ROOT, OUT, pname(wx, V0) + "_AP", "analysis.npz"))
            aA = load(os.path.join(ROOT, OUT, pname(wx, V0) + "_AP", "analysis.json"))
            a1, a2 = axs[0, c], axs[1, c]
            for a in (a1, a2):
                a.grid(True, color=GRID, lw=0.6)
                a.spines[["top", "right"]].set_visible(False)
                a.tick_params(colors=MUTED, labelsize=7)
            a1.plot(zP["x_nm"], 100 * zP["s_x"], color=MUTED, lw=1.4, label="P (Phase 1)")
            a1.plot(zA["x_nm"], 100 * zA["s_x"], color=INK, lw=1.6, label="AP run")
            a1.plot(zA["x_nm"], 50 * zA["n_x"], color=INK, lw=1.0, ls="--", label="n(x)/2, AP run")
            a1.axhline(0, color=MUTED, lw=0.6)
            half = 0.5 * aA["L_D"] + 250
            a1.set_xlim(-half, half)
            a1.set_title(f"$\\hbar\\omega_x$={wx:g}, $V_0$={V0:g}: s(x)" + ("" if aA["converged"] else " NOT CONV."),
                         fontsize=8)
            for z, col, lab in ((zP, MUTED, "P"), (zA, INK, "AP")):
                if "eps0" in z:
                    mu = load(os.path.join(ROOT, OUT, pname(wx, V0), "analysis.json"))["mu_meV"]
                    a2.plot(z["x_nm"], z["eps0"][0] - mu, color=col, lw=1.5, label=f"{lab} up")
                    a2.plot(z["x_nm"], z["eps0"][1] - mu, color=col, lw=1.2, ls="--", label=f"{lab} down")
            a2.axhline(0, color=MUTED, lw=0.8, ls=":")
            a2.set_xlim(-half, half)
            a2.set_xlabel("x (nm)", fontsize=8)
            if c == 0:
                a1.set_ylabel(r"$10^{-2}$ nm$^{-1}$", fontsize=8)
                a2.set_ylabel(r"$\epsilon_{0,\sigma}(x)-\mu$ (meV)", fontsize=8)
                a1.legend(frameon=False, fontsize=6.5)
                a2.legend(frameon=False, fontsize=6.5, ncol=2)
        fig.savefig(os.path.join(ROOT, "figures", "two_e", "fig4_P_vs_AP.png"), dpi=170, bbox_inches="tight")
        plt.close(fig)
        pts = [(wx, p3["Delta_meV"], p3["Omega_stable"]) for wx, V0, a, _, p3 in rows
               if p3 and not p3.get("skipped") and V0 == 3.0]
        if pts:
            fig, a = plt.subplots(figsize=(4.0, 3.0))
            a.grid(True, color=GRID, lw=0.6)
            a.spines[["top", "right"]].set_visible(False)
            for wx, dO, ok in pts:
                a.plot([wx], [1000 * dO], "o", ms=8, mfc=INK if ok else "white", mec=INK, mew=1.4)
            a.axhline(0, color=MUTED, lw=0.8)
            a.set_xlabel(r"$\hbar\omega_x$ (meV)", fontsize=8)
            a.set_ylabel(r"$\Omega_{AP}-\Omega_P$ ($\mu$eV)", fontsize=8)
            a.set_title(r"$V_0$ = 3.0 meV (hollow: $\Omega$ not stable to 0.1 $\mu$eV)", fontsize=8)
            fig.savefig(os.path.join(ROOT, "figures", "two_e", "fig5_dOmega.png"), dpi=170, bbox_inches="tight")
            plt.close(fig)
    print(f"notes/TWO_E_PHASE2.md: {len(done)} of {len(CASES)} cases")


# ---------------------------------------------------------------------------------------- queue
def jobs():
    return [(f"TS{wx:g}_{V0:g}", 300, f"AP ramp + Phase 3, wx {wx} V0 {V0}",
             [[ov.PY, "scripts/two_e_stag.py", "--case", str(wx), str(V0)]]) for wx, V0 in CASES]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", nargs=2, type=float, metavar=("WX", "V0"), help="(internal) one case")
    ap.add_argument("--maxiter", type=int, default=300, help="(internal, smoke tests only)")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--force", action="store_true", help="start even if a Phase 1 job is still running")
    a = ap.parse_args()
    if a.case:
        return run_case(a.case[0], a.case[1], a.maxiter)
    if a.summary:
        return summary()
    if a.status:
        for jid, mins, desc, _ in jobs():
            print(f"{jid:10s} {ov.marker(jid) or '-':8s} {desc}")
        return
    busy = subprocess.run(["pgrep", "-f", "two_e.py --case"], capture_output=True, text=True).stdout.strip()
    if busy and not a.force:
        print(f"a Phase 1 job is still running (pid {busy.split()[0]}); wait for it (or use --force)")
        return
    n = ov.physical_cores()
    env = dict(os.environ, OMP_NUM_THREADS=str(n), OPENBLAS_NUM_THREADS=str(n), MKL_NUM_THREADS=str(n),
               PYTHONUNBUFFERED="1")
    ov.say(f"two_e_stag start: threads = {n}")
    for jid, mins, desc, cmds in jobs():
        if ov.marker(jid) in ("ok", "skipped"):
            ov.say(f"{jid}: ok earlier; skipped")
            continue
        ov.say(f"{jid} START [{mins} min] {desc}")
        t0 = time.time()
        code = ov.run_job(jid, mins, cmds, env)
        ov.say(f"{jid} END {'ok' if code == 0 else 'FAILED (' + str(code) + ')'} after {(time.time() - t0) / 60:.1f} min")
        r = subprocess.run([ov.PY, "scripts/two_e_stag.py", "--summary"], cwd=ROOT, env=env, capture_output=True, text=True)
        ov.say(f"{jid}: summary -> {(r.stdout.strip().splitlines() or ['(no output)'])[-1]}"
               + ("" if r.returncode == 0 else f" [exit {r.returncode}: {r.stderr.strip()[-300:]}]"))
        extra = glob.glob(os.path.join(ROOT, OUT, "*_AP", "*.json")) + glob.glob(os.path.join(ROOT, OUT, "*_AP", "*_levels.txt"))
        ov.git_sync(jid, msg=f"two_e Phase 2/3: {jid} {desc}", extra=extra)
    ov.say("two_e_stag queue finished")


if __name__ == "__main__":
    main()
