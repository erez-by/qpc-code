"""Two localized electrons in a longer QPC -- Phase 1 (uniform-field scan), docs/TWO_ELECTRON_SPEC.md.

Model C (hartree_scale 0.8, interacting leads), interp exchange, E_cut 15 meV, Ly 320 nm, Lx 5000 nm,
kT 0.05 meV, uniform Zeeman field. Folders results/two_e/C_exch_wx<W>_V<V0>[_lx7500][_kT0.02]/ (Folder
layout of scripts/qpc_driver.py: unpol, wire_B*, B<B>.npz/.json, energies.json) plus analysis.json /
analysis.npz (qpc/analysis_moments.py) of the B = 0 state; results/two_e/summary.csv and
notes/TWO_E_SUMMARY.md are rebuilt after every job.

Per case: Janak ramp B = 5, 4, 2, 1, 0.5, 0.25, 0 T, each field from the previous converged state; mu(B)
from the model C clean wire; the 5 T start of V0 = 2.5 and 3.5 is the converged 5 T density of V0 = 3.0
at the same hbar w_x (V0 = 3.0 starts from the clean wire, as the existing ramps). Then
Omega_pol - Omega_unpol at B = 0 and the moment analysis (lobes, D, N_D, eps_0,s(x), LDOS at the lobes).

Jobs (markers results/done/<id>.ok; a rerun skips finished jobs, the SCF resumes from checkpoints):
  TE01         wx 1.0, V0 3.0 (also: 5 T ramp vs the existing 6 T run; d Omega / d mu = -N check)
  TE<wx>_<V0>  the other 14 cases of wx {1.0, 0.7, 0.5, 0.35, 0.25} x V0 {2.5, 3.0, 3.5}
  TEL          wx 0.25, V0 3.0 at Lx = 7500 nm (cell-size check: s(x) and Omega vs Lx = 5000)
  TEK_<case>   kT = 0.02 reruns of the cases with two lobes or a broad single lobe (decided at run time;
               5 T start from the kT 0.05 5 T state)

Usage:
  python scripts/two_e.py                 runs TE01 only, then writes notes/two_e_estimate.txt and STOPS
  nohup python scripts/two_e.py --go >> logs/two_e.log 2>&1 &      everything (after you read the estimate)
  python scripts/two_e.py --status | --summary
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import overnight as ov                                       # noqa: E402

ROOT = ov.ROOT
OUT = os.environ.get("TWO_E_OUT", "results/two_e")     # override only for smoke tests
RAMP5 = [5.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.0]
WX = [1.0, 0.7, 0.5, 0.35, 0.25]
V0S = [3.0, 2.5, 3.5]                    # 3.0 first: it seeds the other two
EST = os.path.join(ROOT, "notes", "two_e_estimate.txt")
DMU_MEV = 1e-3


def cname(wx, V0, lx=5000.0, kT=0.05):
    return (f"C_exch_wx{float(wx)}_V{float(V0)}" + ("" if float(lx) == 5000.0 else f"_lx{float(lx):g}")
            + ("" if float(kT) == 0.05 else f"_kT{float(kT):g}"))


def log(m):
    print(m, flush=True)


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


# ---------------------------------------------------------------------------------------- one case
def run_case(wx, V0, lx, kT, seed, dmu_check, maxiter=300):
    import numpy as np
    from qpc_driver import Folder, jdump
    from qpc.analysis import n1d
    from qpc.analysis_moments import analyse
    from qpc.scf import SCFResult
    F = Folder("C", "exchange", wx, lx=lx, kT=kT, maxiter=maxiter, log=log, level_diag=True, V0=V0, root=OUT,
               name=cname(wx, V0, lx, kT))
    t0 = time.time()
    start = seed if seed and os.path.exists(seed) else None
    log(f"[two_e] {F.dir}: 5 T start from {start or 'the clean wire'}")
    F.ramp(RAMP5, start_from=start)
    F.energies(min_mloc=-np.inf)
    info, res = write_analysis(F, wx, V0, lx, kT, start)
    if dmu_check:
        dmu_check_case(F, res, info)
    jdump(F.path("timing.json"), dict(wall_s=time.time() - t0, finished=time.ctime()))


def write_analysis(F, wx, V0, lx, kT, start):
    """analysis.json / analysis.npz of the B = 0 state of folder F (qpc/analysis_moments.analyse)."""
    import numpy as np
    from qpc_driver import jdump
    from qpc.analysis import n1d
    from qpc.analysis_moments import analyse
    from qpc.scf import SCFResult
    w = np.load(F.path("wire_B0.npz"))
    n_lead = float(n1d(w["n_up"] + w["n_dn"], F.grid).mean() / F.U.length_nm)
    res = SCFResult.load_npz(F.path("B0.npz"))
    info, arrays = analyse(res, F.ham, F.q, n_lead, units=F.U)
    b0 = load(F.path("B0.json")) or {}
    en = load(F.path("energies.json")) or {}
    info.update(hbar_wx=wx, V0=V0, d_nm=F.U.au_to_nm(F.q.d), lx=lx, kT=kT, M_loc=b0.get("M_loc"),
                M_win300=b0.get("M_win300"), dOmega_pol_unpol_meV=en.get("dOmega_meV"), seed=start,
                ramp=[dict(B=B, **{k: (load(F.path(f"B{B:g}.json")) or {}).get(k) for k in ("M_loc", "converged",
                                                                                            "iterations")})
                      for B in RAMP5])
    np.savez_compressed(F.path("analysis.npz"), **arrays)
    jdump(F.path("analysis.json"), info)
    log(f"[two_e] {F.dir}: {info['cls']} ({info['n_lobes']} moments in D; raw rule {info['cls_raw']}), "
        f"L_D {info['L_D']:.0f} nm, N_D {info['N_D']:.3f}, net spin {info['net_spin']:.3f}, M_loc {info['M_loc']}")
    return info, res


def reanalyse():
    """Recompute analysis.json of every finished case (no SCF), e.g. after a change of the lobe rules."""
    from qpc_driver import Folder
    for d in sorted(glob.glob(os.path.join(ROOT, OUT, "*/"))):
        a = load(d + "analysis.json")
        if not a or not os.path.exists(d + "B0.npz"):
            continue
        F = Folder("C", "exchange", a["hbar_wx"], lx=a["lx"], kT=a["kT"], log=log, V0=a["V0"], root=OUT,
                   name=os.path.basename(d.rstrip("/")))
        write_analysis(F, a["hbar_wx"], a["V0"], a["lx"], a["kT"], a.get("seed"))


def dmu_check_case(F, r0, info):
    """d Omega / d mu = -N at B = 0 (spec Phase 3 test): two SCFs at mu +- 1e-3 meV from the converged state,
    tol 1e-6, Omega from qpc.energy.total_energy; relative deviation |dOmega/dmu + N| / N (target < 1e-3)."""
    from qpc_driver import jdump
    from qpc.energy import total_energy
    from qpc.scf import run_scf
    if os.path.exists(F.path("dOmega_dmu.json")):
        return
    lead, mu, V_ext = F.lead(0.0)
    p = F.params(0.0, tol=1e-6, maxiter=max(F.maxiter, 2))
    dmu = F.U.meV_to_au(DMU_MEV)
    Om, Ns, conv = [], [], []
    for sg in (1, -1):
        r = run_scf(F.ham, F.hart, V_ext, p, mu=mu + sg * dmu, n_init=(r0.n_up, r0.n_dn), verbose=True)
        E = total_energy(r, F.ham, F.hart, V_ext, p)
        Om.append(E["Omega"]); Ns.append(E["N"]); conv.append(bool(r.converged))
    deriv = (Om[0] - Om[1]) / (2 * dmu)
    N = 0.5 * (Ns[0] + Ns[1])
    out = dict(cls=info["cls"], dmu_meV=DMU_MEV, dOmega_dmu=deriv, N=N, rel_dev=abs(deriv + N) / N,
               converged=conv, Omega_meV=[F.U.au_to_meV(o) for o in Om], pass_1e3=abs(deriv + N) / N < 1e-3)
    jdump(F.path("dOmega_dmu.json"), out)
    log(f"[dmu check] dOmega/dmu = {deriv:.5f}, -N = {-N:.5f}, rel. dev. {out['rel_dev']:.2e} "
        f"({'PASS' if out['pass_1e3'] else 'FAIL'} at 1e-3)")


# ---------------------------------------------------------------------------------------- summary
def summary():
    import numpy as np
    rows, csv = [], []
    from qpc.analysis_moments import CSV_COLS, csv_row
    for d in sorted(glob.glob(os.path.join(ROOT, OUT, "*/"))):
        a = load(d + "analysis.json")
        if a:
            rows.append((os.path.basename(d.rstrip("/")), d, a))
            csv.append(csv_row(a, a["hbar_wx"], a["V0"], a["lx"], a["kT"]))
    os.makedirs(os.path.join(ROOT, OUT), exist_ok=True)
    open(os.path.join(ROOT, OUT, "summary.csv"), "w").write(",".join(CSV_COLS) + "\n" + "\n".join(csv) + "\n")

    def f(x, fmt="{:.3f}"):
        try:
            return fmt.format(x)
        except Exception:
            return "-"
    out = [f"# Two-electron scan, Phase 1 ({time.strftime('%Y-%m-%d %H:%M')})", "",
           "Model C, exchange, E_cut 15 meV, Ly 320 nm, uniform field, ramp 5 -> 0 T. B = 0 states. L_D: length of "
           "D = {n < 0.9 n_lead} [nm]; N_D electrons in D; lobes in D as x(spin) [nm (e)]; Friedel: mean same-sign "
           "spacing of the lobes outside D vs pi/k_F = 2/n_lead [nm]; eps0max-mu: top of eps_0,s(x) minus mu (up/dn) "
           "[meV]; dOmega = Omega_pol - Omega_unpol [meV]; LDOS peaks below mu at the lobes (up/dn).", "",
           "class: moments in D after merging same-sign lobes with a shallow dip (s > 0.5 peak) and dropping lobes "
           "with |spin| < 0.3 (satellites); raw: the spec's literal peak count (OPEN_QUESTIONS #13).", "",
           "| case | d | conv/it | M_loc | net spin | L_D | N_D | class (raw) | moments x(spin) | satellites | broad | "
           "Friedel / 2/n | eps0max-mu | dOmega | LDOS pk<mu |", "|" + "---|" * 15]
    for name, d, a in rows:
        lob = " ".join(f"{x:+.0f}({s:+.2f})" for x, s in zip(a["lobe_x"], a["lobe_spins"])) or "-"
        em = a.get("eps0_max_minus_mu", [None, None])
        pk = "; ".join(f"{l['n_peaks_below_mu'][0]}/{l['n_peaks_below_mu'][1]}" for l in a.get("ldos", [])) or "-"
        conv = f"{'y' if a['converged'] else 'NO'}/{a['iterations']}"
        sat = " ".join("%+.0f(%+.2f)" % (l["x"], l["spin"]) for l in a.get("satellites", [])) or "-"
        out.append(f"| {name} | {f(a['d_nm'], '{:.0f}')} | {conv} | {f(a['M_loc'])} | {f(a['net_spin'])} | "
                   f"{f(a['L_D'], '{:.0f}')} | {f(a['N_D'])} | {a['cls']} ({a.get('cls_raw', '-')}) | {lob} | {sat} | "
                   f"{'y' if a['broad'] else '-'} | "
                   f"{f(a['friedel_spacing'], '{:.0f}')} / {f(a['friedel_estimate'], '{:.0f}')} | "
                   f"{f(em[0], '{:+.3f}')}/{f(em[1], '{:+.3f}')} | {f(a['dOmega_pol_unpol_meV'], '{:+.5f}')} | {pk} |")
    out += ["", "## Ramp: M_loc at each field (NO = not converged)", "",
            "| case | " + " | ".join(f"{B:g} T" for B in RAMP5) + " |", "|" + "---|" * (len(RAMP5) + 1)]
    for name, d, a in rows:
        cells = []
        for r in a.get("ramp", []):
            cells.append("-" if r.get("M_loc") is None else f(r["M_loc"]) + ("" if r.get("converged") else " NO"))
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    # 5 T vs 6 T ramp start
    old = load(os.path.join(ROOT, "results/C_exch_wx1.0/B0.json"))
    new = load(os.path.join(ROOT, OUT, cname(1.0, 3.0), "B0.json"))
    out += ["", "## Ramp start: 5 T (this scan) vs 6 T (existing C_exch_wx1.0), wx 1.0, V0 3.0, B = 0", ""]
    if old and new:
        out.append(f"M_loc 6 T start {old['M_loc']:.4f} ({old['iterations']} it), 5 T start {new['M_loc']:.4f} "
                   f"({new['iterations']} it): difference {new['M_loc'] - old['M_loc']:+.4f}; zeta0 "
                   f"{old['zeta0']:.3f} vs {new['zeta0']:.3f}; M_win300 {old['M_win300']:.4f} vs {new['M_win300']:.4f}")
    else:
        out.append("(not yet)")
    dm = load(os.path.join(ROOT, OUT, cname(1.0, 3.0), "dOmega_dmu.json"))
    out += ["", "## d Omega / d mu = -N check (wx 1.0, V0 3.0, B = 0, mu +- 1e-3 meV, tol 1e-6)", ""]
    out.append(f"class {dm['cls']}: dOmega/dmu = {dm['dOmega_dmu']:.5f}, -N = {-dm['N']:.5f}, relative deviation "
               f"{dm['rel_dev']:.2e} -> {'PASS' if dm['pass_1e3'] else 'FAIL'} (1e-3); converged {dm['converged']}"
               if dm else "(not yet)")
    # cell size
    out += ["", "## Cell size: wx 0.25, V0 3.0, Lx 7500 vs 5000 nm (B = 0)", ""]
    a5 = os.path.join(ROOT, OUT, cname(0.25, 3.0), "")
    a7 = os.path.join(ROOT, OUT, cname(0.25, 3.0, lx=7500), "")
    j5, j7 = load(a5 + "analysis.json"), load(a7 + "analysis.json")
    if j5 and j7:
        z5, z7 = np.load(a5 + "analysis.npz"), np.load(a7 + "analysis.npz")
        sel = np.abs(z5["x_nm"]) < 2000
        s7 = np.interp(z5["x_nm"][sel], z7["x_nm"], z7["s_x"])
        ds = np.abs(s7 - z5["s_x"][sel])
        out.append(f"max |s_7500 - s_5000| for |x| < 2000 nm: {ds.max():.2e} nm^-1 (max |s| {np.abs(z5['s_x']).max():.2e}); "
                   f"net spin {j5['net_spin']:.4f} vs {j7['net_spin']:.4f}; class {j5['cls']} vs {j7['cls']}; "
                   f"L_D {j5['L_D']:.0f} vs {j7['L_D']:.0f} nm")
        out.append(f"Omega_pol - Omega_unpol: {f(j5['dOmega_pol_unpol_meV'], '{:+.5f}')} vs "
                   f"{f(j7['dOmega_pol_unpol_meV'], '{:+.5f}')} meV (absolute Omega is extensive: N differs)")
    else:
        out.append("(not yet)")
    e = os.path.exists(EST) and open(EST).read().strip()
    out += ["", "## Timing estimate", "```", e or "(not yet)", "```", ""]
    open(os.path.join(ROOT, "notes", "TWO_E_SUMMARY.md"), "w").write("\n".join(out))
    print(f"notes/TWO_E_SUMMARY.md: {len(rows)} cases")


# ---------------------------------------------------------------------------------------- queue
def case_cmd(wx, V0, lx=5000.0, kT=0.05, seed=None, dmu=False):
    c = [ov.PY, "scripts/two_e.py", "--case", str(wx), str(V0), "--lx", str(lx), "--kT", str(kT)]
    if seed:
        c += ["--seed", seed]
    if dmu:
        c.append("--dmu-check")
    return c


def jobs():
    J = [("TE01", 240, "wx 1.0 V0 3.0 (+ dOmega/dmu check)", [case_cmd(1.0, 3.0, dmu=True)])]
    for wx in WX:
        for V0 in V0S:
            if (wx, V0) == (1.0, 3.0):
                continue
            seed = None if V0 == 3.0 else os.path.join(OUT, cname(wx, 3.0), "B5.npz")
            J.append((f"TE{wx:g}_{V0:g}", 240, f"wx {wx} V0 {V0}", [case_cmd(wx, V0, seed=seed)]))
    J.append(("TEL", 360, "wx 0.25 V0 3.0, Lx 7500 nm (cell-size check)", [case_cmd(0.25, 3.0, lx=7500.0)]))
    return J


def kT_jobs():
    """kT = 0.02 reruns: Lx 5000, kT 0.05 cases with two lobes in D or a broad single lobe."""
    J = []
    for wx in WX:
        for V0 in V0S:
            a = load(os.path.join(ROOT, OUT, cname(wx, V0), "analysis.json"))
            if a and (a["n_lobes"] == 2 or a["broad"]):
                J.append((f"TEK{wx:g}_{V0:g}", 240, f"kT 0.02 rerun wx {wx} V0 {V0} ({a['cls']}, broad {a['broad']})",
                          [case_cmd(wx, V0, kT=0.02, seed=os.path.join(OUT, cname(wx, V0), "B5.npz"))]))
    return J


def post(jid, env, desc):
    import subprocess
    r = subprocess.run([ov.PY, "scripts/two_e.py", "--summary"], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=900)
    ov.say(f"{jid}: summary -> {(r.stdout.strip().splitlines() or ['(no output)'])[-1]}"
           + ("" if r.returncode == 0 else f" [exit {r.returncode}: {r.stderr.strip()[-300:]}]"))
    extra = (glob.glob(os.path.join(ROOT, OUT, "*", "*.json")) + glob.glob(os.path.join(ROOT, OUT, "*", "*_levels.txt"))
             + [os.path.join(ROOT, OUT, "summary.csv")])
    ov.git_sync(jid, msg=f"two_e: {jid} {desc}", extra=[p for p in extra if os.path.exists(p)])


def run_list(J, env):
    for jid, mins, desc, cmds in J:
        if ov.marker(jid) in ("ok", "skipped"):
            ov.say(f"{jid}: ok earlier; skipped")
            continue
        ov.say(f"{jid} START [{mins} min] {desc}")
        t0 = time.time()
        try:
            code = ov.run_job(jid, mins, cmds, env)
        except Exception as exc:
            code = f"exception {exc!r}"
            open(os.path.join(ov.DONE, f"{jid}.failed"), "w").write(f"{time.ctime()} {code}\n")
        ov.say(f"{jid} END {'ok' if code == 0 else 'FAILED (' + str(code) + ')'} after {(time.time() - t0) / 60:.1f} min")
        try:
            post(jid, env, desc)
        except Exception as exc:
            ov.say(f"{jid}: post-job hook error (continuing): {exc!r}")


def write_estimate():
    t = load(os.path.join(ROOT, OUT, cname(1.0, 3.0), "timing.json"))
    if not t:
        return None
    n5000 = len(jobs()) - 2                              # the other 5000 nm cases
    t1 = t["wall_s"] / 60
    lines = [f"TE01 (wx 1.0, V0 3.0, ramp 5 -> 0 T + unpol + energies + analysis + dOmega/dmu check): {t1:.1f} min",
             f"remaining Lx 5000 cases: {n5000} x ~{t1:.0f} min = {n5000 * t1 / 60:.1f} h "
             "(longer QPCs may need more iterations per field: treat as a lower bound)",
             f"Lx 7500 check: ~{3.4 * t1:.0f} min (dense diagonalisation ~ n_pw^3: x 3.4)",
             f"kT 0.02 reruns: ~{t1:.0f} min each, number decided by the scan (0 ... 15)",
             f"Phase 1 total without the reruns: ~{(n5000 * t1 + 3.4 * t1) / 60:.1f} h"]
    open(EST, "w").write("\n".join(lines) + "\n")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", nargs=2, type=float, metavar=("WX", "V0"), help="(internal) one case")
    ap.add_argument("--lx", type=float, default=5000.0)
    ap.add_argument("--kT", type=float, default=0.05)
    ap.add_argument("--seed", default=None)
    ap.add_argument("--dmu-check", action="store_true")
    ap.add_argument("--maxiter", type=int, default=300, help="(internal, smoke tests only)")
    ap.add_argument("--go", action="store_true", help="run the whole queue (after the TE01 estimate)")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--reanalyse", action="store_true", help="recompute analysis.json of finished cases")
    a = ap.parse_args()
    if a.case:
        return run_case(a.case[0], a.case[1], a.lx, a.kT, a.seed, a.dmu_check, a.maxiter)
    if a.summary:
        return summary()
    if a.reanalyse:
        reanalyse()
        return summary()
    if a.status:
        for jid, mins, desc, _ in jobs() + kT_jobs():
            print(f"{jid:12s} {ov.marker(jid) or '-':8s} {desc}")
        if os.path.exists(EST):
            print("\n" + open(EST).read())
        return
    n = ov.physical_cores()
    env = dict(os.environ, OMP_NUM_THREADS=str(n), OPENBLAS_NUM_THREADS=str(n), MKL_NUM_THREADS=str(n),
               PYTHONUNBUFFERED="1")
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    ov.say(f"two_e start: threads = {n}, {'full queue' if a.go else 'TE01 + estimate only'}")
    J = jobs()
    run_list(J[:1], env)
    est = write_estimate()
    if est:
        ov.say("estimate:\n  " + "\n  ".join(est))
    if not a.go:
        ov.say("stopping after TE01 (read notes/two_e_estimate.txt, then run with --go)")
        return
    run_list(J[1:], env)
    run_list(kT_jobs(), env)
    ov.say("two_e Phase 1 queue finished")


if __name__ == "__main__":
    main()
