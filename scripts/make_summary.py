"""notes/OVERNIGHT_SUMMARY.md from results/*/ JSON files only (fast; works with whatever exists).

One table per folder (all fields), a final table of the B = 0 values of every folder next to the
PRL targets, the clean-wire numbers of each model, spin-gain and energy results, and the list of
unconverged states with their M_loc statistics. Also writes notes/spin_gain_overnight.txt.
Folders tagged lx10000_kT0.02 (scripts/lowT.py) go to an appended low-temperature section with a
B = 0 comparison against the kT = 0.05, Lx = 5000 C_exch runs.
"""
import glob
import json
import os
import time

TARGETS = {
    1.0: "net spin 0.85; up peaks ~0.6 at +-75 nm, centre ~0.4; dn ~1.18; zeta(0) ~0.68; Gamma ~0.1; U ~0.6",
    1.5: "net spin 0.93; up centre ~0.58, sides ~0.33 at +-60 nm; dn ~1.3; zeta(0) ~0.74; U ~0.6, Gamma ~0.3",
    2.0: "net spin 0.90; up peak ~0.88, sides ~0.18; dn ~1.5; zeta(0) ~0.70; Gamma ~0.6; U ~0.6",
}
ALL = "all: mu - e_0(far) ~ 1.1 meV; Gamma ~0.1 (1.0) -> ~0.6 (2.0); U ~0.6 nearly constant"


def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def f(x, fmt="{:.3f}"):
    try:
        return fmt.format(x)
    except Exception:
        return "-"


def peaks(pk):
    return " ".join(f"{h:.2f}@{x:+.0f}" for x, h in pk) if pk else "-"


def bkey(name):
    try:
        return float(os.path.basename(name)[1:-5])
    except ValueError:
        return 1e9


LOWT_TAG = "lx10000_kT0.02"
LOWT_REF = "C_exch_wx{}"                    # kT = 0.05, Lx = 5000 comparison folders
LOWT_RUNS = (("kT 0.05, Lx 5000 (ramp)", ""),           # (label, folder tag) shown in the low-T section
             ("kT 0.02, Lx 5000 (B=0 seeded)", "_kT0.02"),
             ("kT 0.02, Lx 10000 (B=0 seeded)", "_lx10000_kT0.02_seed"),
             ("kT 0.02, Lx 10000 (ramp)", "_" + LOWT_TAG))


def folder_block(d, acc):
    """Lines of one folder's table; fills acc (b0rows, unconv, wires, gains, energies)."""
    out = []
    name = os.path.basename(d.rstrip("/"))
    js = sorted(glob.glob(d + "B*.json"), key=bkey)
    un = load(d + "unpol.json")
    out += [f"## {name}", ""]
    if un:
        out.append(f"unpolarised B = 0: top {f(un['top_unpol'])} meV, n1d0 {f(100 * un['n1d_0'])}, "
                   f"mu - e0(far) {f(un['mu_e0far'])}, converged {un['converged']} ({un['iterations']} it)")
        out.append("")
    if js:
        out += ["| B | conv | it | M_loc | M_win | zeta0 | n1d0 | up peaks | e0up(0) | dn peaks | D_dn | D_up0 | "
                "mu-e0far | res-mu | Gamma | U |", "|" + "---|" * 16]
    for j in js:
        r = load(j)
        if not r:
            continue
        L = r.get("ldos", {})
        out.append(f"| {r['B']:g} | {'y' if r['converged'] else 'NO'} | {r['iterations']} | {f(r['M_loc'])} | "
                   f"{f(r['M_win300'])} | {f(r['zeta0'], '{:.2f}')} | {f(100 * r['n1d_0'], '{:.3f}')} | "
                   f"{peaks(r['peaks_up'])} | {f(r['e0_0_up'])} | {peaks(r['peaks_dn'])} | {f(r['D_dn'], '{:+.3f}')} | "
                   f"{f(r['D_up0'], '{:+.3f}')} | {f(r['mu_e0far'])} | {f(L.get('res_e_minus_mu_mean'), '{:+.3f}')} | "
                   f"{f(L.get('Gamma_mean'))} | {f(L.get('U_mean'))} |")
        if not r["converged"]:
            s = r.get("M_loc_last100", {})
            line = (f"- {name} B = {r['B']:g}: {r['iterations']} it, M_loc last100 mean "
                    f"{f(s.get('mean'))} std {f(s.get('std'))} (min {f(s.get('min'))}, max {f(s.get('max'))})")
            lc = r.get("level_crossing_last100")
            if lc:
                line += (f"; swings |dM_loc| > std: {lc['swings']}, of which with a change of the level count "
                         f"within +-0.1 meV of mu: {lc['swings_with_count_change']}; count changes overall "
                         f"{lc['count_changes']}; corr(|dM|, count change) {f(lc['corr_dM_dcount'])}; "
                         f"closest level to mu {f(lc['min_abs_dmin_meV'], '{:.4f}')} meV")
            acc["unconv"].append(line)
        if r["B"] == 0:
            acc["b0rows"].append((name, r))
    w0 = load(d + "wire_B0.json")
    if w0:
        acc["wires"].append(f"| {name} | {f(w0['mu_meV'], '{:.4f}')} | "
                            f"{f(w0['subbands_up_meV'][1] - w0['subbands_up_meV'][0])} | "
                            f"{f(w0['mu_meV'] - w0['subbands_up_meV'][0])} | {f(2 * w0['N_sub_up'][1], '{:.2f}')} | "
                            f"{w0.get('hartree_scale', 1.0)} | {w0.get('kT', '-')} |")
    g = load(d + "spin_gain.json")
    if g:
        acc["gains"].append(f"{name:28s} lambda(2-step, last 5) = {g['lambda_2step']:.4f}   "
                            f"r_{len(g['r'])} = {g['r_last']:.4f}")
    e = load(d + "energies.json")
    if e:
        acc["energies"].append(f"- {name}: Omega_pol - Omega_unpol (B = 0) = {e['dOmega_meV']:+.5f} meV")
    out.append("")
    return out


def b0_table(rows):
    out = ["| folder | conv | M_loc | zeta0 | n1d0 | up peaks | e0up(0) | dn peaks | D_dn | D_up0 | mu-e0far | "
           "res-mu | Gamma | U |", "|" + "---|" * 14]
    for name, r in rows:
        L = r.get("ldos", {})
        out.append(f"| {name} | {'y' if r['converged'] else 'NO'} | {f(r['M_loc'])} | {f(r['zeta0'], '{:.2f}')} | "
                   f"{f(100 * r['n1d_0'], '{:.3f}')} | {peaks(r['peaks_up'])} | {f(r['e0_0_up'])} | {peaks(r['peaks_dn'])} | "
                   f"{f(r['D_dn'], '{:+.3f}')} | {f(r['D_up0'], '{:+.3f}')} | {f(r['mu_e0far'])} | "
                   f"{f(L.get('res_e_minus_mu_mean'), '{:+.3f}')} | {f(L.get('Gamma_mean'))} | {f(L.get('U_mean'))} |")
    return out


def tail(acc):
    return (["## Clean wire (B = 0) per folder", "",
             "| folder | mu | e1-e0 | mu-e0 | N(subband 1) | hartree_scale | kT |", "|---|---|---|---|---|---|---|"]
            + acc["wires"] + [""]
            + ["## Spin gain (B = 0, linear mixing, 1 % seed)", "```"] + (acc["gains"] or ["(none yet)"]) + ["```", ""]
            + ["## Grand potential Omega_pol - Omega_unpol at B = 0", ""] + (acc["energies"] or ["(none yet)"]) + [""]
            + ["## Unconverged states (maxiter reached)", ""] + (acc["unconv"] or ["(none)"]) + [""])


def lowt_section():
    """Low-temperature test (model C, exchange, Lx = 10000 nm, kT = 0.02 meV) vs the kT = 0.05, Lx = 5000 runs."""
    dirs = [f"results/C_exch_wx{w}{t}/" for w in (2.0, 1.5, 1.0) for _, t in LOWT_RUNS[1:]]
    dirs = [d for d in dirs if glob.glob(d + "*.json")]
    if not dirs:
        return []
    acc = dict(b0rows=[], unconv=[], wires=[], gains=[], energies=[])
    out = ["", "# Low-temperature test: model C, exchange, kT = 0.02 meV (Lx 5000 and 10000 nm)", "",
           "Everything else as in C_exch (E_cut 15 meV, Ly 320 nm, Pulay, tol 1e-4); N = n_1D Lx (fixed "
           "n_1D = 2.8e-2 nm^-1). 'seeded': B = 0 started from a converged state (scripts/lowT_quick.py), "
           "'ramp': Janak ramp 6 T -> 0 (scripts/lowT.py). Same columns as above.", ""]
    for d in dirs:
        out += [l.replace("## ", "### ", 1) if l.startswith("## ") else l for l in folder_block(d, acc)]
        sd = load(d + "seed.json")
        if sd:
            out += [f"seed: {sd['source']}, maxiter {sd['maxiter']} (~{sd['t_iter_est_s']:.1f} s per iteration)", ""]
        lv = d + "B0_levels.txt"
        if not load(d + "B0.json") and os.path.exists(lv):
            rows = [l.split() for l in open(lv) if l.strip() and not l.startswith("#")]
            if rows:
                m = [float(r[1]) for r in rows[-20:]]
                out += [f"B = 0 UNFINISHED (time limit): {len(rows)} iterations logged, M_loc last {len(m)}: "
                        f"mean {sum(m) / len(m):.3f}, min {min(m):.3f}, max {max(m):.3f}", ""]
    out += ["### B = 0: kT = 0.02 meV, Lx = 10000 nm vs kT = 0.05 meV, Lx = 5000 nm", "",
            "| wx | run | conv | it | M_loc | M_win | zeta0 | n1d0 | up peaks | dn peaks | mu-e0far | res-mu | "
            "Gamma | U | dOmega |", "|" + "---|" * 15]
    for w in (2.0, 1.5, 1.0):
        for lab, t in LOWT_RUNS:
            d = f"results/C_exch_wx{w}{t}/"
            r = load(d + "B0.json")
            if not r:
                continue
            L = r.get("ldos", {})
            e = load(d + "energies.json")
            out.append(f"| {w} | {lab} | {'y' if r['converged'] else 'NO'} | {r['iterations']} | {f(r['M_loc'])} | "
                       f"{f(r['M_win300'])} | {f(r['zeta0'], '{:.2f}')} | {f(100 * r['n1d_0'], '{:.3f}')} | {peaks(r['peaks_up'])} | "
                       f"{peaks(r['peaks_dn'])} | {f(r['mu_e0far'])} | {f(L.get('res_e_minus_mu_mean'), '{:+.3f}')} | "
                       f"{f(L.get('Gamma_mean'))} | {f(L.get('U_mean'))} | "
                       f"{f(e['dOmega_meV'], '{:+.5f}') if e else '-'} |")
    out += ["", "PRL (T = 0.1 K): net spin 0.85 (1.0), 0.93 (1.5), 0.90 (2.0) meV; mu - e_0(far) ~ 1.1 meV", ""]
    out += [l.replace("## ", "### ", 1) if l.startswith("## ") else l for l in tail(acc)]
    pre = "notes/lowT_preflight.txt"
    if os.path.exists(pre):
        out += ["### Preflight (timing, memory, clean wire)", "```"] + open(pre).read().rstrip().splitlines() + ["```", ""]
    return out


def main():
    lowt = tuple(f"results/C_exch_wx{w}{t}/" for w in (2.0, 1.5, 1.0) for _, t in LOWT_RUNS[1:])
    folders = sorted(d for d in glob.glob("results/*/") if glob.glob(d + "*.json") and d not in lowt)
    out = [f"# Overnight summary ({time.strftime('%Y-%m-%d %H:%M')})", "",
           "Columns: M_loc (local moment), M_win (|x|<300 nm), zeta0 = spin polarisation of n_1D at x = 0, "
           "n1d0 [1e-2 nm^-1], barrier heights in meV above e_0(far) as height@x(nm), e0up(0) = spin-up value at "
           "x = 0, D_dn = top_dn - top_unpol, D_up0 = e0up(0) - top_unpol, LDOS (eta 0.05/0.1 mean): resonance "
           "e - mu, Gamma = FWHM_fit - 2 eta, U (HMW definition), all meV.", ""]
    acc = dict(b0rows=[], unconv=[], wires=[], gains=[], energies=[])
    for d in folders:
        out += folder_block(d, acc)
    out += ["## B = 0 of every folder vs PRL targets", ""] + b0_table(acc["b0rows"])
    out += ["", "PRL targets (user's readings, polarised, B = 0):"] + [f"- wx {k}: {v}" for k, v in TARGETS.items()] + [f"- {ALL}", ""]
    out += tail(acc)
    out += lowt_section()
    os.makedirs("notes", exist_ok=True)
    open("notes/OVERNIGHT_SUMMARY.md", "w").write("\n".join(out))
    if acc["gains"]:
        open("notes/spin_gain_overnight.txt", "w").write(
            "Spin gain at B = 0 (lambda from the last 5 two-step ratios sqrt(M_(n+2)/M_n); lambda > 1 unstable)\n"
            + "\n".join(acc["gains"]) + "\n")
    print(f"notes/OVERNIGHT_SUMMARY.md: {len(folders)} folders, {len(acc['b0rows'])} B = 0 states")


if __name__ == "__main__":
    main()
