"""notes/OVERNIGHT_SUMMARY.md from results/*/ JSON files only (fast; works with whatever exists).

One table per folder (all fields), a final table of the B = 0 values of every folder next to the
PRL targets, the clean-wire numbers of each model, spin-gain and energy results, and the list of
unconverged states with their M_loc statistics. Also writes notes/spin_gain_overnight.txt.
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


def main():
    folders = sorted(d for d in glob.glob("results/*/") if glob.glob(d + "*.json"))
    out = [f"# Overnight summary ({time.strftime('%Y-%m-%d %H:%M')})", "",
           "Columns: M_loc (local moment), M_win (|x|<300 nm), zeta0 = spin polarisation of n_1D at x = 0, "
           "n1d0 [1e-2 nm^-1], barrier heights in meV above e_0(far) as height@x(nm), e0up(0) = spin-up value at "
           "x = 0, D_dn = top_dn - top_unpol, D_up0 = e0up(0) - top_unpol, LDOS (eta 0.05/0.1 mean): resonance "
           "e - mu, Gamma = FWHM_fit - 2 eta, U (HMW definition), all meV.", ""]
    b0rows, unconv, wires, gains, energies = [], [], [], [], []
    for d in folders:
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
                unconv.append(f"- {name} B = {r['B']:g}: {r['iterations']} it, M_loc last100 mean "
                              f"{f(s.get('mean'))} std {f(s.get('std'))} (min {f(s.get('min'))}, max {f(s.get('max'))})")
            if r["B"] == 0:
                b0rows.append((name, r))
        w0 = load(d + "wire_B0.json")
        if w0:
            wires.append(f"| {name} | {f(w0['mu_meV'], '{:.4f}')} | {f(w0['subbands_up_meV'][1] - w0['subbands_up_meV'][0])} | "
                         f"{f(w0['mu_meV'] - w0['subbands_up_meV'][0])} | {f(2 * w0['N_sub_up'][1], '{:.2f}')} | "
                         f"{w0.get('hartree_scale', 1.0)} |")
        g = load(d + "spin_gain.json")
        if g:
            gains.append(f"{name:28s} lambda(2-step, last 5) = {g['lambda_2step']:.4f}   r_{len(g['r'])} = {g['r_last']:.4f}")
        e = load(d + "energies.json")
        if e:
            energies.append(f"- {name}: Omega_pol - Omega_unpol (B = 0) = {e['dOmega_meV']:+.5f} meV")
        out.append("")
    out += ["## B = 0 of every folder vs PRL targets", "",
            "| folder | conv | M_loc | zeta0 | n1d0 | up peaks | e0up(0) | dn peaks | D_dn | D_up0 | mu-e0far | "
            "res-mu | Gamma | U |", "|" + "---|" * 14]
    for name, r in b0rows:
        L = r.get("ldos", {})
        out.append(f"| {name} | {'y' if r['converged'] else 'NO'} | {f(r['M_loc'])} | {f(r['zeta0'], '{:.2f}')} | "
                   f"{f(100 * r['n1d_0'], '{:.3f}')} | {peaks(r['peaks_up'])} | {f(r['e0_0_up'])} | {peaks(r['peaks_dn'])} | "
                   f"{f(r['D_dn'], '{:+.3f}')} | {f(r['D_up0'], '{:+.3f}')} | {f(r['mu_e0far'])} | "
                   f"{f(L.get('res_e_minus_mu_mean'), '{:+.3f}')} | {f(L.get('Gamma_mean'))} | {f(L.get('U_mean'))} |")
    out += ["", "PRL targets (user's readings, polarised, B = 0):"] + [f"- wx {k}: {v}" for k, v in TARGETS.items()] + [f"- {ALL}", ""]
    out += ["## Clean wire (B = 0) per folder", "", "| folder | mu | e1-e0 | mu-e0 | N(subband 1) | hartree_scale |",
            "|---|---|---|---|---|---|"] + wires + [""]
    out += ["## Spin gain (B = 0, linear mixing, 1 % seed)", "```"] + (gains or ["(none yet)"]) + ["```", ""]
    out += ["## Grand potential Omega_pol - Omega_unpol at B = 0", ""] + (energies or ["(none yet)"]) + [""]
    out += ["## Unconverged states (maxiter reached)", ""] + (unconv or ["(none)"]) + [""]
    os.makedirs("notes", exist_ok=True)
    open("notes/OVERNIGHT_SUMMARY.md", "w").write("\n".join(out))
    if gains:
        open("notes/spin_gain_overnight.txt", "w").write(
            "Spin gain at B = 0 (lambda from the last 5 two-step ratios sqrt(M_(n+2)/M_n); lambda > 1 unstable)\n"
            + "\n".join(gains) + "\n")
    print(f"notes/OVERNIGHT_SUMMARY.md: {len(folders)} folders, {len(b0rows)} B = 0 states")


if __name__ == "__main__":
    main()
