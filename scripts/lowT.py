"""Low-temperature test of the local moment in model C (exchange): Lx = 10000 nm, kT = 0.02 meV.

PRL (HMW 2003) is at T = 0.1 K (kT = 0.0086 meV); our runs use kT = 0.05 meV at Lx = 5000 nm (level
spacing near mu ~0.06 meV). Doubling Lx halves the spacing (~0.03 meV) so kT = 0.02 meV stays
comparable to it. Everything else as in the C_exch runs (E_cut 15 meV, Ly 320 nm, Pulay, tol 1e-4,
maxiter 300); N = n_1D Lx with n_1D = 2.8e-2 nm^-1 (doubles). Folders results/C_exch_wx<W>_lx10000_kT0.02/.

Queue (one job at a time, markers results/done/<job>.ok|.failed, rerun skips .ok jobs; every SCF is
resumable from its checkpoint):
  lowT-pre     preflight: time per SCF iteration (dense), peak memory incl. the full LDOS
               diagonalisation, clean wire at B = 0 (mu - e_0, spacing, electrons in subband 1)
               -> notes/lowT_preflight.txt. GATE: the queue stops here if an iteration takes more
               than 60 s or the peak memory exceeds 75 % of the RAM Linux sees (use --force to go on).
  lowT-wx2.0   unpolarised B = 0 ref + Janak ramp 6, 4, 2, 1, 0.5, 0.25, 0 T (each from the previous),
               clean wire at every B, LDOS / U / Gamma / barrier / zeta0 / M_loc / M_win300 per state,
               KS levels near mu per iteration (B<B>_levels.txt; crossing statistics if B = 0 does not
               converge), Omega_pol - Omega_unpol at B = 0, spin gain lambda at B = 0
  lowT-wx1.5   the same without the spin gain
  lowT-wx1.0   the same without the spin gain
After every wx job: make_summary.py (low-T section appended to notes/OVERNIGHT_SUMMARY.md),
make_figs.py --tags lx10000_kT0.02, git commit + push.

Usage:  nohup python scripts/lowT.py >> logs/lowT.log 2>&1 &
        python scripts/lowT.py --status      |   python scripts/lowT.py --dry-run
        python scripts/lowT.py --force       (ignore a failed preflight gate)
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import overnight as ov                                       # noqa: E402

ROOT = ov.ROOT
TAG = "lx10000_kT0.02"
LX, KT = "10000", "0.02"
PRE_JSON = os.path.join(ROOT, "results", "done", "lowT_preflight.json")
PRE_TXT = os.path.join(ROOT, "notes", "lowT_preflight.txt")
MAX_IT_S = 60.0
MEM_FRAC = 0.75


def q(wx, *extra):
    return ov.RQ + ["--model", "C", "--interp", "exchange", "--wx", str(wx), "--lx", LX, "--kT", KT,
                    "--tag", TAG] + list(extra)


def wx_job(wx, gain=False):
    cmds = [q(wx, "--level-diag", "--B", *ov.RAMP), q(wx, "--energies", "--energies-all")]
    if gain:
        cmds.append(q(wx, "--spin-gain", "40"))
    return cmds


JOBS = [  # (id, minutes, description, commands)
    ("lowT-pre", 60, "preflight: timing, memory, clean wire at Lx 10000, kT 0.02",
     [[ov.PY, "scripts/lowT.py", "--preflight"]]),
    ("lowT-wx2.0", 900, "C exch wx2.0 Lx10000 kT0.02: unpol + ramp + dOmega + spin gain", wx_job(2.0, gain=True)),
    ("lowT-wx1.5", 900, "C exch wx1.5 Lx10000 kT0.02: unpol + ramp + dOmega", wx_job(1.5)),
    ("lowT-wx1.0", 900, "C exch wx1.0 Lx10000 kT0.02: unpol + ramp + dOmega", wx_job(1.0)),
]


# ---------------------------------------------------------------------------------------- preflight
def preflight():
    """The three pre-launch checks; writes notes/lowT_preflight.txt and results/done/lowT_preflight.json."""
    import resource
    import numpy as np
    from qpc_driver import Folder
    from qpc.scf import run_scf

    def rss_gb():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2

    mem = {l.split(":")[0]: int(l.split()[1]) / 1024 ** 2 for l in open("/proc/meminfo") if ":" in l}
    lines = [f"lowT preflight ({time.strftime('%Y-%m-%d %H:%M')}), threads OMP_NUM_THREADS = "
             f"{os.environ.get('OMP_NUM_THREADS', '?')}"]
    F = Folder("C", "exchange", 2.0, lx=float(LX), kT=float(KT), tag=TAG, log=lambda m: print(m, flush=True))
    lines.append(f"grid {F.grid.Nx} x {F.grid.Ny}, n_pw = {F.ham.n_pw} (Lx 5000: 3379), N = {F.N:g}")

    # 3. clean wire
    t0 = time.perf_counter()
    lead, mu, V_ext = F.lead(0.0)
    t_lead = time.perf_counter() - t0
    w = json.load(open(F.path("wire_B0.json")))
    e0, e1 = w["subbands_up_meV"][0], w["subbands_up_meV"][1]
    lines += ["", "3. clean wire, B = 0, model C, kT = 0.02 meV, Lx = 10000 nm "
              f"({t_lead:.0f} s, converged {w.get('converged', '-')}, {w.get('iterations', '-')} it):",
              f"   mu = {w['mu_meV']:.5f} meV, mu - e_0 = {w['mu_meV'] - e0:.4f} meV, "
              f"spacing e_1 - e_0 = {e1 - e0:.4f} meV",
              f"   electrons in subband 1 = {2 * w['N_sub_up'][1]:.3f} (kT 0.05, Lx 5000: 1.90), "
              f"subband 0 = {2 * w['N_sub_up'][0]:.3f}",
              f"   (kT 0.05, Lx 5000: mu - e_0 = 1.072, spacing 1.201 meV)"]

    # 1. time per SCF iteration (QPC, polarised, dense)
    p = F.params(0.0, maxiter=3)
    res = run_scf(F.ham, F.hart, V_ext, p, mu=mu, n_init=(lead.n_up, lead.n_dn), verbose=True, stop=False)
    dts = [h[5] for h in res.history]
    t_it = float(np.median(dts[1:])) if len(dts) > 1 else dts[0]
    nb = res.eigs[0].size
    lines += ["", f"1. QPC SCF iteration (dense, 2 spins, nb = {nb}): " + ", ".join(f"{d:.1f}" for d in dts)
              + f" s  -> {t_it:.1f} s per iteration (Lx 5000: ~3.5 s)"]

    # LDOS step: one full dense diagonalisation per spin (largest memory use)
    t0 = time.perf_counter()
    F.ham.set_potential(res.V[0])
    wv, v = F.ham.eigh_dense()
    t_full = time.perf_counter() - t0
    del wv, v
    lines.append(f"   full dense diagonalisation (LDOS step, per spin): {t_full:.1f} s")

    # 2. memory
    peak = rss_gb()
    tot, avail = mem.get("MemTotal", float("nan")), mem.get("MemAvailable", float("nan"))
    ok_mem = peak < MEM_FRAC * tot
    ok_t = t_it <= MAX_IT_S
    lines += ["", f"2. memory: peak RSS {peak:.2f} GB; Linux (WSL) sees MemTotal {tot:.2f} GB "
              f"(available at start {avail:.2f} GB) -> {'OK' if ok_mem else 'TOO LARGE'} "
              f"(limit {MEM_FRAC:.0%} of MemTotal)"]
    n_it = 30 + 6 * 25 + 60                      # unpol + 6 fields + B = 0 (typical); <= 300 per state
    lines += ["", f"estimate per wx: ~{n_it} iterations x {t_it:.0f} s + 8 x 2 x {t_full:.0f} s LDOS "
              f"= {(n_it * t_it + 16 * t_full) / 3600:.1f} h (an unconverged B = 0 adds up to "
              f"{240 * t_it / 3600:.1f} h)",
              f"GATE: time per iteration {'OK' if ok_t else 'ABOVE'} {MAX_IT_S:.0f} s, memory "
              f"{'OK' if ok_mem else 'NOT OK'} -> {'PASS, queue continues' if ok_t and ok_mem else 'STOP'}"]
    os.makedirs(os.path.dirname(PRE_TXT), exist_ok=True)
    open(PRE_TXT, "w").write("\n".join(lines) + "\n")
    os.makedirs(os.path.dirname(PRE_JSON), exist_ok=True)
    json.dump(dict(t_iter_s=t_it, t_full_diag_s=t_full, peak_rss_gb=peak, mem_total_gb=tot, n_pw=F.ham.n_pw,
                   ok_time=ok_t, ok_mem=ok_mem, wire=w), open(PRE_JSON, "w"), indent=1)
    print("\n".join(lines), flush=True)


# ---------------------------------------------------------------------------------------- queue
def post(job, env, wx=None):
    for c in (["scripts/make_summary.py"], ["scripts/make_figs.py", "--tags", TAG]):
        r = subprocess.run([ov.PY] + c, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
        ov.say(f"{job}: {c[0]} -> {(r.stdout.strip().splitlines() or ['(no output)'])[-1]}"
               + ("" if r.returncode == 0 else f" [exit {r.returncode}: {r.stderr.strip()[-200:]}]"))
    if wx is not None:
        import glob
        extra = glob.glob(os.path.join(ROOT, "results", f"C_exch_wx{wx}_{TAG}", "*_levels.txt"))
        ov.git_sync(job, msg=f"lowT: C exch wx{wx} Lx 10000 nm kT 0.02 meV", extra=extra)


def gate_ok():
    try:
        j = json.load(open(PRE_JSON))
        return j["ok_time"] and j["ok_mem"]
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preflight", action="store_true", help="(internal) run the checks only")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="continue even if the preflight gate fails")
    a = ap.parse_args()
    if a.preflight:
        return preflight()
    if a.status or a.dry_run:
        for jid, mins, desc, cmds in JOBS:
            print(f"{jid:11s} {ov.marker(jid) or '-':8s} [{mins} min] {desc}")
            if a.dry_run:
                for c in cmds:
                    print("      " + " ".join(c[1:]))
        if os.path.exists(PRE_TXT):
            print("\n" + open(PRE_TXT).read())
        return
    n = ov.physical_cores()
    env = dict(os.environ, OMP_NUM_THREADS=str(n), OPENBLAS_NUM_THREADS=str(n), MKL_NUM_THREADS=str(n),
               PYTHONUNBUFFERED="1")
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    ov.say(f"lowT start: threads = {n}")
    for jid, mins, desc, cmds in JOBS:
        if ov.marker(jid) in ("ok", "skipped"):
            ov.say(f"{jid}: ok earlier; skipped")
        else:
            t0 = time.time()
            ov.say(f"{jid} START [{mins} min] {desc}")
            try:
                code = ov.run_job(jid, mins, cmds, env)
            except Exception as exc:
                code = f"exception {exc!r}"
                open(os.path.join(ov.DONE, f"{jid}.failed"), "w").write(f"{time.ctime()} {code}\n")
            ov.say(f"{jid} END {'ok' if code == 0 else 'FAILED (' + str(code) + ')'} "
                   f"after {(time.time() - t0) / 60:.1f} min")
            wx = jid.split("wx")[1] if "wx" in jid else None
            try:
                post(jid, env, wx)
            except Exception as exc:
                ov.say(f"{jid}: post-job hook error (continuing): {exc!r}")
        if jid == "lowT-pre" and not gate_ok():
            ov.say("preflight gate NOT passed (see notes/lowT_preflight.txt)"
                   + ("; --force given, continuing" if a.force else "; stopping. Rerun with --force to go on."))
            if not a.force:
                return
    ov.say("lowT queue finished")


if __name__ == "__main__":
    main()
