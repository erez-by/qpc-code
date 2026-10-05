"""Quick (<= 90 min) low-temperature test of the moment at hbar w_x = 2.0 meV, model C, exchange, B = 0.

The full Janak-ramp version is scripts/lowT.py (~10+ h). Here only B = 0, each state SEEDED from a
converged state instead of the 6 T ramp:
  Q1  Lx = 5000 nm, kT = 0.02 meV  (results/C_exch_wx2.0_kT0.02/): clean wire, unpol ref, B = 0 from
      results/C_exch_wx2.0/B0.npz (kT 0.05), Omega_pol - Omega_unpol, spin gain lambda. ~20 min.
  Q2  Lx = 10000 nm, kT = 0.02 meV (results/C_exch_wx2.0_lx10000_kT0.02_seed/): clean wire, unpol ref,
      B = 0 from the Q1 state (or the kT 0.05 state) stretched to the 10 um cell: the old density for
      |x| < 1.5 um, clean-wire density (Lx 10000, kT 0.02) for |x| > 2 um, linear blend in between.
      maxiter is chosen from the measured time per iteration so that the run ends before the deadline
      (unconverged -> M_loc last-100 statistics + level-crossing statistics; LDOS always).
Per iteration the KS levels within +-0.1 meV of mu are logged (B0_levels.txt). After each stage:
make_summary.py (low-T section of notes/OVERNIGHT_SUMMARY.md), make_figs.py --tags, commit + push.
Markers results/done/lowQ-q1.ok, lowQ-q2.ok; a rerun skips finished stages and resumes checkpoints.

Usage:  nohup python scripts/lowT_quick.py >> logs/lowT_quick.log 2>&1 &
        python scripts/lowT_quick.py --status
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
BUDGET_MIN = 90
TAG1, TAG2 = "kT0.02", "lx10000_kT0.02_seed"
SRC = "results/C_exch_wx2.0/B0.npz"


def log(m):
    print(m, flush=True)


def run_b0(F, seed, maxiter=None):
    """B = 0 polarised state at the leads' mu, started from seed = (n_up, n_dn) (as Folder.ramp)."""
    if F.done("B0"):
        log("[B0] finished earlier; skipped")
        return
    if maxiter is not None:
        F.maxiter = int(maxiter)
    lead, mu, V_ext = F.lead(0.0)
    res, t, resumed = F._scf("B0", F.params(0.0), mu, V_ext, seed)
    F.summarise("B0", 0.0, res, t, resumed)
    ck = F.path("B0_ckpt.npz")
    if os.path.exists(ck):
        os.remove(ck)


def stage_q1():
    import numpy as np
    from qpc_driver import Folder
    from qpc.scf import SCFResult
    F = Folder("C", "exchange", 2.0, lx=5000.0, kT=0.02, maxiter=300, tag=TAG1, log=log, level_diag=True)
    F.unpol()
    s = SCFResult.load_npz(SRC)
    run_b0(F, (s.n_up, s.n_dn))
    F.energies(min_mloc=-np.inf)
    if not os.path.exists(F.path("spin_gain.json")):
        F.spin_gain(40)


def stretch(n_old, g_old, g_new, n_lead, U):
    """Density of a 5 um cell on the 10 um grid: old for |x| < 1500 nm, lead for |x| > 2000 nm."""
    import numpy as np
    from qpc.fourier import min_image
    assert g_old.Ny == g_new.Ny and abs(g_old.Ly - g_new.Ly) < 1e-12, "y grids differ"
    xo = min_image(g_old.real_axes()[0], g_old.Lx)
    xn = min_image(g_new.real_axes()[0], g_new.Lx)
    o = np.argsort(xo)
    out = np.empty((g_new.Nx, g_new.Ny))
    for j in range(g_new.Ny):
        out[:, j] = np.interp(xn, xo[o], n_old[o, j])
    w = np.clip((U.nm_to_au(2000.0) - np.abs(xn)) / U.nm_to_au(500.0), 0.0, 1.0)[:, None]
    return w * out + (1 - w) * n_lead


def stage_q2():
    import numpy as np
    from qpc_driver import Folder, jdump
    from qpc.grid import grid_from_cutoff
    from qpc.scf import SCFResult
    deadline = float(os.environ.get("LOWQ_DEADLINE", time.time() + 3600))
    F = Folder("C", "exchange", 2.0, lx=10000.0, kT=0.02, maxiter=300, tag=TAG2, log=log, level_diag=True)
    U = F.U
    unp = F.unpol()
    dts = [h[5] for h in (unp.history or [])]
    t_it = 2.0 * float(np.median(dts[1:])) if len(dts) > 2 else 30.0      # polarised = 2 diagonalisations
    lead, mu, V_ext = F.lead(0.0)
    q1 = f"results/C_exch_wx2.0_{TAG1}/B0.npz"
    src = q1 if os.path.exists(q1) else SRC
    if not F.done("B0"):
        s = SCFResult.load_npz(src)
        g_old = grid_from_cutoff(U.nm_to_au(5000.0), U.nm_to_au(320.0), U.meV_to_au(15.0))
        seed = (stretch(s.n_up, g_old, F.grid, lead.n_up, U), stretch(s.n_dn, g_old, F.grid, lead.n_dn, U))
        if os.path.exists(F.path("B0_ckpt.npz")):
            seed = None                                           # _scf resumes from the checkpoint
        reserve = 3 * t_it + 240                                 # LDOS (2 full diagonalisations) + summary
        mi = int(np.clip((deadline - time.time() - reserve) / t_it, 15, 300))
        jdump(F.path("seed.json"), dict(source=src, t_iter_est_s=t_it, maxiter=mi,
                                        blend_nm=[1500, 2000], time=time.ctime()))
        log(f"[q2] seed {src}; ~{t_it:.1f} s per iteration -> maxiter {mi} "
            f"({(deadline - time.time()) / 60:.0f} min left)")
        run_b0(F, seed if seed is not None else (lead.n_up, lead.n_dn), maxiter=mi)
    if time.time() < deadline - 60:
        F.energies(min_mloc=-np.inf)


def post(job, env, msg):
    import subprocess
    for c in (["scripts/make_summary.py"], ["scripts/make_figs.py", "--tags", f"{TAG1},{TAG2}"]):
        r = subprocess.run([ov.PY] + c, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
        ov.say(f"{job}: {c[0]} -> {(r.stdout.strip().splitlines() or ['(no output)'])[-1]}"
               + ("" if r.returncode == 0 else f" [exit {r.returncode}: {r.stderr.strip()[-200:]}]"))
    extra = glob.glob(os.path.join(ROOT, "results", "C_exch_wx2.0_*", "*_levels.txt"))
    ov.git_sync(job, msg=msg, extra=extra)


JOBS = [("lowQ-q1", "Q1: C exch wx2.0 Lx 5000 kT 0.02, B = 0 seeded", "--stage q1"),
        ("lowQ-q2", "Q2: C exch wx2.0 Lx 10000 kT 0.02, B = 0 seeded", "--stage q2")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["q1", "q2"], default=None, help="(internal)")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.stage:
        return {"q1": stage_q1, "q2": stage_q2}[a.stage]()
    if a.status:
        for jid, desc, _ in JOBS:
            print(f"{jid}  {ov.marker(jid) or '-':8s} {desc}")
        return
    t_start = time.time()
    deadline = t_start + 60 * BUDGET_MIN
    n = ov.physical_cores()
    env = dict(os.environ, OMP_NUM_THREADS=str(n), OPENBLAS_NUM_THREADS=str(n), MKL_NUM_THREADS=str(n),
               PYTHONUNBUFFERED="1", LOWQ_DEADLINE=str(deadline - 300))
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    ov.say(f"lowT_quick start: threads = {n}, budget {BUDGET_MIN} min (ends ~{time.ctime(deadline)})")
    for jid, desc, arg in JOBS:
        if ov.marker(jid) in ("ok", "skipped"):
            ov.say(f"{jid}: ok earlier; skipped")
            continue
        left = (deadline - time.time()) / 60 - 3
        lim = min(35.0, left) if jid == "lowQ-q1" else left
        if lim < 10:
            ov.say(f"{jid}: only {left:.0f} min left; not started (rerun later)")
            continue
        ov.say(f"{jid} START [{lim:.0f} min] {desc}")
        t0 = time.time()
        code = ov.run_job(jid, lim, [[ov.PY, "scripts/lowT_quick.py"] + arg.split()], env)
        ov.say(f"{jid} END {'ok' if code == 0 else 'FAILED (' + str(code) + ')'} after {(time.time() - t0) / 60:.1f} min")
        try:
            post(jid, env, f"lowT quick: {desc}")
        except Exception as exc:
            ov.say(f"{jid}: post-job hook error (continuing): {exc!r}")
    ov.say(f"lowT_quick finished after {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
