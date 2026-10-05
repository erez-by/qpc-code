"""Overnight orchestrator: runs the job queue J01..J20 one job at a time.

Each job = one or more subprocess calls with a common time limit; markers results/done/<JOB>.ok
(.failed with the exit code / 'timeout', .skipped for unmet conditions); jobs with .ok or .skipped
are skipped on a rerun (the SCF itself is resumable from checkpoints). Per job: logs/<JOB>.log;
one-line status (start, end, duration, ETA) to stdout (-> logs/overnight.log). After EVERY job:
make_summary.py + make_figs.py, then git add notes figures results/*/*.json (no npz), commit
"overnight: <JOB>", push (3 tries; failures never stop the queue).

Usage: python scripts/overnight.py [--dry-run] [--status] [--only J05,J06]
       nohup python scripts/overnight.py >> logs/overnight.log 2>&1 &
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = sys.executable
RQ = [PY, "scripts/run_qpc.py"]
RAMP = ["6", "4", "2", "1", "0.5", "0.25", "0"]
RAMP6 = ["6", "2", "1", "0.5", "0.25", "0"]
DONE = os.path.join(ROOT, "results", "done")


def q(model, interp, wx, *extra):
    return RQ + ["--model", model, "--interp", interp, "--wx", str(wx)] + list(extra)


def fig2_fields(model, wx=1.5):
    d = f"results/{model}_exch_wx{wx}"
    return [q(model, "exchange", wx, "--B", "5", "--start-from", f"{d}/B6.npz"),
            q(model, "exchange", wx, "--B", "3", "--start-from", f"{d}/B4.npz"),
            q(model, "exchange", wx, "--B", "7", "8", "9", "10", "--up-from", "6")]


def c_moment():
    """C exch wx1.5 M_loc(B = 0) > 0.3 ?"""
    p = os.path.join(ROOT, "results/C_exch_wx1.5/B0.json")
    try:
        return json.load(open(p))["M_loc"] > 0.3
    except Exception:
        return False


def energy_cmds():
    cmds = []
    for p in sorted(glob.glob(os.path.join(ROOT, "results/*/B0.json"))):
        try:
            j = json.load(open(p))
        except Exception:
            continue
        if j.get("M_loc", 0) > 0.3 and j.get("model") in ("A", "B", "C"):
            c = q(j["model"], j["interp"], j["wx"], "--energies", "--ecut", str(j["ecut"]),
                  "--lx", str(j["lx"]), "--kT", str(j["kT"]))
            if j.get("tag"):
                c += ["--tag", j["tag"]]
            cmds.append(c)
    return cmds


JOBS = [  # (id, minutes, description, commands or callable, condition or None)
    ("J01", 90, "A exch wx2.0: unpol ref + ramp", [q("A", "exchange", 2.0, "--B", *RAMP)], None),
    ("J02", 90, "A exch wx1.5 Fig. 2 fields 5, 3 and 7-10 (up from 6)", fig2_fields("A"), None),
    ("J03", 30, "post-process JSON + LDOS of existing states (no SCF)",
     [q("A", "exchange", 1.0, "--postprocess"), q("A", "exchange", 1.5, "--postprocess"),
      q("A", "quadratic", 1.0, "--postprocess"), q("B", "exchange", 1.0, "--postprocess")], None),
    ("J04", 30, "C clean wire B = 0, 6 T + C unpol refs wx 1.0, 1.5, 2.0",
     [q("C", "exchange", 1.5, "--wire-check", "0", "6")]
     + [q("C", "exchange", w, "--unpol-only") for w in (1.0, 1.5, 2.0)], None),
    ("J05", 90, "C exch wx1.5 ramp", [q("C", "exchange", 1.5, "--B", *RAMP)], None),
    ("J06", 90, "C exch wx1.0 ramp", [q("C", "exchange", 1.0, "--B", *RAMP)], None),
    ("J07", 90, "C exch wx2.0 ramp", [q("C", "exchange", 2.0, "--B", *RAMP)], None),
    ("J08", 15, "interim figures + summary", [], None),
    ("J09", 150, "A exch wx1.5 ecut 20 meV", [q("A", "exchange", 1.5, "--ecut", "20", "--tag", "ecut20",
                                                "--B", *RAMP6)], None),
    ("J10", 180, "A exch wx1.5 Lx 7500 nm", [q("A", "exchange", 1.5, "--lx", "7500", "--tag", "lx7500",
                                               "--B", *RAMP6)], None),
    ("J11", 120, "A exch wx1.25 and wx1.75", [q("A", "exchange", w, "--B", *RAMP6) for w in (1.25, 1.75)], None),
    ("J12", 120, "C exch wx1.25 and wx1.75 (if C wx1.5 has a moment)",
     [q("C", "exchange", w, "--B", *RAMP6) for w in (1.25, 1.75)], c_moment),
    ("J13", 90, "C exch wx1.5 Fig. 2 fields (if C wx1.5 has a moment)", fig2_fields("C"), c_moment),
    ("J14", 180, "C quad ramps wx1.0 and wx1.5", [q("C", "quadratic", w, "--B", *RAMP) for w in (1.0, 1.5)], None),
    ("J15", 90, "A power:1.3333 ramp wx1.5", [q("A", "power:1.3333", 1.5, "--B", *RAMP)], None),
    ("J16", 90, "B exch ramp wx1.5 (bare leads)", [q("B", "exchange", 1.5, "--B", *RAMP)], None),
    ("J17", 120, "kT sensitivity at B = 0, A exch wx1.5 (kT 0.03, 0.08)",
     [q("A", "exchange", 1.5, "--B", "0", "--start-from", "results/A_exch_wx1.5/B0.25.npz", "--kT", kt,
        "--maxiter", "300", "--tag", f"kT{kt}") for kt in ("0.03", "0.08")], None),
    ("J18", 120, "spin-gain table (C exch / C quad wx 1.0, 1.5, 2.0; A quad wx 2.0)",
     [q("C", i, w, "--spin-gain", "40") for i in ("exchange", "quadratic") for w in (1.0, 1.5, 2.0)]
     + [q("A", "quadratic", 2.0, "--spin-gain", "40")], None),
    ("J19", 30, "grand potential Omega_pol - Omega_unpol at B = 0 (folders with M_loc > 0.3)", energy_cmds, None),
    ("J20", 20, "final figures + summary", [], None),
]


def physical_cores():
    try:
        out = subprocess.run(["lscpu", "-p=Core,Socket"], capture_output=True, text=True).stdout
        return max(1, len({l for l in out.splitlines() if l and not l.startswith("#")}))
    except Exception:
        return os.cpu_count() or 1


def marker(job):
    for ext in ("ok", "skipped", "failed"):
        if os.path.exists(os.path.join(DONE, f"{job}.{ext}")):
            return ext
    return None


def say(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def git_sync(job, msg=None, extra=()):
    """git add notes figures results/*/*.json (+ extra paths, forced), commit, push (3 tries)."""
    def run(c):
        return subprocess.run(c, cwd=ROOT, capture_output=True, text=True)
    run(["git", "add", "notes", "figures"])
    jsons = glob.glob(os.path.join(ROOT, "results", "*", "*.json"))
    if jsons:
        run(["git", "add", "-f"] + [os.path.relpath(p, ROOT) for p in jsons])
    if extra:
        run(["git", "add", "-f"] + [os.path.relpath(p, ROOT) for p in extra])
    if run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
        say(f"{job}: nothing to commit")
        return
    c = run(["git", "commit", "-m", f"{msg or 'overnight: ' + job}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"])
    say(f"{job}: commit {'ok' if c.returncode == 0 else 'FAILED: ' + c.stderr.strip()[:200]}")
    for k in range(3):
        p = run(["git", "push", "-q", "origin", "HEAD"])
        if p.returncode == 0:
            say(f"{job}: push ok")
            return
        time.sleep(20 * (k + 1))
    say(f"{job}: push FAILED (continuing): {p.stderr.strip()[:200]}")


def post_job(job, env):
    for s in ("scripts/make_summary.py", "scripts/make_figs.py"):
        r = subprocess.run([PY, s], cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
        say(f"{job}: {s} -> {(r.stdout.strip().splitlines() or ['(no output)'])[-1]}"
            + ("" if r.returncode == 0 else f" [exit {r.returncode}: {r.stderr.strip()[-200:]}]"))
    git_sync(job)


def run_job(job, minutes, cmds, env):
    os.makedirs(DONE, exist_ok=True)
    for ext in ("failed",):
        p = os.path.join(DONE, f"{job}.{ext}")
        if os.path.exists(p):
            os.remove(p)
    deadline = time.time() + 60 * minutes
    log = open(os.path.join(ROOT, "logs", f"{job}.log"), "a")
    code = 0
    for c in cmds:
        left = deadline - time.time()
        if left <= 0:
            code = "timeout"
            break
        log.write(f"\n===== {time.ctime()}  {' '.join(c)}\n")
        log.flush()
        try:
            r = subprocess.run(c, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=left)
            if r.returncode != 0:
                code = r.returncode
                log.write(f"===== exit code {r.returncode}\n")
        except subprocess.TimeoutExpired:
            code = "timeout"
            log.write("===== TIMEOUT\n")
            break
    log.close()
    ext = "ok" if code == 0 else "failed"
    with open(os.path.join(DONE, f"{job}.{ext}"), "w") as fh:
        fh.write(f"{time.ctime()} exit {code}\n")
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    jobs = [j for j in JOBS if only is None or j[0] in only]
    if a.status:
        for jid, mins, desc, _, _ in jobs:
            print(f"{jid}  {marker(jid) or '-':8s} {desc}")
        return
    if a.dry_run:
        tot = 0
        for jid, mins, desc, cmds, cond in jobs:
            m = marker(jid)
            cl = cmds() if callable(cmds) else cmds
            note = "" if cond is None else "  [conditional]"
            print(f"{jid} [{mins:4d} min] {desc}{note}  (marker: {m or '-'})")
            for c in cl:
                print("      " + " ".join(c[1:]))
            if callable(cmds):
                print("      (commands built at run time from the B0.json files)")
            if m not in ("ok", "skipped"):
                tot += mins
        print(f"\ntotal of the time limits of the pending jobs: {tot} min = {tot / 60:.1f} h (upper bound)")
        return
    n = physical_cores()
    env = dict(os.environ, OMP_NUM_THREADS=str(n), OPENBLAS_NUM_THREADS=str(n), MKL_NUM_THREADS=str(n),
               PYTHONUNBUFFERED="1")
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    df = subprocess.run(["df", "-h", ROOT], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    say(f"overnight start: {len(jobs)} jobs, threads = {n}, disk: {df}")
    for k, (jid, mins, desc, cmds, cond) in enumerate(jobs):
        m = marker(jid)
        if m in ("ok", "skipped"):
            say(f"{jid}: {m} earlier; skipped")
            continue
        if cond is not None and not cond():
            os.makedirs(DONE, exist_ok=True)
            open(os.path.join(DONE, f"{jid}.skipped"), "w").write(f"{time.ctime()} condition not met\n")
            say(f"{jid}: condition not met -> skipped ({desc})")
            continue
        eta = sum(j[1] for j in jobs[k:] if marker(j[0]) not in ("ok", "skipped"))
        t0 = time.time()
        say(f"{jid} START [{mins} min] {desc}; pending queue <= {eta} min")
        cl = cmds() if callable(cmds) else cmds
        try:
            code = run_job(jid, mins, cl, env)
        except Exception as exc:
            code = f"exception {exc!r}"
            open(os.path.join(DONE, f"{jid}.failed"), "w").write(f"{time.ctime()} {code}\n")
        say(f"{jid} END {'ok' if code == 0 else 'FAILED (' + str(code) + ')'} after {(time.time() - t0) / 60:.1f} min")
        try:
            post_job(jid, env)
        except Exception as exc:
            say(f"{jid}: post-job hook error (continuing): {exc!r}")
    say("overnight queue finished")


if __name__ == "__main__":
    main()
