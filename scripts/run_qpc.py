"""M5.2 / M5.3: QPC runs at fixed mu = mu_wire (docs/SPEC.md M5).

Model: bare parabola + V_QPC + V_H[n] + v_xc,s[n] - sigma_s E_Z/2 (a_image = 100 nm).

--unpolarised (M5.2): B = 0; mu_wire from the interacting clean wire (N = 140), whose density is
    the start. Saves results/qpc_wx{wx}_unpol.npz.
--B B1 B2 ... (M5.3, Janak ramp; HMW: "first apply an in-plane magnetic field of up to B = 6 T and
    then solve ... while reducing the magnetic field to zero": --B 6 5 4 3 2 1 0.5 0.25 0;
    Fig. 2 states afterwards with --B 7 8 9 10 --up-from 6, NOT part of the Janak protocol):
    for each B in the given order: (1) clean wire spin-polarised at B, fixed N = 140 (Zeeman in
    the leads, HMW Eq. 2) -> mu_wire(B), saved to results/wire_B{B}.npz; (2) QPC at fixed
    mu = mu_wire(B), starting from the previous converged QPC density (the first B: from the
    clean-wire spin densities of that field). Saves results/qpc_wx{wx}_B{B}.npz; converged files
    are skipped (resumable) and used as the start of the next B.
--seed-b0: B = 0 from the UNPOLARISED solution (results/qpc_wx{wx}_unpol.npz) with a 1 % seed
    polarisation for |x| < 300 nm (n_up *= 1.01, n_dn *= 0.99). Saves results/qpc_wx{wx}_B0_seed.npz.
Prints per B: iterations, time, M, M_loc (= M - M_wire(B)), far-field n_up/n_dn, barrier top
and mu - top per spin, N; a table is appended to results/ramp_wx{wx}.txt.
Usage: python scripts/run_qpc.py --wx 1.5 --unpolarised
       python scripts/run_qpc.py --wx 1.5 --B 6 5 4 3 2 1 0.5 0.25 0
       python scripts/run_qpc.py --wx 1.5 --B 7 8 9 10 --up-from 6
       python scripts/run_qpc.py --wx 1.5 --seed-b0
"""
import argparse
import dataclasses
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qpc.analysis import (barrier_features, barrier_profile, format_features, n1d,  # noqa: E402
                          net_spin_local, net_spin_window)
from qpc.fourier import min_image                                 # noqa: E402
from qpc.grid import grid_from_cutoff                             # noqa: E402
from qpc.hamiltonian import Hamiltonian                           # noqa: E402
from qpc.hartree import Hartree                                   # noqa: E402
from qpc.potential import QPCParams                               # noqa: E402
from qpc.scf import (SCFParams, SCFResult, clean_wire, external_potential,  # noqa: E402
                     lead_state, physical_xy, run_scf)
from qpc.units import Units                                       # noqa: E402

LX_NM, LY_NM, ECUT_MEV = 5000.0, 320.0, 15.0
N1D_NM = 2.8e-2
FAR_NM = 1500.0


def summarise(res, ham, U, mu_wire, label, t_run=None):
    g = ham.grid
    meV = U.au_to_meV
    x_nm = U.au_to_nm(min_image(g.real_axes()[0], g.Lx))
    line = n1d(res.n_up + res.n_dn, g) / U.length_nm
    far = np.abs(x_nm) > FAR_NM
    e0 = barrier_profile(res, ham)                       # (Nx, 2) Ha*
    print(f"\n=== {label} ===")
    print(f"mu (leads, N=140)             : {meV(mu_wire):.5f} meV")
    print(f"converged                     : {res.converged} in {res.iterations} iterations"
          + (f", {t_run:.1f} s" if t_run is not None else ""))
    if res.history:
        dts = [h[5] for h in res.history]
        print(f"time per iteration            : mean {np.mean(dts):.2f} s, max {np.max(dts):.2f} s")
    print(f"N                             : {res.N:.4f}")
    print(f"M = N_up - N_dn               : {res.M:.5f}")
    print(f"far-field n_1D (|x|>1.5um)    : {line[far].mean():.6f} nm^-1  "
          f"(max dev {np.abs(line[far] / N1D_NM - 1).max() * 100:.3f} %)")
    print(f"n_1D(x = 0)                   : {line[0]:.6f} nm^-1")
    for s, name in enumerate(("up", "dn")):
        far_s = e0[far, s].mean()
        top = e0[:, s].max()
        x_top = x_nm[np.argmax(e0[:, s])]
        print(f"[{name}] e_0(far)                 : {meV(far_s):.5f} meV")
        print(f"[{name}] barrier top - e_0(far)   : {meV(top - far_s):.5f} meV  (top at x = {x_top:.1f} nm;"
              f" e_0(0) - e_0(far) = {meV(e0[0, s] - far_s):.5f}; bare 3.0)")
        print(f"[{name}] mu - e_0(far)            : {meV(res.mu - far_s):.5f} meV")
        print(f"[{name}] mu - barrier top         : {meV(res.mu - top):+.5f} meV")


def setup(wx):
    U = Units()
    grid = grid_from_cutoff(U.nm_to_au(LX_NM), U.nm_to_au(LY_NM), U.meV_to_au(ECUT_MEV))
    ham = Hamiltonian(grid, U.meV_to_au(ECUT_MEV))
    hart = Hartree(grid, SCFParams().a_metal_au(U))
    return U, grid, ham, hart, QPCParams(hbar_wx_meV=wx)


def bstr(B):
    return f"{B:g}"


# barrier top of the UNPOLARISED B = 0 state of the same model / length [meV above e_0(far)]
TOP_UNPOL = {("interacting", 1.0): 0.6282, ("interacting", 1.5): 0.7094, ("interacting", 2.0): 0.8366,
             ("bare", 1.0): 0.5785, ("bare", 1.5): 0.6344, ("bare", 2.0): 0.7133}


def spin_summary(res, wire, ham, U, wx=None, lead=None):
    """Dict of the per-B quantities (meV / nm^-1), incl. barrier_features ('feat') and
    D_dn = top_dn - top_unpol, D_up0 = e_0,up(0) - top_unpol (TOP_UNPOL of the same model/length)."""
    g = ham.grid
    meV = U.au_to_meV
    x_nm = U.au_to_nm(min_image(g.real_axes()[0], g.Lx))
    far = np.abs(x_nm) > FAR_NM
    e0 = barrier_profile(res, ham)
    top = e0.max(axis=0)
    feat = barrier_features(res, ham, units=U)
    tu = TOP_UNPOL.get((lead, wx), np.nan)
    top_dn = meV(top[1] - e0[far, 1].mean())
    return dict(M=res.M, M_loc=net_spin_local(res, wire, g), M_win=net_spin_window(res, wire, g),
                N=res.N, M_wire=wire.M,
                nup_far=(n1d(res.n_up, g) / U.length_nm)[far].mean(),
                ndn_far=(n1d(res.n_dn, g) / U.length_nm)[far].mean(),
                top_up=meV(top[0] - e0[far, 0].mean()), top_dn=top_dn,
                mu_top_up=meV(res.mu - top[0]), mu_top_dn=meV(res.mu - top[1]),
                mu=meV(res.mu), D_dn=top_dn - tu, D_up0=feat["centre"][0] - tu, feat=feat)


HEADER = (f"{'B[T]':>5} {'M_loc':>8} {'M_win300':>8} {'M':>9} {'M_wire':>8} {'it':>4} {'t[s]':>7} {'N':>9} "
          f"{'nup_far':>9} {'ndn_far':>9} {'top_up':>7} {'top_dn':>7} {'mu-top_up':>9} {'mu-top_dn':>9} "
          f"{'D_dn':>7} {'D_up0':>7} {'conv':>5}")


def row(B, d, it, t, conv):
    return (f"{B:5g} {d['M_loc']:8.4f} {d['M_win']:8.4f} {d['M']:9.4f} {d['M_wire']:8.4f} {it:4d} {t:7.1f} {d['N']:9.4f} "
            f"{d['nup_far']:9.6f} {d['ndn_far']:9.6f} {d['top_up']:7.4f} {d['top_dn']:7.4f} "
            f"{d['mu_top_up']:+9.4f} {d['mu_top_dn']:+9.4f} {d['D_dn']:+7.3f} {d['D_up0']:+7.3f} {str(conv):>5}")


# HMW targets: user's readings of PRL Fig. 1 (polarised state, B = 0, mu - e_0(far) ~ 1.1 meV in the
# paper; barrier heights in meV above the far band bottom). Observation tracked: in HMW
# top_dn - top_unpol ~ +0.6 meV for all three lengths and the spin-up peak ~ the unpolarised top
# (D_dn ~ +0.6, D_up0 ~ -0.18 at wx = 1.0).
_LDOS_T = dict(res="~ -0.5 meV", onset="~ +0.1 meV", U="~0.6 meV", gamma="~0.1 meV")
HMW_TARGETS = {
    1.0: dict(peaks_up="two peaks ~0.6 meV at x ~ +-80 nm (nearly flat top over ~160 nm)",
              centre_up="~0.45 meV (shallow dip ~0.15 meV)", peaks_dn="one peak ~1.2 meV (just above mu)",
              mu_e0far="~1.1 meV", n1d_0="~1.5e-2 nm^-1 total", M_loc="~0.85",
              D="D_dn ~ +0.6, D_up0 ~ -0.18", **_LDOS_T),
    1.5: dict(peaks_up="~0.6 meV", peaks_dn="~1.3 meV", mu_e0far="~1.1 meV", n1d_0="~1.4e-2 nm^-1 total",
              M_loc="~0.93", D="D_dn ~ +0.6", **_LDOS_T),
    2.0: dict(peaks_up="single peak ~0.8 meV at x = 0", peaks_dn="single peak ~1.5 meV", mu_e0far="~1.1 meV",
              M_loc="~0.90", D="D_dn ~ +0.6", **_LDOS_T),
}


def model_tag(lead, interp):
    """'A' / 'B' + '' (exchange) | '_quad' (quadratic) | '_mixed{w}' (mixed:w)."""
    if interp == "exchange":
        suf = ""
    elif interp == "quadratic":
        suf = "_quad"
    elif interp.startswith("power:"):
        suf = "_power" + interp.split(":", 1)[1]
    elif interp.startswith("mixed:"):
        suf = "_mixed" + interp.split(":", 1)[1]
    else:
        suf = "_" + interp
    return ("A" if lead == "interacting" else "B") + suf


def qpc_path(wx, lead, interp, B, suffix=""):
    return f"results/qpc_wx{wx}_{model_tag(lead, interp)}_B{bstr(B)}{suffix}.npz"


def print_features(res, ham, U, d, wx, B, ldos=True):
    """Feature block (barrier maxima, centre, n_1D(0), mu - e_0(far), D columns) and, for
    M_loc > 0.3, the LDOS numbers at x = 0 (eta = 0.05, 0.1); HMW targets printed at B = 0."""
    from qpc.analysis import ldos_1d, ldos_features
    f = d["feat"]
    t = HMW_TARGETS.get(wx, {}) if B == 0 else {}
    print(f"  M = {d['M']:.4f}, M_loc = {d['M_loc']:.4f}" + (f"   # HMW: {t['M_loc']}" if t else "")
          + f", M_win300 = {d['M_win']:.4f}", flush=True)
    print(format_features(f, t), flush=True)
    print(f"  D_dn = top_dn - top_unpol = {d['D_dn']:+.4f} meV, D_up0 = e_0,up(0) - top_unpol = "
          f"{d['D_up0']:+.4f} meV" + (f"   # HMW: {t['D']}" if t else ""), flush=True)
    if ldos and d["M_loc"] > 0.3:
        mu_meV = U.au_to_meV(res.mu)
        e_max = min(U.au_to_meV(np.max(res.eigs[s])) for s in range(2))
        E = np.arange(f["e0_far"][0] - 0.3, e_max - 0.15, 0.002)
        for eta in (0.05, 0.1):
            _, rho = ldos_1d(res, ham, 0.0, E, eta_meV=eta, units=U)
            lf = ldos_features(E, rho, mu_meV, f["e0_far"][0])
            print(f"  LDOS eta={eta}: up resonance e-mu = {lf['res_e']:+.4f} meV"
                  + (f" (# HMW {t['res']})" if t else "")
                  + f", FWHM = {lf['fwhm']:.4f} meV" + (f" (# HMW Gamma {t['gamma']}, incl. 2 eta)" if t else "")
                  + f"; dn onset e-mu = {lf['onset_dn']:+.4f} meV" + (f" (# HMW {t['onset']})" if t else "")
                  + f"; U = {lf['U']:.4f} meV" + (f" (# HMW {t['U']})" if t else ""), flush=True)
    return f


def ramp_end_lines(feats):
    """Summary after a ramp: B with two spin-up peaks, B at the minimum of M_loc(B), M_loc(B = 0)."""
    two = [B for B, d, f in feats if len(f["peaks"][0]) >= 2]
    Bmin, dmin, _ = min(feats, key=lambda t: t[1]["M_loc"])
    b0 = [d["M_loc"] for B, d, f in feats if B == 0]
    return (f"largest B with two spin-up barrier peaks: {max(two) if two else 'none'} T\n"
            f"B at the minimum of M_loc(B): {Bmin:g} T (M_loc = {dmin['M_loc']:.4f})\n"
            f"M_loc(B = 0): " + (f"{b0[0]:.4f}" if b0 else "not in ramp") + "\n"
            f"M_loc(B) in ramp order: {[(B, round(d['M_loc'], 4)) for B, d, f in feats]}")


def run_ramp(wx, B_list, table_path, lead="interacting", up_from=None, interp="quadratic"):
    """Janak ramp (docs/SPEC.md M5.3): B_list in order, e.g. 6 5 4 3 2 1 0.5 0.25 0; the first
    QPC state starts from the lead densities at that field, each next one from the previous
    converged state. up_from = B0: Fig. 2 states ramped UP from the converged B0 state (NOT part
    of the Janak protocol; labelled as such)."""
    U, grid, ham, hart, q = setup(wx)
    N = N1D_NM * LX_NM
    prev, wire_prev = None, None
    label = "Janak ramp (down)"
    if up_from is not None:
        prev = SCFResult.load_npz(qpc_path(wx, lead, interp, up_from))
        if not prev.converged:
            raise RuntimeError(f"B = {up_from} T start state not converged")
        label = f"ramp-UP from the converged {up_from:g} T state (Fig. 2 states, NOT Janak)"
    print(f"{label}: wx = {wx} meV, lead = {lead} (model {model_tag(lead, interp)}), interp = {interp}, "
          f"B = {list(B_list)}", flush=True)
    lines, feats = [], []
    for B in B_list:
        p = SCFParams(B_T=B, lead_reference=lead, interp=interp)
        t0 = time.perf_counter()
        wire_init = None if wire_prev is None else (wire_prev.n_up, wire_prev.n_dn)
        wire, mu_lead, V_ext = lead_state(ham, hart, q, p, N, n_init=wire_init)
        if lead == "interacting":
            wire.res.save_npz(f"results/wire_{interp}_B{bstr(B)}.npz")
        print(f"\n##### B = {B:g} T: leads ({lead}) mu = {U.au_to_meV(mu_lead):.5f} meV, "
              f"M_wire = {wire.M:.4f} ({time.perf_counter() - t0:.1f} s)", flush=True)
        wire_prev = wire
        out = qpc_path(wx, lead, interp, B)
        ckpt = out.replace(".npz", "_ckpt.npz")
        if os.path.exists(out) and SCFResult.load_npz(out).converged:
            res = SCFResult.load_npz(out)
            print(f"{out} exists and is converged; skipped", flush=True)
            t_run = float("nan")
        else:
            if os.path.exists(ckpt):
                z = np.load(ckpt)
                n_init = (z["n_up"], z["n_dn"])
                print(f"resuming from checkpoint {ckpt} (iteration {int(z['it'])})", flush=True)
            elif prev is None:
                n_init = (wire.n_up, wire.n_dn)
            else:
                n_init = (prev.n_up, prev.n_dn)
            t1 = time.perf_counter()
            res = run_scf(ham, hart, V_ext, p, mu=mu_lead, n_init=n_init,
                          X_init=None if prev is None else prev.X, checkpoint=ckpt, checkpoint_every=1)
            t_run = time.perf_counter() - t1
            res.save_npz(out)
            print(f"saved {out}", flush=True)
        d = spin_summary(res, wire, ham, U, wx, lead)
        print(HEADER + "\n" + row(B, d, res.iterations, t_run, res.converged), flush=True)
        lines.append(row(B, d, res.iterations, t_run, res.converged))
        feats.append((B, d, print_features(res, ham, U, d, wx, B)))
        if not res.converged:
            print(f"WARNING: B = {B:g} not converged; continuing from it", flush=True)
        prev = res
    with open(table_path, "a") as fh:
        fh.write(f"# {label}, wx = {wx} meV, lead = {lead}, B = {list(B_list)}  ({time.ctime()})\n{HEADER}\n"
                 + "\n".join(lines) + "\n")
    print("\n" + HEADER + "\n" + "\n".join(lines))
    print("\n" + ramp_end_lines(feats), flush=True)


def run_large_seed(wx, table_path, lead, interp, maxiter=200, kT=0.05, sigma_nm=60.0, M0=1.0):
    """Large-seed coexistence test at B = 0 (Pulay, tol 1e-4): from the unpolarised state,
    n_up *= (1 + s f(x)), n_dn *= (1 - s f(x)), f = exp(-x^2 / (2 sigma^2)), s such that M_init = M0."""
    unpol, _, _, ham, hart, grid, q, _ = unpolarised_state(wx, lead, kT, verbose=False)
    U = Units()
    p = SCFParams(B_T=0.0, kT=kT, interp=interp, lead_reference=lead, maxiter=maxiter)
    lead_obj, mu, V_ext = lead_state(ham, hart, q, p, N1D_NM * LX_NM)
    X, _ = physical_xy(grid)
    f = np.exp(-U.au_to_nm(X) ** 2 / (2 * sigma_nm ** 2))
    dA = grid.dx * grid.dy
    s = M0 / float((f * (unpol.n_up + unpol.n_dn)).sum() * dA)
    if s * f.max() >= 1:
        raise ValueError(f"seed amplitude s = {s:.3f} would make n_dn negative")
    n_up, n_dn = unpol.n_up * (1 + s * f), unpol.n_dn * (1 - s * f)
    print(f"large-seed B = 0: wx = {wx}, model {model_tag(lead, interp)}, sigma = {sigma_nm} nm, s = {s:.4f}, "
          f"initial M = {(n_up - n_dn).sum() * dA:.4f}, Pulay, maxiter = {maxiter}", flush=True)
    out = qpc_path(wx, lead, interp, 0.0, "_largeseed")
    t1 = time.perf_counter()
    res = run_scf(ham, hart, V_ext, p, mu=mu, n_init=(n_up, n_dn),
                  checkpoint=out.replace(".npz", "_ckpt.npz"), checkpoint_every=1)
    t_run = time.perf_counter() - t1
    res.save_npz(out)
    d = spin_summary(res, lead_obj, ham, U, wx, lead)
    line = row(0.0, d, res.iterations, t_run, res.converged)
    print("\n" + HEADER + "\n" + line)
    print_features(res, ham, U, d, wx, 0.0)
    with open(table_path, "a") as fh:
        fh.write(f"# large seed (M0 = {M0}, sigma = {sigma_nm} nm) B = 0, wx = {wx}, model "
                 f"{model_tag(lead, interp)} ({time.ctime()})\n{HEADER}\n{line}\n")


def run_seed_b0(wx, table_path, lead="interacting", interp="quadratic", maxiter=150, kT=0.05):
    """Seeded B = 0 (Pulay, tol 1e-4): start from the converged UNPOLARISED state of the model
    with n_up *= 1.01, n_dn *= 0.99 for |x| < 300 nm. Saves results/qpc_wx{wx}_B0_seed_{lead}_{interp}.npz."""
    unpol, _, _, ham, hart, grid, q, _ = unpolarised_state(wx, lead, kT, verbose=False)
    U = Units()
    p = SCFParams(B_T=0.0, kT=kT, interp=interp, lead_reference=lead, maxiter=maxiter)
    lead_obj, mu, V_ext = lead_state(ham, hart, q, p, N1D_NM * LX_NM)
    X, _ = physical_xy(grid)
    near = np.abs(U.au_to_nm(X)) < 300.0
    n_up = np.where(near, unpol.n_up * 1.01, unpol.n_up)
    n_dn = np.where(near, unpol.n_dn * 0.99, unpol.n_dn)
    print(f"seeded B = 0, wx = {wx} meV, lead = {lead}, interp = {interp}, Pulay, maxiter = {maxiter}: "
          f"initial M = {(n_up - n_dn).sum() * grid.dx * grid.dy:.4f}", flush=True)
    out = f"results/qpc_wx{wx}_B0_seed_{lead}_{interp}" + ("" if kT == 0.05 else f"_kT{kT}") + ".npz"
    t1 = time.perf_counter()
    res = run_scf(ham, hart, V_ext, p, mu=mu, n_init=(n_up, n_dn),
                  checkpoint=out.replace(".npz", "_ckpt.npz"), checkpoint_every=1)
    t_run = time.perf_counter() - t1
    res.save_npz(out)
    d = spin_summary(res, lead_obj, ham, U, wx, lead)
    line = row(0.0, d, res.iterations, t_run, res.converged)
    print("\n" + HEADER + "\n" + line)
    summarise(res, ham, U, mu, f"seeded B = 0: wx = {wx}, lead = {lead}, interp = {interp}", t_run)
    with open(table_path, "a") as fh:
        fh.write(f"# B = 0 from unpolarised + 1 % seed, wx = {wx} meV, lead = {lead}, interp = {interp} "
                 f"({time.ctime()})\n{HEADER}\n{line}\n")


def main():
    if "--model" in sys.argv:                     # folder-based driver (overnight runs)
        import qpc_driver
        return qpc_driver.main(sys.argv[1:])
    ap = argparse.ArgumentParser()
    ap.add_argument("--wx", type=float, default=1.5, help="hbar w_x in meV")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--unpolarised", action="store_true")
    mode.add_argument("--B", type=float, nargs="+",
                      help="Janak ramp, fields in T in order (6 5 4 3 2 1 0.5 0.25 0)")
    mode.add_argument("--seed-b0", action="store_true")
    mode.add_argument("--large-seed", action="store_true", help="B = 0, Gaussian seed with M_init = 1")
    ap.add_argument("--lead", choices=["interacting", "bare"], default="interacting",
                    help="lead reference (model A / B)")
    ap.add_argument("--interp", default=None,   # quadratic | exchange | mixed:w
                    help="default: SCFParams default (quadratic)")
    ap.add_argument("--maxiter", type=int, default=None, help="--seed-b0 (150) / --large-seed (200)")
    ap.add_argument("--kT", type=float, default=0.05, help="kT in meV; --unpolarised / --seed-b0")
    ap.add_argument("--up-from", type=float, default=None,
                    help="with --B 7 8 9 10: ramp UP from the converged state at this field (Fig. 2)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    table = f"results/ramp_wx{args.wx}.txt"
    if args.B is not None:
        return run_ramp(args.wx, args.B, table, lead=args.lead, up_from=args.up_from,
                        interp=args.interp or SCFParams().interp)
    if args.large_seed:
        return run_large_seed(args.wx, table, args.lead, args.interp or SCFParams().interp,
                              args.maxiter or 200, args.kT)
    if args.seed_b0:
        return run_seed_b0(args.wx, table, args.lead, args.interp or "quadratic", args.maxiter or 150, args.kT)
    run_unpolarised(args)


def unpol_path(wx, lead="interacting", kT=0.05):
    """results/qpc_wx{wx}_unpol[_bare][_kT{kT}].npz (model A at kT = 0.05: the M5.2 file)."""
    return (f"results/qpc_wx{wx}_unpol" + ("" if lead == "interacting" else "_bare")
            + ("" if kT == 0.05 else f"_kT{kT}") + ".npz")


def unpolarised_state(wx, lead="interacting", kT=0.05, force=False, verbose=True):
    """Converged UNPOLARISED QPC at B = 0, fixed mu of the leads of model `lead` (computed and
    saved if missing). Returns (res, lead_obj, mu, ham, hart, grid, q, t_run)."""
    U, grid, ham, hart, q = setup(wx)
    p = SCFParams(spin_polarized=False, B_T=0.0, kT=kT, lead_reference=lead)
    t0 = time.perf_counter()
    lead_obj, mu, V_ext = lead_state(ham, hart, q, p, N1D_NM * LX_NM)
    print(f"leads ({lead}): mu = {U.au_to_meV(mu):.5f} meV ({time.perf_counter() - t0:.1f} s)", flush=True)
    out = unpol_path(wx, lead, kT)
    ckpt = out.replace(".npz", "_ckpt.npz")
    t_run = None
    if os.path.exists(out) and not force and SCFResult.load_npz(out).converged:
        print(f"{out} exists and is converged; loading (use --force to recompute)", flush=True)
        res = SCFResult.load_npz(out)
    else:
        n_init = (lead_obj.n_up, lead_obj.n_dn)
        if os.path.exists(ckpt) and not force:
            z = np.load(ckpt)
            n_init = (z["n_up"], z["n_dn"])
            print(f"resuming from checkpoint {ckpt} (iteration {int(z['it'])})", flush=True)
        t0 = time.perf_counter()
        res = run_scf(ham, hart, V_ext, p, mu=mu, n_init=n_init, checkpoint=ckpt,
                      checkpoint_every=1, verbose=verbose)
        t_run = time.perf_counter() - t0
        res.save_npz(out)
        print(f"saved {out}", flush=True)
    return res, lead_obj, mu, ham, hart, grid, q, t_run


def run_unpolarised(args):
    U = Units()
    res, lead_obj, mu, ham, hart, grid, q, t_run = unpolarised_state(args.wx, args.lead, args.kT,
                                                                     args.force)
    model = "A (interacting leads)" if args.lead == "interacting" else "B (bare leads, delta formulation)"
    summarise(res, ham, U, mu, f"QPC hbar w_x = {args.wx} meV, unpolarised, B = 0, model {model}, "
                               f"kT = {args.kT} meV", t_run)

if __name__ == "__main__":
    main()
