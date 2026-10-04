"""Folder-based QPC driver (overnight runs). Used by scripts/run_qpc.py when --model is given.

Folder: results/<model>_<interp>_wx<W>[_<tag>]/ with
  unpol.npz / unpol.json      unpolarised B = 0 reference (fixed mu of the leads), eigenvectors kept
  wire_B<B>.npz / .json       clean-wire lead state at field B (model A/C: interacting; B: bare reference)
  B<B>.npz / B<B>.json        QPC state at field B (+ LDOS arrays); B<B>_ckpt.npz checkpoint (every 25 it)
  spin_gain.json, energies.json
Models: A = interacting leads; B = bare leads (delta formulation, non-interacting reference);
        C = A with hartree_scale = 0.8 (SENSITIVITY knob, not physical).
A field is skipped when B<B>.json exists with "finished": true (converged, or maxiter reached:
then converged = false and M_loc statistics over the last 100 iterations are recorded).
"""
import dataclasses
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qpc.analysis import barrier_features, ldos_1d, n1d                 # noqa: E402
from qpc.fourier import min_image                                       # noqa: E402
from qpc.grid import grid_from_cutoff                                   # noqa: E402
from qpc.hamiltonian import Hamiltonian                                 # noqa: E402
from qpc.hartree import Hartree                                         # noqa: E402
from qpc.mixing import Pulay                                            # noqa: E402,F401
from qpc.potential import QPCParams                                     # noqa: E402
from qpc.scf import SCFParams, SCFResult, lead_state, physical_xy, run_scf   # noqa: E402
from qpc.units import Units                                             # noqa: E402

N1D_NM = 2.8e-2
LY_NM = 320.0
MODELS = {"A": dict(lead_reference="interacting", hartree_scale=1.0),
          "B": dict(lead_reference="bare", hartree_scale=1.0),
          "C": dict(lead_reference="interacting", hartree_scale=0.8)}


def interp_short(interp):
    if interp == "exchange":
        return "exch"
    if interp == "quadratic":
        return "quad"
    if interp.startswith("power:"):
        return "pow" + interp.split(":", 1)[1]
    if interp.startswith("mixed:"):
        return "mix" + interp.split(":", 1)[1]
    return interp


def bstr(B):
    return f"{B:g}"


def jdump(path, d):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))
    with open(path + ".tmp", "w") as fh:
        json.dump(d, fh, indent=1, default=conv)
    os.replace(path + ".tmp", path)


def lorentz_c(e, A, e0, w, c):
    return A * (0.5 * w) ** 2 / ((e - e0) ** 2 + (0.5 * w) ** 2) + c


class Folder:
    def __init__(self, model, interp, wx, ecut=15.0, lx=5000.0, kT=0.05, maxiter=300, tag=None,
                 log=print):
        self.model, self.interp, self.wx = model, interp, float(wx)
        self.ecut, self.lx, self.kT, self.maxiter = float(ecut), float(lx), float(kT), int(maxiter)
        self.tag = tag
        self.log = log
        name = f"{model}_{interp_short(interp)}_wx{self.wx}" + (f"_{tag}" if tag else "")
        self.dir = os.path.join("results", name)
        os.makedirs(self.dir, exist_ok=True)
        U = self.U = Units()
        self.grid = grid_from_cutoff(U.nm_to_au(self.lx), U.nm_to_au(LY_NM), U.meV_to_au(self.ecut))
        self.ham = Hamiltonian(self.grid, U.meV_to_au(self.ecut))
        p0 = self.params(0.0)
        self.hart = Hartree(self.grid, p0.a_metal_au(U), verbose=False)
        self.q = QPCParams(hbar_wx_meV=self.wx)
        self.N = N1D_NM * self.lx
        x_nm = U.au_to_nm(min_image(self.grid.real_axes()[0], self.grid.Lx))
        self.x_nm = x_nm
        self.far_nm = 0.3 * self.lx
        self._leads = {}
        log(f"[folder] {self.dir}: grid {self.grid.Nx}x{self.grid.Ny}, n_pw {self.ham.n_pw}, N = {self.N:g}, "
            f"kT = {self.kT}, maxiter = {self.maxiter}")

    # ------------------------------------------------------------------ helpers
    def params(self, B, polarized=True, **kw):
        base = dict(B_T=float(B), kT=self.kT, interp=self.interp, maxiter=self.maxiter,
                    spin_polarized=polarized, **MODELS[self.model])
        base.update(kw)
        return SCFParams(**base)

    def path(self, name):
        return os.path.join(self.dir, name)

    def keep_X(self, B):
        return B == 0 or abs(self.wx - 1.5) < 1e-9

    def lead(self, B, n_init=None):
        """(lead, mu, V_ext) at field B, cached; saves wire_B<B>.npz / .json (densities + subbands)."""
        if B in self._leads:
            return self._leads[B]
        t0 = time.perf_counter()
        p = self.params(B)
        lead, mu, V_ext = lead_state(self.ham, self.hart, self.q, p, self.N, n_init=n_init)
        U = self.U
        info = dict(B=B, model=self.model, interp=self.interp, mu_meV=U.au_to_meV(mu), M=float(lead.M),
                    subbands_up_meV=U.au_to_meV(np.asarray(lead.subbands[0][:3])).tolist(),
                    subbands_dn_meV=U.au_to_meV(np.asarray(lead.subbands[1][:3])).tolist(),
                    N_sub_up=np.asarray(lead.N_sub[0][:3]).tolist(), N_sub_dn=np.asarray(lead.N_sub[1][:3]).tolist(),
                    hartree_scale=p.hartree_scale, kT=self.kT, time=time.perf_counter() - t0)
        if hasattr(lead, "res") and hasattr(lead.res, "converged") and lead.res is not lead:
            info["converged"] = bool(lead.res.converged)
            info["iterations"] = int(lead.res.iterations)
        np.savez_compressed(self.path(f"wire_B{bstr(B)}.npz"), n_up=lead.n_up, n_dn=lead.n_dn, mu=mu, M=lead.M)
        jdump(self.path(f"wire_B{bstr(B)}.json"), info)
        self.log(f"[lead] B = {B:g} T: mu = {info['mu_meV']:.5f} meV, M_wire = {lead.M:.4f}, "
                 f"N_sub up {np.round(info['N_sub_up'], 3)} dn {np.round(info['N_sub_dn'], 3)}")
        self._leads[B] = (lead, mu, V_ext)
        return self._leads[B]

    def wire_M(self, B):
        lead, _, _ = self.lead(B)
        return float(lead.M)

    def done(self, name):
        j = self.path(name + ".json")
        if not os.path.exists(j):
            return False
        try:
            return bool(json.load(open(j)).get("finished", False))
        except Exception:
            return False

    # ------------------------------------------------------------------ SCF of one state
    def _scf(self, name, p, mu, V_ext, n_init):
        ckpt = self.path(name + "_ckpt.npz")
        resumed = 0
        if os.path.exists(ckpt):
            z = np.load(ckpt)
            n_init = (z["n_up"], z["n_dn"])
            resumed = int(z["it"])
            self.log(f"[{name}] resuming from checkpoint (iteration {resumed})")
        t0 = time.perf_counter()
        res = run_scf(self.ham, self.hart, V_ext, p, mu=mu, n_init=n_init, checkpoint=ckpt,
                      checkpoint_every=25, verbose=True)
        return res, time.perf_counter() - t0, resumed

    def unpol(self):
        """Unpolarised B = 0 reference (fixed mu of the B = 0 leads); unpol.npz / unpol.json."""
        if self.done("unpol"):
            return SCFResult.load_npz(self.path("unpol.npz"))
        lead, mu, V_ext = self.lead(0.0)
        if os.path.exists(self.path("unpol.npz")):
            res = SCFResult.load_npz(self.path("unpol.npz"))
            t, resumed = float("nan"), 0
        else:
            res, t, resumed = self._scf("unpol", self.params(0.0, polarized=False), mu, V_ext,
                                        (lead.n_up, lead.n_dn))
            res.save_npz(self.path("unpol.npz"))
        f = barrier_features(res, self.ham, far_nm=self.far_nm, units=self.U)
        info = dict(name="unpol", finished=True, converged=bool(res.converged), iterations=int(res.iterations),
                    time=t, N=float(res.N), mu_meV=self.U.au_to_meV(res.mu), mu_e0far=f["mu_e0far"][0],
                    top_unpol=max([h for _, h in f["peaks"][0]] + [f["centre"][0]]),
                    n1d_0=f["n1d_0"], peaks=f["peaks"][0], centre=f["centre"][0])
        jdump(self.path("unpol.json"), info)
        self.log(f"[unpol] converged {res.converged} in {res.iterations} it: top = {info['top_unpol']:.4f} meV, "
                 f"n_1D(0) = {f['n1d_0']:.6f}, N = {res.N:.4f}")
        return res

    def top_unpol(self):
        self.unpol()
        return float(json.load(open(self.path("unpol.json")))["top_unpol"])

    # ------------------------------------------------------------------ LDOS + summary
    def ldos_analysis(self, res, B, feat):
        """One extra full dense diagonalisation per spin (eigenpairs with e < mu + 1.5 meV), LDOS at
        x = 0 for eta = 0.05 and 0.1, resonance fit (Lorentzian + constant), Gamma, U."""
        U = self.U
        mu_meV = U.au_to_meV(res.mu)
        eigs, Xs = [], []
        for s in range(2):
            self.ham.set_potential(res.V[s])
            w, v = self.ham.eigh_dense()
            keep = U.au_to_meV(w) < mu_meV + 1.5
            eigs.append(w[keep]); Xs.append(v[:, keep])
        e0f = feat["e0_far"][0]
        E = np.arange(e0f - 0.3, mu_meV + 1.5 + 1e-9, 0.005)
        fake = type("R", (), {})()
        fake.eigs, fake.X = eigs, Xs
        out, arrays = {}, {"ldos_e_minus_e0far": E - e0f, "ldos_e_minus_mu": E - mu_meV}
        for eta in (0.05, 0.1):
            _, rho = ldos_1d(fake, self.ham, 0.0, E, eta_meV=eta, units=U)
            arrays[f"ldos_up_eta{eta}"] = rho[0]
            arrays[f"ldos_dn_eta{eta}"] = rho[1]
            r = dict(eta=eta)
            sel = (E >= e0f) & (E <= mu_meV)
            if sel.sum() > 3:
                i = np.nonzero(sel)[0][np.argmax(rho[0][sel])]
                pk, half = rho[0][i], 0.5 * rho[0][i]
                j1 = i
                while j1 > 0 and rho[0][j1] > half:
                    j1 -= 1
                j2 = i
                while j2 < E.size - 1 and rho[0][j2] > half:
                    j2 += 1
                w0 = max(E[j2] - E[j1], 2 * eta)
                win = np.abs(E - E[i]) <= 1.5 * w0
                try:
                    popt, _ = curve_fit(lorentz_c, E[win], rho[0][win], p0=[pk, E[i], w0, 0.0], maxfev=5000)
                    fwhm, e_res = abs(popt[2]), popt[1]
                except Exception:
                    fwhm, e_res = float("nan"), E[i]
                if not (e0f - 0.05 <= e_res <= mu_meV + 0.05) or not np.isfinite(fwhm) or fwhm > 4 * w0:
                    fwhm, e_res = w0, E[i]                   # fit failed / left the window: raw peak
                    r["fit"] = "fallback_halfmax"
                d = np.gradient(rho[1], E)
                sd = (E >= e_res) & (E <= mu_meV + 1.5)
                e_on = E[np.nonzero(sd)[0][np.argmax(d[sd])]] if sd.any() else float("nan")
                r.update(res_e_minus_mu=e_res - mu_meV, res_e_minus_e0far=e_res - e0f, fwhm_fit=fwhm,
                         Gamma=fwhm - 2 * eta, onset_dn_minus_mu=e_on - mu_meV, U=e_on - e_res,
                         peak_height=float(pk))
            out[f"eta{eta}"] = r
        a, b = out["eta0.05"], out["eta0.1"]
        for k in ("U", "Gamma", "res_e_minus_mu"):
            if k in a and k in b:
                out[f"{k}_mean"] = 0.5 * (a[k] + b[k])
                out[f"{k}_diff"] = a[k] - b[k]
        return out, arrays

    def summarise(self, name, B, res, t, resumed=0, ldos=True):
        """B<B>.json (and LDOS arrays into B<B>.npz)."""
        U, g = self.U, self.grid
        dA = g.dx * g.dy
        lead, mu, V_ext = self.lead(B)
        w = getattr(lead, "res", lead)
        win = np.abs(self.x_nm) < 300.0
        M = float((res.n_up - res.n_dn).sum() * dA)
        M_wire = float((w.n_up - w.n_dn).sum() * dA)
        M_loc = M - M_wire
        M_win = float((res.n_up - res.n_dn)[win].sum() * dA - (w.n_up - w.n_dn)[win].sum() * dA)
        f = barrier_features(res, self.ham, far_nm=self.far_nm, units=U)
        nu0, nd0 = f["n1d_0_s"]
        tu = self.top_unpol()
        top_dn = max([h for _, h in f["peaks"][1]] + [f["centre"][1]])
        info = dict(name=name, B=B, model=self.model, interp=self.interp, wx=self.wx, ecut=self.ecut, lx=self.lx,
                    kT=self.kT, tag=self.tag, finished=True, converged=bool(res.converged),
                    iterations=int(res.iterations), resumed_from_it=resumed, time=t,
                    mu_meV=U.au_to_meV(res.mu), mu_e0far=f["mu_e0far"][0], N=float(res.N), M=M, M_wire=M_wire,
                    M_loc=M_loc, M_win300=M_win, zeta0=(nu0 - nd0) / (nu0 + nd0) if nu0 + nd0 > 0 else 0.0,
                    n1d_0=f["n1d_0"], n1d_0_up=nu0, n1d_0_dn=nd0,
                    peaks_up=f["peaks"][0], peaks_dn=f["peaks"][1], e0_0_up=f["centre"][0],
                    e0_0_dn=f["centre"][1], top_dn=top_dn, top_unpol=tu, D_dn=top_dn - tu,
                    D_up0=f["centre"][0] - tu)
        if not res.converged and res.history:
            Ms = np.array([h[4] for h in res.history])[-100:] - M_wire
            info["M_loc_last100"] = dict(mean=float(Ms.mean()), std=float(Ms.std()),
                                         min=float(Ms.min()), max=float(Ms.max()), n=int(Ms.size))
        arrays = {}
        if ldos:
            try:
                info["ldos"], arrays = self.ldos_analysis(res, B, f)
            except Exception as exc:                 # never lose the state because of the analysis
                info["ldos_error"] = repr(exc)
        res.save_npz(self.path(f"{name}.npz"), extra=arrays, drop_X=not self.keep_X(B))
        jdump(self.path(f"{name}.json"), info)
        lt = info.get("ldos", {}).get("eta0.05", {})
        self.log(f"[{name}] conv {res.converged} it {res.iterations} t {t:.0f}s: M_loc {M_loc:.4f} M_win {M_win:.4f} "
                 f"zeta0 {info['zeta0']:.3f} n1d0 {f['n1d_0']:.5f} D_dn {info['D_dn']:+.3f} D_up0 {info['D_up0']:+.3f}"
                 + (f" | res {lt.get('res_e_minus_mu', float('nan')):+.3f} Gamma {lt.get('Gamma', float('nan')):.3f} "
                    f"U {lt.get('U', float('nan')):.3f}" if lt else "")
                 + ("" if res.converged else f" | last100 {info.get('M_loc_last100')}"))
        return info

    # ------------------------------------------------------------------ modes
    def ramp(self, B_list, start_from=None, up_from=None):
        self.unpol()
        prev = None
        if up_from is not None:
            prev = SCFResult.load_npz(self.path(f"B{bstr(up_from)}.npz"))
        elif start_from is not None:
            prev = SCFResult.load_npz(start_from)
        lead_prev = None
        for B in B_list:
            name = f"B{bstr(B)}"
            lead, mu, V_ext = self.lead(B, n_init=None if lead_prev is None else (lead_prev.n_up, lead_prev.n_dn))
            lead_prev = lead
            if self.done(name):
                self.log(f"[{name}] finished earlier; skipped")
                prev = SCFResult.load_npz(self.path(name + ".npz"))
                continue
            res, t, resumed = None, float("nan"), 0
            start = (lead.n_up, lead.n_dn) if prev is None else (prev.n_up, prev.n_dn)
            if os.path.exists(self.path(name + ".npz")) and not os.path.exists(self.path(name + "_ckpt.npz")):
                old = SCFResult.load_npz(self.path(name + ".npz"))      # e.g. migrated, no JSON yet
                if old.converged or old.iterations >= self.maxiter:
                    res = old
                    t = sum(h[5] for h in old.history) if old.history else float("nan")
                else:
                    start = (old.n_up, old.n_dn)
            if res is None:
                res, t, resumed = self._scf(name, self.params(B), mu, V_ext, start)
            self.summarise(name, B, res, t, resumed)
            ck = self.path(name + "_ckpt.npz")
            if os.path.exists(ck):
                os.remove(ck)
            prev = res

    def postprocess(self, ldos=True):
        """Re-create JSON + LDOS for every existing B<B>.npz (no SCF)."""
        self.unpol()
        for fn in sorted(os.listdir(self.dir)):
            if fn.startswith("B") and fn.endswith(".npz") and not fn.endswith("_ckpt.npz"):
                name = fn[:-4]
                try:
                    B = float(name[1:])
                except ValueError:
                    continue
                res = SCFResult.load_npz(self.path(fn))
                if res.V is None:
                    self.log(f"[{name}] no stored potentials; skipped")
                    continue
                t = sum(h[5] for h in res.history) if res.history else float("nan")
                self.summarise(name, B, res, t, ldos=ldos)

    def wire_check(self, B_list):
        for B in B_list:
            self.lead(B)

    def spin_gain(self, n_it=40, alpha=0.2):
        """Linear-response spin gain of the unpolarised B = 0 state (1 % seed for |x| < 300 nm, linear
        mixing, n_it iterations, no stop): spin_gain.json with ratios, lambda (last 5 two-step ratios), r_40."""
        unpol = self.unpol()
        lead, mu, V_ext = self.lead(0.0)
        X, _ = physical_xy(self.grid)
        near = np.abs(self.U.au_to_nm(X)) < 300.0
        n_up = np.where(near, unpol.n_up * 1.01, unpol.n_up)
        n_dn = np.where(near, unpol.n_dn * 0.99, unpol.n_dn)
        dA = self.grid.dx * self.grid.dy
        Ms = []
        p = self.params(0.0, alpha=alpha, history=1, maxiter=n_it)
        res = run_scf(self.ham, self.hart, V_ext, p, mu=mu, n_init=(n_up, n_dn), verbose=False, stop=False,
                      callback=lambda it, mi, mo, *a: Ms.append(mi))
        M = np.array(Ms + [float((res.n_up - res.n_dn).sum() * dA)])
        r = M[1:] / M[:-1]
        rho2 = np.sqrt(M[2:] / M[:-2])
        lam = 1 + (rho2[-5:].mean() - 1) / alpha
        info = dict(n_it=n_it, alpha=alpha, M=M.tolist(), r=r.tolist(), rho2=rho2.tolist(), lambda_2step=lam,
                    r_last=float(r[-1]), r_20=float(r[19]) if r.size >= 20 else None)
        jdump(self.path("spin_gain.json"), info)
        self.log(f"[spin_gain] {self.dir}: lambda = {lam:.4f}, r_{r.size} = {r[-1]:.4f}")
        return info

    def energies(self):
        """Omega_pol(B=0) - Omega_unpol (only if B0 has M_loc > 0.3)."""
        from qpc.energy import total_energy
        if not self.done("B0"):
            self.log("[energies] no B0 state; skipped")
            return None
        j = json.load(open(self.path("B0.json")))
        if j["M_loc"] <= 0.3:
            self.log(f"[energies] M_loc(B=0) = {j['M_loc']:.3f} <= 0.3; skipped")
            return None
        lead, mu, V_ext = self.lead(0.0)
        out = {}
        for name in ("B0", "unpol"):
            r = SCFResult.load_npz(self.path(name + ".npz"))
            if r.X[0] is None:
                self.log(f"[energies] {name} has no eigenvectors; skipped")
                return None
            E = total_energy(r, self.ham, self.hart, V_ext, r.params)
            out[name] = {k: self.U.au_to_meV(v) if k != "N" else v for k, v in E.items()}
        out["dOmega_meV"] = out["B0"]["Omega"] - out["unpol"]["Omega"]
        jdump(self.path("energies.json"), out)
        self.log(f"[energies] {self.dir}: Omega_pol - Omega_unpol = {out['dOmega_meV']:+.5f} meV")
        return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--interp", default="exchange", help="quadratic | exchange | power:p | mixed:w")
    ap.add_argument("--wx", type=float, required=True)
    ap.add_argument("--B", type=float, nargs="*", default=None)
    ap.add_argument("--up-from", type=float, default=None)
    ap.add_argument("--start-from", default=None)
    ap.add_argument("--ecut", type=float, default=15.0)
    ap.add_argument("--lx", type=float, default=5000.0)
    ap.add_argument("--kT", type=float, default=0.05)
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--unpol-only", action="store_true")
    ap.add_argument("--postprocess", action="store_true")
    ap.add_argument("--wire-check", type=float, nargs="*", default=None)
    ap.add_argument("--spin-gain", type=int, default=None, metavar="N_IT")
    ap.add_argument("--energies", action="store_true")
    a = ap.parse_args(argv)
    print(f"[run_qpc] {' '.join(sys.argv)}  ({time.ctime()})", flush=True)
    F = Folder(a.model, a.interp, a.wx, a.ecut, a.lx, a.kT, a.maxiter, a.tag,
               log=lambda m: print(m, flush=True))
    if a.wire_check is not None:
        return F.wire_check(a.wire_check)
    if a.unpol_only:
        return F.unpol()
    if a.postprocess:
        return F.postprocess()
    if a.spin_gain is not None:
        return F.spin_gain(a.spin_gain)
    if a.energies:
        return F.energies()
    if a.B:
        return F.ramp(a.B, start_from=a.start_from, up_from=a.up_from)
    ap.error("nothing to do (give --B, --unpol-only, --postprocess, --wire-check, --spin-gain or --energies)")


if __name__ == "__main__":
    main()
