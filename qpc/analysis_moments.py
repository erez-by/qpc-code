"""Localized moments in a long QPC (docs/TWO_ELECTRON_SPEC.md, analysis module).

All lengths in nm, line densities in nm^-1, energies in meV at the interface (atomic units inside).
For a converged B = 0 state (SCFResult with n_up, n_dn, V, and eigenpairs if the LDOS is wanted):
  1. line profiles n(x) = int dy (n_up + n_dn), s(x) = int dy (n_up - n_dn);
  2. depleted region D = {x : n(x) < 0.9 n_lead} (the connected part containing the minimum of n), L_D;
  3. lobes of s(x): peaks of s and -s with prominence > 0.3e-2 nm^-1 (scipy.signal.find_peaks); inside D
     they are localized electrons (0: none, 1: single, 2 same sign: P, 2 opposite: AP), outside D they
     are Friedel oscillations (spacing ~ pi/k_F = 2/n_lead), reported but never counted;
  4. net spin int s dx and N_D = int_D n dx;
  5. eps_{0,s}(x): lowest eigenvalue of -(hbar^2/2m*) d^2/dy^2 + V_eff,s(x, y) by finite differences in y;
  6. LDOS at the lobe positions (qpc.analysis.ldos_1d), with its peaks.
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.signal import find_peaks

from .analysis import ldos_1d, n1d
from .fourier import min_image
from .potential import V_qpc
from .scf import external_potential
from .units import Units

PROMINENCE_NM = 0.3e-2          # lobe threshold on |s(x)| [nm^-1] (spec)
DEPLETION_FRAC = 0.9            # D = {n < 0.9 n_lead} (spec)
VALLEY = 0.5                    # PHYSICS-CHOICE #13: same-sign lobes are ONE moment unless s dips below 0.5 x peak
MIN_SPIN = 0.3                  # PHYSICS-CHOICE #13: a moment carries |int s dx| >= 0.3; weaker lobes = satellites


# ----------------------------------------------------------------------------- 1. profiles
def line_profiles(res, grid, units=Units()):
    """x [nm] ascending, n(x) and s(x) [nm^-1] (n1d of n_up +- n_dn), and the sort order of the grid."""
    x = units.au_to_nm(min_image(grid.real_axes()[0], grid.Lx))
    o = np.argsort(x)
    nu, nd = n1d(res.n_up, grid)[o], n1d(res.n_dn, grid)[o]
    return x[o], (nu + nd) / units.length_nm, (nu - nd) / units.length_nm, o


# ----------------------------------------------------------------------------- 2. depleted region
def depleted_region(x, n, n_lead, frac=DEPLETION_FRAC):
    """Mask of D = {x : n(x) < frac n_lead}, restricted to the connected interval that contains the
    minimum of n (other dips, e.g. Friedel minima far out, are not part of the constriction).
    Returns (mask, L_D [nm], (x_left, x_right)); L_D = 0 if n never drops below frac n_lead."""
    below = n < frac * n_lead
    mask = np.zeros_like(below)
    i0 = int(np.argmin(n))
    if not below[i0]:
        return mask, 0.0, (np.nan, np.nan)
    lo = i0
    while lo > 0 and below[lo - 1]:
        lo -= 1
    hi = i0
    while hi < n.size - 1 and below[hi + 1]:
        hi += 1
    mask[lo:hi + 1] = True
    dx = x[1] - x[0]
    return mask, float((hi - lo + 1) * dx), (float(x[lo]), float(x[hi]))


# ----------------------------------------------------------------------------- 3. lobes
def find_lobes(x, s, D, prominence=PROMINENCE_NM):
    """Peaks of s and of -s with the given prominence [nm^-1] whose own height sign * s(x0) is at least
    half of it (a flat s = 0 stretch, or a tiny lobe, between two opposite-sign lobes has a large
    prominence but is not a lobe). Each lobe: position x0 [nm], sign, peak value s(x0), its interval and
    the integrated spin over it (int s dx, electrons), and whether x0 lies in D. Sorted by position.

    Interval: between the neighbouring zeros of s (spec); two adjacent SAME-sign lobes (P case, no zero
    in between) are split at the minimum of |s| between them, so their spins are not double counted."""
    dx = x[1] - x[0]
    out = []
    for sign in (1, -1):
        v = sign * s
        idx, _ = find_peaks(v, prominence=prominence, height=0.5 * prominence)
        for k, i in enumerate(idx):
            lo = i
            while lo > 0 and v[lo - 1] > 0:
                lo -= 1
            hi = i
            while hi < s.size - 1 and v[hi + 1] > 0:
                hi += 1
            half = 0.0                                          # the shared saddle point counts half to each side
            if k > 0 and idx[k - 1] >= lo:                      # same-sign neighbour without a zero between
                lo = idx[k - 1] + int(np.argmin(v[idx[k - 1]:i + 1]))
                half += 0.5 * s[lo]
            if k < len(idx) - 1 and idx[k + 1] <= hi:
                hi = i + int(np.argmin(v[i:idx[k + 1] + 1]))
                half += 0.5 * s[hi]
            out.append(dict(x=float(x[i]), sign=int(sign), peak=float(s[i]), x_lo=float(x[lo]),
                            x_hi=float(x[hi]), width=float((hi - lo + 1) * dx),
                            spin=float((s[lo:hi + 1].sum() - half) * dx), in_D=bool(D[i]), n_merged=1))
    return sorted(out, key=lambda l: l["x"])


def merge_lobes(lobes, x, s, valley=VALLEY):
    """Merge adjacent same-sign lobes into one moment when sign * s between them never drops below
    valley x the smaller peak (a single moment with a central dip, e.g. the PRL Fig. 1(a) spin-up density
    with two peaks); spins add, the position is the spin-weighted centre. Returns a new sorted list."""
    out = []
    for l in sorted(lobes, key=lambda l: l["x"]):
        if out and out[-1]["sign"] == l["sign"]:
            m = out[-1]
            sel = (x >= m["x"]) & (x <= l["x"])
            floor = float(np.min(l["sign"] * s[sel])) if sel.any() else 0.0
            if floor >= valley * min(abs(m["peak"]), abs(l["peak"])):
                sp = m["spin"] + l["spin"]
                out[-1] = dict(m, x=float((m["x"] * m["spin"] + l["x"] * l["spin"]) / sp) if sp else m["x"],
                               peak=m["peak"] if abs(m["peak"]) >= abs(l["peak"]) else l["peak"],
                               x_hi=l["x_hi"], width=l["x_hi"] - m["x_lo"] + (x[1] - x[0]), spin=sp,
                               n_merged=m.get("n_merged", 1) + l.get("n_merged", 1), dip=floor)
                continue
        out.append(dict(l))
    return out


def moments(lobes_D, x, s, valley=VALLEY, min_spin=MIN_SPIN):
    """(moments, satellites): merged lobes in D with |spin| >= min_spin, and the weaker ones."""
    m = merge_lobes(lobes_D, x, s, valley)
    return [l for l in m if abs(l["spin"]) >= min_spin], [l for l in m if abs(l["spin"]) < min_spin]


def classify(lobes_D):
    """'none' (0 lobes in D), 'single' (1), 'P' (2, same sign), 'AP' (2, opposite), 'multi<k>' (k > 2)."""
    k = len(lobes_D)
    if k == 0:
        return "none"
    if k == 1:
        return "single"
    if k == 2:
        return "P" if lobes_D[0]["sign"] == lobes_D[1]["sign"] else "AP"
    return f"multi{k}"


def is_broad(lobes_D, L_D, frac=0.75, spin_max=1.5):
    """PHYSICS-CHOICE (docs/OPEN_QUESTIONS.md #13): a single moment is 'broad' (room for two overlapping
    electrons; kT = 0.02 rerun candidate) when its width between the zeros of s exceeds 0.75 L_D or its
    integrated |spin| exceeds 1.5. (At wx 1.0, V0 3.0 the known single moment has width 0.56 L_D, spin 1.30.)"""
    if len(lobes_D) != 1 or L_D <= 0:
        return False
    l = lobes_D[0]
    return bool(l["width"] > frac * L_D or abs(l["spin"]) > spin_max)


def friedel_spacing(lobes_out):
    """Mean distance between consecutive same-sign lobes outside D on each side [nm] (nan if < 2)."""
    d = []
    for side in (-1, 1):
        for sign in (-1, 1):
            xs = sorted(l["x"] for l in lobes_out if l["sign"] == sign and np.sign(l["x"]) == side)
            d += list(np.diff(xs))
    return float(np.mean(d)) if d else float("nan")


# ----------------------------------------------------------------------------- 3b. peaks of n(x)
N_PROMINENCE_NM = 0.1e-2        # PHYSICS-CHOICE #14: n(x) peak threshold inside D [nm^-1] ("strong" peaks)
N_PROMINENCE_WEAK_NM = 0.05e-2  # weaker peaks, reported separately


def density_peaks(x, n, D, prominence=N_PROMINENCE_NM):
    """Peaks of the line density n(x) inside D (localized electrons as density maxima, Wigner-like):
    scipy find_peaks on n restricted to D with the given prominence [nm^-1]. Returns (positions [nm],
    prominences [nm^-1], spacings between neighbouring peaks [nm])."""
    i = np.nonzero(D)[0]
    if i.size < 3:
        return [], [], []
    k, pr = find_peaks(n[i], prominence=prominence)
    xs = [float(x[i][j]) for j in k]
    return xs, [float(p) for p in pr["prominences"]], [float(d) for d in np.diff(xs)]


# ----------------------------------------------------------------------------- 5. eps_0,s(x)
def _fourier_y(Vs, grid, y_au):
    """Trigonometric interpolation in y of the periodic grid function Vs (Nx, Ny) at the points y_au."""
    G = grid.G_axes()[1]                                    # grid points y_j = j dy, so phases exp(i G y)
    F = np.fft.fft(Vs, axis=1) / grid.Ny
    return np.real(F @ np.exp(1j * G[None, :] * np.asarray(y_au)[:, None]).T)


def eps0_fd(V, grid, qpc_params, dy_nm=0.5, units=Units(), n_levels=1):
    """Lowest transverse eigenvalues of -(1/2) d^2/dy^2 + V(x, y) at every grid x [Ha*, (Nx, n_levels),
    grid order], by second-order finite differences on a fine y grid (spacing <= dy_nm) with hard walls
    at y = +-Ly/2 (scipy eigh_tridiagonal).

    V on the fine grid = V_ext(x, y) evaluated analytically (parabola + V_QPC, model A/C) + the
    trigonometric (exact Fourier-series) interpolation in y of the smooth remainder V - V_ext
    (V_H + v_xc + Zeeman). The y^2 confinement is not band-limited (kink at the periodic y boundary),
    so it is never Fourier-interpolated itself. V = None: bare V_ext only."""
    U = units
    M = int(np.ceil(U.au_to_nm(grid.Ly) / dy_nm))
    h = grid.Ly / M
    y = -0.5 * grid.Ly + h * np.arange(1, M)                                    # interior points
    x = min_image(grid.real_axes()[0], grid.Lx)
    Vext_f = 0.5 * qpc_params.wy ** 2 * y[None, :] ** 2 + V_qpc(x[:, None], y[None, :], qpc_params)
    if V is not None:
        rest = np.asarray(V) - external_potential(grid, qpc_params, include_qpc=True)
        Vext_f = Vext_f + _fourier_y(rest, grid, y)
    off = np.full(M - 2, -0.5 / h ** 2)
    out = np.empty((grid.Nx, n_levels))
    for j in range(grid.Nx):
        out[j] = eigh_tridiagonal(1.0 / h ** 2 + Vext_f[j], off, select="i",
                                  select_range=(0, n_levels - 1), eigvals_only=True)
    return out


# ----------------------------------------------------------------------------- 6. LDOS at the lobes
def ldos_at(res, ham, x_list_nm, mu_meV, e_lo_meV, eta_meV=0.05, e_hi_above_mu=1.0, units=Units(),
            rel_prominence=0.1):
    """LDOS rho_s(x0, e) at each x0 (one full dense diagonalisation per spin, eigenpairs with
    e < mu + e_hi_above_mu + 0.5 meV) on e in [e_lo, mu + e_hi_above_mu]; peaks per spin with
    prominence > rel_prominence * max. Returns (E, {x0: rho (2, nE)}, {x0: [[peaks up - mu], [dn - mu]]})."""
    U = units
    eigs, Xs = [], []
    for s in range(2):
        ham.set_potential(res.V[s])
        w, v = ham.eigh_dense()
        keep = U.au_to_meV(w) < mu_meV + e_hi_above_mu + 0.5
        eigs.append(w[keep]); Xs.append(v[:, keep])
    fake = type("R", (), {})()
    fake.eigs, fake.X = eigs, Xs
    E = np.arange(e_lo_meV, mu_meV + e_hi_above_mu + 1e-9, 0.005)
    rhos, peaks = {}, {}
    for x0 in x_list_nm:
        _, rho = ldos_1d(fake, ham, x0, E, eta_meV=eta_meV, units=U)
        rhos[x0] = rho
        pk = []
        for s in range(2):
            idx, _ = find_peaks(rho[s], prominence=rel_prominence * max(rho[s].max(), 1e-30))
            pk.append([float(E[i] - mu_meV) for i in idx])
        peaks[x0] = pk
    return E, rhos, peaks


# ----------------------------------------------------------------------------- all of it
def analyse(res, ham, qpc_params, n_lead_nm, units=Units(), ldos=True, eta_meV=0.05):
    """Steps 1-6 for one state. n_lead_nm: clean-wire line density [nm^-1]. Returns (info, arrays):
    info is JSON-able, arrays holds the profiles, eps_0,s(x) and the LDOS curves."""
    U, g = units, ham.grid
    x, n, s, o = line_profiles(res, g, U)
    D, L_D, (xl, xr) = depleted_region(x, n, n_lead_nm)
    dx = x[1] - x[0]
    lobes = find_lobes(x, s, D)
    raw = [l for l in lobes if l["in_D"]]
    out_D = [l for l in lobes if not l["in_D"]]
    in_D, sats = moments(raw, x, s)
    i0 = int(np.argmin(np.abs(x)))
    mu = U.au_to_meV(res.mu)
    info = dict(L_D=L_D, D_edges=[xl, xr], N_D=float(n[D].sum() * dx), net_spin=float(s.sum() * dx),
                n_lead=float(n_lead_nm), n_center=float(n[i0]), LD_estimate=float(2.0 / n[i0]) if n[i0] > 0 else None,
                n_lobes=len(in_D), cls=classify(in_D), broad=is_broad(in_D, L_D),
                lobe_x=[l["x"] for l in in_D], lobe_signs=[l["sign"] for l in in_D],
                lobe_spins=[l["spin"] for l in in_D], lobes=in_D, satellites=sats,
                n_lobes_raw=len(raw), cls_raw=classify(raw), lobes_raw=raw, friedel_lobes=out_D,
                friedel_spacing=friedel_spacing(out_D), friedel_estimate=float(2.0 / n_lead_nm),
                mu_meV=mu, converged=bool(res.converged), iterations=int(res.iterations))
    arrays = dict(x_nm=x, n_x=n, s_x=s, D_mask=D)
    if res.V is not None:
        e0 = np.stack([U.au_to_meV(eps0_fd(res.V[k], g, qpc_params, units=U)[:, 0])[o] for k in range(2)])
        arrays["eps0"] = e0
        far = np.abs(x) > 0.3 * U.au_to_nm(g.Lx)
        info["eps0_far"] = float(e0[:, far].mean())
        info["eps0_max"] = [float(e0[k].max()) for k in range(2)]
        info["eps0_max_minus_mu"] = [float(e0[k].max() - mu) for k in range(2)]
        if ldos:
            xs = [l["x"] for l in in_D] or [0.0]
            E, rhos, peaks = ldos_at(res, ham, xs, mu, info["eps0_far"] - 0.3, eta_meV=eta_meV, units=U)
            arrays["ldos_e_meV"] = E
            for k, x0 in enumerate(xs):
                arrays[f"ldos_{k}"] = rhos[x0]
            info["ldos"] = [dict(x=x0, peaks_up_minus_mu=peaks[x0][0], peaks_dn_minus_mu=peaks[x0][1],
                                 n_peaks_below_mu=[sum(p < 0 for p in peaks[x0][k]) for k in range(2)])
                            for x0 in xs]
    return info, arrays


CSV_COLS = ["hbar_wx", "V0", "lx", "kT", "branch", "L_D", "N_D", "net_spin", "n_lobes", "cls", "lobe_signs",
            "lobe_spins", "converged"]


def csv_row(info, hbar_wx, V0, lx, kT, branch="uniform"):
    """One summary.csv line (spec columns + cell, kT and branch)."""
    v = dict(hbar_wx=hbar_wx, V0=V0, lx=lx, kT=kT, branch=branch, L_D=f"{info['L_D']:.1f}",
             N_D=f"{info['N_D']:.4f}", net_spin=f"{info['net_spin']:.4f}", n_lobes=info["n_lobes"], cls=info["cls"],
             lobe_signs=" ".join(f"{s:+d}" for s in info["lobe_signs"]),
             lobe_spins=" ".join(f"{s:+.3f}" for s in info["lobe_spins"]), converged=info["converged"])
    return ",".join(str(v[c]) for c in CSV_COLS)
