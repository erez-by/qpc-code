"""Analysis of converged SCF states (docs/SPEC.md M5.2 / M6). Atomic units inside."""
import numpy as np

from .occupation import fermi
from .solver import XAveragedPreconditioner

_PAD = 1e5          # XAveragedPreconditioner pads its blocks with 1e6 on the diagonal


def transverse_levels(V, ham, n_levels=1, basis="grid"):
    """Lowest eigenvalues of the transverse operator h(x) = -(1/2) d^2/dy^2 + V(x, y) at every x.

    Plane waves in y: <G_y|h(x)|G_y'> = G_y^2/2 delta + V_y(x)[(iy - iy') mod Ny],
    V_y(x) = fft_y(V(x,.))/Ny.
      basis="grid" (default): all Ny plane waves of the y grid -- exact for the grid-sampled V
                     (bare QPC barrier top (V0/2 + (w_y + V0)/2 = 4.000 meV) reproduced to 1e-5).
      basis="cutoff": only the G_x = 0 plane waves with |G_y|^2/2 <= e_cut (the KS basis); 17
                     waves at E_cut = 15 meV, which overestimates the stiff x = 0 level (+0.013 meV
                     for the bare barrier).
    V: (Nx, Ny) Ha*. Returns (Nx, n_levels) Ha*.
    """
    g = ham.grid
    if basis == "grid":
        iy = np.arange(g.Ny)
        kin = 0.5 * g.G_axes()[1] ** 2
    elif basis == "cutoff":
        sel = ham.ix == 0
        iy = ham.iy[sel]
        kin = ham.kinetic[sel]
    else:
        raise ValueError(basis)
    Vy = np.fft.fft(np.asarray(V, dtype=float), axis=1) / g.Ny            # (Nx, Ny)
    H = Vy[:, (iy[:, None] - iy[None, :]) % g.Ny] + np.diag(kin)[None]    # (Nx, m, m)
    return np.linalg.eigvalsh(H)[:, :n_levels]


def barrier_profile(res, ham, basis="grid"):
    """Effective 1D barrier e_0,s(x) (PRL Fig. 1(a)-(c)): lowest eigenvalue of
    -(1/2) d_y^2 + V_eff,s(x, y) at each x, per spin (transverse_levels).
    Returns (Nx, 2) Ha*, in grid order (x_j = j dx; physical x = min_image)."""
    if res.V is None:
        raise ValueError("SCFResult has no stored potentials (V)")
    return np.stack([transverse_levels(res.V[s], ham, basis=basis)[:, 0] for s in range(2)], axis=1)


def n1d(n, grid):
    """Line density n_1D(x) = int n(x, y) dy  [a*^-1], shape (Nx,)."""
    return np.asarray(n).sum(axis=1) * grid.dy


def wire_subbands(ham, V, mu, kT):
    """Transverse subbands of an x-independent potential V (clean wire), one spin channel.

    For x-independent V the KS states are e^{i G_x x} phi_{n,G_x}(y); the G_x blocks of
    XAveragedPreconditioner give them exactly (column n = transverse index n, ascending).
    Returns (bottoms, N_sub): bottoms[n] = e_n(k_x = 0) [Ha*]; N_sub[n] = sum_{G_x} f(e_n(G_x))
    = electrons in subband n for this spin (multiply by 2 for an unpolarised wire).
    """
    ham.set_potential(V)
    P = XAveragedPreconditioner(ham)
    d = np.where(P.d > _PAD, np.inf, P.d)
    g0 = int(np.nonzero(np.unique(ham.ix) == 0)[0][0])
    return np.sort(d[g0]), fermi(d, mu, kT).sum(axis=0)


def net_spin(res, grid):
    """M = int (n_up - n_dn) dx dy  (electrons)."""
    return float((res.n_up - res.n_dn).sum()) * grid.dx * grid.dy


def net_spin_local(res, wire_res, grid):
    """Local moment of the QPC: M_loc = int (n_up - n_dn) dx dy - (N_up - N_dn of the clean wire at
    the same B, same cell). Removes the Zeeman polarisation of the leads; equals M at B = 0.
    This is the number to compare with HMW (0.85, 0.93, 0.90). wire_res: SCFResult or CleanWire."""
    w = getattr(wire_res, "res", wire_res)
    return net_spin(res, grid) - net_spin(w, grid)


def net_spin_window(res, wire_res, grid, x_half_nm=300.0, units=None):
    """Window moment M_win = int_{|x| < x_half} (n_up - n_dn) dx dy  minus the same window
    integral of the clean wire at the same B (removes the lead polarisation inside the window).
    x_half_nm in nm; physical x = min_image(j dx, Lx). wire_res: SCFResult, CleanWire or
    BareReference."""
    from .fourier import min_image
    from .units import Units
    U = units or Units()
    x = min_image(grid.real_axes()[0], grid.Lx)
    win = np.abs(U.au_to_nm(x)) < x_half_nm
    w = getattr(wire_res, "res", wire_res)
    dA = grid.dx * grid.dy
    return float(((res.n_up - res.n_dn)[win]).sum() * dA - ((w.n_up - w.n_dn)[win]).sum() * dA)


def barrier_features(res, ham, far_nm=1500.0, min_height_meV=0.1, units=None):
    """Barrier shape and centre densities of a converged state (all at grid points, no windows).

    Per spin s, from the barrier profile e_0,s(x) (barrier_profile, full y grid):
      e0_far[s]   mean of e_0,s over |x| > far_nm                         [meV]
      peaks[s]    list of (x_peak [nm], height [meV above e0_far]) of ALL local maxima with
                  height > min_height_meV (periodic neighbours; a flat top counts once)
      centre[s]   e_0,s(0) - e0_far[s]  (dip depth when there are two peaks)     [meV]
      mu_e0far[s] mu - e0_far[s]                                              [meV]
    and n1d_0 (total), n1d_0_s = (n_up(0), n_dn(0)): line densities at x = 0   [nm^-1].
    """
    from .fourier import min_image
    from .units import Units
    U = units or Units()
    g = ham.grid
    meV = U.au_to_meV
    x_nm = U.au_to_nm(min_image(g.real_axes()[0], g.Lx))
    far = np.abs(x_nm) > far_nm
    e0 = barrier_profile(res, ham)
    out = dict(e0_far=[], peaks=[], centre=[], mu_e0far=[])
    for s in range(2):
        e = meV(e0[:, s])
        ef = e[far].mean()
        h = e - ef
        left, right = np.roll(h, 1), np.roll(h, -1)
        is_max = (h > left) & (h >= right) & (h > min_height_meV)
        idx = np.nonzero(is_max)[0]
        idx = idx[np.argsort(x_nm[idx])]
        out["e0_far"].append(ef)
        out["peaks"].append([(float(x_nm[i]), float(h[i])) for i in idx])
        out["centre"].append(float(h[0]))
        out["mu_e0far"].append(float(meV(res.mu) - ef))
    l_up = n1d(res.n_up, g)[0] / U.length_nm
    l_dn = n1d(res.n_dn, g)[0] / U.length_nm
    out["n1d_0"] = float(l_up + l_dn)
    out["n1d_0_s"] = (float(l_up), float(l_dn))
    return out


def format_features(f, targets=None):
    """Multi-line text block of barrier_features, with optional HMW target comments."""
    t = targets or {}
    lines = []
    for s, name in enumerate(("up", "dn")):
        pk = ", ".join(f"x={x:+.1f} nm h={h:.4f}" for x, h in f["peaks"][s]) or "none > 0.1 meV"
        lines.append(f"  [{name}] peaks: {pk}" + (f"   # HMW: {t['peaks_' + name]}" if 'peaks_' + name in t else ""))
        lines.append(f"  [{name}] e_0(0)-e_0(far) = {f['centre'][s]:.4f} meV"
                     + (f"   # HMW: {t['centre_' + name]}" if 'centre_' + name in t else "")
                     + f";  mu - e_0(far) = {f['mu_e0far'][s]:.4f} meV"
                     + (f"   # HMW: {t['mu_e0far']}" if 'mu_e0far' in t else ""))
    lines.append(f"  n_1D(0) = {f['n1d_0']:.6f} nm^-1 (up {f['n1d_0_s'][0]:.6f}, dn {f['n1d_0_s'][1]:.6f})"
                 + (f"   # HMW: {t['n1d_0']}" if 'n1d_0' in t else ""))
    return "\n".join(lines)


def ldos_1d(res, ham, x0=0.0, energies_meV=None, eta_meV=0.05, shape="lorentz", units=None):
    """Local 1D density of states at x = x0, per spin (HMW Figs. 1-2 insets / Fig. 2):

        rho_s(x0, e) = sum_i [ int dy |psi_is(x0, y)|^2 ] L_eta(e - e_is)        [1/(meV nm)]

    with the physical orbital psi_is = ham.to_real_space(X_is) / sqrt(Lx Ly) (sum|c|^2 = 1), the
    x0 slice taken at the nearest grid point (x0 in nm, physical coordinate), the y integral as
    sum * dy. L_eta: Lorentzian (eta/pi)/(e^2 + eta^2) ("lorentz") or a normalised Gaussian of
    standard deviation eta ("gauss"). Uses the stored eigenpairs res.eigs[s], res.X[s].

    LIMIT: only the computed KS states enter; they are converged up to the band margin
    (~ mu + 15 kT = mu + 0.75 meV at kT = 0.05 meV, more if nb was larger). Energies above the
    highest stored eigenvalue minus a few eta are not meaningful. eta must not be smaller than
    the level spacing of the finite cell (~0.06 meV near mu for Lx = 5 um); default 0.05 meV.
    Returns (energies_meV, rho) with rho shape (2, n_e) in 1/(meV nm).
    """
    from .fourier import min_image
    from .units import Units
    U = units or Units()
    g = ham.grid
    x_nm = U.au_to_nm(min_image(g.real_axes()[0], g.Lx))
    ix = int(np.argmin(np.abs(x_nm - x0)))
    e = np.asarray(energies_meV, dtype=float)
    rho = np.zeros((2, e.size))
    for s in range(2):
        X = np.asarray(res.X[s])
        w = np.empty(X.shape[1])
        for start in range(0, X.shape[1], 32):
            psi = ham.to_real_space(X[:, start:start + 32])[ix]            # (Ny, k)
            w[start:start + 32] = (np.abs(psi) ** 2).sum(axis=0) * g.dy / (g.Lx * g.Ly)
        w = w / U.length_nm                                                 # a*^-1 -> nm^-1
        de = e[:, None] - U.au_to_meV(np.asarray(res.eigs[s]))[None, :]
        if shape == "lorentz":
            L = (eta_meV / np.pi) / (de ** 2 + eta_meV ** 2)
        elif shape == "gauss":
            L = np.exp(-de ** 2 / (2 * eta_meV ** 2)) / (np.sqrt(2 * np.pi) * eta_meV)
        else:
            raise ValueError(shape)
        rho[s] = L @ w
    return e, rho


def ldos_features(e, rho, mu_meV, e0_far_meV, lo_meV=0.1):
    """Spin-up resonance and spin-down onset from ldos_1d output (energies in meV).

    Resonance: maximum of rho_up for e in [e0_far + lo, mu]; FWHM from the half-maximum crossings
    around it (linear interpolation). U-onset: energy of max d rho_dn/de for e > e0_far.
    U = onset - resonance (HMW definition). Returns dict (energies relative to mu)."""
    sel = (e >= e0_far_meV + lo_meV) & (e <= mu_meV)
    i = np.nonzero(sel)[0][np.argmax(rho[0][sel])]
    peak = rho[0][i]
    half = 0.5 * peak

    def cross(direction):
        j = i
        while 0 < j < e.size - 1 and rho[0][j] > half:
            j += direction
        j0 = j - direction
        y0, y1 = rho[0][j0], rho[0][j]
        return e[j0] + (half - y0) * (e[j] - e[j0]) / (y1 - y0) if y1 != y0 else e[j]

    fwhm = cross(+1) - cross(-1)
    d = np.gradient(rho[1], e)
    sel_d = e > e0_far_meV
    k = np.nonzero(sel_d)[0][np.argmax(d[sel_d])]
    return dict(res_e=e[i] - mu_meV, res_height=peak, fwhm=fwhm, onset_dn=e[k] - mu_meV,
                U=e[k] - e[i])
