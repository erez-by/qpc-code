"""Transmission from the eigenstates of the periodic cell ("ring") by even/odd phase shifts (Meir's method;
docs/RING_TRANSMISSION_SPEC.md).

A mirror-symmetric single-channel scatterer has standing-wave eigenstates of definite x-parity. In the leads
    psi_e ~ cos(k|x| + d_e),   psi_o ~ sign(x) sin(k|x| + d_o),
and the scattering amplitudes are t = (e^{2 i d_e} + e^{2 i d_o})/2, r = (e^{2 i d_e} - e^{2 i d_o})/2, so
    T(E) = |t|^2 = cos^2(d_e(E) - d_o(E)).
Ring quantisation (psi_e' = 0, psi_o = 0 at x = +-Lx/2) gives k_n Lx/2 + d(E_n) = n pi for both parities.
Two ways to get the phases of the ring levels:
  M1 (level_phase): d_n = n pi - k(E_n) Lx/2, k(E) from the exact lead dispersion, n = level count;
  M2 (lead_phase):  fit of the channel-1 wavefunction psi_1(x) = int dy phi_1(y) psi(x, y) to C cos(k|x| + d)
                    (even) or C sign(x) sin(k|x| + d) (odd) in a lead window x1 <= |x| <= x2.
Each level gives one phase; d_e(E) and d_o(E) are interpolated (PCHIP on the unwrapped phases) to a common
energy grid before T = cos^2(d_e - d_o) is formed.

Single channel: the method needs one open channel of the symmetry sector. Since V(x, y) = V(x, -y) exactly,
y-odd modes never mix with y-even ones; the y-even sector has one channel up to E3 (the second y-EVEN lead
subband), above the spec's E2 (the first y-odd one). Both edges are reported.
Units: meV and nm at the interface; the 2D Hamiltonian works in Ha*, a* (qpc.units).
"""
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.optimize import minimize_scalar

from .fourier import min_image
from .solver import XAveragedPreconditioner
from .units import Units

U = Units()


# ----------------------------------------------------------------------------- phases from 1D data
def fit_phase(x, psi, k, parity):
    """Least-squares phase of a standing wave on x > 0 at fixed k:
        even: psi = C cos(k x + d) = a cos kx + b sin kx  ->  d = atan2(-b, a)
        odd:  psi = C sin(k x + d) = a cos kx + b sin kx  ->  d = atan2(a, b)
    Returns (d mod 2 pi, C, relative rms residual)."""
    A = np.column_stack([np.cos(k * x), np.sin(k * x)])
    (a, b), *_ = np.linalg.lstsq(A, psi, rcond=None)
    d = np.arctan2(-b, a) if parity > 0 else np.arctan2(a, b)
    res = psi - A @ np.array([a, b])
    return float(d), float(np.hypot(a, b)), float(np.sqrt(np.mean(res ** 2)) / max(np.sqrt(np.mean(psi ** 2)), 1e-300))


def lead_phase(x, psi, k_guess, parity, x1, x2, fit_k=True, rel_k=0.05):
    """Method M2. psi: real channel wavefunction on the physical grid x (any order, contains +-x pairs).
    Fits on the x > 0 window [x1, x2] (by parity the left side is identical). fit_k: k is a free parameter
    (bounded to k_guess (1 +- rel_k)) minimising the residual; otherwise k = k_guess.
    Returns dict(d, k, C, resid, d_fixed_k, resid_fixed_k)."""
    sel = (x >= x1) & (x <= x2)
    xs, ps = x[sel], psi[sel]
    d0, C0, r0 = fit_phase(xs, ps, k_guess, parity)
    out = dict(d=d0, k=float(k_guess), C=C0, resid=r0, d_fixed_k=d0, resid_fixed_k=r0)
    if fit_k:
        m = minimize_scalar(lambda kk: fit_phase(xs, ps, kk, parity)[2], bounds=(k_guess * (1 - rel_k),
                            k_guess * (1 + rel_k)), method="bounded", options=dict(xatol=1e-10 * k_guess))
        d, C, r = fit_phase(xs, ps, m.x, parity)
        out.update(d=d, k=float(m.x), C=C, resid=r)
    return out


def chiral_flux(psi, dpsi, k):
    """a = (psi + psi'/(ik))/2, b = (psi - psi'/(ik))/2 and the flux |a|^2 - |b|^2 (= 0 for a standing wave)."""
    a = 0.5 * (psi + dpsi / (1j * k))
    b = 0.5 * (psi - dpsi / (1j * k))
    return a, b, np.abs(a) ** 2 - np.abs(b) ** 2


def spectral_derivative(psi, L):
    """d psi / dx of a periodic sample (uniform grid in FFT order, period L) by FFT."""
    G = 2 * np.pi * np.fft.fftfreq(psi.size, d=L / psi.size)
    return np.fft.ifft(1j * G * np.fft.fft(psi))


def level_phase(E, k_of_E, L):
    """Method M1: d_n = n pi - k(E_n) L/2 for the levels E (one parity, ascending, consecutive levels of the
    scattering continuum); n = 0, 1, 2, ... (the offset is a multiple of pi, irrelevant for T). Unwrapped."""
    E = np.asarray(E, float)
    return np.pi * np.arange(E.size) - k_of_E(E) * L / 2.0


def unwrap_pi(d):
    """Unwrap a phase defined modulo pi along its index (np.unwrap of 2d, halved)."""
    return np.unwrap(2.0 * np.asarray(d, float)) / 2.0


def transmission(E_grid, E_e, d_e, E_o, d_o, interp="pchip"):
    """T(E) = cos^2(d_e(E) - d_o(E)) on E_grid, with d_e, d_o interpolated (PCHIP or cubic) from the levels.
    Points outside the common range of even and odd levels are NaN."""
    I = PchipInterpolator if interp == "pchip" else CubicSpline
    fe, fo = I(E_e, d_e), I(E_o, d_o)
    E_grid = np.asarray(E_grid, float)
    T = np.cos(fe(E_grid) - fo(E_grid)) ** 2
    lo, hi = max(E_e[0], E_o[0]), min(E_e[-1], E_o[-1])
    T[(E_grid < lo) | (E_grid > hi)] = np.nan
    return T


def fermi_average(E, T, mu, kT, n_kT=2.0):
    """<T> = int T(E)(-df/dE) dE / int (-df/dE) dE over |E - mu| <= n_kT kT (NaN-free part)."""
    E = np.asarray(E, float)
    sel = (np.abs(E - mu) <= n_kT * kT) & np.isfinite(T)
    w = 1.0 / (4 * kT * np.cosh((E[sel] - mu) / (2 * kT)) ** 2)
    return float(np.trapezoid(w * T[sel], E[sel]) / np.trapezoid(w, E[sel]))


def parity_of(psi, mirror):
    """<psi|P|psi>/<psi|psi> with P psi = psi[mirror] (mirror: index map x -> -x along axis 0)."""
    return float(np.real(np.vdot(psi, psi[mirror])) / np.real(np.vdot(psi, psi)))


def split_parity(E, Psi, mirror, tol=1e-9):
    """Make degenerate (|dE| < tol) groups parity eigenstates: diagonalise P within each group.
    Psi: (..., n) states along the last axis (spatial axis 0 is x). Returns (E, Psi, parities)."""
    Psi = np.array(Psi, copy=True)
    n = E.size
    i = 0
    while i < n:
        j = i + 1
        while j < n and E[j] - E[j - 1] < tol:
            j += 1
        if j - i > 1:
            B = Psi[..., i:j].reshape(-1, j - i)
            Pm = np.array([[np.vdot(B[:, a], Psi[..., i + b][mirror].reshape(-1)) for b in range(j - i)]
                           for a in range(j - i)])
            w, v = np.linalg.eigh(0.5 * (Pm + Pm.conj().T))
            Psi[..., i:j] = (B @ v).reshape(Psi[..., i:j].shape)
        i = j
    par = np.array([parity_of(Psi[..., m], mirror) for m in range(n)])
    return E, Psi, par


def real_phase(psi):
    """Global phase that makes a standing wave real: psi e^{-i theta}, theta = arg(sum psi^2)/2."""
    th = 0.5 * np.angle(np.sum(psi ** 2))
    return np.real(psi * np.exp(-1j * th)), float(np.max(np.abs(np.imag(psi * np.exp(-1j * th)))))


def check_mirror(V, axis=0, tol=1e-9):
    """Raise ValueError unless V is mirror symmetric along axis (x -> -x on the periodic grid) to tol
    (absolute, units of V): the even/odd method is invalid for an asymmetric scatterer."""
    V = np.asarray(V)
    m = (-np.arange(V.shape[axis])) % V.shape[axis]
    dev = float(np.max(np.abs(V - np.take(V, m, axis=axis))))
    if dev > tol:
        raise ValueError(f"potential not mirror symmetric (max |V(x) - V(-x)| = {dev:.3e} > {tol:.1e}): "
                         "the even/odd phase-shift method does not apply")
    return dev


# ----------------------------------------------------------------------------- 1D ring (tests, checks)
def ring_1d(V, L, hb2_2m=0.5):
    """Eigenpairs of -hb2_2m d^2/dx^2 + V(x) on a periodic grid of N points (period L) with the exact
    spectral (FFT) kinetic matrix. x_j = j L/N in FFT order (x = 0 at index 0, mirror j -> -j)."""
    N = V.size
    G = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    F = np.fft.fft(np.eye(N), axis=0)
    K = np.real(np.fft.ifft(hb2_2m * G[:, None] ** 2 * F, axis=0))
    H = 0.5 * (K + K.T) + np.diag(V)
    return np.linalg.eigh(H)


def ring_T_1d(V, L, E_grid, x1, x2, hb2_2m=0.5, e_lead=0.0, method="M1"):
    """Full method on a 1D ring (tests): levels, parities, phases (M1 or M2), T on E_grid."""
    N = V.size
    check_mirror(V, tol=1e-9 * max(1.0, float(np.max(np.abs(V)))))
    x = min_image(np.arange(N) * L / N, L)
    mirror = (-np.arange(N)) % N
    E, Psi = ring_1d(V, L, hb2_2m)
    sel = E > e_lead + 1e-9
    E, Psi = E[sel], Psi[:, sel]
    E, Psi, par = split_parity(E, Psi, mirror)
    k_of_E = lambda e: np.sqrt((np.asarray(e) - e_lead) / hb2_2m)          # noqa: E731
    out = {}
    for p, key in ((1, "e"), (-1, "o")):
        m = par * p > 0.99
        Ep = E[m]
        if method == "M1":
            d = level_phase(Ep, k_of_E, L)
        else:
            d = unwrap_pi([lead_phase(x, real_phase(Psi[:, j])[0], k_of_E(Ep[q]), p, x1, x2)["d"]
                           for q, j in enumerate(np.nonzero(m)[0])])
        out[key] = (Ep, d)
    return transmission(E_grid, *out["e"], *out["o"]), out


# ----------------------------------------------------------------------------- 2D cell
def lead_modes(ham, V_col):
    """Transverse lead modes of the clean-wire column V_col(y) (Ny, Ha*) in the full y plane-wave basis of
    the grid: eigenvalues (Ha*) and real-space modes phi_n(y_j) (sum_j phi^2 = 1, real, y-parity sign)."""
    g = ham.grid
    G = g.G_axes()[1]
    Vy = np.fft.fft(V_col) / g.Ny
    iy = np.arange(g.Ny)
    H = Vy[(iy[:, None] - iy[None, :]) % g.Ny] + np.diag(0.5 * G ** 2)
    w, c = np.linalg.eigh(H)
    y = g.real_axes()[1]
    phi = np.exp(1j * y[:, None] * G[None, :]) @ c                     # (Ny, n)
    phi = np.array([real_phase(phi[:, n])[0] for n in range(phi.shape[1])]).T
    phi /= np.sqrt(np.sum(phi ** 2, axis=0))[None, :]
    return w, phi


def lead_dispersion(ham, V_col, sector_level=0):
    """Exact lead dispersion E_n(k) of the KS Hamiltonian (plane-wave cutoff circle included): the clean wire
    V(x, y) = V_col(y) is block diagonal in G_x; level number sector_level (0: lowest, y-even channel 1;
    2: next y-even) of each block at k = |G_x|. Returns (k [1/a*], E [Ha*]) for k >= 0, ascending in k, and a
    callable k_of_E (inverse by a monotone spline)."""
    g = ham.grid
    ham.set_potential(np.tile(np.asarray(V_col, float), (g.Nx, 1)))
    P = XAveragedPreconditioner(ham)
    Gx = g.G_axes()[0]
    cols = np.unique(ham.ix)
    k = np.abs(Gx[cols])
    E = np.sort(P.d, axis=1)[:, sector_level]
    o = np.argsort(k)
    k, E = k[o], E[o]
    keep = np.concatenate([[True], np.diff(k) > 1e-12])
    k, E = k[keep], E[keep]
    inv = PchipInterpolator(E, k)
    return k, E, (lambda e: inv(np.clip(np.asarray(e, float), E[0], E[-1])))


def ring_states(V, ham, E_lo, E_hi):
    """All eigenpairs of T + V (V: (Nx, Ny) Ha*, frozen) with E_lo < E < E_hi [Ha*], by dense diagonalisation.
    Returns (E [Ha*], Psi (Nx, Ny, n) complex real-space orbitals, sum |c|^2 = 1)."""
    ham.set_potential(V)
    w, c = ham.eigh_dense()
    sel = (w > E_lo) & (w < E_hi)
    return w[sel], ham.to_real_space(c[:, sel])


def classify_2d(E, Psi, phi1, grid, lead_win_nm, units=U, tol=1e-9):
    """x-parity (degenerate groups rotated), y-parity and channel-1 overlap of the 2D ring states.
    overlap = sum_{x in lead window} |sum_y phi1 psi|^2 / sum_{x in lead window} sum_y |psi|^2.
    Returns dict(E, Psi, px, py, overlap, psi1 (Nx, n) real channel-1 wavefunction)."""
    Nx, Ny = grid.Nx, grid.Ny
    mx = (-np.arange(Nx)) % Nx
    my = (-np.arange(Ny)) % Ny
    E, Psi, px = split_parity(E, Psi, mx, tol=tol)
    py = np.array([float(np.real(np.vdot(Psi[..., m], Psi[:, my, m])) / np.real(np.vdot(Psi[..., m], Psi[..., m])))
                   for m in range(E.size)])
    x = units.au_to_nm(min_image(grid.real_axes()[0], grid.Lx))
    win = (np.abs(x) >= lead_win_nm[0]) & (np.abs(x) <= lead_win_nm[1])
    proj = np.einsum("y,xyn->xn", phi1, Psi)
    ov = np.sum(np.abs(proj[win]) ** 2, axis=0) / np.sum(np.abs(Psi[win]) ** 2, axis=(0, 1))
    psi1 = np.empty((Nx, E.size))
    imag = np.empty(E.size)
    for m in range(E.size):
        psi1[:, m], imag[m] = real_phase(proj[:, m])
    return dict(E=E, Psi=Psi, px=px, py=py, overlap=ov, psi1=psi1, imag_resid=imag, x_nm=x)


WINDOWS_NM = [(400.0, 1200.0), (600.0, 1500.0), (800.0, 2000.0), (1000.0, 2400.0)]


def ring_transmission_2d(V, ham, E_grid_meV, lead_win_nm=(600.0, 1500.0), windows_nm=WINDOWS_NM, top="E3",
                         margin_meV=0.02, ov_min=0.95, e_min_meV=0.05, units=U):
    """Full method for one spin channel of the 2D cell (frozen V (Nx, Ny) Ha*):
      1. mirror checks in x and y (tol 1e-9 meV);
      2. lead: column at the cell edge; band edges E1 (lowest, y-even), E2 (y-odd), E3 (2nd y-even) from the
         exact lead dispersion (lead_dispersion); k1(E) of channel 1; lead mode phi_1(y);
      3. ring states with E1 < E < top - margin (top = E3 by y-parity decoupling, or E2 as in the spec);
      4. classification: keep |<P_x>| > 0.99, <P_y> > 0.99, channel-1 overlap > ov_min; list the dropped ones;
      5. phases: M1 (levels) and M2 (lead fit, free k, default window) per parity; M2 for all windows;
      6. T(E) = cos^2(d_e - d_o) on E_grid_meV (M1 and M2), plus the M2 window scatter; NaN for
         E < E1 + e_min_meV: near the band bottom (k -> 0) the phases of the first levels are ill-defined and the
         interpolation fails (Phase 0: errors up to 4e-2 at E1 + 0.002 meV, < 4e-3 above E1 + 0.05 meV).
    Returns a dict (energies in meV, k in 1/nm)."""
    g = ham.grid
    meV = units.au_to_meV
    check_mirror(V, 0, tol=units.meV_to_au(1e-9))
    check_mirror(V, 1, tol=units.meV_to_au(1e-9))
    Nx = g.Nx
    i_edge = int(np.argmax(np.abs(min_image(g.real_axes()[0], g.Lx))))
    V_col = np.asarray(V)[i_edge]
    kd, Ed, k1 = lead_dispersion(ham, V_col, 0)
    E1 = float(Ed[0])
    _, E2d, _ = lead_dispersion(ham, V_col, 1)
    _, E3d, _ = lead_dispersion(ham, V_col, 2)
    E2, E3 = float(E2d.min()), float(E3d.min())
    _, phi = lead_modes(ham, V_col)
    phi1 = phi[:, 0]
    E_top = (E3 if top == "E3" else E2) - units.meV_to_au(margin_meV)
    Es, Psi = ring_states(V, ham, E1 + 1e-12, E_top)
    c = classify_2d(Es, Psi, phi1, g, lead_win_nm, units)
    keep = (np.abs(c["px"]) > 0.99) & (c["py"] > 0.99) & (c["overlap"] > ov_min)
    dropped = [dict(E_meV=float(meV(c["E"][m])), px=float(c["px"][m]), py=float(c["py"][m]),
                    overlap=float(c["overlap"][m])) for m in np.nonzero(~keep)[0]
               if c["py"][m] > -0.99]                                    # y-odd (channel 2) states are expected
    x_au = min_image(g.real_axes()[0], g.Lx)
    x_nm = units.au_to_nm(x_au)
    res = dict(E1=meV(E1), E2=meV(E2), E3=meV(E3), E_top=meV(E_top), n_states=int(Es.size),
               n_kept=int(keep.sum()), n_y_odd=int(np.sum(c["py"] < -0.99)), dropped=dropped,
               max_imag_resid=float(np.max(c["imag_resid"][keep])) if keep.any() else None)
    phases = {}
    for p, key in ((1, "e"), (-1, "o")):
        m = keep & (c["px"] * p > 0.99)
        Ep = c["E"][m]
        dM1 = level_phase(Ep, k1, g.Lx)
        fits = {w: [lead_phase(x_au, c["psi1"][:, j], float(k1(Ep[q])), p, units.nm_to_au(w[0]),
                               units.nm_to_au(w[1])) for q, j in enumerate(np.nonzero(m)[0])] for w in windows_nm}
        main = fits[tuple(lead_win_nm)]
        dM2 = unwrap_pi([f["d"] for f in main])
        flux = [float(np.max(np.abs(chiral_flux(c["psi1"][:, j], np.real(spectral_derivative(c["psi1"][:, j], g.Lx)),
                                                float(k1(Ep[q])))[2])) / max(np.max(c["psi1"][:, j] ** 2), 1e-300))
                for q, j in enumerate(np.nonzero(m)[0])]
        phases[key] = dict(E=meV(Ep), d_M1=dM1, d_M2=dM2,
                           d_M2_windows={f"{w[0]:g}-{w[1]:g}": unwrap_pi([f["d"] for f in fits[w]]) for w in windows_nm},
                           k_fit=np.array([f["k"] for f in main]) / units.length_nm,
                           k_lead=np.array(k1(Ep)) / units.length_nm,
                           resid=np.array([f["resid"] for f in main]), flux_rel=np.array(flux))
    res["phases"] = phases
    Eg = np.asarray(E_grid_meV, float)
    if len(phases["e"]["E"]) > 1 and len(phases["o"]["E"]) > 1:
        pe, po = phases["e"], phases["o"]
        res["E_grid"] = Eg
        res["T_M1"] = transmission(Eg, pe["E"], pe["d_M1"], po["E"], po["d_M1"])
        res["T_M2"] = transmission(Eg, pe["E"], pe["d_M2"], po["E"], po["d_M2"])
        res["T_M2_windows"] = {w: transmission(Eg, pe["E"], pe["d_M2_windows"][w], po["E"], po["d_M2_windows"][w])
                               for w in pe["d_M2_windows"]}
        low = Eg < res["E1"] + e_min_meV
        for key in ("T_M1", "T_M2"):
            res[key][low] = np.nan
        for w in res["T_M2_windows"]:
            res["T_M2_windows"][w][low] = np.nan
        # M1 vs M2 phase agreement (mod pi), per level
        res["dM1_vs_M2_max"] = float(max(np.max(np.abs(np.angle(np.exp(2j * (ph["d_M1"] - ph["d_M2"]))))) / 2
                                         for ph in (pe, po)))
    return res
