"""Landauer transmission of the self-consistent Kohn-Sham potentials (docs/TRANSPORT_SPEC.md).

Interface units: energies in meV, lengths in nm. hbar^2 / (2 m*) = HB2_2M = (1/2) Ha* a*^2 = 568.65 meV nm^2.
Spins are independent (no spin-orbit); each spin sigma sees V_eff,sigma(x, y) = V_ext + s_H V_H + v_xc,sigma
-+ E_Z/2 (SCFResult.V), and mu is measured from the same energy zero (bottom of the y parabola).

1D adiabatic route (lowest transverse mode only):
    eps_0,sigma(x) = lowest eigenvalue of -(hbar^2/2m*) d^2/dy^2 + V_eff,sigma(x, y)      (transverse_modes)
    -(hbar^2/2m*) psi'' + eps_0,sigma(x) psi = E psi,  T_sigma(E) from the scattering state (scatter_1d)
2D route: tight-binding lattice of the full V_eff,sigma(x, y) with Kwant (build_kwant, transmission_2d_kwant).
Conductance (Landauer-Buttiker): G = (e^2/h) sum_sigma int dE (-df/dE) T_sigma(E)   (thermal_conductance).

References: Landauer, IBM J. Res. Dev. 1, 223 (1957); Buttiker, PRB 41, 7906 (1990) (saddle point);
Landau & Lifshitz, QM, sec. 25 problem 4 (Poschl-Teller); Groth et al., New J. Phys. 16, 063065 (2014) (Kwant).
"""
from dataclasses import dataclass
import json

import numpy as np
from scipy.linalg import eigh_tridiagonal

from .grid import grid_from_cutoff
from .potential import QPCParams, V_qpc
from .scf import external_potential
from .units import Units

U = Units()
HB2_2M = 0.5 * U.energy_meV * U.length_nm ** 2          # hbar^2 / (2 m*) [meV nm^2]
LY_NM = 320.0


# ----------------------------------------------------------------------------- potential on fine points
def smooth_coeffs(V, grid, qpc_params):
    """Fourier coefficients F = fft2(V - V_ext) / (Nx Ny) of the smooth part of a KS potential [Ha*].
    V_ext = (1/2) w_y^2 y^2 + V_QPC is not band-limited (kink of y^2 at the periodic y boundary), so it is
    never Fourier-interpolated; it is added back analytically in veff_on_grid."""
    return np.fft.fft2(np.asarray(V, dtype=float) - external_potential(grid, qpc_params)) / (grid.Nx * grid.Ny)


def veff_on_grid(F, grid, qpc_params, x_nm, y_nm, units=U, chunk=2000):
    """V_eff(x, y) [meV] at arbitrary physical points (outer product x_nm x y_nm):

        V_eff = Re sum_{m,n} F_mn exp(i (Gx_m x + Gy_n y)) + (1/2) w_y^2 y^2 + V_QPC(x, y)

    i.e. the exact Fourier series of the smooth part on the SCF grid (reproduces the grid values) plus the
    analytic external potential. No spline interpolation."""
    Gx, Gy = grid.G_axes()
    x = units.nm_to_au(np.asarray(x_nm, dtype=float))
    y = units.nm_to_au(np.asarray(y_nm, dtype=float))
    Ey = np.exp(1j * Gy[:, None] * y[None, :])                                  # (Ny, ny)
    FEy = F @ Ey                                                                # (Nx, ny)
    out = np.empty((x.size, y.size))
    for s in range(0, x.size, chunk):
        xs = x[s:s + chunk]
        sm = np.real(np.exp(1j * xs[:, None] * Gx[None, :]) @ FEy)
        out[s:s + chunk] = sm + 0.5 * qpc_params.wy ** 2 * y[None, :] ** 2 + V_qpc(xs[:, None], y[None, :], qpc_params)
    return units.au_to_meV(out)


def transverse_modes(V_xy, y_nm, n_modes=2, return_vectors=False):
    """Lowest transverse levels eps_n(x) [meV] of -(hbar^2/2m*) d^2/dy^2 + V(x, y) for every x row of
    V_xy (nx, ny) [meV] on the uniform interior points y_nm (hard walls one step outside), by second-order
    finite differences: diagonal 2 t_y + V, off-diagonal -t_y, t_y = HB2_2M / dy^2 (eigh_tridiagonal).
    return_vectors: also chi_n(x, y) with sum_j chi^2 = 1 (discrete normalisation, as on a lattice)."""
    V_xy = np.atleast_2d(V_xy)
    dy = float(y_nm[1] - y_nm[0])
    ty = HB2_2M / dy ** 2
    off = np.full(V_xy.shape[1] - 1, -ty)
    eps = np.empty((V_xy.shape[0], n_modes))
    vecs = np.empty((V_xy.shape[0], V_xy.shape[1], n_modes)) if return_vectors else None
    for i, row in enumerate(V_xy):
        if return_vectors:
            w, v = eigh_tridiagonal(2 * ty + row, off, select="i", select_range=(0, n_modes - 1))
            vecs[i] = v * np.sign(v[np.argmax(np.abs(v[:, 0])), 0])
            eps[i] = w
        else:
            eps[i] = eigh_tridiagonal(2 * ty + row, off, select="i", select_range=(0, n_modes - 1),
                                      eigvals_only=True)
    return (eps, vecs) if return_vectors else eps


def y_points(W=100.0, dy=0.5):
    """Uniform interior points of [-W, W] with hard walls at +-W (spacing dy)."""
    n = int(round(2 * W / dy))
    return -W + dy * np.arange(1, n)


# ----------------------------------------------------------------------------- saved states
@dataclass
class TransportState:
    """Smooth-part coefficients of the KS potentials of one saved SCF state (both spins) and its mu."""
    F: list                       # [F_up, F_dn], fft2 of V - V_ext (Ha*)
    grid: object
    qpc: QPCParams
    mu_meV: float
    converged: bool
    label: str = ""


def load_state(npz_path, hbar_wx, V0, lx=5000.0, ecut=15.0, label=""):
    """TransportState from an SCFResult .npz (only V_up, V_dn, mu, converged are read)."""
    z = np.load(npz_path)
    grid = grid_from_cutoff(U.nm_to_au(lx), U.nm_to_au(LY_NM), U.meV_to_au(ecut))
    if z["V_up"].shape != (grid.Nx, grid.Ny):
        raise ValueError(f"{npz_path}: V shape {z['V_up'].shape} != grid {(grid.Nx, grid.Ny)}")
    q = QPCParams(hbar_wx_meV=hbar_wx, V0_meV=V0)
    F = [smooth_coeffs(z["V_up"], grid, q), smooth_coeffs(z["V_dn"], grid, q)]
    return TransportState(F=F, grid=grid, qpc=q, mu_meV=U.au_to_meV(float(z["mu"])),
                          converged=bool(z["converged"]), label=label)


def mode_profile(state, s, x_nm, W=100.0, dy=0.5, n_modes=2):
    """eps_n,s(x) [meV], shape (nx, n_modes). PHYSICS-CHOICE (OPEN_QUESTIONS #17): hard walls at |y| = W =
    100 nm, where V_eff > mu + 12 meV (shift of eps_0 < 1e-5 meV against the cell edge at 160 nm)."""
    y = y_points(W, dy)
    return transverse_modes(veff_on_grid(state.F[s], state.grid, state.qpc, x_nm, y), y, n_modes)


def check_energy_zero(state, wire_json, analysis_json=None, tol_mu=1e-6, tol_e=1e-3):
    """Explicit check that V_eff and mu share the energy zero: (i) mu of the state equals mu of the clean
    wire (and of analysis.json); (ii) the transverse levels e0, e1 of the edge column x = Lx/2 equal the
    clean-wire subband bottoms. Raises ValueError on failure; returns the numbers."""
    w = json.load(open(wire_json))
    out = dict(mu=state.mu_meV, mu_wire=w["mu_meV"])
    if abs(state.mu_meV - w["mu_meV"]) > tol_mu:
        raise ValueError(f"mu {state.mu_meV} != wire mu {w['mu_meV']}")
    if analysis_json:
        a = json.load(open(analysis_json))
        out["mu_analysis"] = a["mu_meV"]
        if abs(state.mu_meV - a["mu_meV"]) > tol_mu:
            raise ValueError(f"mu {state.mu_meV} != analysis mu {a['mu_meV']}")
    xe = np.array([0.5 * U.au_to_nm(state.grid.Lx)])
    for s in range(2):
        e = mode_profile(state, s, xe, W=150.0, dy=0.25)[0]
        out[f"e0_e1_edge_s{s}"] = e.tolist()
        ref = w["subbands_up_meV"][:2] if s == 0 else w["subbands_dn_meV"][:2]
        if np.max(np.abs(e - np.asarray(ref))) > tol_e:
            raise ValueError(f"edge levels {e} != wire subbands {ref} (spin {s})")
    out["mu_minus_e0_edge"] = state.mu_meV - out["e0_e1_edge_s0"][0]
    return out


# ----------------------------------------------------------------------------- 1D scattering
def lead_cut(x, eps, tol=1e-4, cap=None):
    """Smallest half-length X with |eps(x) - eps(edge of that side)| < tol for all |x| >= X (x symmetric,
    uniform). PHYSICS-CHOICE (OPEN_QUESTIONS #16): outside +-X the potential is frozen at eps(-+X).
    Returns (X, residual_at_X); X = cap (and residual reported) if the tail never relaxes that far."""
    x = np.asarray(x)
    cap = cap if cap is not None else float(np.max(np.abs(x)))
    X = 0.0
    for side in (x < 0, x > 0):
        xs, es = x[side], eps[side]
        edge = es[np.argmax(np.abs(xs))]
        bad = np.abs(es - edge) >= tol
        if bad.any():
            X = max(X, float(np.max(np.abs(xs[bad]))) + float(x[1] - x[0]))
    X = min(X, cap)
    sel = np.abs(x) >= X - 1e-9
    res = float(max(np.max(np.abs(eps[sel & (x < 0)] - eps[np.argmin(x)])),
                    np.max(np.abs(eps[sel & (x > 0)] - eps[np.argmax(x)])))) if sel.any() else 0.0
    return X, res


def _numerov_q(w, h):
    """Discrete wavenumber of the Numerov recursion in a flat region: cos(q h) = (6 - 5 w) / w."""
    return np.arccos(np.clip((6.0 - 5.0 * w) / w, -1.0, 1.0)) / h


def scatter_1d(x, eps, E, want_psi=False, npad=4):
    """Scattering of -(hbar^2/2m*) psi'' + eps(x) psi = E psi on the uniform grid x [nm] (eps [meV]), with
    flat leads eps(x_0) on the left and eps(x_-1) on the right. E [meV] scalar or array.

    Numerov, f = (E - eps)/HB2_2M, w = 1 + h^2 f / 12:  w_{i-1} psi_{i-1} + w_{i+1} psi_{i+1} = (12 - 10 w_i) psi_i.
    Incident from the LEFT: psi = t e^{i q_R x} on the right (start there, march left: stable through
    barriers); on the left psi = A e^{i q_L x} + B e^{-i q_L x} (2x2 solve on two pad points); t = 1/A,
    r = B/A. The recursion conserves the discrete current J = w_i w_{i+1} Im(psi_i^* psi_{i+1}), which in a
    flat region is w^2 sin(q h) (|A|^2 - |B|^2), hence exactly
        T = [w_R^2 sin(q_R h)] / [w_L^2 sin(q_L h)] / |A|^2,   R = |B|^2 / |A|^2,   T + R = 1,
    with q the Numerov wavenumber (cos qh = (6 - 5w)/w; equals k = sqrt(f) to O(h^4)).
    Incident from the RIGHT: mirror procedure (psi_R). T_LR = T_RL for a real potential (time reversal);
    psi_L and psi_R differ when eps(x) is not symmetric.
    Closed channel (E <= lead bottom on either side): T = 0, R = 1, psi = None.
    Returns dict(T, R, T_rev, R_rev, r, t, psi_L, psi_R, x_full) (arrays over E; psi normalised to unit
    incident amplitude, shape (nE, nx_full))."""
    x = np.asarray(x, dtype=float)
    h = float(x[1] - x[0])
    E = np.atleast_1d(np.asarray(E, dtype=float))
    eL, eR = float(eps[0]), float(eps[-1])
    xf = np.concatenate([x[0] - h * np.arange(npad, 0, -1), x, x[-1] + h * np.arange(1, npad + 1)])
    ef = np.concatenate([np.full(npad, eL), eps, np.full(npad, eR)])
    N = xf.size
    nE = E.size
    out = dict(T=np.zeros(nE), R=np.ones(nE), T_rev=np.zeros(nE), R_rev=np.ones(nE),
               r=np.full(nE, np.nan + 0j), t=np.zeros(nE, complex), x_full=xf, psi_L=None, psi_R=None)
    op = E > max(eL, eR)
    if not op.any():
        return out
    Eo = E[op]
    w = 1.0 + h * h * (Eo[:, None] - ef[None, :]) / HB2_2M / 12.0          # (nE, N)
    qL, qR = _numerov_q(w[:, 0], h), _numerov_q(w[:, -1], h)
    jL, jR = w[:, 0] ** 2 * np.sin(qL * h), w[:, -1] ** 2 * np.sin(qR * h)

    # incident from the left: start on the right, march left
    pL = np.zeros((Eo.size, N), complex)
    pL[:, -1] = np.exp(1j * qR * xf[-1])
    pL[:, -2] = np.exp(1j * qR * xf[-2])
    for i in range(N - 2, 0, -1):
        pL[:, i - 1] = ((12.0 - 10.0 * w[:, i]) * pL[:, i] - w[:, i + 1] * pL[:, i + 1]) / w[:, i - 1]
    A, B = _two_wave(pL[:, 0], pL[:, 1], xf[0], xf[1], qL, +1)
    # incident from the right: start on the left, march right
    pR = np.zeros((Eo.size, N), complex)
    pR[:, 0] = np.exp(-1j * qL * xf[0])
    pR[:, 1] = np.exp(-1j * qL * xf[1])
    for i in range(1, N - 1):
        pR[:, i + 1] = ((12.0 - 10.0 * w[:, i]) * pR[:, i] - w[:, i - 1] * pR[:, i - 1]) / w[:, i + 1]
    A2, B2 = _two_wave(pR[:, -1], pR[:, -2], xf[-1], xf[-2], qR, -1)
    out["T"][op] = jR / jL / np.abs(A) ** 2
    out["R"][op] = np.abs(B) ** 2 / np.abs(A) ** 2
    out["T_rev"][op] = jL / jR / np.abs(A2) ** 2
    out["R_rev"][op] = np.abs(B2) ** 2 / np.abs(A2) ** 2
    out["r"][op] = B / A
    out["t"][op] = 1.0 / A
    if want_psi:
        full_L = np.full((nE, N), np.nan + 0j)
        full_R = np.full((nE, N), np.nan + 0j)
        full_L[op] = pL / A[:, None]
        full_R[op] = pR / A2[:, None]
        out["psi_L"], out["psi_R"] = full_L, full_R
    return out


def _two_wave(p0, p1, x0, x1, q, direction):
    """Amplitudes of psi = A e^{i d q x} + B e^{-i d q x} (d = direction = +-1, A = incident) from the
    values p0, p1 at two points of a flat region."""
    a0, a1 = np.exp(1j * direction * q * x0), np.exp(1j * direction * q * x1)
    det = a0 / a1 - a1 / a0
    A = (p0 / a1 - p1 / a0) / det
    B = (p1 * a0 - p0 * a1) / det
    return A, B


def transmission_1d(x, eps, E):
    """(T, R) of the 1D problem (scatter_1d)."""
    o = scatter_1d(x, eps, E)
    return o["T"], o["R"]


def transmission_1d_ivp(x, eps, E):
    """Independent check: continuum shooting with scipy solve_ivp (DOP853, rtol 1e-10, atol 1e-12) of
    psi'' = (eps(x) - E)/HB2_2M psi, eps(x) a cubic spline of the samples. psi = e^{i k_R x} at x_R,
    integrated to x_L, A = e^{-i k_L x_L}(i k_L psi + psi')/(2 i k_L), B = e^{i k_L x_L}(i k_L psi - psi')/(2 i k_L),
    T = (k_R/k_L)/|A|^2, R = |B|^2/|A|^2. Scalar E."""
    from scipy.integrate import solve_ivp
    from scipy.interpolate import CubicSpline
    x = np.asarray(x, float)
    cs = CubicSpline(x, eps)
    eL, eR = float(eps[0]), float(eps[-1])
    if E <= max(eL, eR):
        return 0.0, 1.0
    kL, kR = np.sqrt((E - eL) / HB2_2M), np.sqrt((E - eR) / HB2_2M)

    def rhs(xx, y):
        f = (cs(xx) - E) / HB2_2M
        return [y[2], y[3], f * y[0], f * y[1]]
    p = np.exp(1j * kR * x[-1])
    dp = 1j * kR * p
    sol = solve_ivp(rhs, (x[-1], x[0]), [p.real, p.imag, dp.real, dp.imag], method="DOP853", rtol=1e-10,
                    atol=1e-12)
    psi = sol.y[0, -1] + 1j * sol.y[1, -1]
    dpsi = sol.y[2, -1] + 1j * sol.y[3, -1]
    A = np.exp(-1j * kL * x[0]) * (1j * kL * psi + dpsi) / (2j * kL)
    B = np.exp(1j * kL * x[0]) * (1j * kL * psi - dpsi) / (2j * kL)
    return float((kR / kL) / abs(A) ** 2), float(abs(B) ** 2 / abs(A) ** 2)


def transmission_1d_steps(x, eps, E):
    """Independent check: piecewise-constant potential (value eps_j on the cell around x_j), 2x2 transfer
    matrices for local plane waves A_j e^{i k_j (x - x_j)} + B_j e^{-i k_j (x - x_j)} with continuity of
    psi and psi' at the cell boundaries. With (A_N, B_N) = M (A_0, B_0) and B_N = 0:
    T = (k_N / k_0) |det M / M_11|^2,  R = |M_10 / M_11|^2 (no use of flux conservation). Scalar E."""
    x = np.asarray(x, float)
    h = float(x[1] - x[0])
    k = np.sqrt((E - np.asarray(eps, float) + 0j) / HB2_2M)
    if not (k[0].real > 0 and k[-1].real > 0 and abs(k[0].imag) < 1e-14 and abs(k[-1].imag) < 1e-14):
        return 0.0, 1.0
    M = np.eye(2, dtype=complex)
    for j in range(k.size - 1):
        kj, kn = k[j], k[j + 1]
        e = np.exp(1j * kj * h)
        r = kj / kn
        M = (0.5 * np.array([[1 + r, 1 - r], [1 - r, 1 + r]])) @ np.array([[e, 0], [0, 1 / e]]) @ M
    T = (k[-1].real / k[0].real) * abs(np.linalg.det(M) / M[1, 1]) ** 2
    R = abs(M[1, 0] / M[1, 1]) ** 2
    return float(T), float(R)


# ----------------------------------------------------------------------------- conductance
def fermi_window(E, mu, kT):
    """-df/dE = 1 / (4 kT cosh^2((E - mu)/(2 kT))) [1/meV]."""
    return 1.0 / (4.0 * kT * np.cosh((np.asarray(E) - mu) / (2.0 * kT)) ** 2)


def thermal_conductance(E, T_up, T_dn, mu, kT):
    """Landauer conductance per spin and total, in units of e^2/h (G) and 2e^2/h (g = G/2):
    T = 0: G = T_up(mu) + T_dn(mu); finite kT: G = sum_s int T_s(E)(-df/dE) dE / int (-df/dE) dE over the
    sampled window (normalised, so T = 1 gives exactly 1 per spin). E uniform, should cover mu +- 8 kT."""
    E = np.asarray(E, float)
    wgt = fermi_window(E, mu, kT)
    norm = np.trapezoid(wgt, E)
    s_up = float(np.trapezoid(wgt * T_up, E) / norm)
    s_dn = float(np.trapezoid(wgt * T_dn, E) / norm)
    T0_up, T0_dn = float(np.interp(mu, E, T_up)), float(np.interp(mu, E, T_dn))
    return dict(T_up_mu=T0_up, T_dn_mu=T0_dn, G_e2h_T0=T0_up + T0_dn, G_e2h_kT=s_up + s_dn,
                g_2e2h_T0=0.5 * (T0_up + T0_dn), g_2e2h_kT=0.5 * (s_up + s_dn), T_up_kT=s_up, T_dn_kT=s_dn,
                window_norm=float(norm))


# ----------------------------------------------------------------------------- 2D (Kwant)
def kwant_system(V, a, t=None):
    """Kwant system of a rectangular region with onsite 4t + V[i, j] (V (nx, ny) meV on x_i = i a,
    y_j = j a), hopping -t, t = HB2_2M / a^2, and two translationally invariant leads equal to the boundary
    columns V[0, :] (left, lead 0) and V[-1, :] (right, lead 1): no step at the interfaces. Hard walls in y
    (no sites beyond the given rows)."""
    import kwant
    t = HB2_2M / a ** 2 if t is None else t
    nx, ny = V.shape
    lat = kwant.lattice.square(a, norbs=1)
    syst = kwant.Builder()
    for i in range(nx):
        for j in range(ny):
            syst[lat(i, j)] = 4 * t + V[i, j]
    syst[lat.neighbors()] = -t
    for side, col, sym in ((0, V[0], (-a, 0)), (1, V[-1], (a, 0))):
        lead = kwant.Builder(kwant.TranslationalSymmetry(sym))
        for j in range(ny):
            lead[lat(0, j)] = 4 * t + col[j]
        lead[lat.neighbors()] = -t
        syst.attach_lead(lead)
    return syst.finalized()


def kwant_region(state, s, a=2.0, Xc=None, W=100.0, tol=1e-4, x_cap=2400.0):
    """V_eff,s on the lattice x = -Xc ... Xc, y = -W + a ... W - a (step a) and the cut Xc: the smallest
    multiple of a with max_y |V(x, y) - V(edge of that side, y)| < tol for |x| >= Xc (capped). Returns
    (V (nx, ny), xs, ys, Xc, tail_residual)."""
    ys = y_points(W, a)
    if Xc is None:
        xx = np.arange(-x_cap, x_cap + 1e-9, a)
        Vb = veff_on_grid(state.F[s], state.grid, state.qpc, xx, ys)
        dev = np.maximum(np.max(np.abs(Vb - Vb[0]), axis=1) * (xx < 0), np.max(np.abs(Vb - Vb[-1]), axis=1) * (xx > 0))
        bad = dev >= tol
        Xc = min(float(np.max(np.abs(xx[bad]))) + a, x_cap) if bad.any() else a
    Xc = a * np.ceil(Xc / a)
    xs = np.arange(-Xc, Xc + 1e-9, a)
    V = veff_on_grid(state.F[s], state.grid, state.qpc, xs, ys)
    Vb = veff_on_grid(state.F[s], state.grid, state.qpc, np.array([-x_cap, x_cap]), ys)
    res = float(max(np.max(np.abs(V[0] - Vb[0])), np.max(np.abs(V[-1] - Vb[1]))))
    return V, xs, ys, float(Xc), res


def transmission_2d(fsyst, E):
    """T (lead 0 -> 1, summed over modes), T_rev, R, number of open modes; asserts T + R = n_open."""
    import kwant
    S = kwant.smatrix(fsyst, E)
    T, Tr, R = S.transmission(1, 0), S.transmission(0, 1), S.transmission(0, 0)
    n = S.num_propagating(0)
    if n and abs(T + R - n) > 1e-6:
        raise RuntimeError(f"flux error {T + R - n}")
    return dict(T=float(T), T_rev=float(Tr), R=float(R), n_open=int(n))


def wavefunctions_2d(fsyst, E, nx, ny):
    """Scattering states at E incident from the left (lead 0) and right (lead 1), rescaled from Kwant's
    unit-current normalisation to unit incident amplitude: psi * sqrt(|v_m|) (v_m = mode velocity of the
    incident lead, so that a plane wave in a clean wire has sum_y |psi|^2 = 1). Returns dict(psi_L, psi_R)
    with arrays (n_modes, nx, ny) on the region lattice (tags (i, j) = indices)."""
    import kwant
    wf = kwant.wave_function(fsyst, E)
    out = {}
    for lead, key in ((0, "psi_L"), (1, "psi_R")):
        modes, _ = fsyst.leads[lead].modes(energy=E)
        n = modes.wave_functions.shape[1] // 2
        v = np.abs(modes.velocities[:n])
        w = wf(lead)                                                   # (n, nsites)
        grid = np.zeros((w.shape[0], nx, ny), complex)
        tags = np.array([s.tag for s in fsyst.sites])
        grid[:, tags[:, 0], tags[:, 1]] = w * np.sqrt(v)[:, None]
        out[key] = grid
    return out
