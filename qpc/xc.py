"""2D local spin-density approximation (LSDA): exchange + Tanatar-Ceperley correlation.

Atomic units (Ha*, a*). n = n_up + n_dn, rs = 1/sqrt(pi n), zeta = (n_up - n_dn)/n.
E_xc = int n eps_xc(rs, zeta) d^2r; potentials (2D: n d rs/dn = -rs/2,
n d zeta/d n_up = 1 - zeta, n d zeta/d n_dn = -(1 + zeta)):

    v_xc,up = eps - (rs/2) d eps/d rs + (1 - zeta) d eps/d zeta
    v_xc,dn = eps - (rs/2) d eps/d rs - (1 + zeta) d eps/d zeta

References: Tanatar & Ceperley, PRB 39, 5005 (1989) [TC] (HMW ref. [20]); Attaccalite et al.,
PRL 88, 256601 (2002) on the spin interpolation.
"""
import numpy as np

N_FLOOR = 1e-10            # a*^-2; below it eps = v = 0 (outside the wire)

CX = 4.0 * np.sqrt(2.0) / (3.0 * np.pi)          # 0.60021 : eps_x(zeta=0) = -CX/rs Ha*

# Tanatar-Ceperley Pade fit eps_c(rs) = a0 (1 + a1 x)/(1 + a1 x + a2 x^2 + a3 x^3), x = sqrt(rs),
# energies in Ry* = Ha*/2. Source: TC 1989, Pade fit to the fixed-node DMC energies (fit
# equation and coefficient table in the paper). Values as given in docs/SPEC.md M3.
# Status: zeta = 0 row as in the spec; zeta = 1 row UNVERIFIED (user to check against TC).
TC_UNPOL = (-0.3568, 1.1300, 0.9052, 0.4165)                  # zeta = 0 (a0, a1, a2, a3)
TC_POL = (-0.0515, 340.5813, 75.2293, 37.0170)                # UNVERIFIED  zeta = 1 (a0, a1, a2, a3)
RY = 0.5                                                      # 1 Ry* = 0.5 Ha*


def eps_x(rs, zeta):
    """Exchange energy per electron of the 2D gas [Ha*] (exact):
    eps_x = -(4 sqrt2 / (3 pi rs)) [(1+zeta)^{3/2} + (1-zeta)^{3/2}] / 2.
    Returns (eps_x, d eps_x/d rs, d eps_x/d zeta)."""
    p, m = 1.0 + zeta, 1.0 - zeta
    phi = 0.5 * (p ** 1.5 + m ** 1.5)
    dphi = 0.75 * (np.sqrt(p) - np.sqrt(m))
    e = -CX * phi / rs
    return e, -e / rs, -CX * dphi / rs


def eps_c_pade(rs, coeffs):
    """TC Pade correlation energy per electron [Ha*] and d/d rs:
    eps_c = RY * a0 (1 + a1 x)/(1 + a1 x + a2 x^2 + a3 x^3), x = sqrt(rs)  (TC 1989; Ry* -> Ha*)."""
    a0, a1, a2, a3 = coeffs
    x = np.sqrt(rs)
    num = 1.0 + a1 * x
    den = 1.0 + a1 * x + a2 * x * x + a3 * x ** 3
    dden = a1 + 2 * a2 * x + 3 * a3 * x * x
    e = RY * a0 * num / den
    de_dx = RY * a0 * (a1 * den - num * dden) / den ** 2
    return e, de_dx / (2.0 * x)


def spin_interp(zeta, interp):
    """Spin interpolation f(zeta) and f'(zeta), f(0) = 0, f(1) = 1.

    PHYSICS-CHOICE (docs/OPEN_QUESTIONS.md #2):
      "vbh":       f = [(1+z)^{3/2} + (1-z)^{3/2} - 2] / (2^{3/2} - 2)   (2D von Barth-Hedin form)
      "quadratic": f = z^2                                               (TC's quadratic form)
    """
    if interp == "vbh":
        c = 2.0 ** 1.5 - 2.0
        p, m = 1.0 + zeta, 1.0 - zeta
        return (p ** 1.5 + m ** 1.5 - 2.0) / c, 1.5 * (np.sqrt(p) - np.sqrt(m)) / c
    if interp == "quadratic":
        return zeta ** 2, 2.0 * zeta
    raise ValueError(interp)


def exc_vxc(n_up, n_dn, interp="vbh"):
    """2D LSDA: eps_xc per electron and v_xc,up, v_xc,dn, all in Ha*, same shape as n_up.

    eps_xc = eps_x(rs, zeta) + eps_c^0(rs) + f(zeta) [eps_c^1(rs) - eps_c^0(rs)]
    with derivatives taken analytically; v_s from the formulas in the module docstring.
    Density floor n < N_FLOOR -> eps = v = 0; zeta clipped to [-1, 1].
    """
    n_up = np.asarray(n_up, dtype=float)
    n_dn = np.asarray(n_dn, dtype=float)
    n = n_up + n_dn
    eps = np.zeros(n.shape)
    v_up = np.zeros(n.shape)
    v_dn = np.zeros(n.shape)
    ok = n >= N_FLOOR
    if not np.any(ok):
        return eps, v_up, v_dn
    nn = n[ok]
    rs = 1.0 / np.sqrt(np.pi * nn)
    z = np.clip((n_up[ok] - n_dn[ok]) / nn, -1.0, 1.0)

    ex, dex_rs, dex_z = eps_x(rs, z)
    e0, de0 = eps_c_pade(rs, TC_UNPOL)
    e1, de1 = eps_c_pade(rs, TC_POL)
    f, df = spin_interp(z, interp)

    e = ex + e0 + f * (e1 - e0)
    de_rs = dex_rs + de0 + f * (de1 - de0)
    de_z = dex_z + df * (e1 - e0)

    common = e - 0.5 * rs * de_rs
    eps[ok] = e
    v_up[ok] = common + (1.0 - z) * de_z
    v_dn[ok] = common - (1.0 + z) * de_z
    return eps, v_up, v_dn


def exc_energy(n_up, n_dn, dA, interp="vbh"):
    """E_xc = sum n eps_xc dA  [Ha*], dA = dx dy."""
    eps, _, _ = exc_vxc(n_up, n_dn, interp)
    return float(np.sum((np.asarray(n_up) + np.asarray(n_dn)) * eps)) * dA
