# Model: spin-DFT of a quantum point contact (reproduction of HMW 2003)

Target: Hirose, Meir & Wingreen, PRL 90, 026804 (2003) [HMW], Figs. 1-2: a local moment
(net spin ~ 1/2 x 2 = 1 electron) bound at a QPC in a 2DEG wire, within 2D LSDA.
Units inside the code: GaAs effective atomic units (hbar = m* = e^2/(4 pi eps0 eps) = 1;
m* = 0.067, eps = 12.9: 1 Ha* = 10.956 meV, 1 a* = 10.189 nm).

## Kohn-Sham Hamiltonian (HMW Eq. 2)

For spin s (sigma_s = +1 up, -1 down; up = Zeeman-favoured):

    H_s = -1/2 d_x^2 + [ -1/2 d_y^2 + 1/2 w_y^2 y^2 + V_H[rho0](y) + v_xc,s[rho0](y) ]
          + V_QPC(x, y) + delta V_H(x, y) + delta V_xc,s(x, y) - sigma_s E_Z / 2

- The bracket is the transverse problem of the clean wire rho0 (its subbands).
- delta V_H = V_H[rho] - V_H[rho0], delta V_xc,s = v_xc,s[rho] - v_xc,s[rho0] (HMW Eq. 2).
- rho0 is the INTERACTING clean wire of Hirose & Wingreen (HW), cond-mat/0106581, Eq. (1):
  bare parabola + gated Hartree + TC xc, no positive background. Hence
  V_H[rho0] + delta V_H = V_H[rho], and the code solves the equivalent full problem
  V_s = 1/2 w_y^2 y^2 + V_QPC + V_H[n] + v_xc,s[n_up, n_dn] - sigma_s E_Z/2 (`qpc/scf.py`).
- Zeeman: g mu_B B sigma with sigma = +-1/2, i.e. -+E_Z/2, E_Z = g* mu_B B, in the QPC AND in the
  leads (rho0 carries spin indices: the clean wire is solved spin-polarised at the same B).

QPC potential, HMW Eq. (1) (`qpc/potential.py`), the difference from the clean-wire parabola:

    V_QPC = V(x)/2 + 1/2 (w_y + V(x))^2 y^2 - 1/2 w_y^2 y^2,   V(x) = V0 / cosh^2(x/d),
    d = sqrt(2 V0/m*)/w_x  (82.6, 55.1, 41.3 nm for hbar w_x = 1.0, 1.5, 2.0 meV).

## Hartree (gate-screened)

Kernel (HW Eq. 1, HMW delta V_H): v(rho) = 1/rho - 1/sqrt(rho^2 + a_image^2), a_image = 100 nm
the distance charge <-> image charge (metal plane at a_m = a_image/2 = 50 nm; `Hartree(grid, a_m)`).
Numerics (`qpc/hartree.py`): Fourier series in x (periodic: the wire continues), direct
convolution with the exact mixed kernel w(k, y) = 2[K0(|k||y|) - K0(|k| sqrt(y^2 + a_image^2))]
in y (OPEN boundary: no periodic image wires), product integration with n quadratic per y cell.

## Exchange-correlation (2D LSDA)

- Exchange: exact 2D form, eps_x = -(4 sqrt2/(3 pi rs)) [(1+z)^{3/2} + (1-z)^{3/2}]/2.
- Correlation: Tanatar & Ceperley, PRB 39, 5005 (1989) (HMW ref. [20]): Pade form Eq. (14),
  coefficients Table IV, energies in Ry* (Sec. I), x = sqrt(rs):
  eps_c = a0 (1 + a1 x)/(1 + a1 x + a2 x^2 + a3 x^3);
  z = 0: (-0.3568, 1.1300, 0.9052, 0.4165), fit rs = 1-50; z = 1: (-0.0515, 340.5813, 75.2293,
  37.0170), fit rs = 5-75 (the wire peak has rs ~ 2: z = 1 row extrapolated).
- Spin interpolation (PHYSICS-CHOICE, not stated by HMW): default "quadratic", TC's own
  prescription E_c(rs, z) = E_c(rs, 0) + z^2 [E_c(rs, 1) - E_c(rs, 0)] with exact exchange.
  Comparison option "exchange" (Koskinen, Manninen & Reimann, PRL 79, 1389 (1997)) in M8.

## Protocol

1. Clean wire (V_QPC = 0) at field B, spin-polarised, fixed N = n_1D Lx = 140 -> mu_wire(B).
   x-independent potential: exact states e^{i k x} phi_n(y) (block-diagonal route).
2. QPC at FIXED mu = mu_wire(B) (grand-canonical: the leads fix the chemical potential; N floats).
3. Janak ramp (HMW: "first solve in a polarizing field, then reduce the field to zero"; Janak,
   PRB 16, 255 (1977)): start at B_max (10 T for hbar w_x = 1.5 meV) from the clean-wire spin
   densities of that field, then step B down, each QPC state started from the previous converged
   density. Every state is saved (results/qpc_wx{wx}_B{B}.npz).
4. Local moment: M_loc = int (n_up - n_dn) - (N_up - N_dn)_clean wire(B); at B = 0, M_loc = M.
   M_loc is compared with HMW's net spins 0.85, 0.93, 0.90 (hbar w_x = 1.0, 1.5, 2.0 meV;
   Fig. 1 panels (a)/(d), (b)/(e), (c)/(f)).
5. Barrier profile (PRL Fig. 1(a)-(c)): e_0,s(x) = lowest eigenvalue of -1/2 d_y^2 + V_eff,s(x, y),
   shown relative to the band bottom far in the wire (mean over |x| > 1.5 um), mu marked.
   All barrier and density numbers are taken at the single grid point x = 0 (top = max over x);
   no averaging window (the three-point |x| < 10 nm window caveat does not apply).

## Parameters

| quantity | value | source |
|---|---|---|
| hbar w_y | 2.0 meV | HMW |
| V0 | 3.0 meV | HMW Eq. (1) |
| hbar w_x | 1.0, 1.5, 2.0 meV | HMW Fig. 1 |
| n_1D | 2.8e-2 nm^-1 (total, both spins) | HMW |
| g* | 0.44, mu_B = 5.788e-2 meV/T (E_Z = 0.153 meV at 6 T) | bulk GaAs |
| a_image | 100 nm | HW Eq. (1), HMW |
| epsilon | 12.9 (HW wire paper: 13.1) | HMW |
| m* | 0.067 | GaAs |
| kT | 0.05 meV (numerical smearing) | this work |

## Numerics

Periodic cell Lx x Ly = 5000 nm x 320 nm; plane waves |G|^2/2 <= 15 meV (n_pw = 3379, grid
525 x 35); dense diagonalisation of the lowest bands per spin; Fermi occupations at fixed mu;
Pulay/DIIS density mixing (alpha = 0.2, history 8); converged when sum_s int |n_out - n_in| <
1e-4 electrons and |dM| < 1e-4.

## Deviations from HMW

- HMW: open system, scattering states with a recursion-transfer-matrix method, T = 0.1 K.
  Here: periodic 5 um supercell (the QPC sees its periodic image 5 um away), discrete
  levels (spacing ~0.06 meV near mu), hence kT = 0.05 meV as a numerical smearing.
  In the clean wire kT = 0.0086 vs 0.05 meV changes mu - e_0 by ~0.02 meV and the subband-1
  occupation by ~2 electrons; Lx and kT are checked in M8.
- Spin interpolation of the TC functional: not stated in HMW (default quadratic, see above).
- At n_1D = 2.8e-2 nm^-1 the clean wire's second subband is partly occupied (11.5 electrons at
  kT = 0.05 meV; edge pinned near mu, as in HW); these electrons are reflected by the QPC
  (n = 1 barrier ~9 meV) and do not carry current. In a field they polarise the leads strongly
  (M_wire = 25.4 at 6 T), which is why M_loc is used.
