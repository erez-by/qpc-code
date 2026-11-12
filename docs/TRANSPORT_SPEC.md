# Spec: transmission T(V_0) for the QPC (for Claude Code)

## Goal
For model C (interacting clean-wire lead, single subband), exchange-like spin interpolation, hbar*omega_x = 1.5 meV:
for each barrier amplitude V_0 in a sweep, run the usual Zeeman ramp 5 T -> 0 T, then compute the Kohn-Sham
transmission T_sigma(mu) of the converged effective potential. Produce g(V_0) = (T_up + T_dn)/2 = G/(2e^2/h),
computed with two solvers (1D adiabatic and 2D Kwant), and compare with an unpolarized reference.

## Step 0: inspect the repo first (report before coding)
Read the code and report, briefly:
1. The exact form and parameters of V_QPC in `qpc/potential.py`. Which parameter is the barrier amplitude V_0?
   Is the potential parametrized by (V_0, hbar*omega_x) or by (V_0, length alpha)?
   - REQUIREMENT: hold hbar*omega_x = 1.5 meV fixed while V_0 varies. If the code is parametrized by alpha,
     add a helper that computes alpha(V_0, hbar*omega_x) from hbar*omega_x = hbar*sqrt(2 V_0 / (m* alpha^2))
     (for V_0/cosh^2(x/alpha)), or the equivalent for the actual functional form.
     If the form makes this ambiguous, STOP and ask.
2. How the existing ramp works (field list, how mu(B) is taken from the clean wire, how model C and the
   exchange-like interpolation are selected, how results are saved, the done-marker logic in scripts/overnight.py).
3. Whether V_eff,sigma = V_QPC + V_H + V_xc,sigma + Zeeman is saved, or must be rebuilt from the saved densities.
   Use the existing hartree/xc functions; do not reimplement them.
Reuse the existing modules and do not change SCF physics.

## Units and conventions
GaAs effective units as in `qpc/units.py`; energies in meV and lengths in nm at the transport-module interface.
hbar^2/(2 m*) = 38.10/0.067 meV nm^2 (take m* from units.py). Spin sigma = +1 (up), -1 (dn).
The energy zero must be the same for V_eff and mu, so check it explicitly.

## Module: `qpc/transport.py`

### 1. `veff_on_grid(state, sigma, x, y) -> ndarray`
Evaluate V_eff,sigma at arbitrary points by exact Fourier-series evaluation of the plane-wave coefficients (or by FFT
zero-padding to a finer grid). Do not use spline interpolation from the coarse 10 nm grid.

### 2. `transverse_modes(veff_xy, y, n_modes) -> eps[n, x]`
For each x column, solve [-(hbar^2/2m*) d^2/dy^2 + V_eff(x,y)] chi = eps chi by finite differences in y
(dy <= 1 nm, hard wall at the cell edges, where V is large anyway) with scipy.linalg.eigh_tridiagonal.
Return eps_{n,sigma}(x) for the lowest n_modes (default 3).

### 3. `transmission_1d(x, eps, E) -> (T, R)`
Shooting method for -(hbar^2/2m*) psi'' + eps(x) psi = E psi:
- Asymptotic points x_L, x_R: the first/last points where |eps(x) - eps(edge)| < 1e-4 meV (warn if the potential has
  not relaxed to the lead value inside the cell). k_{L,R} = sqrt(2m*(E - eps(x_{L,R})))/hbar.
  If E <= eps at either end, the channel is closed: return (0, 1).
- Set psi(x_R) = exp(i k_R x_R), psi'(x_R) = i k_R psi(x_R); integrate leftward with solve_ivp
  (complex system split into real and imaginary parts, method DOP853, rtol=1e-10, atol=1e-12), with eps(x) from a
  cubic interpolant of the finely sampled eps.
- A = exp(-i k_L x_L) (i k_L psi + psi') / (2 i k_L),  B = exp(+i k_L x_L) (i k_L psi - psi') / (2 i k_L).
- T = (k_R/k_L) / |A|^2,  R = |B|^2/|A|^2.  Assert |T + R - 1| < 1e-8.
`g_1d(state, E)`: sum over modes and spins, returning T per (sigma, n).

### 4. `transmission_2d_kwant(state, sigma, E, a=2.0, Xc=None) -> T`
- Square lattice, spacing a nm, t = hbar^2/(2 m* a^2) (about 142 meV at a = 2 nm; E - V << t holds).
- Scattering region |x| <= Xc, with Xc the smallest half-length where max_y |V_eff(x,y) - V_eff(edge column)| < 1e-4 meV.
- Onsite energy 4t + V_eff,sigma(x_i, y_j); hopping -t.
- Leads: translationally invariant in x, with onsite energy 4t + V_eff evaluated on the boundary column of the
  scattering region (the lead potential must match the region edge exactly; no artificial step).
- T = kwant.smatrix(fsyst, energy=E).transmission(1, 0). Spins are independent (no spin-orbit), so do two runs.
- `pip install kwant` in the venv (if pip fails, try conda-forge or report back).

### 5. `thermal_g(T_of_E, mu, kT)`
g = 1/2 sum_sigma int dE (-df/dE) T_sigma(E), evaluated on E in mu +- 8 kT (about 41 points) with the 1D solver only.
kT = 0.05 meV (the same as the SCF).

## Tests: `tests/test_transport.py` (all must pass before any production run)
1. Clean wire, harmonic transverse confinement, no barrier: T(E) is an integer staircase equal to the number of open
   subbands (1D and 2D).
2. Flux conservation: |T + R - 1| < 1e-8 for random smooth barriers.
3. 1D Poschl-Teller U_0/cosh^2(x/alpha) against the analytic result (Landau & Lifshitz QM sec. 25, problem 4):
   T = sinh^2(pi k alpha) / [sinh^2(pi k alpha) + cosh^2((pi/2) sqrt(8 m* U_0 alpha^2/hbar^2 - 1))]
   for 8 m* U_0 alpha^2/hbar^2 > 1 (cos instead of cosh for < 1). This formula was quoted from memory:
   if it disagrees with the numerics, cross-check against an independent piecewise-constant transfer-matrix solver.
   If the two numerical solvers agree with each other but not with the formula, report it; do not "fix" the formula
   until it matches.
4. 2D saddle point V = V_0 - (1/2) m* w_x^2 x^2 + (1/2) m* w_y^2 y^2 in Kwant against Buttiker (PRB 41, 7906):
   T_n = 1/(1 + exp(-2 pi eps_n)), eps_n = (E - hbar w_y (n + 1/2) - V_0)/(hbar w_x).
5. Adiabatic limit: for a long smooth constriction (large alpha) the 1D and 2D results agree to < 1e-2.
6. Lattice convergence: the 2D result at a = 2 nm vs a = 1 nm differs by < 1e-2 on a real V_eff.

## Script 1: `scripts/vsweep_range.py` (choose the V_0 window automatically)
1. Non-interacting scan: V_eff = V_QPC + clean-wire model C potential (for the lead offset), at mu(B=0),
   200 values of V_0, 1D T. Find V_0 where T = 0.98 and T = 0.02.
2. Pilot: about 8 V_0 values over that window extended by 50% on each side; unpolarized SCF at B = 0 (no ramp),
   1D T. Find V_0(T=0.98) and V_0(T=0.02) on the interacting curve.
3. Final grid: 25 points over [V_0(0.98) - 15% width, V_0(0.02) + 15% width]: 10 uniform plus 15 placed at
   uniform T levels on the pilot curve (merge, sort, deduplicate).
   Save `results/vsweep_C_exch_1.5/grid.json`, including the pilot data.

## Script 2: `scripts/vsweep.py` (production, resumable)
For each V_0 in grid.json, in increasing order (from the open side):
- Ramp B = 5 -> 0 T using the existing ramp field list (starting at 5 T); mu(B) from the cached model C clean wire.
- Seed: initial density at B = 5 T = the converged B = 5 T density of the previous V_0 (save every point's 5 T
  density for this). The first point starts as the existing ramp does.
- At every B in the ramp: save net spin, n_up, n_dn and the 1D T_sigma(mu(B)) (cheap).
- At B = 0: also the 2D Kwant T_sigma, thermal_g, eps_{0,sigma}(x), the effective barrier top
  max_x eps_{0,sigma}(x) - mu, SCF residual and iteration count, and the converged flag.
  Non-converged points are flagged and excluded from the plots, never used silently.
- Unpolarized reference at every V_0: one SCF at B = 0 with spin polarization off, 1D + 2D T.
- Hysteresis check: after the upward sweep, run a downward sweep over the middle 9 points, seeded from the upper
  neighbour's 5 T density.
- Output: `results/vsweep_C_exch_1.5/V0_<value>.npz` + a done-marker per point; `summary.csv` updated after each
  point (V0, B, net_spin, T_up_1d, T_dn_1d, T_up_2d, T_dn_2d, g_1d, g_2d, g_thermal, barrier_top, converged,
  direction).
- Same resumability style as scripts/overnight.py.
- Before launching: time one full point, print the estimated total runtime, and wait for confirmation.

## Script 3: `scripts/plot_vsweep.py` -> figures/
1. g vs V_0 at B = 0: spin-polarized 1D, 2D, thermal; unpolarized reference; T_up and T_dn as thin lines.
   Secondary x-axis: effective barrier top - mu.
2. Net spin vs V_0 (same x-axis, stacked under fig. 1): shows where the moment exists relative to the step.
3. eps_{0,up}(x) and eps_{0,dn}(x) for three V_0 values (open, mid-step, nearly closed), with mu marked.
4. g vs V_0 for each B of the ramp (1D): the field-induced spin splitting.
5. Up and down sweeps overlaid (hysteresis).

## Do not
- Change the SCF, the xc or the Hartree physics.
- Rigidly shift a converged V_eff to emulate a new V_0 (every V_0 requires its own SCF).
- Interpolate V_eff with splines from the 10 nm grid.
