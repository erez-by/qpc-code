# QPC spin-DFT (reproduction of Hirose, Meir & Wingreen, PRL 90, 026804 (2003)) — results so far

Code: github.com/erez-by/qpc-code. Full model with sources: docs/MODEL.md; open questions: docs/OPEN_QUESTIONS.md.

## Model and numerics (fixed)
- 2D Kohn-Sham LSDA, GaAs units (m* = 0.067, eps = 12.9; 1 Ha* = 10.956 meV, 1 a* = 10.189 nm).
- V_s = 1/2 w_y^2 y^2 + V_QPC(x,y) [HMW Eq. 1] + V_H[n] + v_xc,s[n_up,n_dn] - sigma_s E_Z/2;
  hbar w_y = 2 meV, V0 = 3 meV, n_1D = 2.8e-2 nm^-1 (total), g* = 0.44.
- Hartree: gate-screened kernel 1/rho - 1/sqrt(rho^2 + a_image^2), a_image = 100 nm (metal plane at
  50 nm); periodic in x, open boundary in y.
- xc: exact 2D exchange + Tanatar-Ceperley correlation (PRB 39, 5005, Eq. 14, Table IV, Ry*).
  Spin interpolation f(z) of E_c (HMW do not state it): "quadratic" z^2 (TC's own, code default),
  "exchange" ((1+z)^1.5+(1-z)^1.5-2)/(2^1.5-2), "power:p" family, "mixed:w".
- Model A (default): leads = interacting clean wire at the same B (HMW Eq. 2 with rho0 = HW wire).
  Model B (comparison): delta formulation with a NON-interacting clean-wire reference.
- Periodic cell 5000 x 320 nm, plane waves to 15 meV (525 x 35 grid), dense diagonalisation,
  fixed mu = mu_wire(B) (grand canonical), kT = 0.05 meV numerical smearing (level spacing of the
  5 um cell ~0.06 meV), Pulay mixing (alpha 0.2, history 8), tol 1e-4 electrons.
- Janak protocol: start at B = 6 T from the clean-wire spin densities, ramp down 6, 4, 2, 1, 0.5,
  0.25, 0 T, each state from the previous converged one.
- M_loc = int(n_up - n_dn) - (same for the clean wire at that B): local moment (lead Zeeman
  polarisation removed; = M at B = 0). M_win300 = same over |x| < 300 nm.
- Barrier profile e_0,s(x) = lowest eigenvalue of -1/2 d_y^2 + V_eff,s(x,y); heights in meV above
  the far band bottom e_0(far); all values at grid points (x = 0, max over x), no windows.
- D_dn = top_dn - top_unpol, D_up0 = e_0,up(0) - top_unpol (top_unpol = barrier top of the
  unpolarised B = 0 state, same model/length: A 0.6282 / 0.7094 / 0.8366 meV for wx 1.0/1.5/2.0).

## Validation (all passed)
- Clean wire (model A, N = 140): mu_wire = 8.9565 meV, subband spacing 0.954 meV, mu - e_0 =
  0.930 meV, 11.5 electrons in subband 1 (reflected by the QPC, n=1 barrier ~9 meV); density scan
  N = 100..185 reproduces an independent run to all printed digits.
- Unpolarised QPC B = 0 (fixed mu): wx 1.5 A: n_1D(0) 0.012061, top 0.7094, mu - top +0.2202 meV,
  N 138.42; wx 1.5 B: 0.017085 / 0.6344 / +0.4676; wx 1.0 A: 0.013818 / 0.6282 / +0.3014;
  wx 1.0 B: 0.018419 / 0.5785 / +0.5235; wx 2.0 A: 0.010488 / 0.8366 / +0.0930 (independent run);
  wx 2.0 B: 0.015831 / 0.7133 / +0.3886 (independent run).
- Clean-wire lead polarisation is large (subband-1 edge pinned at mu): N_up - N_dn = 25.4 at 6 T
  (quadratic), 29.0 (exchange).

## Results before this run
1. wx = 1.5, model A, quadratic, old protocol 10 -> 0 T: M_loc peaks 0.28 at 3 T, -> 0 at B = 0
   (barrier top 0.22 meV below mu, QPC open). Seeded B = 0 runs (1 % seed, Pulay) return to
   M = 0 for A/B x quadratic/exchange.
2. Spin linear-response gain lambda of the unpolarised B = 0 state (linear mixing, 40 it):
   wx 1.5: A quad 0.71, A exch 0.89, B quad 0.66, B exch 0.84 (all < 1, stable);
   wx 1.0: A exch 1.04 (unstable), B exch 0.76.
3. wx = 1.0, model A, EXCHANGE, Janak 6 -> 0 T: M_loc = 0.084, 0.125, 0.098, 0.259, 0.495,
   0.720, 1.074 (B = 6, 4, 2, 1, 0.5, 0.25, 0). Large-seed B = 0 (M_init = 1) -> same state
   (M = 1.073). B = 0: spin-up barrier two peaks 0.416 meV at x = +-85.7 nm, centre -0.03 meV;
   spin-down one peak 1.220 meV (mu - top = -0.29); n_1D(0) = 0.01298 (up 0.01235, dn 0.00063);
   D_dn = +0.592, D_up0 = -0.658; LDOS (eta 0.05): spin-up resonance -0.79 meV below mu
   (0.14 meV above e_0(far)), FWHM 0.13 meV (incl. 2 eta), spin-down onset +0.27 meV, U = 1.06 meV.
4. wx = 1.0, model B, exchange: ramp ends in a spin-textured state with M_loc = 0.03 (alternating
   spin, n_up(0) 0.0144 vs n_dn(0) 0.0046); large seed -> unpolarised (M = 0). No moment.
5. wx = 1.0, model A, QUADRATIC, Janak 6 -> 0 T: M_loc = 0.298, 0.311, 0.404, 0.630, 0.809,
   0.900 (0.25 T), 0.911 at B = 0 but NOT converged (300 it, M_loc swinging 0.82-1.09) ->
   M_loc(B -> 0) = 0.9 +- 0.1. B = 0 (unconverged): spin-up peaks 0.40 meV at +-76 nm, centre
   +0.02; spin-down 1.31 meV; n_1D(0) 0.0134; D_dn +0.68, D_up0 -0.61; LDOS: resonance -0.70 meV,
   FWHM 0.155, onset +0.20, U = 0.90 meV.

## HMW targets (user's readings of PRL Fig. 1, polarised, B = 0, mu - e_0(far) ~ 1.1 meV)
- wx 1.0: net spin 0.85; spin-up two peaks ~0.6 meV at +-80 nm, centre ~0.45; spin-down ~1.2;
  n_1D(0) ~ 1.5e-2; resonance ~ -0.5 meV below mu, Gamma ~0.1, spin-down onset ~ +0.1, U ~ 0.6 meV.
- wx 1.5: net spin 0.93; spin-up ~0.6; spin-down ~1.3; n_1D(0) ~ 1.4e-2.
- wx 2.0: net spin 0.90; spin-up single peak ~0.8 at x = 0; spin-down ~1.5.
- In HMW D_dn ~ +0.6 meV for all lengths and the spin-up peak ~ the unpolarised top (D_up0 ~ -0.18).
