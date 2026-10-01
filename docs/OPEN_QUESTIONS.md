# Open questions and physics choices

Choices not fixed by the PRL (HMW 2003) or by docs/SPEC.md. Each one is marked in the code with
`# PHYSICS-CHOICE:`. Ask the user / advisor.

## From docs/SPEC.md
1. ~~Tanatar-Ceperley coefficients and energy unit~~ **Resolved:** both rows verified against
   TC, PRB 39, 5005 (1989): Eq. (14), Table IV, energies in Ry* (Sec. I). Fit ranges rs = 1-50
   (zeta = 0), 5-75 (zeta = 1); wire peak rs ~ 2 (zeta = 1 row extrapolated). Tests: Pade vs
   printed Table I/II E_c (rel. 1e-2) -- all pass except zeta = 1, rs = 10 (Pade -0.018500 vs
   printed -0.0183, 1.09 %; strict xfail, reported); E(rs,1) - E(rs,0) changes sign in (30, 40).
2. **PHYSICS-CHOICE: spin interpolation.** HMW do not state it. Default `quadratic` (TC's own
   prescription: E_c(rs,z) = E_c(rs,0) + z^2 [E_c(rs,1) - E_c(rs,0)], exact exchange), in
   `xc.py` and `SCFParams`. The `exchange` run (Koskinen, Manninen & Reimann, PRL 79, 1389
   (1997); alias `vbh`) is a convergence/sensitivity check in M8.
3. Gate model and distance a (image plane at 2a, a = 100 nm): not stated in the PRL.
4. Temperature / smearing used by HMW (we use k_B T = 0.05 meV).
5. Which HMW Fig. 1 panel corresponds to which hbar w_x.

## Added during implementation
6. **Hartree y-quadrature (numerical, M2).** SPEC M2 prescribes the plain cell-averaged kernel
   (n piecewise constant per y cell). Its error is O(dy^2): 1.7e-3 relative for the 25 nm line
   charge of test (a) at production dy = 9.14 nm, so it fails the spec's 1e-3 tolerance (and is
   -4.6e-4 / -1.2e-4 at dy/2, dy/4, confirming pure discretisation error). Default is now
   `Hartree(..., y_rule="quadratic")`: same exact kernel w(k, y), but n quadratic (central
   differences) inside each cell, error O(dy^4): 8.9e-5 at production dy. `y_rule="cell"` keeps
   the spec's rule. Public interface (`Hartree(grid, a)`, `potential`, `energy`) unchanged.
7. **Clean wire / lead reference — resolved (user decision): delta formulation.** The lead is
   the NON-interacting harmonic clean wire n_ref (hbar w_y = 2 meV, N = 140, kT), mu_wire =
   2.1019 meV, mu - e_0 = 1.1019 meV (k_F^2/2 = 1.1000): single subband, M5.1 PASS. The KS
   potential is dV_s = V_QPC + V_H[n - n_ref] + (v_xc,s[n] - v_xc[n_ref/2, n_ref/2]),
   implemented as V_ext_eff = parabola + V_QPC - V_H[n_ref] - v_xc[ref] (`delta_external`,
   `SCFParams.reference = "delta"`). The unpolarised reference v_xc is subtracted for both spins.
   Diagnostic kept as `reference = "full"` / `run_wire.py --full`: the interacting wire with full
   Hartree + xc has subband spacing 0.889 meV, the n = 1 subband at mu (14.8 of 140 electrons),
   and mu - e_0 = 0.883 meV (notes/m5_wire_full_output.txt).
