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
7. **Clean wire / M5.1 — resolved (user decision).** The model is the full one (bare parabola +
   V_QPC + V_H[n] + v_xc,s[n] + Zeeman): HMW Eq. (2) has delta V_H = V_H[rho] - V_H[rho0] with
   rho0 the INTERACTING clean wire of HW (cond-mat/0106581, Eq. 1), so V_H[rho0] + delta V_H =
   V_H[rho]. No reference-subtracted mode. The earlier failure (spacing 0.889 meV) came from
   the Hartree convention bug (a = 100 nm used as metal distance; notes/m5_wire_diag_a_metal100.txt).
   With a_image = 100 nm: spacing 0.954 meV, mu - e_0 = 0.930 meV, 11.5 electrons in subband 1
   at kT = 0.05 meV; acceptance rewritten (subband-0 density), PASS.
