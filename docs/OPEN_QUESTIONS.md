# Open questions and physics choices

Choices not fixed by the PRL (HMW 2003) or by docs/SPEC.md. Each one is marked in the code with
`# PHYSICS-CHOICE:`. Ask the user / advisor.

## From docs/SPEC.md
1. Tanatar-Ceperley coefficients and energy unit (Ry*): verify against PRB 39, 5005 (1989) (M3).
   Status: zeta = 0 row confirmed from a secondary source quoting TC (arXiv cond-mat/0103541),
   not from TC itself. zeta = 1 row (`TC_POL`) not verified (no open source found), marked
   `# UNVERIFIED`. Regression test: polarised/unpolarised total-energy crossing at rs = 35.9,
   required to lie in (30, 40).
2. Spin interpolation used by HMW (`exchange` vs `quadratic`): not stated in the PRL.
   `exchange` = exchange-like interpolation (Koskinen, Manninen & Reimann, PRL 79, 1389 (1997));
   previously mislabelled "von Barth-Hedin", `vbh` kept as an alias.
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
7. **Clean wire is not single-subband (M5.1 acceptance FAILS).** With hbar w_y = 2 meV,
   n_1D = 2.8e-2 nm^-1, gate a = 100 nm, LSDA (exchange interp.), kT = 0.05 meV: the
   self-consistent subband spacing is 0.889 meV (bare 2.0 meV; Hartree ~15.9 meV at the centre
   flattens the parabola into a shallow double well), mu_wire = 12.868 meV sits 6 ueV below the
   n = 1 subband bottom, and n = 1 holds 14.8 of 140 electrons. mu - e_0 = 0.883 meV instead
   of k_F^2/2 = 1.100 meV (consistent with the 125.2 electrons left in n = 0: 0.880 meV).
   Not tuned. Options: smaller a (stronger screening), different reading of HMW's n_1D
   (per spin vs total), larger hbar w_y, positive background. Needs a decision before M5.2.
