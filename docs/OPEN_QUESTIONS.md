# Open questions and physics choices

Choices not fixed by the PRL (HMW 2003) or by docs/SPEC.md. Each one is marked in the code with
`# PHYSICS-CHOICE:`. Ask the user / advisor.

## From docs/SPEC.md
1. Tanatar-Ceperley coefficients and energy unit (Ry*): verify against PRB 39, 5005 (1989) (M3).
2. Spin interpolation used by HMW (`vbh` vs `quadratic`): not stated in the PRL.
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
