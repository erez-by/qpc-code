# QPC spin-DFT project (educational)

Reproduce Hirose, Meir, Wingreen, PRL 90, 026804 (2003): 2D plane-wave spin-DFT (LSDA) of a
quantum point contact. Target: a local moment at the QPC, i.e. integrated net spin density
N_up - N_down ~ 1 (one electron spin, S = 1/2); paper Fig. 1 (spin densities, 3 QPC lengths)
and Fig. 2 (local DOS at the QPC centre vs in-plane B).

The full task list, formulas and acceptance tests are in `docs/SPEC.md`. Work through it in order.

## Workflow rules
- One module at a time: implement -> tests in `tests/` -> `python -m pytest -q` green -> commit
  (`git commit -m "M<k>: <module>"`) -> push. Never commit with failing tests.
- Every new function: docstring with the formula it implements, units, and the paper / reference.
- Every module gets at least one test against an analytic result or an independent method.
- Do not change the conventions below or the public interfaces of existing modules
  (units, grid, potential, fourier, hamiltonian, solver). If a change is unavoidable, stop and
  explain why first.
- If a physics choice is not fixed by docs/SPEC.md, do NOT guess silently: implement the
  documented default, add a `# PHYSICS-CHOICE:` comment, and list it in `docs/OPEN_QUESTIONS.md`.
- Long runs: scripts must checkpoint every converged state to `results/*.npz` and be resumable.
  Print progress (iteration, residual, mu, net spin) at least every iteration.
- Python 3.12, numpy/scipy/matplotlib/mpmath only. .py modules and scripts, no notebooks.

## Conventions (fixed)
- Effective GaAs atomic units inside the code: hbar = m* = e^2/(4 pi eps0 eps_r) = 1,
  m* = 0.067, eps_r = 12.9: 1 Ha* = 10.956 meV, 1 a* = 10.189 nm (qpc/units.py).
  Convert to meV / nm / T only at input and output.
- Periodic cell Lx x Ly = 5000 nm x 320 nm, grid from `grid_from_cutoff` (Nx=525, Ny=35 at
  E_cut = 15 meV, n_pw = 3379). QPC centred at x = 0, wire at y = 0; physical coordinates are
  `min_image(x, Lx)`, `min_image(y, Ly)`.
- Plane-wave coefficients: order `np.nonzero(mask)`; psi(r) = sum_p c_p exp(i G_p.r) with
  sum|c|^2 = 1, so the physical orbital is psi/sqrt(Lx Ly) and
  density per spin n_s(r) = sum_i f_is |psi_is(r)|^2 / (Lx Ly)   [a*^-2].
- `V_hat = fft2(V)/(Nx Ny)`. Eigenpairs via `qpc.solver.lowest_states(ham, nb, method=...)`.
- Spin index s = 0 (up, parallel to the Zeeman-favoured direction), 1 (down).

## Commands
    source .venv/bin/activate
    python -m pytest -q
    python scripts/<script>.py
