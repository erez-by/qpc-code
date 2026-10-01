# QPC spin-DFT project (educational)
Goal: 2D plane-wave spin-DFT of a quantum point contact, reproducing Hirose, Meir,
Wingreen, PRL 90, 026804 (2003): self-consistent spin densities and the local moment.

## Workflow
- Claude writes modules, tests and scripts; the user runs, reviews and writes LyX notes.
- Python 3.12, numpy/scipy/mpmath, .py modules and scripts only, no notebooks.
- Every module has tests against analytic results in tests/. Run: python -m pytest -q
- Explain physics choices briefly, cite the paper, and say when unsure.

## Units and conventions
- Effective GaAs atomic units inside the code: hbar = m* = e^2/(4 pi eps0 eps_r) = 1
  (m* = 0.067, kappa = 12.9: 1 Ha* = 10.96 meV, 1 a* = 10.19 nm). Convert only at I/O.
- Basis: plane waves exp(i G.r) with |G|^2/2 <= e_cut; basis order = np.nonzero(mask),
  stored once on Hamiltonian (ix, iy).
- psi(r) = sum_p c_p exp(i G_p . r), sum|c|^2 = 1; physical wavefunction psi/sqrt(Lx*Ly).
- V_hat = fft2(V_grid)/(Nx*Ny); H_pq = V_hat[(ix_p-ix_q)%Nx, (iy_p-iy_q)%Ny] + G_p^2/2 delta_pq.
- Grid from cutoff: N >= 2 G_cut L / pi, FFT-friendly (grid_from_cutoff).
- V_QPC (paper Eq. 1) is the difference from the clean wire; fold x and y with min_image.

## Parameters (paper)
m* = 0.067, kappa = 12.9, g = 0.44, hbar w_y = 2 meV, V0 = 3 meV,
hbar w_x = 1.0 / 1.5 / 2.0 meV (d = 82.6 / 55.1 / 41.3 nm), n1D = 2.8e-2 /nm,
T = 0.1 K (plan uses 0.05 meV smearing), image plane a = 100 nm,
xc: Tanatar-Ceperley 2D LSDA, Zeeman term +-g mu_B B (factor 1, as printed in the paper),
ramp B = 6 T -> 0.
Numerics: Lx = 5 um, Ly = 320 nm, E_cut = 15 meV (test 20, 25 later).

## Layout
qpc/: units, grid, fourier, potential, hamiltonian, solver, occupation, hartree, xc,
mixing, scf, analysis. scripts/: dayNN_*.py, figures/: output.
