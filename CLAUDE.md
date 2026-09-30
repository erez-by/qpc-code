# QPC spin-DFT project (educational)
Goal: 2D plane-wave spin-DFT of a quantum point contact, reproducing the
spin density / local moment of Hirose, Meir, Wingreen, PRL 90, 026804 (2003).

## Rules
- Python 3.12, numpy/scipy/mpmath, .py modules only, no notebooks.
- Effective GaAs atomic units internally (hbar = m* = e^2/(4 pi eps0 eps_r) = 1).
  Convert to nm/meV only at input and output.
- Every function gets a docstring stating units and the formula it implements.
- Do NOT write potential.py or the mpmath Fourier transform: the user writes
  them. Review them and suggest tests.
- Explain physics choices briefly and cite the paper; say when unsure.
- Every module gets a test against an analytic result in tests/.
- Run tests with: source .venv/bin/activate && pytest

## Layout
qpc/: units, grid, fourier, potential, hamiltonian, solver, occupation,
hartree, xc, mixing, scf, analysis. Scripts in scripts/, figures in figures/.

## Parameters (current plan)
Lx ~ 5 um, Ly ~ 320 nm, ~10 nm grid, E_cut ~ 15 meV, smearing ~ 0.05 meV.
m* = 0.067 m0, kappa = 12.9 (verify against the article).
