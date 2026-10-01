# SPEC: remaining code for the QPC spin-DFT project (M0-M8)

Goal: reproduce Hirose, Meir, Wingreen, PRL 90, 026804 (2003) [HMW], Fig. 1 (spin densities for
three QPC lengths, net spin ~1) and Fig. 2 (local DOS at the QPC centre for B = 0..10 T).
All formulas in effective atomic units (hbar = m* = e^2/4 pi eps0 eps_r = 1) unless stated.

Physical parameters (HMW and project decisions):

| symbol | value | note |
|---|---|---|
| hbar w_y | 2.0 meV | wire confinement |
| V0 | 3.0 meV | barrier, Eq. (1) |
| hbar w_x | 1.0, 1.5, 2.0 meV | three QPC lengths (d = sqrt(2 V0/m*)/w_x = 82.6, 55.1, 41.3 nm) |
| n_1D | 2.8e-2 nm^-1 total (1.40e-2 per spin, HMW text) | lead density |
| a_image | 100 nm | charge <-> image-charge distance (HW Eq. 1, HMW delta V_H); metal plane at a_m = a_image/2 = 50 nm |
| g* | 0.44 (|g| of bulk GaAs), mu_B = 5.788e-2 meV/T | E_Z = g* mu_B B = 0.153 meV at 6 T |
| k_B T | 0.05 meV (Fermi smearing) | PHYSICS-CHOICE |
| E_cut, Lx, Ly | 15 meV, 5000 nm, 320 nm | Day 1-2 decisions |
| xc | 2D LSDA, Tanatar-Ceperley, PRB 39, 5005 (1989) | HMW ref. [20] |

---------------------------------------------------------------------------------------------
## M0. Install Day-2 files (already written and verified)

`qpc/solver.py`, `tests/test_solver.py`, `scripts/day02_benchmarks.py` are provided by the user.
Run `python -m pytest -q` and `python scripts/day02_benchmarks.py | tee notes/day02_output.txt`.
Acceptance: all tests pass; benchmark errors as in the docstring (wire < 3e-5 meV,
Poschl-Teller < 3e-5 meV). Note which solver the timing block chooses; use it as
`DEFAULT_METHOD` in `qpc/scf.py`.

If an interface mismatch appears (attribute names in `Hamiltonian`), adapt `solver.py`, not
`hamiltonian.py`, and tell the user.

---------------------------------------------------------------------------------------------
## M1. `qpc/occupation.py`

```python
def fermi(e, mu, kT)                      # 1/(1+exp((e-mu)/kT)), overflow-safe (use scipy.special.expit(-(e-mu)/kT))
def find_mu(eigs_list, N, kT)             # mu such that sum_s sum_i fermi(e_is, mu, kT) = N; brentq, bracket from min/max eig +- 40 kT
def density(ham, X, f)                    # n(r) = sum_i f_i |psi_i(r)|^2 / (Lx Ly), psi_i = ham.to_real_space(X[:, i]); returns (Nx, Ny) real, a*^-2
def check_band_margin(eigs, mu, kT, margin=15)   # True if max(eigs) - mu > margin*kT (enough bands computed)
```
Process `density` in chunks of ~32 columns (memory: (Nx, Ny, k) complex).

Tests: (i) fermi(mu)=1/2, symmetry f(mu+x)+f(mu-x)=1, no overflow at |e-mu|/kT = 1e4;
(ii) find_mu reproduces N to 1e-10 on random spectra, also with an exactly degenerate level
at the Fermi energy; (iii) density integrates to sum_i f_i (Parseval):
`density.sum()*dx*dy == f.sum()` to 1e-10.

---------------------------------------------------------------------------------------------
## M2. `qpc/hartree.py` — gate-screened Hartree, periodic in x, OPEN in y

Interaction of two electrons in the 2DEG with a metal plane (HW Eq. (1), HMW delta V_H), with
a_image = 100 nm the distance charge <-> image charge (metal plane at a_m = a_image/2):

    v(rho) = 1/rho - 1/sqrt(rho^2 + a_image^2)      (e^2/eps already = 1 in a.u.)

Code convention: `Hartree(grid, a)` takes the metal-plane distance a = a_m = a_image/2, and its
kernel is written 1/rho - 1/sqrt(rho^2 + 4 a^2) (identical). `SCFParams.a_image_nm = 100`.

**Why not a 2D FFT:** the cell is periodic in y with Ly = 320 nm. A periodic FFT would add image
wires at y = +-Ly, +-2Ly, .... A line charge lambda at distance D with the gate gives
2 lambda ln(sqrt(D^2+a_image^2)/D): for D = 320 nm this is 2*0.164*lambda per image, vs about
2*2.3*lambda for the wire itself: a ~14 % error. So: Fourier series in x (periodic is
physical: the wire continues), direct convolution in y (open boundary).

1D Fourier transform in x of v at fixed y (k = G_x):

    w(k, y) = int dx e^{-ikx} v(x, y) = 2 [ K0(|k||y|) - K0(|k| sqrt(y^2 + a_image^2)) ]   (k != 0)
    w(0, y) = ln( (y^2 + a_image^2) / y^2 )                                               (k = 0)

(uses int dx cos(kx)/sqrt(x^2+c^2) = 2 K0(|k| c); the k=0 limit follows from
K0(z) ~ -ln(z/2) - gamma.) w has an integrable log singularity at y = 0, so use the
**cell-averaged kernel** on the y grid (y_j = min_image(j dy)):

    W[k, Delta] = (1/dy) int_{Delta - dy/2}^{Delta + dy/2} w(k, y') dy',   Delta = (i - j) dy,

computed once (scipy.integrate.quad; split the Delta = 0 cell at y' = 0; quad handles the
log endpoint singularity; or integrate -2 ln|y| analytically and the smooth rest numerically).
w depends only on |k|, so compute for k >= 0 only.

Hartree potential:

    n_k(y_j)   = (1/Nx) sum_i n(x_i, y_j) e^{-i k x_i}            (np.fft.rfft along x / Nx)
    V_k(y_i)   = sum_j W[k, i - j] n_k(y_j) dy
    V_H(x, y)  = sum_k V_k(y) e^{ikx}                            (irfft * Nx)

```python
class Hartree:
    def __init__(self, grid, a)          # precompute W (shape (Nx//2+1, Ny, Ny)); log the time
    def potential(self, n)               # n: (Nx, Ny) total density a*^-2 -> V_H (Nx, Ny) Ha*
    def energy(self, n)                  # 0.5 * sum n V_H dx dy
```

Tests (tolerances relative 1e-3 unless stated; here a = a_m, the metal-plane distance passed to Hartree):
(a) x-uniform Gaussian line charge n(y) = (lambda/(sqrt(2pi) s)) exp(-y^2/2s^2), s = 25 nm:
    V_H(y=0) = int dy' n(y') ln(1 + 4a^2/y'^2), reference by scipy.quad.
(b) 2D Gaussian blob n = Q/(2 pi s^2) exp(-rho^2/2s^2), s = 30 nm, at the cell centre:
    V_H(0) = int_0^inf 2 pi rho n(rho) [1/rho - 1/sqrt(rho^2+4a^2)] d rho  (quad).
    (Periodic images in x at 5 um contribute ~ Q (2a)^2/Lx^3: negligible.)
(c) open boundary in y: V_H of a wire density on Ly = 320 nm equals (on the common y points,
    1e-8) the result with Ly = 640 nm (same dy, zero-padded density).
(d) positivity: energy(n) > 0 for random smooth n; linearity; V_H real.
(e) the k = 0 and k -> 0 kernels join smoothly: |W[1,:] - W[0,:]| small for k_1 = 2pi/Lx.

---------------------------------------------------------------------------------------------
## M3. `qpc/xc.py` — 2D LSDA

rs and polarisation (a.u.): n = n_up + n_dn, rs = 1/sqrt(pi n), zeta = (n_up - n_dn)/n.

Exchange (exact for the 2D gas):

    eps_x(rs, zeta) = -(4 sqrt(2) / (3 pi rs)) * [ (1+zeta)^{3/2} + (1-zeta)^{3/2} ] / 2     Ha*

(zeta = 0: -0.6002/rs Ha*.)

Correlation, Tanatar-Ceperley Pade form, x = sqrt(rs), **energies in Ry* = Ha*/2**:

    eps_c^P(rs) = a0 (1 + a1 x) / (1 + a1 x + a2 x^2 + a3 x^3)
    zeta = 0:  a0 = -0.3568, a1 = 1.1300,   a2 = 0.9052,  a3 = 0.4165
    zeta = 1:  a0 = -0.0515, a1 = 340.5813, a2 = 75.2293, a3 = 37.0170

**VERIFY these 8 numbers and the Rydberg unit against TC 1989 (Eq. for the fit / table)
before production runs** — they are written from memory and unchecked. Write the source
(equation/table number) into the docstring. Sanity: as rs -> 0, eps_c(zeta=0) -> a0 = -0.357 Ry,
close to the exact 2D high-density constant -0.38 Ry.

Spin interpolation (PHYSICS-CHOICE, switch `interp=`):
- `"exchange"` (alias `"vbh"`): eps_c(rs,zeta) = eps_c^0 + f(zeta) (eps_c^1 - eps_c^0),
  f(zeta) = [(1+zeta)^{3/2} + (1-zeta)^{3/2} - 2] / (2^{3/2} - 2)   (exchange-like interpolation, Koskinen, Manninen & Reimann, PRL 79, 1389 (1997))
- `"quadratic"` (default since M5.2 review): f = zeta^2 (TC's own quadratic interpolation; Attaccalite et al.,
  PRL 88, 256601 (2002) show it underestimates the spin susceptibility).
The local moment depends on this choice: run the B = 0, hbar w_x = 1.5 meV case with both.

Potentials (eps = eps_x + eps_c per electron, E_xc = int n eps):

    v_xc,up = eps - (rs/2) d eps/d rs + (1 - zeta) d eps/d zeta
    v_xc,dn = eps - (rs/2) d eps/d rs - (1 + zeta) d eps/d zeta

(derivation: in 2D rs ~ n^{-1/2} so n d rs/dn = -rs/2; n d zeta/d n_up = 1 - zeta,
n d zeta/d n_dn = -(1 + zeta).) Implement the derivatives analytically.
Density floor: where n < 1e-10 a*^-2 set eps = v = 0 (outside the wire); clip |zeta| <= 1.

```python
def exc_vxc(n_up, n_dn, interp="quadratic") -> (eps_xc, v_up, v_dn)     # arrays, Ha*
```

Tests: (i) functional derivative: for random smooth positive n_up, n_dn and perturbation dn,
[E_xc(n + h dn_up) - E_xc(n - h dn_up)]/(2h) == sum v_up dn_up dx dy (rel 1e-6), same for dn;
(ii) zeta = 0 and zeta = 1 reproduce the two Pade fits exactly; (iii) eps_x(zeta=0) = -0.6002/rs;
(iv) v_up == v_dn when n_up == n_dn; (v) zeros below the floor, no NaN anywhere.

---------------------------------------------------------------------------------------------
## M4. `qpc/mixing.py` — Pulay / DIIS density mixing

State vector: rho = concat(n_up.ravel(), n_dn.ravel()); residual F = rho_out - rho_in.
Pulay, Chem. Phys. Lett. 73, 393 (1980); form of Kresse & Furthmueller, PRB 54, 11169 (1996):

    minimise || sum_k c_k F_k ||^2  subject to  sum_k c_k = 1
    -> [B 1; 1^T 0][c; lambda] = [0; 1],  B_ij = <F_i, F_j>   (<.,.> = sum * dx dy)
    rho_in,next = sum_k c_k (rho_in,k + alpha F_k)

```python
class Pulay:
    def __init__(self, alpha=0.2, history=8)
    def step(self, rho_in, rho_out) -> rho_next        # clip negatives to 0
    def reset(self)
```
Use lstsq / drop the oldest entry if B is ill-conditioned (cond > 1e12).
Tests: (i) history=1 equals linear mixing; (ii) on a linear contraction
rho -> A rho + b (A symmetric, spectral radius 0.95, n = 200) DIIS reaches 1e-10 in far fewer
steps than linear mixing with the same alpha; (iii) output non-negative.

---------------------------------------------------------------------------------------------
## M5. `qpc/scf.py` — Kohn-Sham self-consistency

Effective potential for spin s (s = 0 up, 1 down; sigma_0 = +1, sigma_1 = -1):

    V_s = V_ext + V_H[n_up + n_dn] + v_xc,s[n_up, n_dn] - sigma_s E_Z / 2,   E_Z = g* mu_B B
    V_ext = (1/2) w_y^2 y^2 + V_QPC(x, y)     (V_QPC = 0 for the clean wire)

```python
@dataclass
class SCFParams: kT, B_T, g=0.44, a_image_nm=100.0, interp="quadratic", alpha=0.2, history=8,
                 tol=1e-4, maxiter=300, method=DEFAULT_METHOD, nb_init=160, spin_polarized=True

def run_scf(ham, hartree, V_ext, p, mu=None, N=None, n_init=None, X_init=None) -> SCFResult
```
Loop: build V_s -> `lowest_states` per spin (warm start from previous X when LOBPCG) ->
occupations (fixed mu: f = fermi(e - mu); fixed N: mu = find_mu) -> enlarge nb by 20 and redo
if `check_band_margin` fails -> n_out,s -> Pulay -> repeat.
Converged when sum_s int |n_out,s - n_in,s| dx dy < tol (electrons) AND the change of the net
spin M = int (n_up - n_dn) < tol. Print per iteration: it, residual, mu (meV), N, M, time.
`SCFResult`: n_up, n_dn, eigs per spin, X per spin, mu, N, M, iterations, converged flag, params.
`save_npz(path)` / `load_npz(path)`.
With spin_polarized=False: one KS problem, n_up = n_dn = n/2 (clean wire, unpolarised QPC).

Calculations, in this order:
1. **Clean-wire reference** (V_QPC = 0, B = 0, unpolarised, fixed N = n_1D Lx = 140).
   The potential is x-independent, so the KS states are exactly e^{i G_x x} phi(y): the
   block-diagonal eigenpairs of `XAveragedPreconditioner` (solver.py) ARE the exact
   eigenpairs — use them (exact, milliseconds). Output: mu_wire.
   The clean wire is the INTERACTING wire of Hirose & Wingreen (cond-mat/0106581, Eq. 1:
   bare parabola + gated Hartree with a_image = 100 nm + TC xc, no positive background).
   Acceptance (rewritten; the earlier "single subband, mu - e_0 = k_F^2/2 with the total n_1D"
   was a wrong premise: at n_1D = 2.8e-2 nm^-1 the 2nd subband is already partly filled, its
   onset is at n_1D ~ 2.6e-2 and its band edge is then pinned near mu, as in HW):
   - SCF converged, N = 140;
   - mu - e_0 = (pi n0 / 2)^2 / 2 with n0 = N0 / Lx from subband 0 ONLY, within
     2 kT + level spacing (reference: 0.951 predicted vs 0.954 meV found at kT = 0.0086 meV);
   - report the electrons in subband 1 (reference 9.8 at kT = 0.0086, 11.5 at kT = 0.05 meV).
   `scripts/run_wire.py --scan [--kT]` tabulates N = 100..185 (tests/test_wire.py).
   Subband-1 electrons are reflected by the QPC (n = 1 adiabatic barrier ~9 meV >> mu), so they
   do not carry current; HMW's "only the lowest two spin subbands contribute to transport" is
   about transport, not occupation.
2. **QPC, unpolarised**, fixed mu = mu_wire, B = 0, start from the clean-wire density.
   Check: far from the QPC (|x| > 1.5 um) n_1D(x) -> 2.8e-2 nm^-1 within ~1 %.
   Regression targets (hbar w_x = 1.5 meV, kT = 0.05 meV, a_image = 100 nm, mu = mu_wire(N=140)):
   mu_wire = 8.9565 meV; ~30 iterations; far-field n_1D = 0.02800 nm^-1; N = 138.42;
   e_0(far) = 8.027 meV; mu - e_0(far) = 0.930 meV; n_1D(x=0) = 0.012061 nm^-1;
   barrier top - e_0(far) = 0.7094 meV; mu - top = +0.2202 meV.
   Measurement convention: n_1D and the barrier at the single grid point x = 0 (top = max over
   x of e_0(x), here at x = 0); e_0(x) = lowest eigenvalue of -1/2 d_y^2 + V_eff(x, y) on the full
   y grid; e_0(far) = mean over |x| > 1.5 um. (An average over |x| < 10 nm, i.e. the 3 grid points
   x = 0, +-9.52 nm, gives 0.012346 / 0.6905 / +0.2391 instead.)
3. **QPC, spin-polarised, Janak ramp** (HMW: "first solve in a polarizing field, then reduce
   the field to zero"; Janak, PRB 16, 255 (1977)): start at B_max from the unpolarised
   density plus a small seed polarisation (n_up *= 1.01, n_dn *= 0.99 near the QPC),
   converge, then step B down, each step starting from the previous converged state
   (density AND eigenvectors). Save each state to `results/qpc_wx{wx}_B{B}.npz`.
   - Fig. 2 run: hbar w_x = 1.5 meV, B = 10, 9, ..., 0 T.
   - Fig. 1 runs: hbar w_x = 1.0 and 2.0 meV, B = 6, 4, 2, 1, 0.5, 0 T
     (1.5 meV reuses the Fig. 2 run).
   If the moment disappears at B = 0 (M -> 0), that is a physics result: report it, do not
   force it.

Scripts: `scripts/run_wire.py`, `scripts/run_qpc.py --wx 1.5 --B 10 9 8 ... 0` (resumable:
skip B values whose npz exists and is converged).

Cost estimate: ~6 s per SCF iteration (2 spins x dense 3379), 50-100 iterations per B point:
Fig. 2 ~1 h, Fig. 1 extra ~1 h. Run in the background (`nohup ... &> logs/x.log &`).

---------------------------------------------------------------------------------------------
## M6. `qpc/analysis.py`

```python
def n1d(n, grid)                          # int n dy -> (Nx,), a*^-1  (report in nm^-1)
def net_spin(res)                         # M = int (n_up - n_dn) dx dy  (HMW: 0.85, 0.93, 0.90)
def ldos_1d(ham, eigs, X, x0, energies, eta, shape="lorentz")
    # local 1D DOS at x = x0 (HMW Figs. 1-3 insets, Fig. 2):
    # rho_s(x0, e) = sum_i [ int dy |psi_is(x0, y)|^2 / (Lx Ly) ] * L_eta(e - e_is)
    # units 1/(meV nm). Needs states up to mu + 3 meV: after convergence, recompute the
    # final KS spectrum with the dense solver (all eigenpairs) for the analysis.
```
Broadening: eta = 0.05 meV (Gaussian or Lorentzian, keep the choice in the caption).
Finite Lx gives discrete levels with spacing ~0.06 meV near mu: eta must not be much smaller.
Test: for the clean wire the LDOS integrates (over e) to the local 1D density of states
normalisation: int rho_s de over all states = int dy (sum_i |psi_i(x0,y)|^2)/(LxLy);
for a single free subband away from the edge, rho(e) ~ 1/(pi hbar v(e)) per spin
(compare to 1/(pi sqrt(2 (e - e0))) in a.u., within the discreteness).

---------------------------------------------------------------------------------------------
## M7. Figures: `scripts/make_figures.py`

- `figures/fig1_spin_density.png`: three panels (hbar w_x = 1.0, 1.5, 2.0 meV; check which
  length goes with which HMW panel in the paper caption). Solid: n_1D,up - n_1D,dn;
  dashed: (n_1D,up + n_1D,dn)/2 (asymptote 1.40e-2 nm^-1). x in nm, |x| < 1000 nm. Net spin M in
  each panel title. Inset/extra row: LDOS at x = 0, both spins, energy axis e - mu (meV).
- `figures/fig2_ldos_vs_B.png`: hbar w_x = 1.5 meV, B = 0..10 T, traces offset vertically,
  up solid, down dashed, energy e - mu.
- `figures/wire_reference.png`: n(y) of the clean wire, V_eff(y), subband energies, mu.
- Write a short `results/summary.txt`: mu_wire, net spins, iteration counts, run times.

---------------------------------------------------------------------------------------------
## M8. Convergence spot checks (only after M7)

For hbar w_x = 1.5 meV, B = 0: E_cut 20 meV; Lx = 7.5 um; kT 0.025 meV; interp = "quadratic".
Report the change of M and of the peak of n_1D,up - n_1D,dn. Do not start these before the
Fig. 1/2 data exist.

---------------------------------------------------------------------------------------------
## Open questions (also in docs/OPEN_QUESTIONS.md; ask the user / advisor)
1. TC coefficients and units: verify against the paper (M3).
2. Spin interpolation used by HMW (exchange vs quadratic) — not stated in the PRL.
3. Gate model and distance a (image plane at 2a) — not stated in the PRL.
4. Temperature / smearing of HMW.
5. Which HMW Fig. 1 panel corresponds to which hbar w_x.
