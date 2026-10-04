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

## New results (2026-10-04)

### 1) B = 0 rerun, wx 1.0, A, quadratic, from the converged 0.25 T state (Pulay alpha 0.1, history 12, maxiter 400)
```
B = 0 rerun: wx = 1.0, model A_quad, start results/qpc_wx1.0_A_quad_B0.25.npz, alpha = 0.1, history = 12, maxiter = 400, kT = 0.05 meV, mu = 8.95648 meV
  it    M_loc M_win300      N_up      N_dn    nup(0)    ndn(0) cnt_up cnt_dn  dmin_up  dmin_dn     resid
   1   2.3796   1.1153   69.9882   67.6087  0.012662  0.000574     24     23  -0.0037  -0.0127  7.04e-01
  10   1.0195   1.0361   69.2866   68.2671  0.012651  0.000620     23     25  +0.0016  +0.0151  1.32e-02
  20   0.9474   0.9661   69.2514   68.3039  0.012578  0.000810     23     25  +0.0020  +0.0145  1.10e-02
  30   0.9552   0.9656   69.2547   68.2995  0.012580  0.000814     23     25  +0.0020  +0.0146  1.20e-02
  40   0.9998   1.0114   69.2760   68.2762  0.012639  0.000685     23     25  +0.0018  +0.0149  1.18e-02
  50   0.9927   1.0073   69.2728   68.2801  0.012636  0.000693     23     25  +0.0018  +0.0149  9.80e-03
  60   1.0083   1.0218   69.2799   68.2716  0.012655  0.000654     23     25  +0.0017  +0.0150  9.91e-03
  70   0.9558   0.9727   69.2552   68.2994  0.012594  0.000787     23     25  +0.0020  +0.0146  7.53e-03
  80   0.9415   0.9563   69.2482   68.3067  0.012578  0.000831     23     25  +0.0020  +0.0145  7.70e-03
  90   0.8951   0.9100   69.2258   68.3307  0.012523  0.000965     23     25  +0.0022  +0.0141  1.17e-02
 100   1.0105   1.0240   69.2809   68.2705  0.012661  0.000647     23     25  +0.0017  +0.0150  9.44e-03
 110   1.0362   1.0527   69.2929   68.2567  0.012698  0.000571     23     25  +0.0016  +0.0153  1.13e-02
 120   0.8780   0.8924   69.2173   68.3394  0.012503  0.001017     23     25  +0.0023  +0.0140  1.40e-02
 130   0.8314   0.8452   69.1942   68.3628  0.012448  0.001160     23     25  -0.0022  +0.0137  2.36e-02
 140   0.9046   0.9194   69.2304   68.3258  0.012534  0.000938     23     25  +0.0022  +0.0142  1.01e-02
 150   1.0448   1.0613   69.2969   68.2521  0.012709  0.000549     23     25  +0.0015  +0.0153  1.25e-02
 160   1.0507   1.0672   69.2996   68.2489  0.012717  0.000535     23     25  +0.0015  +0.0154  1.37e-02
 170   1.0683   1.0853   69.3077   68.2394  0.012741  0.000490     23     24  +0.0014  +0.0155  1.74e-02
 180   0.9552   0.9705   69.2548   68.2995  0.012595  0.000792     23     25  +0.0020  +0.0146  6.54e-03
 190   0.9909   1.0066   69.2717   68.2808  0.012639  0.000692     23     25  +0.0018  +0.0149  6.88e-03
 200   1.0628   1.0795   69.3052   68.2424  0.012733  0.000504     23     24  +0.0014  +0.0155  1.61e-02
 210   1.1100   1.1273   69.3268   68.2167  0.012799  0.000392     23     24  +0.0011  +0.0159  2.92e-02
 220   0.9160   0.9308   69.2360   68.3200  0.012547  0.000904     23     25  +0.0022  +0.0143  8.69e-03
 230   1.0188   1.0349   69.2848   68.2661  0.012674  0.000617     23     25  +0.0016  +0.0151  8.76e-03
 240   0.9949   1.0107   69.2737   68.2788  0.012644  0.000681     23     25  +0.0018  +0.0149  6.60e-03
 250   0.9731   0.9886   69.2634   68.2903  0.012616  0.000741     23     25  +0.0019  +0.0147  5.94e-03
 260   0.9735   0.9890   69.2635   68.2900  0.012617  0.000740     23     25  +0.0019  +0.0147  6.05e-03
 270   0.8854   0.8997   69.2210   68.3356  0.012511  0.000996     23     25  +0.0023  +0.0141  1.27e-02
 280   1.0851   1.1021   69.3154   68.2303  0.012764  0.000450     23     24  +0.0013  +0.0157  2.15e-02
 290   1.0122   1.0283   69.2818   68.2696  0.012666  0.000634     23     25  +0.0017  +0.0151  8.04e-03
 300   0.9647   0.9801   69.2594   68.2947  0.012606  0.000764     23     25  +0.0019  +0.0147  5.85e-03
 310   0.9517   0.9670   69.2532   68.3015  0.012590  0.000801     23     25  +0.0020  +0.0146  6.13e-03
 320   0.9061   0.9207   69.2312   68.3251  0.012535  0.000934     23     25  +0.0022  +0.0142  9.61e-03
 330   0.9073   0.9219   69.2317   68.3245  0.012537  0.000930     23     25  +0.0022  +0.0142  9.51e-03
 340   0.7993   0.8125   69.1781   68.3787  0.012411  0.001259     23     25  -0.0014  +0.0134  3.15e-02
 350   0.9238   0.9386   69.2397   68.3159  0.012556  0.000882     23     25  +0.0021  +0.0144  8.01e-03
 360   0.9051   0.9197   69.2306   68.3255  0.012534  0.000937     23     25  +0.0022  +0.0142  1.03e-02
 370   0.9501   0.9653   69.2523   68.3022  0.012589  0.000806     23     25  +0.0020  +0.0146  6.87e-03
 380   0.9233   0.9381   69.2394   68.3161  0.012556  0.000884     23     25  +0.0021  +0.0144  8.26e-03
 390   0.9025   0.9170   69.2293   68.3268  0.012531  0.000945     23     25  +0.0022  +0.0142  1.06e-02
 400   0.9427   0.9578   69.2488   68.3060  0.012580  0.000827     23     25  +0.0020  +0.0145  7.07e-03

converged: False after 400 iterations (1391 s); table: notes/wx1.0_A_quad_B0_rerun_table.txt
 B[T]    M_loc M_win300         M   M_wire   it    t[s]         N   nup_far   ndn_far  top_up  top_dn mu-top_up mu-top_dn    D_dn   D_up0  conv
    0   1.0151   1.0312    0.9427  -0.0000  400  1391.4  137.5548  0.014000  0.014000  0.3975  1.3386   +0.5321   -0.4090  +0.710  -0.621 False
  M = 0.9427, M_loc = 1.0151   # HMW: ~0.85, M_win300 = 1.0312
  [up] peaks: x=-76.2 nm h=0.3975, x=+76.2 nm h=0.3975   # HMW: two peaks ~0.6 meV at x ~ +-80 nm (nearly flat top over ~160 nm)
  [up] e_0(0)-e_0(far) = 0.0074 meV   # HMW: ~0.45 meV (shallow dip ~0.15 meV);  mu - e_0(far) = 0.9296 meV   # HMW: ~1.1 meV
  [dn] peaks: x=+0.0 nm h=1.3386   # HMW: one peak ~1.2 meV (just above mu)
  [dn] e_0(0)-e_0(far) = 1.3386 meV;  mu - e_0(far) = 0.9296 meV   # HMW: ~1.1 meV
  n_1D(0) = 0.013281 nm^-1 (up 0.012666, dn 0.000615)   # HMW: ~1.5e-2 nm^-1 total
  D_dn = top_dn - top_unpol = +0.7104 meV, D_up0 = e_0,up(0) - top_unpol = -0.6208 meV   # HMW: D_dn ~ +0.6, D_up0 ~ -0.18
  LDOS eta=0.05: up resonance e-mu = -0.7196 meV (# HMW ~ -0.5 meV), FWHM = 0.1495 meV (# HMW Gamma ~0.1 meV, incl. 2 eta); dn onset e-mu = +0.2624 meV (# HMW ~ +0.1 meV); U = 0.9820 meV (# HMW ~0.6 meV)
  LDOS eta=0.1: up resonance e-mu = -0.7156 meV (# HMW ~ -0.5 meV), FWHM = 0.2632 meV (# HMW Gamma ~0.1 meV, incl. 2 eta); dn onset e-mu = +0.1384 meV (# HMW ~ +0.1 meV); U = 0.8540 meV (# HMW ~0.6 meV)
NOT converged: M_loc over the last 100 iterations: mean = 0.9234, std = 0.0778, min = 0.5283, max = 1.1249
swings |dM_loc| > std: 12 of 99 steps; with a level-count change near mu: 7; count changes overall: 8; corr(|dM_loc|, count change) = 0.704
```
(full per-iteration table: notes/wx1.0_A_quad_B0_rerun_table.txt)

### 2) same B = 0 rerun with kT = 0.08 meV (smearing sensitivity)
```
B = 0 rerun: wx = 1.0, model A_quad, start results/qpc_wx1.0_A_quad_B0.25.npz, alpha = 0.1, history = 12, maxiter = 300, kT = 0.08 meV, mu = 8.92578 meV
  it    M_loc M_win300      N_up      N_dn    nup(0)    ndn(0) cnt_up cnt_dn  dmin_up  dmin_dn     resid

converged: False after 300 iterations (1108 s); table: notes/wx1.0_A_quad_B0_rerun_kT0.08_table.txt
 B[T]    M_loc M_win300         M   M_wire   it    t[s]         N   nup_far   ndn_far  top_up  top_dn mu-top_up mu-top_dn    D_dn   D_up0  conv
  M = 0.9651, M_loc = 0.9730   # HMW: ~0.85, M_win300 = 0.9769
  [up] peaks: x=-76.2 nm h=0.3621, x=+76.2 nm h=0.3621   # HMW: two peaks ~0.6 meV at x ~ +-80 nm (nearly flat top over ~160 nm)
  [up] e_0(0)-e_0(far) = -0.0153 meV   # HMW: ~0.45 meV (shallow dip ~0.15 meV);  mu - e_0(far) = 0.9099 meV   # HMW: ~1.1 meV
  [dn] peaks: x=+0.0 nm h=1.3680   # HMW: one peak ~1.2 meV (just above mu)
  [dn] e_0(0)-e_0(far) = 1.3680 meV;  mu - e_0(far) = 0.9099 meV   # HMW: ~1.1 meV
  n_1D(0) = 0.013394 nm^-1 (up 0.012663, dn 0.000731)   # HMW: ~1.5e-2 nm^-1 total
  D_dn = top_dn - top_unpol = +0.7398 meV, D_up0 = e_0,up(0) - top_unpol = -0.6435 meV   # HMW: D_dn ~ +0.6, D_up0 ~ -0.18
  LDOS eta=0.05: up resonance e-mu = -0.7259 meV (# HMW ~ -0.5 meV), FWHM = 0.1490 meV (# HMW Gamma ~0.1 meV, incl. 2 eta); dn onset e-mu = +0.3461 meV (# HMW ~ +0.1 meV); U = 1.0720 meV (# HMW ~0.6 meV)
  LDOS eta=0.1: up resonance e-mu = -0.7219 meV (# HMW ~ -0.5 meV), FWHM = 0.2634 meV (# HMW Gamma ~0.1 meV, incl. 2 eta); dn onset e-mu = +0.1641 meV (# HMW ~ +0.1 meV); U = 0.8860 meV (# HMW ~0.6 meV)
NOT converged: M_loc over the last 100 iterations: mean = 1.0641, std = 0.1753, min = 0.6432, max = 1.4360
swings |dM_loc| > std: 17 of 99 steps; with a level-count change near mu: 8; count changes overall: 21; corr(|dM_loc|, count change) = 0.290
```
(per-iteration table: notes/wx1.0_A_quad_B0_rerun_kT0.08_table.txt)

### 3/4) Janak ramp ramp_wx1.5_A_exch (6 -> 0 T); M_loc(B=0) exch = 0.7836
```
 B[T]    M_loc M_win300         M   M_wire   it    t[s]         N   nup_far   ndn_far  top_up  top_dn mu-top_up mu-top_dn    D_dn   D_up0  conv
    6   0.1142   0.0699   29.1420  29.0279   44   156.2  138.3470  0.016904  0.011097  0.2066  1.2500   +0.8328   -0.5568  +0.541  -0.503  True
    4   0.1636   0.1325   23.7079  23.5443   36   127.3  138.3652  0.016355  0.011645  0.2121  1.2277   +0.8018   -0.4679  +0.518  -0.497  True
    2   0.2584   0.2379   17.5201  17.2617   37   133.5  138.3846  0.015727  0.012274  0.2283  1.2097   +0.7591   -0.3808  +0.500  -0.481  True
    1   0.3878   0.3642   12.9037  12.5159   38   135.3  138.3956  0.015252  0.012748  0.2412  1.2041   +0.7290   -0.3355  +0.495  -0.468  True
  0.5   0.5339   0.5037    9.0142   8.4804   45   158.9  138.4013  0.014849  0.013152  0.2482  1.2042   +0.7089   -0.3107  +0.495  -0.461  True
 0.25   0.6588   0.6293    5.8215   5.1628   43   155.9  138.4036  0.014517  0.013484  0.2502  1.2069   +0.6964   -0.2974  +0.498  -0.459  True
    0   0.7837   0.7948    0.7836  -0.0000   59   213.6  138.4046  0.014000  0.014000  0.2484  1.2130   +0.6812   -0.2833  +0.504  -0.461  True
largest B with two spin-up barrier peaks: 6.0 T
B at the minimum of M_loc(B): 6 T (M_loc = 0.1142)
M_loc(B = 0): 0.7837
M_loc(B) in ramp order: [(6.0, 0.1142), (4.0, 0.1636), (2.0, 0.2584), (1.0, 0.3878), (0.5, 0.5339), (0.25, 0.6588), (0.0, 0.7837)]
-- per-B features:
##### B = 6 T: leads (interacting) mu = 8.86248 meV, M_wire = 29.0279 (2.2 s)
  M = 29.1420, M_loc = 0.1142, M_win300 = 0.0699
  [up] peaks: x=-85.7 nm h=0.1258, x=+0.0 nm h=0.2066, x=+85.7 nm h=0.1258
  [up] e_0(0)-e_0(far) = 0.2066 meV;  mu - e_0(far) = 1.0394 meV
  [dn] peaks: x=+0.0 nm h=1.2500
  [dn] e_0(0)-e_0(far) = 1.2500 meV;  mu - e_0(far) = 0.6932 meV
  n_1D(0) = 0.011813 nm^-1 (up 0.011438, dn 0.000375)
  D_dn = top_dn - top_unpol = +0.5406 meV, D_up0 = e_0,up(0) - top_unpol = -0.5028 meV
##### B = 4 T: leads (interacting) mu = 8.89359 meV, M_wire = 23.5443 (1.3 s)
  M = 23.7079, M_loc = 0.1636, M_win300 = 0.1325
  [up] peaks: x=-76.2 nm h=0.1583, x=+0.0 nm h=0.2121, x=+76.2 nm h=0.1583
  [up] e_0(0)-e_0(far) = 0.2121 meV;  mu - e_0(far) = 1.0139 meV
  [dn] peaks: x=+0.0 nm h=1.2277
  [dn] e_0(0)-e_0(far) = 1.2277 meV;  mu - e_0(far) = 0.7597 meV
  n_1D(0) = 0.011757 nm^-1 (up 0.011222, dn 0.000535)
  D_dn = top_dn - top_unpol = +0.5183 meV, D_up0 = e_0,up(0) - top_unpol = -0.4973 meV
##### B = 2 T: leads (interacting) mu = 8.92526 meV, M_wire = 17.2617 (1.3 s)
  M = 17.5201, M_loc = 0.2584, M_win300 = 0.2379
  [up] peaks: x=-76.2 nm h=0.1938, x=+0.0 nm h=0.2283, x=+76.2 nm h=0.1938
  [up] e_0(0)-e_0(far) = 0.2283 meV;  mu - e_0(far) = 0.9874 meV
  [dn] peaks: x=+0.0 nm h=1.2097
  [dn] e_0(0)-e_0(far) = 1.2097 meV;  mu - e_0(far) = 0.8289 meV
  n_1D(0) = 0.011791 nm^-1 (up 0.011015, dn 0.000775)
  D_dn = top_dn - top_unpol = +0.5003 meV, D_up0 = e_0,up(0) - top_unpol = -0.4811 meV
##### B = 1 T: leads (interacting) mu = 8.94172 meV, M_wire = 12.5159 (1.3 s)
  M = 12.9037, M_loc = 0.3878, M_win300 = 0.3642
  [up] peaks: x=-76.2 nm h=0.2030, x=+0.0 nm h=0.2412, x=+76.2 nm h=0.2030
  [up] e_0(0)-e_0(far) = 0.2412 meV;  mu - e_0(far) = 0.9702 meV
  [dn] peaks: x=+0.0 nm h=1.2041
  [dn] e_0(0)-e_0(far) = 1.2041 meV;  mu - e_0(far) = 0.8686 meV
  n_1D(0) = 0.011869 nm^-1 (up 0.010907, dn 0.000963)
  D_dn = top_dn - top_unpol = +0.4947 meV, D_up0 = e_0,up(0) - top_unpol = -0.4682 meV
  LDOS eta=0.05: up resonance e-mu = -0.6962 meV, FWHM = 0.3451 meV; dn onset e-mu = +0.1898 meV; U = 0.8860 meV
  LDOS eta=0.1: up resonance e-mu = -0.6822 meV, FWHM = 0.4920 meV; dn onset e-mu = +0.1318 meV; U = 0.8140 meV
##### B = 0.5 T: leads (interacting) mu = 8.95024 meV, M_wire = 8.4804 (1.4 s)
  M = 9.0142, M_loc = 0.5339, M_win300 = 0.5037
  [up] peaks: x=-66.7 nm h=0.2110, x=+0.0 nm h=0.2482, x=+66.7 nm h=0.2110
  [up] e_0(0)-e_0(far) = 0.2482 meV;  mu - e_0(far) = 0.9571 meV
  [dn] peaks: x=+0.0 nm h=1.2042
  [dn] e_0(0)-e_0(far) = 1.2042 meV;  mu - e_0(far) = 0.8935 meV
  n_1D(0) = 0.011938 nm^-1 (up 0.010842, dn 0.001096)
  D_dn = top_dn - top_unpol = +0.4948 meV, D_up0 = e_0,up(0) - top_unpol = -0.4612 meV
  LDOS eta=0.05: up resonance e-mu = -0.6631 meV, FWHM = 0.3547 meV; dn onset e-mu = +0.1649 meV; U = 0.8280 meV
  LDOS eta=0.1: up resonance e-mu = -0.6511 meV, FWHM = 0.5003 meV; dn onset e-mu = +0.1069 meV; U = 0.7580 meV
##### B = 0.25 T: leads (interacting) mu = 8.95427 meV, M_wire = 5.1628 (1.5 s)
  M = 5.8215, M_loc = 0.6588, M_win300 = 0.6293
  [up] peaks: x=-66.7 nm h=0.2126, x=+0.0 nm h=0.2502, x=+66.7 nm h=0.2126
  [up] e_0(0)-e_0(far) = 0.2502 meV;  mu - e_0(far) = 0.9466 meV
  [dn] peaks: x=+0.0 nm h=1.2069
  [dn] e_0(0)-e_0(far) = 1.2069 meV;  mu - e_0(far) = 0.9095 meV
  n_1D(0) = 0.011984 nm^-1 (up 0.010803, dn 0.001181)
  D_dn = top_dn - top_unpol = +0.4975 meV, D_up0 = e_0,up(0) - top_unpol = -0.4592 meV
  LDOS eta=0.05: up resonance e-mu = -0.6466 meV, FWHM = 0.3604 meV; dn onset e-mu = +0.1494 meV; U = 0.7960 meV
  LDOS eta=0.1: up resonance e-mu = -0.6326 meV, FWHM = 0.5063 meV; dn onset e-mu = +0.0914 meV; U = 0.7240 meV
##### B = 0 T: leads (interacting) mu = 8.95648 meV, M_wire = -0.0000 (1.7 s)
  M = 0.7836, M_loc = 0.7837   # HMW: ~0.93, M_win300 = 0.7948
  [up] peaks: x=-66.7 nm h=0.2089, x=+0.0 nm h=0.2484, x=+66.7 nm h=0.2089   # HMW: ~0.6 meV
  [up] e_0(0)-e_0(far) = 0.2484 meV;  mu - e_0(far) = 0.9296 meV   # HMW: ~1.1 meV
  [dn] peaks: x=+0.0 nm h=1.2130   # HMW: ~1.3 meV
  [dn] e_0(0)-e_0(far) = 1.2130 meV;  mu - e_0(far) = 0.9296 meV   # HMW: ~1.1 meV
  n_1D(0) = 0.012040 nm^-1 (up 0.010755, dn 0.001285)   # HMW: ~1.4e-2 nm^-1 total
  D_dn = top_dn - top_unpol = +0.5036 meV, D_up0 = e_0,up(0) - top_unpol = -0.4610 meV   # HMW: D_dn ~ +0.6
  LDOS eta=0.05: up resonance e-mu = -0.6236 meV (# HMW ~ -0.5 meV), FWHM = 0.3664 meV (# HMW Gamma ~0.1 meV, incl. 2 eta); dn onset e-mu = +0.1284 meV (# HMW ~ +0.1 meV); U = 0.7520 meV (# HMW ~0.6 meV)
  LDOS eta=0.1: up resonance e-mu = -0.6116 meV (# HMW ~ -0.5 meV), FWHM = 0.5139 meV (# HMW Gamma ~0.1 meV, incl. 2 eta); dn onset e-mu = +0.0744 meV (# HMW ~ +0.1 meV); U = 0.6860 meV (# HMW ~0.6 meV)
B at the minimum of M_loc(B): 6 T (M_loc = 0.1142)
```

### 5) figures
```
figures/fig1_wx1.0_quad.png: results/qpc_wx1.0_A_quad_B0.npz, M_loc = 0.911, mu - e0(far) = 0.930 meV
figures/fig1_wx1.0_exch.png: results/qpc_wx1.0_A_B0.npz, M_loc = 1.074, mu - e0(far) = 0.930 meV
```
Figures: figures/fig1_wx1.0_quad.png, figures/fig1_wx1.0_exch.png (PRL Fig. 1 axes; n_up - n_dn
below 0 in the Friedel oscillations is clipped by the paper's 0 ... 1.5 axis).
