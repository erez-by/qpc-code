"""Iterative eigensolver for the plane-wave Hamiltonian: block LOBPCG.

Problem: the lowest k eigenpairs of the Hermitian H (n_pw x n_pw) with H only available
as a block matrix-vector product, apply_H(X) -> H X (Hamiltonian.apply, FFT based).

Algorithm (Knyazev, SIAM J. Sci. Comput. 23, 517 (2001); the stable basis choice of
Hetmaniuk & Lehoucq, J. Comput. Phys. 218, 324 (2006)):

    X_0  orthonormal start block (n x k), Rayleigh-Ritz in span(X_0)
    loop:
        R = H X - X Theta                  residuals (columns r_i = H x_i - theta_i x_i)
        W = K R                            preconditioned residuals (search directions)
        P                                  previous update directions (the "conjugate" part)
        Rayleigh-Ritz in span[X, W, P]     -> new X, Theta, and P = the [W, P] part of new X

LOBPCG is the block version of locally optimal CG on the Rayleigh quotient
rho(x) = x^H H x / x^H x: in every step it takes the best vector in the 3k-dimensional
trial space instead of doing a line search along one direction.

Preconditioner K ~ (H - sigma)^-1, applied to the residuals. Two choices:

(1) "block" (default): the x-averaged Hamiltonian H_0 = T + <V>_x, i.e. H restricted to
    the couplings with Delta G_x = 0. Since <V>_x depends only on y, H_0 is block diagonal:
    one small block (<= ~17 plane waves in G_y) per G_x. Each block is diagonalised once
    (cost ~ms), then K = (H_0 - sigma)^-1 is exact on its own eigenbasis. For the clean wire
    H_0 = H, and for the QPC the barrier is a localised perturbation of H_0. This is the
    same physics as the "mode-space" description: transverse modes x longitudinal waves.
    sigma = min(theta_0, lowest eigenvalue of H_0) - delta keeps K positive definite,
    which the LOBPCG convergence theory assumes (Knyazev 2001, Sec. 3).

(2) "tpa": Teter-Payne-Allan, Phys. Rev. B 40, 12255 (1989), the standard choice in
    plane-wave electronic-structure codes, diagonal in G:   s = T_G / E_kin,i,
    K_i(G) = (27 + 18 s + 12 s^2 + 8 s^3) / (27 + 18 s + 12 s^2 + 8 s^3 + 16 s^4)
K -> 1 for T_G << E_kin,i (low-G components untouched) and K ~ 1/(2 s) = E_kin,i/(2 T_G)
for T_G >> E_kin,i: it damps the high-G components, where H is dominated by T_G, like
an approximate inverse (T_G - theta)^-1 would, but smooth and never singular.
    It works when the stiff part of H is the kinetic energy (atoms, solids). Here it is
    not: the confinement (1/2) w_y^2 y^2 reaches ~45 meV at the cell edge, far above
    E_cut = 15 meV, so the large eigenvalues of H come from V, and TPA barely helps
    (measured on Day 2: 216 iterations with TPA, 40 with "block").

Conventions follow qpc/hamiltonian.py: a column is a coefficient vector over the basis
(order np.nonzero(mask)), normalised as sum |c|^2 = 1. Energies in Ha*.
"""
from dataclasses import dataclass, field
import numpy as np
import scipy.linalg as la


# ----------------------------------------------------------------------------- helpers
def _hermitize(A):
    return 0.5 * (A + A.conj().T)


def svqb(S, drop_tol=1e-10):
    """Orthonormalise the columns of S, dropping (near-)linearly dependent directions.

    'SVQB' of Stathopoulos & Wu, SIAM J. Sci. Comput. 23, 2165 (2002):
    M = S^H S, scale to unit diagonal, M~ = U diag(lam) U^H,
    Q = S D U lam^{-1/2}  (keeping only lam > drop_tol * max lam).
    Unlike Gram-Schmidt / Cholesky, this does not break down when S is almost
    rank deficient, which happens in LOBPCG close to convergence (W and P become
    nearly parallel).
    """
    norms = np.linalg.norm(S, axis=0)
    keep = norms > 0
    S = S[:, keep] / norms[keep]
    if S.shape[1] == 0:
        return S
    M = _hermitize(S.conj().T @ S)
    lam, U = la.eigh(M)
    good = lam > drop_tol * lam.max()
    return S @ (U[:, good] / np.sqrt(lam[good]))


def project_out(S, X):
    """S - X (X^H S), applied twice ('twice is enough', Giraud et al. 2005)."""
    for _ in range(2):
        S = S - X @ (X.conj().T @ S)
    return S


def tpa_preconditioner(kinetic, X):
    """TPA weights K (n x m), one column per vector of X (kinetic = T_G on the basis)."""
    e_kin = np.real(np.einsum("g,gi->i", kinetic, np.abs(X) ** 2)) / np.sum(np.abs(X) ** 2, axis=0)
    s = kinetic[:, None] / np.maximum(e_kin, 1e-12)[None, :]
    num = 27.0 + 18.0 * s + 12.0 * s ** 2 + 8.0 * s ** 3
    return num / (num + 16.0 * s ** 4)


class XAveragedPreconditioner:
    """K = (H_0 - sigma)^-1 with H_0 = T + <V>_x (block diagonal in G_x), see module docstring.

    Matrix elements: <G|H_0|G'> = delta(Gx, Gx') [ T_G delta(Gy, Gy') + V_hat[0, iy - iy'] ],
    because V_hat[0, m] is the Fourier coefficient of the x-average of V.
    Rebuild it whenever the potential changes (every SCF step); it costs a few ms.
    """

    def __init__(self, ham, delta=0.01):
        Ny = ham.grid.Ny
        Vy = ham.V_hat[0, :]
        cols, gid = np.unique(ham.ix, return_inverse=True)        # one group per G_x
        sizes = np.bincount(gid)
        order = np.argsort(gid, kind="stable")
        pos = np.empty(ham.n_pw, dtype=int)
        pos[order] = np.concatenate([np.arange(m) for m in sizes])
        smax = sizes.max()
        big = 1e6                                                   # padding rows: K ~ 0 there
        A = np.zeros((cols.size, smax, smax), dtype=complex)
        A[:] = big * np.eye(smax)
        for g in range(cols.size):
            idx = np.nonzero(gid == g)[0]
            iy = ham.iy[idx]
            A[g, :idx.size, :idx.size] = Vy[(iy[:, None] - iy[None, :]) % Ny]
            A[g, :idx.size, :idx.size] += np.diag(ham.kinetic[idx])
        self.d, self.U = np.linalg.eigh(A)                          # batched (G, s), (G, s, s)
        self.gid, self.pos, self.delta = gid, pos, float(delta)
        self.e0 = float(self.d.min())                               # lowest level of H_0

    def __call__(self, R, theta):
        sigma = min(float(np.min(theta)), self.e0) - self.delta
        Rp = np.zeros(self.d.shape + (R.shape[1],), dtype=complex)
        Rp[self.gid, self.pos] = R
        Y = np.swapaxes(self.U, 1, 2).conj() @ Rp
        Y /= (self.d - sigma)[..., None]
        return (self.U @ Y)[self.gid, self.pos]


@dataclass
class EigResult:
    eigvals: np.ndarray          # (k,) ascending, Ha*
    X: np.ndarray                # (n, k) orthonormal eigenvector block
    resnorms: np.ndarray         # (k,) ||H x_i - theta_i x_i||
    n_iter: int
    n_matvec: int                # number of single-vector H applications
    converged: bool
    history: list = field(default_factory=list)   # max residual of the wanted bands per iteration


# ----------------------------------------------------------------------------- LOBPCG
def lobpcg(apply_H, X0, n_wanted=None, kinetic=None, tol=1e-6, maxiter=300,
           precond="tpa", drop_tol=1e-10):
    """Lowest eigenpairs of a Hermitian operator by block LOBPCG with soft locking.

    Parameters
    ----------
    apply_H  : callable, (n, m) -> (n, m), the block product H X
    X0       : (n, k) start block (warm start: previous eigenvectors). Need not be orthonormal.
    n_wanted : how many of the k lowest pairs must converge (default k). The remaining
               k - n_wanted columns are a buffer: they make the convergence of band n_wanted
               depend on the gap to band k+1, not to band n_wanted+1.
    kinetic  : (n,) T_G, needed for precond="tpa"
    tol      : residual tolerance ||H x - theta x|| in Ha*. Eigenvalue error ~ tol^2 / gap.
    precond  : None, "tpa", or a callable K(R, theta) -> W acting on the active residual
               columns (e.g. XAveragedPreconditioner)
    """
    X = np.array(X0, dtype=complex, copy=True)
    n, k = X.shape
    n_wanted = k if n_wanted is None else int(n_wanted)
    if not (1 <= n_wanted <= k):
        raise ValueError("need 1 <= n_wanted <= k")
    if 3 * k > n:
        raise ValueError(f"block size k={k} too large for n={n}: use the dense solver")
    if precond == "tpa" and kinetic is None:
        raise ValueError("precond='tpa' needs the kinetic diagonal")

    # initial Rayleigh-Ritz in span(X0)
    X = svqb(X, drop_tol)
    if X.shape[1] < k:                                  # rank-deficient start: pad randomly
        rng = np.random.default_rng(1)
        extra = rng.standard_normal((n, k - X.shape[1])) + 0j
        X = svqb(np.hstack([X, project_out(extra, X)]), drop_tol)
    HX = apply_H(X)
    n_matvec = k
    theta, C = la.eigh(_hermitize(X.conj().T @ HX))
    X, HX = X @ C, HX @ C
    P = HP = None
    history = []

    for it in range(1, maxiter + 1):
        R = HX - X * theta
        resn = np.linalg.norm(R, axis=0)
        history.append(float(resn[:n_wanted].max()))
        if np.all(resn[:n_wanted] < tol):
            return EigResult(theta, X, resn, it - 1, n_matvec, True, history)

        # soft locking: converged columns stay in X (and in Rayleigh-Ritz) but get no
        # new search directions -> fewer matvecs, no loss of orthogonality
        active = resn >= tol
        W = R[:, active]
        if precond == "tpa":
            W = tpa_preconditioner(kinetic, X[:, active]) * W
        elif callable(precond):
            W = precond(W, theta[active])

        blocks = [W] if P is None else [W, P[:, active]]
        S = svqb(project_out(np.hstack(blocks), X), drop_tol)
        S = svqb(project_out(S, X), drop_tol)            # second pass for orthogonality
        HS = apply_H(S)
        n_matvec += S.shape[1]

        # Rayleigh-Ritz in the orthonormal basis [X, S]
        XHS = X.conj().T @ HS
        G = np.block([[np.diag(theta).astype(complex), XHS],
                      [XHS.conj().T, S.conj().T @ HS]])
        theta, C = la.eigh(_hermitize(G), subset_by_index=[0, k - 1])
        Cx, Cs = C[:k], C[k:]
        P, HP = S @ Cs, HS @ Cs                          # Hetmaniuk-Lehoucq: P from [W, P] part
        X, HX = X @ Cx + P, HX @ Cx + HP

    R = HX - X * theta
    resn = np.linalg.norm(R, axis=0)
    return EigResult(theta, X, resn, maxiter, n_matvec, bool(np.all(resn[:n_wanted] < tol)), history)


# ----------------------------------------------------------------------------- front end
def random_start(n, k, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, k)) + 1j * rng.standard_normal((n, k))


def dense_lowest(ham, n_bands):
    """Exact lowest n_bands pairs from the dense matrix; only that part of the spectrum is
    computed (LAPACK ?syevr / ?heevr via subset_by_index), usually ~2x faster than all."""
    H = ham.dense()
    if np.max(np.abs(H.imag)) < 1e-12 * np.max(np.abs(H)):
        H = H.real
    return la.eigh(H, subset_by_index=[0, n_bands - 1], driver="evr")


def lowest_states(ham, n_bands, method="lobpcg", X0=None, buffer=None, tol=1e-6,
                  maxiter=300, precond="block"):
    """Lowest n_bands eigenpairs of a Hamiltonian object (set_potential already called).

    method="dense"  : ham.eigh_dense (exact, O(n_pw^3), for tests and small problems)
    method="lobpcg" : matrix-free; X0 = previous eigenvectors for a warm start (the SCF
                      loop passes the last result's .X_full). buffer default max(5, 10%).
    precond         : "block" (default), "tpa" or None
    Returns EigResult with eigvals/X restricted to n_bands.
    """
    if method == "dense":
        w, v = dense_lowest(ham, n_bands)
        return EigResult(w, v.astype(complex), np.zeros(n_bands), 0, 0, True)
    if method != "lobpcg":
        raise ValueError(method)
    buffer = max(5, int(0.1 * n_bands)) if buffer is None else int(buffer)
    k = n_bands + buffer
    if X0 is None:
        X0 = random_start(ham.n_pw, k)
    elif X0.shape[1] < k:
        X0 = np.hstack([X0, random_start(ham.n_pw, k - X0.shape[1], seed=7)])
    else:
        X0 = X0[:, :k]
    if precond == "block":
        precond = XAveragedPreconditioner(ham)
    res = lobpcg(ham.apply, X0, n_wanted=n_bands, kinetic=ham.kinetic, tol=tol,
                 maxiter=maxiter, precond=precond)
    # keep the full block (with buffer) in X_full for the next warm start
    out = EigResult(res.eigvals[:n_bands], res.X[:, :n_bands], res.resnorms[:n_bands],
                    res.n_iter, res.n_matvec, res.converged, res.history)
    out.X_full = res.X
    return out
