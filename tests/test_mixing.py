import numpy as np

from qpc.mixing import Pulay


def contraction(n=200, radius=0.95, seed=0):
    """rho -> A rho + b with A symmetric, spectral radius 0.95, fixed point rho* > 0."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    lam = np.linspace(-radius, radius, n)
    A = (Q * lam) @ Q.T
    rho_star = 1.0 + 0.1 * rng.uniform(size=n)
    b = rho_star - A @ rho_star
    return A, b, rho_star


def iterate(mixer, A, b, rho0, tol=1e-10, maxit=5000):
    rho = rho0.copy()
    for it in range(1, maxit + 1):
        out = A @ rho + b
        if np.linalg.norm(out - rho) < tol:
            return it, rho
        rho = mixer.step(rho, out)
    return maxit, rho


def test_history_one_is_linear_mixing():
    rng = np.random.default_rng(1)
    p = Pulay(alpha=0.3, history=1)
    for _ in range(4):
        rin, rout = rng.uniform(size=50), rng.uniform(size=50)
        np.testing.assert_allclose(p.step(rin, rout), rin + 0.3 * (rout - rin), atol=1e-15)


def test_diis_faster_than_linear():
    A, b, rho_star = contraction()
    rho0 = np.ones_like(b)
    it_lin, r_lin = iterate(Pulay(alpha=0.2, history=1), A, b, rho0)
    it_diis, r_diis = iterate(Pulay(alpha=0.2, history=8), A, b, rho0)
    assert np.abs(r_diis - rho_star).max() < 1e-8
    assert it_diis < 200
    assert it_diis * 5 < it_lin, (it_diis, it_lin)


def test_output_nonnegative():
    rng = np.random.default_rng(2)
    p = Pulay(alpha=0.5, history=4)
    for _ in range(10):
        out = p.step(rng.uniform(0, 1, 100), rng.uniform(-1, 1, 100))
        assert out.min() >= 0


def test_ill_conditioned_history_dropped():
    """Repeating the same pair makes B singular; the mixer must stay finite."""
    p = Pulay(alpha=0.2, history=8)
    rin, rout = np.full(10, 1.0), np.full(10, 1.1)
    for _ in range(5):
        r = p.step(rin, rout)
    assert np.all(np.isfinite(r))
    np.testing.assert_allclose(r, rin + 0.2 * (rout - rin))
