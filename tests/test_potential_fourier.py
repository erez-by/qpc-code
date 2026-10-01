import numpy as np
import mpmath as mp
import pytest
from qpc.units import Units
from qpc.potential import QPCParams, V_barrier, V_qpc
from qpc.fourier import min_image, ft_numpy, ft_sech2_exact, ft_mpmath


@pytest.mark.parametrize("hwx, d_nm", [(1.0, 82.6), (1.5, 55.0), (2.0, 41.3)])
def test_decay_length_matches_paper(hwx, d_nm):
    p = QPCParams(hbar_wx_meV=hwx)
    assert abs(p.units.au_to_nm(p.d) - d_nm) < 0.15      # paper rounds d


def test_V_qpc_limits():
    p = QPCParams()
    y = np.linspace(-5, 5, 11)
    assert np.allclose(V_qpc(1e3, y, p), 0.0, atol=1e-12)            # -> clean wire
    x = np.linspace(-20, 20, 9)
    assert np.allclose(V_qpc(x, 0.0, p), 0.5 * V_barrier(x, p))       # y = 0: V(x)/2
    assert np.isclose(V_barrier(0.0, p), p.V0)


def test_barrier_no_overflow():
    p = QPCParams()
    assert V_barrier(1e6, p) == 0.0


def test_fft_matches_exact_sech2():
    p = QPCParams(hbar_wx_meV=2.0)                     # sharpest barrier, d = 41.3 nm
    u = Units()
    L = u.nm_to_au(5000.0)
    N = 512
    dx = L / N
    x = min_image(np.arange(N) * dx, L)
    Vt = ft_numpy(V_barrier(x, p), dx)
    G = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    scale = 2 * p.V0 * p.d
    for m in range(1, 40):
        exact = float(ft_sech2_exact(G[m], p.V0, p.d))
        assert abs(Vt[m].real - exact) < 1e-12 * scale
        assert abs(Vt[m].imag) < 1e-12 * scale
    assert np.isclose(Vt[0].real, scale)


def test_mpmath_matches_exact():
    mp.mp.dps = 40
    d, V0 = mp.mpf("4.05"), mp.mpf("0.27")
    f = lambda x: V0 / mp.cosh(x / d) ** 2
    for k in [0, 0.1, 0.5, 1.0, 2.0]:
        got = ft_mpmath(f, mp.mpf(k), h=mp.mpf("0.05"), x_max=mp.mpf(40) * d)
        assert abs(got - ft_sech2_exact(k, V0, d)) < mp.mpf(10) ** -25
