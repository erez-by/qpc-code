import numpy as np
from qpc.units import Units
from qpc.grid import Grid


def test_unit_values():
    u = Units()
    assert abs(u.energy_meV - 10.96) < 0.05
    assert abs(u.length_nm - 10.19) < 0.05


def test_unit_roundtrip():
    u = Units()
    assert np.isclose(u.au_to_nm(u.nm_to_au(123.4)), 123.4)
    assert np.isclose(u.au_to_meV(u.meV_to_au(1.5)), 1.5)


def test_fft_roundtrip():
    g = Grid(Lx=50.0, Ly=20.0, Nx=64, Ny=32)
    f = np.random.default_rng(0).normal(size=(g.Nx, g.Ny))
    assert np.allclose(np.fft.ifft2(np.fft.fft2(f)).real, f)


def test_laplacian_of_cosine():
    """-G^2 in Fourier space must reproduce the analytic second derivative."""
    g = Grid(Lx=50.0, Ly=20.0, Nx=64, Ny=32)
    x, y = g.real_axes()
    X, Y = np.meshgrid(x, y, indexing="ij")
    m = 3
    f = np.cos(2 * np.pi * m * X / g.Lx)
    lap = np.fft.ifft2(-g.G2() * np.fft.fft2(f)).real
    assert np.allclose(lap, -(2 * np.pi * m / g.Lx) ** 2 * f)
