"""Fourier transforms of the barrier V(x). Atomic units.

Convention: Vt(k) = integral f(x) exp(-i k x) dx  (continuous transform).
"""
import numpy as np
import mpmath as mp


def min_image(x, L):
    """Fold coordinates into [-L/2, L/2) so a function centred at x = 0 is periodic."""
    return x - L * np.round(x / L)


def ft_numpy(f_samples, dx):
    """Trapezoid/FFT estimate Vt(G_m) = dx * sum_j f(x_j) exp(-i G_m x_j).

    Expects samples at x_j = j*dx (numpy FFT order, as in Grid.real_axes) of a
    function already folded with min_image so its centre sits at index 0.
    Returns the array matching G = 2*pi*np.fft.fftfreq(N, d=dx).
    """
    return dx * np.fft.fft(f_samples)


def ft_sech2_exact(k, V0, d):
    """Exact transform of V0 sech^2(x/d): V0 * pi * k * d^2 / sinh(pi k d / 2); 2 V0 d at k = 0."""
    k = mp.mpf(k)
    if k == 0:
        return 2 * mp.mpf(V0) * mp.mpf(d)
    return mp.mpf(V0) * mp.pi * k * mp.mpf(d) ** 2 / mp.sinh(mp.pi * k * mp.mpf(d) / 2)


def ft_mpmath(f, k, h, x_max, dps=40):
    """YOUR TASK: Vt(k) = integral f(x) exp(-i k x) dx in arbitrary precision,
    with your own quadrature (no mp.quad for the final version).

    f: callable taking an mpf, returning an mpf. h: step. x_max: half-width.
    Return an mpf (f is even, so the transform is real).
    """
    mp.mp.dps = dps
    n = int(x_max /h)
    h = mp.mpf(h)
    k = mp.mpf(k)
    x_max = mp.mpf(x_max)
    total = f(0)/2
    for j in range(1, n + 1):
        x = j * h
        total += f(x) * mp.cos(k * x)
    return 2*h*total
