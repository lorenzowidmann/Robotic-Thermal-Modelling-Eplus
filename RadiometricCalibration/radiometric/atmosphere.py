"""Atmospheric transmission in the LWIR band.

Implements the standard single-band atmospheric transmission model used by
FLIR cameras (a fit to LOWTRAN simulations): the water vapour content of the
air is estimated from relative humidity and air temperature, and the
transmittance over a path follows a double-exponential decay in
sqrt(distance).

All functions are numpy-vectorized: `distance` may be a scalar or a 2-D map
(one LiDAR distance per thermal-image pixel), and tau is returned with the
same shape — this is what makes the correction per-pixel.
"""

import numpy as np

# Empirical constants of the FLIR/LOWTRAN atmospheric transmission model.
X = 1.9  # weighting between the two attenuation components
ALPHA1 = 0.006569  # attenuation for atmosphere without water vapour, term 1
ALPHA2 = 0.01262  # attenuation for atmosphere without water vapour, term 2
BETA1 = -0.002276  # attenuation for water vapour, term 1
BETA2 = -0.00667  # attenuation for water vapour, term 2

# Polynomial fit of the saturated water vapour content of air (T in deg C).
H1 = 1.5587
H2 = 6.939e-2
H3 = -2.7816e-4
H4 = 6.8455e-7


def water_vapour_content(relative_humidity: float, air_temp: float) -> float:
    """Water vapour content of the air in g/m^3.

    relative_humidity: relative humidity in percent (0-100)
    air_temp: air temperature in deg C
    """
    saturated = np.exp(H1 + H2 * air_temp + H3 * air_temp**2 + H4 * air_temp**3)
    return float(relative_humidity / 100.0 * saturated)


def transmittance(distance, relative_humidity: float, air_temp: float):
    """Atmospheric transmittance tau over `distance` metres.

    distance: scalar or numpy array (e.g. the LiDAR distance map in metres).
        Non-positive or NaN distances (LiDAR holes) yield NaN.
    Returns tau in [0, 1] with the same shape as `distance`.
    """
    d = np.asarray(distance, dtype=float)
    d = np.where(d > 0, d, np.nan)

    omega = water_vapour_content(relative_humidity, air_temp)
    sqrt_d = np.sqrt(d)
    sqrt_omega = np.sqrt(omega)

    tau = X * np.exp(-sqrt_d * (ALPHA1 + BETA1 * sqrt_omega)) + (1.0 - X) * np.exp(
        -sqrt_d * (ALPHA2 + BETA2 * sqrt_omega)
    )
    tau = np.clip(tau, 0.0, 1.0)
    return float(tau) if np.isscalar(distance) else tau
