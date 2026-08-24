"""Radiance <-> temperature conversion.

The draft uses the Stefan-Boltzmann T^4 form of the measurement formula
(radiation proportional to the 4th power of the absolute temperature), the
standard approximation when the camera's Planck calibration constants
(R1, R2, B, F, O) are not known. The class boundary is designed so that a
Planck-based model for the actual thermal camera can be swapped in later
without touching the correction code.
"""

import numpy as np

ZERO_C = 273.15  # 0 deg C in Kelvin


class RadianceModel:
    """T^4 radiance model. Temperatures are in deg C at the interface."""

    def radiance(self, temp_c):
        """Relative radiance W(T) for temperature(s) in deg C."""
        t_k = np.asarray(temp_c, dtype=float) + ZERO_C
        return t_k**4

    def temperature(self, w):
        """Inverse W^-1: temperature(s) in deg C from relative radiance."""
        w = np.asarray(w, dtype=float)
        w = np.where(w > 0, w, np.nan)
        return w**0.25 - ZERO_C
