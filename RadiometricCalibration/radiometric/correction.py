"""Per-pixel radiometric correction of apparent temperatures.

The thermal camera reports an apparent-temperature map assuming a black body
(eps = 1) and a perfectly transparent atmosphere (tau = 1). The true object
temperature is recovered by inverting the radiation balance

    W(T_app) = eps*tau*W(T_obj) + (1-eps)*tau*W(T_refl) + (1-tau)*W(T_atm)

per pixel: `apparent_temp`, `emissivity` and `tau` may each be a scalar or a
2-D map of the same shape (numpy broadcasting applies), so nearby pixels
(high tau) and distant pixels (low tau) are corrected differently.
"""

import numpy as np

from .radiance import RadianceModel


def correct_temperature(
    apparent_temp,
    emissivity,
    tau,
    reflected_temp: float,
    air_temp: float,
    model: RadianceModel | None = None,
):
    """True object temperature(s) in deg C.

    apparent_temp: deg C, scalar or 2-D map (from the thermal camera)
    emissivity: 0-1, scalar or 2-D map (from the ZED material classification)
    tau: 0-1, scalar or 2-D map (from atmosphere.transmittance of the LiDAR
        distance map)
    reflected_temp: deg C, reflected apparent temperature (global scalar)
    air_temp: deg C, atmosphere temperature (global scalar)
    """
    model = model or RadianceModel()

    w_tot = model.radiance(apparent_temp)
    w_refl = model.radiance(reflected_temp)
    w_atm = model.radiance(air_temp)

    eps = np.asarray(emissivity, dtype=float)
    tau = np.asarray(tau, dtype=float)

    w_obj = (w_tot - (1.0 - eps) * tau * w_refl - (1.0 - tau) * w_atm) / (eps * tau)
    result = model.temperature(w_obj)
    return float(result) if result.ndim == 0 else result
