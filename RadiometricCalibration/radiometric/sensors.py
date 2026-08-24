"""Hardware input stubs for the field setup (not yet available on this PC).

Mirrors EmissivityCalculation/emissivity/sources.py: each class is a
placeholder that raises with integration guidance until the real drivers are
installed. During development the CLI takes the same quantities from files
and arguments instead.
"""


class ThermalCameraSource:
    """Apparent-temperature map from the thermal camera (model t.b.d.)."""

    def __init__(self):
        raise NotImplementedError(
            "Thermal camera SDK not integrated yet. Once the camera model is "
            "chosen, implement grab() here to return the apparent-temperature "
            "map (deg C, 2-D float array). Until then pass --thermal <file>."
        )


class LidarSource:
    """Per-pixel distance map from the LiDAR, projected onto the thermal image."""

    def __init__(self):
        raise NotImplementedError(
            "LiDAR driver not integrated yet. Implement grab() to return the "
            "distance map (metres, 2-D float array) co-registered with the "
            "thermal image. Until then pass --distance-map <file> or "
            "--distance <metres>."
        )


class HygrometerSource:
    """Relative humidity and air temperature from the weather sensor."""

    def __init__(self):
        raise NotImplementedError(
            "Humidity/temperature sensor not integrated yet. Implement read() "
            "to return (relative_humidity_percent, air_temp_c). Until then "
            "pass --humidity and --air-temp."
        )
