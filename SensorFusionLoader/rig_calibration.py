"""Load the canonical rig calibration (rig_calibration.yaml): FLIR/ZED
intrinsics and LiDAR<->camera extrinsics.

Single source of truth for anything that projects LiDAR points into a
camera image or needs a camera's own K/distortion -- edit the YAML, not
code, when a calibration is redone.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

DEFAULT_PATH = Path(__file__).resolve().parent / "rig_calibration.yaml"


@dataclass
class CameraIntrinsics:
    K: np.ndarray      # 3x3
    dist: np.ndarray   # (5,) k1, k2, p1, p2, k3 -- cv2.projectPoints order
    width: int
    height: int


@dataclass
class RigCalibration:
    flir: CameraIntrinsics
    flir_rotated_180: bool  # FLIR frames must be rotated 180deg before use (see YAML)
    zed_calib: CameraIntrinsics  # ZED K at its own calibration resolution
    T_lidar_to_flir: np.ndarray  # 4x4, laser frame -> FLIR camera frame
    T_lidar_to_zed: np.ndarray   # 4x4, laser frame -> ZED camera frame

    def zed_K_for(self, width: int, height: int) -> np.ndarray:
        """ZED K rescaled from its calibration resolution to an actual
        capture size (e.g. 1920x1080 recording sessions vs the 1280x720
        calibration images). Distortion coefficients are resolution-
        independent (normalized-coordinate model) and are used as-is via
        self.zed_calib.dist."""
        sx = width / self.zed_calib.width
        sy = height / self.zed_calib.height
        K = self.zed_calib.K.copy()
        K[0, 0] *= sx  # fx
        K[0, 2] *= sx  # cx
        K[1, 1] *= sy  # fy
        K[1, 2] *= sy  # cy
        return K


def _camera_intrinsics(block: dict) -> CameraIntrinsics:
    K = np.array([
        [block["fx"], block.get("skew", 0.0), block["cx"]],
        [0.0, block["fy"], block["cy"]],
        [0.0, 0.0, 1.0],
    ])
    dist = np.array(block["dist_coeffs"], dtype=float)
    size = block.get("image_size") or block.get("calibration_image_size")
    return CameraIntrinsics(K=K, dist=dist, width=size["width"], height=size["height"])


def load_rig_calibration(path: str | Path = DEFAULT_PATH) -> RigCalibration:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data.get("schema") != "rig_calibration/v1":
        raise ValueError(f"Unexpected schema in {path}: {data.get('schema')!r}")

    return RigCalibration(
        flir=_camera_intrinsics(data["flir"]),
        flir_rotated_180=bool(data["flir"].get("rotated_180_before_calibration", False)),
        zed_calib=_camera_intrinsics(data["zed"]),
        T_lidar_to_flir=np.array(data["lidar_to_flir"]["T_lidar_to_cam"], dtype=float),
        T_lidar_to_zed=np.array(data["lidar_to_zed"]["T_lidar_to_cam"], dtype=float),
    )
