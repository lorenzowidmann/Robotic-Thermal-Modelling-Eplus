"""Project LiDAR points into a camera's pixel space.

Generic: works for either FLIR or ZED, just pass the matching intrinsics
(K, dist) and extrinsic (T_lidar_to_cam) from rig_calibration.py.

Points from /cloud_registered are already in world/map frame (the SLAM node
applies the current LiDAR pose before publishing). To reach the camera's own
frame at that instant: world -> LiDAR-local (undo that same pose) ->
camera-local (T_lidar_to_cam, a fixed rig extrinsic) -> pixels (K + dist).

Also provides flir_fov_bbox_in_zed(): which part of a ZED frame the FLIR can
actually see, for cropping work down to the overlap.
"""

import cv2
import numpy as np


def quat_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """xyzw quaternion (ROS convention, as stored in sync_manifest.json's
    triplet[i].lidar.orientation) -> 3x3 rotation matrix."""
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z
    return np.array([
        [1.0 - (yY + zZ), xY - wZ, xZ + wY],
        [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
        [xZ - wY, yZ + wX, 1.0 - (xX + yY)],
    ])


def world_to_lidar_local(points_world: np.ndarray, position: np.ndarray, orientation_xyzw: np.ndarray) -> np.ndarray:
    """Undo the SLAM pose (lidar-local -> world) to bring world points back
    into the LiDAR's own frame at that instant."""
    R = quat_to_rotation_matrix(*orientation_xyzw)
    return (points_world - position) @ R  # R.T @ (p - t), vectorized as (p-t) @ R


def project_lidar_to_camera(
    points_world: np.ndarray,
    lidar_position: np.ndarray,
    lidar_orientation_xyzw: np.ndarray,
    T_lidar_to_cam: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world-frame LiDAR points into one camera's pixel space.

    Returns:
        uv:    (N, 2) float pixel coordinates (undefined/garbage where invalid)
        depth: (N,) float, camera-frame forward distance (z)
        valid: (N,) bool, True where depth > 0 and the pixel falls inside
               [0, width) x [0, height)
    """
    points_lidar = world_to_lidar_local(points_world, lidar_position, lidar_orientation_xyzw)

    R_lc = T_lidar_to_cam[:3, :3]
    t_lc = T_lidar_to_cam[:3, 3]
    points_cam = points_lidar @ R_lc.T + t_lc

    depth = points_cam[:, 2]
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    uv, _ = cv2.projectPoints(points_cam.astype(np.float64), rvec, tvec, K, dist)
    uv = uv.reshape(-1, 2)

    valid = (
        (depth > 0)
        & (uv[:, 0] >= 0) & (uv[:, 0] < width)
        & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )
    return uv, depth, valid


def flir_fov_bbox_in_zed(
    cal,
    zed_width: int,
    zed_height: int,
    depths: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 50.0),
    margin_px: int = 0,
) -> tuple[int, int, int, int]:
    """Bounding box, in ZED pixels, of everything the FLIR can see.

    There is no direct FLIR<->ZED calibration, but both extrinsics are
    expressed in the same LiDAR frame, so composing them gives one:

        T_zed_to_flir = T_lidar_to_flir @ inv(T_lidar_to_zed)

    The FLIR image border is back-projected to rays, placed at each depth,
    moved into the ZED frame and projected with the ZED intrinsics; the union
    over depths is returned. Depth only matters through the FLIR<->ZED
    baseline (~13 cm on this rig), so the box is nearly depth-independent and
    a single static crop covers the whole session.

    `margin_px` pads the result before clipping. Worth using: this composes
    two extrinsics (~5.8 cm and ~6.8 cm RMSE) and the FLIR cx/cy are flagged
    as weakly constrained in rig_calibration.yaml, which is roughly +-20 ZED
    pixels of slop at 5 m.

    Returns (x0, y0, x1, y1), clipped to the frame, x1/y1 exclusive.
    """
    T_flir_to_zed = np.linalg.inv(cal.T_lidar_to_flir @ np.linalg.inv(cal.T_lidar_to_zed))
    R_fz = T_flir_to_zed[:3, :3]
    t_fz = T_flir_to_zed[:3, 3]

    # Dense border rather than the 4 corners: lens distortion bows the edges
    # outwards, so the corners alone would under-cover the real footprint.
    n = 200
    s = np.linspace(0, 1, n)
    w, h = cal.flir.width, cal.flir.height
    border = np.concatenate([
        np.stack([s * (w - 1), np.zeros(n)], 1),
        np.stack([np.full(n, w - 1.0), s * (h - 1)], 1),
        np.stack([s * (w - 1), np.full(n, h - 1.0)], 1),
        np.stack([np.zeros(n), s * (h - 1)], 1),
    ]).astype(np.float64)

    rays = cv2.undistortPoints(border.reshape(-1, 1, 2), cal.flir.K, cal.flir.dist).reshape(-1, 2)
    rays = np.concatenate([rays, np.ones((len(rays), 1))], 1)   # z = 1 normalized

    zed_K = cal.zed_K_for(zed_width, zed_height)
    lo = np.array([np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf])
    for d in depths:
        pts_zed = (R_fz @ (rays * d).T).T + t_fz
        uv, _ = cv2.projectPoints(pts_zed, np.zeros(3), np.zeros(3), zed_K, cal.zed_calib.dist)
        uv = uv.reshape(-1, 2)
        lo = np.minimum(lo, uv.min(axis=0))
        hi = np.maximum(hi, uv.max(axis=0))

    x0 = int(max(0, np.floor(lo[0]) - margin_px))
    y0 = int(max(0, np.floor(lo[1]) - margin_px))
    x1 = int(min(zed_width, np.ceil(hi[0]) + margin_px))
    y1 = int(min(zed_height, np.ceil(hi[1]) + margin_px))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("FLIR FOV does not intersect the ZED frame -- check the calibration")
    return x0, y0, x1, y1
