from typing import List, Optional
import numpy as np
import gtsam
from functools import partial


def GtsamMatrix(row, col):
    return np.zeros((row, col), order='F')


def relativeBearing(d: gtsam.Point2, H: Optional[np.ndarray]) -> gtsam.Rot2:
    x = d[0]
    y = d[1]
    d2 = x * x + y * y
    n = np.sqrt(d2)
    if np.abs(n) > 1e-5:
        if H is not None:
            H[0, 0] = -y / d2
            H[0, 1] = x / d2
        return gtsam.Rot2.fromCosSin(x / n, y / n)
    else:
        if H is not None:
            H[0, 0] = 0.0
            H[0, 1] = 0.0
        return gtsam.Rot2()


def point2line_error(line_start: gtsam.Point2, line_end: gtsam.Point2,
                     this: gtsam.CustomFactor, v: gtsam.Values, H: List[np.ndarray]):
    key_point = this.keys()[0]
    point = v.atPoint2(key_point)

    # calculate the error
    bearing_line = relativeBearing(line_end-line_start, None)

    J_bearing_wrt_point = GtsamMatrix(1, 2)
    bearing_point = relativeBearing(point-line_start, J_bearing_wrt_point)

    error = bearing_line.localCoordinates(bearing_point)

    if H is not None:
        H[0] = J_bearing_wrt_point
    return error


def Point2ToLine2Factor(line_start: gtsam.Point2, line_end: gtsam.Point2, point_key: int, noise_model):
    keys = [point_key]
    return gtsam.CustomFactor(keys=keys, noiseModel=noise_model, errorFunction=partial(point2line_error, line_start, line_end))


def point3_project_line2_error(line_start: gtsam.Point2, line_end: gtsam.Point2,
                               k: gtsam.Cal3DS2, cam_pose: gtsam.Pose3,
                               this: gtsam.CustomFactor, v: gtsam.Values, H: List[np.ndarray]):
    camera = gtsam.PinholePoseCal3DS2(pose=cam_pose, K=k)
    key_point = this.keys()[0]
    world_point = v.atPoint3(key_point)

    J_point2_wrt_camera_pose = GtsamMatrix(2, 6)
    J_point2_wrt_point3 = GtsamMatrix(2, 3)
    J_point2_wrt_cam_cal = GtsamMatrix(2, 9)
    try:
        point2 = camera.project(
            point=world_point,
            Dpose=J_point2_wrt_camera_pose,
            Dpoint=J_point2_wrt_point3,
            Dcal=J_point2_wrt_cam_cal)

        # calculate the error
        bearing_line = relativeBearing(line_end-line_start, None)

        J_bearing_wrt_point = GtsamMatrix(1, 2)
        bearing_point = relativeBearing(point2-line_start, J_bearing_wrt_point)

        error = bearing_line.localCoordinates(bearing_point)

        if H is not None:
            H[0] = J_bearing_wrt_point @ J_point2_wrt_point3
    except:
        error = np.array([1e8])

    return error


def Point3ProjectLine2Factor(
        line_start: gtsam.Point2, line_end: gtsam.Point2, k: gtsam.Cal3DS2, camera_pose: gtsam.Pose3,
        point_key: int,  noise_model):
    keys = [point_key]
    return gtsam.CustomFactor(keys=keys,
                              noiseModel=noise_model,
                              errorFunction=partial(point3_project_line2_error, line_start, line_end, k, camera_pose))
