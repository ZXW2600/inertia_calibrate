from typing import List, Optional
import numpy as np
import gtsam
from functools import partial


def GtsamMatrix(row, col):
    return np.zeros((row, col), order='F')

# Pose3 Pose3::Create(const Rot3& R, const Point3& t, OptionalJacobian<6, 3> HR, OptionalJacobian<6, 3> Ht)


def CreatePose3(R: gtsam.Rot3, t: gtsam.Point3, HR: Optional[np.ndarray] = None, Ht: Optional[np.ndarray] = None) -> gtsam.Pose3:
    if HR is not None:
        HR[:3, :3] = np.eye(3)
        HR[:3, 3:] = np.zeros((3,0))
    # if Ht is not None:
    #     Ht[:3, :3] = np.zeros((3, 3))
    #     Ht[:3, 3:] = R.transpose().reshape((3,0))
    return gtsam.Pose3(R, t)


def RotateAxisError(world_pose_before: gtsam.Pose3, world_pose_after: gtsam.Pose3,
                    this: gtsam.CustomFactor, v: gtsam.Values, H: List[np.ndarray]):
    key_rotate_axis_pose, key_rotate_angle = this.keys()[0], this.keys()[1]
    rotate_angle = v.atDouble(key_rotate_angle)
    world_T_rotate_axis_pose = v.atPose3(key_rotate_axis_pose)

    J_rotate_ref_T_pose_before_wrt_world_T_rotate_axis_pose = GtsamMatrix(6, 6)
    J_rotate_ref_T_pose_before_wrt_world_pose_before = GtsamMatrix(6, 6)
    rotate_ref_T_pose_before = world_T_rotate_axis_pose.transformPoseTo(
        world_pose_before,
        J_rotate_ref_T_pose_before_wrt_world_T_rotate_axis_pose,
        J_rotate_ref_T_pose_before_wrt_world_pose_before)

    J_rotate_ref_T_pose_after_wrt_world_T_rotate_axis_pose = GtsamMatrix(6, 6)
    J_rotate_ref_T_pose_after_wrt_world_pose_after = GtsamMatrix(6, 6)
    rotate_ref_T_pose_after = world_T_rotate_axis_pose.transformPoseTo(
        world_pose_after)

    J_rotate_ref_T_rotated_ref_Rot_wrt_angle = GtsamMatrix(3, 1)
    _J0 = GtsamMatrix(3,1)
    _J1 = GtsamMatrix(3,1)
    J_rotate_ref_T_rotated_ref_wrt_rotate_ref_T_rotated_ref_Rot = GtsamMatrix(
        6, 3)
    J_rotate_ref_T_rotated_ref_wrt_rotate_ref_T_rotated_ref_t = GtsamMatrix(
        6, 3)
    rotate_ref_T_rotated_ref = CreatePose3(
        R=gtsam.Rot3.RzRyRx(x=0, y=0, z=rotate_angle, Hx=_J0,
                            Hy=_J1, Hz=J_rotate_ref_T_rotated_ref_Rot_wrt_angle),
        t=gtsam.Point3(0, 0, 0),
        HR=J_rotate_ref_T_rotated_ref_wrt_rotate_ref_T_rotated_ref_Rot,
        Ht=J_rotate_ref_T_rotated_ref_wrt_rotate_ref_T_rotated_ref_t
    )

    J_rotated_ref_T_pose_before_wrt_rotate_ref_T_rotated_ref = GtsamMatrix(
        6, 6)
    J_rotated_ref_T_pose_before_wrt_rotate_ref_T_pose_before = GtsamMatrix(
        6, 6)
    rotated_ref_T_pose_before = rotate_ref_T_rotated_ref.transformPoseTo(
        rotate_ref_T_pose_before,
        J_rotated_ref_T_pose_before_wrt_rotate_ref_T_rotated_ref,
        J_rotated_ref_T_pose_before_wrt_rotate_ref_T_pose_before)

    J_pose_error_wrt_rotate_ref_T_pose_after = GtsamMatrix(6, 6)
    J_pose_error_wrt_rotated_ref_T_pose_before = GtsamMatrix(6, 6)
    pose_error = rotate_ref_T_pose_after.between(rotated_ref_T_pose_before,
                                                 J_pose_error_wrt_rotate_ref_T_pose_after,
                                                 J_pose_error_wrt_rotated_ref_T_pose_before)

    J_error_wrt_pose_error = GtsamMatrix(6, 6)
    error = pose_error.localCoordinates(gtsam.Pose3(), J_error_wrt_pose_error)

    if H is not None:
        H[0] = J_error_wrt_pose_error@J_pose_error_wrt_rotated_ref_T_pose_before@J_rotated_ref_T_pose_before_wrt_rotate_ref_T_pose_before@J_rotate_ref_T_pose_before_wrt_world_T_rotate_axis_pose\
            + J_error_wrt_pose_error@J_pose_error_wrt_rotate_ref_T_pose_after@J_rotate_ref_T_pose_after_wrt_world_T_rotate_axis_pose
        H[1] = J_error_wrt_pose_error@J_pose_error_wrt_rotated_ref_T_pose_before\
            @ J_rotated_ref_T_pose_before_wrt_rotate_ref_T_rotated_ref \
            @ J_rotate_ref_T_rotated_ref_wrt_rotate_ref_T_rotated_ref_Rot @ J_rotate_ref_T_rotated_ref_Rot_wrt_angle

    return error


def RotAxisFactor(pose_before: gtsam.Pose3, pose_after: gtsam.Pose3,
                  key_rotate_axis_pose: int, key_rotate_angle: int,
                  noise_model: gtsam.noiseModel.Gaussian):
    """
    Create a custom factor to represent the rotation around an axis,

    :param pose_before: the pose before the rotation
    :param pose_after: the pose after the rotation
    :param key_rotate_axis_pose: the key of the pose that the rotation axis is in, the rotation axis is the z axis of the pose
    :param key_rotate_angle: the key of the rotation angle
    :param noise_model: the noise model of the factor
    """
    keys = [key_rotate_axis_pose, key_rotate_angle]
    return gtsam.CustomFactor(errorFunction=partial(RotateAxisError, pose_before, pose_after),
                              keys=keys,
                              noiseModel=noise_model)
