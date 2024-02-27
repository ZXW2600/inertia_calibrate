from typing import List
import numpy as np
import gtsam
from functools import partial


def GtsamMatrix(row, col):
    return np.zeros((row, col), order='F')


def RotateAxisError(world_pose_before: gtsam.Pose3, world_pose_after: gtsam.Pose3, this: gtsam.CustomFactor, v: gtsam.Values, H: List[np.ndarray]):
    key_rotate_axis_pose, key_rotate_angle = this.keys()[0], this.keys()[1]
    rotate_angle = v.atDouble(key_rotate_angle)
    world_rotate_axis_pose = v.atPose3(key_rotate_axis_pose)

    # transform the pose to the rotate axis frame
    J_axis_T_pose_before_wrt_world_pose_before = GtsamMatrix(6, 6)
    J_axis_T_pose_before_wrt_world_rotate_axis_pose = GtsamMatrix(6, 6)
    axis_T_pose_before = world_rotate_axis_pose.transformPoseFrom(
        world_rotate_axis_pose,
        J_axis_T_pose_before_wrt_world_rotate_axis_pose,
        J_axis_T_pose_before_wrt_world_pose_before,
    )

    # transform the pose to the rotate axis frame
    J_axis_T_pose_after_wrt_world_pose_after = GtsamMatrix(6, 6)
    J_axis_T_pose_after_wrt_world_rotate_axis_pose = GtsamMatrix(6, 6)
    axis_T_pose_after = world_rotate_axis_pose.transformPoseFrom(
        world_pose_after,
        J_axis_T_pose_after_wrt_world_rotate_axis_pose,
        J_axis_T_pose_after_wrt_world_pose_after,
    )

    # calculate the rotation in the rotate axis frame
    J_R_wrt_x = GtsamMatrix(1, 3)
    J_R_wrt_y = GtsamMatrix(1, 3)
    J_R_wrt_z = GtsamMatrix(1, 3)

    rotate = gtsam.Rot3.RzRyRx(
        x=0, y=0, z=rotate_angle,
        Hx=J_R_wrt_x, Hy=J_R_wrt_y, Hz=J_R_wrt_z)

    axis_T_pose_after_predic=axis_T_pose_before
    
    pass


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
