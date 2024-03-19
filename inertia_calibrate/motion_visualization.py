import argparse
from functools import partial
from apriltag_calibrate.visualization import draw_camera, draw_axes, draw_tag, draw_axis
from apriltag_calibrate.visualization.utils import draw_line
from matplotlib import pyplot as plt
import numpy as np
import yaml
from apriltag_calibrate.configparase import TagBundle
import gtsam

# get image path from command line
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--motion", required=True,
                help="path to the motion yaml path")

ap.add_argument("-b", "--bundle", required=False,
                help="path to the bundle calibration file")

parms = ap.parse_args()
motion_path = parms.motion

flag_draw_tag = False
bundle_path = parms.bundle
if bundle_path is not None:
    flag_draw_tag = True
    bundle = TagBundle()
    bundle.load(bundle_path)
    print(f"bundle loaded: {bundle_path}")

# read the motion
with open(motion_path, 'r') as f:
    motion = yaml.safe_load(f)
    # output_data = {}
    # output_data["header"] = {
    #     "camera_param": camera_param_path,
    #     "bundle_param": bundle_param_path,
    #     "video_path": video_path,
    #     "video_param": {
    #         "res": [frame_width, frame_height],
    #         "frame_rate": frame_rate,
    #         "total_frame": total_frame
    #     }
    # }
    # output_data["data"] = {
    #     "pose": {},
    # }
    pose_dict = motion["data"]["pose"]
    angle_dict = motion["data"]["angle"]
    angle_np = np.array([angle_dict[i] for i in sorted(angle_dict.keys())])
    pose_np = np.array([pose_dict[i]
                       for i in sorted(pose_dict.keys())]).reshape(-1, 4, 4)
    camera_pose = np.array(motion["data"]["camera_pose"])
    axis_pose = np.array(motion["data"]["axis_pose"])
    print(f"motion header: video  {motion['header']['video_path']}")
    print(f"motion header: camera {motion['header']['camera_param']}")
    print(f"motion header: bundle {motion['header']['bundle_param']}")
    print(f"motion header: video param {motion['header']['video_param']}")

    figure = plt.figure()
    axes = figure.add_subplot(111, projection='3d')
    figure_all = plt.figure()
    axes_all = figure_all.add_subplot(111, projection='3d')

    draw_axes(axes_all, axis_pose, size=0.02)

    for frame_id, (frame, world_T_bundle_pose) in enumerate(pose_dict.items()):
        axes._children = []
        angle_i = angle_dict[frame_id]
        R = np.eye(4)
        R[:3, :3] = np.array(world_T_bundle_pose)[:3, :3]
        world_T_bundle_pose = np.array(world_T_bundle_pose)
        pose_i_np = pose_np[frame_id]
        pose_np_ref:np.ndarray = np.linalg.inv(world_T_bundle_pose)@pose_np
        print(pose_np_ref.shape)
        def angle_axis(T):
            axis,angle=gtsam.Rot3(R=T[:3, :3]).axisAngle()
            pt=np.array(axis.point3()).tolist()
            return np.array([*pt,angle])
        
        axis_angle_vec = np.vectorize(angle_axis,signature='(n,m)->(n)',otypes='O')
        axis_angle=axis_angle_vec(pose_np_ref)
        angle_np=axis_angle[:,3]
        max_id=np.argmax(angle_np)
        print(angle_np.shape)
        draw_axes(axes, axis_pose)
        draw_axes(axes, np.array(world_T_bundle_pose))

        # find max angle distance frame
        # angle_distance = np.abs(angle_np-angle_i)
        # max_id = np.argmax(angle_distance)
        # max_angle = angle_np[max_id]
        max_world_T_bundle_pose = pose_dict[max_id]

        between_pose = np.array(world_T_bundle_pose).dot(
            np.linalg.inv(max_world_T_bundle_pose))
        R_max = np.eye(4)
        R_max[:3, :3] = between_pose[:3, :3]
        axis_max, angle_max = gtsam.Rot3(R=R_max[:3, :3]).axisAngle()

        if angle_max < angle_i:
            print("error max")

        if frame_id == 0:
            init_axis = gtsam.Unit3(-axis_max.point3())
        cos = init_axis.dot(axis_max)
        if cos < 0:
            axis_max = gtsam.Unit3(-axis_max.point3())

        line_start = np.zeros(3)
        # line_start=np.array(world_T_bundle_pose)[:3,3]
        line_end = line_start+axis_max.point3()
        draw_line(axes_all, line_start, line_end, color="r", alpha=0.2)

        # draw_axes(axes,R)
        axis, angle = gtsam.Rot3(R=R[:3, :3]).axisAngle()
        line_start = np.zeros(3)
        # line_start=np.array(world_T_bundle_pose)[:3,3]
        line_end = line_start+axis.point3()
        if angle > 0.2:
            draw_line(axes, line_start, line_end)

        draw_line(axes_all, line_start, line_end, alpha=0.2)

        if flag_draw_tag:
            for tag_id, bundle_T_tag_pose in bundle.tag_pose.items():
                world_T_tag_pose = np.array(
                    world_T_bundle_pose)@np.array(bundle_T_tag_pose)
                draw_tag(axes, np.array(world_T_tag_pose),
                         bundle.tag_size, tag_id)
        plt.show(block=False)
        plt.pause(0.1)
