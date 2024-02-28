import argparse
from apriltag_calibrate.visualization import draw_camera, draw_axes, draw_tag,draw_axis
from matplotlib import pyplot as plt
import numpy as np
import yaml
from apriltag_calibrate.configparase import TagBundle

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
    camera_pose=np.array(motion["data"]["camera_pose"])
    axis_pose = np.array(motion["data"]["axis_pose"])
    print(f"motion header: video  {motion['header']['video_path']}")
    print(f"motion header: camera {motion['header']['camera_param']}")
    print(f"motion header: bundle {motion['header']['bundle_param']}")
    print(f"motion header: video param {motion['header']['video_param']}")

    figure = plt.figure()
    axes = figure.add_subplot(111, projection='3d')
    for frame, world_T_bundle_pose in pose_dict.items():
        axes._children=[]
        
        draw_camera(axes,camera_pose)
        draw_axes(axes, axis_pose)
        draw_axes(axes, np.array(world_T_bundle_pose))
        if flag_draw_tag:
            for tag_id, bundle_T_tag_pose in bundle.tag_pose.items():
                world_T_tag_pose = np.array(
                    world_T_bundle_pose)@np.array(bundle_T_tag_pose)
                draw_tag(axes, np.array(world_T_tag_pose),
                         bundle.tag_size, tag_id)
        plt.show(block=False)
        plt.pause(0.1)
