import apriltag

import cv2
import numpy as np
import argparse
import os
import yaml

from tqdm import tqdm

import gtsam

from apriltag_calibrate.utils.ImageLoader import ImageLoader
from apriltag_calibrate.configparase import TagBundle, Camera
from apriltag_calibrate.utils.TagPnp import TagPnP
from apriltag_calibrate.visualization import draw_axes, draw_camera
from apriltag_calibrate.utils.Geometery import Rtvec2HomogeousT

from inertia_calibrate.custom_factor.RotAxisFactor import RotAxisFactor


# get image path from command line
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path to the video file")
ap.add_argument("-o", "--output", required=True,
                help="path to the result path")
ap.add_argument("-c", "--camera", required=True,
                help="path to the camera calibration file")
ap.add_argument("-b", "--bundle", required=True,
                help="path to the bundle calibration file")


args = ap.parse_args()
video_path = args.video
output_path = args.output
camera_param_path = args.camera
bundle_param_path = args.bundle

# read the video
cap = cv2.VideoCapture(video_path)

# get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = cap.get(cv2.CAP_PROP_FPS)
total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"video res:{frame_width}x{frame_height}, frame rate:{frame_rate}")

# config parse
# get camera parameters
camera = Camera(camera_param_path)

# get bundle parameters
bundle = TagBundle()
bundle.load(bundle_param_path)

# create detector
detector_options = apriltag.DetectorOptions(families=bundle.tag_family,
                                            border=1,
                                            nthreads=4,
                                            quad_decimate=4,
                                            quad_blur=0.0,
                                            refine_edges=True,
                                            refine_decode=False,
                                            refine_pose=False,
                                            debug=False,
                                            quad_contours=True)
detector = apriltag.Detector(detector_options)

# setup gtsam graph
graph = gtsam.NonlinearFactorGraph()
initial_estimate = gtsam.Values()
key_axis_pose = gtsam.symbol('r', 0)

pose_noise = gtsam.noiseModel.Robust.Create(
    robust=gtsam.noiseModel.mEstimator.Huber.Create(1.345),
    noise=gtsam.noiseModel.Isotropic.Sigma(6, 1))


# output data
output_data = {}
output_data["header"] = {
    "camera_param": camera_param_path,
    "bundle_param": bundle_param_path,
    "video_path": video_path,
    "video_param": {
        "res": [frame_width, frame_height],
        "frame_rate": frame_rate,
        "total_frame": total_frame
    }
}
output_data["data"] = {
    "pose": {},
    "angle": {},

}


# walk through the video
world_T_first_bundle_pose = np.eye(4)
for frame_id in tqdm(range(total_frame)):
    ret, img = cap.read()
    if not ret:
        print("error reading video")
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)

    pnp = TagPnP()
    pnp.add_tag(detections, bundle)
    ret, rvecs, tvecs = pnp.solve(camera)

    camera_T_bundle_pose = Rtvec2HomogeousT(rvecs, tvecs)
    bundle_T_camera_pose = np.linalg.inv(camera_T_bundle_pose)

    if frame_id == 0:
        # first tag
        world_T_first_bundle_pose = camera_T_bundle_pose
        output_data["data"]["camera_pose"] = bundle_T_camera_pose.tolist()

    first_bundle_T_bundle_i_pose = np.linalg.inv(
        world_T_first_bundle_pose)@camera_T_bundle_pose
    key_angle = gtsam.symbol('b', frame_id)
    graph.add(RotAxisFactor(pose_before=gtsam.Pose3(),
                            pose_after=gtsam.Pose3(
                                mat=first_bundle_T_bundle_i_pose),
                            key_rotate_angle=key_angle,
                            key_rotate_axis_pose=key_axis_pose,
                            noise_model=pose_noise))
    initial_estimate.insert(key_angle, 0)
    output_data["data"]["pose"][frame_id] = first_bundle_T_bundle_i_pose.tolist()
cap.release()

init_axes_pose = gtsam.Pose3(
    r=gtsam.Rot3(), t=gtsam.Point3(0.01,0.01,0.01))
initial_estimate.insert(key_axis_pose, init_axes_pose)

# solve
params = gtsam.LevenbergMarquardtParams()
params.setVerbosityLM("SUMMARY")

optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)

result = optimizer.optimize()
opt_axes_pose: gtsam.Pose3 = result.atPose3(key_axis_pose)

output_data["data"]["axis_pose"] = opt_axes_pose.matrix().tolist()
for frame_id in output_data["data"]["pose"].keys():
    key_angle = gtsam.symbol('b', frame_id)
    output_data["data"]["angle"][frame_id] = result.atDouble(key_angle)

# save data
with open(output_path, 'w') as f:
    yaml.dump(output_data, f)
    print(f"output data saved to {output_path}")
