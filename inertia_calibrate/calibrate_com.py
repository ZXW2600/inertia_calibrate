import apriltag
import cv2
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

import numpy as np
import argparse

from tqdm import tqdm

from apriltag_calibrate.utils.ImageLoader import ImageLoader
from apriltag_calibrate.configparase import TagBundle, Camera
from apriltag_calibrate.utils.TagPnp import TagPnP
from apriltag_calibrate.visualization import draw_axes, draw_camera, draw_3dpoints, draw_tag

from apriltag_calibrate.utils.Geometery import Rtvec2HomogeousT

import gtsam

from inertia_calibrate.custom_factor import Point3ProjectLine2Factor

# Finds the intersection of two lines, or returns False.
# The lines are defined by (o1, p1) and (o2, p2).


def line_intersection(line1, line2,
                      on_line1_segment=True, on_line2_segment=True):
    # Convert lines to a vector form
    x_diff = np.array([line1[0][0] - line1[1][0], line2[0][0] - line2[1][0]])
    y_diff = np.array([line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(x_diff, y_diff)
    if div == 0:
        return False  # Lines don't intersect

    d = (det(*line1), det(*line2))
    x = det(d, x_diff) / div
    y = det(d, y_diff) / div

    if on_line1_segment:
        if not (min(line1[0][0], line1[1][0]) <= x <= max(line1[0][0], line1[1][0]) and
                min(line1[0][1], line1[1][1]) <= y <= max(line1[0][1], line1[1][1])):
            return False, (0, 0)
    if on_line2_segment:
        if not (min(line2[0][0], line2[1][0]) <= x <= max(line2[0][0], line2[1][0]) and
                min(line2[0][1], line2[1][1]) <= y <= max(line2[0][1], line2[1][1])):
            return False, (0, 0)

    return True, (x, y)  # Lines don't intersect

def draw_convexhull(axes:Axes, convexhull,color='r',line_width=1):
       # Loop over pairs of points in the convex hull
    for i in range(len(convexhull)):
        # Get the current point and the next point (wrapping around at the end)
        pt1 = tuple(convexhull[i][0])
        pt2 = tuple(convexhull[(i+1) % len(convexhull)][0])
        axes.plot([pt1[0],pt2[0]],[pt1[1],pt2[1]],color=color)

def line_intersects_convexhull(line, convexhull):
    line_start = np.array(line[0])
    line_end = np.array(line[1])

    convexhull_center = np.mean(convexhull, axis=0)

    # Loop over pairs of points in the convex hull
    for i in range(len(convexhull)):
        # Get the current point and the next point (wrapping around at the end)
        pt1 = tuple(convexhull[i][0])
        pt2 = tuple(convexhull[(i+1) % len(convexhull)][0])

        # Check if the line intersects the current line segment
        inter_flag, inter_point = line_intersection(
            (line_start, line_end), (pt1, pt2), False, True)

        if inter_flag:
            outter_length = 0
            inner_length = 0
            distance_start = np.linalg.norm(line_start-convexhull_center)
            distance_end = np.linalg.norm(line_end-convexhull_center)
            if distance_start > distance_end:
                outter_point = line_start
                inner_point = line_end
            else:
                outter_point = line_end
                inner_point = line_start
            outter_length = np.linalg.norm(outter_point-inter_point)
            inner_length = np.linalg.norm(inner_point-inter_point)

            return True, outter_length, inner_length

    # If we haven't returned yet, the line doesn't intersect the convex hull
    return False, 0, 0


# get image path from command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the image folder")
ap.add_argument("-c", "--camera", required=True,
                help="path to the camera calibration file")
ap.add_argument("-b", "--bundle", required=True,
                help="path to the bundle calibration file")

args = ap.parse_args()
image_path = args.image
camera_param_path = args.camera
bundle_param_path = args.bundle

# read the image
imageset = ImageLoader(image_path)
imageset.load()

# get camera parameters
camera = Camera(camera_param_path)

# get bundle parameters
bundle = TagBundle()
bundle.load(bundle_param_path)

# bundle_center = np.array([bundle.tag_points[i]
#                          for i in bundle.tag_keys]).reshape(-1, 3).mean(axis=0)

bundle_center=np.array([0,0,0])

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

lsd_detector = cv2.createLineSegmentDetector(refine=1)


print("processing images")
image_draw = []


graph = gtsam.NonlinearFactorGraph()
initial_estimate = gtsam.Values()
point_key = gtsam.symbol('p', 0)
initial_estimate.insert(point_key, gtsam.Point3(bundle_center))
K = gtsam.Cal3DS2(
    fx=camera.fx,
    fy=camera.fy,
    s=0,
    u0=camera.cx,
    v0=camera.cy,
    k1=camera.k1,
    k2=camera.k2,
    p1=camera.p1,
    p2=camera.p2,

)

figure_3d = plt.figure()
ax_3d = figure_3d.add_subplot(111, projection='3d')

figure_2d = plt.figure()
ax_2d_dict = {}

for id in bundle.tag_pose.keys():
    draw_tag(ax_3d, bundle.tag_pose[id], bundle.tag_size, tag_id=id)

camera_pose_list = []
tag_points = np.array([pts for pts in bundle.tag_points.values()],dtype=np.float32).reshape(-1,3)


cv2.namedWindow("debug",cv2.WINDOW_GUI_NORMAL)
for img_id, img in enumerate(tqdm(imageset.images)):

    axes_2d = figure_2d.add_subplot(5,5, img_id+1)
    axes_2d.set_xlim(0, img.shape[1])
    axes_2d.set_ylim(0, img.shape[0])

    ax_2d_dict[img_id] = axes_2d

    w, h = img.shape[1], img.shape[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)

    pnp = TagPnP()
    pnp.add_tag(detections, bundle)
    ret, rvecs, tvecs = pnp.solve(camera)

    pose = Rtvec2HomogeousT(rvecs, tvecs)
    camera_pose = np.linalg.inv(pose)

    p, J = cv2.projectPoints(tag_points, rvecs,
                             tvecs, camera.cameraMatrix, camera.distCoeffs)
    camera_pose_list.append((camera_pose, rvecs, tvecs))
    p = p.reshape(-1, 2).astype(np.float32)

    convexHull = cv2.convexHull(p)

    cv2.drawContours(img, [convexHull.astype(np.int32)], -1, (0, 0, 255), 30)

    gray_downsampled = cv2.pyrDown(gray)
    gray_downsampled = cv2.pyrDown(gray_downsampled)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(
        gray, (kernel_size, kernel_size), 0)
    lines = lsd_detector.detect(gray_downsampled)[0]

    accept_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        x1, y1, x2, y2 = x1*4, y1*4, x2*4, y2*4
        line = [np.array((x1, y1)), np.array((x2, y2))]
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        cv2.line(img,line[0].astype(np.int32),line[1].astype(np.int32),(255,0,0),1)
        if length > h*0.2:
            cross, inner_length, outter_lengt = line_intersects_convexhull(
                line, convexHull)
            cv2.line(img,line[0].astype(np.int32),line[1].astype(np.int32),(255,0,0),5)

            if cross:
                dx = np.abs(x1-x2)/length
                accept_lines.append((line, dx))
                # print(f"{inner_length=}, {outter_lengt=}")
                # if outter_lengt > inner_length:
                cv2.line(img,line[0].astype(np.int32),line[1].astype(np.int32),(0,0,255),5)


    sorted_lines = sorted(accept_lines, key=lambda x: x[1])
    best_line = sorted_lines[0][0]
    d=best_line[0]-best_line[1]
    best_line[0]+=d*10
    best_line[1]-=d*10


    noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(
            1.345), gtsam.noiseModel.Isotropic.Sigma(1, 1.0)
    )
    graph.push_back(
        Point3ProjectLine2Factor(best_line[0], best_line[1],
                                 K, gtsam.Pose3(mat=camera_pose), point_key, noise)

    )

    # draw 3d
    draw_camera(ax_3d, camera_pose, 0.05, 0.15)

    axes_2d.plot([best_line[0][0],best_line[1][0]],[best_line[0][1],best_line[1][1]])
    draw_convexhull(axes_2d,convexHull)
    # print(sorted_lines[:,1])
    # cv2.line(img, (int(sorted_lines[0][0][0]), int(sorted_lines[0][0][1])), (int(sorted_lines[0][0][2]), int(sorted_lines[0][0][3])),
    #          (0, 255, 0), 5, cv2.LINE_AA)
    cv2.line(img,best_line[0].astype(np.int32),best_line[1].astype(np.int32),(0,255,255),10)

    cv2.imshow("debug", img)
    cv2.waitKey(1)

# optimize
params = gtsam.LevenbergMarquardtParams()
params.setVerbosityLM("SUMMARY")
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
for i, ax in ax_2d_dict.items():
        camera_pose, rvecs, tvecs = camera_pose_list[i]
        p, J = cv2.projectPoints(
            np.array([bundle_center]).astype(np.float32), rvecs, tvecs, camera.cameraMatrix, camera.distCoeffs)
        p = p.reshape(-1, 2).astype(np.float32)
        ax.scatter(p[0, 0], p[0, 1], c='b',marker='x')
        plt.show(block=False)

opt_cnt=0
while optimizer.lambda_()>1e-8 and   opt_cnt<20:
    opt_cnt+=1
    optimizer.iterate()
    result = optimizer.values()
    point_opt = result.atPoint3(point_key)
    for i, ax in ax_2d_dict.items():
        camera_pose, rvecs, tvecs = camera_pose_list[i]
        p, J = cv2.projectPoints(
            np.array([point_opt]), rvecs, tvecs, camera.cameraMatrix, camera.distCoeffs)
        p = p.reshape(-1, 2).astype(np.float32)
        ax.scatter(p[0, 0], p[0, 1], c='r',marker='x',alpha=opt_cnt/40.0+0.5)
        plt.show(block=False)
draw_3dpoints(ax_3d, bundle_center, size=0.1, color='r')
draw_3dpoints(ax_3d, point_opt, size=0.2,line_width=3)

# result: gtsam.Values = optimizer.optimize()
# point_opt = result.atPoint3(point_key)

plt.show()
