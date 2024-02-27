from inertia_calibrate.custom_factor.Point2LineFactor import Point2ToLine2Factor, Point3ProjectLine2Factor
from apriltag_calibrate.visualization import draw_3dpoints, draw_camera

import gtsam
import numpy as np
import matplotlib.pyplot as plt


def test_Point2ToLine2Factor():
    # prepare data
    line_start = gtsam.Point2(0, 0)

    line_end_list = []

    for i in range(10):
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()

        point_key = gtsam.symbol('p', i)
        point = np.random.rand(2)*10+np.array([-1, 1])
        initial_estimate.insert(point_key, point)

        end_point = np.random.rand(2)*5+np.array([1, 1])
        line_end_list.append(end_point)

        noise = gtsam.noiseModel.Isotropic.Sigma(1, 1)
        graph.add(Point2ToLine2Factor(line_start, end_point, point_key, noise))

        # optimize
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("SUMMARY")
        optimizer = gtsam.LevenbergMarquardtOptimizer(
            graph, initial_estimate, params)

        point_init = initial_estimate.atPoint2(point_key)
        # plot
        plt.figure()
        plt.plot([line_start[0], end_point[0]], [
                 line_start[1], end_point[1]], 'r')

        plt.scatter(x=point_init[0], y=point_init[1], c='r')

        opt_values = [point_init]
        opt_cnt=0
        while optimizer.lambda_() > 1e-6 and opt_cnt<20:
            opt_cnt+=1
            optimizer.iterate()
            result = optimizer.values()
            # result:gtsam.Values = optimizer.optimize()

            point_opt = result.atPoint2(point_key)
            opt_values.append(point_opt)
            plt.scatter(x=point_opt[0], y=point_opt[1], c='b', marker='x')

        for i in range(len(opt_values)-1):
            plt.plot([opt_values[i][0], opt_values[i+1][0]],
                     [opt_values[i][1], opt_values[i+1][1]], 'b')

        plt.show()


def test_Point3ProjectLine2Factor():
    K = gtsam.Cal3DS2(
        fx=320,
        fy=240,
        s=0,
        u0=320,
        v0=240,
        k1=0,
        k2=0,
        p1=0,
        p2=0
    )
    point3 = gtsam.Point3(0, 0, 0)
    point3_noise = gtsam.Point3(0.3,0.2,0.4)
    line_start = gtsam.Point3(0, 0, -1)
    line_end = gtsam.Point3(0, 0, 1)

    # visualize camera pose and fake image
    camera_pose_figure = plt.figure()
    camera_pose_axes = camera_pose_figure.add_subplot(111, projection='3d')
    camera_pose_axes.set_xlim(-1.2, 1.2)
    camera_pose_axes.set_ylim(-1.2, 1.2)
    camera_pose_axes.set_zlim(-1.2, 1.2)

    draw_3dpoints(camera_pose_axes, point3, size=0.5,line_width=1, color="r")
    draw_3dpoints(camera_pose_axes, point3_noise, size=0.1, color='b',)

    image_figure = plt.figure()

    # create graph
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    point_key = gtsam.symbol('p', 0)
    initial_estimate.insert(point_key, gtsam.Point3(0.1,0.2,0.1))

    image_num = 25

    image_axes_list = []
    camera_pose_list = []
    line_list = []

    for i in range(image_num):
        # add image axes
        image_axes = image_figure.add_subplot(5,5, i+1)
        image_axes.set_xlim(0, 640)
        image_axes.set_ylim(0, 480)
        image_axes_list.append(image_axes)
        image_axes.set_title(f"image {i}")
        # generate fake camera pose
        rand_rot = gtsam.Rot3.RzRyRx(np.random.rand(3) * np.pi * 2)
        rand_tf = gtsam.Pose3(rand_rot, gtsam.Point3(0, 0, 0))

        camera_point = gtsam.Point3(-1, 0, 0)
        camera_pose = gtsam.Pose3(gtsam.Rot3.Ry(
            np.pi/2).compose(gtsam.Rot3.AxisAngle(np.random.randn(3), np.random.rand()*0.1)), camera_point)

        # camera_pose=rand_tf.transformFrom(camera_pose)

        camera_pose_rand = rand_tf.transformPoseTo(camera_pose)

        # generate camera and project point
        camera = gtsam.PinholePoseCal3DS2(
            pose=camera_pose_rand, K=K)
        point2 = camera.project(point3)
        try:
            point2_noise = camera.project(point3_noise)
        except:
            point2_noise = point2
            print("point2_noise failed")

        image_line_start = camera.project(line_start)
        image_line_end = camera.project(line_end)
        line_list.append([image_line_start, image_line_end])
        camera_pose_list.append(camera_pose_rand.compose(
            gtsam.Pose3(r=gtsam.Rot3.AxisAngle(np.random.randn(3), np.random.rand()*0.1),t=gtsam.Point3(0,0,0))
        ))

        # add factor

        # visualize image
        image_axes.plot([image_line_start[0], image_line_end[0]], [
                        image_line_start[1], image_line_end[1]], 'r')
        image_axes.scatter(point2[0], point2[1], c='r', marker='x')
        image_axes.scatter(point2_noise[0], point2_noise[1], c='b')
        draw_camera(camera_pose_axes, camera_pose_rand.matrix(),
                    focal_len_scaled=0.10, aspect_ratio=0.3, color='r')

    # build graph
    for line, camera_pose in zip(line_list, camera_pose_list):
        img_line_start, img_line_end = line

        graph.push_back(
            Point3ProjectLine2Factor(gtsam.Point2(img_line_start[0],img_line_start[1]), gtsam.Point2(img_line_end[0],img_line_end[1]), K,
                                     camera_pose, point_key, gtsam.noiseModel.Isotropic.Sigma(1, 1)))

    # optimize
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY")
    optimizer = gtsam.LevenbergMarquardtOptimizer(
        graph, initial_estimate, params)
    try:
        result = optimizer.optimize()
    except:
        print("Optimization failed")
        result = initial_estimate

    # visualize result
    point_opt = result.atPoint3(point_key)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

    camera_pose_axes.add_collection3d(
        Line3DCollection([[line_start, line_end]], facecolors="r", linewidths=0.1, edgecolors="r", alpha=0.9))

    draw_3dpoints(camera_pose_axes, point_opt, size=0.1, color='g',)

    plt.show()


# test_Point2ToLine2Factor()
test_Point3ProjectLine2Factor()
