

import gtsam
import numpy as np
from apriltag_calibrate.visualization import draw_axes,draw_3dpoints
import matplotlib.pyplot as plt
from inertia_calibrate.custom_factor.RotAxisFactor import RotAxisFactor

def test_rotate_factor():
    world_T_rotate_axis_pose=gtsam.Pose3(
        r=gtsam.Rot3.AxisAngle(np.random.randn(3),np.random.rand()),
        t=np.array([1,1,1])
    )
    angle=0
    
    world_T_init_pose=gtsam.Pose3()
    rotate_ref_T_init_pose=world_T_rotate_axis_pose.transformPoseTo(world_T_init_pose)

    figure=plt.figure()
    axes=figure.add_subplot(111, projection='3d')
    axes.set_xlim(-2,2)
    axes.set_ylim(-2,2)
    axes.set_zlim(-2,2)
    draw_axes(axes,world_T_init_pose.matrix(),size=0.2,alpha=0.3)
    draw_axes(axes,world_T_rotate_axis_pose.matrix(),size=0.2,alpha=0.3)


    pose_list=[]
    pose_noise_list=[]
    angle_list=[]

    avr_t=gtsam.Point3(0,0,0)
    point_num=100
    for i in range(point_num):
        angle=np.random.rand()*0.7

        rotate_pose=gtsam.Pose3(
            r=gtsam.Rot3.Rz(angle),t=gtsam.Point3(0,0,0)
        )
        noise_pose=gtsam.Pose3(
            r=gtsam.Rot3.AxisAngle(axis=np.random.randn(3),angle=np.random.rand()*0.3),
            t=np.zeros(3)
        )
        ref_pose_i=rotate_pose.compose(rotate_ref_T_init_pose)
        world_pose_i=world_T_rotate_axis_pose.transformPoseFrom(ref_pose_i)
        world_pose_i_noise=world_pose_i.compose(noise_pose)
        draw_axes(axes,world_pose_i_noise.matrix())
        draw_axes(axes,world_pose_i.matrix())
        
        pose_list.append(world_pose_i)
        pose_noise_list.append(world_pose_i_noise)
        angle_list.append(angle)
        avr_t+=world_pose_i_noise.translation()/point_num

    graph=gtsam.NonlinearFactorGraph()
    initial_estimate=gtsam.Values()
    key_axis_pose=gtsam.symbol('a',0)

    init_axes_pose=gtsam.Pose3(
        r=gtsam.Rot3(),t=avr_t)
    initial_estimate.insert(key_axis_pose,init_axes_pose)

    pose_noise=gtsam.noiseModel.Isotropic.Sigma(6,0.1)
    angle_index=0
    for i,pose in enumerate(pose_list):
    # for i,pose in enumerate(pose_noise_list):
        key_angle=gtsam.symbol('b',angle_index)
        angle_index+=1
        graph.add(RotAxisFactor(pose_before=world_T_init_pose,
                                pose_after=pose,
                                key_rotate_angle=key_angle,
                                key_rotate_axis_pose=key_axis_pose,
                                noise_model=pose_noise))
        initial_estimate.insert(key_angle,0)
    
    # solve
    params=gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY")

    optimizer=gtsam.LevenbergMarquardtOptimizer(graph,initial_estimate,params)
    result=optimizer.optimize()


    # draw
    rotate_axes_opt=result.atPose3(key_axis_pose)
    draw_axes(axes,rotate_axes_opt.matrix(),size=0.2)
    # draw_axes(axes,init_axes_pose.matrix(),size=0.4)

    print("ground truth axis pose:",world_T_rotate_axis_pose)
    print("optimized axis pose:",rotate_axes_opt)

    angle_error_list=[]
    for i,angle in enumerate(angle_list):
        key_angle=gtsam.symbol('b',i)
        opt_angle=-result.atDouble(key_angle)


        gt_rot=gtsam.Rot2(angle)
        opt_rot=gtsam.Rot2(opt_angle)
        # print(f"gt rot:{gt_rot}, opt rot:{opt_rot}")
        print(f"gt angle:{gt_rot.theta()}, opt angle:{opt_rot.theta()}")
        angle_error_list.append(np.abs(gt_rot.between(opt_rot).theta()))

    # print(angle_error_list)
    print("mean angle error:",np.mean(angle_error_list))
    print("max angle error:",np.max(angle_error_list))
    # print(f"init axes :{init_axes_pose}")
    plt.show()


test_rotate_factor()