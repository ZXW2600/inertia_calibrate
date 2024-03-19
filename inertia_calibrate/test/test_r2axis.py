from inertia_calibrate.utils.geometery import RotationMatrix2AxisAngle

def test_RotationMatrix2AxisAngle():
    import gtsam
    import numpy as np
    test_num=10000
    for i in range(test_num):
        axis=np.random.randn(3)
        axis=axis/np.linalg.norm(axis)

        angle=i/test_num*np.pi*2-np.pi
        if np.allclose(angle,0):
            continue
        if np.allclose(angle,np.pi):
            continue
        if np.allclose(angle,-np.pi):
            continue
        R=gtsam.Rot3.AxisAngle(axis,angle).matrix()   
        axis1,angle1=RotationMatrix2AxisAngle(R)

        print("--"*50)
        print(f"test angle {angle/np.pi*180} cal angle {angle1/np.pi*180}")
        if np.allclose(axis,-axis1,atol=1e-6):
            angle2=-angle1
        elif np.allclose(axis,axis1,atol=1e-6):
            angle2=angle1
        else:
            print(f"axis:{axis} axis cal {axis1} ")

            raise ValueError
        print(f"axis:{axis} axis cal {axis1} ")
        d_angle=gtsam.Rot2(angle-angle2).theta()
    
        print(f"cal angle {angle2/np.pi*180} dangle {d_angle/np.pi*180}")
        assert np.allclose(d_angle,0,atol=1e-6)

def test_gtsam_rot_2_axisangle():
    import gtsam
    import numpy as np
    for i in range(1000):
        axis=np.random.randn(3)
        axis=axis/np.linalg.norm(axis)
        angle=(i+1)/1000.0*np.pi*2-np.pi
        if np.allclose(angle,0):
            continue
        print("_"*50)
        print(f"angle:{angle/np.pi*180}")

        R=gtsam.Rot3.AxisAngle(axis,angle).matrix()   

        axis1,angle1=gtsam.Rot3(R=R).axisAngle()
        axis2=axis1.point3()

        if np.allclose(axis,-axis2,atol=1e-2):
            angle2=-angle1
        elif np.allclose(axis,axis2,atol=1e-2):
            angle2=angle1
        else:
            print(f"axis:{axis} axis cal {axis2} ")

            raise ValueError
        print(f"axis:{axis} axis cal {axis2} ")
        d_angle=gtsam.Rot2(angle-angle2).theta()
    
        print(f"cal angle {angle2/np.pi*180} dangle {d_angle/np.pi*180}")
        assert np.allclose(d_angle,0,atol=1e-2)

test_RotationMatrix2AxisAngle()
# test_gtsam_rot_2_axisangle()