import numpy as np

def RotationMatrix2AxisAngle(R: np.ndarray) -> np.ndarray:
    """
    Convert a rotation matrix to axis angle representation
    :param R: 3x3 rotation matrix
    :return: 3x1 axis angle representation
    """
    cos=(np.trace(R)-1)/2
    XYZ=R-R.transpose()
    sin=np.sqrt(XYZ[2,1]**2+XYZ[0,2]**2+XYZ[1,0]**2)/2
    angle=np.arctan2(sin,cos)
    axis=np.array([XYZ[2,1],XYZ[0,2],XYZ[1,0]])/(2*sin)
    return axis,angle

