from setuptools import setup, find_packages

setup(
    name='inertia_calibrate',
    version='1.0.0',
    description='Inertia Params Calibrate Tools',
    author='ZXW2600',
    author_email='zhaoxinwei74@gmail.com',
    packages=find_packages(),
    install_requires=[
        "gtsam",
        "tqdm",
        "pyyaml",
        "apriltag",
        "opencv-python",
        "matplotlib",
    ]
)
