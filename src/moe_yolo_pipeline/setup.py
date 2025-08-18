import os
import glob
from setuptools import find_packages, setup

package_name = 'moe_yolo_pipeline'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # include launch files
        (os.path.join('share', package_name, 'launch'),
         glob.glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools'],  # ROS deps come from package.xml
    zip_safe=True,
    maintainer='nvidia',
    maintainer_email='nvidia@todo.todo',
    description='YOLO pipeline with multi-camera inference, visualization, and web video bridge',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_publisher = moe_yolo_pipeline.camera_publisher:main',
            'yolo_inference_node = moe_yolo_pipeline.yolo_inference_node:main',
            'yolo_visualization_node = moe_yolo_pipeline.yolo_visualization_node:main',
            # NEW: Flask-based multi-topic web streamer
            'web_video_bridge = moe_yolo_pipeline.web_video_bridge:main',
        ],
    },
)
