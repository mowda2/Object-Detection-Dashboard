from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os

def generate_launch_description():
    bag_path = os.path.expanduser('~/RISP-july')  # Change if needed

    return LaunchDescription([
        # 1. ROS2 Bag Playback
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', bag_path, '--loop'],
            output='screen'
        ),

        # 2. YOLO Inference Node
        Node(
            package='moe_yolo_pipeline',
            executable='yolo_inference_node',
            name='yolo_inference_node',
            output='screen',
            remappings=[('/camera/image_raw', '/flir_boson/image_raw')]
        ),

        # 3. YOLO Visualization Node
        Node(
            package='moe_yolo_pipeline',
            executable='yolo_visualization_node',
            name='yolo_visualization_node',
            output='screen',
            remappings=[('/camera/image_raw', '/flir_boson/image_raw')]
        ),
    ])
