from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

import time, glob
import cv2  # used to probe which /dev/videoN can actually open

def _parse_indices_arg(s: str):
    s = (s or "").strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts if p.isdigit()]

def _find_video_indices(wait_seconds: float = 3.0):
    """Find existing /dev/video* (presence only)."""
    deadline = time.time() + wait_seconds
    devices = []
    while time.time() < deadline:
        devices = sorted(glob.glob("/dev/video*"))
        if devices:
            break
        time.sleep(0.2)
    indices = []
    for dev in devices:
        tail = dev.replace("/dev/video", "")
        if tail.isdigit():
            indices.append(int(tail))
    return sorted(set(indices))

def _can_open(idx: int) -> bool:
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    ok = cap.isOpened()
    cap.release()
    return ok

def _launch_setup(context, *args, **kwargs):
    # Read CLI args / defaults
    idx_arg = LaunchConfiguration("indices").perform(context)
    requested = _parse_indices_arg(idx_arg) or _find_video_indices()

    model_path     = LaunchConfiguration("model_path").perform(context)
    width          = float(LaunchConfiguration("width").perform(context))
    height         = float(LaunchConfiguration("height").perform(context))
    fps            = float(LaunchConfiguration("fps").perform(context))
    fourcc         = LaunchConfiguration("fourcc").perform(context)
    min_conf       = float(LaunchConfiguration("min_conf").perform(context))
    publish_overlay= LaunchConfiguration("publish_overlay").perform(context).lower() in ("1","true","yes","on")

    print(f"[multi_camera_yolo] requested indices: {requested}")

    usable = [i for i in requested if _can_open(i)]
    print(f"[multi_camera_yolo] openable indices: {usable}")

    nodes = []
    for idx in usable:
        label = f"cam{idx}"
        nodes.append(
            Node(
                package="moe_yolo_pipeline",
                executable="yolo_inference_node",
                namespace=label,                        # ← topics become /camX/...
                name=f"yolo_inference_{label}",
                output="screen",
                parameters=[{
                    "camera_index": idx,
                    "camera_label": label,
                    "model_path": model_path,
                    "width": int(width),
                    "height": int(height),
                    "fps": fps,
                    "fourcc": fourcc,
                    "min_conf": min_conf,
                    "publish_overlay": publish_overlay,
                    # Separate CSV per cam for convenience:
                    "csv_path": f"/home/nvidia/yolo_detections_{label}.csv",
                }],
            )
        )

    if not nodes:
        print("[multi_camera_yolo] No usable cameras found. Exiting.")
    return nodes

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "indices",
            default_value="",
            description="Comma-separated camera indices (e.g. '0,4'). Leave empty to auto-detect."
        ),
        DeclareLaunchArgument("model_path",      default_value="yolov8n.pt"),
        DeclareLaunchArgument("width",           default_value="640"),
        DeclareLaunchArgument("height",          default_value="480"),
        DeclareLaunchArgument("fps",             default_value="15.0"),
        DeclareLaunchArgument("fourcc",          default_value="MJPG"),
        DeclareLaunchArgument("min_conf",        default_value="0.25"),
        DeclareLaunchArgument("publish_overlay", default_value="true"),
        OpaqueFunction(function=_launch_setup),
    ])
