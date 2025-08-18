import os, json, csv, cv2
from datetime import datetime

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO


# ---------------- CSV helper (modular & reusable) ----------------
class CsvLogger:
    def __init__(
        self,
        enabled: bool,
        directory: str,
        filename: str,
        per_camera: bool,
        camera_label: str,
        include_header: bool,
        append: bool,
        datetime_fmt: str,
        flush_every_n: int,
    ):
        self.enabled = bool(enabled)
        self.datetime_fmt = datetime_fmt
        self.flush_every_n = max(int(flush_every_n), 0)
        self._rows_since_flush = 0
        self.fp = None
        self.writer = None

        if not self.enabled:
            return

        if not directory:
            directory = os.path.expanduser("~/yolo_logs")
        os.makedirs(directory, exist_ok=True)

        if filename:
            path = os.path.join(directory, filename)
        else:
            base = f"{camera_label}.csv" if per_camera else "yolo_detections_log.csv"
            path = os.path.join(directory, base)

        mode = "a" if append else "w"
        new_file = not os.path.exists(path) or not append
        self.fp = open(path, mode, newline="")
        self.writer = csv.writer(self.fp)

        if include_header and new_file:
            self.writer.writerow(["Camera", "Timestamp", "Label", "Confidence", "X", "Y", "Width", "Height"])
            self._maybe_flush(force=True)

    def log_detections(self, camera_label: str, detections: list):
        if not self.enabled or not self.writer or not detections:
            return
        ts = datetime.now().strftime(self.datetime_fmt)
        for d in detections:
            self.writer.writerow([
                camera_label, ts, d["label"], f'{d["confidence"]:.2f}',
                d["x"], d["y"], d["width"], d["height"]
            ])
        self._maybe_flush()

    def _maybe_flush(self, force: bool = False):
        if not self.enabled or not self.fp:
            return
        if force:
            self.fp.flush()
            self._rows_since_flush = 0
            return
        if self.flush_every_n == 0:
            return
        self._rows_since_flush += 1
        if self._rows_since_flush >= self.flush_every_n:
            self.fp.flush()
            self._rows_since_flush = 0

    def close(self):
        try:
            if self.fp:
                self.fp.flush()
                self.fp.close()
        except Exception:
            pass
        finally:
            self.fp = None
            self.writer = None


# ---------------- Inference node ----------------
class YOLOInferenceNode(Node):
    def __init__(self):
        super().__init__("yolo_inference_node")

        # -------- Parameters (all optional) --------
        # Camera / model
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("camera_label", "")
        self.declare_parameter("model_path", "yolov8n.pt")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 15.0)
        self.declare_parameter("fourcc", "MJPG")
        self.declare_parameter("min_conf", 0.25)
        self.declare_parameter("publish_overlay", True)
        self.declare_parameter("print_detections", False)
        # NEW: raw stream toggle
        self.declare_parameter("publish_raw", True)

        # CSV group
        self.declare_parameter("csv.enabled", True)
        self.declare_parameter("csv.dir", os.path.expanduser("~/yolo_logs"))
        self.declare_parameter("csv.filename", "")
        self.declare_parameter("csv.per_camera", True)
        self.declare_parameter("csv.include_header", True)
        self.declare_parameter("csv.append", True)
        self.declare_parameter("csv.datetime_format", "%Y-%m-%d %H:%M:%S")
        self.declare_parameter("csv.flush_every_n", 0)

        # Resolve params
        self.camera_index = int(self.get_parameter("camera_index").value)
        label_param = self.get_parameter("camera_label").value
        self.camera_label = label_param if label_param else f"cam{self.camera_index}"
        model_path = str(self.get_parameter("model_path").value)
        self.target_w = int(self.get_parameter("width").value)
        self.target_h = int(self.get_parameter("height").value)
        self.target_fps = float(self.get_parameter("fps").value)
        self.fourcc_name = str(self.get_parameter("fourcc").value).upper()
        self.min_conf = float(self.get_parameter("min_conf").value)
        self.publish_overlay = bool(self.get_parameter("publish_overlay").value)
        self.print_detections = bool(self.get_parameter("print_detections").value)
        self.publish_raw = bool(self.get_parameter("publish_raw").value)  # NEW

        # CSV logger
        self.csv = CsvLogger(
            enabled=self.get_parameter("csv.enabled").value,
            directory=str(self.get_parameter("csv.dir").value),
            filename=str(self.get_parameter("csv.filename").value),
            per_camera=bool(self.get_parameter("csv.per_camera").value),
            camera_label=self.camera_label,
            include_header=bool(self.get_parameter("csv.include_header").value),
            append=bool(self.get_parameter("csv.append").value),
            datetime_fmt=str(self.get_parameter("csv.datetime_format").value),
            flush_every_n=int(self.get_parameter("csv.flush_every_n").value),
        )

        self.get_logger().info(
            f"Starting on /dev/video{self.camera_index} "
            f"({self.target_w}x{self.target_h}@{self.target_fps} {self.fourcc_name}), "
            f"label='{self.camera_label}', model='{model_path}'"
        )

        # -------- Model / bridge --------
        self.model = YOLO(model_path)
        self.bridge = CvBridge()

        # -------- Publishers --------
        self.det_pub = self.create_publisher(String, "yolo/detections", 10)
        if self.publish_overlay:
            self.overlay_pub = self.create_publisher(Image, "yolo/image_overlay", 10)
        if self.publish_raw:
            # IMPORTANT: publish raw at top-level in the namespace, not under /yolo/
            self.raw_pub = self.create_publisher(Image, "image_raw", 10)

        # -------- Open camera --------
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error(f"❌ Failed to open /dev/video{self.camera_index}")
            raise RuntimeError(f"Cannot open camera at index {self.camera_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_h)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        def set_fourcc(name: str) -> bool:
            code = cv2.VideoWriter_fourcc(*name)
            return bool(self.cap.set(cv2.CAP_PROP_FOURCC, code))

        if not set_fourcc(self.fourcc_name) and self.fourcc_name != "YUYV":
            if set_fourcc("YUYV"):
                self.get_logger().warn(f"FOURCC {self.fourcc_name} not accepted; fell back to YUYV")

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.get_logger().info(f"Camera configured -> {actual_w}x{actual_h} @ {actual_fps:.1f} fps")

        self.timer = self.create_timer(1.0 / max(1.0, self.target_fps), self.process_frame)

    def process_frame(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.get_logger().warn("⚠️ Failed to read frame")
            return

        # --- publish RAW first (unmodified frame) ---
        if self.publish_raw:
            raw_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            raw_msg.header.stamp = self.get_clock().now().to_msg()
            self.raw_pub.publish(raw_msg)

        # Inference
        res = self.model(frame, verbose=False)[0]
        detections = []
        for box in res.boxes:
            conf = float(box.conf[0])
            if conf < self.min_conf:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label_id = int(box.cls[0])
            label = self.model.names[label_id]
            detections.append({
                "x": x1, "y": y1,
                "width": x2 - x1, "height": y2 - y1,
                "label": label, "confidence": conf
            })

            if self.publish_overlay:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Publish JSON
        self.det_pub.publish(String(data=json.dumps(detections)))

        # Publish overlay
        if self.publish_overlay:
            msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            self.overlay_pub.publish(msg)

        # Log CSV (buffered)
        self.csv.log_detections(self.camera_label, detections)

        if self.print_detections and detections:
            labels = ", ".join(f"{d['label']}:{d['confidence']:.2f}" for d in detections[:5])
            self.get_logger().info(f"{self.camera_label}: {len(detections)} detections -> {labels}")

    def destroy_node(self):
        try:
            if hasattr(self, "cap") and self.cap:
                self.cap.release()
            if hasattr(self, "csv") and self.csv:
                self.csv.close()
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = YOLOInferenceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
