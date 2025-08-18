import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2, json

class YOLOVisualizationNode(Node):
    def __init__(self):
        super().__init__('yolo_visualization_node')
        self.bridge = CvBridge()
        self.detections = []
        self.create_subscription(String, '/yolo/detections', self.detection_callback, 10)
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.overlay_pub = self.create_publisher(Image, '/yolo/image_overlay', 10)

    def detection_callback(self, msg):
        self.detections = json.loads(msg.data)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        for det in self.detections:
            x, y, w, h = det["x"], det["y"], det["width"], det["height"]
            label, conf = det["label"], det["confidence"]
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        self.overlay_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))

def main(args=None):
    rclpy.init(args=args)
    node = YOLOVisualizationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
