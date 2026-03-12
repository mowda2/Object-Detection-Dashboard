import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class DualCameraPublisher(Node):
    def __init__(self, topic_name):
        super().__init__('dual_camera_publisher')
        self.bridge = CvBridge()

        # Create publishers for each camera
        self.cam1_pub = self.create_publisher(Image, topic_name, 10)
        self.cam2_pub = self.create_publisher(Image, '/image_cam2', 10)

        # Open both cameras (adjust device indices as needed)
        self.cap1 = cv2.VideoCapture(0)  # iCatchtek SPCA6350
        self.cap2 = cv2.VideoCapture(4)  # Insta360 X3

        # Set timers to publish frames
        self.timer1 = self.create_timer(0.03, self.publish_cam1)
        self.timer2 = self.create_timer(0.03, self.publish_cam2)

    def publish_cam1(self):
        ret, frame = self.cap1.read()
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
            self.cam1_pub.publish(msg)

    def publish_cam2(self):
        ret, frame = self.cap2.read()
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
            self.cam2_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = DualCameraPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
