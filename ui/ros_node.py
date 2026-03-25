#!/usr/bin/env python3

import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class ImageSubscriber(Node):
    def __init__(self, topic_name, image_callback, keep_ros_node):
        """
        Initialize the image subscriber node.
        
        :param topic_name: ROS2 topic name to subscribe for images.
        :param image_callback: Callback function that will be called with a cv2 image.
        """
        rclpy.init(args=None)

        super().__init__('image_subscriber_node')
        self.bridge = CvBridge()
        self.user_callback = image_callback

        # Create a subscription to the specified image topic.
        self.subscription = self.create_subscription(
            Image,
            topic_name,
            self._internal_callback,
            10  # QoS history depth
        )
        self.subscription  # prevent unused variable warning
        self.get_logger().info(f"Subscribed to {topic_name}")

        try:
            while keep_ros_node():
                rclpy.spin_once(self)
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.destroy_node()
            rclpy.shutdown()

    def _internal_callback(self, msg):
        """
        Internal callback that converts the ROS Image message to an OpenCV image,
        then calls the user-supplied callback.
        """
        try:
            # Convert the ROS Image message to an OpenCV image (BGR8 format).
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return
        
        # Call the user-supplied callback with the converted image.
        self.user_callback(cv_image)

def main_ros_loop(topic_name, ros_image_callback, keep_ros_node, args=None):

    ros_node = ImageSubscriber(topic_name, ros_image_callback, keep_ros_node)

    # rclpy.init(args=args)

    # # Define a simple callback function that prints out image information.
    # def my_image_callback(cv_image):
    #     print("Received an image of shape:", cv_image.shape)
    #     # Add further processing for the OpenCV image here.

    # # Create the image subscriber node.
    # image_subscriber = ImageSubscriber(topic_name, my_image_callback)

    # try:
    #     rclpy.spin(image_subscriber)
    # except KeyboardInterrupt:
    #     pass
    # finally:
    #     image_subscriber.destroy_node()
    #     rclpy.shutdown()
