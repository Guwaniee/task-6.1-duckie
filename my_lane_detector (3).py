#!/usr/bin/env python3

# Python Libs
import sys, time

# numpy
import numpy as np

# OpenCV
import cv2
from cv_bridge import CvBridge

# ROS Libraries
import rospy
import roslib

# ROS Message Types
from sensor_msgs.msg import CompressedImage

class Lane_Detector:
    def __init__(self):
        self.cv_bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/duckiegk/camera_node/image/compressed', CompressedImage, self.image_callback, queue_size=1)
        rospy.init_node("my_lane_detector")

    def image_callback(self, msg):
        rospy.loginfo("image_callback")
        
        # Convert to OpenCV image
        img = self.cv_bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

        # Crop the image to focus on the road (adjust these values as necessary)
        height, width, _ = img.shape
        crop_img = img[int(height/2):, :]

        # Convert to HSV color space
        hsv_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

        # Color filtering for white pixels
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 25, 255])
        white_mask = cv2.inRange(hsv_img, lower_white, upper_white)
        white_output = cv2.bitwise_and(crop_img, crop_img, mask=white_mask)

        # Color filtering for yellow pixels
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
        yellow_output = cv2.bitwise_and(crop_img, crop_img, mask=yellow_mask)

        # Apply Canny Edge Detector to the cropped image
        edges = cv2.Canny(crop_img, 50, 150)

        # Hough Transform for white filtered image
        white_lines = cv2.HoughLinesP(white_mask, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)

        # Hough Transform for yellow filtered image
        yellow_lines = cv2.HoughLinesP(yellow_mask, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)

        # Draw lines found on both Hough Transforms on the cropped image
        line_image = np.copy(crop_img)
        self.draw_lines(line_image, white_lines, (255, 255, 255))
        self.draw_lines(line_image, yellow_lines, (0, 255, 255))

        # Show image in a window
        cv2.imshow('Cropped Image', crop_img)
        cv2.imshow('White Filtered Image', white_output)
        cv2.imshow('Yellow Filtered Image', yellow_output)
        cv2.imshow('Edges', edges)
        cv2.imshow('Lane Lines', line_image)
        cv2.waitKey(1)

    def draw_lines(self, img, lines, color):
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img, (x1, y1), (x2, y2), color, 2)

    def run(self):
        rospy.spin()  # Spin forever but listen to message callbacks

if __name__ == "__main__":
    try:
        lane_detector_instance = Lane_Detector()
        lane_detector_instance.run()
    except rospy.ROSInterruptException:
        pass
