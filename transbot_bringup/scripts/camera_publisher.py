#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image

def main():
    rospy.init_node('camera_publisher')
    pub = rospy.Publisher('/image', Image, queue_size=10)
    rate = rospy.Rate(30)
    rospy.sleep(5)  # wait for other nodes to finish initializing

    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    if not cap.isOpened():
        rospy.logerr("Failed to open /dev/video0")
        return

    # Warm up: discard first few frames, MJPG cameras need buffer time
    for _ in range(10):
        cap.read()

    rospy.loginfo("Camera publisher started on /image")

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            rospy.logwarn("Failed to grab frame, retrying...")
            continue
        msg = Image()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'camera_link'
        msg.height = frame.shape[0]
        msg.width = frame.shape[1]
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = frame.shape[1] * 3
        msg.data = frame.tobytes()
        pub.publish(msg)
        rate.sleep()

    cap.release()

if __name__ == '__main__':
    main()

