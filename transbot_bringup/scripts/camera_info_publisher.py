#!/usr/bin/env python3
import rospy
import yaml
from sensor_msgs.msg import CameraInfo

def load_camera_info(yaml_path):
    with open(yaml_path, 'r') as f:
        calib = yaml.safe_load(f)

    info = CameraInfo()
    info.width = calib['image_width']
    info.height = calib['image_height']
    info.K = calib['camera_matrix']['data']
    info.D = calib['distortion_coefficients']['data']
    info.R = calib['rectification_matrix']['data']
    info.P = calib['projection_matrix']['data']
    info.distortion_model = calib['camera_model']
    return info

if __name__ == '__main__':
    rospy.init_node('camera_info_publisher')

    yaml_path = rospy.get_param('~yaml_path', '/home/pi/.ros/camera_info/camera.yaml')
    frame_id = rospy.get_param('~frame_id', 'camera_link')
    rate_hz = rospy.get_param('~rate', 30)

    info = load_camera_info(yaml_path)
    info.header.frame_id = frame_id

    pub = rospy.Publisher('/camera_info', CameraInfo, queue_size=10)
    rate = rospy.Rate(rate_hz)

    rospy.loginfo("Publishing camera_info from %s on /camera_info", yaml_path)

    while not rospy.is_shutdown():
        info.header.stamp = rospy.Time.now()
        pub.publish(info)
        rate.sleep()
