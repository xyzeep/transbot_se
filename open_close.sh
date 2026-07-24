#!/bin/bash


echo "start"
rostopic pub -1 /TargetAngle transbot_msgs/Arm '{joint: [{id: 7, angle: 225, run_time: 500}]}'
echo "7 up 225"
rostopic pub -1 /TargetAngle transbot_msgs/Arm '{joint: [{id: 8, angle: 270, run_time: 500}]}'
echo "8 up 270"
rostopic pub -1 /TargetAngle transbot_msgs/Arm '{joint: [{id: 9, angle: 30, run_time: 500}]}'
echo "9 close"

rostopic pub -1 /TargetAngle transbot_msgs/Arm '{joint: [{id: 7, angle: 0, run_time: 500}]}'
echo "7 down 0"
rostopic pub -1 /TargetAngle transbot_msgs/Arm '{joint: [{id: 8, angle: 210, run_time: 500}]}'
echo "8 down 210"
rostopic pub -1 /TargetAngle transbot_msgs/Arm '{joint: [{id: 9, angle: 180, run_time: 500}]}'
echo "9 close 180"
