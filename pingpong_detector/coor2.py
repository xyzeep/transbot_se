import cv2
import numpy as np
import time
import rospy
from geometry_msgs.msg import Twist
from transbot_msgs.msg import Arm, Joint

# ?????????????????????????????????????????
#  CONFIG
# ?????????????????????????????????????????
FRAME_W, FRAME_H = 640, 480

# Target pixel coordinate the ball's center should reach.
# TARGET_X: horizontal center (usually frame center, keeps ball centered left/right)
# TARGET_Y: vertical position that corresponds to "ball is close enough to pick up".
#           CALIBRATE THIS: place the ball exactly where you want the bot to stop,
#           read the printed (x, y) on screen, and set TARGET_Y to that y value.
TARGET_X = FRAME_W // 2 + 60
TARGET_Y = 380   # placeholder ? calibrate this on your setup

TOLERANCE_X = 15   # pixels, dead zone for turning
TOLERANCE_Y = 15   # pixels, dead zone for stopping

# Orange HSV range ? tuned from your color-picked values
LOWER_ORANGE = np.array([10, 170, 100])
UPPER_ORANGE = np.array([22, 255, 255])

MIN_AREA        = 30
MIN_CIRCULARITY = 0.55

MAX_LINEAR  = 0.15
MAX_ANGULAR = 0.4

LAST_SEEN_TIMEOUT = 0.8   # seconds ? keep acting on last known position within this window
# Arm joint IDs and angles for the pickup sequence ? tune these to your setup
SHOULDER_ID = 7
ELBOW_ID    = 8
GRIPPER_ID  = 9

GRIPPER_OPEN   = 30
GRIPPER_CLOSED = 80

SHOULDER_READY = 120   # arm raised, out of camera's way, gripper open
SHOULDER_DOWN  = 60    # arm lowered toward ball
ELBOW_READY    = 230
ELBOW_DOWN     = 250

ARM_STEP_DELAY = 0.8   # seconds to wait between joint moves, let servos catch up


# ?????????????????????????????????????????
#  GRAYSCALE ROUNDNESS CHECK (kept from yesterday ? rejects skin blobs)
# ?????????????????????????????????????????
def confirm_circle_bnw(frame, cx, cy, radius):
    pad = int(radius * 1.5) + 10
    x1 = max(0, cx - radius - pad)
    y1 = max(0, cy - radius - pad)
    x2 = min(frame.shape[1], cx + radius + pad)
    y2 = min(frame.shape[0], cy + radius + pad)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 2)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=gray.shape[0],
        param1=80,
        param2=25,
        minRadius=max(3, int(radius * 0.5)),
        maxRadius=int(radius * 1.8) + 5
    )
    return circles is not None


# ?????????????????????????????????????????
#  FIND BALL ? color mask candidates + b&w circle confirmation
# ?????????????????????????????????????????
def find_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        candidates.append((int(cx), int(cy), int(radius), area * circularity))

    candidates.sort(key=lambda c: c[3], reverse=True)

    for (cx, cy, radius, _) in candidates[:5]:
        if confirm_circle_bnw(frame, cx, cy, radius):
            return (cx, cy, radius), mask

    return None, mask


# ?????????????????????????????????????????
#  POSITION-BASED CONTROL (no depth used)
# ?????????????????????????????????????????
def compute_velocity(cx, cy):
    x_error = cx - TARGET_X
    y_error = TARGET_Y - cy   # positive => ball is above target => still far away, drive forward

    # turning: always correct horizontal centering
    if abs(x_error) <= TOLERANCE_X:
        angular = 0.0
    else:
        angular = max(-MAX_ANGULAR, min(MAX_ANGULAR, -0.004 * x_error))

    # forward/stop: drive until ball's y reaches target row
    if abs(y_error) <= TOLERANCE_Y:
        linear = 0.0
        reached = True
    else:
        linear = max(0.0, min(MAX_LINEAR, 0.005 * y_error))
        reached = False

    return linear, angular, reached


def stop_robot(cmd_pub):
    cmd_pub.publish(Twist())

def move_joint(arm_pub, joint_id, angle, run_time=1000):
    joint = Joint()
    joint.id = joint_id
    joint.angle = angle
    joint.run_time = run_time

    msg = Arm()
    msg.joint = [joint]
    arm_pub.publish(msg)

def move_joints_multi(arm_pub, id_angle_list, run_time=1000):
    joints = []
    for joint_id, angle in id_angle_list:
        j = Joint()
        j.id = joint_id
        j.angle = angle
        j.run_time = run_time
        joints.append(j)

    msg = Arm()
    msg.joint = joints
    arm_pub.publish(msg)

def pickup_sequence(arm_pub):
    print("Starting pickup sequence...")

    move_joint(arm_pub, 7, 0)
    time.sleep(ARM_STEP_DELAY)

    move_joint(arm_pub, 8, 250)
    time.sleep(ARM_STEP_DELAY)

    move_joint(arm_pub, 9, 120)
    time.sleep(ARM_STEP_DELAY)

    # move_joints_multi(arm_pub, [(8, 250), (9, 120)])
    # time.sleep(ARM_STEP_DELAY)

    print("Pickup sequence done.")


# ?????????????????????????????????????????
#  MAIN
# ?????????????????????????????????????????
def detect():
    rospy.init_node('pingpong_coord_detector', anonymous=True)
    cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    arm_pub = rospy.Publisher('/TargetAngle', Arm, queue_size=1)

    while arm_pub.get_num_connections() == 0 and not rospy.is_shutdown():
        print("Waiting for /TargetAngle subscriber...")
        rospy.sleep(0.2)
    print("Arm publisher connected.")

    use_realsense = False
    pipeline = None
    cap = None
    try:
        import pyrealsense2 as rs
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, 30)
        pipeline.start(config)
        use_realsense = True
        print("RealSense D435 detected and started! (color only, no depth needed)")
    except Exception as e:
        print(f"RealSense not available ({e})")
        print("Falling back to webcam...")
        cap = cv2.VideoCapture(4)
        if not cap.isOpened():
            print("ERROR: No camera found.")
            return

    print("Press Q to quit\n")

    last_cx = None
    last_cy = None
    last_seen_time = 0.0
    pickup_done = False
    arm_active = False   # once True, detection stops entirely, only arm logic runs
    reached_count = 0
    REACHED_CONFIRM_FRAMES = 60   # must stay "reached" for this many consecutive frames

    try:
        while not rospy.is_shutdown():
            if use_realsense:
                frames = pipeline.wait_for_frames()
                color_f = frames.get_color_frame()
                if not color_f:
                    continue
                frame = np.asanyarray(color_f.get_data())
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            h, w = frame.shape[:2]

            if arm_active:
                # detection is fully paused while the arm handles pickup
                cv2.putText(frame, "Arm active ? detection paused", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                cv2.imshow("Ping Pong Detector ? Coordinate Target", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            ball, mask = find_ball(frame)

            # draw target crosshair for calibration/visual reference
            cv2.drawMarker(frame, (TARGET_X, TARGET_Y), (255, 0, 255),
                            markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

            now = time.time()

            if ball is not None:
                # fresh detection ? always wins over stale last-known position
                cx, cy, radius = ball
                last_cx, last_cy = cx, cy
                last_seen_time = now
                using_stale = False
            elif last_cx is not None and (now - last_seen_time) < LAST_SEEN_TIMEOUT:
                # no detection this frame, but still within grace window ?
                # keep acting on the last known position instead of stopping
                cx, cy, radius = last_cx, last_cy, 10
                using_stale = True
            else:
                cx = cy = radius = None
                using_stale = False

            if cx is not None:
                linear, angular, reached = compute_velocity(cx, cy)

                twist = Twist()
                twist.linear.x = linear
                twist.angular.z = angular
                cmd_pub.publish(twist)

                if reached:
                    reached_count += 1
                    color = (0, 255, 0)
                    status = f"TARGET REACHED ? confirming ({reached_count}/{REACHED_CONFIRM_FRAMES})"

                    if reached_count >= REACHED_CONFIRM_FRAMES and not pickup_done:
                        stop_robot(cmd_pub)
                        arm_active = True
                        pickup_sequence(arm_pub)
                        pickup_done = True


                elif using_stale:
                    reached_count = 0
                    color = (0, 165, 255)
                    status = "Ball LOST ? using last known position"
                else:
                    reached_count = 0
                    color = (0, 200, 255)
                    status = "Ball DETECTED ? moving"

                cv2.circle(frame, (cx, cy), radius, color, 2)
                cv2.circle(frame, (cx, cy), 5, color, -1)
                cv2.putText(frame, status, (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"Ball (x={cx}, y={cy})  Target (x={TARGET_X}, y={TARGET_Y})",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            else:
                stop_robot(cmd_pub)
                cv2.putText(frame, "No ball", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            mode_str = "RealSense D435 (color only)" if use_realsense else "Webcam"
            cv2.putText(frame, mode_str, (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            mask_small = cv2.resize(mask, (w // 5, h // 5))
            mask_color = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
            frame[h - h // 5 - 20:h - 20, 0:w // 5] = mask_color

            cv2.imshow("Ping Pong Detector ? Coordinate Target", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        stop_robot(cmd_pub)
        if use_realsense and pipeline:
            pipeline.stop()
        elif cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detect()

