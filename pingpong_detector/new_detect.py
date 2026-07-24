import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Twist

# ?????????????????????????????????????????
#  CONFIG
# ?????????????????????????????????????????
FRAME_W, FRAME_H = 640, 480
FRAME_CENTER_X   = FRAME_W // 2

# Orange HSV range ? tuned from your color-picked values
LOWER_ORANGE = np.array([10, 170, 100])
UPPER_ORANGE = np.array([22, 255, 255])

MIN_AREA        = 30
MIN_CIRCULARITY = 0.55   # used only as a first-pass area/shape sanity check

TARGET_DIST = 0.20
MAX_LINEAR  = 0.15
MAX_ANGULAR = 0.4


# ?????????????????????????????????????????
#  DEPTH LOOKUP
# ?????????????????????????????????????????
def get_depth(depth_frame, cx, cy, radius):
    depths = []
    r = max(3, radius // 3)
    for dy in range(-r, r):
        for dx in range(-r, r):
            px, py = cx + dx, cy + dy
            if 0 <= px < depth_frame.get_width() and 0 <= py < depth_frame.get_height():
                d = depth_frame.get_distance(px, py)
                if d > 0:
                    depths.append(d)
    return float(np.median(depths)) if depths else None


# ?????????????????????????????????????????
#  GRAYSCALE ROUNDNESS CHECK (the b&w experiment)
# ?????????????????????????????????????????
def confirm_circle_bnw(frame, cx, cy, radius):
    """
    Crop the candidate region, convert to grayscale, and run Hough Circle
    detection on it. Returns True only if a real circle is found ? this is
    what should reject skin blobs (irregular shape) while accepting the
    actual ball (round shape), regardless of color similarity.
    """
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
        minDist=gray.shape[0],           # only expect one circle in this crop
        param1=80,                        # Canny high threshold
        param2=25,                        # accumulator threshold ? lower = more lenient
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
#  VELOCITY CONTROL
# ?????????????????????????????????????????
def compute_velocity(ball_x, dist_m):
    if dist_m is None or dist_m <= TARGET_DIST:
        return 0.0, 0.0
    x_error = ball_x - FRAME_CENTER_X
    angular = max(-MAX_ANGULAR, min(MAX_ANGULAR, -0.003 * x_error))
    linear  = max(0.0, min(MAX_LINEAR, 0.3 * (dist_m - TARGET_DIST)))
    return linear, angular


def stop_robot(cmd_pub):
    cmd_pub.publish(Twist())


# ?????????????????????????????????????????
#  MAIN
# ?????????????????????????????????????????
def detect():
    rospy.init_node('pingpong_detector', anonymous=True)
    cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    use_realsense = False
    pipeline = None
    cap = None
    try:
        import pyrealsense2 as rs
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, FRAME_W, FRAME_H, rs.format.z16, 30)
        align = rs.align(rs.stream.color)
        pipeline.start(config)
        use_realsense = True
        print("RealSense D435 detected and started!")
    except Exception as e:
        print(f"RealSense not available ({e})")
        print("Falling back to webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: No camera found.")
            return

    print("Press Q to quit\n")

    try:
        while not rospy.is_shutdown():
            depth_frame = None
            if use_realsense:
                frames = pipeline.wait_for_frames()
                aligned = align.process(frames)
                color_f = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_f or not depth_frame:
                    continue
                frame = np.asanyarray(color_f.get_data())
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            h, w = frame.shape[:2]
            ball, mask = find_ball(frame)

            if ball is not None:
                cx, cy, radius = ball
                dist_m = get_depth(depth_frame, cx, cy, radius) if use_realsense else None

                linear, angular = compute_velocity(cx, dist_m)
                twist = Twist()
                twist.linear.x = linear
                twist.angular.z = angular
                cmd_pub.publish(twist)

                cv2.circle(frame, (cx, cy), radius, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                dist_str = f"  Dist:{dist_m:.2f}m" if dist_m is not None else "  Dist:N/A"
                cv2.putText(frame, f"Ball DETECTED{dist_str}", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                stop_robot(cmd_pub)
                cv2.putText(frame, "No ball", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            mode_str = "RealSense D435" if use_realsense else "Webcam"
            cv2.putText(frame, mode_str, (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            mask_small = cv2.resize(mask, (w // 5, h // 5))
            mask_color = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
            frame[h - h // 5 - 20:h - 20, 0:w // 5] = mask_color

            cv2.imshow("Ping Pong Detector", frame)
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
