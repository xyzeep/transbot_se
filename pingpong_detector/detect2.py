import cv2
import numpy as np
import time
import rospy
from geometry_msgs.msg import Twist

# ?????????????????????????????????????????
#  CONFIG
# ?????????????????????????????????????????
FRAME_W, FRAME_H = 640, 480
FRAME_CENTER_X   = FRAME_W // 2

# Orange HSV range ? wide on purpose, tune with the trackbars if needed
LOWER_ORANGE = np.array([10, 170, 100])
UPPER_ORANGE = np.array([22, 255, 255])

MIN_AREA        = 30      # minimum blob area in pixels, catches far/small balls
MIN_CIRCULARITY = 0.55    # lower = more lenient shape matching

TARGET_DIST = 0.4   # meters ? stop here
MAX_LINEAR    = 0.35
MAX_ANGULAR   = 0.5
SLOWDOWN_DIST = 0.15  # start braking this far before the target
MIN_LINEAR    = 0.04  # crawl speed near target, prevents overshoot from momentum


# ?????????????????????????????????????????
#  DEPTH LOOKUP
# ?????????????????????????????????????????

def get_depth(depth_frame, cx, cy, radius):
    depths = []
    r = max(5, min(radius, 15))  # stay inside the ball, don't spill into background
    for dy in range(-r, r):
        for dx in range(-r, r):
            px, py = cx + dx, cy + dy
            if 0 <= px < depth_frame.get_width() and 0 <= py < depth_frame.get_height():
                d = depth_frame.get_distance(px, py)
                if d > 0:
                    depths.append(d)

    if not depths:
        return None

    # reject outliers (background pixels that snuck into the window)
    depths = np.array(depths)
    med = np.median(depths)
    filtered = depths[np.abs(depths - med) < 0.05]  # keep only within 5cm of median
    return float(np.median(filtered)) if len(filtered) > 0 else float(med)


# ?????????????????????????????????????????
#  FIND BALL ? pure color + shape, no ML
# ?????????????????????????????????????????
def find_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = 0
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
        score = area * circularity
        if score > best_score:
            best_score = score
            best = (int(cx), int(cy), int(radius))

    return best, mask


# ?????????????????????????????????????????
#  VELOCITY CONTROL
# ?????????????????????????????????????????
def compute_velocity(ball_x, dist_m):
    if dist_m is None:
        return 0.0, 0.0

    x_error = ball_x - FRAME_CENTER_X
    angular = max(-MAX_ANGULAR, min(MAX_ANGULAR, -0.003 * x_error))

    if dist_m <= TARGET_DIST:
        linear = 0.0
    elif dist_m <= TARGET_DIST + SLOWDOWN_DIST:
        # crawl for the final stretch, so momentum doesn't carry it past the target
        linear = MIN_LINEAR
    else:
        linear = max(0.0, min(MAX_LINEAR, 0.6 * (dist_m - TARGET_DIST - SLOWDOWN_DIST)))

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
        hole_filling = rs.hole_filling_filter()
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


    last_dist = None
    last_dist_time = 0.0
    DIST_HOLD_SECONDS = 1.0
    smoothed_dist = None
    DIST_SMOOTHING = 0.7  # higher = smoother but slower to react



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
                depth_frame = hole_filling.process(depth_frame).as_depth_frame()

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


                now = time.time()
                if dist_m is not None:
                    last_dist = dist_m
                    last_dist_time = now
                elif last_dist is not None and (now - last_dist_time) < DIST_HOLD_SECONDS:
                    dist_m = last_dist
                # else: dist_m stays None, too stale to trust

                if dist_m is not None:
                    if smoothed_dist is None:
                        smoothed_dist = dist_m
                    else:
                        smoothed_dist = DIST_SMOOTHING * smoothed_dist + (1 - DIST_SMOOTHING) * dist_m
                    dist_m = smoothed_dist

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
                # no ball this frame -> stop immediately, no grace period
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
