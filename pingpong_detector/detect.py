import cv2
import numpy as np
import os
import pickle
import rospy
from geometry_msgs.msg import Twist

# ?????????????????????????????????????????
#  CONFIG
# ?????????????????????????????????????????
MODEL_PATH = "models/ensemble_models.pkl"
SCALER_PATH = "models/ensemble_scaler.pkl"
IMG_SIZE   = (64, 64)
HOG_WIN    = (64, 64)
HOG_BLOCK  = (16, 16)
HOG_STRIDE = (8, 8)
HOG_CELL   = (8, 8)
HOG_BINS   = 9

# ?????????????????????????????????????????
#  HOG EXTRACTOR
# ?????????????????????????????????????????
hog = cv2.HOGDescriptor(HOG_WIN, HOG_BLOCK, HOG_STRIDE, HOG_CELL, HOG_BINS)

def extract_hog(image):
    image = cv2.resize(image, IMG_SIZE)
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return hog.compute(gray).flatten()

# ?????????????????????????????????????????
#  ENSEMBLE PREDICT ? SVM + RF + GB vote
# ?????????????????????????????????????????
def ensemble_predict(models, feat):
    svm_prob = models["svm"].predict_proba(feat)[0][1]
    rf_prob  = models["rf"].predict_proba(feat)[0][1]
    gb_prob  = models["gb"].predict_proba(feat)[0][1]
    avg_prob = (svm_prob + rf_prob + gb_prob) / 3.0
    votes    = sum([svm_prob > 0.5, rf_prob > 0.5, gb_prob > 0.5])
    return avg_prob, votes

# ?????????????????????????????????????????
#  GET DEPTH from RealSense frame
# ?????????????????????????????????????????
def get_depth(depth_frame, cx, cy, radius):
    depths = []
    r = max(3, radius // 3)
    for dy in range(-r, r):
        for dx in range(-r, r):
            px, py = cx + dx, cy + dy
            if (0 <= px < depth_frame.get_width() and
                    0 <= py < depth_frame.get_height()):
                d = depth_frame.get_distance(px, py)
                if d > 0:
                    depths.append(d)
    return float(np.median(depths)) if depths else None

FRAME_CENTER_X = 320
TARGET_DIST    = 0.20
STOP_DIST      = 0.20

def compute_velocity(ball_x, dist_m):
    linear  = 0.0
    angular = 0.0
    if dist_m is None:
        return linear, angular
    x_error = ball_x - FRAME_CENTER_X
    angular = -0.003 * x_error
    angular = max(-0.4, min(0.4, angular))
    if dist_m > TARGET_DIST:
        linear = min(0.15, 0.3 * (dist_m - STOP_DIST))
    else:
        linear = 0.0
    return linear, angular

# ?????????????????????????????????????????
#  MAIN DETECT
# ?????????????????????????????????????????
def detect():
    # ?? Load models ??
    if not os.path.exists(MODEL_PATH):
        print("No model found! Run:  python train.py  first.")
        return

    print("Loading ensemble models...")
    with open(MODEL_PATH, "rb") as f:
        models = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    rospy.init_node('pingpong_detector', anonymous=True)
    cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    # ?? Try RealSense D435 ??
    use_realsense = False
    pipeline      = None
    try:
        import pyrealsense2 as rs
        pipeline = rs.pipeline()
        config   = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
        align    = rs.align(rs.stream.color)
        pipeline.start(config)
        use_realsense = True
        print("RealSense D435 detected and started!")
    except Exception as e:
        print(f"RealSense not available ({e})")
        print("Falling back to webcam...")
        cap = cv2.VideoCapture(2)
        if not cap.isOpened():
            print("ERROR: No camera found.")
            return

    print("Press Q to quit\n")

    # ?? State ??
    last_x = last_y = last_r = None
    last_dist         = None
    frames_since_seen = 0
    MAX_LOST          = 90   # 3 seconds
    smooth_x = smooth_y = None
    SMOOTH   = 0.5

    # MeanShift
    track_window = None
    roi_hist     = None
    tracking     = False
    term_crit    = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    try:
        while True:
            # ?? Get frame ??
            depth_frame = None
            if use_realsense:
                frames      = pipeline.wait_for_frames()
                aligned     = align.process(frames)
                color_f     = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_f or not depth_frame:
                    continue
                frame = np.asanyarray(color_f.get_data())
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            h, w = frame.shape[:2]
            hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # ?? STEP 1: CV color filter ??
            lower_orange = np.array([8,  180, 120])
            upper_orange = np.array([25, 255, 255])
            mask = cv2.inRange(hsv, lower_orange, upper_orange)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
            mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            # ?? STEP 2: Find circular contours ??
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            candidates = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 40:
                    continue
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)

                min_circularity = 0.6 if area < 150 else 0.72
                if circularity < min_circularity:
                    continue

                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                cx, cy, radius   = int(cx), int(cy), int(radius)
                pad  = max(5, int(radius * 0.3))
                x1   = max(0, cx - radius - pad)
                y1   = max(0, cy - radius - pad)
                x2   = min(w, cx + radius + pad)
                y2   = min(h, cy + radius + pad)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                candidates.append((x1, y1, x2, y2, cx, cy,
                                   radius, crop, area * circularity))

            candidates.sort(key=lambda x: x[8], reverse=True)

            # ?? STEP 3: Ensemble ML confirms ??
            ball_found = False
            for (x1, y1, x2, y2, cx, cy, radius, crop, _) in candidates[:5]:
                feat = extract_hog(crop).reshape(1, -1)
                feat = scaler.transform(feat)
                avg_prob, votes = ensemble_predict(models, feat)

                if votes >= 2 and avg_prob > 0.75:
                    # ignore big jumps
                    if last_x is not None:
                        d = np.sqrt((cx-last_x)**2 + (cy-last_y)**2)
                        if d > 250:
                            continue

                    # smooth
                    if smooth_x is None:
                        smooth_x, smooth_y = float(cx), float(cy)
                    else:
                        smooth_x = SMOOTH*smooth_x + (1-SMOOTH)*cx
                        smooth_y = SMOOTH*smooth_y + (1-SMOOTH)*cy
                    sx, sy = int(smooth_x), int(smooth_y)

                    # get depth if RealSense available
                    dist_m = get_depth(depth_frame, sx, sy, radius) \
                             if use_realsense else None

                    last_x, last_y, last_r = sx, sy, radius
                    last_dist              = dist_m
                    frames_since_seen      = 0
                    ball_found             = True
		

                    last_x, last_y, last_r = sx, sy, radius
                    last_dist              = dist_m
                    frames_since_seen      = 0
                    ball_found             = True

                    linear, angular = compute_velocity(sx, dist_m)
                    twist = Twist()
                    twist.linear.x  = linear
                    twist.angular.z = angular
                    cmd_pub.publish(twist)



                    # init MeanShift
                    # init MeanShift
                    tw = max(10, radius*2+20)
                    th = max(10, radius*2+20)
                    tx = max(0, sx-radius-10)
                    ty = max(0, sy-radius-10)
                    tw = min(tw, w-tx)
                    th = min(th, h-ty)
                    if tw > 0 and th > 0:
                        roi      = hsv[ty:ty+th, tx:tx+tw]
                        roi_mask = mask[ty:ty+th, tx:tx+tw]
                        roi_hist = cv2.calcHist([roi],[0],roi_mask,
                                                [180],[0,180])
                        cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
                        track_window = (tx, ty, tw, th)
                        tracking     = True

                    # draw green box
                    bx1 = max(0, sx-radius-8)
                    by1 = max(0, sy-radius-8)
                    bx2 = min(w, sx+radius+8)
                    by2 = min(h, sy+radius+8)
                    cv2.rectangle(frame,(bx1,by1),(bx2,by2),(0,255,0),2)
                    cv2.circle(frame,(sx,sy),5,(0,255,0),-1)

                    size_label = ("close" if radius>80
                                  else "medium" if radius>40 else "far")
                    dist_str   = f"  Dist:{dist_m:.2f}m" if dist_m else ""
                    cv2.putText(frame,
                        f"Ping Pong Ball  {avg_prob*100:.0f}%{dist_str}  ({size_label})",
                        (bx1, max(15,by1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    cv2.putText(frame, "Ball DETECTED",
                        (10,35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,255,0),2)

                    # SVM/RF/GB individual votes display
                    cv2.putText(frame,
                        f"SVM+RF+GB votes: {votes}/3",
                        (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1)
                    break

            # ?? STEP 4: MeanShift fallback ??
            if not ball_found and tracking and roi_hist is not None:
                dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
                dst &= mask
                _, track_window = cv2.meanShift(dst, track_window, term_crit)
                tx, ty, tw, th  = track_window
                cx = tx + tw//2
                cy = ty + th//2

                region_mask  = mask[max(0,ty):min(h,ty+th),
                                    max(0,tx):min(w,tx+tw)]
                orange_ratio = np.sum(region_mask>0) / max(1, tw*th)

                if orange_ratio > 0.15:
                    dist_m = get_depth(depth_frame, cx, cy,
                                       max(tw,th)//2) \
                             if use_realsense else None
                    last_x, last_y, last_r = cx, cy, max(tw,th)//2
                    last_dist              = dist_m
                    frames_since_seen      = 0
                    ball_found             = True

                    last_x, last_y, last_r = cx, cy, max(tw,th)//2
                    last_dist              = dist_m
                    frames_since_seen      = 0
                    ball_found             = True

                    linear, angular = compute_velocity(cx, dist_m)
                    twist = Twist()
                    twist.linear.x  = linear
                    twist.angular.z = angular
                    cmd_pub.publish(twist)

                    dist_str = f"  Dist:{dist_m:.2f}m" if dist_m else ""
                    dist_str = f"  Dist:{dist_m:.2f}m" if dist_m else ""
                    cv2.rectangle(frame,(tx,ty),(tx+tw,ty+th),(255,150,0),2)
                    cv2.circle(frame,(cx,cy),5,(255,150,0),-1)
                    cv2.putText(frame, f"Tracking Ball{dist_str}",
                        (tx,max(15,ty-10)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,150,0),2)
                    cv2.putText(frame,"Ball TRACKING",
                        (10,35),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,150,0),2)
                else:
                    tracking = False
                    frames_since_seen += 1

            # ?? STEP 5: Last known position ??
            if not ball_found:
                frames_since_seen += 1
                if last_x is not None and frames_since_seen <= MAX_LOST:
                    cv2.rectangle(frame,
                        (max(0,last_x-last_r-8),max(0,last_y-last_r-8)),
                        (min(w,last_x+last_r+8),min(h,last_y+last_r+8)),
                        (0,100,255),2)
                    cv2.circle(frame,(last_x,last_y),5,(0,100,255),-1)
                    secs     = frames_since_seen//30
                    dist_str = f"  {last_dist:.2f}m" if last_dist else ""
                    cv2.putText(frame,
                        f"Last seen here{dist_str}  ({secs}s ago)",
                        (max(0,last_x-last_r),max(15,last_y-last_r-10)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,100,255),2)
                    cv2.putText(frame,"Ball OUT OF FRAME",
                        (10,35),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,100,255),2)

                else:
                    if frames_since_seen > MAX_LOST:
                        last_x=last_y=last_r=None
                        last_dist=None
                        smooth_x=smooth_y=None
                        tracking=False
                        roi_hist=None
                        twist = Twist()
                        twist.linear.x  = 0.0
                        twist.angular.z = 0.0
                        cmd_pub.publish(twist)
                    cv2.putText(frame,"No ball",
                        (10,35),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)
            # legend
            mode_str = "RealSense D435" if use_realsense else "Webcam"
            cv2.putText(frame,
                f"CV+SVM+RF+GB Ensemble  |  {mode_str}",
                (10,h-30),cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,200,200),1)
            cv2.putText(frame,
                "GREEN=detected  BLUE=tracking  ORANGE=last known",
                (10,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,200,200),1)

            # small mask preview
            mask_small = cv2.resize(mask,(w//5,h//5))
            mask_color = cv2.cvtColor(mask_small,cv2.COLOR_GRAY2BGR)
            frame[h-h//5-20:h-20, 0:w//5] = mask_color

            cv2.imshow("Ping Pong Detector ? CV + Ensemble + Depth", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if use_realsense and pipeline:
            pipeline.stop()
        elif not use_realsense:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detect()
