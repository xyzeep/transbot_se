# Autonomous Object Detection and Retrieval Using an Ensemble Machine Learning Pipeline on the Yahboom Transbot SE: A Low-Cost Robotic Solution for Nepal’s Emerging Industrial Automation

<p align="center">
  <img src="images/transbot3.png" alt="Yahboom Transbot SE platform overview" width="720"/>
</p>

<p align="center">
  <em>Yahboom Transbot SE — tracked mobile base, 3-DOF servo arm, and camera PTZ mount.</em>
</p>

---

## Abstract

This project presents a low-cost autonomous robotic system that **detects**, **approaches**, and **retrieves** an orange ping pong ball using the Yahboom Transbot SE. Perception combines classical computer vision with an ensemble machine-learning confirmation pipeline (SVM + Random Forest + Gradient Boosting on HOG features). Navigation and manipulation are implemented as a ROS Melodic intelligent agent: RGB sensing → perception → visual servoing on `/cmd_vel` → multi-frame confirmation → ordered arm control on `/TargetAngle`.

The work targets practical industrial-automation scenarios in resource-constrained settings (e.g. Nepal), where expensive commercial robots are inaccessible but Raspberry Pi–class platforms and open-source ROS stacks remain viable.

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [Team & Contributions](#2-team--contributions)
3. [Hardware Platform](#3-hardware-platform)
4. [Software Environment](#4-software-environment)
5. [System Architecture](#5-system-architecture)
6. [Intelligent Agent Design](#6-intelligent-agent-design)
7. [Machine Learning Pipeline](#7-machine-learning-pipeline)
8. [Computer Vision Detection](#8-computer-vision-detection)
9. [Depth Sensing Investigation](#9-depth-sensing-investigation)
10. [Navigation & Visual Servoing](#10-navigation--visual-servoing)
11. [Arm Pickup Sequence](#11-arm-pickup-sequence)
12. [ROS Topics & Integration](#12-ros-topics--integration)
13. [Simulation Attempt](#13-simulation-attempt)
14. [Repository Structure](#14-repository-structure)
15. [How to Run](#15-how-to-run)
16. [Calibration](#16-calibration)
17. [Results & Known Limitations](#17-results--known-limitations)
18. [Future Work](#18-future-work)

---

## 1. Project Goal

Detect an orange ping pong ball with the onboard camera, drive the robot toward it while keeping the ball centered in the image, stop at a calibrated grasp pose, then execute a fixed servo sequence to **pick up**, **turn**, **drop**, and **reset**.

| Stage | Behavior |
|-------|----------|
| Sense | RGB frames from Intel RealSense D435 |
| Perceive | HSV + circularity (+ optional ML ensemble / Hough confirm) |
| Act (drive) | Publish `geometry_msgs/Twist` to `/cmd_vel` |
| Decide | Multi-frame “target reached” confirmation |
| Act (arm) | Publish `transbot_msgs/Arm` to `/TargetAngle`; pause detection |

---

## 2. Team & Contributions

| Member | Ownership | Documented focus |
|--------|-----------|------------------|
| **Pawan** | ROS software environment, integration, control logic | Bringup, Python/ROS bridging, state machine (detect ↔ drive ↔ arm), `/cmd_vel` + `/TargetAngle` wiring |
| **Pratik** | Arm movement & navigation | Joint sequences, visual-servoing approach/stop, turn–drop–reset maneuvers |
| **Dikshant** | Machine learning + ping-pong CV | HOG + SVM/RF/GB ensemble, Roboflow dataset, HSV/Hough detection used in production |
| **Sneha** | Depth sensor (RealSense) | librealsense build, depth sampling/smoothing, why depth was dropped for stopping |
| **Simon** | Simulation | Gazebo / MoveIt exploration and decision to stay on physical hardware |
| **Nishan** | Hardware setup | Mechanical mount, cabling, joint ranges, camera servo defaults, power |

---

## 3. Hardware Platform

<p align="center">
  <img src="images/transbot1.png" alt="Transbot SE top-down hardware view" width="480"/>
  &nbsp;
  <img src="images/transbot2.png" alt="Transbot SE front view with RealSense and gripper" width="480"/>
</p>

<p align="center">
  <em>Left: top-down view (tracks, RealSense mount, 3-DOF arm). Right: front view showing gripper reach and elevated D435.</em>
</p>

### 3.1 Robot

| Item | Specification |
|------|----------------|
| Platform | **Yahboom Transbot SE** |
| Drive | Tracked differential drive (520 motors with encoders) |
| Chassis | Aluminum alloy |
| Battery | 12 V lithium, 4400 mAh |
| Compute | **Raspberry Pi 4B** on multi-function expansion board |
| Arm | **3-DOF bus-servo arm** |
| Camera mount | 2-DOF PTZ; default PWM servo angle **120°** |

### 3.2 Arm joints

| Joint ID | Role | Valid range |
|----------|------|-------------|
| **7** | Shoulder | 0° – 225° |
| **8** | Elbow | 200° – 270° |
| **9** | Gripper | 30° – 180° |

### 3.3 Camera

| Item | Detail |
|------|--------|
| Sensor | **Intel RealSense D435** |
| Interface | USB 3.0 |
| Streams used (final) | Color (BGR, 640×480 @ 30 FPS) |
| Streams explored | Aligned depth (abandoned for stop logic; see §9) |

### 3.4 Development machines

| Machine | Role |
|---------|------|
| Raspberry Pi 4B | Runtime: ROS, detection, motor/arm control |
| MacBook (Apple Silicon) | Model training attempts; hit build failures on old numpy/scipy (pre–Apple Silicon wheels) |

Access: **RealVNC** for remote desktop; **ssh / scp** for file transfer between Pi and Mac.

---

## 4. Software Environment

| Layer | Version / note |
|-------|----------------|
| OS | Ubuntu **18.04** (64-bit) |
| Middleware | **ROS 1 Melodic** |
| System Python | **3.6.9** (required by Melodic + Transbot_Lib; left untouched) |
| Detection Python | **3.9.18** via **pyenv**, scoped to the detection project folder |
| RealSense SDK | **librealsense v2.50.0** built from source (Python 3.6/3.9 bindings; newer SDK required Python ≥ 3.7) |
| Vision stack | OpenCV, NumPy (venv on 3.9 for compatibility) |

### Why two Pythons?

ROS Melodic and `Transbot_Lib` (egg under Python 3.6) must stay on system Python. Trained models and modern NumPy/OpenCV needed Python ≥ 3.9. Solution: isolate detection under pyenv 3.9 without breaking ROS.

### Key library paths

- Transbot library (runtime): `/usr/local/lib/python3.6/dist-packages/Transbot_Lib-…`
- Editable source (reinstall after changes): `~/py_install-V3.2.5/py_install/Transbot_Lib/` then `sudo python3 setup.py install`
- Joint 8 elbow angle formula was fixed in `Transbot_Lib.py`
- Camera servo default: `self.bot.set_pwm_servo(2, 120)` in `transbot_driver.py`

### Main bringup

```bash
roslaunch transbot_bringup bringup.launch
```

---

## 5. System Architecture

<p align="center">
  <img src="images/architecture.png" alt="System architecture: sense → perceive → control → decide → arm" width="900"/>
</p>

<p align="center">
  <em>End-to-end architecture: sensing, perception, visual servoing, state decision, and arm sequence on ROS Melodic.</em>
</p>

### Pipeline (five stages)

```
┌─────────────┐    RGB     ┌──────────────────┐   (cx,cy)   ┌─────────────────┐
│ 1. Sensing  │ ─────────► │ 2. Perception    │ ──────────► │ 3. Visual       │
│ RealSense   │            │ HSV + shape      │             │ Servoing        │
│ D435 color  │            │ (+ ML / Hough)   │             │ /cmd_vel        │
└─────────────┘            │ last-seen grace  │             └────────┬────────┘
                           │ multi-frame OK   │                      │
                           └──────────────────┘                      ▼
                                                    ┌────────────────────────┐
                                                    │ 4. Target reached?     │
                                                    │ false → keep tracking  │
                                                    │ true  → pause detect   │
                                                    └────────────┬───────────┘
                                                                 ▼
                                                    ┌────────────────────────┐
                                                    │ 5. Arm sequence        │
                                                    │ /TargetAngle           │
                                                    │ lower → grip → lift →  │
                                                    │ turn → drop → reset    │
                                                    └────────────────────────┘
```

### Abandoned approaches (documented for honesty)

| Approach | Why abandoned |
|----------|----------------|
| ML ensemble as **required** runtime gate | Cross-machine pickle / NumPy / sklearn version hell |
| Depth-based stop distance | Sparse/noisy IR returns on small glossy balls |
| Gazebo full-system sim | URDF lacked wheeled-base drive; Pi too weak for Gazebo GUI |

---

## 6. Intelligent Agent Design

Framed as a **hybrid intelligent agent** (model-based reflex + finite-state control), not a pure open-loop script.

### 6.1 Agent type

| Property | Design choice |
|----------|----------------|
| Sensors | RGB camera (primary); depth explored then dropped for stop logic |
| Effectors | Differential tracks (`/cmd_vel`); 3-DOF arm (`/TargetAngle`) |
| World model | Ball pixel `(cx, cy)`, last-seen timestamp, reached-frame counter, `arm_active` flag |
| Control law | Proportional visual servoing on pixel error (hill-climb toward target pixel) |
| Deliberation | Ordered pickup sequence = fixed task plan (STRIPS-like: approach → center → stop → grasp → relocate → reset) |

### 6.2 State machine

```
IDLE / SEARCH
    │ ball detected
    ▼
TRACK & SERVO  ◄──── last-seen grace (≤ ~0.5–0.8 s)
    │ cy ≈ TARGET_Y for N consecutive frames
    ▼
CONFIRM REACHED
    │ stop base; set arm_active = true
    ▼
ARM SEQUENCE (detection fully paused)
    │
    ▼
DONE (one-shot; no auto re-hunt)
```

### 6.3 Knowledge / predicates (informal KR)

Useful facts the controller reasons over:

- `seen(ball, cx, cy, t)`
- `stale(ball)` if `now − t > LAST_SEEN_TIMEOUT`
- `centered_x` if `|cx − TARGET_X| ≤ TOLERANCE_X`
- `in_grasp_zone` if `|cy − TARGET_Y| ≤ TOLERANCE_Y`
- `confirmed` if `reached_count ≥ REACHED_CONFIRM_FRAMES`
- `arm_active` ⇒ do not call `find_ball`

Inference example: `confirmed ∧ ¬pickup_done ⇒ stop ∧ arm_active ∧ run_pickup_sequence`.

---

## 7. Machine Learning Pipeline

*(Primary owner: Dikshant; training support on secondary Mac)*

### 7.1 Motivation

Color alone confuses orange skin and background clutter. An **ensemble** was trained to answer: *“Is this crop a ping pong ball?”*

### 7.2 Dataset

- Source: [Roboflow — Ping Pong Detection](https://universe.roboflow.com/pingpong-ojuhj/ping-pong-detection-0guzq1) (YOLO format)
- Positives: labeled ball crops  
- Negatives: random non-ball regions from the same images  

### 7.3 Features & models

| Step | Detail |
|------|--------|
| Preprocess | Resize crop to **64×64** |
| Features | **HOG** (win 64, block 16, stride 8, cell 8, 9 bins) |
| Classifiers | **SVM**, **Random Forest**, **Gradient Boosting** |
| Decision | Average probability + vote count (e.g. ≥ 2/3 votes, confidence threshold) |
| Artifacts | `models/ensemble_models.pkl`, `models/ensemble_scaler.pkl` |

### 7.4 Intended folder layout (training)

```
pingpong_detector/
├── train.py
├── detect.py          # CV + ensemble + optional depth
├── dataset/
│   ├── train/{images,labels}
│   └── valid/{images,labels}
└── models/
    ├── ensemble_models.pkl
    └── ensemble_scaler.pkl
```

### 7.5 Cross-platform friction (important lesson)

| Issue | Cause | Mitigation explored |
|-------|--------|---------------------|
| `numpy._core` pickle errors | Mac NumPy 2.x vs Pi NumPy 1.x | Match versions / pyenv 3.9 on Pi |
| sklearn install failures on Pi 3.6 | sklearn 1.x needs Python ≥ 3.9 | pyenv 3.9 + matching Mac versions |
| Apple Silicon build of old NumPy/SciPy | No wheels; `faltivec` compile failures | Windows retrain or drop ML at runtime |
| OpenCV 5 vs HOG | API / NumPy ABI mismatch | Pin OpenCV 4.x compatible with chosen NumPy |

### 7.6 Final production choice

Runtime reliability favored **pure CV** (HSV + circularity + grayscale Hough confirm) in `coor.py` / `coor2.py`. The ensemble remains part of the **ML research / agent-perception story** and lives in `detect.py` for comparison and documentation—not as a hard dependency of the demo path.

---

## 8. Computer Vision Detection

*(Final demo path — Dikshant / Pawan)*

### Pipeline

1. Convert frame BGR → **HSV**
2. Threshold orange range (tuned from color picker: ball `#dd7818` vs skin tones)  
   - Example: `LOWER_ORANGE = [10, 170, 100]`, `UPPER_ORANGE = [22, 255, 255]`  
   - High **saturation** floor rejects skin
3. Morphology open/close (small elliptical kernel)
4. Contours: `area ≥ MIN_AREA`, `circularity ≥ MIN_CIRCULARITY`
5. Second pass: **HoughCircles** on grayscale crop to reject non-round orange blobs
6. Best candidate → `(cx, cy, radius)`

### Reliability features

| Mechanism | Purpose |
|-----------|---------|
| **Last-seen grace** (~0.5–0.8 s) | Keep servoing on last `(cx, cy)` through brief misses |
| **Reached debounce** (many consecutive frames) | Avoid false arm triggers from one noisy frame |
| **Mutual exclusion** | When `arm_active`, detection loop is skipped entirely |

---

## 9. Depth Sensing Investigation

*(Primary owner: Sneha)*

### What was tried

- Enable RealSense **depth** + align to color  
- Sample a window around ball center; median + outlier reject + temporal smooth / hold  
- Stop when `dist_m ≤ TARGET_DIST` (e.g. 0.2–0.3 m)

### Why it failed for ping pong balls

Small, glossy, curved surfaces return **sparse IR**, so depth flickers (`N/A` ↔ 0.4 m ↔ 0.5 m). That caused jerky `/cmd_vel` (drive–stop–drive). Hole-filling helped somewhat but not enough for precise grasp stopping.

### Outcome

Final controller uses **pixel-row visual servoing** (`TARGET_Y`) instead of meters. Depth work remains valuable documentation of sensor limits on this object class.

---

## 10. Navigation & Visual Servoing

*(Pratik / Pawan)*

Camera looks slightly downward. Far ball → smaller **y**; near ball → larger **y**.

| Signal | Control |
|--------|---------|
| `x_error = cx − TARGET_X` | Angular velocity (center horizontally) |
| `y_error = TARGET_Y − cy` | Linear velocity (approach until grasp row) |
| Dead zones `TOLERANCE_X/Y` | Prevent chatter |

```text
Publish Twist { linear.x, angular.z } → /cmd_vel → /transbot_node (PWM / tracks)
```

Default demo tuning lives in `pingpong_detector/coor.py` (`MAX_LINEAR`, `MAX_ANGULAR`, `TARGET_X/Y`).

---

## 11. Arm Pickup Sequence

*(Primary owner: Pratik)*

Messages use **`transbot_msgs/Arm`** containing a list of **`Joint`** `{id, angle, run_time}` so multiple joints can move in one publish.

### Example default / grasp angles (tune on hardware)

| Phase | Joints (id : angle) |
|-------|---------------------|
| Default | 9:30, 8:200, 7:60 |
| Pickup (ordered) | 8:250 → 7:0 → 9:120 → 7:50 |
| Drop | 8:200 → 9:30 |
| Reset | back to default |
| Relocate | in-place ~180° turn via `/cmd_vel`, then reverse turn |

**Critical integration rule:** once the target is confirmed, **stop the base**, set `arm_active`, run the sequence; **do not** keep detecting in parallel.

Manual experiment (bringup running, detector stopped):

```bash
rostopic pub /TargetAngle transbot_msgs/Arm \
  "{joint: [{id: 9, angle: 90, run_time: 1000}]}" -1
```

---

## 12. ROS Topics & Integration

| Topic | Type | Role |
|-------|------|------|
| `/cmd_vel` | `geometry_msgs/Twist` | Base motion |
| `/TargetAngle` | `transbot_msgs/Arm` | Arm joints |
| `/PWMServo` | (servo msgs) | Camera pan/tilt |
| `/odom` | odometry | Wheel feedback (available; not required by final stop logic) |
| `/camera/color/image_raw` | image | RealSense RGB (when using ROS camera stack) |
| `/camera/aligned_depth_to_color/image_raw` | image | Aligned depth (investigated) |

### Runtime graph (conceptual)

```
pingpong_coord_detector ──pub──► /cmd_vel ──sub──► transbot_node ──► tracks
         │
         └──pub──► /TargetAngle ──sub──► arm driver ──► joints 7/8/9
```

**Note:** Only one publisher should command `/cmd_vel` during demos (disable joystick `/transbot_joy` if it fights the detector).

---

## 13. Simulation Attempt

*(Primary owner: Simon)*

Explored `transbot_se_moveit_config/demo_gazebo.launch` (Gazebo 9 + MoveIt already on the Pi). Findings:

- MoveIt demo is **arm-centric**; URDF effectively lacked a usable wheeled `/cmd_vel` plugin path for this task  
- Pi 4B struggled with Gazebo GUI FPS  
- Risk of commanding the **physical** `/transbot_node` if bringup and sim share a master  

**Decision:** abandon full sim for the deadline; continue physical detect → drive → pick.

---

## 14. Repository Structure

```
transbot_se/
├── README.md
├── images/
│   ├── architecture.png      # system architecture diagram
│   ├── transbot1.png         # top-down hardware photo
│   ├── transbot2.png         # front view (camera + gripper)
│   └── transbot3.png         # annotated platform overview
├── pingpong_detector/        # project application code
│   ├── detect.py             # CV + ML ensemble (+ depth experiments)
│   ├── detect2.py            # CV / depth iteration
│   ├── coor.py / coor2.py    # final pixel-target servoing + arm hook
│   └── new_detect.py         # additional detection variants
├── transbot_bringup/         # robot bringup, drivers, navigation params
├── transbot_msgs/            # Arm, Joint, and other custom messages
├── transbot_se_moveit_config/# MoveIt + Gazebo demo (sim exploration)
├── transbot_ctrl/            # joystick / keyboard teleop
└── …                         # stock Yahboom packages (vision, track, etc.)
```

---

## 15. How to Run

### 15.1 Hardware bringup

```bash
# On Transbot Wi-Fi (ROS_MASTER_URI must match Pi IP)
roslaunch transbot_bringup bringup.launch
```

### 15.2 Detection environment (Pi)

```bash
cd ~/transbot_ws/src/pingpong_detector   # or this repo's pingpong_detector
# activate pyenv 3.9 + venv as configured on the robot, e.g.:
detectenv   # alias: venv + PYTHONPATH to pyrealsense2 for 3.9
python3 coor.py
```

### 15.3 Optional: ML ensemble path

```bash
# After models/ is populated and deps match training versions:
python3 detect.py
```

### 15.4 Quit

Press **Q** in the OpenCV window; the node publishes zero velocity on shutdown.

---

## 16. Calibration

| Parameter | How to set |
|-----------|------------|
| `TARGET_X`, `TARGET_Y` | Place ball at ideal grasp pose; read on-screen `(x, y)`; write into config |
| HSV orange bounds | Color-pick ball vs skin; raise lower **saturation** to reject skin |
| Arm angles / `run_time` | `rostopic pub /TargetAngle …` until grasp is reliable |
| `LAST_SEEN_TIMEOUT` | Longer → smoother through flicker; shorter → safer stop |
| `REACHED_CONFIRM_FRAMES` | Higher → fewer false pickups; slower trigger |

---

## 17. Results & Known Limitations

### Achieved

- Reliable orange ball detection at near and mid range with CV path  
- Visual servoing approach + horizontal centering without depth  
- Debounced stop and mutually exclusive arm takeover  
- End-to-end physical demo path under ROS Melodic on Pi 4B  

### Limitations

- Arm angles and `TARGET_Y` remain setup-specific  
- One-shot: no automatic multi-ball reset loop  
- ML ensemble not required at runtime due to environment/pickle fragility  
- Depth unsuitable as sole stop metric for this object  
- Simulation not production-ready on this Pi + URDF  

---

## 18. Future Work

1. Calibrate grasp geometry (hand–eye / empirical lookup)  
2. Multi-ball loop: reset `arm_active` and resume detection  
3. Containerize matched NumPy/sklearn/OpenCV for portable ensemble inference  
4. Odometry-closed-loop 180° turns instead of open-loop timing  
5. Lightweight headless Gazebo or remote sim host for CI-style tests  

---

## Citation / Context

**Platform:** Yahboom Transbot SE · Raspberry Pi 4B · Ubuntu 18.04 · ROS Melodic · Intel RealSense D435  

**Theme:** Combining **ensemble machine learning** (perception research track) with **ROS-based intelligent agent control** (sensing, visual servoing, finite-state retrieval) as a **low-cost automation** case study relevant to emerging industrial needs in Nepal.

---

## License & Acknowledgments

Built on Yahboom Transbot SE ROS packages and Intel RealSense librealsense. Team: Pawan, Pratik, Dikshant, Sneha, Simon, Nishan.
