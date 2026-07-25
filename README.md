# Ping Pong Ball Detector & Pickup

A ROS-based system for the Yahboom Transbot SE that detects an orange ping pong ball, drives toward it, centers it in frame, stops at the right spot, and picks it up with the arm.

## How it works

1. **Detection** - A two-stage pipeline. First, HSV color filtering and shape/circularity checks find candidate regions in the frame. Each candidate is then passed through a trained ensemble of three classifiers (SVM, Random Forest, and Gradient Boosting) trained on HOG features, which confirms whether the region is actually a ball rather than a similarly colored object or skin tone. A candidate only counts as a detection if the ensemble agrees with enough confidence.
2. **Navigation** - No depth camera needed. The robot uses the ball's pixel position in the camera frame: it turns to keep the ball horizontally centered, and drives forward until the ball reaches a calibrated target row, meaning it's close enough to grab.
3. **Reliability** - If the ball briefly disappears from a frame, the robot keeps moving based on its last known position for a short grace window instead of stopping instantly. A confirmation counter also prevents a single noisy frame from falsely triggering pickup.
4. **Pickup** - Once the target position is confirmed for several consecutive frames, detection pauses completely and the arm runs a fixed sequence of joint moves, one command at a time, to pick up the ball and drop it elsewhere.

## Machine learning

The classification stage is trained in `train.py` using a labeled ping pong ball dataset (Roboflow, YOLO format). HOG features are extracted from each labeled crop plus randomly sampled negative regions, then used to train three classifiers whose predictions are combined by averaging their confidence scores. The resulting model and scaler are saved and loaded at runtime in the detection script to confirm candidates found by the color/shape filter.

## Hardware

- Yahboom Transbot SE
- Raspberry Pi 4B, Ubuntu 18.04, ROS Melodic
- Intel RealSense D435 (color stream only)

## Running it

```bash
roslaunch transbot_bringup bringup.launch
```

Train the model first if the model files are not already present:

```bash
python3 train.py
```

Then run detection:

```bash
python3 detect.py
```

## Calibration notes

- `TARGET_X` / `TARGET_Y` in the config section need to be tuned to your camera's mount position. Place the ball where you want the robot to stop, read the printed coordinates on screen, and update the values.
- Arm joint angles in the sequence are tuned for this specific robot's arm and will need adjusting for a different physical setup.
- The ensemble's confidence threshold can be tuned depending on how strict or lenient you want detection to be.

## Status

Working end to end: detect, drive, center, stop, pick up, turn, drop, reset. Built under a tight deadline, so treat the joint angles, timing values, and confidence thresholds as a starting point, not a final calibration.
