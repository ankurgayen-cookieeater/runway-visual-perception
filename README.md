# Runway Visual Perception for UAV Landing Using Onboard Camera Video

A computer vision system for detecting runway geometry from forward-facing UAV onboard-camera video.

## Overview

The project focuses on runway visual perception during UAV approach.

Given a forward-facing UAV video frame, the system detects four runway keypoints:

- Left-bottom runway point
- Left-top runway point
- Right-bottom runway point
- Right-top runway point

These keypoints are used to represent the visible runway boundaries and derive the runway centerline.

## Approach

The system follows this pipeline:

UAV onboard-camera video
→ YOLO Pose
→ Four runway keypoints
→ Runway boundary geometry
→ Runway centerline

## Dataset

The project uses runway footage containing both real and simulated UAV video.

A mixed-domain dataset was prepared for training and evaluation.

The complete datasets are not included in this repository because of their size.

## Model

The final model is a YOLO Pose model trained specifically for runway keypoint detection.

The model predicts:

- 1 runway object
- 4 keypoints
- 2 coordinates per keypoint

## Evaluation

The final model was evaluated on a completely held-out runway video.

Final test results:

| Metric | Result |
|---|---:|
| Box Precision | 1.000 |
| Box Recall | 1.000 |
| Box mAP50 | 0.995 |
| Box mAP50-95 | 0.985 |
| Pose Precision | 1.000 |
| Pose Recall | 1.000 |
| Pose mAP50 | 0.995 |
| Pose mAP50-95 | 0.995 |

The test video was not used during model training.

## Project Status

The final Train-7 model is frozen.

The repository contains the code and documentation required to reproduce and understand the final project.
