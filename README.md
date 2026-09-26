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

The project uses the **UAV Flights Dataset**, published through the **University of Sheffield ORDA (Online Research Data)** repository.

The dataset is part of the **Swarm of UAVs Innovate UK Project, Future Flights Strand 3** and contains UAV flight data recorded during landing approaches in both real-world and simulated environments. The recordings include onboard-camera video from real UAV flights as well as simulation flights using **X-Plane 11**. :contentReference[oaicite:0]{index=0}

The dataset includes runway-approach footage and corresponding ground-truth annotations describing the two runway sidelines. The data was created to support research in vision-based navigation, runway detection, trajectory evaluation, and autonomous UAV landing. :contentReference[oaicite:1]{index=1}

For this project, selected real and simulated runway sequences from the dataset were used to prepare a mixed-domain dataset for YOLO Pose training and evaluation.

The final dataset contains:

- 1,400 training frames
- 802 validation frames
- 756 test frames

The selected sequences include:

- `GX010028Trim2.mp4` — real UAV flight
- `GX010035Trim1.mp4` — real UAV flight
- `runway_02.mp4` — simulated UAV flight
- `runway_video20230228-103640.mp4` — simulated UAV flight

The original runway side-line annotations were converted into four keypoints:

- Left-bottom runway point
- Left-top runway point
- Right-bottom runway point
- Right-top runway point

These keypoints are used to represent the visible runway geometry for the YOLO Pose model.

The complete UAV Flights Dataset is not included in this repository because of its size. The original dataset is available from the University of Sheffield ORDA repository under DOI **10.15131/shef.data.25712577.v1**. :contentReference[oaicite:2]{index=2}

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

The repository contains the code, experiments, results, and documentation required to reproduce and understand the final project.
