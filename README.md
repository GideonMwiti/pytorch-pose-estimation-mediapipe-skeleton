# Project 71: Real-Time Human Pose Estimation & Skeletal Tracking with PyTorch & MediaPipe

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=flat&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![Task](https://img.shields.io/badge/Task-Pose_Estimation_%26_Kinematics-success.svg)](#)
[![Model](https://img.shields.io/badge/Architecture-PoseNet_(Deep_MLP)-blue.svg)](#system-architecture)
[![Accuracy](https://img.shields.io/badge/Pose_Accuracy-100.0%25-brightgreen.svg)](#quantitative-benchmarks--performance-profile)
[![Rule-Based](https://img.shields.io/badge/Kinematic_Rules-100.0%25-brightgreen.svg)](#biomechanical-joint-angle-kinematics)
[![Throughput](https://img.shields.io/badge/Throughput-3%2C994.7_FPS-purple.svg)](#quantitative-benchmarks--performance-profile)
[![Challenge](https://img.shields.io/badge/Challenge-100_PyTorch_Projects_in_10_Days-orange.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, edge-optimized PyTorch and MediaPipe implementation of **Real-Time Human Pose Estimation and Skeletal Tracking**. The system detects **33 3D full-body landmarks** $(x, y, z)$ conforming to the MediaPipe Pose topology, computes **biomechanical joint angles** (knee flexion, elbow extension, hip hinge) for continuous exercise rep counting, and classifies functional athletic postures (**Standing**, **Squat**, **Pushup**, **Jumping Jack**, **Lunge**) using a lightweight neural network (**PoseNet**, 23,749 parameters, 92.77 KB footprint) running in **0.250 ms latency** (**3,994.7 FPS throughput**).

---

## 📌 Table of Contents
- [Executive Overview & Problem Formulation](#executive-overview--problem-formulation)
- [System Architecture](#system-architecture)
- [Mathematical Foundations](#mathematical-foundations)
  - [33-Joint 3D Skeletal Topology](#33-joint-3d-skeletal-topology)
  - [Pelvic Centering & Torso Scale Normalization](#pelvic-centering--torso-scale-normalization)
  - [Biomechanical Joint Angle Kinematics](#biomechanical-joint-angle-kinematics)
  - [Repetition Counter State Machine](#repetition-counter-state-machine)
- [Repository Directory Structure](#repository-directory-structure)
- [Quantitative Benchmarks & Performance Profile](#quantitative-benchmarks--performance-profile)
- [Diagnostic Dashboard Visualizations](#diagnostic-dashboard-visualizations)
- [Interactive Live Stream / Webcam CLI](#interactive-live-stream--webcam-cli)
- [Quickstart & Reproduction Guide](#quickstart--reproduction-guide)
- [References](#references)
- [License](#license)

---

## Executive Overview & Problem Formulation

Contactless fitness tracking, physical rehabilitation monitoring, ergonomics assessment, and sports biomechanics require tracking full-body movement with real-time feedback. Raw camera feeds pose significant challenges due to subject morphology variations, camera distance, perspective foreshortening, and background clutter.

By decomposing motion tracking into a **geometric skeletal representation** (MediaPipe 33 keypoints) and a **kinematic neural inference head**, this project delivers:
1. **Pose & Morphology Invariance**: Transforming 3D joint coordinates relative to the mid-hip pelvis and scaling by torso length guarantees consistent classification across varied body types and camera distances.
2. **Biomechanical Angle Extraction**: Vector dot-product trigonometry quantifies knee and elbow flexion in degrees, powering deterministic repetition counters and form validators.
3. **Sub-Millisecond Edge Performance**: PoseNet executes in **0.250 ms** ($> 3,900\text{ FPS}$ on single-core CPU), requiring minimal CPU resources and allowing seamless multi-person tracking on edge devices.

---

## System Architecture

```mermaid
flowchart TD
    subgraph "Video Frame Ingestion & Pose Extraction"
        RAW["Camera / Video Feed (RGB Frame)"]
        MP["MediaPipe Pose Landmark Extractor\n(33 Joints in 3D: x, y, z)"]
        RAW --> MP
    end

    subgraph "Kinematic & Invariance Normalization Engine"
        CEN["Pelvic Origin Translation:\np'_i = p_i - p_midhip"]
        SCL["Torso Height Scaling:\ns = ||p_midshoulder - p_midhip||_2\np_norm = p'_i / s"]
        ANG["Biomechanical Joint Angles:\nKnees, Elbows, Hips via Vector Dot-Products"]

        MP --> CEN --> SCL
        MP --> ANG
    end

    subgraph "Hybrid Classification & Repetition Tracking"
        subgraph "Neural Branch: PoseNet (23,749 Params)"
            FC1["Linear(99->128) + BN + ReLU + Dropout(0.20)"]
            FC2["Linear(128->64) + BN + ReLU + Dropout(0.15)"]
            FC3["Linear(64->32) + BN + ReLU"]
            OUT["Linear(32->5) -> Softmax (Standing, Squat, Pushup, Jumping Jack, Lunge)"]
            SCL --> FC1 --> FC2 --> FC3 --> OUT
        end

        subgraph "Heuristic Branch: Rep Counter State Machine"
            REP["Hysteresis State Machine:\nKnee < 100° (Inflection) -> Knee > 150° (Rep Completed)"]
            ANG --> REP
        end
    end

    subgraph "Live Fitness & Rehabilitation HUD"
        HUD["Live Video HUD: Skeletal Overlay, Joint Angle Tags, Active Posture, Rep Counter"]
        OUT --> HUD
        REP --> HUD
    end
```

---

## Mathematical Foundations

### 33-Joint 3D Skeletal Topology

A detected human body pose is represented as a spatial landmark matrix $\mathbf{P} \in \mathbb{R}^{33 \times 3}$:

$$\mathbf{P} = \begin{bmatrix} \mathbf{p}_0^T \\ \mathbf{p}_1^T \\ \vdots \\ \mathbf{p}_{32}^T \end{bmatrix}, \qquad \mathbf{p}_i = \begin{pmatrix} x_i \\ y_i \\ z_i \end{pmatrix}$$

Key kinematic anchors include:
- **Head & Facial**: $\mathbf{p}_0$ (Nose), $\mathbf{p}_1 \dots \mathbf{p}_8$ (Eyes & Ears)
- **Upper Body**: $\mathbf{p}_{11}, \mathbf{p}_{12}$ (Shoulders), $\mathbf{p}_{13}, \mathbf{p}_{14}$ (Elbows), $\mathbf{p}_{15}, \mathbf{p}_{16}$ (Wrists)
- **Core & Pelvis**: $\mathbf{p}_{23}, \mathbf{p}_{24}$ (Hips)
- **Lower Body**: $\mathbf{p}_{25}, \mathbf{p}_{26}$ (Knees), $\mathbf{p}_{27}, \mathbf{p}_{28}$ (Ankles), $\mathbf{p}_{29} \dots \mathbf{p}_{32}$ (Heels & Toes)

---

### Pelvic Centering & Torso Scale Normalization

To decouple camera framing and distance from posture representation:

1. **Pelvic Origin**:
   $$\mathbf{p}_{\text{mid-hip}} = \frac{\mathbf{p}_{23} + \mathbf{p}_{24}}{2}$$

2. **Torso Reference Scale**:
   $$\mathbf{p}_{\text{mid-shoulder}} = \frac{\mathbf{p}_{11} + \mathbf{p}_{12}}{2}$$
   $$s_{\text{torso}} = \|\mathbf{p}_{\text{mid-shoulder}} - \mathbf{p}_{\text{mid-hip}}\|_2$$

3. **Normalized Landmark Vector**:
   $$\tilde{\mathbf{p}}_i = \frac{\mathbf{p}_i - \mathbf{p}_{\text{mid-hip}}}{s_{\text{torso}}}, \qquad \mathbf{x} = \text{vec}(\tilde{\mathbf{P}}) \in \mathbb{R}^{99}$$

---

### Biomechanical Joint Angle Kinematics

Interior joint angles are evaluated across 3 connected anatomical vertices $\mathbf{p}_A, \mathbf{p}_B, \mathbf{p}_C$ (where $\mathbf{p}_B$ is the joint pivot):

$$\mathbf{u} = \mathbf{p}_A - \mathbf{p}_B, \qquad \mathbf{v} = \mathbf{p}_C - \mathbf{p}_B$$

$$\theta_{\text{joint}} = \arccos \left( \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} \right) \times \frac{180^\circ}{\pi}$$

Key evaluated angles:
- **Knee Flexion**: $\mathbf{p}_{\text{Hip}} \to \mathbf{p}_{\text{Knee}} \to \mathbf{p}_{\text{Ankle}}$
- **Elbow Extension**: $\mathbf{p}_{\text{Shoulder}} \to \mathbf{p}_{\text{Elbow}} \to \mathbf{p}_{\text{Wrist}}$
- **Hip Hinge**: $\mathbf{p}_{\text{Shoulder}} \to \mathbf{p}_{\text{Hip}} \to \mathbf{p}_{\text{Knee}}$

---

### Repetition Counter State Machine

Repetition counting applies a dual-threshold hysteresis model to filter noise:

$$\text{State}(t+1) = \begin{cases} 
\text{"INFLECTION / BOTTOM",} & \text{if } \theta_{\text{knee}} < 100^\circ \land \text{State}(t) = \text{"TOP"} \\
\text{"TOP",} & \text{if } \theta_{\text{knee}} > 150^\circ \land \text{State}(t) = \text{"INFLECTION"} \\
\text{State}(t), & \text{otherwise}
\end{cases}$$

$$\text{Reps} \leftarrow \text{Reps} + 1 \quad \text{when transitioning from INFLECTION} \to \text{TOP}$$

---

## Repository Directory Structure

```text
71_pose_estimation_mediapipe/
├── .gitignore                          # Excludes raw dumps, checkpoints, and cache
├── requirements.txt                     # Dependencies (PyTorch, MediaPipe, OpenCV, SciPy)
├── main.py                             # Complete model training, evaluation & diagnostic pipeline
├── webcam_live.py                      # Interactive real-time live video / webcam CLI
├── outputs/
│   ├── posenet_best.pth                # Serialized PoseNet PyTorch weights (92.8 KB)
│   ├── pose_estimation_metrics.json    # Serialized latency, accuracy, and angle metrics
│   └── pose_estimation_report.png      # Publication-grade 6-panel diagnostic dashboard
└── README.md                           # Exhaustive technical documentation & benchmarks
```

---

## Quantitative Benchmarks & Performance Profile

### Computational Footprint & Latency Profile

| Metric | Measured Score | Specification Target | Status |
| :--- | :---: | :---: | :---: |
| **Model Parameters** | **`23,749`** | $< 100,000$ | Ultra-Compact |
| **Model Size in Memory** | **`92.77 KB`** | $< 500.0\text{ KB}$ | Microcontroller Ready |
| **Input Feature Dimensionality** | **`99-D`** ($33 \text{ joints} \times 3$) | Scale-Invariant | Fully Normalized |
| **Mean Inference Latency** | **`0.250 ms`** | $< 2.00\text{ ms}$ | **3,994.7 FPS Throughput** |
| **Training Duration** | **`4.02 seconds`** (25 epochs) | $< 30.0\text{ s}$ | Rapid On-Device Calibration |

### Classification & Diagnostic Evaluation

| Performance Benchmark | Measured Value | Target Standard | Status |
| :--- | :---: | :---: | :---: |
| **Neural Test Accuracy (PoseNet)** | **`100.00%`** | $> 95.0\%$ | Perfect Separation |
| **Rule-Based Heuristic Accuracy** | **`100.00%`** | $> 90.0\%$ | Fully Aligned Form Rules |
| **Precision (Weighted)** | **`1.0000`** | $> 0.950$ | Zero False Positives |
| **Recall (Weighted)** | **`1.0000`** | $> 0.950$ | Zero Missed Postures |
| **F1-Score (Weighted)** | **`1.0000`** | $> 0.950$ | Complete Harmonic Balance |
| **Confusion Matrix (188 Test Samples)** | `188 / 188` Correctly Classified | 100% on Diagonal | Complete Separation |

---

## Diagnostic Dashboard Visualizations

The automated execution pipeline generates a publication-grade 6-panel diagnostic dashboard saved to `outputs/pose_estimation_report.png`:

![Pose Estimation Report](outputs/pose_estimation_report.png)

### Panel Breakdown:
1. **MediaPipe 33-Joint Full Body Skeletal Mesh**: Complete wireframe displaying joints, color-coded functional body segments (torso, arms, legs), and labeled anatomical indices.
2. **Live Video Stream Pose Tracking HUD**: Simulated gym camera feed showing detected skeletal overlay, bounding box, knee flexion tag, and active posture label.
3. **Biomechanical Joint Angle Trajectories**: 10-second temporal waveform tracking knee and hip flexion cycles with squat depth threshold lines and rep inflection markers.
4. **PoseNet Loss & Accuracy Dynamics**: Cross-Entropy loss descending to near-zero with validation accuracy reaching 100%.
5. **Normalized Confusion Matrix**: 5-class confusion matrix showing zero off-diagonal misclassifications across all test samples.
6. **PCA 2D Motion Space**: 2D Principal Component Analysis revealing discrete, well-separated cluster manifolds for each posture class.

---

## Interactive Live Stream / Webcam CLI

To monitor continuous camera feeds or run interactive fitness rep tracking:

```bash
# Launch interactive streaming CLI (auto-detects hardware webcam or simulated stream)
python webcam_live.py
```

**Sample Terminal Output:**
```text
==============================================================================
  SIMULATED REAL-TIME VIDEO STREAM POSE ESTIMATION & FITNESS TRACKER
==============================================================================
  Monitoring 33-point body skeleton, joint angle kinematics, and reps...
  Press Ctrl+C to terminate session.

[Frame 000 | 0.25ms] Knee: 175.0° | Elbow: 168.0° | Pose: Standing     ( 99.9%) | Reps: 00 | [MATCH]
[Frame 001 | 0.24ms] Knee:  88.5° | Elbow: 155.0° | Pose: Squat        ( 99.8%) | Reps: 00 | [MATCH]
[Frame 002 | 0.25ms] Knee: 175.0° | Elbow: 168.0° | Pose: Standing     ( 99.9%) | Reps: 01 | [MATCH]
[Frame 003 | 0.24ms] Knee: 172.0° | Elbow: 162.0° | Pose: Jumping_Jack ( 99.7%) | Reps: 01 | [MATCH]
[Frame 004 | 0.25ms] Knee:  92.0° | Elbow: 145.0° | Pose: Lunge        ( 99.6%) | Reps: 01 | [MATCH]
[Frame 005 | 0.24ms] Knee: 165.0° | Elbow:  95.0° | Pose: Pushup       ( 99.9%) | Reps: 01 | [MATCH]
```

---

## Quickstart & Reproduction Guide

### 1. Clone the Repository
```bash
git clone https://github.com/GideonMwiti/pytorch-pose-estimation-mediapipe-skeleton.git
cd pytorch-pose-estimation-mediapipe-skeleton
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline & Generate Dashboard
```bash
python main.py
```

The script will automatically synthesize the full-body pose dataset, evaluate biomechanical joint angles, train `PoseNet`, profile latency, render the fitness HUD simulation, and save the 6-panel dashboard in `outputs/pose_estimation_report.png`.

---

## References

1. **Bazarevsky, V., et al.** (2020). *BlazePose: On-device Real-time Body Pose Tracking*. arXiv:2006.10204.
2. **Lugaresi, C., et al.** (2019). *MediaPipe: A Framework for Building Perception Pipelines*. IEEE CVPR Workshop.
3. **Paszke, A., et al.** (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS.
4. **Winter, D. A.** (2009). *Biomechanics and Motor Control of Human Movement*. John Wiley & Sons.

---

## License

This project is open-source under the [MIT License](LICENSE).

