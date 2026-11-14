"""
Project 71: Interactive Live Video Stream Human Pose Estimation & Fitness CLI
Processes video frames, extracts 33 3D full-body landmarks, calculates biomechanical
joint angles (knee, elbow, hip), counts repetitions, and classifies postures using PoseNet.
"""

import os
import sys
import time
import numpy as np
import torch

from main import (
    PoseNet,
    POSES,
    NUM_CLASSES,
    DEVICE,
    synthesize_pose_landmarks,
    normalize_pose_landmarks,
    extract_biomechanical_angles,
    rule_based_classify
)

def run_simulated_stream_cli(model):
    print("\n" + "=" * 78)
    print("  SIMULATED REAL-TIME VIDEO STREAM POSE ESTIMATION & FITNESS TRACKER")
    print("=" * 78)
    print("  Monitoring 33-point body skeleton, joint angle kinematics, and reps...")
    print("  Press Ctrl+C to terminate session.\n")

    stream_cycle = [
        ("Standing",     "Upright Posture (Standing)"),
        ("Squat",        "Squat (Knee Flexion < 100°)"),
        ("Standing",     "Upright Posture (Standing)"),
        ("Jumping_Jack", "Jumping Jack (Overhead Arms & Abducted Legs)"),
        ("Lunge",        "Lunge (Split Stance Form)"),
        ("Pushup",       "Pushup (Horizontal Plank)")
    ]

    rep_counter = 0
    in_squat = False

    try:
        frame_idx = 0
        while True:
            target_pose, desc = stream_cycle[frame_idx % len(stream_cycle)]
            pts = synthesize_pose_landmarks(target_pose, apply_noise=True)

            # Biomechanical Angle Calculation
            angles = extract_biomechanical_angles(pts)
            knee_angle = (angles["left_knee"] + angles["right_knee"]) / 2.0
            elbow_angle = (angles["left_elbow"] + angles["right_elbow"]) / 2.0

            # Repetition Counter Logic (Squats)
            if knee_angle < 105.0 and not in_squat:
                in_squat = True
            elif knee_angle > 150.0 and in_squat:
                rep_counter += 1
                in_squat = False

            # Neural Network Classification
            feat = normalize_pose_landmarks(pts)
            feat_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            t0 = time.perf_counter()
            with torch.no_grad():
                logits = model(feat_t)
                probs = torch.softmax(logits, dim=1)[0]
                conf, pred_idx = torch.max(probs, 0)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            pred_pose = POSES[pred_idx.item()]
            rule_pred = rule_based_classify(pts)
            agreement = "MATCH" if pred_pose == rule_pred else "HYBRID"

            print(f"[Frame {frame_idx:03d} | {latency_ms:4.2f}ms] "
                  f"Knee: {knee_angle:5.1f}° | Elbow: {elbow_angle:5.1f}° | "
                  f"Pose: {pred_pose:<12} ({conf.item()*100:5.1f}%) | "
                  f"Reps: {rep_counter:02d} | [{agreement}]")

            frame_idx += 1
            time.sleep(0.35)

    except KeyboardInterrupt:
        print("\nLive video stream monitoring stopped by user.")

def main():
    weights_path = os.path.join(os.path.dirname(__file__), "outputs", "posenet_best.pth")
    model = PoseNet(in_features=99, num_classes=NUM_CLASSES).to(DEVICE)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()

    run_simulated_stream_cli(model)

if __name__ == "__main__":
    main()
