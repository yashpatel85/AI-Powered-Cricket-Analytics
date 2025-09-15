import cv2
import argparse
import yaml
import os
import numpy as np
from collections import Counter

from utils.pose import PoseEstimator
from utils import metrics, evaluation, phases as phase_module

# --- Feedback thresholds ---
FEEDBACK_THRESHOLDS = {
    "elbow": (90, 130),
    "spine_lean": (-10, 10),
    "head_over_knee": 20,
    "foot_angle": (-15, 15),
}

def get_feedback(elbow, spine, head_knee, foot):
    feedbacks = []
    if elbow is not None:
        feedbacks.append("✅ Good elbow bend" if FEEDBACK_THRESHOLDS["elbow"][0] <= elbow <= FEEDBACK_THRESHOLDS["elbow"][1] else "⚠️ Elbow angle off")
    if spine is not None:
        feedbacks.append("✅ Spine posture OK" if FEEDBACK_THRESHOLDS["spine_lean"][0] <= spine <= FEEDBACK_THRESHOLDS["spine_lean"][1] else "⚠️ Spine posture off")
    if head_knee is not None:
        feedbacks.append("✅ Head over knee" if head_knee <= FEEDBACK_THRESHOLDS["head_over_knee"] else "⚠️ Head not over knee")
    if foot is not None:
        feedbacks.append("✅ Foot aligned" if FEEDBACK_THRESHOLDS["foot_angle"][0] <= foot <= FEEDBACK_THRESHOLDS["foot_angle"][1] else "⚠️ Foot misaligned")
    return feedbacks

def overlay_phase(frame, phase_name, position=(10, 250)):
    cv2.putText(frame, f"Phase: {phase_name}", position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(frame, f"Phase: {phase_name}", position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2, cv2.LINE_AA)

# --- String-compatible phase smoothing ---
def smooth_phases(detected_phases, window_size=5):
    smoothed = []
    n = len(detected_phases)
    for i in range(n):
        start = max(0, i - window_size//2)
        end = min(n, i + window_size//2 + 1)
        window = detected_phases[start:end]
        most_common = Counter(window).most_common(1)[0][0]
        smoothed.append(most_common)
    return smoothed

def analyze_video(input_path, output_path=None, config=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open {input_path}")
        return

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    pose_estimator = PoseEstimator()
    all_metrics = []
    y0, dy = 30, 30

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        landmarks, annotated_frame = pose_estimator.process_frame(frame, draw=True)
        elbow_angle = spine_lean = head_knee_dist = foot_angle = None

        if landmarks:
            elbow_angle = metrics.compute_elbow_angle(landmarks, side="LEFT")
            spine_lean = metrics.compute_spine_lean(landmarks, side="LEFT")
            head_knee_dist = metrics.compute_head_over_knee(landmarks, side="LEFT")
            foot_angle = metrics.compute_foot_direction(landmarks, side="LEFT")

            print(f"Elbow: {elbow_angle:.1f}° | Spine: {spine_lean:.1f}° | Head-Knee: {head_knee_dist:.1f}px | Foot: {foot_angle:.1f}°")

            # Overlay metrics
            overlay_texts = [
                f"Elbow Angle: {elbow_angle:.1f}°" if elbow_angle else "Elbow Angle: N/A",
                f"Spine Lean: {spine_lean:.1f}°" if spine_lean else "Spine Lean: N/A",
                f"Head-Knee X Dist: {head_knee_dist:.1f}px" if head_knee_dist else "Head-Knee: N/A",
                f"Foot Dir: {foot_angle:.1f}°" if foot_angle else "Foot Dir: N/A",
            ]
            for i, text in enumerate(overlay_texts):
                y = y0 + i*dy
                cv2.putText(annotated_frame, text, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4, cv2.LINE_AA)
                cv2.putText(annotated_frame, text, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            # Overlay feedback
            feedbacks = get_feedback(elbow_angle, spine_lean, head_knee_dist, foot_angle)
            for i, fb in enumerate(feedbacks):
                y = y0 + (len(overlay_texts)+i)*dy
                color = (0,255,0) if "✅" in fb else (0,0,255)
                cv2.putText(annotated_frame, fb, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4, cv2.LINE_AA)
                cv2.putText(annotated_frame, fb, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        frame_metrics = {
            "elbow": elbow_angle,
            "spine": spine_lean,
            "head_knee": head_knee_dist,
            "foot": foot_angle,
            "landmarks": landmarks
        }
        all_metrics.append(frame_metrics)

        # Phase detection + smoothing
        detected_phases = phase_module.detect_phases(all_metrics)
        smoothed_phases = smooth_phases(detected_phases, window_size=5)
        current_phase = smoothed_phases[-1] if smoothed_phases else "Stance"
        overlay_phase(annotated_frame, current_phase)

        # Display & write
        cv2.imshow("Cover Drive Analysis", annotated_frame)
        if writer:
            writer.write(annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Final evaluation
    eval_result = evaluation.evaluate_shot(all_metrics)
    eval_result["phases"] = smoothed_phases
    os.makedirs("results", exist_ok=True)
    evaluation.save_evaluation(eval_result, "results/evaluation.json")
    print("✅ Evaluation saved to results/evaluation.json")
    print(eval_result)

def main():
    parser = argparse.ArgumentParser(description="Cricket Cover Drive Pose Analysis")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", help="Path to save annotated output video")
    parser.add_argument("--config", help="YAML config file (optional)")
    args = parser.parse_args()

    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    analyze_video(args.input, args.output, config)

if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()
