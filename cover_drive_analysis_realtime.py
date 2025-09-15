import cv2
import argparse
import yaml
import os

from utils.pose import PoseEstimator
from utils import metrics, evaluation

# --- Feedback thresholds ---
FEEDBACK_THRESHOLDS = {
    "elbow": (90, 130),       # degrees
    "spine_lean": (-10, 10),  # degrees from vertical
    "head_over_knee": 20,     # pixels
    "foot_angle": (-15, 15),  # degrees
}


def get_feedback(elbow, spine, head_knee, foot):
    """Return list of feedback strings per metric"""
    feedbacks = []

    if elbow is not None:
        feedbacks.append(
            "✅ Good elbow bend"
            if FEEDBACK_THRESHOLDS["elbow"][0] <= elbow <= FEEDBACK_THRESHOLDS["elbow"][1]
            else "⚠️ Elbow angle off"
        )
    if spine is not None:
        feedbacks.append(
            "✅ Spine posture OK"
            if FEEDBACK_THRESHOLDS["spine_lean"][0] <= spine <= FEEDBACK_THRESHOLDS["spine_lean"][1]
            else "⚠️ Spine posture off"
        )
    if head_knee is not None:
        feedbacks.append(
            "✅ Head over knee"
            if head_knee <= FEEDBACK_THRESHOLDS["head_over_knee"]
            else "⚠️ Head not over knee"
        )
    if foot is not None:
        feedbacks.append(
            "✅ Foot aligned"
            if FEEDBACK_THRESHOLDS["foot_angle"][0] <= foot <= FEEDBACK_THRESHOLDS["foot_angle"][1]
            else "⚠️ Foot misaligned"
        )

    return feedbacks


def analyze_video(input_path, output_path=None, config=None):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: Could not open {input_path}")
        return

    # Setup output writer if needed
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Pose estimator
    pose_estimator = PoseEstimator()

    # List to collect per-frame metrics
    all_metrics = []

    # Overlay positions
    y0, dy = 30, 30

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get landmarks + annotated frame
        landmarks, annotated_frame = pose_estimator.process_frame(frame, draw=True)

        elbow_angle = spine_lean = head_knee_dist = foot_angle = None

        if landmarks:
            # --- Compute metrics ---
            elbow_angle = metrics.compute_elbow_angle(landmarks, side="LEFT")
            spine_lean = metrics.compute_spine_lean(landmarks, side="LEFT")
            head_knee_dist = metrics.compute_head_over_knee(landmarks, side="LEFT")
            foot_angle = metrics.compute_foot_direction(landmarks, side="LEFT")

            # --- Log to console ---
            print(
                f"Elbow: {elbow_angle:.1f}° | Spine: {spine_lean:.1f}° | "
                f"Head-Knee: {head_knee_dist:.1f}px | Foot: {foot_angle:.1f}°"
                if elbow_angle and spine_lean and head_knee_dist and foot_angle
                else "Pose detected but some metrics missing"
            )

            # --- Overlay numeric metrics ---
            overlay_texts = [
                f"Elbow Angle: {elbow_angle:.1f}°" if elbow_angle else "Elbow Angle: N/A",
                f"Spine Lean: {spine_lean:.1f}°" if spine_lean else "Spine Lean: N/A",
                f"Head-Knee X Dist: {head_knee_dist:.1f}px" if head_knee_dist else "Head-Knee: N/A",
                f"Foot Dir: {foot_angle:.1f}°" if foot_angle else "Foot Dir: N/A",
            ]

            for i, text in enumerate(overlay_texts):
                y = y0 + i * dy
                cv2.putText(annotated_frame, text, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(annotated_frame, text, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # --- Overlay feedback cues ---
            feedbacks = get_feedback(elbow_angle, spine_lean, head_knee_dist, foot_angle)
            for i, fb in enumerate(feedbacks):
                y = y0 + (len(overlay_texts) + i) * dy
                color = (0, 255, 0) if "✅" in fb else (0, 0, 255)
                cv2.putText(annotated_frame, fb, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(annotated_frame, fb, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # --- Collect metrics for evaluation ---
        frame_metrics = {
            "elbow": elbow_angle,
            "spine": spine_lean,
            "head_knee": head_knee_dist,
            "foot": foot_angle
        }
        all_metrics.append(frame_metrics)

        # --- Live display ---
        cv2.imshow("Cover Drive Analysis", annotated_frame)

        # --- Write output video ---
        if writer:
            writer.write(annotated_frame)

        # Quit early on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # --- Final evaluation ---
    eval_result = evaluation.evaluate_shot(all_metrics)
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
    # Suppress TensorFlow warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()
