import cv2
import argparse
import yaml
import os

from utils.pose import PoseEstimator
from utils import metrics


def analyze_video(input_path, output_path=None, config=None):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"❌ Error: Could not open {input_path}")
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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get landmarks + annotated frame
        landmarks, annotated_frame = pose_estimator.process_frame(frame, draw=True)

        if landmarks:
            # --- Compute metrics ---
            elbow_angle = metrics.compute_elbow_angle(landmarks, side="LEFT")
            spine_lean = metrics.compute_spine_lean(landmarks, side="LEFT")
            head_knee_dist = metrics.compute_head_over_knee(landmarks, side="LEFT")
            foot_angle = metrics.compute_foot_direction(landmarks, side="LEFT")

            # --- Overlay results ---
            overlay_texts = [
                f"Elbow Angle: {elbow_angle:.1f}°" if elbow_angle else "Elbow Angle: N/A",
                f"Spine Lean: {spine_lean:.1f}°" if spine_lean else "Spine Lean: N/A",
                f"Head-Knee X Dist: {head_knee_dist:.1f}px" if head_knee_dist else "Head-Knee: N/A",
                f"Foot Dir: {foot_angle:.1f}°" if foot_angle else "Foot Dir: N/A",
            ]

            y0, dy = 30, 30
            for i, text in enumerate(overlay_texts):
                y = y0 + i * dy
                cv2.putText(
                    annotated_frame,
                    text,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        # Write or display
        if writer:
            writer.write(annotated_frame)
        else:
            cv2.imshow("Cover Drive Analysis", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


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
    main()
