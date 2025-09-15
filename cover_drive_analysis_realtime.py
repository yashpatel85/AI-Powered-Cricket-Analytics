import argparse
import os
import cv2
import yaml
import json
from utils.pose import PoseEstimator


def analyze_video(input_path, output_path, config):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize pose estimator
    pose_estimator = PoseEstimator(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Pose estimation
        landmarks, annotated_frame = pose_estimator.process_frame(frame, draw = True)

        # Debug: print some key landmarks if detected
        if landmarks:
            if "LEFT_ELBOW" in landmarks and "RIGHT_ELBOW" in landmarks:
                print(
                    f"Frame {frame_idx}: LEFT_ELBOW={landmarks['LEFT_ELBOW']}, "
                    f"RIGHT_ELBOW={landmarks['RIGHT_ELBOW']}"
                )

        out.write(annotated_frame)
        frame_idx += 1

    # Release resources
    cap.release()
    out.release()
    pose_estimator.close()

    # Dummy evaluation results (to be replaced with real metrics)
    results = {
        "Footwork": {"score": 7, "feedback": "Decent, but needs improvement."},
        "Head Position": {"score": 8, "feedback": "Good alignment."},
        "Swing Control": {"score": 6, "feedback": "Work on follow-through."},
        "Balance": {"score": 7, "feedback": "Maintain more stability."},
        "Follow-through": {"score": 8, "feedback": "Well executed."},
    }

    eval_path = os.path.join(os.path.dirname(output_path), "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Annotated video saved to {output_path}")
    print(f"Evaluation saved to {eval_path}")


def main():
    parser = argparse.ArgumentParser(description = "Real-time Cover Drive Analysis")
    parser.add_argument("--input", required = True, help = "Path to input video file")
    parser.add_argument("--output", default = "output/annotated_video.mp4", help = "Path to output video file")
    parser.add_argument("--config", default = "config/config.yaml", help = "Path to config file")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok = True)

    # Load config if available
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    analyze_video(args.input, args.output, config)


if __name__ == "__main__":
    main()
