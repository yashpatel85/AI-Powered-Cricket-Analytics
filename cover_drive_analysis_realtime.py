import cv2
import os
from utils.pose import PoseEstimator
from utils import metrics

def analyze_video(input_path, output_path="output/annotated_video.mp4", config=None):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    # Setup output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    pose_estimator = PoseEstimator()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks, annotated_frame = pose_estimator.process_frame(frame, draw=True)

        if landmarks:
            # Compute metrics
            elbow_angle = metrics.compute_elbow_angle(landmarks, side="LEFT")
            spine_lean = metrics.compute_spine_lean(landmarks, side="LEFT")
            head_knee_dist = metrics.compute_head_over_knee(landmarks, side="LEFT")
            foot_angle = metrics.compute_foot_direction(landmarks, side="LEFT")

            # Overlay text on frame
            overlay_texts = [
                f"Elbow Angle: {elbow_angle:.1f}°" if elbow_angle else "Elbow Angle: N/A",
                f"Spine Lean: {spine_lean:.1f}°" if spine_lean else "Spine Lean: N/A",
                f"Head-Knee X Dist: {head_knee_dist:.1f}px" if head_knee_dist else "Head-Knee: N/A",
                f"Foot Dir: {foot_angle:.1f}°" if foot_angle else "Foot Dir: N/A",
            ]

            y0, dy = 30, 30
            for i, text in enumerate(overlay_texts):
                y = y0 + i * dy
                cv2.putText(annotated_frame, text, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(annotated_frame, text, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(annotated_frame)

    cap.release()
    writer.release()
    return output_path
