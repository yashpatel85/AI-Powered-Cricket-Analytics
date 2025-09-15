import cv2
import mediapipe as mp
import numpy as np

class PoseEstimator:

    def __init__(self, static_image_mode = False, model_complexity = 1,
                 enable_segmentation = False, min_detection_confidence = 0.5,
                 min_tracking_confidence = 0.5):
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode = static_image_mode,
            model_complexity = model_complexity,
            enable_segmentation = enable_segmentation,
            min_detection_confidence = min_detection_confidence,
            min_tracking_confidence = min_tracking_confidence
        )
        self.drawer = mp.solutions.drawing_utils
        self.drawer_styles = mp.solutions.drawing_styles

    def process_frame(self, frame, draw = False):

        rgb = cv2.cvtColor(frame, cv2.BGR2RGB)
        results = self.pose.process(rgb)

        landmarks = {}
        if results.pose_landmarks:
            h, w, _ = frame.shape
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks[self.mp_pose.PoseLandmark(idx).name] = (cx, cy, lm.z)

            if draw:
                self.drawer.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmakr_drawing_spec = self.drawer_styles.get_default_pose_landmarks_style()
                )

        return landmarks, frame
    
    def close(self):
        self.pose.close()