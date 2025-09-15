import numpy as np

def compute_velocity(points_prev, points_curr):
    """
    Compute simple frame-to-frame velocity for LEFT wrist + elbow.
    Assumes points_prev and points_curr are dictionaries with keys:
        'LEFT_WRIST', 'LEFT_ELBOW'
    Returns 0 if any landmarks are missing.
    """
    if points_prev is None or points_curr is None:
        return 0.0

    try:
        prev_wrist = np.array(points_prev['LEFT_WRIST'])
        prev_elbow = np.array(points_prev['LEFT_ELBOW'])
        curr_wrist = np.array(points_curr['LEFT_WRIST'])
        curr_elbow = np.array(points_curr['LEFT_ELBOW'])
    except KeyError:
        return 0.0

    return np.linalg.norm(np.concatenate([curr_wrist, curr_elbow]) -
                          np.concatenate([prev_wrist, prev_elbow]))


def detect_phases(metrics_per_frame):
    """
    Detect cricket cover drive phases per frame.
    Phases: Stance → Stride → Downswing → Impact → Follow-through → Recovery
    """
    phases = []
    prev_landmarks = None

    for frame_metrics in metrics_per_frame:
        phase = "Stance"
        elbow = frame_metrics.get("elbow")
        curr_landmarks = frame_metrics.get("landmarks")
        wrist_velocity = compute_velocity(prev_landmarks, curr_landmarks)

        if elbow is not None:
            if wrist_velocity < 0.01 and elbow > 140:
                phase = "Stance"
            elif 0.01 <= wrist_velocity < 0.03 and elbow > 140:
                phase = "Stride"
            elif wrist_velocity >= 0.03 and elbow > 90:
                phase = "Downswing"
            elif elbow <= 90 and wrist_velocity >= 0.05:
                phase = "Impact"
            elif elbow > 90 and wrist_velocity < 0.05:
                phase = "Follow-through"
            else:
                phase = "Recovery"

        phases.append(phase)
        prev_landmarks = curr_landmarks

    return phases
