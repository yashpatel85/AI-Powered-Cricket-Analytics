import math
from typing import Dict, Tuple, Optional

def _get_point(landmarks: Dict[str, Tuple[int, int, float]], key: str) -> Optional[Tuple[float, float]]:

    if key not in landmarks:
        return None
    x, y, _ = landmarks[key]
    return float(x), float(y)

def angle_between_points(a, b, c) -> Optional[float]:

    if not (a and b and c):
        return None
    
    ax, ay = a
    bx, by = b
    cx, cy = c

    ab = (ax - bx, ay - by)
    cb = (cx - bx, cy - by)

    dot = ab[0] * cb[0] + ab[1] * cb[1]
    mag_ab = math.hypot(*ab)
    mag_cb = math.hypot(*cb)

    if mag_ab == 0 or mag_cb == 0:
        return None
    
    cos_angle = dot / (mag_ab * mag_cb)
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    angle = math.degrees(math.acos(cos_angle))

    return angle



def compute_elbow_angle(landmarks: Dict[str, Tuple[int, int, float]], side: str = "LEFT") -> Optional[float]:

    side = side.upper()
    shoulder = _get_point(landmarks, f"{side}_SHOULDER")
    elbow = _get_point(landmarks, f"{side}_ELBOW")
    wrist = _get_point(landmarks, f"{side}_WRIST")
    return angle_between_points(shoulder, elbow, wrist)


def compute_spine_lean(landmarks: Dict[str, Tuple[int, int, float]], side: str = "LEFT") -> Optional[float]:

    side = side.upper()
    hip = _get_point(landmarks, f"{side}_HIP")
    shoulder = _get_point(landmarks, f"{side}_SHOULDER")

    if not (hip and shoulder):
        return None
    
    dx = shoulder[0] - hip[0]
    dy = shoulder[0] - hip[1]

    angle = math.degrees(math.atan2(dx, dy))
    return angle


def compute_head_over_knee(landmarks: Dict[str, Tuple[int, int, float]], side: str = "LEFT") -> Optional[float]:
   
    side = side.upper()
    head = _get_point(landmarks, "NOSE") or _get_point(landmarks, "HEAD_TOP")
    knee = _get_point(landmarks, f"{side}_KNEE")

    if not (head and knee):
        return None

    return abs(head[0] - knee[0])  # pixel distance in x-axis


def compute_foot_direction(landmarks: Dict[str, Tuple[int, int, float]], side: str = "LEFT") -> Optional[float]:
    
    side = side.upper()
    ankle = _get_point(landmarks, f"{side}_ANKLE")
    foot_index = _get_point(landmarks, f"{side}_FOOT_INDEX")

    if not (ankle and foot_index):
        return None

    dx = foot_index[0] - ankle[0]
    dy = foot_index[1] - ankle[1]

    angle = math.degrees(math.atan2(dy, dx))  # relative to horizontal x-axis
    return angle