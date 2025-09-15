import json

# Thresholds (can match metrics.py thresholds)
THRESHOLDS = {
    "elbow": (90, 130),
    "spine_lean": (-10, 10),
    "head_over_knee": 20,  # pixels
    "foot_angle": (-15, 15),
}


def score_percentage(frames_ok, total_frames):
    """Map percentage of good frames to score 1–10."""
    if total_frames == 0:
        return 0
    percent = frames_ok / total_frames
    return max(1, round(percent * 10))


def evaluate_shot(metrics_per_frame):
    """
    Args:
        metrics_per_frame: list of dicts per frame, e.g.,
        [{"elbow": 115, "spine": 5, "head_knee": 12, "foot": 10}, ...]
    Returns:
        dict with category scores + feedback
    """
    total_frames = len(metrics_per_frame)
    counters = {"elbow": 0, "spine": 0, "head_knee": 0, "foot": 0}

    for m in metrics_per_frame:
        if m.get("elbow") is not None and THRESHOLDS["elbow"][0] <= m["elbow"] <= THRESHOLDS["elbow"][1]:
            counters["elbow"] += 1
        if m.get("spine") is not None and THRESHOLDS["spine_lean"][0] <= m["spine"] <= THRESHOLDS["spine_lean"][1]:
            counters["spine"] += 1
        if m.get("head_knee") is not None and m["head_knee"] <= THRESHOLDS["head_over_knee"]:
            counters["head_knee"] += 1
        if m.get("foot") is not None and THRESHOLDS["foot_angle"][0] <= m["foot"] <= THRESHOLDS["foot_angle"][1]:
            counters["foot"] += 1

    # Compute scores
    scores = {
        "Footwork": score_percentage(counters["foot"], total_frames),
        "Head Position": score_percentage(counters["head_knee"], total_frames),
        "Swing Control": score_percentage(counters["elbow"], total_frames),
        "Balance": score_percentage(counters["spine"], total_frames),
        "Follow-through": score_percentage(
            min(counters["elbow"], counters["spine"], counters["head_knee"]), total_frames
        ),
    }

    # Generate feedback
    feedback = {}
    for cat, score in scores.items():
        if score >= 8:
            feedback[cat] = "Excellent"
        elif score >= 5:
            feedback[cat] = "Good, some improvement needed"
        else:
            feedback[cat] = "Needs improvement"

    return {"scores": scores, "feedback": feedback}


def save_evaluation(eval_dict, output_path="evaluation.json"):
    with open(output_path, "w") as f:
        json.dump(eval_dict, f, indent=4)
