from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


Point = Tuple[float, float, float]
Keypoints = Dict[str, Point]


KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

ANGLE_POINTS = {
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_shoulder": ("left_elbow", "left_shoulder", "left_hip"),
    "right_shoulder": ("right_elbow", "right_shoulder", "right_hip"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
}

MODEL_PATH = "yolov8n-pose.pt"
CONFIDENCE_THRESHOLD = 0.25
INFERENCE_IMAGE_SIZE = 640
DEFAULT_TOLERANCE = 38

_model = None


@dataclass(frozen=True)
class PoseTemplate:
    pose_id: int
    name: str
    angles: Dict[str, float]
    relations: Tuple[Tuple[str, str, str, float], ...] = ()
    tolerance: int = DEFAULT_TOLERANCE


def calculate_angle(a: Iterable[float], b: Iterable[float], c: Iterable[float]) -> float:
    """Return the angle ABC in degrees."""
    point_a = np.array(list(a)[:2], dtype=np.float32)
    point_b = np.array(list(b)[:2], dtype=np.float32)
    point_c = np.array(list(c)[:2], dtype=np.float32)

    ba = point_a - point_b
    bc = point_c - point_b
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator == 0:
        return 0.0

    cosine = np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def load_model(model_path: str = MODEL_PATH):
    global _model
    if _model is None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is required. Install it with: pip install ultralytics"
            ) from exc
        _model = YOLO(model_path)
    return _model


def detect_pose(frame, model=None) -> Tuple[Keypoints, Optional[object]]:
    """Run YOLO Pose on one frame and return the largest detected person."""
    if model is None:
        model = load_model()

    results = model.predict(
        frame,
        imgsz=INFERENCE_IMAGE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
    )
    if not results:
        return {}, None

    result = results[0]
    if result.keypoints is None or result.keypoints.xy is None:
        return {}, result

    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
    all_xy = result.keypoints.xy.cpu().numpy()
    all_conf = result.keypoints.conf.cpu().numpy()
    if len(all_xy) == 0:
        return {}, result

    best_index = _largest_person_index(boxes, len(all_xy))
    frame_height, frame_width = frame.shape[:2]
    keypoints: Keypoints = {}

    for index, name in enumerate(KEYPOINT_NAMES):
        x, y = all_xy[best_index][index]
        confidence = all_conf[best_index][index]
        keypoints[name] = (float(x / frame_width), float(y / frame_height), float(confidence))

    return keypoints, result


def check_pose(
    keypoints: Keypoints,
    pose: PoseTemplate | int,
    tolerance: Optional[int] = None,
) -> bool:
    template = get_pose_by_id(pose) if isinstance(pose, int) else pose
    if not template or not _has_required_points(keypoints, template):
        return False

    angle_tolerance = template.tolerance if tolerance is None else tolerance
    for angle_name, target_angle in template.angles.items():
        measured = _measure_angle(keypoints, angle_name)
        if measured is None or abs(measured - target_angle) > angle_tolerance:
            return False

    for relation in template.relations:
        if not _check_relation(keypoints, relation):
            return False

    return True


def get_next_pose(current_pose_id: int = 0) -> PoseTemplate:
    next_index = current_pose_id % len(POSE_TEMPLATES)
    return POSE_TEMPLATES[next_index]


def get_pose_by_id(pose_id: int) -> Optional[PoseTemplate]:
    for pose in POSE_TEMPLATES:
        if pose.pose_id == pose_id:
            return pose
    return None


def draw_keypoints(frame, keypoints: Keypoints) -> None:
    height, width = frame.shape[:2]
    for name_a, name_b in _SKELETON:
        if _visible(keypoints, name_a) and _visible(keypoints, name_b):
            ax, ay, _ = keypoints[name_a]
            bx, by, _ = keypoints[name_b]
            cv2.line(
                frame,
                (int(ax * width), int(ay * height)),
                (int(bx * width), int(by * height)),
                (0, 220, 255),
                2,
            )

    for name, (x, y, confidence) in keypoints.items():
        if confidence >= CONFIDENCE_THRESHOLD:
            color = (70, 255, 140) if "wrist" in name else (255, 255, 255)
            cv2.circle(frame, (int(x * width), int(y * height)), 4, color, -1)


def _largest_person_index(boxes, person_count: int) -> int:
    if len(boxes) == 0:
        return 0
    areas = []
    for box in boxes[:person_count]:
        x1, y1, x2, y2 = box
        areas.append(max(0, x2 - x1) * max(0, y2 - y1))
    return int(np.argmax(areas)) if areas else 0


def _has_required_points(keypoints: Keypoints, template: PoseTemplate) -> bool:
    required = set()
    for angle_name in template.angles:
        required.update(ANGLE_POINTS[angle_name])
    for _, point_a, point_b, _ in template.relations:
        required.add(point_a)
        required.add(point_b)
    return all(_visible(keypoints, name) for name in required)


def _visible(keypoints: Keypoints, name: str) -> bool:
    return name in keypoints and keypoints[name][2] >= CONFIDENCE_THRESHOLD


def _measure_angle(keypoints: Keypoints, angle_name: str) -> Optional[float]:
    names = ANGLE_POINTS.get(angle_name)
    if names is None or not all(_visible(keypoints, name) for name in names):
        return None
    return calculate_angle(keypoints[names[0]], keypoints[names[1]], keypoints[names[2]])


def _body_scale(keypoints: Keypoints) -> float:
    if _visible(keypoints, "left_shoulder") and _visible(keypoints, "right_shoulder"):
        left = keypoints["left_shoulder"]
        right = keypoints["right_shoulder"]
        return max(0.08, abs(left[0] - right[0]))
    return 0.18


def _check_relation(keypoints: Keypoints, relation: Tuple[str, str, str, float]) -> bool:
    relation_type, point_a, point_b, margin = relation
    a = keypoints[point_a]
    b = keypoints[point_b]
    body_scale = _body_scale(keypoints)
    scaled_margin = margin * body_scale * 0.75

    if relation_type == "above":
        return a[1] < b[1] - scaled_margin
    if relation_type == "below":
        return a[1] > b[1] + scaled_margin
    if relation_type == "left_of":
        return a[0] < b[0] - scaled_margin
    if relation_type == "right_of":
        return a[0] > b[0] + scaled_margin
    if relation_type == "near":
        distance = np.linalg.norm(np.array(a[:2]) - np.array(b[:2]))
        return distance <= margin * max(body_scale, 0.16) * 1.25
    if relation_type == "far":
        distance = np.linalg.norm(np.array(a[:2]) - np.array(b[:2]))
        return distance >= margin * max(body_scale, 0.16) * 0.8
    if relation_type == "same_y":
        return abs(a[1] - b[1]) <= scaled_margin * 1.35
    return False


def _relations(*items: Tuple[str, str, str, float]) -> Tuple[Tuple[str, str, str, float], ...]:
    return tuple(items)


# ANSWER / VALIDATION TEMPLATES
# These are the real pose answers used by the game logic.
# Each template defines the angles and body-point relations that YOLO Pose must match.
# If the player's detected keypoints pass these rules, the pose is counted as correct.
POSE_TEMPLATES: List[PoseTemplate] = [
    PoseTemplate(1, "Hands Up", {"left_elbow": 155, "right_elbow": 155}, _relations(("above", "left_wrist", "left_shoulder", 0.15), ("above", "right_wrist", "right_shoulder", 0.15))),
    PoseTemplate(2, "T Pose", {"left_elbow": 160, "right_elbow": 160}, _relations(("same_y", "left_wrist", "left_shoulder", 0.8), ("same_y", "right_wrist", "right_shoulder", 0.8), ("far", "left_wrist", "right_wrist", 1.8))),
    PoseTemplate(3, "Left Hand Up", {"left_elbow": 145}, _relations(("above", "left_wrist", "left_shoulder", 0.2))),
    PoseTemplate(4, "Right Hand Up", {"right_elbow": 145}, _relations(("above", "right_wrist", "right_shoulder", 0.2))),
    PoseTemplate(5, "Heart Pose", {"left_elbow": 80, "right_elbow": 80}, _relations(("near", "left_wrist", "right_wrist", 1.0), ("above", "left_wrist", "nose", 0.1), ("above", "right_wrist", "nose", 0.1))),
    PoseTemplate(6, "Hands On Head", {"left_elbow": 70, "right_elbow": 70}, _relations(("near", "left_wrist", "nose", 1.2), ("near", "right_wrist", "nose", 1.2))),
    PoseTemplate(7, "Left Hand On Head", {"left_elbow": 85}, _relations(("near", "left_wrist", "nose", 1.5))),
    PoseTemplate(8, "Right Hand On Head", {"right_elbow": 85}, _relations(("near", "right_wrist", "nose", 1.5))),
    PoseTemplate(9, "Left Hand On Hip", {}, _relations(("near", "left_wrist", "left_hip", 1.2))),
    PoseTemplate(10, "Right Hand On Hip", {}, _relations(("near", "right_wrist", "right_hip", 1.2))),
    PoseTemplate(11, "Both Hands On Hips", {"left_elbow": 80, "right_elbow": 80}, _relations(("near", "left_wrist", "left_hip", 0.9), ("near", "right_wrist", "right_hip", 0.9))),
    PoseTemplate(12, "Wide Arms High", {"left_elbow": 150, "right_elbow": 150}, _relations(("above", "left_wrist", "left_shoulder", 0.05), ("above", "right_wrist", "right_shoulder", 0.05), ("far", "left_wrist", "right_wrist", 1.8))),
    PoseTemplate(13, "Arms Down Straight", {"left_elbow": 145, "right_elbow": 145}, _relations(("below", "left_wrist", "left_shoulder", 0.15), ("below", "right_wrist", "right_shoulder", 0.15))),
    PoseTemplate(14, "Left Arm Side", {"left_elbow": 145}, _relations(("same_y", "left_wrist", "left_shoulder", 0.8))),
    PoseTemplate(15, "Right Arm Side", {"right_elbow": 145}, _relations(("same_y", "right_wrist", "right_shoulder", 0.8))),
    PoseTemplate(16, "Left Elbow Bent", {"left_elbow": 90}, _relations(("above", "left_wrist", "left_elbow", -0.05))),
    PoseTemplate(17, "Right Elbow Bent", {"right_elbow": 90}, _relations(("above", "right_wrist", "right_elbow", -0.05))),
    PoseTemplate(18, "Cross Arms", {"left_elbow": 80, "right_elbow": 80}, _relations(("near", "left_wrist", "right_elbow", 1.0), ("near", "right_wrist", "left_elbow", 1.0))),
    PoseTemplate(19, "Left Knee Up", {"left_knee": 100}, _relations(("above", "left_knee", "right_knee", 0.25))),
    PoseTemplate(20, "Right Knee Up", {"right_knee": 100}, _relations(("above", "right_knee", "left_knee", 0.25))),
    PoseTemplate(21, "Squat", {"left_knee": 110, "right_knee": 110}, _relations(("below", "left_hip", "left_knee", 0.05), ("below", "right_hip", "right_knee", 0.05))),
    PoseTemplate(22, "Straight Stand", {"left_knee": 160, "right_knee": 160}, _relations(("below", "left_wrist", "left_shoulder", 0.1), ("below", "right_wrist", "right_shoulder", 0.1))),
    PoseTemplate(23, "Left Lean", {}, _relations(("left_of", "nose", "left_hip", 0.05))),
    PoseTemplate(24, "Right Lean", {}, _relations(("right_of", "nose", "right_hip", 0.05))),
    PoseTemplate(25, "Left Punch", {"left_elbow": 145}, _relations(("same_y", "left_wrist", "left_shoulder", 0.9), ("left_of", "left_wrist", "left_shoulder", 0.35))),
    PoseTemplate(26, "Right Punch", {"right_elbow": 145}, _relations(("same_y", "right_wrist", "right_shoulder", 0.9), ("right_of", "right_wrist", "right_shoulder", 0.35))),
    PoseTemplate(27, "Goalkeeper", {"left_elbow": 110, "right_elbow": 110}, _relations(("above", "left_wrist", "left_shoulder", 0.1), ("above", "right_wrist", "right_shoulder", 0.1), ("far", "left_wrist", "right_wrist", 1.3))),
    PoseTemplate(28, "Small Circle Arms", {"left_elbow": 65, "right_elbow": 65}, _relations(("near", "left_wrist", "right_wrist", 0.8), ("above", "left_wrist", "left_hip", 0.3))),
    PoseTemplate(29, "Diagonal Left", {"left_elbow": 165, "right_elbow": 165}, _relations(("above", "left_wrist", "left_shoulder", 0.4), ("below", "right_wrist", "right_hip", 0.0))),
    PoseTemplate(30, "Victory Pose", {"left_elbow": 165, "right_elbow": 165}, _relations(("above", "left_wrist", "left_shoulder", 0.4), ("above", "right_wrist", "right_shoulder", 0.4), ("far", "left_wrist", "right_wrist", 1.2))),
]


_SKELETON = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]
