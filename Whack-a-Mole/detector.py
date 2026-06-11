"""Webcam hand-skeleton detector based on OpenCV frames and MediaPipe Hands."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp

import config


@dataclass
class DetectedHand:
    """A detected hand with a closeness score."""

    landmarks: object
    cursor_point: tuple[int, int]
    closeness_score: float


class SkeletonDetector:
    """Detects hand skeletons and returns the hand closest to the camera."""

    def __init__(self) -> None:
        self.hands_module = mp.solutions.hands
        self.pose_module = mp.solutions.pose
        self.drawing = mp.solutions.drawing_utils
        self.hands = self.hands_module.Hands(
            static_image_mode=False,
            max_num_hands=config.HAND_MAX_NUM_HANDS,
            model_complexity=config.HAND_MODEL_COMPLEXITY,
            min_detection_confidence=config.HAND_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.HAND_MIN_TRACKING_CONFIDENCE,
        )
        self.pose = self.pose_module.Pose(
            static_image_mode=False,
            model_complexity=config.POSE_FALLBACK_MODEL_COMPLEXITY,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=config.POSE_FALLBACK_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.POSE_FALLBACK_MIN_TRACKING_CONFIDENCE,
        )
        self.smoothed_point: tuple[int, int] | None = None
        self.previous_raw_point: tuple[int, int] | None = None

    def detect(self, frame):
        """Return a processed frame, closest-hand point, hand landmarks, and mask."""
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (config.CAMERA_WIDTH, config.CAMERA_HEIGHT))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if not results.multi_hand_landmarks:
            fallback_point = self.detect_pose_wrist_fallback(rgb_frame)
            if fallback_point is None:
                self.smoothed_point = None
                self.previous_raw_point = None
                return frame, None, None, None

            cursor_point = self.smooth_point(fallback_point)
            cv2.circle(frame, cursor_point, 10, config.SKELETON_CURSOR_COLOR, -1)
            cv2.circle(frame, cursor_point, 18, config.SKELETON_CURSOR_COLOR, 2)
            return frame, cursor_point, None, None

        hands = [
            self.make_detected_hand(hand_landmarks)
            for hand_landmarks in results.multi_hand_landmarks
        ]
        closest_hand = max(hands, key=lambda hand: hand.closeness_score)
        cursor_point = self.smooth_point(closest_hand.cursor_point)

        for hand in hands:
            is_selected = hand is closest_hand
            self.draw_hand_skeleton(frame, hand.landmarks, is_selected)

        cv2.circle(frame, cursor_point, 10, config.SKELETON_CURSOR_COLOR, -1)
        cv2.circle(frame, cursor_point, 18, config.SKELETON_CURSOR_COLOR, 2)

        return frame, cursor_point, closest_hand.landmarks, None

    def detect_pose_wrist_fallback(self, rgb_frame) -> tuple[int, int] | None:
        """Use the nearest visible wrist when tilted-hand landmarks are lost."""
        results = self.pose.process(rgb_frame)
        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark
        wrist_indices = (
            self.pose_module.PoseLandmark.RIGHT_WRIST.value,
            self.pose_module.PoseLandmark.LEFT_WRIST.value,
        )
        visible_wrists = [
            landmarks[index]
            for index in wrist_indices
            if landmarks[index].visibility >= config.POSE_FALLBACK_MIN_WRIST_VISIBILITY
        ]
        if not visible_wrists:
            return None

        nearest_wrist = min(visible_wrists, key=lambda landmark: landmark.z)
        return self.normalized_to_pixel(nearest_wrist.x, nearest_wrist.y)

    def make_detected_hand(self, hand_landmarks) -> DetectedHand:
        """Convert MediaPipe landmarks into a scored hand target."""
        landmarks = hand_landmarks.landmark
        cursor_point = self.get_palm_center(landmarks)

        xs = [landmark.x for landmark in landmarks]
        ys = [landmark.y for landmark in landmarks]
        zs = [landmark.z for landmark in landmarks]
        box_area = max(0.0, max(xs) - min(xs)) * max(0.0, max(ys) - min(ys))
        average_z = sum(zs) / len(zs)

        # Larger projected hand area usually means closer to the camera.
        # MediaPipe z is smaller when the hand is closer, so subtract it.
        closeness_score = (box_area * config.HAND_AREA_SCORE_WEIGHT) - average_z
        return DetectedHand(hand_landmarks, cursor_point, closeness_score)

    def get_palm_center(self, landmarks) -> tuple[int, int]:
        """Use the palm center instead of fingertips for a steadier cursor."""
        palm_indices = (
            self.hands_module.HandLandmark.WRIST.value,
            self.hands_module.HandLandmark.THUMB_CMC.value,
            self.hands_module.HandLandmark.INDEX_FINGER_MCP.value,
            self.hands_module.HandLandmark.MIDDLE_FINGER_MCP.value,
            self.hands_module.HandLandmark.RING_FINGER_MCP.value,
            self.hands_module.HandLandmark.PINKY_MCP.value,
        )
        x = sum(landmarks[index].x for index in palm_indices) / len(palm_indices)
        y = sum(landmarks[index].y for index in palm_indices) / len(palm_indices)
        return self.normalized_to_pixel(x, y)

    def normalized_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        """Convert normalized MediaPipe coordinates into camera pixels."""
        pixel_x = int(x * config.CAMERA_WIDTH)
        pixel_y = int(y * config.CAMERA_HEIGHT)
        pixel_x = max(0, min(config.CAMERA_WIDTH - 1, pixel_x))
        pixel_y = max(0, min(config.CAMERA_HEIGHT - 1, pixel_y))
        return pixel_x, pixel_y

    def smooth_point(self, point: tuple[int, int]) -> tuple[int, int]:
        """Smooth small shakes while letting big movements stay fast."""
        if self.smoothed_point is None:
            self.smoothed_point = point
            self.previous_raw_point = point
            return point

        dx = point[0] - self.smoothed_point[0]
        dy = point[1] - self.smoothed_point[1]
        distance = (dx * dx + dy * dy) ** 0.5
        if distance <= config.HAND_CURSOR_DEAD_ZONE:
            self.previous_raw_point = point
            return self.smoothed_point

        predicted_point = point
        if self.previous_raw_point is not None:
            raw_dx = point[0] - self.previous_raw_point[0]
            raw_dy = point[1] - self.previous_raw_point[1]
            predicted_point = (
                point[0] + int(raw_dx * config.HAND_CURSOR_PREDICTION),
                point[1] + int(raw_dy * config.HAND_CURSOR_PREDICTION),
            )

        speed_ratio = min(1.0, distance / config.HAND_CURSOR_FAST_DISTANCE)
        alpha = config.HAND_CURSOR_MIN_SMOOTHING + (
            config.HAND_CURSOR_MAX_SMOOTHING - config.HAND_CURSOR_MIN_SMOOTHING
        ) * speed_ratio
        x = int(self.smoothed_point[0] * (1 - alpha) + predicted_point[0] * alpha)
        y = int(self.smoothed_point[1] * (1 - alpha) + predicted_point[1] * alpha)
        x = max(0, min(config.CAMERA_WIDTH - 1, x))
        y = max(0, min(config.CAMERA_HEIGHT - 1, y))
        self.smoothed_point = (x, y)
        self.previous_raw_point = point
        return self.smoothed_point

    def draw_hand_skeleton(self, frame, hand_landmarks, is_selected: bool) -> None:
        """Draw the chosen hand brightly and other detected hands muted."""
        if is_selected:
            joint_color = config.SELECTED_HAND_JOINT_COLOR
            bone_color = config.SELECTED_HAND_BONE_COLOR
            joint_radius = 4
            bone_thickness = 3
        else:
            joint_color = config.UNSELECTED_HAND_JOINT_COLOR
            bone_color = config.UNSELECTED_HAND_BONE_COLOR
            joint_radius = 2
            bone_thickness = 1

        self.drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.hands_module.HAND_CONNECTIONS,
            landmark_drawing_spec=self.drawing.DrawingSpec(
                color=joint_color,
                thickness=joint_radius,
                circle_radius=joint_radius,
            ),
            connection_drawing_spec=self.drawing.DrawingSpec(
                color=bone_color,
                thickness=bone_thickness,
            ),
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.hands.close()
        self.pose.close()


HandDetector = SkeletonDetector
