from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import pygame

from pose_utils import (
    check_pose,
    detect_pose,
    draw_keypoints,
    get_next_pose,
    load_model,
)
from scoreboard import draw_scoreboard, save_score


CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720
ROUND_SECONDS = 30
SCORE_PER_POSE = 100
MAX_POSES = 30
TARGET_FPS = 30
MATCH_COOLDOWN_SECONDS = 0.65
POSE_HINT_DELAY_SECONDS = 3.0

STATE_NAME_INPUT = "name_input"
STATE_PLAYING = "playing"
STATE_SCOREBOARD = "scoreboard"


@dataclass
class GameSession:
    player_name: str = ""
    score: int = 0
    poses_completed: int = 0
    start_time: float = 0.0
    last_match_time: float = 0.0
    current_pose_id: int = 1
    pose_started_time: float = 0.0
    saved_entry: Optional[dict] = None

    def reset_for_player(self, player_name: str) -> None:
        self.player_name = player_name.strip() or "Player"
        self.score = 0
        self.poses_completed = 0
        self.start_time = time.time()
        self.last_match_time = 0.0
        self.current_pose_id = 1
        self.pose_started_time = self.start_time
        self.saved_entry = None

    @property
    def current_pose(self):
        return get_next_pose(self.current_pose_id - 1)

    def time_left(self) -> int:
        elapsed = int(time.time() - self.start_time)
        return max(0, ROUND_SECONDS - elapsed)

    def complete_pose(self) -> None:
        self.poses_completed += 1
        self.score = min(MAX_POSES * SCORE_PER_POSE, self.score + SCORE_PER_POSE)
        self.current_pose_id += 1
        self.last_match_time = time.time()
        self.pose_started_time = self.last_match_time

    def finished(self) -> bool:
        return self.time_left() <= 0 or self.poses_completed >= MAX_POSES


class PoseMatchGame:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Pose Match")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.fonts = self._create_fonts()
        self.capture = self._open_camera()
        self.model = load_model()
        self.session = GameSession()
        self.state = STATE_NAME_INPUT
        self.name_buffer = ""
        self.last_frame_surface: Optional[pygame.Surface] = None
        self.running = True

    def run(self) -> None:
        while self.running:
            self._handle_events()
            if self.state == STATE_NAME_INPUT:
                self._draw_name_input()
            elif self.state == STATE_PLAYING:
                self._update_gameplay()
            elif self.state == STATE_SCOREBOARD:
                draw_scoreboard(self.screen, self.fonts, self.session.saved_entry or {})

            pygame.display.flip()
            self.clock.tick(TARGET_FPS)

        self._shutdown()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event)

    def _handle_keydown(self, event: pygame.event.Event) -> None:
        if self.state == STATE_NAME_INPUT:
            if event.key == pygame.K_RETURN and self.name_buffer.strip():
                self.session.reset_for_player(self.name_buffer)
                self.state = STATE_PLAYING
            elif event.key == pygame.K_BACKSPACE:
                self.name_buffer = self.name_buffer[:-1]
            elif event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.unicode and len(self.name_buffer) < 16 and event.unicode.isprintable():
                self.name_buffer += event.unicode
        elif self.state == STATE_SCOREBOARD:
            if event.key == pygame.K_r:
                self.name_buffer = self.session.player_name
                self.session.reset_for_player(self.name_buffer)
                self.state = STATE_PLAYING
            elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                self.running = False
        elif self.state == STATE_PLAYING and event.key == pygame.K_ESCAPE:
            self._finish_game()

    def _update_gameplay(self) -> None:
        success, frame = self.capture.read()
        if not success:
            self._draw_camera_error()
            return

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
        keypoints, _ = detect_pose(frame, self.model)
        matched = check_pose(keypoints, self.session.current_pose)

        if matched and time.time() - self.session.last_match_time >= MATCH_COOLDOWN_SECONDS:
            self.session.complete_pose()

        draw_keypoints(frame, keypoints)
        self.last_frame_surface = self._frame_to_surface(frame)
        self._draw_gameplay(keypoints_detected=bool(keypoints), matched=matched)

        if self.session.finished():
            self._finish_game()

    def _finish_game(self) -> None:
        if self.session.saved_entry is None:
            self.session.saved_entry = save_score(
                self.session.player_name,
                self.session.score,
                self.session.poses_completed,
            )
        self.state = STATE_SCOREBOARD

    def _draw_name_input(self) -> None:
        self.screen.fill((14, 18, 24))
        self._draw_center("Pose Match", self.fonts["title"], 120, (255, 255, 255))
        self._draw_center("Nickname Input", self.fonts["large"], 220, (0, 220, 255))

        input_rect = pygame.Rect(260, 295, 440, 58)
        pygame.draw.rect(self.screen, (28, 34, 44), input_rect, border_radius=6)
        pygame.draw.rect(self.screen, (0, 220, 255), input_rect, width=2, border_radius=6)
        name_text = self.name_buffer or "Type nickname"
        color = (255, 255, 255) if self.name_buffer else (120, 130, 145)
        self.screen.blit(self.fonts["medium"].render(name_text, True, color), (input_rect.x + 18, input_rect.y + 15))

        self._draw_center("Press Enter To Start", self.fonts["medium"], 410, (220, 226, 232))

    def _draw_gameplay(self, keypoints_detected: bool, matched: bool) -> None:
        self.screen.fill((8, 10, 14))
        if self.last_frame_surface:
            self.screen.blit(self.last_frame_surface, (0, 0))

        panel = pygame.Rect(0, CAMERA_HEIGHT, WINDOW_WIDTH, WINDOW_HEIGHT - CAMERA_HEIGHT)
        pygame.draw.rect(self.screen, (14, 18, 24), panel)

        pose = self.session.current_pose
        status = "MATCHED" if matched else ("DETECTING" if keypoints_detected else "NO BODY DETECTED")
        status_color = (70, 255, 140) if matched else ((255, 220, 90) if keypoints_detected else (255, 110, 110))
        show_pose_hint = time.time() - self.session.pose_started_time >= POSE_HINT_DELAY_SECONDS

        self.screen.blit(self.fonts["medium"].render(f"Score: {self.session.score}", True, (255, 255, 255)), (24, 566))
        self.screen.blit(self.fonts["medium"].render(f"Time: {self.session.time_left()}", True, (255, 255, 255)), (250, 566))
        self.screen.blit(self.fonts["medium"].render(f"Pose: {self.session.current_pose_id}/30", True, (255, 255, 255)), (430, 566))
        self.screen.blit(self.fonts["small"].render(status, True, status_color), (760, 570))

        self.screen.blit(self.fonts["small"].render("Target Pose:", True, (170, 180, 195)), (24, 625))
        self.screen.blit(self.fonts["large"].render(pose.name, True, (0, 220, 255)), (24, 648))
        if show_pose_hint:
            self._draw_pose_preview(pose, pygame.Rect(680, 590, 240, 112))
        else:
            self._draw_hint_waiting(pygame.Rect(680, 590, 240, 112))

    def _draw_hint_waiting(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, (24, 30, 38), rect, border_radius=6)
        pygame.draw.rect(self.screen, (70, 82, 96), rect, width=1, border_radius=6)
        wait_left = max(
            0,
            int(POSE_HINT_DELAY_SECONDS - (time.time() - self.session.pose_started_time)) + 1,
        )
        self.screen.blit(
            self.fonts["small"].render("Example hidden", True, (170, 180, 195)),
            (rect.x + 12, rect.y + 22),
        )
        self.screen.blit(
            self.fonts["small"].render(f"Hint in {wait_left}s", True, (0, 220, 255)),
            (rect.x + 12, rect.y + 56),
        )

    def _draw_pose_preview(self, pose, rect: pygame.Rect) -> None:
        # SAMPLE POSE IMAGE
        # This draws the small example pose shown to the player during gameplay.
        # It appears only after the player has not matched the current pose for 3 seconds.
        # It is only a visual guide and is not used to judge whether the answer is correct.
        pygame.draw.rect(self.screen, (24, 30, 38), rect, border_radius=6)
        pygame.draw.rect(self.screen, (70, 82, 96), rect, width=1, border_radius=6)
        self.screen.blit(
            self.fonts["small"].render("Example", True, (170, 180, 195)),
            (rect.x + 12, rect.y + 8),
        )

        points = self._preview_points(pose.pose_id)
        skeleton = [
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

        def to_screen(point_name: str) -> tuple[int, int]:
            x, y = points[point_name]
            return (rect.x + int(x * rect.width), rect.y + int(y * rect.height))

        for start, end in skeleton:
            pygame.draw.line(self.screen, (0, 220, 255), to_screen(start), to_screen(end), 4)
        pygame.draw.circle(self.screen, (255, 255, 255), to_screen("nose"), 9)

        for name in points:
            if name != "nose":
                pygame.draw.circle(self.screen, (245, 245, 245), to_screen(name), 4)

    def _preview_points(self, pose_id: int) -> dict[str, tuple[float, float]]:
        # SAMPLE POSE DATA
        # These normalized stick-figure points create the example image only.
        # The real answer checking is in pose_utils.py inside POSE_TEMPLATES.
        base = {
            "nose": (0.50, 0.18),
            "left_shoulder": (0.38, 0.32),
            "right_shoulder": (0.62, 0.32),
            "left_elbow": (0.34, 0.50),
            "right_elbow": (0.66, 0.50),
            "left_wrist": (0.32, 0.70),
            "right_wrist": (0.68, 0.70),
            "left_hip": (0.43, 0.58),
            "right_hip": (0.57, 0.58),
            "left_knee": (0.42, 0.78),
            "right_knee": (0.58, 0.78),
            "left_ankle": (0.40, 0.96),
            "right_ankle": (0.60, 0.96),
        }
        presets = {
            1: {"left_elbow": (0.32, 0.18), "left_wrist": (0.30, 0.02), "right_elbow": (0.68, 0.18), "right_wrist": (0.70, 0.02)},
            2: {"left_elbow": (0.22, 0.32), "left_wrist": (0.08, 0.32), "right_elbow": (0.78, 0.32), "right_wrist": (0.92, 0.32)},
            3: {"left_elbow": (0.32, 0.16), "left_wrist": (0.30, 0.02)},
            4: {"right_elbow": (0.68, 0.16), "right_wrist": (0.70, 0.02)},
            5: {"left_elbow": (0.35, 0.10), "left_wrist": (0.48, 0.03), "right_elbow": (0.65, 0.10), "right_wrist": (0.52, 0.03)},
            6: {"left_elbow": (0.32, 0.25), "left_wrist": (0.43, 0.18), "right_elbow": (0.68, 0.25), "right_wrist": (0.57, 0.18)},
            7: {"left_elbow": (0.32, 0.25), "left_wrist": (0.43, 0.18)},
            8: {"right_elbow": (0.68, 0.25), "right_wrist": (0.57, 0.18)},
            9: {"left_elbow": (0.35, 0.48), "left_wrist": (0.43, 0.58)},
            10: {"right_elbow": (0.65, 0.48), "right_wrist": (0.57, 0.58)},
            11: {"left_elbow": (0.35, 0.48), "left_wrist": (0.43, 0.58), "right_elbow": (0.65, 0.48), "right_wrist": (0.57, 0.58)},
            12: {"left_elbow": (0.24, 0.22), "left_wrist": (0.12, 0.08), "right_elbow": (0.76, 0.22), "right_wrist": (0.88, 0.08)},
            13: {"left_elbow": (0.36, 0.48), "left_wrist": (0.34, 0.76), "right_elbow": (0.64, 0.48), "right_wrist": (0.66, 0.76)},
            14: {"left_elbow": (0.22, 0.32), "left_wrist": (0.08, 0.32)},
            15: {"right_elbow": (0.78, 0.32), "right_wrist": (0.92, 0.32)},
            16: {"left_elbow": (0.28, 0.43), "left_wrist": (0.34, 0.28)},
            17: {"right_elbow": (0.72, 0.43), "right_wrist": (0.66, 0.28)},
            18: {"left_elbow": (0.38, 0.45), "left_wrist": (0.61, 0.44), "right_elbow": (0.62, 0.45), "right_wrist": (0.39, 0.44)},
            19: {"left_knee": (0.37, 0.60), "left_ankle": (0.30, 0.78)},
            20: {"right_knee": (0.63, 0.60), "right_ankle": (0.70, 0.78)},
            21: {"left_hip": (0.40, 0.64), "right_hip": (0.60, 0.64), "left_knee": (0.32, 0.78), "right_knee": (0.68, 0.78), "left_ankle": (0.26, 0.96), "right_ankle": (0.74, 0.96)},
            22: {},
            23: {"nose": (0.38, 0.18), "left_shoulder": (0.30, 0.32), "right_shoulder": (0.54, 0.32)},
            24: {"nose": (0.62, 0.18), "left_shoulder": (0.46, 0.32), "right_shoulder": (0.70, 0.32)},
            25: {"left_elbow": (0.20, 0.35), "left_wrist": (0.04, 0.35)},
            26: {"right_elbow": (0.80, 0.35), "right_wrist": (0.96, 0.35)},
            27: {"left_elbow": (0.22, 0.20), "left_wrist": (0.10, 0.15), "right_elbow": (0.78, 0.20), "right_wrist": (0.90, 0.15)},
            28: {"left_elbow": (0.43, 0.33), "left_wrist": (0.49, 0.28), "right_elbow": (0.57, 0.33), "right_wrist": (0.51, 0.28)},
            29: {"left_elbow": (0.30, 0.18), "left_wrist": (0.20, 0.04), "right_elbow": (0.66, 0.60), "right_wrist": (0.72, 0.86)},
            30: {"left_elbow": (0.30, 0.18), "left_wrist": (0.22, 0.04), "right_elbow": (0.70, 0.18), "right_wrist": (0.78, 0.04)},
        }
        base.update(presets.get(pose_id, {}))
        return base

    def _draw_camera_error(self) -> None:
        self.screen.fill((14, 18, 24))
        self._draw_center("Camera not available", self.fonts["large"], 310, (255, 110, 110))
        self._draw_center("Press ESC to save and leave gameplay", self.fonts["small"], 360, (220, 226, 232))

    def _draw_center(self, text: str, font: pygame.font.Font, y: int, color: tuple[int, int, int]) -> None:
        surface = font.render(text, True, color)
        rect = surface.get_rect(center=(WINDOW_WIDTH // 2, y))
        self.screen.blit(surface, rect)

    def _frame_to_surface(self, frame) -> pygame.Surface:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return pygame.image.frombuffer(rgb.tobytes(), rgb.shape[1::-1], "RGB")

    def _create_fonts(self) -> Dict[str, pygame.font.Font]:
        return {
            "title": pygame.font.SysFont("arial", 64, bold=True),
            "large": pygame.font.SysFont("arial", 36, bold=True),
            "medium": pygame.font.SysFont("arial", 26, bold=True),
            "small": pygame.font.SysFont("consolas", 20),
        }

    def _open_camera(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        capture.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        return capture

    def _shutdown(self) -> None:
        self.capture.release()
        pygame.quit()
        sys.exit()


def main() -> None:
    game = PoseMatchGame()
    game.run()


if __name__ == "__main__":
    main()
