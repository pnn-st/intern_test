"""Whack-a-Mole Vision Game.

Run this file with Python 3.11+:
    python game.py
"""

from __future__ import annotations

import csv
import os
import random
import time
from dataclasses import dataclass

import cv2
import pygame

import config
from detector import SkeletonDetector


@dataclass
class Mole:
    """A mole target that can be hit by the player's detected hand."""

    x: int
    y: int
    radius: int
    created_at: float
    visible_seconds: float
    hit: bool = False

    def is_expired(self, now: float) -> bool:
        return now - self.created_at >= self.visible_seconds

    def contains_point(self, point: tuple[int, int]) -> bool:
        px, py = point
        distance_squared = (self.x - px) ** 2 + (self.y - py) ** 2
        return distance_squared <= self.radius**2

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.ellipse(
            surface,
            config.HOLE_COLOR,
            (
                self.x - self.radius - 12,
                self.y + self.radius // 3,
                (self.radius + 12) * 2,
                self.radius,
            ),
        )
        color = config.MOLE_HIT_COLOR if self.hit else config.MOLE_COLOR
        pygame.draw.circle(surface, color, (self.x, self.y), self.radius)
        pygame.draw.circle(surface, (92, 58, 38), (self.x - 14, self.y - 10), 5)
        pygame.draw.circle(surface, (92, 58, 38), (self.x + 14, self.y - 10), 5)
        pygame.draw.arc(
            surface,
            (65, 38, 28),
            (self.x - 18, self.y - 2, 36, 22),
            0,
            3.14,
            3,
        )


class Leaderboard:
    """Small CSV-backed score table."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.ensure_file()

    def ensure_file(self) -> None:
        if os.path.exists(self.filename):
            return
        with open(self.filename, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["name", "score", "date"])

    def add_score(self, name: str, score: int) -> None:
        clean_name = name.strip() or "Player"
        with open(self.filename, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([clean_name[:18], score, time.strftime("%Y-%m-%d %H:%M")])

    def top_scores(self, limit: int = config.LEADERBOARD_LIMIT) -> list[dict[str, str]]:
        self.ensure_file()
        with open(self.filename, "r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        rows.sort(key=lambda row: int(row.get("score", 0)), reverse=True)
        return rows[:limit]


class Button:
    """Simple pygame button used by menu and result screens."""

    def __init__(self, rect: pygame.Rect, text: str) -> None:
        self.rect = rect
        self.text = text

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        mouse_pos = pygame.mouse.get_pos()
        color = config.BUTTON_HOVER_COLOR if self.rect.collidepoint(mouse_pos) else config.BUTTON_COLOR
        pygame.draw.rect(surface, color, self.rect, border_radius=8)
        label = font.render(self.text, True, config.TEXT_COLOR)
        label_rect = label.get_rect(center=self.rect.center)
        surface.blit(label, label_rect)

    def was_clicked(self, event: pygame.event.Event) -> bool:
        return (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self.rect.collidepoint(event.pos)
        )


class WhackAMoleGame:
    """Main application controller."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption(config.WINDOW_TITLE)
        self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        self.title_font = pygame.font.Font(config.FONT_NAME, 54)
        self.large_font = pygame.font.Font(config.FONT_NAME, 38)
        self.medium_font = pygame.font.Font(config.FONT_NAME, 28)
        self.small_font = pygame.font.Font(config.FONT_NAME, 20)

        self.leaderboard = Leaderboard(config.LEADERBOARD_FILE)
        self.detector = SkeletonDetector()
        self.camera = cv2.VideoCapture(config.CAMERA_INDEX)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

        self.running = True
        self.state = "menu"
        self.player_name = ""
        self.score = 0
        self.moles: list[Mole] = []
        self.next_spawn_time = 0.0
        self.game_started_at = 0.0
        self.last_camera_surface: pygame.Surface | None = None
        self.hand_point: tuple[int, int] | None = None
        self.score_saved = False

        self.play_area = pygame.Rect(0, 0, config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        self.start_button = Button(pygame.Rect(380, 430, 200, 58), "Start")
        self.quit_button = Button(pygame.Rect(380, 505, 200, 58), "Quit")
        self.play_again_button = Button(pygame.Rect(350, 500, 260, 58), "Play Again")

    def run(self) -> None:
        while self.running:
            if self.state == "menu":
                self.handle_menu()
            elif self.state == "playing":
                self.handle_gameplay()
            elif self.state == "result":
                self.handle_result()

        self.shutdown()

    def start_game(self) -> None:
        self.score = 0
        self.moles = []
        self.hand_point = None
        self.score_saved = False
        self.game_started_at = time.time()
        self.next_spawn_time = self.game_started_at + 0.5
        self.state = "playing"

    def handle_menu(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    self.player_name = self.player_name[:-1]
                elif event.key == pygame.K_RETURN:
                    self.start_game()
                elif len(self.player_name) < 18 and event.unicode.isprintable():
                    self.player_name += event.unicode
            elif self.start_button.was_clicked(event):
                self.start_game()
            elif self.quit_button.was_clicked(event):
                self.running = False

        self.draw_menu()
        pygame.display.flip()
        self.clock.tick(config.FPS)

    def handle_gameplay(self) -> None:
        now = time.time()
        elapsed = now - self.game_started_at
        remaining = max(0, int(config.GAME_DURATION_SECONDS - elapsed))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.state = "menu"

        self.update_camera()
        self.spawn_moles(now)
        self.update_moles(now)
        self.check_hits()

        if elapsed >= config.GAME_DURATION_SECONDS:
            self.save_result_once()
            self.state = "result"

        self.draw_gameplay(remaining)
        pygame.display.flip()
        self.clock.tick(config.FPS)

    def handle_result(self) -> None:
        self.save_result_once()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                self.start_game()
            elif self.play_again_button.was_clicked(event):
                self.start_game()

        self.draw_result()
        pygame.display.flip()
        self.clock.tick(config.FPS)

    def update_camera(self) -> None:
        success, frame = self.camera.read()
        if not success:
            self.hand_point = None
            return

        processed_frame, center, _contour, _mask = self.detector.detect(frame)
        self.last_camera_surface = self.cv_frame_to_surface(processed_frame)

        if center is None:
            self.hand_point = None
            return

        hand_x = int(center[0] / config.CAMERA_WIDTH * self.play_area.width)
        hand_y = int(center[1] / config.CAMERA_HEIGHT * self.play_area.height)
        self.hand_point = self.apply_cursor_gain((hand_x, hand_y))

    def apply_cursor_gain(self, point: tuple[int, int]) -> tuple[int, int]:
        """Increase cursor travel from the screen center for faster aiming."""
        center_x = self.play_area.centerx
        center_y = self.play_area.centery
        x = center_x + int((point[0] - center_x) * config.GAME_CURSOR_GAIN)
        y = center_y + int((point[1] - center_y) * config.GAME_CURSOR_GAIN)
        x = max(self.play_area.left, min(self.play_area.right - 1, x))
        y = max(self.play_area.top, min(self.play_area.bottom - 1, y))
        return x, y

    def cv_frame_to_surface(self, frame) -> pygame.Surface:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = rgb_frame.swapaxes(0, 1)
        return pygame.surfarray.make_surface(rgb_frame)

    def spawn_moles(self, now: float) -> None:
        if now < self.next_spawn_time:
            return

        margin = config.MOLE_RADIUS + 40
        x = random.randint(margin, config.SCREEN_WIDTH - margin)
        y = random.randint(145, config.SCREEN_HEIGHT - margin)
        self.moles.append(
            Mole(
                x=x,
                y=y,
                radius=config.MOLE_RADIUS,
                created_at=now,
                visible_seconds=config.MOLE_VISIBLE_SECONDS,
            )
        )
        self.next_spawn_time = now + random.uniform(
            config.MOLE_SPAWN_INTERVAL_MIN,
            config.MOLE_SPAWN_INTERVAL_MAX,
        )

    def update_moles(self, now: float) -> None:
        self.moles = [mole for mole in self.moles if not mole.is_expired(now) and not mole.hit]

    def check_hits(self) -> None:
        if self.hand_point is None:
            return

        for mole in self.moles:
            if mole.contains_point(self.hand_point):
                mole.hit = True
                self.score += config.MOLE_POINTS
                break

    def save_result_once(self) -> None:
        if self.score_saved:
            return
        self.leaderboard.add_score(self.player_name, self.score)
        self.score_saved = True

    def draw_menu(self) -> None:
        self.screen.fill(config.BACKGROUND_COLOR)
        self.draw_title("Whack-a-Mole", "Vision Game")

        name_label = self.medium_font.render("Player name", True, config.MUTED_TEXT_COLOR)
        self.screen.blit(name_label, (330, 285))

        input_rect = pygame.Rect(330, 320, 300, 54)
        pygame.draw.rect(self.screen, config.PANEL_COLOR, input_rect, border_radius=8)
        pygame.draw.rect(self.screen, config.ACCENT_COLOR, input_rect, 2, border_radius=8)
        name_text = self.medium_font.render(self.player_name or "Player", True, config.TEXT_COLOR)
        self.screen.blit(name_text, (input_rect.x + 16, input_rect.y + 12))

        self.start_button.draw(self.screen, self.medium_font)
        self.quit_button.draw(self.screen, self.medium_font)
        self.draw_leaderboard(650, 300)

    def draw_gameplay(self, remaining: int) -> None:
        self.screen.fill(config.BACKGROUND_COLOR)

        for mole in self.moles:
            mole.draw(self.screen)

        if self.hand_point is not None:
            pygame.draw.circle(self.screen, config.HAND_CURSOR_COLOR, self.hand_point, 18, 4)
            pygame.draw.circle(self.screen, config.HAND_CURSOR_COLOR, self.hand_point, 4)

        self.draw_hud(remaining)
        self.draw_camera_preview()

    def draw_result(self) -> None:
        self.screen.fill(config.BACKGROUND_COLOR)
        self.draw_title("Game Over", f"Score: {self.score}")
        self.draw_leaderboard(350, 300)
        self.play_again_button.draw(self.screen, self.medium_font)

    def draw_title(self, line_one: str, line_two: str) -> None:
        title = self.title_font.render(line_one, True, config.TEXT_COLOR)
        subtitle = self.large_font.render(line_two, True, config.ACCENT_COLOR)
        self.screen.blit(title, title.get_rect(center=(config.SCREEN_WIDTH // 2, 120)))
        self.screen.blit(subtitle, subtitle.get_rect(center=(config.SCREEN_WIDTH // 2, 180)))

    def draw_hud(self, remaining: int) -> None:
        hud_rect = pygame.Rect(20, 18, 420, 72)
        pygame.draw.rect(self.screen, config.PANEL_COLOR, hud_rect, border_radius=8)
        score_text = self.medium_font.render(f"Score: {self.score}", True, config.TEXT_COLOR)
        time_text = self.medium_font.render(f"Time: {remaining}", True, config.ACCENT_COLOR)
        self.screen.blit(score_text, (42, 40))
        self.screen.blit(time_text, (260, 40))

    def draw_camera_preview(self) -> None:
        preview_rect = pygame.Rect(config.SCREEN_WIDTH - 230, 18, 210, 158)
        pygame.draw.rect(self.screen, config.PANEL_COLOR, preview_rect, border_radius=8)

        if self.last_camera_surface is None:
            label = self.small_font.render("Camera loading", True, config.MUTED_TEXT_COLOR)
            self.screen.blit(label, label.get_rect(center=preview_rect.center))
            return

        preview = pygame.transform.scale(self.last_camera_surface, (200, 150))
        self.screen.blit(preview, (preview_rect.x + 5, preview_rect.y + 4))

    def draw_leaderboard(self, x: int, y: int) -> None:
        title = self.medium_font.render("Top Scores", True, config.ACCENT_COLOR)
        self.screen.blit(title, (x, y))

        scores = self.leaderboard.top_scores()
        if not scores:
            empty = self.small_font.render("No scores yet", True, config.MUTED_TEXT_COLOR)
            self.screen.blit(empty, (x, y + 42))
            return

        for index, row in enumerate(scores, start=1):
            name = row.get("name", "Player")
            score = row.get("score", "0")
            line = self.small_font.render(f"{index}. {name} - {score}", True, config.TEXT_COLOR)
            self.screen.blit(line, (x, y + 36 + index * 30))

    def shutdown(self) -> None:
        self.detector.close()
        if self.camera.isOpened():
            self.camera.release()
        pygame.quit()


if __name__ == "__main__":
    WhackAMoleGame().run()
