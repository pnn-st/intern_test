from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pygame


SCORE_FILE = Path("scores.json")
ScoreEntry = Dict[str, object]


def load_scores() -> List[ScoreEntry]:
    if not SCORE_FILE.exists():
        return []
    try:
        data = json.loads(SCORE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, list):
        return []
    return [entry for entry in data if isinstance(entry, dict)]


def save_score(name: str, score: int, poses_completed: int) -> ScoreEntry:
    entry: ScoreEntry = {
        "name": name.strip() or "Player",
        "score": int(score),
        "poses_completed": int(poses_completed),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    scores = load_scores()
    scores.append(entry)
    scores.sort(key=lambda item: int(item.get("score", 0)), reverse=True)
    SCORE_FILE.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    return entry


def get_top_scores(limit: int = 10) -> List[ScoreEntry]:
    scores = load_scores()
    scores.sort(key=lambda item: int(item.get("score", 0)), reverse=True)
    return scores[:limit]


def get_player_rank(player_entry: ScoreEntry) -> int:
    scores = load_scores()
    scores.sort(key=lambda item: int(item.get("score", 0)), reverse=True)
    target_timestamp = player_entry.get("timestamp")
    for index, entry in enumerate(scores, start=1):
        if entry.get("timestamp") == target_timestamp:
            return index
    return len(scores)


def draw_scoreboard(
    screen: pygame.Surface,
    fonts: Dict[str, pygame.font.Font],
    player_entry: ScoreEntry,
) -> None:
    screen.fill((14, 18, 24))
    width, _ = screen.get_size()
    rank = get_player_rank(player_entry)
    top_scores = get_top_scores(8)

    _draw_center(screen, fonts["title"], "Game Over", 70, (255, 255, 255))
    _draw_center(
        screen,
        fonts["large"],
        f"{player_entry.get('name', 'Player')}  Score: {player_entry.get('score', 0)}",
        140,
        (0, 220, 255),
    )
    _draw_center(screen, fonts["medium"], f"Ranking: #{rank}", 190, (235, 235, 235))

    header_y = 255
    screen.blit(fonts["medium"].render("Top Scores", True, (255, 255, 255)), (width // 2 - 180, header_y))
    for index, entry in enumerate(top_scores, start=1):
        y = header_y + 42 + index * 34
        line = (
            f"{index:>2}. {str(entry.get('name', 'Player'))[:14]:<14}"
            f" {int(entry.get('score', 0)):>4}"
        )
        color = (0, 220, 255) if entry.get("timestamp") == player_entry.get("timestamp") else (220, 226, 232)
        screen.blit(fonts["small"].render(line, True, color), (width // 2 - 220, y))

    _draw_center(screen, fonts["small"], "R = Restart     Q = Quit", 680, (190, 195, 205))


def _draw_center(
    screen: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    y: int,
    color: tuple[int, int, int],
) -> None:
    surface = font.render(text, True, color)
    rect = surface.get_rect(center=(screen.get_width() // 2, y))
    screen.blit(surface, rect)
