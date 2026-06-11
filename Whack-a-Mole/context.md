# Project Name

Whack-a-Mole Vision Game

# Overview

This project is a computer vision game for Science Week.

Players stand in front of a webcam and use their hand to hit moles that appear on the screen.

The game uses:

* OpenCV
* Background Subtraction (MOG2)
* Contour Detection
* Pygame

No MediaPipe.
No YOLO.

The objective is to score as many points as possible within 40 seconds.

---

# Project Structure

whack_a_mole/

├── config.py
├── detector.py
├── game.py
├── leaderboard.csv

Only 3 Python files are allowed.

---

# Gameplay

## Start Screen

Display:

WHACK-A-MOLE

Enter Name:

[Player Name]

Press ENTER to Start

---

## Round

Duration:

40 seconds

Player uses hand movement in front of webcam.

Hand position becomes game cursor.

Moles randomly appear.

When hand cursor overlaps mole:

* Mole disappears
* Score +1
* New mole spawns

---

## Result Screen

Show:

Game Over

Player Name

Final Score

Top 10 Leaderboard

Press R to Restart

Press ESC to Quit

---

# Hand Detection

Use OpenCV only.

Pipeline:

Webcam
↓
Background Subtraction (MOG2)
↓
Threshold
↓
Morphology
↓
Contour Detection
↓
Largest Contour
↓
Bounding Box
↓
Center Point
↓
Cursor Position

The center point of the largest contour is considered the player's hand.

Ignore contours smaller than a minimum area threshold.

---

# Mole System

Create Mole class.

Properties:

* x
* y
* radius
* active

Methods:

* draw()
* respawn()

Respawn position must be random.

Keep mole fully visible on screen.

---

# Collision Detection

Use circle collision.

Formula:

distance = sqrt(
(x1 - x2)^2 +
(y1 - y2)^2
)

If distance <= mole_radius:

Hit detected.

---

# Score System

Hit:

+1 point

Store score in memory during game.

---

# Leaderboard

Use CSV.

Columns:

name,score

When game ends:

Append player score.

Load leaderboard.

Sort descending by score.

Display top 10.

---

# Config File

Store constants only.

Examples:

WINDOW_WIDTH
WINDOW_HEIGHT

FPS

ROUND_TIME

MOLE_RADIUS

MIN_CONTOUR_AREA

Colors

Fonts

---

# Detector File

Create HandDetector class.

Responsibilities:

* OpenCV processing
* MOG2
* Threshold
* Morphology
* Contour detection
* Return hand center position

API:

detector = HandDetector()

x, y = detector.get_hand_position(frame)

---

# Game File

Contains:

* Main menu
* Mole class
* Game loop
* Collision detection
* Score system
* Result screen
* Leaderboard display

Use pygame for rendering.

Keep code readable and beginner-friendly.

The final project should be stable enough to be demonstrated at a university science exhibition.
