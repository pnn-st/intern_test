# Pose Match Game

## Project Information

Project Name:
Pose Match

Project Type:
Computer Vision Game

Main Goal:
Create a pose matching game using OpenCV, YOLO Pose, Image Processing, and Pygame.

The game will be used as an educational project for learning:

- Image Processing
- OpenCV
- YOLO Pose
- Computer Vision
- Game Development
- Human Pose Estimation

---

# Technology Stack

## Required

Python

OpenCV

Pygame

YOLO Pose

NumPy

JSON

---

# File Constraints

Maximum Python files:

3 files only

Required files:

main.py

pose_utils.py

scoreboard.py

No additional Python files.

---

# Gameplay

The player must perform body poses shown by the game.

The webcam captures the player's body.

YOLO Pose detects body keypoints.

The game compares the player's pose against the target pose.

If the pose is correct:

- Add score
- Move to next pose

---

# Game Flow

## Step 1

Launch game

## Step 2

Show player name input screen

Player enters nickname

Press ENTER

## Step 3

Start gameplay immediately

No Start Menu

No Main Menu

No Settings Menu

## Step 4

Play for 30 seconds

## Step 5

End game

## Step 6

Save score

## Step 7

Show scoreboard

---

# Game Rules

Total Time:
30 seconds

Total Poses:
30 poses

Score Per Pose:
100 points

Maximum Score:
3000 points

If player completes all poses:

End game immediately

Score = 3000

If timer reaches zero:

End game immediately

---

# Pose System

The game contains 30 predefined poses.

Each pose has:

- Name
- Pose ID
- Validation logic

Example:

Pose 1:
Hands Up

Pose 2:
T Pose

Pose 3:
Left Hand Up

Pose 4:
Right Hand Up

Pose 5:
Heart Pose

...

Pose 30:
Victory Pose

The architecture must allow adding more poses later.

---

# Pose Detection

Use YOLO Pose.

Detect keypoints:

- Nose
- Eyes
- Ears
- Shoulders
- Elbows
- Wrists
- Hips
- Knees
- Ankles

The system must:

1. Detect keypoints
2. Calculate body angles
3. Compare with pose template
4. Return True or False

---

# Matching Strategy

Each pose should be represented as a template.

Example:

{
"left_arm_angle": 170,
"right_arm_angle": 170,
"left_leg_angle": 180,
"right_leg_angle": 180
}

Use tolerance thresholds.

Example:

±15 degrees

If all conditions pass:

Return matched

Otherwise:

Return not matched

---

# User Interface

## Name Input Screen

Show:

Pose Match

Nickname Input

Press Enter To Start

---

## Gameplay Screen

Show:

Webcam Feed

Current Pose Name

Current Score

Remaining Time

Pose Progress

Example:

Score: 500

Time: 22

Pose: 5/30

Target Pose:
T Pose

---

## Scoreboard Screen

Show:

Game Over

Player Name

Final Score

Player Ranking

Top Scores

Controls:

R = Restart

Q = Quit

---

# Scoreboard System

Use:

scores.json

Example:

[
{
"name": "John",
"score": 1500,
"poses_completed": 15
}
]

Store:

Player Name

Score

Poses Completed

Timestamp

Sort by score descending.

---

# File Responsibilities

## main.py

Responsibilities:

- Pygame initialization
- OpenCV camera
- Game loop
- State management
- Name input
- Gameplay
- Timer
- Drawing UI
- Calling pose detection
- Calling scoreboard

---

## pose_utils.py

Responsibilities:

- YOLO Pose model loading
- Keypoint extraction
- Angle calculations
- Pose matching
- Pose templates
- Pose validation

Required Functions:

calculate_angle()

detect_pose()

check_pose()

get_next_pose()

---

## scoreboard.py

Responsibilities:

load_scores()

save_score()

get_top_scores()

get_player_rank()

draw_scoreboard()

---

# Development Roadmap

Phase 1

Open webcam

Display webcam in pygame

---

Phase 2

YOLO Pose integration

Display keypoints

---

Phase 3

Angle calculation

Pose validation

---

Phase 4

Scoring system

---

Phase 5

Timer system

---

Phase 6

Scoreboard system

---

Phase 7

Add all 30 poses

---

# Performance Requirements

Maintain smooth webcam feed

Maintain responsive pygame loop

Avoid unnecessary computations

Process only one frame at a time

Target FPS:

30+

---

# MVP Requirements

The first version must:

- Open webcam
- Detect pose
- Match poses
- Give score
- Run timer
- Save scores
- Show scoreboard

The game must be playable before adding advanced features.
