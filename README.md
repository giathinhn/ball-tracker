# Ball Tracker ⚽

## 📌 Introduction

This repository contains the source code, final report, and presentation slides for the Computer Vision course project at **University of Science, VNU-HCM (HCMUS)**.

The project proposes a robust, modular two-pass pipeline for **detecting and tracking fast-moving sports balls** (such as footballs, tennis balls, and basketballs) in broadcast sports videos.

## 👥 Group Members (Class: CV-TH)

1.  **Hoàng Kim Trí** - MSSV: `23120098`
2.  **Đỗ Trần Minh Phúc** - MSSV: `23120156`
3.  **Nguyễn Gia Thịnh** - MSSV: `23120167`

## 📂 Codebase Structure

The project separates the CLI/IO logic from the core computer vision processing pipelines:

- **[main.py](file:///d:/ball-tracker/main.py)**: The main entrypoint. Handles command-line arguments, processes the video frames, writes the outputs, and manages the CV2 display window.
- **[sports/pipelines.py](file:///d:/ball-tracker/sports/pipelines.py)**: The processing hub. Houses the `Mode` enum, annotator instances, utility helpers (`get_crops`, `resolve_goalkeepers_team_id`), and all detection, tracking, classification, and radar pipelines.

## 💻 install

We don't have a Python package yet. Install from source in a
[**Python>=3.8**](https://www.python.org/) environment.

**Step 1 — Navigate to the main source code folder**

```bash
cd ball-tracker
```

**Step 2 — Create a virtual environment (recommended)**

```bash
# Tạo môi trường
python -m venv .venv

# Create environment
# - Windows (PowerShell):
.venv\Scripts\Activate.ps1
# - Linux/macOS:
source .venv/bin/activate
```

**Step 3 — Install Dependencies**

```bash
pip install -r requirements.txt
```

**Step 4 - Download .pt**

- **Windows (PowerShell/CMD):**
  ```cmd
  .\setup.bat
  ```
- **Linux/macOS:**
  ```bash
  ./setup.sh
  ```

## 🛠️ modes

> [!TIP]
> The `--device` parameter specifies which hardware to run the model on. You can use:
> - `cpu` (default, works on any machine)
> - `cuda` (if you have an NVIDIA GPU on Windows/Linux)
> - `mps` (if you are on macOS Apple Silicon)

- `PITCH_DETECTION` - Detects the soccer field boundaries and key points in the video.
  Useful for identifying and visualizing the layout of the soccer pitch.

  ```bash
  python main.py --source_video_path data/demo.mp4 --target_video_path data/output-pitch-detection.mp4 --device cpu --mode PITCH_DETECTION
  ```

  https://github.com/user-attachments/assets/cf4df75a-89fe-4c6f-b3dc-e4d63a0ed211

- `PLAYER_DETECTION` - Detects players, goalkeepers, referees, and the ball in the
  video. Essential for identifying and tracking the presence of players and other
  entities on the field.

  ```bash
  python main.py --source_video_path data/demo.mp4 --target_video_path data/output-player-detection.mp4 --device cpu --mode PLAYER_DETECTION
  ```

  https://github.com/user-attachments/assets/c36ea2c1-b03e-4ffe-81bd-27391260b187

- `BALL_DETECTION` - Detects the ball in the video frames and tracks its position.
  Useful for following ball movements throughout the match.

  ```bash
  python main.py --source_video_path data/demo.mp4 --target_video_path data/output-ball-detection.mp4 --device cpu --mode BALL_DETECTION
  ```

  https://github.com/user-attachments/assets/2fd83678-7790-4f4d-a8c0-065ef38ca031

- `PLAYER_TRACKING` - Tracks players across video frames, maintaining consistent
  identification. Useful for following player movements and positions throughout the
  match.

  ```bash
  python main.py --source_video_path data/demo.mp4 --target_video_path data/output-player-tracking.mp4 --device cpu --mode PLAYER_TRACKING
  ```

  https://github.com/user-attachments/assets/69be83ac-52ff-4879-b93d-33f016feb839

- `TEAM_CLASSIFICATION` - Classifies detected players into their respective teams based
  on their visual features. Helps differentiate between players of different teams for
  analysis and visualization.

  ```bash
  python main.py --source_video_path data/demo.mp4 --target_video_path data/output-team-classification.mp4 --device cpu --mode TEAM_CLASSIFICATION
  ```

  https://github.com/user-attachments/assets/239c2960-5032-415c-b330-3ddd094d32c7

- `RADAR` - Combines pitch detection, player detection, tracking, and team
  classification to generate a radar-like visualization of player positions on the
  soccer field. Provides a comprehensive overview of player movements and team formations
  on the field.

  ```bash
  python main.py --source_video_path data/demo.mp4 --target_video_path data/output-radar.mp4 --device cpu --mode RADAR
  ```

  https://github.com/user-attachments/assets/263b4cd0-2185-4ed3-9be2-cf4d8f5bfa67

## ⚽ datasets

Original data comes from the [DFL - Bundesliga Data Shootout](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout)
Kaggle competition. This data has been processed to create new datasets, which can be
downloaded from the [Roboflow Universe](https://universe.roboflow.com/).

| use case                        | dataset                                                                                                                                                          | train model                                                                                                                                                                                            |
| :------------------------------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| soccer player detection         | [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow/sports/blob/main/examples/soccer/notebooks/train_player_detector.ipynb)         |
| soccer ball detection           | [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg)    | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow/sports/blob/main/examples/soccer/notebooks/train_ball_detector.ipynb)           |
| soccer pitch keypoint detection | [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi)   | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow/sports/blob/main/examples/soccer/notebooks/train_pitch_keypoint_detector.ipynb) |

## 🤖 models

- [YOLOv8](https://docs.ultralytics.com/models/yolov8/) (Player Detection) - Detects
  players, goalkeepers, referees, and the ball in the video.
- [YOLOv8](https://docs.ultralytics.com/models/yolov8/) (Pitch Detection) - Identifies
  the soccer field boundaries and key points.
- [SigLIP](https://huggingface.co/docs/transformers/en/model_doc/siglip) - Extracts
  features from image crops of players.
- [UMAP](https://umap-learn.readthedocs.io/en/latest/) - Reduces the dimensionality of
  the extracted features for easier clustering.
- [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) -
  Clusters the reduced-dimension features to classify players into two teams.
