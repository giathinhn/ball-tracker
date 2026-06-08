import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from typing import Optional

from sports.common.ball import BallTracker, BallAnnotator, BallSpeedEstimator, KalmanBallTracker
from sports.common.team_smoother import TeamIdSmoother, associate_tracker_ids
from sports.common.dashboard import DashboardRenderer
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

# Calculate models paths relative to the project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))

PLAYER_DETECTION_MODEL_PATH = os.path.join(PROJECT_ROOT, 'model', 'football-player-detection.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PROJECT_ROOT, 'model', 'football-pitch-detection.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(PROJECT_ROOT, 'model', 'football-ball-detection.pt')

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 60
CONFIG = SoccerPitchConfiguration()

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    thickness=2,
    edges=CONFIG.edges,
)
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    base=20,
    height=15,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


def get_crops(frame: np.ndarray, detections: sv.Detections) -> list[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.

    Args:
        frame (np.ndarray): The frame from which to extract crops.
        detections (sv.Detections): Detected objects with bounding boxes.

    Returns:
        List[np.ndarray]: List of cropped images.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.ndarray,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Resolve the team IDs for detected goalkeepers based on the proximity to team
    centroids.

    Args:
        players (sv.Detections): Detections of all players.
        players_team_id (np.ndarray): Array containing team IDs of detected players.
        goalkeepers (sv.Detections): Detections of goalkeepers.

    Returns:
        np.ndarray: Array containing team IDs for the detected goalkeepers.

    This function calculates the centroids of the two teams based on the positions of
    the players. Then, it assigns each goalkeeper to the nearest team's centroid by
    calculating the distance between each goalkeeper and the centroids of the two teams.
    """
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)


def render_radar(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray
) -> np.ndarray:
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)

    radar = draw_pitch(config=CONFIG)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 0],
        face_color=sv.Color.from_hex(COLORS[0]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 1],
        face_color=sv.Color.from_hex(COLORS[1]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 2],
        face_color=sv.Color.from_hex(COLORS[2]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 3],
        face_color=sv.Color.from_hex(COLORS[3]), radius=20, pitch=radar)
    return radar


def run_pitch_detection(source_video_path: str, device: str) -> sv.Detections:
    """
    Run pitch detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
            annotated_frame, keypoints, CONFIG.labels)
        yield annotated_frame


def run_player_detection(source_video_path: str, device: str) -> sv.Detections:
    """
    Run player detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame


def run_ball_detection(source_video_path: str, device: str) -> sv.Detections:
    """
    Run ball detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=callback,
        slice_wh=(640, 640),
    )

    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=0.1)
        detections = ball_tracker.update(detections)
        annotated_frame = frame.copy()
        annotated_frame = ball_annotator.annotate(annotated_frame, detections)
        yield annotated_frame


def run_player_tracking(source_video_path: str, device: str) -> sv.Detections:
    """
    Run player tracking on a video and yield annotated frames with tracked players.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels=labels)
        yield annotated_frame


def run_team_classification(source_video_path: str, device: str) -> sv.Detections:
    """
    Run team classification on a video and yield annotated frames with team colors.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)
    team_smoother = TeamIdSmoother()

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)
        if players.tracker_id is not None:
            players_team_id = team_smoother.smooth(players.tracker_id, players_team_id)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        # Collect tracker IDs before merging
        tracker_ids = np.concatenate([
            players.tracker_id if players.tracker_id is not None else np.array([None] * len(players)),
            goalkeepers.tracker_id if goalkeepers.tracker_id is not None else np.array([None] * len(goalkeepers)),
            referees.tracker_id if referees.tracker_id is not None else np.array([None] * len(referees))
        ])
        
        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
                players_team_id.tolist() +
                goalkeepers_team_id.tolist() +
                [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(int(tracker_id)) if tracker_id is not None else "" for tracker_id in tracker_ids]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup)
        yield annotated_frame


def run_radar(source_video_path: str, device: str) -> sv.Detections:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)
    team_smoother = TeamIdSmoother()

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)
        if players.tracker_id is not None:
            players_team_id = team_smoother.smooth(players.tracker_id, players_team_id)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        # Collect tracker IDs before merging
        tracker_ids = np.concatenate([
            players.tracker_id if players.tracker_id is not None else np.array([None] * len(players)),
            goalkeepers.tracker_id if goalkeepers.tracker_id is not None else np.array([None] * len(goalkeepers)),
            referees.tracker_id if referees.tracker_id is not None else np.array([None] * len(referees))
        ])
        
        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(int(tracker_id)) if tracker_id is not None else "" for tracker_id in tracker_ids]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels,
            custom_color_lookup=color_lookup)

        h, w, _ = frame.shape
        radar = render_radar(detections, keypoints, color_lookup)
        radar = sv.resize_image(radar, (w // 2, h // 2))
        radar_h, radar_w, _ = radar.shape
        rect = sv.Rect(
            x=w // 2 - radar_w // 2,
            y=h - radar_h,
            width=radar_w,
            height=radar_h
        )
        annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)
        yield annotated_frame


def run_ball_tracking(
    source_video_path: str,
    device: str,
    fps: Optional[float] = None,
):
    """
    Combined pipeline: player detection + ball tracking with speed estimation,
    gradient motion trail, and HUD dashboard.

    Detects and annotates players with bounding boxes, tracks the ball with
    a gradient motion trail, estimates ball speed via homography-based pitch
    mapping (ViewTransformer), and overlays a HUD dashboard with speed stats.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').
        fps (Optional[float]): Override video FPS; auto-detected if None.

    Yields:
        Iterator[np.ndarray]: Annotated BGR frames.
    """
    player_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    ball_model   = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    pitch_model  = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)

    video_info    = sv.VideoInfo.from_video_path(source_video_path)
    effective_fps = fps if fps is not None else (video_info.fps or 25.0)

    ball_tracker    = KalmanBallTracker(
        fps=effective_fps,
        max_miss=12,
        min_confidence=0.25,
        max_box_side=80,
        gate_distance_px=160.0,
    )
    ball_annotator  = BallAnnotator(radius=8, buffer_size=30)
    speed_estimator = BallSpeedEstimator(fps=effective_fps, smooth_window=9)
    dashboard       = DashboardRenderer(
        fps=effective_fps,
        max_expected_kmh=120.0,
        panel_alpha=0.80,
        position='top-left',
    )
    player_tracker  = sv.ByteTrack(minimum_consecutive_frames=3)
    team_smoother   = TeamIdSmoother()

    def _detect_ball(image_slice: np.ndarray) -> sv.Detections:
        result = ball_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(callback=_detect_ball, slice_wh=(640, 640))

    # Fit team classifier on player crops
    crops = []
    player_detection_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)
    for frame in tqdm(player_detection_generator, desc='collecting crops'):
        result = player_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    if crops:
        team_classifier.fit(crops)
    else:
        team_classifier = None

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        # 1. Player detection
        player_result     = player_model(frame, imgsz=1280, verbose=False)[0]
        player_detections = sv.Detections.from_ultralytics(player_result)

        players = player_detections[player_detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        
        # Track players separately to avoid filtering/dropping any detected player boxes
        temp_players = sv.Detections(
            xyxy=players.xyxy,
            confidence=players.confidence,
            class_id=players.class_id,
        )
        tracked_players = player_tracker.update_with_detections(temp_players)
        player_tracker_ids = associate_tracker_ids(players, tracked_players)

        if team_classifier is not None and len(crops) > 0:
            players_team_id = team_classifier.predict(crops)
            players_team_id = team_smoother.smooth(player_tracker_ids, players_team_id)
        else:
            players_team_id = np.array([0] * len(players), dtype=int)

        goalkeepers = player_detections[player_detections.class_id == GOALKEEPER_CLASS_ID]
        if len(players) > 0 and len(goalkeepers) > 0:
            goalkeepers_team_id = resolve_goalkeepers_team_id(
                players, players_team_id, goalkeepers)
        else:
            goalkeepers_team_id = np.array([0] * len(goalkeepers), dtype=int)

        referees = player_detections[player_detections.class_id == REFEREE_CLASS_ID]
        balls = player_detections[player_detections.class_id == BALL_CLASS_ID]

        # Merge them back to a single Detections object for annotation
        player_detections = sv.Detections.merge([players, goalkeepers, referees, balls])
        
        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees) +
            [2] * len(balls)
        )
        
        labels = (
            [player_model.names[PLAYER_CLASS_ID]] * len(players) +
            [player_model.names[GOALKEEPER_CLASS_ID]] * len(goalkeepers) +
            [player_model.names[REFEREE_CLASS_ID]] * len(referees) +
            [player_model.names[BALL_CLASS_ID]] * len(balls)
        )

        # 2. Pitch keypoints → ViewTransformer
        pitch_result = pitch_model(frame, verbose=False)[0]
        keypoints    = sv.KeyPoints.from_ultralytics(pitch_result)

        transformer: Optional[ViewTransformer] = None
        kp_xy      = keypoints.xy[0]
        valid_mask = (kp_xy[:, 0] > 1) & (kp_xy[:, 1] > 1)
        if valid_mask.sum() >= 4:
            try:
                transformer = ViewTransformer(
                    source=kp_xy[valid_mask].astype(np.float32),
                    target=np.array(CONFIG.vertices)[valid_mask].astype(np.float32),
                )
            except ValueError:
                transformer = None

        # 3. Ball detection with Kalman-guided search window
        #
        # Flow:
        #   a) predict()           → advance Kalman state to frame i
        #   b) get_search_window() → ±3σ rectangle from diagonal of P̄
        #   c) crop + detect       → YOLO on small ROI only (1 ball max)
        #   d) remap coords        → crop-space → full-frame space
        #   e) correct()           → Kalman measurement update
        #
        frame_h, frame_w = frame.shape[:2]

        # (a) Advance prediction so P̄ is ready for the search window
        ball_tracker.predict()

        # (b) Compute ±3σ search window from predicted covariance P̄
        search_win = ball_tracker.get_search_window(
            frame_wh=(frame_w, frame_h), k_sigma=3.0
        )

        if search_win is not None:
            # (c) Kalman initialized — detect ONLY inside the predicted window
            x1, y1, x2, y2 = search_win
            crop        = frame[y1:y2, x1:x2]
            crop_result = ball_model(crop, imgsz=640, verbose=False)[0]
            crop_det    = sv.Detections.from_ultralytics(crop_result)

            # (d) Remap bounding boxes from crop-space → full-frame space
            if len(crop_det) > 0:
                crop_det.xyxy[:, 0] += x1
                crop_det.xyxy[:, 1] += y1
                crop_det.xyxy[:, 2] += x1
                crop_det.xyxy[:, 3] += y1

            raw_detections = crop_det.with_nms(threshold=0.1)
        else:
            # (c) Not yet initialized — full-frame scan to find ball for the first time
            raw_detections = slicer(frame).with_nms(threshold=0.1)

        # (e) Correction phase: filter → gate → select 1 best → update Kalman
        ball_detection = ball_tracker.correct(raw_detections, transformer)

        # 4. Speed estimation via real-world pitch coordinates
        real_xy: Optional[np.ndarray] = None
        if transformer is not None and len(ball_detection) > 0:
            pixel_xy = ball_detection.get_anchors_coordinates(sv.Position.CENTER)
            try:
                transformed = transformer.transform_points(
                    pixel_xy.astype(np.float32)
                )
                real_xy = transformed[0]
            except Exception:
                real_xy = None

        _speed_ms, speed_kmh = speed_estimator.update(real_xy)

        # 5. Annotate frame
        annotated_frame = frame.copy()

        # Player bounding boxes + class labels with team colors
        annotated_frame = BOX_ANNOTATOR.annotate(
            annotated_frame, player_detections, custom_color_lookup=color_lookup
        )
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(
            annotated_frame, player_detections, labels=labels, custom_color_lookup=color_lookup
        )

        # Ball gradient motion trail + bounding box + speed label
        annotated_frame = ball_annotator.annotate_with_label(
            annotated_frame, ball_detection, speed_kmh=speed_kmh
        )

        # HUD dashboard (speed bar, max/avg, distance, elapsed)
        annotated_frame = dashboard.render(
            annotated_frame,
            speed_kmh=speed_kmh,
            estimator=speed_estimator,
        )

        yield annotated_frame

