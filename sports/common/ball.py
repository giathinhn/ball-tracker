from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np
import supervision as sv

from sports.common.view import ViewTransformer


# ---------------------------------------------------------------------------
# Advanced BallAnnotator — gradient trail with alpha blending
# ---------------------------------------------------------------------------

class BallAnnotator:
    """
    Annotates frames with a gradient motion trail for the ball.

    The trail fades from transparent (oldest) to fully opaque (newest), and
    shifts color from cool (blue-cyan) to warm (yellow-red) to indicate
    recency.  Each dot's radius also grows towards the current position.

    Attributes:
        radius (int): Maximum radius of the ball dot.
        buffer_size (int): Number of past positions to keep.
        thickness (int): Outline thickness (-1 = filled).
    """

    def __init__(self, radius: int, buffer_size: int = 20, thickness: int = -1):
        self.radius = radius
        self.thickness = thickness
        self.buffer: deque = deque(maxlen=buffer_size)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hsv_to_bgr(h: float, s: float = 1.0, v: float = 1.0) -> Tuple[int, int, int]:
        """Convert HSV (h in [0,1]) to BGR tuple."""
        h_deg = h * 179  # OpenCV hue range
        hsv_pixel = np.array([[[int(h_deg), int(s * 255), int(v * 255)]]], dtype=np.uint8)
        bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)
        b, g, r = int(bgr_pixel[0, 0, 0]), int(bgr_pixel[0, 0, 1]), int(bgr_pixel[0, 0, 2])
        return (b, g, r)

    def _trail_params(self, idx: int, total: int) -> Tuple[Tuple[int, int, int], float, int]:
        """
        For a trail point at position `idx` (0 = oldest, total-1 = newest),
        return (bgr_color, alpha, radius).
        """
        t = idx / max(total - 1, 1)          # 0.0 (oldest) → 1.0 (newest)

        # Hue: 0.65 (blue) → 0.15 (yellow) as t increases
        hue = 0.65 - t * 0.50
        bgr = self._hsv_to_bgr(hue)

        # Alpha: 0.08 → 1.0
        alpha = 0.08 + t * 0.92

        # Radius: 2 → self.radius
        r = max(2, int(2 + t * (self.radius - 2)))

        return bgr, alpha, r

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """
        Draw the gradient motion trail and the current ball position.

        Args:
            frame: BGR video frame.
            detections: Ball detections for the current frame.

        Returns:
            Annotated frame.
        """
        if len(detections) > 0:
            xy = detections.get_anchors_coordinates(sv.Position.CENTER).astype(int)
            self.buffer.append(xy)
        else:
            # Keep buffer growing even on missed detections (with last known pos)
            if self.buffer:
                self.buffer.append(self.buffer[-1])

        n = len(self.buffer)
        for idx, positions in enumerate(self.buffer):
            bgr, alpha, r = self._trail_params(idx, n)
            for cx, cy in positions:
                overlay = frame.copy()
                cv2.circle(overlay, (cx, cy), r, bgr, self.thickness)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        return frame

    def annotate_with_label(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        speed_kmh: float = 0.0,
    ) -> np.ndarray:
        """
        Draw trail + bounding box + "BALL #ID  XX km/h" label on the current ball.

        Args:
            frame: BGR video frame.
            detections: Ball detections for the current frame (may carry tracker_id).
            speed_kmh: Current smoothed ball speed in km/h.

        Returns:
            Annotated frame.
        """
        # Draw the gradient trail first
        frame = self.annotate(frame, detections)

        if len(detections) == 0 or not self.buffer:
            return frame

        # Use the latest buffered position as ball centre
        cx, cy = int(self.buffer[-1][0][0]), int(self.buffer[-1][0][1])
        box_r = self.radius + 6

        # --- bounding box (white, 2 px) ---
        cv2.rectangle(
            frame,
            (cx - box_r, cy - box_r),
            (cx + box_r, cy + box_r),
            (255, 255, 255),
            2,
        )

        # --- label text (include tracker ID if available) ---
        track_str = ""
        if (
            detections.tracker_id is not None
            and len(detections.tracker_id) > 0
        ):
            track_str = f" #{int(detections.tracker_id[0])}"
        label = f"BALL{track_str}  {speed_kmh:.1f} km/h"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Position label above the box, clamp to frame top
        lx = cx - box_r
        ly = max(cy - box_r - 6, th + baseline + 4)

        # Semi-transparent dark background pill
        pad = 4
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (lx - pad, ly - th - pad),
            (lx + tw + pad, ly + baseline + pad),
            (20, 20, 20),
            -1,
        )
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Label text in bright yellow-cyan
        cv2.putText(
            frame,
            label,
            (lx, ly),
            font,
            font_scale,
            (0, 230, 255),
            thickness,
            cv2.LINE_AA,
        )

        return frame


# ---------------------------------------------------------------------------
# BallTracker — unchanged logic, tiny refactor for clarity
# ---------------------------------------------------------------------------

class BallTracker:
    """
    Simple ball tracker — picks the detection closest to the historical
    centroid of recent positions.  Used by the lightweight BALL_DETECTION
    pipeline; the full BALL_TRACKING pipeline uses KalmanBallTracker.

    Attributes:
        buffer (deque): Recent ball center coordinates.
    """

    def __init__(self, buffer_size: int = 20):
        self.buffer: deque = deque(maxlen=buffer_size)

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Update the tracker and return the single best detection.

        Args:
            detections: Raw ball detections in this frame.

        Returns:
            The detection closest to the historical centroid, or the original
            (empty) detections if nothing was found.
        """
        xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        self.buffer.append(xy)

        if len(detections) == 0:
            return detections

        centroid = np.mean(np.concatenate(self.buffer), axis=0)
        distances = np.linalg.norm(xy - centroid, axis=1)
        index = int(np.argmin(distances))
        return detections[[index]]


# ---------------------------------------------------------------------------
# KalmanBallTracker — Discrete Kalman Filter for precise 2D ball tracking
# ---------------------------------------------------------------------------

class KalmanBallTracker:
    """
    Discrete Kalman Filter tracker for a single soccer ball.

    Implements the 2D ball tracking formulation described in the literature:

        Plant equation:      a_i = A * a_{i-1}
        Measurement equation: z_i = H * a_i + v_i

    State vector (5-dimensional):
        [x, y, r, vx, vy]
        - (x, y) : ball centre in pixels
        - r      : ball radius in pixels
        - (vx,vy): velocity in pixels / frame

    Measurement vector (3-dimensional):
        [x, y, r]  — the observed centre and radius from YOLO.

    The tracker also pre-filters raw YOLO detections to suppress false
    positives (shoes, field markings, white dots) before Kalman association:
        1. Confidence threshold
        2. Maximum bounding-box size (large objects cannot be the ball)
        3. Aspect-ratio check (ball must be roughly square)
        4. Spatial gate: candidates must be within `gate_distance_px` of
           the Kalman prediction.

    The ball is always assigned tracker_id = 1.

    Args:
        fps (float): Video frame rate (used to set dt = 1/fps).
        max_miss (int): Frames without detection before the track is lost.
        min_confidence (float): Minimum YOLO confidence to keep a candidate.
        max_box_side (int): Maximum pixel length of any bounding-box side.
        gate_distance_px (float): Mahalanobis-style gate radius in pixels.
    """

    BALL_ID: int = 1  # fixed track-ID for the single ball

    def __init__(
        self,
        fps: float = 25.0,
        max_miss: int = 10,
        min_confidence: float = 0.3,
        max_box_side: int = 80,
        gate_distance_px: float = 150.0,
    ):
        if fps <= 0:
            raise ValueError("fps must be positive.")

        self.fps              = fps
        self.max_miss         = max_miss
        self.min_confidence   = min_confidence
        self.max_box_side     = max_box_side
        self.gate_distance_px = gate_distance_px

        self._prev_real_xy: Optional[np.ndarray] = None
        self._prev_real_v: Optional[np.ndarray] = None

        dt = 1.0 / fps

        # ---- State transition matrix A (constant-velocity model) ----
        # State: [x, y, r, vx, vy]
        self.A = np.array([
            [1, 0, 0, dt, 0 ],
            [0, 1, 0, 0,  dt],
            [0, 0, 1, 0,  0 ],
            [0, 0, 0, 1,  0 ],
            [0, 0, 0, 0,  1 ],
        ], dtype=np.float64)

        # ---- Measurement matrix H (observe x, y, r) ----
        self.H = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
        ], dtype=np.float64)

        # ---- Process noise covariance Q ----
        # Larger values for velocity → allow quick changes in ball direction.
        self.Q = np.diag([10.0, 10.0, 2.0, 120.0, 120.0])

        # ---- Measurement noise covariance R ----
        # Reflects typical YOLO localisation uncertainty (~8 px std).
        self.R = np.diag([8.0, 8.0, 3.0])

        # ---- Internal state ----
        self._x: Optional[np.ndarray] = None   # (5,) state estimate
        self._P: Optional[np.ndarray] = None   # (5,5) covariance
        self._initialized: bool       = False
        self._miss_count: int         = 0

    # ------------------------------------------------------------------
    # Kalman predict / update
    # ------------------------------------------------------------------

    def _kf_predict(self) -> None:
        """Propagate state one step forward."""
        self._x = self.A @ self._x
        self._P = self.A @ self._P @ self.A.T + self.Q

    def _kf_update(self, z: np.ndarray) -> None:
        """Correct state with measurement z = [x, y, r]."""
        H, R, P, x = self.H, self.R, self._P, self._x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        self._x = x + K @ (z - H @ x)
        self._P = (np.eye(5) - K @ H) @ P

    def _init_track(self, cx: float, cy: float, r: float) -> None:
        """Bootstrap the filter from the first reliable detection."""
        self._x = np.array([cx, cy, r, 0.0, 0.0], dtype=np.float64)
        self._P = np.diag([50.0, 50.0, 10.0, 250.0, 250.0])
        self._initialized = True
        self._miss_count  = 0

    # ------------------------------------------------------------------
    # Pre-filtering helpers
    # ------------------------------------------------------------------

    def _filter_candidates(self, detections: sv.Detections) -> sv.Detections:
        """
        Discard false positives before Kalman association.

        Filters:
        - Confidence  ≥ min_confidence
        - Bounding-box side  ≤ max_box_side  (removes large non-ball objects)
        - Aspect ratio  ≥ 0.45              (removes elongated detections)
        """
        if len(detections) == 0:
            return detections

        mask = np.ones(len(detections), dtype=bool)

        if detections.confidence is not None:
            mask &= detections.confidence >= self.min_confidence

        xyxy    = detections.xyxy
        widths  = xyxy[:, 2] - xyxy[:, 0]
        heights = xyxy[:, 3] - xyxy[:, 1]

        # Size filter
        mask &= np.maximum(widths, heights) <= self.max_box_side

        # Aspect-ratio filter
        shorter = np.minimum(widths, heights)
        longer  = np.maximum(widths, heights)
        mask   &= (shorter / (longer + 1e-6)) >= 0.45

        return detections[mask]

    def _gate(self, detections: sv.Detections) -> sv.Detections:
        """Keep only candidates within gate_distance_px of the prediction."""
        if len(detections) == 0 or not self._initialized:
            return detections
        centres = detections.get_anchors_coordinates(sv.Position.CENTER)
        dists   = np.linalg.norm(centres - self._x[:2], axis=1)
        return detections[dists <= self.gate_distance_px]

    # ------------------------------------------------------------------
    # State → sv.Detections
    # ------------------------------------------------------------------

    def _to_detections(
        self, ref: Optional[sv.Detections] = None
    ) -> sv.Detections:
        """Wrap the current Kalman state as an sv.Detections object."""
        sx, sy, sr = self._x[0], self._x[1], abs(self._x[2])
        sr = max(sr, 5.0)
        xyxy = np.array([[sx - sr, sy - sr, sx + sr, sy + sr]])
        conf = (
            ref.confidence
            if ref is not None and ref.confidence is not None
            else np.array([1.0])
        )
        cls = (
            ref.class_id
            if ref is not None and ref.class_id is not None
            else None
        )
        return sv.Detections(
            xyxy=xyxy,
            confidence=conf,
            class_id=cls,
            tracker_id=np.array([self.BALL_ID]),
        )

    # ------------------------------------------------------------------
    # Public API  (split interface for Kalman-guided search window)
    # ------------------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        """True once the filter has been bootstrapped from a first detection."""
        return self._initialized

    def predict(self) -> "KalmanBallTracker":
        """
        Advance the Kalman filter one time step (prediction phase).

        Call this BEFORE get_search_window() and correct() so the
        predicted state / covariance are up-to-date when computing the
        search window.

        Returns self for optional method chaining.
        """
        if self._initialized:
            self._kf_predict()
        return self

    def get_search_window(
        self,
        frame_wh: Tuple[int, int],
        k_sigma: float = 3.0,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Return the Kalman-predicted search window as (x1, y1, x2, y2).

        Mathematical basis (from the literature):
        After the prediction phase the filter computes a predicted covariance
        matrix P̄_i. The diagonal elements P̄[0,0] and P̄[1,1] are the
        variances of the predicted x- and y-positions respectively.

        Taking √P̄[j,j] gives the standard deviation σ_j.  Expanding ±k·σ_j
        around the predicted centre captures:
            k=2 → 95.4 % of the probability mass (Gaussian assumption)
            k=3 → 99.7 % of the probability mass

        This window is used to crop the frame before running YOLO, so the
        detector only sees a small region where the ball is almost certainly
        located, eliminating far-away false positives (shoes, field markings).

        Args:
            frame_wh: (width, height) of the full frame in pixels.
            k_sigma:  Number of standard deviations around the predicted
                      centre (default 3 → 99.7 % coverage).

        Returns:
            Integer rectangle (x1, y1, x2, y2) clamped to frame bounds,
            or None if the filter is not yet initialised.
        """
        if not self._initialized or self._P is None or self._x is None:
            return None

        cx, cy = float(self._x[0]), float(self._x[1])

        # σ from diagonal of predicted covariance (position components only)
        sigma_x = float(np.sqrt(max(self._P[0, 0], 0.0)))
        sigma_y = float(np.sqrt(max(self._P[1, 1], 0.0)))

        # Half-extents: at least 50 px so we don't miss a fast-moving ball
        half_w = max(k_sigma * sigma_x, 50.0)
        half_h = max(k_sigma * sigma_y, 50.0)

        fw, fh = frame_wh
        x1 = max(0,  int(cx - half_w))
        y1 = max(0,  int(cy - half_h))
        x2 = min(fw, int(cx + half_w))
        y2 = min(fh, int(cy + half_h))

        # Guard: window must be large enough to be meaningful
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            return None

        return (x1, y1, x2, y2)

    def correct(
        self,
        detections: sv.Detections,
        transformer: Optional[ViewTransformer] = None,
    ) -> sv.Detections:
        """
        Correction phase: associate detections with the predicted state and
        update the Kalman filter.  Always returns at most ONE detection
        (enforcing the one-ball-per-frame constraint).

        Call AFTER predict() and after running detection inside the
        search window returned by get_search_window().

        Pipeline:
        1. Pre-filter (confidence / size / aspect ratio).
        2. Spatial gate — discard candidates outside the predicted region.
        3. Physical constraint validation (max speed / max acceleration).
        4. Select the single best candidate.
        5. Kalman update (or increment miss counter if none found).
        6. Return smoothed sv.Detections with tracker_id = BALL_ID.

        Args:
            detections: sv.Detections from YOLO (already in full-frame coords).
            transformer: ViewTransformer for mapping pixel coordinates to pitch.

        Returns:
            Exactly one sv.Detections (the Kalman estimate), or
            sv.Detections.empty() if the ball is considered lost.
        """
        # 1. Pre-filter false positives
        candidates = self._filter_candidates(detections)

        # 2. Spatial gate around predicted position
        candidates = self._gate(candidates)

        # 3. Physical constraint validation (max speed / max acceleration)
        if len(candidates) > 0 and transformer is not None and self._prev_real_xy is not None:
            dt = 1.0 / self.fps
            a_max = 1200.0          # m/s^2 (max acceleration of a football during a kick)
            v_absolute_max = 45.0  # m/s (absolute speed limit of a football)

            mask = np.ones(len(candidates), dtype=bool)
            centres = candidates.get_anchors_coordinates(sv.Position.CENTER)

            try:
                transformed = transformer.transform_points(centres.astype(np.float32))
                real_coords_m = transformed * 0.01  # convert cm to meters
                
                for idx, xy_m in enumerate(real_coords_m):
                    # calculate speed
                    v_curr = (xy_m - self._prev_real_xy) / dt
                    speed_curr = float(np.linalg.norm(v_curr))

                    if speed_curr > v_absolute_max:
                        mask[idx] = False
                        continue

                    # calculate acceleration
                    if self._prev_real_v is not None:
                        a_curr = (v_curr - self._prev_real_v) / dt
                        accel_curr = float(np.linalg.norm(a_curr))
                        if accel_curr > a_max:
                            mask[idx] = False
                
                candidates = candidates[mask]
            except Exception:
                pass

        # ---- No valid candidate this frame ----
        if len(candidates) == 0:
            if not self._initialized:
                return sv.Detections.empty()
            self._miss_count += 1
            if self._miss_count > self.max_miss:
                self._initialized = False   # ball considered lost
                self._prev_real_xy = None
                self._prev_real_v = None
                return sv.Detections.empty()
            # Coast: return the predicted position as-is
            return self._to_detections()

        # 4. Select single best candidate (one ball per frame)
        centres = candidates.get_anchors_coordinates(sv.Position.CENTER)
        if self._initialized:
            dists    = np.linalg.norm(centres - self._x[:2], axis=1)
            best_idx = int(np.argmin(dists))
        elif candidates.confidence is not None:
            best_idx = int(np.argmax(candidates.confidence))
        else:
            best_idx = 0

        best = candidates[[best_idx]]          # exactly 1 detection
        cx   = float(centres[best_idx, 0])
        cy   = float(centres[best_idx, 1])
        b    = best.xyxy[0]
        r    = float(max(b[2] - b[0], b[3] - b[1]) / 2)

        # 5. Kalman update
        if not self._initialized:
            self._init_track(cx, cy, r)
            # Store initial real position
            if transformer is not None:
                try:
                    transformed = transformer.transform_points(np.array([[cx, cy]], dtype=np.float32))
                    self._prev_real_xy = transformed[0] * 0.01
                    self._prev_real_v = None
                except Exception:
                    self._prev_real_xy = None
                    self._prev_real_v = None
        else:
            self._kf_update(np.array([cx, cy, r]))
            self._miss_count = 0
            # Update real position and velocity
            if transformer is not None:
                try:
                    transformed = transformer.transform_points(np.array([[cx, cy]], dtype=np.float32))
                    curr_real_xy = transformed[0] * 0.01
                    if self._prev_real_xy is not None:
                        self._prev_real_v = (curr_real_xy - self._prev_real_xy) / (1.0 / self.fps)
                    self._prev_real_xy = curr_real_xy
                except Exception:
                    pass

        # 6. Return Kalman-smoothed detection (1 ball max)
        return self._to_detections(best)

    def update(
        self,
        detections: sv.Detections,
        transformer: Optional[ViewTransformer] = None,
    ) -> sv.Detections:
        """
        Convenience wrapper: predict() → correct() in a single call.

        Use this for the simple pipeline (no search-window cropping).
        For the full Kalman-guided pipeline, call predict() /
        get_search_window() / detect on crop / correct() explicitly.
        """
        self.predict()
        return self.correct(detections, transformer)


# ---------------------------------------------------------------------------
# BallSpeedEstimator — NEW
# ---------------------------------------------------------------------------

class BallSpeedEstimator:
    """
    Estimates ball speed using real-world coordinates obtained from a
    ViewTransformer (homography-based pitch mapping).

    Speed is smoothed with a moving-average over `smooth_window` frames to
    reduce jitter caused by detection noise.

    Units:
        - Input coordinates in **centimetres** (matching SoccerPitchConfiguration).
        - Output speed in **m/s** and **km/h**.
        - Distance accumulated in **metres**.

    Args:
        fps (float): Frames per second of the source video.
        smooth_window (int): Window size for moving-average speed smoothing.
    """

    CM_TO_M = 0.01

    def __init__(self, fps: float, smooth_window: int = 7):
        if fps <= 0:
            raise ValueError("FPS must be positive.")
        self.fps = fps
        self._prev_xy: Optional[np.ndarray] = None   # (2,) array in cm
        self._speed_buffer: deque = deque(maxlen=smooth_window)

        # Cumulative statistics
        self.total_distance_m: float = 0.0
        self.max_speed_kmh: float = 0.0
        self._speed_history: list = []               # for avg calculation

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # Maximum physically plausible displacement per frame in metres.
    # At 30 fps, 15 m/frame ≈ 1 620 km/h — anything above this is a
    # homography artefact and should be discarded.
    MAX_DELTA_M_PER_FRAME: float = 15.0

    def update(self, real_xy: Optional[np.ndarray]) -> Tuple[float, float]:
        """
        Supply the current ball position in pitch coordinates and receive
        instantaneous (smoothed) speed.

        Args:
            real_xy: Shape (2,) array [x_cm, y_cm], or None if ball not found.

        Returns:
            Tuple (speed_ms, speed_kmh).  Returns (0.0, 0.0) if speed cannot
            be computed yet.
        """
        if real_xy is None or real_xy.size == 0:
            # Do NOT reset _prev_xy — preserves continuity for next detection.
            # Simply contribute a zero to the smoothing buffer.
            self._speed_buffer.append(0.0)
            return self._smoothed()

        real_xy = np.asarray(real_xy, dtype=np.float64).flatten()[:2]

        if self._prev_xy is None:
            self._prev_xy = real_xy
            self._speed_buffer.append(0.0)
            return self._smoothed()

        # Euclidean distance in cm → metres
        delta_m = float(np.linalg.norm(real_xy - self._prev_xy)) * self.CM_TO_M

        # Outlier filter: discard frames where displacement is physically
        # impossible (homography glitch).  Keep _prev_xy unchanged so the
        # next valid frame can still compute a sensible delta.
        if delta_m > self.MAX_DELTA_M_PER_FRAME:
            self._speed_buffer.append(float(np.mean(self._speed_buffer)) if self._speed_buffer else 0.0)
            return self._smoothed()

        self._prev_xy = real_xy

        # Distance per frame → speed in m/s
        raw_speed_ms = delta_m * self.fps

        self._speed_buffer.append(raw_speed_ms)
        self.total_distance_m += delta_m

        speed_ms, speed_kmh = self._smoothed()
        self.max_speed_kmh = max(self.max_speed_kmh, speed_kmh)
        self._speed_history.append(speed_kmh)
        return speed_ms, speed_kmh

    @property
    def avg_speed_kmh(self) -> float:
        """Average speed over all frames where the ball was detected."""
        if not self._speed_history:
            return 0.0
        return float(np.mean(self._speed_history))

    def reset_stats(self) -> None:
        """Reset cumulative statistics (distance, max/avg speed)."""
        self.total_distance_m = 0.0
        self.max_speed_kmh = 0.0
        self._speed_history.clear()
        self._prev_xy = None
        self._speed_buffer.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _smoothed(self) -> Tuple[float, float]:
        speed_ms = float(np.mean(self._speed_buffer)) if self._speed_buffer else 0.0
        speed_kmh = speed_ms * 3.6
        if speed_kmh > 80.0:
            speed_kmh = 80.0
            speed_ms = 80.0 / 3.6
        return speed_ms, speed_kmh
