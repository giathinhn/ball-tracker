# -*- coding: utf-8 -*-
"""
sports/common/dashboard.py
==========================
HUD dashboard overlay rendered onto video frames.

Displays:
  - Current / max / avg ball speed with animated colour bar
  - Total distance covered by the ball
  - Elapsed time

Usage example::

    renderer = DashboardRenderer(fps=25.0, max_expected_kmh=120.0)
    frame = renderer.render(frame, speed_kmh=42.3, estimator=speed_estimator)

"""

from __future__ import annotations

import time
from typing import Optional, TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from sports.common.ball import BallSpeedEstimator


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _hex_to_bgr(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)


# ---------------------------------------------------------------------------
# DashboardRenderer
# ---------------------------------------------------------------------------

class DashboardRenderer:
    """
    Renders a semi-transparent HUD panel in the top-left corner of a frame.

    Panel layout::

        +------------------------------------------+
        |  ** BALL SPEED                            |
        |  [############.........]  42.3 km/h       |
        |  MAX  67.1 km/h   AVG  35.2 km/h          |
        |                                           |
        |  DISTANCE   1234.5 m                      |
        |  TIME        00:34                        |
        +------------------------------------------+

    Args:
        fps (float): Source video FPS, used to compute elapsed time.
        max_expected_kmh (float): Reference maximum for the speed bar scale.
        panel_alpha (float): Background panel opacity (0‒1).
        position (str): ``'top-left'`` or ``'bottom-left'``.
    """

    # Design tokens
    _PANEL_W = 380
    _PANEL_H = 185
    _PAD = 18
    _BAR_H = 18
    _CORNER_R = 12

    _BG_COLOR = (20, 20, 28)          # near-black
    _ACCENT    = _hex_to_bgr('#00E5FF')  # cyan
    _WHITE     = (240, 240, 240)
    _GREY      = (140, 140, 150)
    _GREEN     = _hex_to_bgr('#00E676')
    _YELLOW    = _hex_to_bgr('#FFD740')
    _RED       = _hex_to_bgr('#FF5252')

    _FONT      = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(
        self,
        fps: float = 25.0,
        max_expected_kmh: float = 120.0,
        panel_alpha: float = 0.78,
        position: str = 'top-left',
    ) -> None:
        self.fps = max(fps, 1.0)
        self.max_expected_kmh = max(max_expected_kmh, 1.0)
        self.panel_alpha = float(np.clip(panel_alpha, 0.0, 1.0))
        self.position = position

        self._frame_count: int = 0
        self._start_wall: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        frame: np.ndarray,
        speed_kmh: float,
        estimator: Optional['BallSpeedEstimator'] = None,
    ) -> np.ndarray:
        """
        Draw the HUD panel onto *frame* (in-place) and return it.

        Args:
            frame: BGR video frame.
            speed_kmh: Current (smoothed) ball speed in km/h.
            estimator: ``BallSpeedEstimator`` instance for max/avg/distance.

        Returns:
            Annotated frame.
        """
        if self._start_wall is None:
            self._start_wall = time.time()
        self._frame_count += 1

        h, w = frame.shape[:2]
        x0, y0 = self._panel_origin(w, h)

        # --- Draw panel background ---
        panel = frame[y0:y0 + self._PANEL_H, x0:x0 + self._PANEL_W].copy()
        panel_bg = np.full_like(panel, self._BG_COLOR, dtype=np.uint8)
        # Rounded-rect mask
        mask = np.zeros((self._PANEL_H, self._PANEL_W), dtype=np.uint8)
        cv2.rectangle(mask, (self._CORNER_R, 0),
                      (self._PANEL_W - self._CORNER_R, self._PANEL_H), 255, -1)
        cv2.rectangle(mask, (0, self._CORNER_R),
                      (self._PANEL_W, self._PANEL_H - self._CORNER_R), 255, -1)
        for cx, cy in [
            (self._CORNER_R, self._CORNER_R),
            (self._PANEL_W - self._CORNER_R, self._CORNER_R),
            (self._CORNER_R, self._PANEL_H - self._CORNER_R),
            (self._PANEL_W - self._CORNER_R, self._PANEL_H - self._CORNER_R),
        ]:
            cv2.circle(mask, (cx, cy), self._CORNER_R, 255, -1)

        blended = cv2.addWeighted(panel_bg, self.panel_alpha,
                                   panel, 1 - self.panel_alpha, 0)
        blended[mask == 0] = panel[mask == 0]       # keep pixels outside mask unchanged
        frame[y0:y0 + self._PANEL_H, x0:x0 + self._PANEL_W] = blended

        # --- Draw cyan top accent bar ---
        cv2.rectangle(frame,
                      (x0 + self._CORNER_R, y0),
                      (x0 + self._PANEL_W - self._CORNER_R, y0 + 3),
                      self._ACCENT, -1)

        # --- Content rows ---
        px = x0 + self._PAD
        py = y0 + self._PAD + 14

        # Title
        self._text(frame, '** BALL SPEED', px, py,
                   color=self._ACCENT, scale=0.52, bold=True)
        py += 26

        # Speed value (large)
        speed_str = f'{speed_kmh:5.1f} km/h'
        self._text(frame, speed_str, px, py,
                   color=self._WHITE, scale=0.90, bold=True)
        py += 8

        # Speed bar
        self._speed_bar(frame, x0, py, speed_kmh)
        py += self._BAR_H + 14

        # Max / Avg row
        max_kmh = estimator.max_speed_kmh if estimator else speed_kmh
        avg_kmh = estimator.avg_speed_kmh if estimator else speed_kmh
        self._text(frame, f'MAX  {max_kmh:5.1f} km/h', px, py,
                   color=self._YELLOW, scale=0.42)
        self._text(frame, f'AVG  {avg_kmh:5.1f} km/h',
                   px + 180, py, color=self._GREY, scale=0.42)
        py += 26

        # Divider
        cv2.line(frame,
                 (x0 + self._PAD, py),
                 (x0 + self._PANEL_W - self._PAD, py),
                 (60, 60, 70), 1)
        py += 14

        # Distance
        dist_m = estimator.total_distance_m if estimator else 0.0
        self._text(frame, 'DISTANCE', px, py, color=self._GREY, scale=0.42)
        self._text(frame, f'{dist_m:,.1f} m',
                   px + 110, py, color=self._WHITE, scale=0.48, bold=True)
        py += 26

        # Time
        elapsed_s = self._frame_count / self.fps
        mm, ss = divmod(int(elapsed_s), 60)
        self._text(frame, 'TIME', px, py, color=self._GREY, scale=0.42)
        self._text(frame, f'{mm:02d}:{ss:02d}',
                   px + 110, py, color=self._WHITE, scale=0.48, bold=True)

        return frame

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _panel_origin(self, w: int, h: int):
        margin = 20
        if self.position == 'top-left':
            return margin, margin
        elif self.position == 'bottom-left':
            return margin, h - self._PANEL_H - margin
        elif self.position == 'top-right':
            return w - self._PANEL_W - margin, margin
        else:  # bottom-right
            return w - self._PANEL_W - margin, h - self._PANEL_H - margin

    def _speed_bar(self, frame: np.ndarray, x0: int, y0: int, speed_kmh: float) -> None:
        """Draw a colour-coded horizontal speed bar."""
        bar_x = x0 + self._PAD
        bar_w = self._PANEL_W - 2 * self._PAD

        # Background track
        cv2.rectangle(frame,
                      (bar_x, y0),
                      (bar_x + bar_w, y0 + self._BAR_H),
                      (50, 50, 60), -1)

        # Filled portion
        ratio = float(np.clip(speed_kmh / self.max_expected_kmh, 0.0, 1.0))
        fill_w = int(ratio * bar_w)
        if fill_w > 0:
            bar_color = self._speed_color(ratio)
            cv2.rectangle(frame,
                          (bar_x, y0),
                          (bar_x + fill_w, y0 + self._BAR_H),
                          bar_color, -1)

        # Border
        cv2.rectangle(frame,
                      (bar_x, y0),
                      (bar_x + bar_w, y0 + self._BAR_H),
                      (80, 80, 90), 1)

    @staticmethod
    def _speed_color(ratio: float) -> tuple:
        """Green → Yellow → Red based on speed ratio."""
        if ratio < 0.5:
            t = ratio * 2                          # 0→1 as ratio 0→0.5
            r = int(t * 255)
            g = 200
            b = 0
        else:
            t = (ratio - 0.5) * 2                  # 0→1 as ratio 0.5→1
            r = 255
            g = int((1 - t) * 200)
            b = 0
        return (b, g, r)

    @staticmethod
    def _text(
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: tuple = (255, 255, 255),
        scale: float = 0.5,
        bold: bool = False,
    ) -> None:
        thickness = 2 if bold else 1
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,
                    cv2.LINE_AA)
