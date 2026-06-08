import argparse
from enum import Enum
from typing import Optional

import torch
import cv2
import supervision as sv

# PyTorch float8 compatibility fix
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

from sports.pipelines import (
    run_pitch_detection,
    run_player_detection,
    run_ball_tracking,
    run_ball_detection,
    run_player_tracking,
    run_team_classification,
    run_radar,
)


class Mode(Enum):
    """
    Enum class representing different modes of operation for Soccer AI video analysis.
    """
    PITCH_DETECTION     = 'PITCH_DETECTION'
    PLAYER_DETECTION    = 'PLAYER_DETECTION'
    BALL_DETECTION      = 'BALL_DETECTION'
    PLAYER_TRACKING     = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR               = 'RADAR'
    BALL_TRACKING       = 'BALL_TRACKING'


def main(
    source_video_path: str,
    target_video_path: str,
    device: str,
    mode: Mode,
    fps: Optional[float] = None,
) -> None:
    # Map 'gpu' to 'cuda' to prevent PyTorch device string issues
    if device.lower() == 'gpu':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Mapped '--device gpu' to '{device}' based on CUDA availability.")

    if mode == Mode.PITCH_DETECTION:
        frame_generator = run_pitch_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_DETECTION:
        frame_generator = run_player_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.BALL_DETECTION:
        frame_generator = run_ball_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_TRACKING:
        frame_generator = run_player_tracking(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.TEAM_CLASSIFICATION:
        frame_generator = run_team_classification(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.RADAR:
        frame_generator = run_radar(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.BALL_TRACKING:
        frame_generator = run_ball_tracking(
            source_video_path=source_video_path, device=device, fps=fps)
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
            sink.write_frame(frame)

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Soccer AI Video Analysis')
    parser.add_argument('--source_video_path', type=str, required=True)
    parser.add_argument('--target_video_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--mode', type=Mode, default=Mode.BALL_TRACKING)
    parser.add_argument(
        '--fps', type=float, default=None,
        help='Override video FPS for speed calculation (auto-detected if omitted)'
    )
    args = parser.parse_args()
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=args.mode,
        fps=args.fps,
    )
