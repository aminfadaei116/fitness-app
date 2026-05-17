#!/usr/bin/env python3
"""
Replay a raw video with pose landmarks drawn from a saved keypoints file.

Usage:
    python replay_landmarks.py --video raw_capture.mp4 --keypoints keypoints.pkl
    python replay_landmarks.py --video raw_capture.mp4 --keypoints keypoints.pkl --record
"""

import argparse
import pickle
import time

import cv2
import numpy as np

from fitness_app.pose import draw_pose_from_array


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay video with saved pose landmarks overlaid.")
    parser.add_argument("--video",     required=True, help="Path to the raw video file (e.g. raw_capture.mp4)")
    parser.add_argument("--keypoints", required=True, help="Path to the keypoints .pkl file")
    parser.add_argument("--record",    action="store_true", help="Save annotated output to replay_output.mp4")
    parser.add_argument("--width",     type=int, default=0, help="Resize display width in pixels (0 = original)")
    parser.add_argument("--visibility", type=float, default=0.5,
                        help="Minimum landmark visibility to draw (default: 0.5)")
    args = parser.parse_args()

    with open(args.keypoints, "rb") as f:
        data = pickle.load(f)

    fps: float = data["fps"]
    keypoints: list = data["keypoints"]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if len(keypoints) != total_video_frames:
        print(
            f"Warning: video has {total_video_frames} frames but keypoints has {len(keypoints)} entries. "
            "Replaying up to whichever ends first."
        )

    writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("replay_output.mp4", fourcc, fps, (w, h))
        print("Recording annotated replay to replay_output.mp4")

    delay = max(1, int(1000 / fps))
    frame_idx = 0

    print("Press  q  to quit,  s  to save a screenshot.")

    while frame_idx < len(keypoints):
        ok, frame = cap.read()
        if not ok:
            break

        lm: np.ndarray | None = keypoints[frame_idx]
        if lm is not None:
            draw_pose_from_array(frame, lm, visibility_threshold=args.visibility)
        else:
            cv2.putText(
                frame, "No pose detected",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
            )

        display = frame
        if args.width > 0:
            ratio = args.width / frame.shape[1]
            display = cv2.resize(frame, (args.width, int(frame.shape[0] * ratio)))

        cv2.imshow("Landmark Replay  (q = quit)", display)

        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            filename = f"replay_screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")

        frame_idx += 1

    detected = sum(1 for k in keypoints[:frame_idx] if k is not None)
    print(f"Replayed {frame_idx} frames  ({detected} with pose, {frame_idx - detected} without)")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
