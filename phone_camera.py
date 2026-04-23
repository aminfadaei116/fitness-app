#!/usr/bin/env python3
"""
View your phone's camera feed in real-time via Wi-Fi.

Usage:
    python phone_camera.py --ip 10.88.111.11:8080
    python phone_camera.py --ip 10.88.111.11:8080 --pose
    python phone_camera.py --url http://10.88.111.11:8080/video --pose --record
"""

import argparse
import time
import cv2

from fitness_app.stream import build_url, open_stream
from fitness_app.pose import run_pose


def main() -> None:
    parser = argparse.ArgumentParser(description="Display phone camera; optionally detect 3-D pose.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ip",  help="Phone IP[:port] (e.g. 10.88.111.11:8080)")
    group.add_argument("--url", help="Full stream URL (e.g. http://10.88.111.11:8080/video)")
    parser.add_argument("--port",   type=int, default=8080, help="Port when not embedded in --ip (default: 8080)")
    parser.add_argument("--pose",   action="store_true",   help="Enable MediaPipe 3-D pose landmark detection")
    parser.add_argument("--record", action="store_true",   help="Save stream to output.mp4")
    parser.add_argument("--width",  type=int, default=0,   help="Resize display width in pixels (0 = original)")
    args = parser.parse_args()

    url = args.url if args.url else build_url(args.ip, args.port)
    cap = open_stream(url)

    # Lazy-load MediaPipe only when --pose is requested
    pose = drawing = mp_pose = None
    if args.pose:
        import mediapipe as mp
        mp_pose   = mp.solutions.pose
        drawing   = mp.solutions.drawing_utils
        pose      = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,          # 0=fast, 1=balanced, 2=accurate
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        print("Pose detection ON  (model_complexity=1)")

    writer = None
    if args.record:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("output.mp4", fourcc, fps, (w, h))
        print("Recording to output.mp4")

    print("Press  q  to quit,  s  to save a screenshot.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Stream lost. Trying to reconnect...")
            cap.release()
            cap = open_stream(url)
            continue

        if args.pose:
            frame = run_pose(frame, pose, drawing, mp_pose)

        display = frame
        if args.width > 0:
            ratio   = args.width / frame.shape[1]
            display = cv2.resize(frame, (args.width, int(frame.shape[0] * ratio)))

        cv2.imshow("Phone Camera  (q = quit)", display)

        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            filename = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")

    cap.release()
    if pose:
        pose.close()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
