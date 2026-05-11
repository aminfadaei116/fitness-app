#!/usr/bin/env python3
"""
View your phone's camera feed or a local video file.

Usage:
    python phone_camera.py --ip 10.88.111.11:8080
    python phone_camera.py --ip 10.88.111.11:8080 --pose
    python phone_camera.py --ip 10.88.111.11:8080 --pose mediapipe
    python phone_camera.py --url http://10.88.111.11:8080/video --pose --record
    python phone_camera.py --file /path/to/video.mp4 --pose
    python phone_camera.py --webcam --coach squat
"""

import argparse
import time
import cv2

from fitness_app.stream import build_url, open_stream, open_file, open_webcam
from fitness_app.pose import build_estimator
from fitness_app.coaching import build_coach, draw_coaching_hud


def main() -> None:
    parser = argparse.ArgumentParser(description="Display phone camera or local video; optionally detect 3-D pose.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ip",   help="Phone IP[:port] (e.g. 10.88.111.11:8080)")
    group.add_argument("--url",  help="Full stream URL (e.g. http://10.88.111.11:8080/video)")
    group.add_argument("--file",   metavar="PATH", help="Path to a local video file (e.g. clip.mp4)")
    group.add_argument("--webcam", metavar="ID",   nargs="?", const=0, type=int, default=None,
                       help="Webcam device index (default: 0)")
    parser.add_argument("--port",   type=int, default=8080,       help="Port when not embedded in --ip (default: 8080)")
    parser.add_argument("--pose",   nargs="?", const="mediapipe", default=None, metavar="MODEL",
                        help="Pose estimation model to use (default: mediapipe)")
    parser.add_argument("--record", action="store_true",           help="Save output to output.mp4")
    parser.add_argument("--width",  type=int, default=0,           help="Resize display width in pixels (0 = original)")
    parser.add_argument("--coach", metavar="NAME", choices=("squat",), default=None,
                        help="Heuristic coaching overlay (requires MediaPipe pose)")
    args = parser.parse_args()

    if args.coach:
        if args.pose is None:
            args.pose = "mediapipe"
        elif args.pose != "mediapipe":
            parser.error("--coach requires --pose mediapipe")

    if args.file:
        cap = open_file(args.file)
        live = False
    elif args.webcam is not None:
        cap = open_webcam(args.webcam)
        live = True
    else:
        url = args.url if args.url else build_url(args.ip, args.port)
        cap = open_stream(url)
        live = True

    estimator = build_estimator(args.pose) if args.pose else None
    squat_coach = build_coach(args.coach) if args.coach else None

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
            if not live:
                break  # end of file
            print("Stream lost. Trying to reconnect...")
            cap.release()
            cap = open_stream(url)
            continue

        if estimator:
            frame = estimator.process(frame)

        if squat_coach is not None:
            lm = getattr(estimator, "last_landmarks", None) if estimator else None
            lines = squat_coach.update(frame, lm)
            draw_coaching_hud(frame, lines)

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
    if estimator:
        estimator.close()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
