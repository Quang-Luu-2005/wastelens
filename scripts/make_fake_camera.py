"""Write a short Y4M clip of one TrashNet test photo for Chrome's fake webcam.

Chrome can replace the webcam with a file (--use-file-for-fake-video-capture), which makes the
webcam tab reproducible for screenshots and tests. The photo is centred on a 1280 x 720 frame
and drifts a few pixels left and right so that the stream is not a frozen frame.

    .venv\\Scripts\\python.exe scripts/make_fake_camera.py outputs/fake-camera.y4m plastic/plastic73.jpg
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def frames(photo: np.ndarray, count: int, drift: int = 8):
    """Yield BGR frames: the photo resized to 960 x 720, centred, swinging by up to `drift` pixels."""
    photo = cv2.resize(photo, (960, 720))
    # Fill the side bars with the photo's own edge colour so the frame looks like one background.
    edge = int(np.median(photo[:, :8].reshape(-1, 3), axis=0).mean())
    for index in range(count):
        shift = int(drift * np.sin(2 * np.pi * index / count))
        canvas = np.full((720, 1280, 3), edge, np.uint8)
        canvas[:, 160 + shift:1120 + shift] = photo
        yield canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path, help="destination .y4m file")
    parser.add_argument("image", help="path under data/trashnet, e.g. plastic/plastic73.jpg")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    photo = cv2.imread(str(ROOT / "data/trashnet" / args.image))
    if photo is None:
        raise SystemExit(f"Cannot read data/trashnet/{args.image}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = max(1, round(args.seconds * args.fps))
    with args.output.open("wb") as file:
        # YUV4MPEG2 header, then one 'FRAME' marker and planar I420 bytes per frame.
        file.write(f"YUV4MPEG2 W1280 H720 F{args.fps}:1 Ip A1:1 C420jpeg\n".encode())
        for frame in frames(photo, count):
            file.write(b"FRAME\n")
            file.write(cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420).tobytes())
    print(f"Wrote {count} frames to {args.output}")


if __name__ == "__main__":
    main()
