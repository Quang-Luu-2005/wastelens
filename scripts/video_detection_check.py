"""Run the video pipeline with per-object detection on real clips and print a per-object summary.

    .venv\\Scripts\\python.exe scripts/video_detection_check.py outputs/test-videos/*.mp4 --output outputs/video-detection
"""
import argparse
import json
from pathlib import Path

from wastelens.classifier import WasteClassifier
from wastelens.ops import OpsParams
from wastelens.profiles import MODEL_PROFILES
from wastelens.streaming import DETECTOR_WEIGHTS, ObjectDetector
from wastelens.video import VideoSettings, objects_csv, process_video


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/video-detection"))
    parser.add_argument("--models", nargs="+", default=list(MODEL_PROFILES), choices=list(MODEL_PROFILES))
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--max-side", type=int, default=960)
    parser.add_argument("--max-objects", type=int, default=3)
    parser.add_argument("--tag", default="", help="suffix for output names, to keep runs with other settings apart")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    detector = ObjectDetector(DETECTOR_WEIGHTS)
    settings = VideoSettings(classify_every=args.every, max_side=args.max_side, detect_objects=True,
                             detector_threshold=args.threshold, max_objects=args.max_objects)
    report = {}
    for video in args.videos:
        for key in args.models:
            classifier = WasteClassifier(MODEL_PROFILES[key].checkpoint)
            stem = f"{video.stem}-{key}{args.tag}"
            destination = args.output / f"{stem}.mp4"
            result = process_video(video, destination, classifier, OpsParams(), settings, detector=detector)
            (args.output / f"{stem}.csv").write_text(objects_csv(result, classifier.classes), encoding="utf-8")
            summary = result.summary()
            report[f"{video.name}/{key}"] = summary
            print(f"== {video.name} · {key}: {summary['analysed_frames']} khung phân tích, "
                  f"{summary['frames_with_objects']} khung có vật, {summary['detections']} phát hiện, "
                  f"{summary['processing_fps']:.1f} khung/s")
            for track in summary["tracks"]:
                print(f"   #{track['object_id']:<3} {track['detector_category']:<14} {track['appearances']:>3} lần "
                      f"{track['first_s']:5.1f}-{track['last_s']:5.1f}s  -> {track['dominant_label']:<10} "
                      f"{track['dominant_share']:.0%} (không chắc {track['uncertain_share']:.0%}, "
                      f"conf TB {track['mean_confidence']:.2f})")
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                                             encoding="utf-8")


if __name__ == "__main__":
    main()
