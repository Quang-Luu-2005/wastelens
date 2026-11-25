"""Video file mode: Module 1 on every frame, periodic classification and an annotated H.264 output.

Pipeline per frame: decode (OpenCV, which applies rotation metadata) -> downscale to `max_side`
-> Module 1 (`ops.apply`) -> classify every `classify_every` frames -> exponential moving
average (EMA) of the class probabilities -> draw the smoothed label -> encode with PyAV.

With `detect_objects`, the classify step instead runs the webcam pipeline on the processed
frame: SSDLite finds objects, an IoU tracker keeps their IDs, each crop is classified, and the
EMA is kept per object ID. The output video then shows a labelled box per object.
"""

import csv
import io
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import ops

UNCERTAIN = "uncertain"
FALLBACK_FPS = 25.0  # Used when a container reports no usable frame rate.
# Box colours per class in BGR, matching the app's class colours; orange marks "uncertain".
CLASS_BGR = {"cardboard": (6, 119, 217), "glass": (199, 132, 2), "metal": (139, 116, 100),
             "paper": (241, 102, 99), "plastic": (105, 150, 5), "trash": (72, 29, 225), UNCERTAIN: (0, 140, 255)}


@dataclass(frozen=True)
class VideoSettings:
    classify_every: int = 5     # Classify one frame in N; the label is held on the frames between.
    max_seconds: float = 60.0   # Stop after this much input video, so long uploads stay interactive.
    max_side: int = 960         # Longest output side in pixels; frames are never upscaled.
    smoothing: float = 0.5      # EMA weight of the newest prediction; 1.0 disables smoothing.
    detect_objects: bool = False      # Detect and classify each object instead of the whole frame.
    detector_threshold: float = 0.45  # Minimum SSDLite score for a box to be kept.
    max_objects: int = 3              # At most this many boxes per analysed frame.

    def __post_init__(self):
        if self.classify_every < 1 or self.max_seconds <= 0 or self.max_side < 64 or not 0 < self.smoothing <= 1:
            raise ValueError("Invalid video settings")
        if not 0 < self.detector_threshold < 1 or self.max_objects < 1:
            raise ValueError("Invalid detection settings")


@dataclass
class FrameRecord:
    frame: int                  # Index in the input video (0-based).
    time_s: float
    top_class: str              # Raw Top-1 of this frame.
    confidence: float
    smoothed_label: str         # Label drawn on the video: EMA Top-1, or "uncertain" below threshold.
    smoothed_confidence: float
    probabilities: dict[str, float]
    smoothed_probabilities: dict[str, float]
    inference_ms: float


@dataclass
class ObjectRecord:
    """One detected object on one analysed frame (detection mode)."""
    frame: int
    time_s: float
    object_id: int              # Tracker ID; stable while the object keeps overlapping its last box.
    box: tuple[int, int, int, int]
    detector_category: str      # COCO class from SSDLite, independent of the waste class.
    detector_confidence: float
    top_class: str              # Raw Top-1 of this crop.
    confidence: float
    smoothed_label: str         # EMA Top-1 of this object ID, or "uncertain" below threshold.
    smoothed_confidence: float
    probabilities: dict[str, float]
    smoothed_probabilities: dict[str, float]
    inference_ms: float


@dataclass
class VideoResult:
    output: Path
    codec: str
    browser_playable: bool      # H.264 plays in browsers; the MPEG-4 Part 2 fallback may not.
    fps: float
    width: int
    height: int
    frames_read: int
    frames_written: int
    input_frames: int           # As reported by the container; 0 when unknown.
    truncated: bool
    records: list[FrameRecord] = field(default_factory=list)
    stage_ms: dict[str, float] = field(default_factory=dict)  # Totals for decode, module1, classify, encode.
    wall_s: float = 0.0
    detection: bool = False     # True when objects were detected and classified one by one.
    objects: list[ObjectRecord] = field(default_factory=list)
    analysed_frames: int = 0    # Frames on which the classify step ran (with or without detections).

    def label_shares(self) -> dict[str, float]:
        """Share of classified frames per smoothed label, largest first."""
        counts = Counter(record.smoothed_label for record in self.records)
        total = sum(counts.values()) or 1
        return {label: count / total for label, count in counts.most_common()}

    def tracks(self) -> list[dict]:
        """Per object ID: when it was seen, what the detector called it and its dominant waste label."""
        by_id: dict[int, list[ObjectRecord]] = {}
        for record in self.objects:
            by_id.setdefault(record.object_id, []).append(record)
        summaries = []
        for object_id, records in sorted(by_id.items()):
            labels = Counter(r.smoothed_label for r in records)
            confident = [label for label, _ in labels.most_common() if label != UNCERTAIN]
            dominant = confident[0] if confident else UNCERTAIN
            summaries.append({
                "object_id": object_id, "appearances": len(records),
                "first_s": records[0].time_s, "last_s": records[-1].time_s,
                "detector_category": Counter(r.detector_category for r in records).most_common(1)[0][0],
                "dominant_label": dominant, "dominant_share": labels[dominant] / len(records),
                "uncertain_share": labels[UNCERTAIN] / len(records),
                "mean_confidence": float(np.mean([r.smoothed_confidence for r in records])),
            })
        return summaries

    def summary(self) -> dict:
        shares = self.label_shares()
        confident = [label for label in shares if label != UNCERTAIN]
        per_frame = max(self.frames_read, 1)
        inference = [r.inference_ms for r in (self.objects if self.detection else self.records)]
        if self.detection:
            return {
                "mode": "detection", "output_codec": self.codec, "browser_playable": self.browser_playable,
                "fps": self.fps, "width": self.width, "height": self.height,
                "frames_read": self.frames_read, "frames_written": self.frames_written,
                "input_frames": self.input_frames, "truncated": self.truncated,
                "analysed_frames": self.analysed_frames, "detections": len(self.objects),
                "frames_with_objects": len({r.frame for r in self.objects}), "tracks": self.tracks(),
                "mean_ms_per_frame": {stage: total / per_frame for stage, total in self.stage_ms.items()},
                "mean_inference_ms": float(np.mean(inference)) if inference else None,
                "processing_fps": self.frames_read / self.wall_s if self.wall_s else None,
                "wall_seconds": self.wall_s,
            }
        return {
            "mode": "frame",
            "output_codec": self.codec, "browser_playable": self.browser_playable,
            "fps": self.fps, "width": self.width, "height": self.height,
            "frames_read": self.frames_read, "frames_written": self.frames_written,
            "input_frames": self.input_frames, "truncated": self.truncated,
            "classified_frames": len(self.records), "dominant_label": next(iter(shares), None),
            # The most frequent class among frames that were not marked uncertain.
            "dominant_confident_label": confident[0] if confident else None,
            "uncertain_share": shares.get(UNCERTAIN, 0.0), "label_shares": shares,
            "mean_ms_per_frame": {stage: total / per_frame for stage, total in self.stage_ms.items()},
            "mean_inference_ms": float(np.mean(inference)) if inference else None,
            "processing_fps": self.frames_read / self.wall_s if self.wall_s else None,
            "wall_seconds": self.wall_s,
        }


def probe(path: Path) -> dict:
    """Frame rate, frame count, size and duration as reported by OpenCV."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open video")
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if np.isfinite(fps) and 1 <= fps <= 240 else FALLBACK_FPS
        frames = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
        ok, frame = capture.read()
        if not ok:
            raise ValueError("Video has no readable frames")
        height, width = frame.shape[:2]  # From a decoded frame, so rotation metadata is applied.
        return {"fps": fps, "frames": frames, "width": width, "height": height,
                "duration_s": frames / fps if frames else None}
    finally:
        capture.release()


def fit_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    """Downscale to `max_side` keeping the aspect ratio; never upscale; even sizes for yuv420p."""
    scale = min(1.0, max_side / max(width, height))
    return max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2)


def draw_label(frame: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
    """Banner in the top-left corner; ASCII only because OpenCV's font has no Vietnamese glyphs."""
    scale = max(0.45, min(frame.shape[1] / 900, 1.2))
    thickness = max(1, round(scale * 2))
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(frame, (0, 0), (width + 16, height + baseline + 14), color, -1)
    cv2.putText(frame, text, (8, height + 7), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


def draw_objects(frame: np.ndarray, objects: list[tuple[int, tuple, str, float]]) -> np.ndarray:
    """A box per object coloured by its label, with "#id label nn%" on a filled tab above it."""
    scale = max(0.45, min(frame.shape[1] / 1100, 1.0))
    thickness = max(2, round(scale * 3))
    for object_id, (x1, y1, x2, y2), label, confidence in objects:
        color = CLASS_BGR.get(label, CLASS_BGR[UNCERTAIN])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        text = f"#{object_id} {label} {confidence:.0%}"
        (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, thickness - 1))
        top = max(0, y1 - height - baseline - 8)  # Above the box, or inside it at the top edge of the frame.
        cv2.rectangle(frame, (x1, top), (x1 + width + 10, top + height + baseline + 8), color, -1)
        cv2.putText(frame, text, (x1 + 5, top + height + 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                    max(1, thickness - 1), cv2.LINE_AA)
    return frame


def _open_writer(path: Path, fps: float, width: int, height: int):
    """H.264 (libx264) when available, otherwise MPEG-4 Part 2, which some browsers cannot play."""
    import av  # Imported lazily: deployments without video support need not install PyAV.
    # faststart moves the index to the front so browsers can start playback before the end.
    container = av.open(str(path), mode="w", options={"movflags": "faststart"})
    for codec, options in (("libx264", {"crf": "23", "preset": "veryfast"}), ("mpeg4", {"qscale": "3"})):
        try:
            av.codec.Codec(codec, "w")
        except av.codec.codec.UnknownCodecError:  # Encoder missing from this FFmpeg build.
            continue
        # PyAV needs a rational rate; 1001 covers NTSC rates such as 30000/1001.
        stream = container.add_stream(codec, rate=Fraction(fps).limit_denominator(1001))
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.options = options
        return container, stream, codec
    container.close()
    raise RuntimeError("No usable video encoder")


def process_video(source: Path, destination: Path, classifier, params: ops.OpsParams, settings: VideoSettings,
                  progress: Callable[[float, str], None] | None = None, detector=None) -> VideoResult:
    """Run the full pipeline and write the annotated video to `destination` (an .mp4 path).

    `classifier` needs `classify(bgr) -> Prediction`, `classes` and `threshold`
    (as `WasteClassifier` provides); the threshold is applied to the smoothed probabilities.
    `detector` (an `ObjectDetector`) is required when `settings.detect_objects` is set.
    """
    import av

    if settings.detect_objects and detector is None:
        raise ValueError("Object detection needs a detector")
    if settings.detect_objects:
        from .streaming import ObjectTracker, recognize_objects  # Lazy: pulls in torchvision.
        tracker, object_ema, drawn = ObjectTracker(), {}, []
    objects: list[ObjectRecord] = []
    analysed = 0
    info = probe(source)
    fps = info["fps"]
    limit = int(round(settings.max_seconds * fps))
    capture = cv2.VideoCapture(str(source))
    container = stream = None
    smoothed = None
    stages = {"decode": 0.0, "module1": 0.0, "classify": 0.0, "encode": 0.0}
    records: list[FrameRecord] = []
    frames_read = frames_written = 0
    size = None
    label, label_confidence, color = "", 0.0, (95, 104, 0)
    started = time.perf_counter()
    try:
        while frames_read < limit:
            t0 = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                break
            if size is None:
                size = fit_size(frame.shape[1], frame.shape[0], settings.max_side)
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            t1 = time.perf_counter()
            processed = ops.apply(frame, params)
            t2 = time.perf_counter()
            if settings.detect_objects and frames_read % settings.classify_every == 0:
                analysed += 1
                # Module 1 already ran on the whole frame, so crops get no second pass (identity params)
                # and the boxes stay in the coordinates of the frame that is drawn and encoded.
                found = recognize_objects(processed, detector, classifier, tracker, ops.OpsParams(),
                                          threshold=settings.detector_threshold, max_objects=settings.max_objects)
                drawn = []
                for item in found:
                    prediction, object_id = item["prediction"], item["id"]
                    vector = np.array([prediction.probabilities[name] for name in classifier.classes])
                    previous = object_ema.get(object_id)
                    # One EMA per object ID, so two objects in the same frame never mix their evidence.
                    ema = vector if previous is None else settings.smoothing * vector + (1 - settings.smoothing) * previous
                    object_ema[object_id] = ema
                    index = int(ema.argmax())
                    object_label = classifier.classes[index] if ema[index] >= classifier.threshold else UNCERTAIN
                    box = tuple(int(v) for v in item["box"])
                    drawn.append((object_id, box, object_label, float(ema[index])))
                    objects.append(ObjectRecord(
                        frame=frames_read, time_s=frames_read / fps, object_id=object_id, box=box,
                        detector_category=item["detector_category"],
                        detector_confidence=float(item["detector_confidence"]), top_class=prediction.top_class,
                        confidence=prediction.confidence, smoothed_label=object_label,
                        smoothed_confidence=float(ema[index]), probabilities=prediction.probabilities,
                        smoothed_probabilities={name: float(v) for name, v in zip(classifier.classes, ema)},
                        inference_ms=prediction.timings_ms["total"]))
            elif not settings.detect_objects and frames_read % settings.classify_every == 0:
                prediction = classifier.classify(processed)
                vector = np.array([prediction.probabilities[name] for name in classifier.classes])
                # EMA: older evidence decays by (1 - smoothing) per classified frame.
                smoothed = vector if smoothed is None else settings.smoothing * vector + (1 - settings.smoothing) * smoothed
                index = int(smoothed.argmax())
                label_confidence = float(smoothed[index])
                label = classifier.classes[index] if label_confidence >= classifier.threshold else UNCERTAIN
                color = (0, 140, 255) if label == UNCERTAIN else (95, 104, 0)  # BGR orange / teal.
                records.append(FrameRecord(
                    frame=frames_read, time_s=frames_read / fps, top_class=prediction.top_class,
                    confidence=prediction.confidence, smoothed_label=label, smoothed_confidence=label_confidence,
                    probabilities=prediction.probabilities,
                    smoothed_probabilities={name: float(value) for name, value in zip(classifier.classes, smoothed)},
                    inference_ms=prediction.timings_ms["total"]))
            t3 = time.perf_counter()
            # Module 1 geometry may change the size (center crop); encode at the processed size.
            if container is None:
                out_size = fit_size(processed.shape[1], processed.shape[0], max(processed.shape[:2]))
                container, stream, codec = _open_writer(destination, fps, *out_size)
            if (processed.shape[1], processed.shape[0]) != (stream.width, stream.height):
                processed = cv2.resize(processed, (stream.width, stream.height), interpolation=cv2.INTER_AREA)
            if settings.detect_objects:
                # Boxes from the last analysed frame are held until the next one, like the frame label.
                overlay = draw_objects(processed.copy(), drawn)
                overlay = draw_label(overlay, f"{len(drawn)} objects  t={frames_read / fps:.1f}s", (60, 60, 60))
            else:
                overlay = draw_label(processed.copy(), f"{label} {label_confidence:.0%}  t={frames_read / fps:.1f}s", color)
            for packet in stream.encode(av.VideoFrame.from_ndarray(overlay, format="bgr24")):
                container.mux(packet)
            frames_written += 1
            t4 = time.perf_counter()
            stages["decode"] += 1000 * (t1 - t0)
            stages["module1"] += 1000 * (t2 - t1)
            stages["classify"] += 1000 * (t3 - t2)
            stages["encode"] += 1000 * (t4 - t3)
            frames_read += 1
            if progress and (frames_read % 10 == 0 or frames_read == limit):
                total = min(limit, info["frames"] or limit)
                progress(min(frames_read / total, 1.0), f"{frames_read}/{total} khung hình")
        if container is None:
            raise ValueError("Video has no readable frames")
        for packet in stream.encode():  # Flush frames buffered inside the encoder.
            container.mux(packet)
        truncated = frames_read == limit and capture.read()[0]
    finally:
        capture.release()
        if container is not None:
            container.close()
    return VideoResult(
        output=destination, codec=codec, browser_playable=codec == "libx264", fps=fps,
        width=stream.width, height=stream.height, frames_read=frames_read, frames_written=frames_written,
        input_frames=info["frames"], truncated=bool(truncated), records=records, stage_ms=stages,
        wall_s=time.perf_counter() - started, detection=settings.detect_objects, objects=objects,
        analysed_frames=analysed if settings.detect_objects else len(records))


def records_csv(result: VideoResult, classes: list[str]) -> str:
    """One row per classified frame: raw and smoothed predictions with all class probabilities."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["frame", "time_s", "top_class", "confidence", "smoothed_label", "smoothed_confidence",
                     "inference_ms", *[f"p_{name}" for name in classes], *[f"ema_{name}" for name in classes]])
    for record in result.records:
        writer.writerow([record.frame, f"{record.time_s:.3f}", record.top_class, f"{record.confidence:.6f}",
                         record.smoothed_label, f"{record.smoothed_confidence:.6f}", f"{record.inference_ms:.2f}",
                         *[f"{record.probabilities[name]:.6f}" for name in classes],
                         *[f"{record.smoothed_probabilities[name]:.6f}" for name in classes]])
    return buffer.getvalue()


def objects_csv(result: VideoResult, classes: list[str]) -> str:
    """One row per detected object per analysed frame, with its box and all class probabilities."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["frame", "time_s", "object_id", "x1", "y1", "x2", "y2", "detector_category",
                     "detector_confidence", "top_class", "confidence", "smoothed_label", "smoothed_confidence",
                     "inference_ms", *[f"p_{name}" for name in classes], *[f"ema_{name}" for name in classes]])
    for record in result.objects:
        writer.writerow([record.frame, f"{record.time_s:.3f}", record.object_id, *record.box,
                         record.detector_category, f"{record.detector_confidence:.4f}", record.top_class,
                         f"{record.confidence:.6f}", record.smoothed_label, f"{record.smoothed_confidence:.6f}",
                         f"{record.inference_ms:.2f}", *[f"{record.probabilities[name]:.6f}" for name in classes],
                         *[f"{record.smoothed_probabilities[name]:.6f}" for name in classes]])
    return buffer.getvalue()


def read_frame(path: Path, index: int) -> np.ndarray | None:
    """Decode one frame of a written video (for snapshots)."""
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        return frame if ok else None
    finally:
        capture.release()


def settings_dict(settings: VideoSettings) -> dict:
    return asdict(settings)


def make_sample_video(images: list[tuple[str, Path]], destination: Path, seconds_each: float = 2.0,
                      fps: int = 15) -> list[dict]:
    """Build a labelled demo clip: each (label, image) is shown with a slow zoom (Ken Burns effect).

    Returns the segments [{"label", "start_s", "end_s"}] so results can be checked against labels.
    """
    import av

    frames_each = max(1, round(seconds_each * fps))
    segments, size = [], None
    with av.open(str(destination), mode="w", options={"movflags": "faststart"}) as container:
        stream = None
        for index, (label, path) in enumerate(images):
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Cannot read {path}")
            if size is None:
                size = fit_size(image.shape[1], image.shape[0], 640)
                stream = container.add_stream("libx264", rate=fps)
                stream.width, stream.height, stream.pix_fmt = *size, "yuv420p"
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            center = (size[0] / 2, size[1] / 2)
            for step in range(frames_each):
                # Zoom from 1.00 to 1.08 so consecutive frames differ like real camera motion.
                matrix = cv2.getRotationMatrix2D(center, 0, 1 + 0.08 * step / frames_each)
                frame = cv2.warpAffine(image, matrix, size, borderMode=cv2.BORDER_REPLICATE)
                for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="bgr24")):
                    container.mux(packet)
            segments.append({"label": label, "start_s": index * frames_each / fps,
                             "end_s": (index + 1) * frames_each / fps})
        for packet in stream.encode():
            container.mux(packet)
    return segments


def segment_agreement(result: VideoResult, segments: list[dict]) -> dict:
    """Compare classified frames with the known segment labels.

    Raw Top-1 is right or wrong. Smoothed labels can also abstain ("uncertain"), so they are
    split into correct, uncertain and wrong; the three shares sum to 1.
    """
    def truth(time_s):
        return next((s["label"] for s in segments if s["start_s"] <= time_s < s["end_s"]), None)
    pairs = [(record, truth(record.time_s)) for record in result.records]
    pairs = [(record, label) for record, label in pairs if label is not None]
    if not pairs:
        return {"frames": 0, "raw_top1": None, "smoothed": None, "smoothed_uncertain": None, "smoothed_wrong": None}
    total = len(pairs)
    correct = sum(r.smoothed_label == label for r, label in pairs)
    uncertain = sum(r.smoothed_label == UNCERTAIN for r, _ in pairs)
    return {"frames": total,
            "raw_top1": sum(r.top_class == label for r, label in pairs) / total,
            "smoothed": correct / total, "smoothed_uncertain": uncertain / total,
            "smoothed_wrong": (total - correct - uncertain) / total}
