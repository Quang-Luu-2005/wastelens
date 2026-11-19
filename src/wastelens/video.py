"""Video file mode: Module 1 on every frame, periodic classification and an annotated H.264 output.

Pipeline per frame: decode (OpenCV, which applies rotation metadata) -> downscale to `max_side`
-> Module 1 (`ops.apply`) -> classify every `classify_every` frames -> exponential moving
average (EMA) of the class probabilities -> draw the smoothed label -> encode with PyAV.
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


@dataclass(frozen=True)
class VideoSettings:
    classify_every: int = 5     # Classify one frame in N; the label is held on the frames between.
    max_seconds: float = 60.0   # Stop after this much input video, so long uploads stay interactive.
    max_side: int = 960         # Longest output side in pixels; frames are never upscaled.
    smoothing: float = 0.5      # EMA weight of the newest prediction; 1.0 disables smoothing.

    def __post_init__(self):
        if self.classify_every < 1 or self.max_seconds <= 0 or self.max_side < 64 or not 0 < self.smoothing <= 1:
            raise ValueError("Invalid video settings")


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

    def label_shares(self) -> dict[str, float]:
        """Share of classified frames per smoothed label, largest first."""
        counts = Counter(record.smoothed_label for record in self.records)
        total = sum(counts.values()) or 1
        return {label: count / total for label, count in counts.most_common()}

    def summary(self) -> dict:
        shares = self.label_shares()
        confident = [label for label in shares if label != UNCERTAIN]
        per_frame = max(self.frames_read, 1)
        return {
            "output_codec": self.codec, "browser_playable": self.browser_playable,
            "fps": self.fps, "width": self.width, "height": self.height,
            "frames_read": self.frames_read, "frames_written": self.frames_written,
            "input_frames": self.input_frames, "truncated": self.truncated,
            "classified_frames": len(self.records), "dominant_label": next(iter(shares), None),
            # The most frequent class among frames that were not marked uncertain.
            "dominant_confident_label": confident[0] if confident else None,
            "uncertain_share": shares.get(UNCERTAIN, 0.0), "label_shares": shares,
            "mean_ms_per_frame": {stage: total / per_frame for stage, total in self.stage_ms.items()},
            "mean_inference_ms": (float(np.mean([r.inference_ms for r in self.records])) if self.records else None),
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
                  progress: Callable[[float, str], None] | None = None) -> VideoResult:
    """Run the full pipeline and write the annotated video to `destination` (an .mp4 path).

    `classifier` needs `classify(bgr) -> Prediction`, `classes` and `threshold`
    (as `WasteClassifier` provides); the threshold is applied to the smoothed probabilities.
    """
    import av

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
            if frames_read % settings.classify_every == 0:
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
        wall_s=time.perf_counter() - started)


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
