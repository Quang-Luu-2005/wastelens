"""Two-stage webcam inference: COCO detection, then TrashNet recognition per crop."""

import base64
import math
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import cv2
import numpy as np
import torch
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large

from . import ops

DETECTOR_WEIGHTS = Path(__file__).resolve().parents[2] / "models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth"


@dataclass
class Detection:
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels of the analysed frame.
    score: float                    # Detector confidence in [0, 1].
    category: str                   # COCO class name, shown separately from the waste class.


def clip_box(box, width: int, height: int, padding: float = 0):
    """Grow a box by `padding` of its size, clamp it to the frame, and reject empty or invalid boxes."""
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        return None
    x1, y1, x2, y2 = map(float, box)
    if x2 <= x1 or y2 <= y1:
        return None
    # A small margin keeps the object's edges inside the crop, as in TrashNet's framing.
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    # Round outwards (floor/ceil) so the clamped crop never cuts into the detected box.
    result = (max(0, min(width, math.floor(x1 - dx))), max(0, min(height, math.floor(y1 - dy))),
              max(0, min(width, math.ceil(x2 + dx))), max(0, min(height, math.ceil(y2 + dy))))
    return result if result[2] > result[0] and result[3] > result[1] else None


def decode_frame(payload: dict) -> np.ndarray:
    """Bound both the compressed payload and image dimensions before OpenCV decoding."""
    from PIL import Image
    import io

    encoded = payload.get("jpeg")
    # The browser sends base64 JPEG text; reject oversized payloads before decoding anything.
    if not isinstance(encoded, str) or len(encoded) > 2_000_000:
        raise ValueError("Khung hình không hợp lệ hoặc quá lớn.")
    try:
        raw = base64.b64decode(encoded, validate=True)
        # PIL reads the header first, so format and size are checked before pixels are decoded.
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "JPEG" or max(image.size) > 1280 or min(image.size) < 16:
                raise ValueError("Khung hình phải là JPEG, kích thước từ 16 đến 1280 pixel.")
            rgb = np.asarray(image.convert("RGB"))
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError("Không đọc được khung hình camera.") from error
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class ObjectDetector:
    """Pretrained COCO detector (SSDLite320 MobileNetV3) that only locates objects for the classifier."""

    def __init__(self, checkpoint: Path = DETECTOR_WEIGHTS):
        if not checkpoint.is_file():
            raise FileNotFoundError("Chạy .venv\\Scripts\\python.exe scripts/prepare_detector.py để tải bộ phát hiện vật.")
        # Build the architecture without downloading anything, then load the hash-checked local weights.
        self.model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=91).eval()
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        self.categories = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT.meta["categories"]
        # Streamlit may call the cached detector from several sessions at once.
        self.lock = RLock()

    def detect(self, frame: np.ndarray, threshold: float = .45, max_objects: int = 3) -> list[Detection]:
        """Boxes above `threshold`, most confident first, at most `max_objects`, people excluded."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV frames are BGR; torchvision expects RGB.
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255)  # HWC uint8 -> CHW float in [0, 1].
        with self.lock, torch.inference_mode():
            output = self.model([tensor])[0]
        height, width = frame.shape[:2]
        found = []
        # torchvision returns detections sorted by score, so the first matches are the most confident.
        for box, score, label in zip(output["boxes"].tolist(), output["scores"].tolist(), output["labels"].tolist()):
            if score < threshold or label == 1:  # Do not classify the camera user's body as waste.
                continue
            clipped = clip_box(box, width, height)
            if clipped:
                found.append(Detection(clipped, score, self.categories[label]))
            if len(found) >= max_objects:
                break
        return found


def iou(a, b):
    """Intersection over union of two (x1, y1, x2, y2) boxes, 0 when they do not overlap."""
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.


class ObjectTracker:
    """Short-lived IoU association; IDs are session-local, not persistent object identities."""
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, detections: list[Detection]) -> list[int]:
        """Return an ID per detection, reusing a track's ID when the boxes overlap enough."""
        # Greedy one-to-one matching: consider every (detection, track) pair from the highest IoU down.
        candidates = sorted(((iou(d.box, track[0]), index, object_id)
                             for index, d in enumerate(detections) for object_id, track in self.tracks.items()), reverse=True)
        assigned, used = {}, set()
        for overlap, index, object_id in candidates:
            if overlap >= .25 and index not in assigned and object_id not in used:
                assigned[index] = object_id
                used.add(object_id)
        # Unmatched tracks survive two more analysed frames, so a briefly missed object keeps its ID.
        previous = {key: (box, age + 1) for key, (box, age) in self.tracks.items() if key not in used and age < 2}
        for index, detection in enumerate(detections):
            if index not in assigned:  # A new object gets the next free ID.
                assigned[index] = self.next_id
                self.next_id += 1
            previous[assigned[index]] = (detection.box, 0)
        self.tracks = previous
        return [assigned[index] for index in range(len(detections))]


def recognize_objects(frame, detector, classifier, tracker, params, threshold=.45, max_objects=3):
    """Detect objects, keep their IDs across frames, and classify each crop into a waste class.

    Module 1 (`params`) is applied to every crop before classification; box coordinates
    stay in the original frame so the browser can draw them over the live video.
    """
    detections = detector.detect(frame, threshold=threshold, max_objects=max_objects)
    identifiers = tracker.update(detections)
    height, width = frame.shape[:2]
    objects = []
    for detection, object_id in zip(detections, identifiers):
        crop_box = clip_box(detection.box, width, height, padding=.04)  # 4% margin on each side.
        if crop_box is None:
            continue
        x1, y1, x2, y2 = crop_box
        processed = ops.apply(frame[y1:y2, x1:x2].copy(), params)  # Copy: Module 1 must not alter the frame.
        prediction = classifier.classify(processed)
        objects.append({"id": object_id, "box": list(detection.box), "detector_category": detection.category,
                        "detector_confidence": detection.score, "prediction": prediction})
    return objects
