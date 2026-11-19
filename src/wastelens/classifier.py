"""Application inference for the baseline MLP, improved MLP ensemble and MobileNet."""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .model import DescriptorEnsemble, load_classifier, prepare_classifier_input
from .profiles import DEFAULT_MODEL, MODEL_PROFILES

DEFAULT_CHECKPOINT = MODEL_PROFILES[DEFAULT_MODEL].checkpoint
UNCERTAIN = "uncertain"


@dataclass
class Prediction:
    """Result of classifying one image."""
    label: str                      # Top-1 class, or "uncertain" below the calibrated threshold.
    top_class: str                  # Top-1 class even when the prediction is uncertain.
    confidence: float               # Top-1 probability in [0, 1]; calibration depends on the model.
    probabilities: dict[str, float]  # Every class, in checkpoint order.
    timings_ms: dict[str, float] = field(default_factory=dict)  # Per stage, plus "total".

    @property
    def uncertain(self) -> bool:
        """True when the calibrated confidence is below the model's threshold."""
        return self.label == UNCERTAIN


def to_pil(image: Image.Image | np.ndarray) -> Image.Image:
    """Accept a PIL image or an OpenCV frame. OpenCV arrays are BGR (or grayscale) uint8."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError("OpenCV frames must be uint8")
    if array.ndim == 2:
        return Image.fromarray(array).convert("RGB")
    if array.ndim == 3 and array.shape[2] in (3, 4):
        # Reverse BGR(A) to RGB; forgetting this swaps the blue and red histograms.
        return Image.fromarray(np.ascontiguousarray(array[:, :, 2::-1]))
    raise ValueError(f"Unsupported frame shape {array.shape}")


class WasteClassifier:
    """Load a checkpoint once, then classify many images with per-stage timing."""

    def __init__(self, checkpoint: Path = DEFAULT_CHECKPOINT, device: str = "cpu", warm_up: int = 3):
        """Load the checkpoint, read its threshold and warm the model up."""
        self.model, self.classes = load_classifier(Path(checkpoint), device)
        self.calibrated = isinstance(self.model, DescriptorEnsemble)
        self.threshold = self.model.uncertain_threshold if self.calibrated else 0.0
        self.device = next(self.model.parameters()).device
        # The first calls allocate memory and pick kernels; keep them out of reported timings.
        blank = Image.new("RGB", (512, 384), "white")
        for _ in range(warm_up):
            self.classify(blank)

    @torch.inference_mode()
    def classify(self, image: Image.Image | np.ndarray) -> Prediction:
        """Classify a PIL image or BGR frame, timing input preparation and the model separately."""
        started = time.perf_counter()
        rgb = to_pil(image)
        inputs = prepare_classifier_input(self.model, rgb)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        described = time.perf_counter()
        output = self.model(inputs)
        probabilities = (output if self.calibrated else output.softmax(dim=1)).squeeze(0).cpu().numpy()
        finished = time.perf_counter()
        index = int(probabilities.argmax())
        confidence = float(probabilities[index])
        top_class = self.classes[index]
        return Prediction(
            label=top_class if confidence >= self.threshold else UNCERTAIN,
            top_class=top_class, confidence=confidence,
            probabilities={label: float(value) for label, value in zip(self.classes, probabilities, strict=True)},
            timings_ms={"features": 1000 * (described - started), "model": 1000 * (finished - described),
                        "total": 1000 * (finished - started)},
        )
