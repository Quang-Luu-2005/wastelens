"""Filter x model experiment: how OpenCV operations change the predictions of each classifier.

Three families of conditions, all applied to the original 512 x 384 test photos exactly as
Module 1 applies them in the app (the classifiers resize afterwards):

1. Module 1 filters on clean images: the app presets and single operations.
2. Corruptions at five severities (ImageNet-C style): noise, blur, illumination, colour, geometry.
3. Rescue: a fixed corruption followed by a restoration filter, beside the corruption alone
   and the restoration filter applied to clean images.

Each condition is a function (BGR uint8 image, per-image seed) -> BGR uint8 image.
"""

from dataclasses import dataclass, replace
from typing import Callable

import cv2
import numpy as np
import torch

from . import ops
from .model import DescriptorEnsemble, prepare_classifier_input

Transform = Callable[[np.ndarray, int], np.ndarray]


# ------------------------------------------------------------ extra operations (not in Module 1)

def low_contrast(image: np.ndarray, alpha: float) -> np.ndarray:
    """Blend towards mid-grey: out = alpha * in + (1 - alpha) * 128, so contrast falls but the mean stays."""
    return cv2.addWeighted(image, alpha, np.full_like(image, 128), 1 - alpha, 0)


def color_cast(image: np.ndarray, strength: float) -> np.ndarray:
    """Warm white-balance error: scale red up and blue down by `strength` (BGR channel order)."""
    gains = np.array([1 - strength, 1.0, 1 + strength], np.float32)
    return np.clip(image.astype(np.float32) * gains, 0, 255).astype(np.uint8)


def gray_world(image: np.ndarray) -> np.ndarray:
    """Gray-world white balance: scale each channel so its mean equals the mean over all channels."""
    means = image.reshape(-1, 3).mean(axis=0)
    gains = means.mean() / np.maximum(means, 1e-6)
    return np.clip(image.astype(np.float32) * gains, 0, 255).astype(np.uint8)


def with_params(**changes) -> Transform:
    """A Module 1 configuration as a transform; noise uses the per-image seed."""
    params = ops.OpsParams(**changes)
    return lambda image, seed: ops.apply(image, replace(params, noise_seed=seed))


# ----------------------------------------------------------------------------- conditions

@dataclass(frozen=True)
class Condition:
    name: str          # Unique identifier used in tables.
    family: str        # "clean", "filter", "corruption", "rescue".
    group: str         # Corruption or filter type, for grouping curves and tables.
    level: float       # Severity or parameter value (0 for the clean reference).
    severity: int      # 1-5 for corruptions, 0 otherwise.
    label: str         # Vietnamese description for figures and the report.
    transform: Transform


# Five severities per corruption, chosen on 512 x 384 photos so level 1 is subtle and level 5 strong.
CORRUPTIONS = {
    "gaussian_noise": ("Nhiễu Gaussian σ", [5, 10, 20, 35, 50], lambda v: with_params(noise_sigma=v)),
    "salt_pepper": ("Nhiễu muối tiêu (tỉ lệ)", [0.01, 0.03, 0.06, 0.10, 0.15], lambda v: with_params(salt_pepper=v)),
    "gaussian_blur": ("Gaussian blur σ", [1, 2, 3, 5, 8], lambda v: with_params(gaussian_sigma=v)),
    "low_contrast": ("Giảm tương phản α", [0.8, 0.6, 0.45, 0.3, 0.2], lambda v: lambda image, seed: low_contrast(image, v)),
    "darken": ("Làm tối (hệ số V)", [0.8, 0.6, 0.45, 0.3, 0.2], lambda v: with_params(value_scale=v)),
    "hue_shift": ("Dịch hue (đơn vị OpenCV)", [5, 10, 20, 30, 45], lambda v: with_params(hue_shift=v)),
    "desaturate": ("Giảm bão hòa (hệ số S)", [0.8, 0.6, 0.4, 0.2, 0.0], lambda v: with_params(saturation_scale=v)),
    "color_cast": ("Ám màu ấm (cường độ)", [0.05, 0.1, 0.15, 0.2, 0.3], lambda v: lambda image, seed: color_cast(image, v)),
    "rotation": ("Xoay (độ)", [10, 20, 45, 90, 180], lambda v: with_params(angle=v)),
}

FILTERS = [
    # (name, group, label, transform) - Module 1 operations on clean images.
    ("preset_nhe", "preset", "Preset Nhẹ (Gaussian σ=0,4)", with_params(gaussian_sigma=0.4)),
    ("preset_vua", "preset", "Preset Vừa (gamma 1,15 · CLAHE 2 · Gaussian σ=0,8)",
     with_params(gamma=1.15, clahe_clip=2.0, gaussian_sigma=0.8)),
    ("preset_manh", "preset", "Preset Mạnh (gamma 1,25 · tương phản 1,15 · CLAHE 3 · Gaussian σ=1,2)",
     with_params(gamma=1.25, contrast=1.15, clahe_clip=3.0, gaussian_sigma=1.2)),
    ("gaussian_1", "smoothing", "Gaussian σ=1", with_params(gaussian_sigma=1.0)),
    ("median_5", "smoothing", "Median 5×5", with_params(median_kernel=5)),
    ("bilateral_9", "smoothing", "Bilateral d=9, σ=75", with_params(bilateral_diameter=9)),
    ("clahe_2", "color", "CLAHE clip 2", with_params(clahe_clip=2.0)),
    ("clahe_4", "color", "CLAHE clip 4", with_params(clahe_clip=4.0)),
    ("gamma_1.5", "color", "Gamma 1,5 (sáng hơn)", with_params(gamma=1.5)),
    ("gamma_0.67", "color", "Gamma 0,67 (tối hơn)", with_params(gamma=0.67)),
    ("gray_world", "color", "Cân bằng trắng gray-world", lambda image, seed: gray_world(image)),
    ("grayscale", "color", "Ảnh xám", with_params(grayscale=True)),
    ("flip_h", "geometry", "Lật ngang", with_params(flip_horizontal=True)),
    ("crop_0.8", "geometry", "Crop giữ 80% trung tâm", with_params(crop_keep=0.8)),
]

RESTORERS = {
    "median_3": ("Median 3×3", with_params(median_kernel=3)),
    "median_5": ("Median 5×5", with_params(median_kernel=5)),
    "gaussian_1": ("Gaussian σ=1", with_params(gaussian_sigma=1.0)),
    "bilateral_9": ("Bilateral d=9", with_params(bilateral_diameter=9)),
    "gamma_2": ("Gamma 2,0 (làm sáng)", with_params(gamma=2.0)),
    "clahe_3": ("CLAHE clip 3", with_params(clahe_clip=3.0)),
    "gray_world": ("Gray-world", lambda image, seed: gray_world(image)),
}
# (corruption, severity 1-5, restorers): the classic pairings of a degradation with its textbook fix.
RESCUES = [
    ("salt_pepper", 3, ["median_3", "median_5"]),
    ("gaussian_noise", 4, ["gaussian_1", "bilateral_9", "median_5"]),
    ("darken", 4, ["gamma_2", "clahe_3"]),
    ("low_contrast", 4, ["clahe_3"]),
    ("color_cast", 4, ["gray_world"]),
]


def build_conditions() -> list[Condition]:
    """Every condition in a fixed order: clean, filters, corruptions, rescues."""
    conditions = [Condition("clean", "clean", "clean", 0, 0, "Ảnh gốc", lambda image, seed: image)]
    for name, group, label, transform in FILTERS:
        conditions.append(Condition(name, "filter", group, 0, 0, label, transform))
    for name, (label, levels, factory) in CORRUPTIONS.items():
        for severity, level in enumerate(levels, start=1):
            conditions.append(Condition(f"{name}_{severity}", "corruption", name, level, severity,
                                        f"{label} = {level:g}", factory(level)))
    # Controls: each restoration filter applied to clean images, to separate "repair" from side effects.
    for restorer in dict.fromkeys(name for _, _, names in RESCUES for name in names):
        restore_label, restore = RESTORERS[restorer]
        conditions.append(Condition(f"control_{restorer}", "control", restorer, 0, 0,
                                    f"Ảnh sạch → {restore_label}", restore))
    for corruption, severity, restorers in RESCUES:
        label, levels, factory = CORRUPTIONS[corruption]
        damage = factory(levels[severity - 1])
        for restorer in restorers:
            restore_label, restore = RESTORERS[restorer]
            conditions.append(Condition(
                f"rescue_{corruption}_{severity}_{restorer}", "rescue", corruption, levels[severity - 1], severity,
                f"{label} = {levels[severity - 1]:g} → {restore_label}",
                # Bind the loop variables now; a plain closure would see only the last pair.
                lambda image, seed, damage=damage, restore=restore: restore(damage(image, seed), seed)))
    return conditions


# ----------------------------------------------------------------------------- inference

@torch.inference_mode()
def predict_batch(model: torch.nn.Module, images_bgr: list[np.ndarray], batch_size: int = 64) -> np.ndarray:
    """Class probabilities (N, classes) through the same input path as the app."""
    from PIL import Image
    outputs = []
    for start in range(0, len(images_bgr), batch_size):
        inputs = [prepare_classifier_input(model, Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
                  for image in images_bgr[start:start + batch_size]]
        if isinstance(inputs[0], dict):
            batch = {key: torch.cat([item[key] for item in inputs]) for key in inputs[0]}
        else:
            batch = torch.cat(inputs)
        output = model(batch)
        # The descriptor ensemble already returns calibrated probabilities; the others return logits.
        probabilities = output if isinstance(model, DescriptorEnsemble) else output.softmax(dim=1)
        outputs.append(probabilities.cpu().numpy())
    return np.concatenate(outputs)
