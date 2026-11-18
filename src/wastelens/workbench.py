"""Processing presets and result identity for the interactive image workbench."""

import hashlib
import json
from dataclasses import asdict

from .ops import OpsParams

# Presets are user-selectable experiments, not an automatically optimal preprocessing policy.
PRESETS = {
    "Gốc": OpsParams(),
    "Nhẹ": OpsParams(gaussian_sigma=0.4),
    "Vừa": OpsParams(gamma=1.15, clahe_clip=2.0, gaussian_sigma=0.8),
    "Mạnh": OpsParams(gamma=1.25, contrast=1.15, clahe_clip=3.0, gaussian_sigma=1.2),
}
PRESET_DESCRIPTIONS = {
    "Gốc": "Giữ nguyên ảnh, không áp dụng bộ lọc.",
    "Nhẹ": "Làm mượt nhẹ với Gaussian σ=0,4.",
    "Vừa": "Gamma 1,15 · CLAHE 2,0 · Gaussian σ=0,8.",
    "Mạnh": "Gamma 1,25 · tương phản 1,15 · CLAHE 3,0 · Gaussian σ=1,2.",
}


def request_signature(image_bytes: bytes, params: OpsParams, model_key: str, checkpoint_version: int) -> str:
    """Bind a result to the exact input, processing settings and versioned model file."""
    settings = json.dumps({"params": asdict(params), "model": model_key, "version": checkpoint_version},
                          sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(image_bytes + settings.encode("utf-8")).hexdigest()


def describe_operations(params: OpsParams) -> str:
    """Describe enabled operations only; inactive filter settings are not operations."""
    steps = []
    if params.gamma != 1:
        steps.append(f"Gamma {params.gamma:g}")
    if params.contrast != 1:
        steps.append(f"Tương phản {params.contrast:g}")
    if params.brightness != 0:
        steps.append(f"Độ sáng {params.brightness:+g}")
    if params.hue_shift != 0 or params.saturation_scale != 1 or params.value_scale != 1:
        steps.append("HSV")
    if params.clahe_clip > 0:
        steps.append(f"CLAHE {params.clahe_clip:g}")
    if params.grayscale:
        steps.append("Ảnh xám")
    if params.angle != 0:
        steps.append(f"Xoay {params.angle:g}°")
    if params.scale != 1:
        steps.append(f"Phóng {params.scale:g}×")
    if params.flip_horizontal:
        steps.append("Lật ngang")
    if params.flip_vertical:
        steps.append("Lật dọc")
    if params.crop_keep != 1:
        steps.append(f"Crop {params.crop_keep:.0%}")
    if params.gaussian_sigma > 0:
        steps.append(f"Gaussian σ={params.gaussian_sigma:g}")
    if params.median_kernel > 1:
        steps.append(f"Median {params.median_kernel}×{params.median_kernel}")
    if params.bilateral_diameter > 0:
        steps.append(f"Bilateral d={params.bilateral_diameter}")
    if params.noise_sigma > 0:
        steps.append(f"Nhiễu Gaussian σ={params.noise_sigma:g}")
    if params.salt_pepper > 0:
        steps.append(f"Nhiễu muối tiêu {params.salt_pepper:.1%}")
    return " · ".join(steps) or "Ảnh gốc · không áp dụng biến đổi"
