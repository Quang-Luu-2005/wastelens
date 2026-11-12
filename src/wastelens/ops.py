"""Module 1: traditional OpenCV operations on BGR uint8 images.

Every function takes and returns a BGR uint8 array, so the same code serves the image,
video and webcam modes and the filter-robustness experiment. Default parameters leave
the image unchanged; `apply()` runs the enabled steps in a fixed order:
color -> geometry -> smoothing -> optional synthetic noise.
"""

from dataclasses import asdict, dataclass

import cv2
import numpy as np


# ---------------------------------------------------------------- color transformations

def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Power-law transform out = 255 * (in / 255) ** (1 / gamma); gamma > 1 brightens."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if gamma == 1:
        return image
    # A 256-entry lookup table applies the curve to every pixel in one pass.
    table = np.clip(255.0 * (np.arange(256) / 255.0) ** (1.0 / gamma) + 0.5, 0, 255).astype(np.uint8)
    return cv2.LUT(image, table)


def adjust_brightness_contrast(image: np.ndarray, contrast: float, brightness: float) -> np.ndarray:
    """Linear transform out = contrast * in + brightness, saturated to [0, 255]."""
    if contrast == 1 and brightness == 0:
        return image
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)


def shift_hsv(image: np.ndarray, hue_shift: int, saturation_scale: float, value_scale: float) -> np.ndarray:
    """Rotate hue (OpenCV units, 0-179 per turn) and scale saturation and value."""
    if hue_shift == 0 and saturation_scale == 1 and value_scale == 1:
        return image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180  # Hue is circular.
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * value_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_clahe(image: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization on the Lab lightness channel only."""
    if clip_limit <= 0:
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    # Equalizing L keeps hue and saturation; equalizing B, G, R separately would shift colors.
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Luminance as a three-channel image, so later steps and the classifier still get BGR."""
    return cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------- geometric transformations

def rotate_scale(image: np.ndarray, angle: float, scale: float) -> np.ndarray:
    """Rotate counter-clockwise by `angle` degrees and zoom by `scale` around the image center."""
    if angle == 0 and scale == 1:
        return image
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    # Replicating the border avoids black corners, which would add a false dark region.
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def flip(image: np.ndarray, horizontal: bool, vertical: bool) -> np.ndarray:
    """Mirror left-right and/or top-bottom."""
    if horizontal and vertical:
        return cv2.flip(image, -1)
    if horizontal:
        return cv2.flip(image, 1)
    if vertical:
        return cv2.flip(image, 0)
    return image


def center_crop(image: np.ndarray, keep: float) -> np.ndarray:
    """Keep the central `keep` fraction of width and height (1.0 keeps everything)."""
    if not 0 < keep <= 1:
        raise ValueError("keep must be in (0, 1]")
    if keep == 1:
        return image
    height, width = image.shape[:2]
    crop_h, crop_w = max(1, round(height * keep)), max(1, round(width * keep))
    top, left = (height - crop_h) // 2, (width - crop_w) // 2
    return image[top:top + crop_h, left:left + crop_w].copy()


# ------------------------------------------------------------------------- smoothing

def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian low-pass filter; the kernel covers about +/- 3 sigma."""
    if sigma <= 0:
        return image
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)


def median_blur(image: np.ndarray, kernel: int) -> np.ndarray:
    """Median of a kernel x kernel window; removes salt-and-pepper noise while keeping edges."""
    if kernel <= 1:
        return image
    return cv2.medianBlur(image, kernel if kernel % 2 else kernel + 1)  # OpenCV needs an odd size.


def bilateral_filter(image: np.ndarray, diameter: int, sigma_color: float, sigma_space: float) -> np.ndarray:
    """Edge-preserving smoothing: neighbours are weighted by both distance and color difference."""
    if diameter <= 0:
        return image
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


# ------------------------------------------------- synthetic corruptions for experiments

def gaussian_noise(image: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    """Add zero-mean Gaussian noise with standard deviation `sigma` (in 0-255 units)."""
    if sigma <= 0:
        return image
    noise = np.random.default_rng(seed).normal(0.0, sigma, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def salt_and_pepper(image: np.ndarray, amount: float, seed: int = 0) -> np.ndarray:
    """Set a fraction `amount` of pixels to pure black or white, half each."""
    if amount <= 0:
        return image
    rng = np.random.default_rng(seed)
    noisy = image.copy()
    mask = rng.random(image.shape[:2])
    noisy[mask < amount / 2] = 0
    noisy[(mask >= amount / 2) & (mask < amount)] = 255
    return noisy


# ------------------------------------------------------------------------- pipeline

@dataclass
class OpsParams:
    """Slider values for Module 1. The defaults are the identity transform."""
    gamma: float = 1.0
    contrast: float = 1.0
    brightness: float = 0.0
    hue_shift: int = 0
    saturation_scale: float = 1.0
    value_scale: float = 1.0
    clahe_clip: float = 0.0
    clahe_tiles: int = 8
    grayscale: bool = False
    angle: float = 0.0
    scale: float = 1.0
    flip_horizontal: bool = False
    flip_vertical: bool = False
    crop_keep: float = 1.0
    gaussian_sigma: float = 0.0
    median_kernel: int = 1
    bilateral_diameter: int = 0
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    noise_sigma: float = 0.0
    salt_pepper: float = 0.0
    noise_seed: int = 0

    def changed(self) -> dict:
        """Parameters that differ from the identity defaults, for display and export."""
        defaults = asdict(OpsParams())
        return {key: value for key, value in asdict(self).items() if value != defaults[key]}


def apply(image: np.ndarray, params: OpsParams) -> np.ndarray:
    """Run every enabled Module 1 step in a fixed order and return a new BGR image."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a BGR uint8 image")
    out = image
    # 1. Color
    out = adjust_gamma(out, params.gamma)
    out = adjust_brightness_contrast(out, params.contrast, params.brightness)
    out = shift_hsv(out, params.hue_shift, params.saturation_scale, params.value_scale)
    out = apply_clahe(out, params.clahe_clip, params.clahe_tiles)
    if params.grayscale:
        out = to_grayscale(out)
    # 2. Geometry
    out = rotate_scale(out, params.angle, params.scale)
    out = flip(out, params.flip_horizontal, params.flip_vertical)
    out = center_crop(out, params.crop_keep)
    # 3. Smoothing
    out = gaussian_blur(out, params.gaussian_sigma)
    out = median_blur(out, params.median_kernel)
    out = bilateral_filter(out, params.bilateral_diameter, params.bilateral_sigma_color, params.bilateral_sigma_space)
    # 4. Synthetic noise (for testing how the classifier copes with degraded input)
    out = gaussian_noise(out, params.noise_sigma, params.noise_seed)
    out = salt_and_pepper(out, params.salt_pepper, params.noise_seed)
    return out if out is not image else image.copy()
