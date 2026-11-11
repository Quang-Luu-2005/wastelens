"""Fixed OpenCV descriptors for the from-scratch MLP; no learned feature extractor."""

import numpy as np
from PIL import Image


PIXEL_FEATURES = 3 * 64 * 64
DESCRIPTOR_FEATURES = 1764 + 21 * 3 * 8


def feature_dimension(representation: str) -> int:
    dimensions = {"pixels": PIXEL_FEATURES, "hog_hsv": DESCRIPTOR_FEATURES,
                  "hybrid": PIXEL_FEATURES + DESCRIPTOR_FEATURES}
    return dimensions[representation]


def image_descriptor(image: Image.Image, representation: str) -> np.ndarray:
    """HOG 8x8 cells plus 8-bin/channel HSV histograms on a 1/2/4 spatial pyramid."""
    import cv2

    rgb = np.asarray(image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR))
    if representation == "pixels":
        return rgb.transpose(2, 0, 1).reshape(-1).astype(np.float32) / 255
    if representation not in {"hog_hsv", "hybrid"}:
        raise ValueError(f"Unknown representation: {representation}")
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    gradients = hog.compute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)).reshape(-1)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    parts = [gradients]
    for cells in (1, 2, 4):
        cell_size = 64 // cells
        for row in range(cells):
            for column in range(cells):
                region = hsv[row * cell_size:(row + 1) * cell_size,
                             column * cell_size:(column + 1) * cell_size]
                for channel, maximum in enumerate((180, 256, 256)):
                    histogram = cv2.calcHist([region], [channel], None, [8], [0, maximum]).reshape(-1)
                    parts.append(histogram / (cell_size * cell_size))
    descriptor = np.concatenate(parts).astype(np.float32)
    assert len(descriptor) == DESCRIPTOR_FEATURES
    if representation == "hybrid":
        pixels = rgb.transpose(2, 0, 1).reshape(-1).astype(np.float32) / 255
        descriptor = np.concatenate((pixels, descriptor))
    return descriptor
