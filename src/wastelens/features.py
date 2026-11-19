"""Fixed OpenCV descriptors for the from-scratch MLP; no learned feature extractor."""

import numpy as np
from PIL import Image, ImageOps


PIXEL_FEATURES = 3 * 64 * 64  # Raw RGB pixels of a 64 x 64 image, as in the Lab02 MLP.
# HOG on 64 x 64: 7 x 7 block positions x (2 x 2 cells x 9 orientations) = 1,764 values.
# Colour pyramid: 1 + 4 + 16 = 21 regions x 3 channels x 8 bins = 504 values per colour space.
DESCRIPTOR_FEATURES = 1764 + 21 * 3 * 8


def feature_dimension(representation: str) -> int:
    """Length of the feature vector for each representation used in the experiments."""
    dimensions = {"pixels": PIXEL_FEATURES, "hog_hsv": DESCRIPTOR_FEATURES,
                  "hybrid": PIXEL_FEATURES + DESCRIPTOR_FEATURES,
                  "hog_only": 1764, "hsv_only": 504, "hog_hsv128": 8100 + 504,  # 128 px: 15 x 15 blocks x 36.
                  "hog_hsv_lab": DESCRIPTOR_FEATURES + 504,                     # Deployed: 2,772 values.
                  "crop_hog_hsv": DESCRIPTOR_FEATURES, "crop_hog_hsv128": 8100 + 504}
    return dimensions[representation]


def descriptor_image(image: Image.Image, representation: str) -> Image.Image:
    """Prepare the exact descriptor image, including optional fixed foreground crop."""
    image = image.convert("RGB")
    if representation.startswith("crop_"):
        import cv2
        rgb = np.asarray(image)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        saturation = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1]
        # TrashNet backgrounds are bright and unsaturated: anything darker or more colourful is object.
        mask = ((gray < 235) | (saturation > 30)).astype(np.uint8)
        # Opening removes isolated specks; closing fills small holes inside the object.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count > 1:
            # Keep the largest foreground component (label 0 is the background).
            component = stats[1 + stats[1:, cv2.CC_STAT_AREA].argmax()]
            x, y, width, height, area = map(int, component)
            if area >= 0.01 * image.width * image.height:  # Ignore masks smaller than 1% of the image.
                margin = round(0.10 * max(width, height))
                image = image.crop((max(0, x - margin), max(0, y - margin),
                                    min(image.width, x + width + margin), min(image.height, y + height + margin)))
                # Pad to a square with white, the TrashNet background, so resizing does not distort shape.
                side = max(image.size)
                image = ImageOps.pad(image, (side, side), color="white", method=Image.Resampling.BILINEAR)
    size = 128 if representation.endswith("128") else 64
    return image.resize((size, size), Image.Resampling.BILINEAR)


def image_descriptor(image: Image.Image, representation: str) -> np.ndarray:
    """HOG 8x8 cells plus 8-bin/channel HSV histograms on a 1/2/4 spatial pyramid."""
    import cv2

    feature_dimension(representation)  # Validate before any expensive preprocessing.
    rgb = np.asarray(descriptor_image(image, representation))
    size = rgb.shape[0]
    if representation == "pixels":
        return rgb.transpose(2, 0, 1).reshape(-1).astype(np.float32) / 255  # CHW order, scaled to [0, 1].
    parts = []
    if representation != "hsv_only":
        # HOG: 16 x 16 blocks sliding by 8 px, 8 x 8 cells, 9 unsigned orientation bins, on grayscale.
        # Block normalization makes it insensitive to global brightness and contrast changes.
        hog = cv2.HOGDescriptor((size, size), (16, 16), (8, 8), (8, 8), 9)
        parts.append(hog.compute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)).reshape(-1))
    # OpenCV 8-bit ranges: hue is 0-179 (degrees / 2); the other channels are 0-255.
    spaces = [] if representation == "hog_only" else [(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), (180, 256, 256))]
    if representation == "hog_hsv_lab":
        spaces.append((cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB), (256, 256, 256)))
    for color, ranges in spaces:
        # Spatial pyramid: the whole image, a 2 x 2 grid and a 4 x 4 grid, so the histograms
        # also record roughly where each colour appears.
        for cells in (1, 2, 4):
            cell_size = size // cells
            for row in range(cells):
                for column in range(cells):
                    region = color[row * cell_size:(row + 1) * cell_size,
                                   column * cell_size:(column + 1) * cell_size]
                    for channel, maximum in enumerate(ranges):
                        histogram = cv2.calcHist([region], [channel], None, [8], [0, maximum]).reshape(-1)
                        # Divide by the pixel count so regions of every size give fractions in [0, 1].
                        parts.append(histogram / (cell_size * cell_size))
    descriptor = np.concatenate(parts).astype(np.float32)
    if representation == "hybrid":
        pixels = rgb.transpose(2, 0, 1).reshape(-1).astype(np.float32) / 255
        descriptor = np.concatenate((pixels, descriptor))
    assert len(descriptor) == feature_dimension(representation)
    return descriptor
