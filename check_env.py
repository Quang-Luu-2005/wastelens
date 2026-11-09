"""Run a small CPU inference check for the WasteLens environment."""

import sys

import cv2
import numpy as np
import torch
import torchvision
from PIL import __version__ as pillow_version
from torchvision.transforms.functional import to_tensor


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__} | torchvision: {torchvision.__version__}")
    print(f"OpenCV: {cv2.__version__} | Pillow: {pillow_version}")
    print(f"CUDA available to PyTorch: {torch.cuda.is_available()}")

    # Exercise the OpenCV-to-PyTorch boundary with a known BGR pixel.
    bgr = np.zeros((16, 16, 3), dtype=np.uint8)
    bgr[:, :, 2] = 255
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = to_tensor(rgb).unsqueeze(0)
    if image[0, :, 0, 0].tolist() != [1.0, 0.0, 0.0]:
        raise RuntimeError("Unexpected channel order after converting BGR to RGB")

    probe = torch.nn.Conv2d(3, 6, kernel_size=1).cpu().eval()
    with torch.inference_mode():
        scores = probe(image).mean(dim=(2, 3))
    if scores.shape != (1, 6) or not torch.isfinite(scores).all().item():
        raise RuntimeError("CPU smoke check did not produce six finite scores")
    print("CPU smoke check passed: OpenCV BGR → RGB → six PyTorch scores.")


if __name__ == "__main__":
    main()
