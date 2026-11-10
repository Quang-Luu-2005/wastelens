"""The Lab02 MLP adapted to six TrashNet classes."""

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision.transforms.functional import to_tensor


ImageSize = int | tuple[int, int]


class Lab02MLP(nn.Module):
    def __init__(self, image_size: ImageSize = 64, num_classes: int = 6):
        super().__init__()
        self.image_size = image_size
        width, height = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * width * height, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )
        # ReLU and Xavier normal match the selected Lab02 experiment.
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # CrossEntropyLoss consumes raw logits and applies log-softmax internally.
        return self.layers(images)


def prepare_image(image: Image.Image, image_size: ImageSize) -> Image.Image:
    """An integer requests a square resize. A (width, height) pair requires the original size."""
    rgb = image.convert("RGB")
    if isinstance(image_size, int):
        return rgb.resize((image_size, image_size), Image.Resampling.BILINEAR)
    if rgb.size != image_size:
        raise ValueError(f"Native-resolution MLP expects {image_size} (width, height), got {rgb.size}")
    return rgb


def preprocess(image: Image.Image, image_size: ImageSize) -> torch.Tensor:
    """Prepare the classifier input and scale RGB values to [0, 1]."""
    return to_tensor(prepare_image(image, image_size))


def load_classifier(path: Path, device: torch.device | str = "cpu") -> tuple[Lab02MLP, list[str]]:
    """Restore the model and the manifest label order for inference."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    classes = checkpoint["class_names"]
    model = Lab02MLP(image_size=checkpoint["image_size"], num_classes=len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), classes


@torch.inference_mode()
def predict_probabilities(model: Lab02MLP, image: Image.Image) -> torch.Tensor:
    """Return a CPU probability vector in the checkpoint's class order."""
    inputs = preprocess(image, model.image_size).unsqueeze(0)
    device = next(model.parameters()).device
    return model(inputs.to(device)).softmax(dim=1).squeeze(0).cpu()
