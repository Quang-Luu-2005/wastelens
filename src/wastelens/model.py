"""The Lab02 MLP adapted to six TrashNet classes."""

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision.transforms.functional import to_tensor


class Lab02MLP(nn.Module):
    def __init__(self, image_size: int = 64, num_classes: int = 6):
        super().__init__()
        self.image_size = image_size
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * image_size**2, 128),
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


def preprocess(image: Image.Image, image_size: int) -> torch.Tensor:
    """Convert a PIL image to RGB, resize as in Lab02 and scale to [0, 1]."""
    resized = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    return to_tensor(resized)


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
