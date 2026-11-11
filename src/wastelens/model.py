"""Six-class TrashNet classifiers: the Lab02 MLP baseline and ImageNet-pretrained CNNs."""

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models
from torchvision.transforms.functional import to_tensor


ImageSize = int | tuple[int, int]

# torchvision builder and weight enum for each supported pretrained backbone.
BACKBONES = {
    "mobilenet_v3_large": "MobileNet_V3_Large_Weights",
    "efficientnet_b0": "EfficientNet_B0_Weights",
    "resnet50": "ResNet50_Weights",
}
ARCHITECTURES = ("lab02-mlp", *BACKBONES)
# Channel statistics of the ImageNet training set, which the pretrained weights expect.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


class MLPVariant(nn.Module):
    """Lab02-style dense network, trained from scratch with explicit ablation settings."""

    def __init__(
        self, image_size: int = 64, num_classes: int = 6,
        hidden_sizes: tuple[int, ...] = (128, 32), dropout: float = 0.0,
        layer_norm: bool = False,
    ):
        super().__init__()
        if not hidden_sizes or any(size < 1 for size in hidden_sizes) or not 0 <= dropout < 1:
            raise ValueError("Expected positive hidden sizes and dropout in [0, 1)")
        self.image_size = image_size
        layers = [nn.Flatten()]
        previous = 3 * image_size * image_size
        for size in hidden_sizes:
            layers.append(nn.Linear(previous, size))
            if layer_norm:
                layers.append(nn.LayerNorm(size))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
            previous = size
        layers.append(nn.Linear(previous, num_classes))
        self.layers = nn.Sequential(*layers)
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)


class PretrainedClassifier(nn.Module):
    """An ImageNet CNN whose 1000-class output layer is replaced by a six-class layer.

    Inputs are RGB tensors in [0, 1], the same contract as the MLP. ImageNet normalization
    happens inside the model so training, the app and exported models share one preprocess().
    """

    def __init__(self, backbone: str, image_size: int = 224, num_classes: int = 6, pretrained: bool = False):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"Unknown backbone {backbone!r}; choose one of {sorted(BACKBONES)}")
        self.image_size = image_size
        weights = getattr(models, BACKBONES[backbone]).DEFAULT if pretrained else None
        self.network = getattr(models, backbone)(weights=weights)
        # ResNet names its output layer `fc`; MobileNetV3 and EfficientNet end `classifier` with it.
        old_head = self.head()
        new_head = nn.Linear(old_head.in_features, num_classes)
        if hasattr(self.network, "fc"):
            self.network.fc = new_head
        else:
            self.network.classifier[-1] = new_head
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def head(self) -> nn.Linear:
        """Return the final linear layer, which is trained from scratch."""
        return self.network.fc if hasattr(self.network, "fc") else self.network.classifier[-1]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Raw logits, as for the MLP; softmax is applied only at inference.
        return self.network((images - self.mean) / self.std)


def build_classifier(
    architecture: str, image_size: ImageSize, num_classes: int = 6, pretrained: bool = False,
) -> nn.Module:
    """Create an untrained MLP, or a backbone with ImageNet weights when `pretrained` is set."""
    if architecture == "lab02-mlp":
        return Lab02MLP(image_size=image_size, num_classes=num_classes)
    if not isinstance(image_size, int):
        raise ValueError("Pretrained backbones take a square integer image size")
    return PretrainedClassifier(architecture, image_size, num_classes, pretrained)


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


def load_classifier(path: Path, device: torch.device | str = "cpu") -> tuple[nn.Module, list[str]]:
    """Restore the model and the manifest label order for inference."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    classes = checkpoint["class_names"]
    image_size = checkpoint["image_size"]
    # Checkpoints written before backbones were added carry no architecture field.
    architecture = checkpoint.get("architecture", "lab02-mlp")
    if architecture == "lab02-mlp-variant":
        model = MLPVariant(image_size=image_size, num_classes=len(classes), **checkpoint["mlp_config"])
    else:
        model = build_classifier(architecture, tuple(image_size) if isinstance(image_size, list) else image_size, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), classes


@torch.inference_mode()
def predict_probabilities(model: nn.Module, image: Image.Image) -> torch.Tensor:
    """Return a CPU probability vector in the checkpoint's class order."""
    inputs = preprocess(image, model.image_size).unsqueeze(0)
    device = next(model.parameters()).device
    return model(inputs.to(device)).softmax(dim=1).squeeze(0).cpu()
