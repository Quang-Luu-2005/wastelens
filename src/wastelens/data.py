"""Load the versioned TrashNet splits without making a new partition."""

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, RandomAffine
from torchvision.transforms.functional import pil_to_tensor

from .model import ImageSize, prepare_image, preprocess


def read_splits(manifest: Path) -> tuple[dict[str, list[tuple[str, int]]], list[str]]:
    splits = {"train": [], "val": [], "test": []}
    class_names = {}
    paths = set()
    with manifest.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            path, label, split = row["path"], row["label"], row["split"]
            class_id = int(row["class_id"])
            if path in paths or split not in splits:
                raise ValueError(f"Invalid or repeated manifest row: {row}")
            if class_names.setdefault(class_id, label) != label:
                raise ValueError(f"Conflicting labels for class {class_id}")
            paths.add(path)
            splits[split].append((path, class_id))

    if set(class_names) != set(range(6)) or len(set(class_names.values())) != 6 or not all(splits.values()):
        raise ValueError("Expected six distinct class IDs (0–5) and three nonempty splits")
    return splits, [class_names[index] for index in range(6)]


class TrashNetDataset(Dataset):
    def __init__(
        self, image_root: Path, samples: list[tuple[str, int]], image_size: ImageSize,
        augment: bool = False, affine_augmentation: bool = False, store_uint8: bool = False,
    ):
        self.labels = torch.tensor([label for _, label in samples], dtype=torch.long)
        self.augment = augment
        # Native-size images always stay as bytes; larger square inputs may opt in.
        as_bytes = store_uint8 or isinstance(image_size, tuple)
        self.affine = (
            RandomAffine(
                degrees=10, translate=(0.0625, 0.0625),
                interpolation=InterpolationMode.BILINEAR, fill=255 if as_bytes else 1.0,
            )
            if affine_augmentation else None
        )
        # Bytes take a quarter of the float32 memory: about 1 GiB for native images, 0.4 GiB at 224 x 224.
        # image_batch_to_device() rescales them to [0, 1] on the device, matching preprocess().
        images = []
        for relative, _ in samples:
            with Image.open(image_root / relative) as image:
                pixels = pil_to_tensor(prepare_image(image, image_size)) if as_bytes else preprocess(image, image_size)
                images.append(pixels)
        self.images = torch.stack(images)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        if self.augment and torch.rand(()).item() < 0.5:
            image = image.flip(-1)
        if self.affine is not None:
            image = self.affine(image)
        return image, self.labels[index]
