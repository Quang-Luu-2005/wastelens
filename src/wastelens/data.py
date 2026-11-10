"""Load the versioned TrashNet splits without making a new partition."""

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .model import preprocess


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
    def __init__(self, image_root: Path, samples: list[tuple[str, int]], image_size: int, augment: bool = False):
        self.labels = torch.tensor([label for _, label in samples], dtype=torch.long)
        self.augment = augment
        # Cache the small resized inputs, as in Lab02, to avoid decoding each epoch.
        images = []
        for relative, _ in samples:
            with Image.open(image_root / relative) as image:
                images.append(preprocess(image, image_size))
        self.images = torch.stack(images)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        if self.augment and torch.rand(()).item() < 0.5:
            image = image.flip(-1)
        return image, self.labels[index]
