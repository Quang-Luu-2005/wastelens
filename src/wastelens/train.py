"""Train and evaluate a six-class classifier (Lab02 MLP or pretrained CNN) on the fixed TrashNet split."""

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from .data import TrashNetDataset, read_splits
from .model import ARCHITECTURES, ImageSize, PretrainedClassifier, build_classifier, load_classifier


# Defaults per model family: the MLP keeps its Lab02 settings, the CNNs use ImageNet-sized inputs
# and a smaller learning rate so fine-tuning does not destroy the pretrained features.
DEFAULT_IMAGE_SIZE = {"lab02-mlp": 64}
DEFAULT_LEARNING_RATE = {"lab02-mlp": 0.001}
BACKBONE_IMAGE_SIZE = 224
BACKBONE_LEARNING_RATE = 0.0001
HEAD_LR_MULTIPLIER = 10  # The new six-class layer starts from random weights and can learn faster.
WEIGHT_DECAY = 0.01


def make_optimizer(model, learning_rate: float, epochs: int):
    """Return the optimizer and an optional per-epoch learning-rate scheduler."""
    if not isinstance(model, PretrainedClassifier):
        # The Lab02 MLP is trained exactly as in the earlier experiments: Adam, constant rate.
        return torch.optim.Adam(model.parameters(), lr=learning_rate), None
    head = list(model.head().parameters())
    head_ids = {id(parameter) for parameter in head}
    backbone = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone, "lr": learning_rate},
            {"params": head, "lr": learning_rate * HEAD_LR_MULTIPLIER},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    # Cosine decay to zero over the run keeps late epochs stable for checkpoint selection.
    return optimizer, torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. On Kaggle, enable a GPU accelerator (T4 or L4).")
        try:
            torch.ones(2, device="cuda").sum().item()
        except RuntimeError as error:
            raise RuntimeError("PyTorch could not execute a CUDA kernel. Check the GPU and PyTorch CUDA build.") from error
    return torch.device(requested)


def image_batch_to_device(images: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    if images.dtype == torch.uint8:
        return images.float().div_(255)
    return images


def train_epoch(model, loader, optimizer, loss_fn, scaler, device) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = image_batch_to_device(images, device)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
            loss = loss_fn(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def classification_metrics(matrix: torch.Tensor, classes: list[str]) -> dict:
    correct = matrix.diag()
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    precision = correct / predicted.clamp_min(1)
    recall = correct / support.clamp_min(1)
    f1 = 2 * correct / (support + predicted).clamp_min(1)
    return {
        "accuracy": (correct.sum() / matrix.sum()).item(),
        "macro_f1": f1.mean().item(),
        "confusion_matrix": matrix.tolist(),
        "per_class": {
            label: {
                "precision": precision[index].item(),
                "recall": recall[index].item(),
                "f1": f1[index].item(),
                "support": support[index].item(),
            }
            for index, label in enumerate(classes)
        },
    }


@torch.inference_mode()
def evaluate(model, loader, loss_fn, device, classes) -> tuple[dict, torch.Tensor, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    true_labels, predictions, confidences = [], [], []
    for images, labels in loader:
        scores = model(image_batch_to_device(images, device))
        loss = loss_fn(scores, labels.to(device, non_blocking=True))
        total_loss += loss.item() * len(labels)
        confidence, predicted = scores.softmax(dim=1).max(dim=1)
        true_labels.append(labels)
        predictions.append(predicted.cpu())
        confidences.append(confidence.cpu())

    targets = torch.cat(true_labels)
    predicted = torch.cat(predictions)
    num_classes = len(classes)
    # Rows are ground truth; columns are model predictions.
    matrix = torch.bincount(num_classes * targets + predicted, minlength=num_classes**2)
    metrics = classification_metrics(matrix.reshape(num_classes, num_classes), classes)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics, predicted, torch.cat(confidences)


def write_csv(path: Path, header, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def train(
    image_root: Path,
    manifest: Path,
    output: Path,
    *,
    architecture: str = "lab02-mlp",
    device: str = "auto",
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float | None = None,
    image_size: int | None = None,
    seed: int = 42,
    native_resolution: bool = False,
    affine_augmentation: bool = False,
) -> dict:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {architecture!r}; choose one of {ARCHITECTURES}")
    pretrained = architecture != "lab02-mlp"
    if native_resolution and (pretrained or image_size is not None):
        raise ValueError("Native resolution applies only to the MLP and replaces --image-size")
    if learning_rate is None:
        learning_rate = DEFAULT_LEARNING_RATE.get(architecture, BACKBONE_LEARNING_RATE)
    if image_size is None:
        image_size = DEFAULT_IMAGE_SIZE.get(architecture, BACKBONE_IMAGE_SIZE)
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or image_size < 1:
        raise ValueError("Epochs, batch size, learning rate and image size must be positive")
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")

    device = select_device(device)
    torch.manual_seed(seed)
    amp = device.type == "cuda"
    device_name = torch.cuda.get_device_name(device) if amp else "CPU"
    print(f"PyTorch {torch.__version__} | {device_name} | CUDA AMP: {amp}", flush=True)

    splits, classes = read_splits(manifest)
    input_size: ImageSize = image_size
    if native_resolution:
        with Image.open(image_root / splits["train"][0][0]) as sample:
            input_size = sample.size

    def make_loader(split: str) -> DataLoader:
        training = split == "train"
        dataset = TrashNetDataset(
            image_root, splits[split], input_size, augment=training,
            affine_augmentation=training and affine_augmentation,
            # CNN inputs are cached as bytes; the 64 x 64 MLP keeps its original float cache.
            store_uint8=pretrained,
        )
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=training, pin_memory=amp,
            generator=torch.Generator().manual_seed(seed),
        )

    train_loader, val_loader = make_loader("train"), make_loader("val")
    # Backbones download ImageNet weights here; reloading a checkpoint later needs no download.
    model = build_classifier(architecture, input_size, len(classes), pretrained=pretrained).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"{architecture}: {parameters:,} parameters | input {input_size} | learning rate {learning_rate}", flush=True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer, scheduler = make_optimizer(model, learning_rate, epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history = []
    best_f1, best_epoch = -1.0, 0
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, scaler, device)
        if scheduler is not None:
            scheduler.step()
        validation, _, _ = evaluate(model, val_loader, loss_fn, device, classes)
        val_f1 = validation["macro_f1"]
        history.append((epoch, train_loss, validation["loss"], validation["accuracy"], val_f1))
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(f"Epoch {epoch:02d}/{epochs} | train loss {train_loss:.4f} | val macro-F1 {val_f1:.4f}", flush=True)

    metadata = {
        "architecture": architecture,
        "class_names": classes,
        "image_size": input_size,
        "preprocessing": {
            "mode": "RGB", "resize": None if native_resolution else "bilinear", "pixel_scale": "0-1",
            # Backbones subtract the ImageNet mean and divide by its std inside forward().
            "normalization": "imagenet, inside model" if pretrained else None,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".wastelens-", dir=output.parent) as temporary:
        bundle = Path(temporary) / "run"
        bundle.mkdir()
        torch.save({**metadata, "state_dict": best_state}, bundle / "best.pt")
        best_model, _ = load_classifier(bundle / "best.pt", device)
        # Open the test split only after selecting and reloading the validation checkpoint.
        test, predicted, confidence = evaluate(best_model, make_loader("test"), loss_fn, device, classes)
        metrics = {
            "device": str(device), "device_name": device_name, "amp": amp,
            "torch_version": str(torch.__version__), "architecture": architecture,
            "parameters": parameters, "epochs": epochs, "seed": seed,
            "batch_size": batch_size, "learning_rate": learning_rate,
            "image_size": input_size, "affine_augmentation": affine_augmentation,
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "best_epoch": best_epoch, "best_val_macro_f1": best_f1, "test": test,
        }
        write_json(bundle / "meta.json", metadata)
        write_json(bundle / "metrics.json", metrics)
        write_csv(bundle / "history.csv", ("epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1"), history)
        write_csv(
            bundle / "confusion-matrix.csv", ("actual/predicted", *classes),
            ([label, *counts] for label, counts in zip(classes, test["confusion_matrix"], strict=True)),
        )
        write_csv(
            bundle / "test-predictions.csv", ("path", "actual", "predicted", "confidence"),
            (
                (path, classes[actual], classes[guess], score)
                for (path, actual), guess, score in zip(splits["test"], predicted.tolist(), confidence.tolist(), strict=True)
            ),
        )
        bundle.rename(output)

    print(f"Test accuracy {test['accuracy']:.4f} | macro-F1 {test['macro_f1']:.4f} | artifacts: {output}", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=Path("data/trashnet"))
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/trashnet.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/lab02-mlp"))
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="lab02-mlp")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, help="default 0.001 for the MLP, 0.0001 for backbones")
    parser.add_argument("--image-size", type=int, help="square input side; default 64 for the MLP, 224 for backbones")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--native-resolution", action="store_true", help="require the same original size for all images")
    parser.add_argument("--affine-augmentation", action="store_true", help="rotate and translate training images slightly")
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
