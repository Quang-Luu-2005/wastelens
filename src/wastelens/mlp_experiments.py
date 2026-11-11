"""Controlled, from-scratch Lab02 MLP ablations; select on validation before test.

Run one seed per GPU using the accompanying Kaggle notebook. Every configuration
uses RGB 64x64, the fixed split, Xavier initialization, batch size 64, and random
horizontal flips only. All candidates receive the same maximum epoch budget.
"""

import argparse
import csv
import hashlib
import json
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .data import TrashNetDataset, read_splits
from .model import MLPVariant, load_classifier
from .train import classification_metrics, write_csv, write_json


@dataclass(frozen=True)
class Experiment:
    name: str
    parent: str | None
    hidden_sizes: tuple[int, ...] = (128, 32)
    dropout: float = 0.0
    layer_norm: bool = False
    optimizer: str = "adam"
    weight_decay: float = 0.0
    cosine: bool = False
    loss: str = "ce"

    def model_config(self) -> dict:
        return {key: asdict(self)[key] for key in ("hidden_sizes", "dropout", "layer_norm")}


BASE = Experiment("baseline", None)
DEEP = replace(BASE, name="deeper", parent="baseline", hidden_sizes=(128, 64, 32))
WIDE = replace(DEEP, name="wider", parent="deeper", hidden_sizes=(256, 64, 32))
DROP = replace(WIDE, name="dropout", parent="wider", dropout=0.25)
NORM = replace(DROP, name="layernorm", parent="dropout", layer_norm=True)
REG = replace(NORM, name="adamw_cosine", parent="layernorm", optimizer="adamw", weight_decay=1e-4, cosine=True)
EXPERIMENTS = (
    BASE, DEEP, WIDE, DROP, NORM, REG,
    replace(REG, name="label_smoothing", parent="adamw_cosine", loss="smooth_ce"),
    replace(REG, name="weighted_ce", parent="adamw_cosine", loss="weighted_ce"),
    replace(REG, name="focal", parent="adamw_cosine", loss="focal"),
)


class FocalLoss(nn.Module):
    """Multiclass focal loss with gamma=2 and no additional class weights."""

    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels, reduction="none")
        return ((1 - torch.exp(-ce)).square() * ce).mean()


def make_loss(name: str, train_labels: torch.Tensor, num_classes: int) -> nn.Module:
    if name == "focal":
        return FocalLoss()
    if name == "weighted_ce":
        # Fit weights on training labels only; square roots limit rare-class amplification.
        counts = torch.bincount(train_labels, minlength=num_classes).float()
        if torch.any(counts == 0):
            raise ValueError("Every class must be present in the training split")
        weights = counts.rsqrt()
        return nn.CrossEntropyLoss(weight=weights / weights.mean())
    if name == "smooth_ce":
        return nn.CrossEntropyLoss(label_smoothing=0.05)
    if name == "ce":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unknown loss: {name}")


@torch.inference_mode()
def measure(model, images, labels, classes, batch_size=256):
    model.eval()
    logits = torch.cat([model(batch) for batch in images.split(batch_size)])
    predicted = logits.argmax(dim=1)
    matrix = torch.bincount(len(classes) * labels + predicted, minlength=len(classes) ** 2)
    metrics = classification_metrics(matrix.reshape(len(classes), len(classes)).float(), classes)
    # Use plain CE for evaluation so loss values remain comparable across training objectives.
    metrics["loss"] = F.cross_entropy(logits, labels).item()
    return metrics, predicted, logits.softmax(dim=1).amax(dim=1)


def cached_split(image_root, samples, device):
    dataset = TrashNetDataset(image_root, samples, 64, augment=False)
    # 1764 training images at 64x64 occupy only about 83 MiB of GPU memory.
    return dataset.images.to(device), dataset.labels.to(device)


def run_seed(image_root: Path, manifest: Path, output: Path, seed: int, epochs: int, device: str,
             batch_size: int = 64, names: list[str] | None = None):
    if epochs < 1 or batch_size < 1:
        raise ValueError("Epochs and batch size must be positive")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(device)
    splits, classes = read_splits(manifest)
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    train_x, train_y = cached_split(image_root, splits["train"], device)
    val_x, val_y = cached_split(image_root, splits["val"], device)
    # The test images are deliberately not loaded by the candidate-training worker.
    experiments = [experiment for experiment in EXPERIMENTS if names is None or experiment.name in names]
    if not experiments or names and set(names) != {experiment.name for experiment in experiments}:
        raise ValueError("Unknown or empty experiment selection")
    for experiment in experiments:
        directory = output / experiment.name
        directory.mkdir()
        torch.manual_seed(seed)
        model = MLPVariant(num_classes=len(classes), **experiment.model_config()).to(device)
        loss_fn = make_loss(experiment.loss, train_y, len(classes)).to(device)
        optimizer_type = torch.optim.AdamW if experiment.optimizer == "adamw" else torch.optim.Adam
        optimizer = optimizer_type(model.parameters(), lr=0.001, weight_decay=experiment.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if experiment.cosine else None
        # Dedicated generators make sample ordering and flip masks identical across configurations.
        order_rng = torch.Generator(device=device).manual_seed(seed + 1000)
        flip_rng = torch.Generator(device=device).manual_seed(seed + 2000)
        history, best_key, best_epoch = [], (-1.0, -1.0), 0
        best_state = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            model.train()
            order = torch.randperm(len(train_y), generator=order_rng, device=device)
            total_loss = torch.zeros((), device=device)
            for indices in order.split(batch_size):
                images, labels = train_x[indices], train_y[indices]
                flip = torch.rand((len(indices), 1, 1, 1), generator=flip_rng, device=device) < 0.5
                images = torch.where(flip, images.flip(-1), images)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(images), labels)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss in {experiment.name}, epoch {epoch}")
                loss.backward()
                optimizer.step()
                total_loss += loss.detach() * len(indices)
            validation, _, _ = measure(model, val_x, val_y, classes)
            key = (validation["macro_f1"], validation["accuracy"])
            if key > best_key:
                best_key, best_epoch = key, epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            history.append((epoch, total_loss.item() / len(train_y), validation["loss"],
                            validation["accuracy"], validation["macro_f1"], optimizer.param_groups[0]["lr"]))
            if scheduler:
                scheduler.step()
            if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
                print(f"{experiment.name} seed={seed} epoch={epoch}/{epochs} val_acc={validation['accuracy']:.4f} "
                      f"val_f1={validation['macro_f1']:.4f} best_f1={best_key[0]:.4f}", flush=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        checkpoint = {
            "architecture": "lab02-mlp-variant", "image_size": 64, "class_names": classes,
            "mlp_config": experiment.model_config(), "state_dict": best_state,
        }
        torch.save(checkpoint, directory / "best.pt")
        restored, restored_classes = load_classifier(directory / "best.pt", device)
        assert restored_classes == classes
        train_metrics, _, _ = measure(restored, train_x, train_y, classes)
        validation, _, _ = measure(restored, val_x, val_y, classes)
        assert (validation["macro_f1"], validation["accuracy"]) == best_key
        result = {
            "experiment": asdict(experiment), "seed": seed, "epochs": epochs, "batch_size": batch_size,
            "learning_rate": 0.001, "image_size": 64, "augmentation": "horizontal_flip_p0.5",
            "precision": "float32", "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "training_seconds": elapsed, "best_epoch": best_epoch,
            "manifest_sha256": manifest_hash, "train": train_metrics, "validation": validation,
            "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "torch_version": torch.__version__,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        }
        write_json(directory / "metrics.json", result)
        write_csv(directory / "history.csv", ("epoch", "train_objective", "val_ce", "val_accuracy", "val_macro_f1", "lr"), history)
        print(f"DONE {experiment.name} seed={seed} best_epoch={best_epoch} seconds={elapsed:.1f}", flush=True)


def summarize(image_root: Path, manifest: Path, output: Path, seeds: list[int], device: str):
    """Lock the winning configuration using mean validation F1, then evaluate test."""
    rows, results = [], {}
    expected_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    for experiment in EXPERIMENTS:
        trials = []
        for seed in seeds:
            path = output / f"seed-{seed}" / experiment.name / "metrics.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            assert result["manifest_sha256"] == expected_hash and result["seed"] == seed
            assert result["experiment"] == json.loads(json.dumps(asdict(experiment)))
            trials.append(result)
        results[experiment.name] = trials
        rows.append({
            "name": experiment.name, "parent": experiment.parent, "parameters": trials[0]["parameters"],
            "val_accuracy_mean": statistics.mean(trial["validation"]["accuracy"] for trial in trials),
            "val_accuracy_std": statistics.stdev(trial["validation"]["accuracy"] for trial in trials) if len(trials) > 1 else 0,
            "val_macro_f1_mean": statistics.mean(trial["validation"]["macro_f1"] for trial in trials),
            "val_macro_f1_std": statistics.stdev(trial["validation"]["macro_f1"] for trial in trials) if len(trials) > 1 else 0,
            "train_accuracy_mean": statistics.mean(trial["train"]["accuracy"] for trial in trials),
            "training_seconds_mean": statistics.mean(trial["training_seconds"] for trial in trials),
        })
    winner = max(rows, key=lambda row: (row["val_macro_f1_mean"], row["val_accuracy_mean"]))["name"]
    summary = {
        "selection": "configuration by mean validation macro-F1 over paired seeds; accuracy breaks ties",
        "winner": winner, "seeds": seeds, "manifest_sha256": expected_hash, "validation": rows,
        "limitations": "The existing test split was inspected in earlier project experiments; this is an exploratory comparison, not a new untouched holdout.",
    }
    # Persist the decision BEFORE loading or evaluating the test split.
    write_json(output / "selection.json", summary)
    write_csv(output / "validation-comparison.csv", list(rows[0]), [row.values() for row in rows])
    splits, classes = read_splits(manifest)
    test_x, test_y = cached_split(image_root, splits["test"], device)
    test_results = {}
    for name in dict.fromkeys(("baseline", winner)):
        trials = []
        for seed in seeds:
            directory = output / f"seed-{seed}" / name
            model, restored_classes = load_classifier(directory / "best.pt", device)
            assert restored_classes == classes
            metrics, predictions, confidence = measure(model, test_x, test_y, classes)
            write_json(directory / "test-metrics.json", metrics)
            write_csv(directory / "test-predictions.csv", ("path", "actual", "predicted", "confidence"),
                      [(path, classes[label], classes[predicted], score) for (path, label), predicted, score
                       in zip(splits["test"], predictions.cpu().tolist(), confidence.cpu().tolist(), strict=True)])
            trials.append({"seed": seed, **metrics})
        test_results[name] = {
            "accuracy_mean": statistics.mean(trial["accuracy"] for trial in trials),
            "macro_f1_mean": statistics.mean(trial["macro_f1"] for trial in trials), "trials": trials,
        }
    summary["test"] = test_results
    summary["accuracy_delta_percentage_points"] = 100 * (test_results[winner]["accuracy_mean"] - test_results["baseline"]["accuracy_mean"])
    write_json(output / "run-summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "summarize"))
    parser.add_argument("--image-root", type=Path, default=Path("data/trashnet"))
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/trashnet.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--experiments", nargs="+")
    args = parser.parse_args()
    if args.mode == "train":
        run_seed(args.image_root, args.manifest, args.output, args.seed, args.epochs, args.device,
                 args.batch_size, args.experiments)
    else:
        summarize(args.image_root, args.manifest, args.output, args.seeds, args.device)


if __name__ == "__main__":
    main()
