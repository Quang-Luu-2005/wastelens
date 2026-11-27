"""Figures for the Lab03 report, one set per section.

The main sections analyse the from-scratch baseline alone; the improvement section compares
baseline and improved MLP; the pretrained section compares improved MLP and MobileNetV3.
Inputs are existing results only (no training or inference):

- outputs/app-model-comparison.json from scripts/evaluate_app_models.py
- reports/filter-robustness-20261119/results.json from scripts/filter_experiment.py
- optionally a training history.csv for the baseline learning curve
- optionally outputs/external-evaluation.json from scripts/evaluate_external.py (RealWaste)

    .venv\\Scripts\\python.exe scripts/report_figures.py --output ../doc/figures
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
VI = {"cardboard": "Bìa carton", "glass": "Thủy tinh", "metal": "Kim loại", "paper": "Giấy",
      "plastic": "Nhựa", "trash": "Rác khác"}
NAMES = {"baseline": "Baseline Lab02", "improved": "MLP cải tiến", "mobilenet": "MobileNetV3"}
COLORS = {"baseline": "#8c9bb0", "improved": "#00695c", "mobilenet": "#e07b00"}


def confusion_figure(matrix: np.ndarray, classes: list[str], title: str, path: Path):
    """One confusion matrix: counts in the cells, colour = share of the true class (recall)."""
    share = matrix / matrix.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 5.0), layout="constrained")
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for row in range(len(classes)):
        for column in range(len(classes)):
            ax.text(column, row, matrix[row, column], ha="center", va="center", fontsize=11,
                    color="white" if share[row, column] > 0.55 else "black")
    labels = [VI[name] for name in classes]
    ax.set_xticks(range(len(classes)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(classes)), labels)
    ax.set(xlabel="Lớp dự đoán", ylabel="Lớp thật", title=title)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def confusion_row(models: dict, classes: list[str], path: Path):
    """The three models' confusion matrices side by side, sharing one colour scale (recall per row)."""
    fig, axes = plt.subplots(1, len(models), figsize=(5.0 * len(models), 5.0), layout="constrained")
    labels = [VI[name] for name in classes]
    for ax, (key, model) in zip(axes, models.items()):
        matrix = np.array(model["confusion_matrix"])
        share = matrix / matrix.sum(1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for row in range(len(classes)):
            for column in range(len(classes)):
                ax.text(column, row, matrix[row, column], ha="center", va="center", fontsize=12,
                        color="white" if share[row, column] > 0.55 else "black")
        ax.set_xticks(range(len(classes)), labels, rotation=35, ha="right", fontsize=11)
        ax.set_yticks(range(len(classes)), labels if ax is axes[0] else [], fontsize=11)
        ax.set_title(f"{NAMES[key]}: {model['accuracy']:.2%}", fontsize=13)
        ax.set_xlabel("Lớp dự đoán", fontsize=11)
    axes[0].set_ylabel("Lớp thật", fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def baseline_errors(model: dict, classes: list[str], path: Path, count: int = 15):
    """Grid of the baseline's most confident mistakes, with true label, prediction and probability."""
    wrong = [r for r in model["predictions"] if r["prediction"] != r["target"]]
    wrong.sort(key=lambda r: -r["probabilities"][r["prediction"]])
    columns = 5
    rows = (min(count, len(wrong)) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(13, 2.95 * rows), layout="constrained")
    for ax, record in zip(axes.flat, wrong[:count]):
        with Image.open(ROOT / "data/trashnet" / record["path"]) as image:
            ax.imshow(image.convert("RGB"))
        probability = record["probabilities"][record["prediction"]]
        ax.set_title(f"Thật: {VI[record['target']]}\nĐoán: {VI[record['prediction']]} ({probability:.0%})", fontsize=10)
        ax.axis("off")
    for ax in list(axes.flat)[len(wrong[:count]):]:
        ax.axis("off")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return len(wrong)


def confidence_figure(models: dict, keys: list[str], path: Path):
    """Histogram of top-1 probability for correct and wrong predictions, one panel per model."""
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 3.6), sharey=True, layout="constrained")
    axes = np.atleast_1d(axes)
    bins = np.linspace(0, 1, 21)
    for ax, key in zip(axes, keys):
        records = models[key]["predictions"]
        top = np.array([r["probabilities"][r["prediction"]] for r in records])
        correct = np.array([r["prediction"] == r["target"] for r in records])
        ax.hist(top[correct], bins=bins, color="#2e7d32", alpha=0.75, label=f"Đúng ({correct.sum()})")
        ax.hist(top[~correct], bins=bins, color="#c62828", alpha=0.75, label=f"Sai ({(~correct).sum()})")
        ax.set(title=NAMES[key], xlabel="Xác suất của lớp được chọn", xlim=(0, 1))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Số ảnh Test")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def severity_figure(rows: list[dict], keys: list[str], path: Path):
    """Accuracy against severity for the 9 corruptions, one line per selected model."""
    corrupt = [r for r in rows if r["family"] == "corruption"]
    groups = list(dict.fromkeys(r["group"] for r in corrupt))

    def accuracy(key, condition):
        return 100 * next(r["accuracy"] for r in rows if r["model"] == key and r["condition"] == condition)

    fig, axes = plt.subplots(3, 3, figsize=(12, 9.2), sharey=True, layout="constrained")
    for ax, group in zip(axes.flat, groups):
        levels = sorted({(r["severity"], r["level"]) for r in corrupt if r["group"] == group})
        title = next(r["label"] for r in corrupt if r["group"] == group).split(" = ")[0]
        for key in keys:
            values = [accuracy(key, "clean")] + [accuracy(key, f"{group}_{severity}") for severity, _ in levels]
            ax.plot(range(len(values)), values, marker="o", color=COLORS[key], label=NAMES[key])
        ax.set_xticks(range(len(levels) + 1), ["0"] + [f"{level:g}" for _, level in levels])
        ax.set(title=title, xlabel="Mức", ylim=(0, 100))
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy (%)")
    axes[0, 0].legend(loc="lower left", fontsize=9)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def learning_curve(history: Path, best_epoch: int | None, path: Path):
    """Train/validation loss and validation macro-F1 of the baseline run."""
    with history.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    epochs = [int(r["epoch"]) for r in rows]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    left.plot(epochs, [float(r["train_loss"]) for r in rows], label="Train loss", color="#1565c0")
    left.plot(epochs, [float(r["val_loss"]) for r in rows], label="Validation loss", color="#e07b00")
    left.set(xlabel="Epoch", ylabel="Cross-entropy", yscale="log", title="Loss")
    right.plot(epochs, [float(r["val_macro_f1"]) for r in rows], color="#00695c", label="Validation macro-F1")
    right.plot(epochs, [float(r["val_accuracy"]) for r in rows], color="#8c9bb0", label="Validation accuracy")
    right.set(xlabel="Epoch", title="Validation", ylim=(0, 1))
    if best_epoch:
        for ax in (left, right):
            ax.axvline(best_epoch, color="black", linestyle="--", linewidth=1, label=f"Epoch tốt nhất ({best_epoch})")
    for ax in (left, right):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def external_samples(external: dict, path: Path, seed: int = 3):
    """One random external image per class with the three models' predictions (correct ones in green)."""
    folder, classes, models = Path(external["folder"]), external["classes"], external["models"]
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, len(classes), figsize=(2.6 * len(classes), 3.7))
    # Manual margins: the prediction lines sit below the axes, where constrained layout does not look.
    fig.subplots_adjust(left=0.005, right=0.995, top=0.9, bottom=0.25, wspace=0.04)
    for ax, name in zip(axes, classes):
        indices = [i for i, r in enumerate(models["baseline"]["predictions"]) if r["target"] == name]
        index = int(rng.choice(indices))
        relative = models["baseline"]["predictions"][index]["path"]
        with Image.open(folder / relative) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(f"Thật: {VI[name]}", fontsize=11, fontweight="bold")
        for line, key in enumerate(NAMES):
            record = models[key]["predictions"][index]
            ax.text(0.0, -0.06 - 0.1 * line, f"{NAMES[key].split()[0]}: {VI[record['prediction']]}", fontsize=10,
                    transform=ax.transAxes, va="top",
                    color="#2e7d32" if record["prediction"] == name else "#c62828")
        ax.axis("off")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, default=ROOT / "outputs/app-model-comparison.json")
    parser.add_argument("--robustness", type=Path, default=ROOT / "reports/filter-robustness-20261119/results.json")
    parser.add_argument("--history", type=Path, help="baseline history.csv for the learning curve")
    parser.add_argument("--best-epoch", type=int)
    parser.add_argument("--external", type=Path, help="outputs/external-evaluation.json from evaluate_external.py")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    classes, models = evaluation["classes"], evaluation["models"]
    for key in models:
        confusion_figure(np.array(models[key]["confusion_matrix"]), classes,
                         f"{NAMES[key]}: accuracy {models[key]['accuracy']:.2%}", args.output / f"cm-{key}.png")
    errors = baseline_errors(models["baseline"], classes, args.output / "baseline-errors.png")
    confidence_figure(models, ["baseline"], args.output / "confidence-baseline.png")
    confidence_figure(models, ["baseline", "improved"], args.output / "confidence-scratch.png")
    confidence_figure(models, list(NAMES), args.output / "confidence-all.png")
    confusion_row(models, classes, args.output / "cm-all.png")

    rows = json.loads(args.robustness.read_text(encoding="utf-8"))["rows"]
    severity_figure(rows, ["baseline"], args.output / "severity-baseline.png")
    severity_figure(rows, ["baseline", "improved"], args.output / "severity-scratch.png")
    severity_figure(rows, ["improved", "mobilenet"], args.output / "severity-pretrained.png")
    severity_figure(rows, list(NAMES), args.output / "severity-all.png")
    if args.history:
        learning_curve(args.history, args.best_epoch, args.output / "baseline-learning-curve.png")
    if args.external:
        external = json.loads(args.external.read_text(encoding="utf-8"))
        for key in external["models"]:
            model = external["models"][key]
            confusion_figure(np.array(model["confusion_matrix"]), external["classes"],
                             f"{NAMES[key]} trên RealWaste: {model['accuracy']:.2%}",
                             args.output / f"external-cm-{key}.png")
        external_samples(external, args.output / "external-samples.png")
    print(f"baseline errors: {errors}")


if __name__ == "__main__":
    main()
