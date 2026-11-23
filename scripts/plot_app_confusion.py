"""Draw the test confusion matrices of the three app models and their per-class metrics.

Reads the JSON written by scripts/evaluate_app_models.py (377 test images, Top-1 on every image,
including those the ensemble marks uncertain) and writes a PNG plus a Markdown table.

    .venv\\Scripts\\python.exe scripts/plot_app_confusion.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"baseline": "Baseline Lab 2", "improved": "MLP cải tiến (ensemble 10 MLP)", "mobilenet": "MobileNetV3 Large"}
VI = {"cardboard": "Bìa carton", "glass": "Thủy tinh", "metal": "Kim loại", "paper": "Giấy",
      "plastic": "Nhựa", "trash": "Rác khác"}


def per_class(matrix: np.ndarray):
    """Precision, recall and F1 per class from a confusion matrix (rows = true, columns = predicted)."""
    true_positive = np.diag(matrix).astype(float)
    precision = np.divide(true_positive, matrix.sum(0), out=np.zeros_like(true_positive), where=matrix.sum(0) > 0)
    recall = true_positive / matrix.sum(1)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(true_positive),
                   where=(precision + recall) > 0)
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/app-model-comparison.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/figures/app-models-test")
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    classes = report["classes"]
    args.output.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(17, 5.6), constrained_layout=True)
    lines = ["| Mô hình | Lớp | Precision | Recall | F1 | Số ảnh test |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for axis, (key, model) in zip(axes, report["models"].items()):
        matrix = np.array(model["confusion_matrix"])
        # Row-normalised colours (recall per true class); the cell text shows the raw counts.
        share = matrix / matrix.sum(1, keepdims=True)
        axis.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for row in range(len(classes)):
            for column in range(len(classes)):
                axis.text(column, row, matrix[row, column], ha="center", va="center", fontsize=10,
                          color="white" if share[row, column] > 0.55 else "black")
        labels = [VI[name] for name in classes]
        axis.set_xticks(range(len(classes)), labels, rotation=35, ha="right")
        axis.set_yticks(range(len(classes)), labels)
        axis.set_xlabel("Lớp dự đoán")
        axis.set_ylabel("Lớp thật")
        axis.set_title(f"{NAMES[key]}\naccuracy {model['accuracy']:.2%} · macro-F1 {model['macro_f1']:.3f}", fontsize=11)
        precision, recall, f1 = per_class(matrix)
        for index, name in enumerate(classes):
            lines.append(f"| {NAMES[key]} | {VI[name]} | {precision[index]:.3f} | {recall[index]:.3f} | "
                         f"{f1[index]:.3f} | {int(matrix[index].sum())} |")
    figure.suptitle(f"Ma trận nhầm lẫn trên tập test ({report['test_images']} ảnh), màu = tỉ lệ theo hàng", fontsize=12)
    figure.savefig(args.output / "confusion-matrices.png", dpi=160)
    (args.output / "per-class-metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output / "confusion-matrices.png")
    print(args.output / "per-class-metrics.md")


if __name__ == "__main__":
    main()
