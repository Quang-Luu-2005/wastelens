"""Run the filter x model experiment on the 377 test images and write tables, figures and a report.

All three deployed models are fixed; nothing is trained or tuned here. Each condition is compared
with the same model's predictions on the clean images by an exact McNemar test, Holm-corrected
over all conditions of that model.

  python scripts/filter_experiment.py --output reports/filter-robustness-20261119
"""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wastelens.data import read_splits  # noqa: E402
from wastelens.model import DescriptorEnsemble, load_classifier  # noqa: E402
from wastelens.profiles import MODEL_PROFILES  # noqa: E402
from wastelens.robustness import CORRUPTIONS, RESCUES, RESTORERS, build_conditions, predict_batch  # noqa: E402
from wastelens.stats import accuracy_and_macro_f1, clopper_pearson, confusion, holm, mcnemar_exact  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = {"baseline": "Baseline Lab 2", "improved": "MLP cải tiến", "mobilenet": "MobileNetV3"}
COLORS = {"baseline": "#94a3b8", "improved": "#00685f", "mobilenet": "#d97706"}


def evaluate(output: Path, threads: int, limit: int | None):
    torch.set_num_threads(threads)
    manifest = ROOT / "data/splits/trashnet.csv"
    splits, classes = read_splits(manifest)
    samples = splits["test"][:limit] if limit else splits["test"]
    labels = np.array([label for _, label in samples])
    images = [cv2.imread(str(ROOT / "data/trashnet" / path)) for path, _ in samples]
    if any(image is None for image in images):
        raise FileNotFoundError("Prepare the dataset first: python scripts/prepare_trashnet.py")
    models = {}
    for key, profile in MODEL_PROFILES.items():
        model, model_classes = load_classifier(profile.checkpoint)
        if model_classes != classes:
            raise ValueError(f"Class order mismatch in {key}")
        models[key] = model
    conditions = build_conditions()
    predictions = {key: {} for key in models}
    confidences = {key: {} for key in models}
    started = time.perf_counter()
    for number, condition in enumerate(conditions, start=1):
        # The same corrupted images go to every model; the per-image seed fixes the random noise.
        transformed = [condition.transform(image, index) for index, image in enumerate(images)]
        for key, model in models.items():
            probabilities = predict_batch(model, transformed)
            predictions[key][condition.name] = probabilities.argmax(1)
            confidences[key][condition.name] = probabilities.max(1)
        print(f"[{number}/{len(conditions)}] {condition.name} ({time.perf_counter() - started:.0f} s)", flush=True)

    rows = []
    for key, model in models.items():
        clean_correct = predictions[key]["clean"] == labels
        threshold = model.uncertain_threshold if isinstance(model, DescriptorEnsemble) else None
        model_rows = []
        for condition in conditions:
            predicted = predictions[key][condition.name]
            correct = predicted == labels
            accuracy, macro_f1 = accuracy_and_macro_f1(labels, predicted, len(classes))
            matrix = confusion(labels, predicted, len(classes))
            row = {
                "model": key, "condition": condition.name, "family": condition.family, "group": condition.group,
                "severity": condition.severity, "level": condition.level, "label": condition.label,
                "correct": int(correct.sum()), "images": len(labels), "accuracy": float(accuracy),
                "accuracy_ci95": clopper_pearson(int(correct.sum()), len(labels)), "macro_f1": float(macro_f1),
                "delta_points": 100 * float(accuracy - clean_correct.mean()),
                # Share of images whose predicted class differs from the clean prediction.
                "flip_rate": float((predicted != predictions[key]["clean"]).mean()),
                "per_class_recall": {name: float(matrix[i, i] / max(matrix[i].sum(), 1)) for i, name in enumerate(classes)},
            }
            if threshold is not None:
                accepted = confidences[key][condition.name] >= threshold
                row.update(uncertain_rate=float(1 - accepted.mean()),
                           accepted_accuracy=float(correct[accepted].mean()) if accepted.any() else None)
            if condition.name != "clean":
                test = mcnemar_exact(clean_correct, correct)
                row.update(mcnemar_p=test["p_value"], only_clean_correct=test["only_a_correct"],
                           only_condition_correct=test["only_b_correct"])
            model_rows.append(row)
        tested = [row for row in model_rows if "mcnemar_p" in row]
        for row, adjusted in zip(tested, holm([row["mcnemar_p"] for row in tested]), strict=True):
            row["mcnemar_p_holm"] = adjusted
        rows += model_rows

    # Did the restoration filter help? Pair each rescue with the corruption alone, per model.
    rescues = []
    for key in models:
        for corruption, severity, restorers in RESCUES:
            damaged = predictions[key][f"{corruption}_{severity}"] == labels
            for restorer in restorers:
                repaired = predictions[key][f"rescue_{corruption}_{severity}_{restorer}"] == labels
                test = mcnemar_exact(damaged, repaired)
                rescues.append({"model": key, "corruption": f"{corruption}_{severity}", "restorer": restorer,
                                "corrupted_correct": int(damaged.sum()), "rescued_correct": int(repaired.sum()),
                                "lost": test["only_a_correct"], "gained": test["only_b_correct"],
                                "mcnemar_p": test["p_value"]})
    for row, adjusted in zip(rescues, holm([row["mcnemar_p"] for row in rescues]), strict=True):
        row["mcnemar_p_holm"] = adjusted

    summary = {}
    for key in models:
        clean = next(r for r in rows if r["model"] == key and r["condition"] == "clean")["accuracy"]
        corrupted = [r["accuracy"] for r in rows if r["model"] == key and r["family"] == "corruption"]
        summary[key] = {"clean_accuracy": clean, "mean_corruption_accuracy": float(np.mean(corrupted)),
                        "relative_drop": float((clean - np.mean(corrupted)) / clean),
                        "significant_conditions_holm": sum(1 for r in rows if r["model"] == key
                                                           and r.get("mcnemar_p_holm", 1) < 0.05)}
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "predictions.npz", labels=labels, paths=np.array([p for p, _ in samples]),
                        **{f"{key}__{name}": value for key in models for name, value in predictions[key].items()})
    report = {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "images": len(labels),
              "classes": classes, "checkpoints": {key: MODEL_PROFILES[key].filename for key in models},
              "seconds": time.perf_counter() - started, "summary": summary, "rows": rows,
              "rescue_vs_corrupted": rescues}
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["model", "condition", "family", "group", "severity", "level", "correct", "images", "accuracy",
                  "macro_f1", "delta_points", "flip_rate", "mcnemar_p", "mcnemar_p_holm", "uncertain_rate",
                  "accepted_accuracy", "label"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return report, images, conditions


def figures(report: dict, images: list[np.ndarray], conditions, output: Path):
    rows = report["rows"]
    keys = list(report["checkpoints"])

    def accuracy(key, name):
        return 100 * next(r["accuracy"] for r in rows if r["model"] == key and r["condition"] == name)

    # 1. Accuracy against severity for every corruption, one line per model.
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharey=True, layout="constrained")
    for ax, (group, (label, levels, _)) in zip(axes.flat, CORRUPTIONS.items()):
        for key in keys:
            values = [accuracy(key, "clean")] + [accuracy(key, f"{group}_{s}") for s in range(1, 6)]
            ax.plot(range(6), values, marker="o", color=COLORS[key], label=MODEL_NAMES[key])
        ax.set_xticks(range(6), ["0"] + [f"{level:g}" for level in levels])
        ax.set(title=label, xlabel="Mức", ylim=(0, 100))
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[2, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].legend(loc="lower left", fontsize=9)
    fig.suptitle(f"Accuracy trên {report['images']} ảnh test theo mức suy giảm (mức 0 = ảnh gốc)")
    fig.savefig(output / "severity-curves.png", dpi=150)
    plt.close(fig)

    # 2. Change in accuracy (points) for each corruption x severity, one panel per model.
    groups = list(CORRUPTIONS)
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 5.2), layout="constrained")
    for ax, key in zip(axes, keys):
        grid = np.array([[accuracy(key, f"{g}_{s}") - accuracy(key, "clean") for s in range(1, 6)] for g in groups])
        image = ax.imshow(grid, cmap="RdBu", vmin=-60, vmax=60)
        for i, j in np.ndindex(grid.shape):
            ax.text(j, i, f"{grid[i, j]:+.0f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(grid[i, j]) > 35 else "black")
        ax.set_xticks(range(5), [f"{s}" for s in range(1, 6)])
        ax.set_yticks(range(len(groups)), [CORRUPTIONS[g][0] for g in groups], fontsize=8)
        ax.set(title=f"{MODEL_NAMES[key]} (gốc {accuracy(key, 'clean'):.1f}%)", xlabel="Mức suy giảm")
    fig.colorbar(image, ax=axes, shrink=0.8, label="Thay đổi accuracy (điểm %)")
    fig.savefig(output / "corruption-heatmap.png", dpi=150)
    plt.close(fig)

    # 3. Module 1 filters on clean images.
    filters = [c for c in conditions if c.family == "filter"]
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    positions = np.arange(len(filters))
    width = 0.27
    for offset, key in zip((-width, 0, width), keys):
        deltas = [accuracy(key, c.name) - accuracy(key, "clean") for c in filters]
        ax.barh(positions + offset, deltas, height=width, color=COLORS[key], label=MODEL_NAMES[key])
    ax.set_yticks(positions, [c.label for c in filters], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Thay đổi accuracy so với ảnh gốc (điểm %)", title="Bộ lọc Module 1 áp lên ảnh sạch")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(output / "module1-filters.png", dpi=150)
    plt.close(fig)

    # 4. Rescue: corruption alone, corruption + restoration, restoration on clean images.
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 5.5), sharey=True, layout="constrained")
    for ax, key in zip(axes, keys):
        labels, values, colors = [], [], []
        for corruption, severity, restorers in RESCUES:
            labels.append(f"{CORRUPTIONS[corruption][0]} = {CORRUPTIONS[corruption][1][severity - 1]:g}")
            values.append(accuracy(key, f"{corruption}_{severity}"))
            colors.append("#e11d48")
            for restorer in restorers:
                labels.append(f"   → {RESTORERS[restorer][0]}")
                values.append(accuracy(key, f"rescue_{corruption}_{severity}_{restorer}"))
                colors.append("#059669")
        ax.barh(range(len(values)), values, color=colors)
        ax.axvline(accuracy(key, "clean"), color="black", linestyle="--", linewidth=1, label="Ảnh gốc")
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.invert_yaxis()
        ax.set(xlim=(0, 100), xlabel="Accuracy (%)", title=MODEL_NAMES[key])
        ax.grid(axis="x", alpha=0.3)
    axes[0].legend(loc="lower right")
    fig.suptitle("Suy giảm (đỏ) và sau khi lọc khôi phục (xanh)")
    fig.savefig(output / "rescue.png", dpi=150)
    plt.close(fig)

    # 5. One test image under every corruption at severity 3.
    index = int(next((i for i, label in enumerate(np.load(output / "predictions.npz")["labels"]) if label == 4), 0))
    picks = [c for c in conditions if c.name == "clean" or (c.family == "corruption" and c.severity == 3)]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), layout="constrained")
    for ax, condition in zip(axes.flat, picks):
        ax.imshow(cv2.cvtColor(condition.transform(images[index], index), cv2.COLOR_BGR2RGB))
        ax.set_title(condition.label, fontsize=9)
        ax.axis("off")
    fig.suptitle("Một ảnh test (nhựa) ở mức suy giảm 3")
    fig.savefig(output / "examples.png", dpi=130)
    plt.close(fig)


def markdown(report: dict, output: Path):
    rows, keys = report["rows"], list(report["checkpoints"])

    def cell(key, name):
        row = next(r for r in rows if r["model"] == key and r["condition"] == name)
        mark = "*" if row.get("mcnemar_p_holm", 1) < 0.05 else ""
        return f"{100 * row['accuracy']:.1f} ({row['delta_points']:+.1f}){mark}"

    lines = ["# Thí nghiệm filter × model (tự sinh)", "",
             f"{report['images']} ảnh test, ba checkpoint cố định. Ô bảng: accuracy % (thay đổi so với ảnh gốc, điểm %). "
             "Dấu * = khác ảnh gốc có ý nghĩa theo McNemar chính xác, hiệu chỉnh Holm trên mọi điều kiện của mô hình đó.", "",
             "## Tóm tắt", "", "| Mô hình | Ảnh gốc | TB 45 điều kiện suy giảm | Giảm tương đối | Số điều kiện khác có ý nghĩa |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for key in keys:
        s = report["summary"][key]
        lines.append(f"| {MODEL_NAMES[key]} | {100 * s['clean_accuracy']:.1f}% | {100 * s['mean_corruption_accuracy']:.1f}% "
                     f"| {100 * s['relative_drop']:.1f}% | {s['significant_conditions_holm']} |")
    header = "| Điều kiện | " + " | ".join(MODEL_NAMES[k] for k in keys) + " |"
    divider = "| --- |" + " ---: |" * len(keys)
    for title, family in (("Bộ lọc Module 1 trên ảnh sạch", "filter"), ("Suy giảm 5 mức", "corruption"),
                          ("Đối chứng: bộ lọc khôi phục trên ảnh sạch", "control"), ("Khôi phục sau suy giảm", "rescue")):
        lines += ["", f"## {title}", "", header, divider]
        for name in dict.fromkeys(r["condition"] for r in rows if r["family"] == family):
            label = next(r["label"] for r in rows if r["condition"] == name)
            lines.append(f"| {label} | " + " | ".join(cell(k, name) for k in keys) + " |")
    lines += ["", "## Khôi phục so với ảnh hỏng (cùng mô hình, theo cặp)", "",
              "Số ảnh đúng khi chỉ bị suy giảm → sau bộ lọc khôi phục; mất/được = ảnh chỉ đúng ở một bên. "
              "p theo McNemar chính xác, hiệu chỉnh Holm trên toàn bộ bảng.", "",
              "| Mô hình | Suy giảm → bộ lọc | Đúng | Thay đổi | Mất / được | p (Holm) |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for r in report["rescue_vs_corrupted"]:
        lines.append(f"| {MODEL_NAMES[r['model']]} | {r['corruption']} → {RESTORERS[r['restorer']][0]} "
                     f"| {r['corrupted_correct']} → {r['rescued_correct']} "
                     f"| {r['rescued_correct'] - r['corrupted_correct']:+d} "
                     f"| {r['lost']} / {r['gained']} | {r['mcnemar_p_holm']:.2g} |")
    (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/filter-robustness-20261119")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--limit", type=int, help="first N test images only, for a quick check")
    args = parser.parse_args()
    report, images, conditions = evaluate(args.output, args.threads, args.limit)
    figures(report, images, conditions, args.output)
    markdown(report, args.output)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
