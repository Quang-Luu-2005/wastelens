"""Run the binary versus multi-class experiment and write tables, figures and a report.

Five-fold stratified CV on the 2,144 train + val images, using the same folds as repeat 0 of the
deployed model's cross-validation. The test split is not read.

  python scripts/binary_vs_multiclass.py --output reports/binary-vs-multiclass-20261119 --workers 5
"""

import argparse
import json
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wastelens.binary_multiclass import SETUPS, TASKS, initial_gradient_check, task_labels, train_fold  # noqa: E402
from wastelens.cv import stratified_folds  # noqa: E402
from wastelens.stats import agreement_metrics, confusion, mcnemar_exact  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/descriptor-cache/train-val-hog_hsv_lab-836d92622ad5.npz"
NAMES = {
    "six_softmax": "6 lớp · softmax + CE", "six_ovr": "6 lớp · 6 sigmoid one-vs-rest + BCE",
    "fiber_sigmoid": "Giấy/khác · 1 sigmoid + BCE", "fiber_softmax2": "Giấy/khác · 2 softmax + CE",
    "trash_sigmoid": "Trash/khác · 1 sigmoid + BCE", "trash_sigmoid_weighted": "Trash/khác · BCE có pos_weight",
    "fiber_from_six": "Giấy/khác · gộp xác suất từ mô hình 6 lớp", "trash_from_six": "Trash/khác · gộp từ mô hình 6 lớp",
}
# One colour per setup in every panel, so lines can be matched across the three plots.
COLORS = {"six_softmax": "#1f77b4", "six_ovr": "#ff7f0e", "fiber_sigmoid": "#2ca02c",
          "fiber_softmax2": "#d62728", "trash_sigmoid": "#9467bd"}
_DATA = {}


def _init(threads: int):
    torch.set_num_threads(threads)
    data = np.load(CACHE)
    _DATA.update(features=data["features"], flipped=data["flipped"], labels=data["labels"])


def _task(job):
    name, fold, seed, folds, epochs = job
    setup = SETUPS[name]
    labels = task_labels(setup.task, _DATA["labels"])
    train, held = folds != fold, folds == fold
    result = train_fold(setup, _DATA["features"][train], _DATA["flipped"][train], labels[train],
                        _DATA["features"][held], labels[held], seed, epochs)
    return name, fold, seed, result


def run(output: Path, seeds: list[int], epochs: int, workers: int, threads: int):
    data = np.load(CACHE)
    six = data["labels"]
    folds = stratified_folds(six, 5, 2026)  # Repeat 0 of the deployed model's CV: identical folds.
    jobs = [(name, fold, seed, folds, epochs) for name in SETUPS for seed in seeds for fold in range(5)]
    oof = {name: {seed: np.zeros((len(six), len(TASKS[SETUPS[name].task][0])), np.float32) for seed in seeds}
           for name in SETUPS}
    histories, extra = {name: [] for name in SETUPS}, {}
    started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=context, initializer=_init, initargs=(threads,)) as pool:
        for done, (name, fold, seed, result) in enumerate(pool.map(_task, jobs), start=1):
            oof[name][seed][folds == fold] = result["probabilities"]
            histories[name].append({"fold": fold, "seed": seed, "history": result["history"]})
            extra[name] = {"parameters": result["parameters"], "head_parameters": result["head_parameters"]}
            print(f"[{done}/{len(jobs)}] {name} fold={fold} seed={seed} {result['seconds']:.0f}s "
                  f"(elapsed {(time.perf_counter() - started) / 60:.1f} min)", flush=True)

    # Collapsed evaluation: the six-class model answers the binary question by summing probabilities.
    for task, group in (("fiber", [0, 3]), ("trash", [5])):
        oof[f"{task}_from_six"] = {seed: np.stack([1 - oof["six_softmax"][seed][:, group].sum(1),
                                                   oof["six_softmax"][seed][:, group].sum(1)], axis=1)
                                   for seed in seeds}
    task_of = {**{name: setup.task for name, setup in SETUPS.items()}, "fiber_from_six": "fiber", "trash_from_six": "trash"}

    rows = {}
    for name, per_seed in oof.items():
        labels = task_labels(task_of[name], six)
        classes = len(TASKS[task_of[name]][0])
        metrics = [agreement_metrics(labels, per_seed[seed].argmax(1), classes) for seed in seeds]
        rows[name] = {"task": task_of[name], "classes": classes, "chance_level": 1 / classes,
                      **{key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]},
                      "per_seed": metrics,
                      "confusion_seed0": confusion(labels, per_seed[seeds[0]].argmax(1), classes).tolist()}
        if name in histories:
            curves = histories[name]
            rows[name].update(extra[name])
            rows[name]["epoch_seconds"] = float(np.mean([h["seconds"] for c in curves for h in c["history"][1:]]))
            rows[name]["final_train_loss"] = float(np.mean([c["history"][-1]["train_loss"] for c in curves]))

    def paired(a, b):
        task = task_of[a]
        labels = task_labels(task, six)
        return [mcnemar_exact(oof[a][s].argmax(1) == labels, oof[b][s].argmax(1) == labels) for s in seeds]

    comparisons = {
        "fiber_sigmoid_vs_fiber_softmax2": paired("fiber_sigmoid", "fiber_softmax2"),
        "fiber_sigmoid_vs_fiber_from_six": paired("fiber_sigmoid", "fiber_from_six"),
        "six_softmax_vs_six_ovr": paired("six_softmax", "six_ovr"),
        "trash_sigmoid_vs_weighted": paired("trash_sigmoid", "trash_sigmoid_weighted"),
    }
    gradients = {name: initial_gradient_check(setup, data["features"], task_labels(setup.task, six))
                 for name, setup in SETUPS.items()}
    output.mkdir(parents=True, exist_ok=True)
    report = {"images": int(len(six)), "folds": 5, "fold_seed": 2026, "seeds": seeds, "epochs": epochs,
              "minutes": (time.perf_counter() - started) / 60, "results": rows, "comparisons": comparisons,
              "initial_gradients": gradients, "note": "Out-of-fold predictions on train + val; test split not read."}
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "histories.json").write_text(json.dumps(histories) + "\n", encoding="utf-8")
    return report, histories


def figures(report, histories, output: Path):
    def mean_curve(name, key):
        curves = [[row.get(key) for row in c["history"]] for c in histories[name]]
        epochs = [row["epoch"] for row in histories[name][0]["history"] if row.get(key) is not None]
        values = np.array([[v for v in curve if v is not None] for curve in curves], dtype=float)
        return epochs, values.mean(axis=0)

    shown = ["six_softmax", "six_ovr", "fiber_sigmoid", "fiber_softmax2", "trash_sigmoid"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
    for name in shown:
        classes = report["results"][name]["classes"]
        epochs, train = mean_curve(name, "train_loss")
        axes[0].plot(epochs, train, label=NAMES[name], color=COLORS[name])
        # Dividing by ln C puts every softmax loss on the same scale: 1.0 is the loss of a uniform guess.
        if SETUPS[name].head == "softmax":
            axes[1].plot(epochs, train / math.log(classes), label=NAMES[name], color=COLORS[name])
        epochs, gradient = mean_curve(name, "logit_gradient_norm")
        axes[2].plot(epochs, gradient, label=NAMES[name], color=COLORS[name])
    axes[0].set(title="Loss huấn luyện", xlabel="Epoch", ylabel="Loss", yscale="log")
    axes[1].set(title="Loss softmax chia cho ln C (1 = đoán đều)", xlabel="Epoch", yscale="log")
    axes[2].set(title="‖∂L/∂z‖ trung bình mỗi mẫu", xlabel="Epoch", yscale="log")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.savefig(output / "training-curves.png", dpi=150)
    plt.close(fig)

    order = ["six_softmax", "six_ovr", "fiber_sigmoid", "fiber_softmax2", "fiber_from_six",
             "trash_sigmoid", "trash_sigmoid_weighted", "trash_from_six"]
    metrics = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced acc."), ("kappa", "Cohen κ"), ("mcc", "MCC")]
    fig, ax = plt.subplots(figsize=(13, 5.5), layout="constrained")
    width = 0.2
    for offset, (key, label) in enumerate(metrics):
        ax.bar(np.arange(len(order)) + (offset - 1.5) * width, [report["results"][n][key] for n in order], width, label=label)
    ax.scatter(np.arange(len(order)), [report["results"][n]["majority_accuracy"] for n in order], color="black",
               marker="_", s=400, zorder=3, label="Accuracy nếu luôn đoán lớp đông nhất")
    ax.set_xticks(np.arange(len(order)), [NAMES[n] for n in order], rotation=20, ha="right", fontsize=8)
    ax.set(ylim=(0, 1.15), title="Chỉ số out-of-fold trên 2.144 ảnh (trung bình 2 seed)")
    ax.legend(fontsize=8, loc="upper left", ncol=5)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(output / "metrics.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    for ax, name in zip(axes, ["six_softmax", "fiber_sigmoid", "fiber_from_six"]):
        matrix = np.array(report["results"][name]["confusion_seed0"])
        classes = TASKS[report["results"][name]["task"]][0]
        ax.imshow(matrix, cmap="Blues")
        for i, j in np.ndindex(matrix.shape):
            ax.text(j, i, matrix[i, j], ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black", fontsize=8)
        ax.set_xticks(range(len(classes)), classes, rotation=30, fontsize=8)
        ax.set_yticks(range(len(classes)), classes, fontsize=8)
        ax.set(title=NAMES[name], xlabel="Dự đoán", ylabel="Thật")
    fig.savefig(output / "confusion.png", dpi=150)
    plt.close(fig)


def markdown(report, output: Path):
    rows = report["results"]
    lines = ["# Binary so với multi-class (tự sinh)", "",
             f"{report['images']} ảnh train + val, CV 5 fold (seed fold {report['fold_seed']}), seed {report['seeds']}, "
             f"{report['epochs']} epoch, cùng descriptor và MLP 256-64-32. Không đọc tập test.", "",
             "| Cấu hình | C | Ngẫu nhiên | Lớp đông nhất | Accuracy | Balanced acc. | Macro-F1 | κ | MCC | Tham số đầu ra | Giây/epoch |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, r in rows.items():
        head = f"{r['head_parameters']}" if "head_parameters" in r else "—"
        seconds = f"{r['epoch_seconds']:.2f}" if "epoch_seconds" in r else "—"
        lines.append(f"| {NAMES[name]} | {r['classes']} | {r['chance_level']:.1%} | {r['majority_accuracy']:.1%} "
                     f"| {r['accuracy']:.1%} | {r['balanced_accuracy']:.1%} | {r['macro_f1']:.3f} | {r['kappa']:.3f} "
                     f"| {r['mcc']:.3f} | {head} | {seconds} |")
    lines += ["", "## So sánh theo cặp (McNemar chính xác, từng seed)", "", "| So sánh | Chỉ A đúng / chỉ B đúng | p |",
              "| --- | --- | --- |"]
    for name, tests in report["comparisons"].items():
        lines.append(f"| {name} | " + "; ".join(f"{t['only_a_correct']} / {t['only_b_correct']}" for t in tests)
                     + " | " + "; ".join(f"{t['p_value']:.2g}" for t in tests) + " |")
    lines += ["", "## Lúc khởi tạo (256 ảnh, chế độ eval)", "", "| Cấu hình | Loss ban đầu | ‖∂L/∂z‖ TB | Sai số giải tích vs autograd |",
              "| --- | ---: | ---: | ---: |"]
    for name, g in report["initial_gradients"].items():
        lines.append(f"| {NAMES[name]} | {g['initial_loss']:.3f} | {g['initial_gradient_norm']:.3f} "
                     f"| {g['analytic_vs_autograd_max_error']:.1e} |")
    (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/binary-vs-multiclass-20261119")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--replot", action="store_true", help="redraw figures from an existing --output")
    args = parser.parse_args()
    if args.replot:  # Redraw figures and tables from saved results without training again.
        report = json.loads((args.output / "results.json").read_text(encoding="utf-8"))
        histories = json.loads((args.output / "histories.json").read_text(encoding="utf-8"))
    else:
        report, histories = run(args.output, args.seeds, args.epochs, args.workers, args.threads)
    figures(report, histories, args.output)
    markdown(report, args.output)
    print(json.dumps({n: {k: round(r[k], 4) for k in ("accuracy", "balanced_accuracy", "kappa", "mcc")}
                      for n, r in report["results"].items()}, indent=2))


if __name__ == "__main__":
    main()
