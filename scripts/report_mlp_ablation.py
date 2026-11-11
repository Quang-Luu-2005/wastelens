"""Render tables and scientific figures from a completed MLP ablation run."""

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def report(root: Path, destination: Path):
    summary = json.loads((root / "run-summary.json").read_text(encoding="utf-8"))
    source = json.loads((root / "source-snapshot.json").read_text(encoding="utf-8"))
    winner, seeds = summary["winner"], summary["seeds"]
    baseline_rows = summary["test"]["baseline"]
    winner_rows = summary["test"][winner]
    rows = summary["validation"]
    by_name = {row["name"]: row for row in rows}
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    names = [row["name"] for row in rows]
    for ax, field, label in zip(axes, ("val_accuracy", "val_macro_f1"), ("Accuracy (%)", "Macro-F1 (%)"), strict=True):
        ax.bar(names, [100 * row[f"{field}_mean"] for row in rows],
               yerr=[100 * row[f"{field}_std"] for row in rows], capsize=3,
               color=["#bd6f37" if name == winner else "#267b69" for name in names])
        ax.tick_params(axis="x", rotation=55)
        ax.set(ylabel=label, ylim=(0, 100), title="Validation: mean and sample SD over two seeds")
    fig.savefig(root / "validation-comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(seeds), figsize=(6 * len(seeds), 7), squeeze=False, layout="constrained")
    for column, seed in enumerate(seeds):
        for name, color in (("baseline", "#7a8490"), (winner, "#267b69")):
            with (root / f"seed-{seed}" / name / "history.csv").open(newline="", encoding="utf-8") as stream:
                history = list(csv.DictReader(stream))
            epochs = [int(row["epoch"]) for row in history]
            axes[0, column].plot(epochs, [float(row["val_accuracy"]) for row in history], label=name, color=color, alpha=0.8)
            axes[1, column].plot(epochs, [float(row["val_ce"]) for row in history], label=name, color=color, alpha=0.8)
        axes[0, column].set(title=f"Seed {seed}", ylabel="Validation accuracy", ylim=(0, 1))
        axes[1, column].set(xlabel="Epoch", ylabel="Validation plain cross-entropy")
        axes[0, column].legend()
        axes[1, column].legend()
    fig.savefig(root / "baseline-winner-learning-curves.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(seeds), figsize=(6 * len(seeds), 5.2), squeeze=False, layout="constrained")
    for ax, trial in zip(axes[0], winner_rows["trials"], strict=True):
        matrix = trial["confusion_matrix"]
        classes = list(trial["per_class"])
        heatmap = ax.imshow(matrix, cmap="Blues")
        threshold = max(max(row) for row in matrix) / 2
        for i, row in enumerate(matrix):
            for j, count in enumerate(row):
                ax.text(j, i, int(count), ha="center", va="center", color="white" if count > threshold else "#172438")
        ax.set(xticks=range(6), yticks=range(6), xticklabels=classes, yticklabels=classes,
               xlabel="Prediction", ylabel="Actual", title=f"{winner}, seed {trial['seed']}: test")
        ax.tick_params(axis="x", rotation=45)
        fig.colorbar(heatmap, ax=ax, shrink=0.8)
    fig.savefig(root / "winner-confusion-matrices.png", dpi=170)
    plt.close(fig)

    lines = [
        "# Kết quả nâng cấp MLP Lab02 trên TrashNet", "",
        "Toàn bộ mô hình khởi tạo ngẫu nhiên, không sử dụng MobileNet hay pretrained.", "",
        f"- Cấu hình được chọn bằng validation: **{winner}**.",
        f"- Test accuracy trung bình baseline: **{baseline_rows['accuracy_mean']:.2%}**.",
        f"- Test accuracy trung bình cấu hình được chọn: **{winner_rows['accuracy_mean']:.2%}**.",
        f"- Chênh lệch accuracy: **{summary['accuracy_delta_percentage_points']:+.2f} điểm phần trăm**.", "",
        "## Validation của tất cả cấu hình", "",
        "Các giá trị là trung bình ± độ lệch chuẩn mẫu trên seed 42 và 43. "
        "Accuracy được ghi tại checkpoint chọn bằng macro-F1, không lấy epoch có accuracy cao nhất riêng biệt.", "",
        "| Cấu hình | Tham số | Train accuracy | Val accuracy | Val macro-F1 | Chênh val accuracy so với cha |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        delta = 100 * (row["val_accuracy_mean"] - by_name[row["parent"]]["val_accuracy_mean"]) if row["parent"] else None
        delta_text = f"{delta:+.2f} đpt" if delta is not None else "—"
        lines.append(f"| {row['name']} | {row['parameters']:,} | {row['train_accuracy_mean']:.2%} | "
                     f"{100*row['val_accuracy_mean']:.2f} ± {100*row['val_accuracy_std']:.2f}% | "
                     f"{row['val_macro_f1_mean']:.4f} ± {row['val_macro_f1_std']:.4f} | {delta_text} |")
    lines += ["", "## Test sau khi chọn cấu hình", "",
              "| Cấu hình | Seed | Epoch được chọn | Test accuracy | Test macro-F1 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    paired_changes = []
    for name in dict.fromkeys(("baseline", winner)):
        for trial in summary["test"][name]["trials"]:
            metrics = json.loads((root / f"seed-{trial['seed']}" / name / "metrics.json").read_text(encoding="utf-8"))
            lines.append(f"| {name} | {trial['seed']} | {metrics['best_epoch']} | {trial['accuracy']:.2%} | {trial['macro_f1']:.4f} |")
    for base_trial, winning_trial in zip(baseline_rows["trials"], winner_rows["trials"], strict=True):
        assert base_trial["seed"] == winning_trial["seed"]
        paired_changes.append(100 * (winning_trial["accuracy"] - base_trial["accuracy"]))
    lines += ["", "Chênh lệch accuracy theo từng seed: " + ", ".join(
              f"seed {seed}: {change:+.2f} đpt" for seed, change in zip(seeds, paired_changes, strict=True)) + ".", "",
              "## Giới hạn và khả năng tái lập", "",
              "- Hai seed là thí nghiệm thăm dò, chưa đủ kết luận ý nghĩa thống kê hay giới hạn tối ưu của MLP.",
              "- Tập test đã được xem ở các thí nghiệm trước. Không phải một holdout mới chưa sử dụng.",
              "- Cùng 600 epoch, batch 64, RGB 64×64, Xavier-normal, lật ngang p=0,5 và float32.",
              "- So sánh chính dùng baseline chạy lại trong cùng pipeline. Không trộn với baseline lịch sử dùng cách sinh batch/augmentation khác.",
              "- Nhánh AdamW/cosine thay cả optimizer, weight decay và lịch LR. Không tách riêng tác động của ba thành phần này.",
              f"- Manifest SHA-256: `{summary['manifest_sha256']}`.",
              f"- Snapshot nguồn SHA-256: `{source['sha256']}`.", "",
              "Xem giao thức và công thức loss trong [mlp-ablation.md](../docs/mlp-ablation.md).", ""]
    lines += ["## Nhận xét", "",
              "- Thêm tầng làm train accuracy tăng mạnh nhưng validation macro-F1 không tăng ở thí nghiệm này. Thêm độ sâu đơn thuần chưa đủ.",
              "- Tăng độ rộng cải thiện validation. Dropout 0,25 dùng riêng gây suy giảm rõ rệt; khi bổ sung LayerNorm, mạng học tốt trở lại.",
              "- Ba loss mới đều có validation macro-F1 thấp hơn cấu hình cha dùng CE trong lần chạy này. Không có bằng chứng chúng cải thiện chất lượng ở các thiết lập đã thử.",
              "- Cấu hình được chọn dùng các tầng ẩn 256→64→32, LayerNorm trước ReLU và dropout 0,25 sau ReLU, CE và Adam với LR 0,001.",
              "- Kết quả cho thấy có thể cải thiện MLP từ Lab02, đồng thời chưa chứng minh đây là giới hạn của kiến trúc này.", ""]
    figure_dir = destination.parent / "figures" / destination.stem
    figure_dir.mkdir(parents=True, exist_ok=True)
    lines += ["## Biểu đồ", ""]
    for filename, title in (("validation-comparison.png", "Đối chiếu validation trên hai seed"),
                            ("baseline-winner-learning-curves.png", "Đường học của baseline và cấu hình được chọn"),
                            ("winner-confusion-matrices.png", "Confusion matrix test của cấu hình được chọn")):
        shutil.copy2(root / filename, figure_dir / filename)
        relative = (figure_dir / filename).relative_to(destination.parent).as_posix()
        lines += [f"![{title}]({relative})", ""]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "run-summary.json", destination.with_suffix(".json"))
    shutil.copy2(root / "validation-comparison.csv", destination.with_suffix(".csv"))
    lines += [f"Dữ liệu kết quả: [JSON]({destination.with_suffix('.json').name}), "
              f"[CSV]({destination.with_suffix('.csv').name}).", ""]
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--report", type=Path, default=Path("reports/mlp-ablation-20261111.md"))
    args = parser.parse_args()
    report(args.run, args.report)
