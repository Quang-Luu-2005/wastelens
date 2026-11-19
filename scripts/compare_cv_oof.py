"""Compare candidates across CV runs that share folds and seeds, using paired out-of-fold scores.

Each candidate is NAME=CONFIG[+CONFIG...]:EPOCHS:MEMBERS, where MEMBERS is 1 (single networks,
scored seed by seed and averaged within each fold) or the number of seeds to average, and
EPOCHS is a number or "best" (the budget with the highest mean fold macro-F1 for that candidate).

Example:
  python scripts/compare_cv_oof.py --runs outputs/kaggle-mlp-cv-20261111/cv outputs/kaggle-mlp-big-cv/cv \
      --candidates one=extra_lab:best:1 ensemble5=extra_lab:300:5 wide=extra_lab_wide:best:1 \
      --reference ensemble5 --output reports/single-vs-ensemble.json
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

from wastelens.cv import CONFIGS, ensemble_log_probs, fold_scores
from wastelens.model import MLPVariant
from wastelens.stats import mcnemar_exact


def load_runs(directories: list[Path]):
    """Merge OOF arrays from several runs after checking they share labels, folds, seeds and budgets."""
    arrays, reference = {}, None
    for directory in directories:
        selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
        data = np.load(directory / "oof-log-probs.npz")
        key = (selection["seeds"], selection["eval_epochs"], selection["manifest_sha256"])
        if reference is None:
            reference = (data["labels"], data["assignments"], key, selection["class_names"])
        elif not (np.array_equal(data["labels"], reference[0]) and np.array_equal(data["assignments"], reference[1])
                  and key == reference[2]):
            raise ValueError(f"{directory} does not share labels, folds, seeds and epoch budgets with the first run")
        for name in data.files:
            if name not in ("labels", "assignments"):
                if name in arrays:
                    raise ValueError(f"Configuration {name} appears in more than one run")
                arrays[name] = data[name]
    labels, assignments, (seeds, eval_epochs, _), classes = reference
    return arrays, labels, assignments, seeds, eval_epochs, classes


def candidate_scores(arrays, labels, assignments, configs, epoch_index, members, num_classes):
    """Fold scores, pooled accuracies, and predictions for the paired first-seed test."""
    folds, pooled_accuracies, predictions = [], [], []
    for repeat, folds_of_repeat in enumerate(assignments):
        if members == 1:
            # Single networks: score each seed, then average the seeds within each fold.
            seed_predictions = arrays[configs[0]][repeat, :, epoch_index].argmax(axis=-1)
            per_seed = [fold_scores(labels, predicted, folds_of_repeat, num_classes)
                        for predicted in seed_predictions]
            folds += [tuple(values) for values in np.mean(per_seed, axis=0)]
            pooled_accuracies.extend((seed_predictions == labels).mean(axis=1))
            predictions.append(seed_predictions[0])
        else:
            stacked = np.concatenate([arrays[name][repeat, :members, epoch_index] for name in configs])
            predicted = ensemble_log_probs(stacked).argmax(1)
            folds += fold_scores(labels, predicted, folds_of_repeat, num_classes)
            pooled_accuracies.append((predicted == labels).mean())
            predictions.append(predicted)
    return folds, pooled_accuracies, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    arrays, labels, assignments, seeds, eval_epochs, classes = load_runs(args.runs)
    num_classes = len(classes)
    results = {}
    for spec in args.candidates:
        name, body = spec.split("=", 1)
        config_part, epochs, members = body.split(":")
        configs, members = config_part.split("+"), int(members)
        if any(config not in arrays for config in configs) or not 1 <= members <= len(seeds):
            raise ValueError(f"Bad candidate {spec}: configs available {sorted(arrays)}, up to {len(seeds)} members")
        budgets = eval_epochs if epochs == "best" else [int(epochs)]
        # "best" picks the budget by mean fold macro-F1, applied the same way to every candidate.
        scored = {budget: candidate_scores(arrays, labels, assignments, configs, eval_epochs.index(budget),
                                           members, num_classes) for budget in budgets}
        budget = max(scored, key=lambda b: statistics.mean(f1 for _, f1 in scored[b][0]))
        folds, pooled_accuracies, predictions = scored[budget]
        parameters = sum(sum(p.numel() for p in MLPVariant(num_classes=num_classes, **CONFIGS[c].model_config()).parameters())
                         for c in configs) * members
        f1 = [score[1] for score in folds]
        results[name] = {
            "configs": configs, "epochs": budget, "members_per_config": members, "parameters": parameters,
            "fold_macro_f1_mean": statistics.mean(f1), "fold_macro_f1_se": statistics.stdev(f1) / math.sqrt(len(f1)),
            "fold_accuracy_mean": statistics.mean(score[0] for score in folds),
            "oof_accuracy_mean": statistics.mean(pooled_accuracies),
            "_folds": folds, "_predictions": predictions,
        }

    reference = results[args.reference]
    for name, result in results.items():
        # Paired differences on identical folds remove most of the split-to-split variation.
        differences = [mine[1] - ref[1] for mine, ref in zip(result["_folds"], reference["_folds"], strict=True)]
        result["delta_vs_reference"] = statistics.mean(differences)
        result["delta_se"] = statistics.stdev(differences) / math.sqrt(len(differences)) if name != args.reference else 0.0
        result["folds_better"] = f"{sum(d > 0 for d in differences)}/{len(differences)}"
        result["mcnemar_per_repeat"] = [mcnemar_exact(ref == labels, mine == labels)["p_value"]
                                        for mine, ref in zip(result["_predictions"], reference["_predictions"], strict=True)]

    print(f"{'candidate':<14}{'epochs':>7}{'params':>11}{'fold F1':>9}{'+-SE':>7}{'OOF acc':>9}"
          f"{'delta vs ' + args.reference:>20}{'+-SE':>7}{'better':>8}")
    for name, r in sorted(results.items(), key=lambda item: -item[1]["fold_macro_f1_mean"]):
        print(f"{name:<14}{r['epochs']:>7}{r['parameters']:>11,}{r['fold_macro_f1_mean']:>9.4f}{r['fold_macro_f1_se']:>7.4f}"
              f"{r['oof_accuracy_mean']:>9.4f}{r['delta_vs_reference']:>+20.4f}{r['delta_se']:>7.4f}{r['folds_better']:>8}")
    if args.output:
        summary = {"runs": [path.as_posix() for path in args.runs], "reference": args.reference,
                   "folds": int(assignments.max()) + 1, "repeats": len(assignments), "seeds": seeds,
                   "note": "Fold SEs treat overlapping folds as independent; use them to rank, not as confidence intervals.",
                   "candidates": {name: {k: v for k, v in r.items() if not k.startswith("_")} for name, r in results.items()}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, default=lambda item: item.item()) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
