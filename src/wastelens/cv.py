"""Repeated stratified cross-validation, seed ensembles and a one-time test for the descriptor MLP.

Workflow (the test split is read only by `final`):

1. `cv`: pool train + val (2,144 images), run k-fold CV repeated R times with S seeds per
   fold, and keep out-of-fold (OOF) probabilities at several epoch budgets. The epoch budget
   is a CV-selected hyperparameter, so no fold ever picks a checkpoint on its own held-out data.
2. Selection: for every (configuration, epochs, single vs ensemble) candidate, average macro-F1
   over the R x k folds, then apply the one-standard-error rule. Post-hoc logit adjustment for
   class priors is tuned by cross-fitting: each fold's tau comes from the other folds.
3. `final`: retrain the selected candidate on all 2,144 images and evaluate the locked test
   split once, with exact and bootstrap confidence intervals and a McNemar test.
"""

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import read_splits
from .mlp_experiments import ROUND3, cached_descriptors, fit_normalization, make_loss
from .model import MLPVariant, load_classifier
from .stats import accuracy_and_macro_f1, clopper_pearson, confusion, mcnemar_exact, paired_bootstrap
from .train import write_csv
from .train import write_json as _write_json


def write_json(path: Path, value) -> None:
    # NumPy scalars (float64, bool_) are converted to their Python equivalents first.
    _write_json(path, json.loads(json.dumps(value, default=lambda item: item.item())))

# Round-three configurations worth re-ranking under CV: the HOG + HSV parent, the selected
# HOG + HSV + Lab winner and the much smaller HSV-only model that nearly tied it.
CONFIGS = {experiment.name: experiment for experiment in ROUND3
           if experiment.name in ("baseline", "extra_lab", "hsv_only")}
TAU_GRID = tuple(round(0.25 * step, 2) for step in range(9))  # 0.0 to 2.0


def stratified_folds(labels: np.ndarray, folds: int, seed: int) -> np.ndarray:
    """Assign each image a fold so every class is spread as evenly as possible."""
    rng = np.random.default_rng(seed)
    assignment = np.empty(len(labels), dtype=np.int64)
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        rng.shuffle(members)
        # A random offset keeps leftover images of small classes from always landing in fold 0.
        assignment[members] = (np.arange(len(members)) + rng.integers(folds)) % folds
    return assignment


def feature_cache(image_root: Path, manifest: Path, representation: str, cache: Path, splits_to_use: tuple[str, ...]):
    """Descriptors (and horizontally flipped descriptors) for the requested splits, cached on disk."""
    splits, classes = read_splits(manifest)
    samples = [sample for split in splits_to_use for sample in splits[split]]
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    path = cache / f"{'-'.join(splits_to_use)}-{representation}-{digest[:12]}.npz"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        features = cached_descriptors(image_root, samples, representation, "cpu").numpy()
        flipped = cached_descriptors(image_root, samples, representation, "cpu", flipped=True).numpy()
        np.savez(path, features=features, flipped=flipped,
                 labels=np.array([label for _, label in samples]), paths=np.array([p for p, _ in samples]))
        print(f"Descriptors {representation} for {len(samples)} images: {time.perf_counter() - started:.1f} s", flush=True)
    data = np.load(path)
    return data["features"], data["flipped"], data["labels"], [str(p) for p in data["paths"]], classes


def fit_and_predict(experiment, train_x, flipped_x, train_y, eval_x, seed, epochs, eval_epochs,
                    num_classes, device="cpu", batch_size=64, return_model=False):
    """Train one MLP exactly as in round three and return eval log-probabilities at each epoch budget.

    Unlike round three, there is no checkpoint selection on held-out data; the cosine schedule,
    when used, spans the full run. Returns {epoch: log-probabilities as a NumPy array}, plus the
    trained model when `return_model` is set.
    """
    device = torch.device(device)
    train_x, flipped_x, eval_x = (torch.as_tensor(array, device=device) for array in (train_x, flipped_x, eval_x))
    train_y = torch.as_tensor(train_y, device=device)
    torch.manual_seed(seed)
    image_size = 128 if experiment.representation.endswith("128") else 64
    model = MLPVariant(image_size=image_size, num_classes=num_classes, **experiment.model_config()).to(device)
    # Standardization statistics come from this fold's training images only.
    fit_normalization(model, train_x, experiment)
    loss_fn = make_loss(experiment.loss, train_y, num_classes).to(device)
    optimizer_type = torch.optim.AdamW if experiment.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_type(model.parameters(), lr=0.001, weight_decay=experiment.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if experiment.cosine else None
    # Same generator offsets as round three, so a seed means the same order and flips.
    order_rng = torch.Generator(device=device).manual_seed(seed + 1000)
    flip_rng = torch.Generator(device=device).manual_seed(seed + 2000)
    outputs = {}
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_y), generator=order_rng, device=device)
        for indices in order.split(batch_size):
            flip = torch.rand((len(indices), 1), generator=flip_rng, device=device) < 0.5
            images = torch.where(flip, flipped_x[indices], train_x[indices])
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), train_y[indices])
            loss.backward()
            optimizer.step()
        if scheduler:
            scheduler.step()
        if epoch in eval_epochs:
            model.eval()
            with torch.inference_mode():
                outputs[epoch] = F.log_softmax(model(eval_x), dim=1).cpu().numpy()
    if not all(np.isfinite(values).all() for values in outputs.values()):
        raise RuntimeError(f"Non-finite predictions in {experiment.name}, seed {seed}")
    return (outputs, model.eval()) if return_model else outputs


# Worker processes load the cached descriptors once and keep them in module state.
_WORKER = {}


def _init_worker(cache_files: dict[str, str], threads: int):
    torch.set_num_threads(threads)
    # Deterministic cuBLAS needs a fixed workspace; set it before the first CUDA call.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    for representation, path in cache_files.items():
        data = np.load(path)
        _WORKER[representation] = (data["features"], data["flipped"], data["labels"])


def _run_task(task):
    name, repeat, fold, seed, folds_of_repeat, epochs, eval_epochs, device = task
    experiment = CONFIGS[name]
    features, flipped, labels = _WORKER[experiment.representation]
    train, held_out = folds_of_repeat != fold, folds_of_repeat == fold
    started = time.perf_counter()
    outputs = fit_and_predict(experiment, features[train], flipped[train], labels[train], features[held_out],
                              seed, epochs, eval_epochs, int(labels.max()) + 1, device)
    return task[:4], outputs, time.perf_counter() - started


def logit_adjust(log_probs: np.ndarray, log_prior: np.ndarray, tau: float) -> np.ndarray:
    """Subtract tau * log prior: tau = 0 keeps argmax(p); tau = 1 targets balanced error."""
    return log_probs - tau * log_prior


def ensemble_log_probs(log_probs: np.ndarray) -> np.ndarray:
    """Average probabilities over the leading (seed) axis and return log-probabilities."""
    return np.log(np.exp(log_probs).mean(axis=0) + 1e-12)


def fold_scores(labels, predicted, folds_of_repeat, num_classes):
    """(accuracy, macro-F1) for each fold of one repeat."""
    return [accuracy_and_macro_f1(labels[folds_of_repeat == fold], predicted[folds_of_repeat == fold], num_classes)
            for fold in range(folds_of_repeat.max() + 1)]


def cross_fitted_tau(scores, labels, folds_of_repeat, log_prior, num_classes):
    """Logit-adjusted predictions where each fold's tau maximizes macro-F1 on the other folds."""
    predicted = np.empty(len(labels), dtype=np.int64)
    taus = []
    for fold in range(folds_of_repeat.max() + 1):
        held_out = folds_of_repeat == fold
        rest = ~held_out
        best_tau = max(TAU_GRID, key=lambda tau: (accuracy_and_macro_f1(
            labels[rest], logit_adjust(scores[rest], log_prior, tau).argmax(1), num_classes)[1], -tau))
        taus.append(best_tau)
        predicted[held_out] = logit_adjust(scores[held_out], log_prior, best_tau).argmax(1)
    return predicted, taus


def summarize_cv(results: dict, labels: np.ndarray, assignments: np.ndarray, classes: list[str],
                 seeds: list[int], eval_epochs: list[int], parameters: dict) -> tuple[list[dict], dict]:
    """Score every candidate; results[name] has shape (repeats, seeds, epochs, images, classes)."""
    num_classes = len(classes)
    log_prior = np.log(np.bincount(labels, minlength=num_classes) / len(labels))
    trash = classes.index("trash")
    rows = []
    for name, log_probs in results.items():
        for epoch_index, epoch in enumerate(eval_epochs):
            for members in (1, len(seeds)) if len(seeds) > 1 else (1,):
                per_fold, pooled, trash_hits, adjusted_fold, tau_values = [], [], 0, [], []
                for repeat, folds_of_repeat in enumerate(assignments):
                    scores_by_seed = log_probs[repeat, :, epoch_index]
                    # A single member is scored seed by seed; an ensemble averages all seeds.
                    variants = ([scores_by_seed[index] for index in range(len(seeds))] if members == 1
                                else [ensemble_log_probs(scores_by_seed)])
                    repeat_folds, repeat_adjusted = [], []
                    for scores in variants:
                        predicted = scores.argmax(1)
                        repeat_folds.append(fold_scores(labels, predicted, folds_of_repeat, num_classes))
                        pooled.append(accuracy_and_macro_f1(labels, predicted, num_classes))
                        trash_hits += int(np.sum((labels == trash) & (predicted == trash)))
                        adjusted, taus = cross_fitted_tau(scores, labels, folds_of_repeat, log_prior, num_classes)
                        repeat_adjusted.append(fold_scores(labels, adjusted, folds_of_repeat, num_classes))
                        tau_values += taus
                    # Average single seeds within each fold, so every candidate has R x k fold scores.
                    per_fold += [tuple(value) for value in np.mean(repeat_folds, axis=0)]
                    adjusted_fold += [tuple(value) for value in np.mean(repeat_adjusted, axis=0)]
                f1 = [score[1] for score in per_fold]
                adjusted_f1 = [score[1] for score in adjusted_fold]
                trash_total = int(np.sum(labels == trash)) * len(pooled)
                rows.append({
                    "config": name, "epochs": epoch, "ensemble_size": members,
                    "parameters": parameters[name] * members,
                    "fold_macro_f1_mean": statistics.mean(f1),
                    # Folds overlap in training data, so this naive SE is optimistic; it only ranks candidates.
                    "fold_macro_f1_se": statistics.stdev(f1) / math.sqrt(len(f1)),
                    "fold_accuracy_mean": statistics.mean(score[0] for score in per_fold),
                    "oof_accuracy_mean": statistics.mean(score[0] for score in pooled),
                    "oof_macro_f1_mean": statistics.mean(score[1] for score in pooled),
                    "trash_recall": trash_hits / trash_total,
                    "adjusted_fold_macro_f1_mean": statistics.mean(adjusted_f1),
                    "adjusted_fold_accuracy_mean": statistics.mean(score[0] for score in adjusted_fold),
                    "tau_median": statistics.median(tau_values),
                })
    best = max(rows, key=lambda row: row["fold_macro_f1_mean"])
    threshold = best["fold_macro_f1_mean"] - best["fold_macro_f1_se"]
    # One-SE rule: the cheapest candidate statistically close to the best (fewer parameters, then epochs).
    eligible = [row for row in rows if row["fold_macro_f1_mean"] >= threshold]
    chosen = min(eligible, key=lambda row: (row["parameters"], row["epochs"]))
    use_adjustment = chosen["adjusted_fold_macro_f1_mean"] > chosen["fold_macro_f1_mean"]
    selection = {
        "rule": "max mean fold macro-F1, then one-SE rule preferring fewer parameters and epochs",
        "best": best, "chosen": chosen, "one_se_threshold": threshold,
        "logit_adjustment": use_adjustment,
    }
    # Paired comparison on OOF predictions: chosen candidate vs the round-three style single model.
    reference = max((row for row in rows if row["config"] == "extra_lab" and row["ensemble_size"] == 1),
                    key=lambda row: row["fold_macro_f1_mean"], default=None)
    if reference is not None and reference is not chosen:
        tests = []
        for repeat in range(len(assignments)):
            def predictions(row):
                scores = results[row["config"]][repeat, :, eval_epochs.index(row["epochs"])]
                return (ensemble_log_probs(scores) if row["ensemble_size"] > 1 else scores[0]).argmax(1)
            tests.append(mcnemar_exact(predictions(reference) == labels, predictions(chosen) == labels))
        selection["oof_mcnemar_vs_reference"] = {"reference": reference, "per_repeat": tests,
                                                 "note": "a = reference (first seed), b = chosen"}
    return rows, selection


def run_cv(args):
    eval_epochs = sorted(set(args.eval_epochs))
    epochs = max(eval_epochs)
    names = args.configs
    representations = {CONFIGS[name].representation for name in names}
    cache_files, labels, classes = {}, None, None
    for representation in representations:
        _, _, rep_labels, _, classes = feature_cache(args.image_root, args.manifest, representation, args.cache, ("train", "val"))
        if labels is not None and not np.array_equal(labels, rep_labels):
            raise RuntimeError("Descriptor caches disagree on image order")
        labels = rep_labels
        digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
        cache_files[representation] = str(args.cache / f"train-val-{representation}-{digest[:12]}.npz")
    assignments = np.stack([stratified_folds(labels, args.folds, args.split_seed + repeat) for repeat in range(args.repeats)])
    args.output.mkdir(parents=True, exist_ok=False)
    grid = [(name, repeat, fold, seed) for name in names for repeat in range(args.repeats)
            for fold in range(args.folds) for seed in args.seeds]
    # With --device cuda, tasks are spread round-robin over all visible GPUs.
    gpus = torch.cuda.device_count() if args.device == "cuda" else 0
    tasks = [(*key, assignments[key[1]], epochs, set(eval_epochs), f"cuda:{index % gpus}" if gpus else args.device)
             for index, key in enumerate(grid)]
    results = {name: np.zeros((args.repeats, len(args.seeds), len(eval_epochs), len(labels), len(classes)), np.float32)
               for name in names}
    started = time.perf_counter()
    print(f"{len(tasks)} fits on {args.workers} worker(s), {epochs} epochs each, scored at {eval_epochs}", flush=True)

    def store(key, outputs, seconds, done):
        name, repeat, fold, seed = key
        held_out = assignments[repeat] == fold
        for epoch_index, epoch in enumerate(eval_epochs):
            results[name][repeat, args.seeds.index(seed), epoch_index, held_out] = outputs[epoch]
        elapsed = time.perf_counter() - started
        print(f"[{done}/{len(tasks)}] {name} repeat={repeat} fold={fold} seed={seed} {seconds:.1f}s "
              f"(elapsed {elapsed / 60:.1f} min)", flush=True)

    if args.workers > 1:
        # Spawned workers never inherit a CUDA context from the parent.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(args.workers, mp_context=context, initializer=_init_worker,
                                 initargs=(cache_files, args.threads)) as pool:
            for done, (key, outputs, seconds) in enumerate(pool.map(_run_task, tasks), start=1):
                store(key, outputs, seconds, done)
    else:
        _init_worker(cache_files, args.threads)
        for done, task in enumerate(tasks, start=1):
            store(*_run_task(task), done)

    parameters = {name: sum(p.numel() for p in MLPVariant(num_classes=len(classes), **CONFIGS[name].model_config()).parameters())
                  for name in names}
    rows, selection = summarize_cv(results, labels, assignments, classes, args.seeds, eval_epochs, parameters)
    np.savez_compressed(args.output / "oof-log-probs.npz", labels=labels, assignments=assignments,
                        **{name: values for name, values in results.items()})
    # tau for the final model: tuned once on all pooled OOF scores of the chosen candidate.
    chosen = selection["chosen"]
    scores = results[chosen["config"]][:, :, eval_epochs.index(chosen["epochs"])]
    pooled = [ensemble_log_probs(scores[r]) if chosen["ensemble_size"] > 1 else scores[r, 0] for r in range(args.repeats)]
    log_prior = np.log(np.bincount(labels, minlength=len(classes)) / len(labels))
    final_tau = 0.0
    if selection["logit_adjustment"]:
        final_tau = max(TAU_GRID, key=lambda tau: (statistics.mean(accuracy_and_macro_f1(
            labels, logit_adjust(s, log_prior, tau).argmax(1), len(classes))[1] for s in pooled), -tau))
    selection.update({
        "final_tau": final_tau, "log_prior": log_prior.tolist(), "class_names": classes,
        "folds": args.folds, "repeats": args.repeats, "split_seed": args.split_seed, "seeds": args.seeds,
        "eval_epochs": eval_epochs, "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "images": int(len(labels)), "cv_minutes": (time.perf_counter() - started) / 60,
        "test_split_loaded": False,
    })
    write_json(args.output / "selection.json", selection)
    write_csv(args.output / "cv-candidates.csv", list(rows[0]), [row.values() for row in rows])
    print_table(rows, selection)


def print_table(rows, selection):
    print(f"\n{'config':<10}{'epochs':>7}{'seeds':>6}{'fold F1':>9}{'±SE':>7}{'OOF acc':>9}{'trash R':>9}{'+LA F1':>8}")
    for row in sorted(rows, key=lambda row: -row["fold_macro_f1_mean"]):
        print(f"{row['config']:<10}{row['epochs']:>7}{row['ensemble_size']:>6}{row['fold_macro_f1_mean']:>9.4f}"
              f"{row['fold_macro_f1_se']:>7.4f}{row['oof_accuracy_mean']:>9.4f}{row['trash_recall']:>9.3f}"
              f"{row['adjusted_fold_macro_f1_mean']:>8.4f}")
    chosen = selection["chosen"]
    print(f"\nChosen: {chosen['config']}, {chosen['epochs']} epochs, {chosen['ensemble_size']} seed(s), "
          f"logit adjustment {'on (tau ' + str(selection['final_tau']) + ')' if selection['logit_adjustment'] else 'off'}")


TEMPERATURE_GRID = tuple(round(0.5 + 0.05 * step, 2) for step in range(91))  # 0.5 to 5.0


def calibrate(log_probs: np.ndarray, temperature: float) -> np.ndarray:
    """Probabilities after dividing the ensemble log-probabilities by a temperature."""
    scaled = log_probs / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    probabilities = np.exp(scaled)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def fit_temperature(log_probs: list[np.ndarray], labels: np.ndarray) -> float:
    """Grid temperature minimizing mean negative log-likelihood over the given OOF repeats."""
    rows = np.arange(len(labels))

    def nll(temperature):
        return statistics.mean(float(-np.log(calibrate(scores, temperature)[rows, labels] + 1e-12).mean())
                               for scores in log_probs)
    return min(TEMPERATURE_GRID, key=nll)


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    """Top-1 ECE with equal-width confidence bins."""
    confidence, predicted = probabilities.max(axis=1), probabilities.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (confidence > low) & (confidence <= high)
        if inside.any():
            gap = float((predicted[inside] == labels[inside]).mean()) - float(confidence[inside].mean())
            error += float(inside.mean()) * abs(gap)
    return error


def reject_threshold(confidence: np.ndarray, correct: np.ndarray, target: float) -> float:
    """Lowest confidence threshold whose accepted predictions still reach the target accuracy."""
    order = np.argsort(-confidence, kind="stable")
    ranked = confidence[order]
    running = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    # Inference accepts every item at the threshold, so only score complete tie groups.
    group_end = np.r_[ranked[:-1] != ranked[1:], True][:len(ranked)]
    passing = np.flatnonzero(group_end & (running >= target))
    return float(ranked[passing[-1]]) if len(passing) else math.nextafter(1.0, math.inf)


def oof_calibration(log_probs: np.ndarray, labels: np.ndarray, assignments: np.ndarray, target: float) -> dict:
    """Temperature and threshold fitted on pooled OOF scores, plus a cross-fitted check of both.

    `log_probs` has shape (repeats, images, classes): one OOF ensemble per repeat.
    """
    num_classes = log_probs.shape[-1]
    repeats = list(log_probs)
    temperature = fit_temperature(repeats, labels)
    calibrated = [calibrate(scores, temperature) for scores in repeats]
    pooled = np.concatenate(calibrated)
    pooled_correct = pooled.argmax(1) == np.tile(labels, len(repeats))
    threshold = reject_threshold(pooled.max(1), pooled_correct, target)
    accepted = pooled.max(1) >= threshold
    # Cross-fitting: each fold gets T and the threshold from the other folds of its repeat.
    held_correct, held_accepted = [], []
    for repeat, folds_of_repeat in enumerate(assignments):
        for fold in range(folds_of_repeat.max() + 1):
            held_out = folds_of_repeat == fold
            rest = ~held_out
            fold_t = fit_temperature([repeats[repeat][rest]], labels[rest])
            rest_probabilities = calibrate(repeats[repeat][rest], fold_t)
            fold_threshold = reject_threshold(rest_probabilities.max(1),
                                              rest_probabilities.argmax(1) == labels[rest], target)
            held = calibrate(repeats[repeat][held_out], fold_t)
            held_correct.append(held.argmax(1) == labels[held_out])
            held_accepted.append(held.max(1) >= fold_threshold)
    held_correct, held_accepted = np.concatenate(held_correct), np.concatenate(held_accepted)
    scores = [accuracy_and_macro_f1(labels, scores.argmax(1), num_classes) for scores in repeats]
    return {
        "temperature": temperature, "uncertain_threshold": threshold, "target_accuracy": target,
        "oof_accuracy": statistics.mean(score[0] for score in scores),
        "oof_macro_f1": statistics.mean(score[1] for score in scores),
        "ece_uncalibrated": statistics.mean(expected_calibration_error(np.exp(s), labels) for s in repeats),
        "ece_calibrated": statistics.mean(expected_calibration_error(c, labels) for c in calibrated),
        "coverage": float(accepted.mean()),
        "accepted_accuracy": float(pooled_correct[accepted].mean()) if accepted.any() else None,
        "cross_fitted_coverage": float(held_accepted.mean()),
        "cross_fitted_accepted_accuracy": float(held_correct[held_accepted].mean()) if held_accepted.any() else None,
    }


def _export_task(task):
    """Train one ensemble member on all train + val images; return its weights and a parity probe."""
    name, seed, epochs, probe, device = task
    experiment = CONFIGS[name]
    features, flipped, labels = _WORKER[experiment.representation]
    started = time.perf_counter()
    outputs, model = fit_and_predict(experiment, features, flipped, labels, features[probe], seed, epochs,
                                     {epochs}, int(labels.max()) + 1, device, return_model=True)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return name, seed, state, outputs[epochs], time.perf_counter() - started


def run_export(args):
    """Train the deployed ensemble on train + val and write one checkpoint; the test split is never read."""
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists; choose a new path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Fail before minutes of training rather than at the final write (10 members take about 17 MiB).
    free_mib = shutil.disk_usage(args.output.parent).free / 2**20
    if free_mib < 64:
        raise OSError(f"Only {free_mib:.1f} MiB free at {args.output.parent}; at least 64 MiB is needed")
    selection = json.loads((args.cv_output / "selection.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if selection["manifest_sha256"] != digest:
        raise RuntimeError("The manifest changed after the CV run")
    members = [(spec.split(":")[0], int(spec.split(":")[1])) for spec in args.members]
    if any(name not in CONFIGS or epochs not in selection["eval_epochs"] for name, epochs in members):
        raise ValueError(f"Members must be CONFIG:EPOCHS with configs {list(CONFIGS)} "
                         f"and epochs {selection['eval_epochs']}")
    seeds = args.seeds if args.seeds is not None else selection["seeds"]
    if not set(seeds) <= set(selection["seeds"]):
        raise ValueError("Calibration needs OOF scores for every exported seed")

    # Calibrate on the OOF scores of exactly these members and seeds, before any training.
    oof = np.load(args.cv_output / "oof-log-probs.npz")
    labels, assignments = oof["labels"], oof["assignments"]
    seed_index = [selection["seeds"].index(seed) for seed in seeds]
    stacked = np.concatenate([oof[name][:, seed_index, selection["eval_epochs"].index(epochs)]
                              for name, epochs in members], axis=1)  # (repeats, members, images, classes)
    ensemble_oof = np.stack([ensemble_log_probs(stacked[repeat]) for repeat in range(len(stacked))])
    calibration = oof_calibration(ensemble_oof, labels, assignments, args.target_accuracy)
    print(json.dumps(calibration, indent=2), flush=True)

    cache_files, classes, paths = {}, None, None
    for name, _ in members:
        representation = CONFIGS[name].representation
        _, _, cache_labels, paths, classes = feature_cache(
            args.image_root, args.manifest, representation, args.cache, ("train", "val"))
        if not np.array_equal(cache_labels, labels):
            raise RuntimeError("Descriptor cache order differs from the OOF file")
        cache_files[representation] = str(args.cache / f"train-val-{representation}-{digest[:12]}.npz")
    probe = np.arange(min(args.probe_images, len(labels)))
    tasks = [(name, seed, epochs, probe, args.device) for name, epochs in members for seed in seeds]
    if args.device == "cuda" and torch.cuda.device_count() > 1:
        devices = torch.cuda.device_count()
        tasks = [(*task[:-1], f"cuda:{index % devices}") for index, task in enumerate(tasks)]
    print(f"Training {len(tasks)} members on {len(labels)} train + val images", flush=True)
    trained = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max(1, args.workers), mp_context=context, initializer=_init_worker,
                             initargs=(cache_files, args.threads)) as pool:
        for name, seed, state, probe_scores, seconds in pool.map(_export_task, tasks):
            print(f"{name} seed={seed} {seconds:.1f}s", flush=True)
            trained.append((name, seed, state, probe_scores))

    epochs_of = dict(members)
    metadata = {
        "architecture": "descriptor-ensemble", "class_names": classes, "image_size": 64,
        "temperature": calibration["temperature"], "uncertain_threshold": calibration["uncertain_threshold"],
        "calibration": calibration,
        "training": {"images": int(len(labels)), "splits": ["train", "val"], "test_split_loaded": False,
                     "manifest_sha256": digest, "cv_output": args.cv_output.as_posix(), "device": args.device,
                     "torch_version": torch.__version__, "opencv_version": __import__("cv2").__version__},
    }
    # Plain containers and tensors only, so torch.load(weights_only=True) can read the file.
    checkpoint = json.loads(json.dumps(metadata, default=lambda item: item.item()))
    checkpoint["members"] = [
        {"config": name, "representation": CONFIGS[name].representation, "seed": seed, "epochs": epochs_of[name],
         "mlp_config": json.loads(json.dumps(CONFIGS[name].model_config())), "state_dict": state}
        for name, seed, state, _ in trained]
    # Write to a temporary name first, so a failed write never leaves a truncated checkpoint.
    partial = args.output.with_name(args.output.name + ".partial")
    try:
        torch.save(checkpoint, partial)
        partial.replace(args.output)
    finally:
        partial.unlink(missing_ok=True)

    # Parity: the reloaded ensemble must reproduce the in-memory members from the image files,
    # which also checks that inference descriptors match the cached training descriptors.
    from PIL import Image
    model, restored_classes = load_classifier(args.output)
    assert restored_classes == classes
    expected = calibrate(ensemble_log_probs(np.stack([scores for *_, scores in trained])), calibration["temperature"])
    with torch.inference_mode():
        for index in probe:
            with Image.open(args.image_root / paths[index]) as image:
                restored = model(model.describe(image)).squeeze(0).numpy()
            if not np.allclose(restored, expected[index], atol=1e-5):
                raise RuntimeError(f"Reloaded ensemble differs on {paths[index]}: {restored} vs {expected[index]}")
    size_mb = args.output.stat().st_size / 2**20
    print(f"Saved {len(trained)} members to {args.output} ({size_mb:.1f} MiB); "
          f"reload parity OK on {len(probe)} images", flush=True)


def run_final(args):
    """Train the CV-selected candidate on train + val, then evaluate the test split exactly once."""
    selection = json.loads((args.cv_output / "selection.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if selection["manifest_sha256"] != digest:
        raise RuntimeError("The manifest changed after CV selection")
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists: the test split is evaluated only once per selection")
    chosen = selection["chosen"]
    experiment = CONFIGS[chosen["config"]]
    seeds = selection["seeds"][:chosen["ensemble_size"]]
    train_x, train_flip, train_y, _, classes = feature_cache(
        args.image_root, args.manifest, experiment.representation, args.cache, ("train", "val"))
    # The test descriptors are computed only now, after the decision is fixed on disk.
    args.output.mkdir(parents=True)
    write_json(args.output / "locked-selection.json", selection)
    test_x, _, test_y, test_paths, _ = feature_cache(
        args.image_root, args.manifest, experiment.representation, args.cache, ("test",))
    _init_worker({}, args.threads)
    epochs = chosen["epochs"]
    member_scores = [fit_and_predict(experiment, train_x, train_flip, train_y, test_x, seed, epochs, {epochs},
                                     len(classes), args.device)[epochs] for seed in seeds]
    log_prior = np.array(selection["log_prior"])
    scores = ensemble_log_probs(np.stack(member_scores)) if len(seeds) > 1 else member_scores[0]
    predicted = logit_adjust(scores, log_prior, selection["final_tau"]).argmax(1)
    single = member_scores[0].argmax(1)  # First seed, no adjustment: the round-three style model.
    num_classes = len(classes)
    accuracy, macro_f1 = accuracy_and_macro_f1(test_y, predicted, num_classes)
    correct = int(np.sum(predicted == test_y))
    matrix = confusion(test_y, predicted, num_classes)
    per_class = {label: {"correct": int(matrix[i, i]), "support": int(matrix[i].sum()),
                         "recall_ci95": clopper_pearson(int(matrix[i, i]), int(matrix[i].sum())),
                         "predicted": int(matrix[:, i].sum())}
                 for i, label in enumerate(classes)}
    single_accuracy, single_f1 = accuracy_and_macro_f1(test_y, single, num_classes)
    result = {
        "candidate": chosen, "final_tau": selection["final_tau"], "seeds": seeds, "test_images": int(len(test_y)),
        "accuracy": accuracy, "accuracy_ci95_clopper_pearson": clopper_pearson(correct, len(test_y)),
        "macro_f1": macro_f1,
        "bootstrap_ci95": paired_bootstrap(test_y, single, predicted, num_classes, seed=args.bootstrap_seed),
        "single_seed": {"seed": seeds[0], "accuracy": single_accuracy, "macro_f1": single_f1},
        "mcnemar_single_vs_final": mcnemar_exact(single == test_y, predicted == test_y),
        "per_class": per_class, "confusion_matrix": matrix.tolist(),
        "note": "bootstrap 'a' is the single seed and the deltas are final minus single seed",
    }
    write_json(args.output / "test-result.json", result)
    write_csv(args.output / "test-predictions.csv", ("path", "actual", "predicted", "single_seed_predicted"),
              [(path, classes[y], classes[p], classes[s]) for path, y, p, s in zip(test_paths, test_y, predicted, single, strict=True)])
    low, high = result["accuracy_ci95_clopper_pearson"]
    print(f"Test accuracy {accuracy:.4f} [{low:.4f}, {high:.4f}] | macro-F1 {macro_f1:.4f} | "
          f"single seed {single_accuracy:.4f} / {single_f1:.4f} | McNemar p={result['mcnemar_single_vs_final']['p_value']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("cv", "final", "export"))
    parser.add_argument("--image-root", type=Path, default=Path("data/trashnet"))
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/trashnet.csv"))
    parser.add_argument("--cache", type=Path, default=Path("outputs/descriptor-cache"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cv-output", type=Path, help="final and export modes: directory written by the cv mode")
    parser.add_argument("--members", nargs="+", default=["extra_lab:300", "hsv_only:300"],
                        help="export mode: CONFIG:EPOCHS for each ensemble view")
    parser.add_argument("--target-accuracy", type=float, default=0.90,
                        help="export mode: accuracy required of predictions not marked uncertain")
    parser.add_argument("--probe-images", type=int, default=16, help="export mode: images used for the reload check")
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--seeds", nargs="+", type=int, help="cv mode default 42-46; export mode default: the CV seeds")
    parser.add_argument("--eval-epochs", nargs="+", type=int, default=[100, 200, 300, 400, 600])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=1, help="parallel processes for cv mode; with --device cuda they share all GPUs")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch threads per process")
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "cv":
        args.seeds = args.seeds or [42, 43, 44, 45, 46]
        run_cv(args)
    elif args.mode == "export":
        if args.cv_output is None:
            parser.error("export mode needs --cv-output")
        run_export(args)
    else:
        if args.cv_output is None:
            parser.error("final mode needs --cv-output")
        run_final(args)


if __name__ == "__main__":
    main()
