"""Evaluate the three deployed checkpoints on an external image set that was never used for training.

The default mapping fits RealWaste (Single et al., 2023, UCI dataset 908, CC BY 4.0): photos of real waste at a
landfill. Its folders that have a TrashNet counterpart are mapped to the six classes; the other three
(Food Organics, Textile Trash, Vegetation) are skipped. Images go through the app's inference path unchanged
(no Module 1, no detector), exactly as for the TrashNet test split.

    python scripts/evaluate_external.py outputs/external/realwaste/realwaste-main/RealWaste
"""

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from wastelens.classifier import WasteClassifier
from wastelens.profiles import MODEL_PROFILES
from wastelens.stats import accuracy_and_macro_f1, clopper_pearson, confusion, holm, mcnemar_exact

ROOT = Path(__file__).resolve().parents[1]
REALWASTE = {"Cardboard": "cardboard", "Glass": "glass", "Metal": "metal", "Paper": "paper",
             "Plastic": "plastic", "Miscellaneous Trash": "trash"}


def collect(folder: Path, mapping: dict[str, str], classes: list[str]) -> list[tuple[Path, int]]:
    """(image path, class index) for every image in a mapped sub-folder, in a stable order."""
    items = []
    for source, target in mapping.items():
        files = sorted(p for p in (folder / source).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not files:
            raise FileNotFoundError(folder / source)
        items += [(path, classes.index(target)) for path in files]
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="Folder with one sub-folder per source class.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/external-evaluation.json")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    classes = WasteClassifier(MODEL_PROFILES["baseline"].checkpoint, warm_up=0).classes
    items = collect(args.folder, REALWASTE, classes)
    labels = np.array([target for _, target in items])
    report = {"folder": str(args.folder), "mapping": REALWASTE, "images": len(items), "classes": classes,
              "per_class": {c: int((labels == i).sum()) for i, c in enumerate(classes)}, "models": {}}
    correct = {}
    for key, profile in MODEL_PROFILES.items():
        classifier = WasteClassifier(profile.checkpoint)
        if classifier.classes != classes:
            raise ValueError(f"Class order mismatch: {key}")
        predicted, accepted, confidence, records = [], [], [], []
        for path, target in items:
            with Image.open(path) as source:
                image = source.convert("RGB")
            result = classifier.classify(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
            predicted.append(classes.index(result.top_class))
            accepted.append(not result.uncertain)
            confidence.append(result.confidence)
            records.append({"path": str(path.relative_to(args.folder)).replace("\\", "/"), "target": classes[target],
                            "prediction": result.top_class, "label": result.label,
                            "confidence": round(result.confidence, 6)})
        predicted, keep, confidence = np.array(predicted), np.array(accepted, bool), np.array(confidence)
        hits = predicted == labels
        correct[key] = hits
        accuracy, macro_f1 = accuracy_and_macro_f1(labels, predicted, len(classes))
        low, high = clopper_pearson(int(hits.sum()), len(hits))
        matrix = confusion(labels, predicted, len(classes))
        with profile.checkpoint.open("rb") as file:
            sha256 = hashlib.file_digest(file, "sha256").hexdigest()
        report["models"][key] = {
            "checkpoint": profile.filename, "sha256": sha256,
            "correct": int(hits.sum()), "accuracy": float(accuracy), "ci95": [low, high], "macro_f1": float(macro_f1),
            "recall": {c: float(matrix[i, i] / matrix[i].sum()) for i, c in enumerate(classes)},
            "predicted_share": {c: float((predicted == i).mean()) for i, c in enumerate(classes)},
            "confusion_matrix": matrix.tolist(),
            "mean_confidence": float(confidence.mean()),
            "mean_confidence_wrong": float(confidence[~hits].mean()) if (~hits).any() else None,
            "calibrated": classifier.calibrated, "uncertain_threshold": classifier.threshold,
            "coverage": float(keep.mean()),
            "accepted_accuracy": float(hits[keep].mean()) if keep.any() else None,
            "predictions": records,
        }
        print(f"{key}: {int(hits.sum())}/{len(hits)} accuracy={accuracy:.2%} [{low:.2%}, {high:.2%}] "
              f"macro-F1={macro_f1:.4f} coverage={keep.mean():.2%}", flush=True)

    pairs = list(combinations(correct, 2))
    tests = [mcnemar_exact(correct[a], correct[b]) for a, b in pairs]
    adjusted = holm([test["p_value"] for test in tests])
    report["mcnemar"] = [{"a": a, "b": b, **test, "p_holm": p} for (a, b), test, p in zip(pairs, tests, adjusted)]
    for row in report["mcnemar"]:
        print(f"McNemar {row['a']} vs {row['b']}: p_holm={row['p_holm']:.3g}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
