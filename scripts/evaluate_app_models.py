"""Re-evaluate all three deployed checkpoints through the app's inference path.

Uses the fixed test split without augmentation or threshold tuning. Writes per-image
probabilities and checkpoint hashes to an ignored output JSON for reproducibility.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from wastelens.classifier import WasteClassifier
from wastelens.data import read_splits
from wastelens.model import predict_probabilities
from wastelens.profiles import MODEL_PROFILES
from wastelens.stats import accuracy_and_macro_f1, confusion

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/app-model-comparison.json")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    manifest = ROOT / "data/splits/trashnet.csv"
    splits, classes = read_splits(manifest)
    labels = np.array([label for _, label in splits["test"]])
    report = {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "test_images": len(labels), "classes": classes, "models": {}}
    for key, profile in MODEL_PROFILES.items():
        classifier = WasteClassifier(profile.checkpoint)
        if classifier.classes != classes:
            raise ValueError(f"Class order mismatch: {key}")
        predictions, accepted, records = [], [], []
        for index, (relative, target) in enumerate(splits["test"]):
            with Image.open(ROOT / "data/trashnet" / relative) as source:
                image = source.convert("RGB")
            bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            result = classifier.classify(bgr)
            probabilities = np.array([result.probabilities[label] for label in classes])
            if not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(), 1, atol=1e-6):
                raise AssertionError(f"Invalid probabilities: {key} {relative}")
            if index == 0:
                # BGR frame, PIL image and the pre-existing API must agree exactly on class order/color.
                np.testing.assert_allclose(probabilities, predict_probabilities(classifier.model, image).numpy(),
                                           rtol=1e-5, atol=1e-6)
                np.testing.assert_allclose(probabilities, list(classifier.classify(image).probabilities.values()),
                                           rtol=1e-5, atol=1e-6)
            predictions.append(int(probabilities.argmax()))
            accepted.append(not result.uncertain)
            records.append({"path": relative, "target": classes[target], "prediction": result.top_class,
                            "label": result.label, "probabilities": result.probabilities})
        predicted, keep = np.array(predictions), np.array(accepted, dtype=bool)
        accuracy, macro_f1 = accuracy_and_macro_f1(labels, predicted, len(classes))
        with profile.checkpoint.open("rb") as file:
            sha256 = hashlib.file_digest(file, "sha256").hexdigest()
        report["models"][key] = {
            "checkpoint": profile.filename, "sha256": sha256, "training": profile.training,
            "parameters": sum(parameter.numel() for parameter in classifier.model.parameters()),
            "correct": int((predicted == labels).sum()), "accuracy": float(accuracy), "macro_f1": float(macro_f1),
            "confusion_matrix": confusion(labels, predicted, len(classes)).tolist(),
            "calibrated": classifier.calibrated, "uncertain_threshold": classifier.threshold,
            "accepted": int(keep.sum()), "coverage": float(keep.mean()),
            "accepted_accuracy": float((predicted[keep] == labels[keep]).mean()) if keep.any() else None,
            "predictions": records,
        }
        print(f"{key}: {int((predicted == labels).sum())}/{len(labels)}; accuracy={accuracy:.4%}; "
              f"macro-F1={macro_f1:.6f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
