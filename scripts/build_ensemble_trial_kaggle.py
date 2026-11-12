"""Package the paired 300/600-epoch ensemble evaluation for Kaggle."""

import argparse
import hashlib
import json
from pathlib import Path

from build_mlp_kaggle import code, snapshot


def build(staging: Path, cv_output: Path, slug: str):
    encoded, digest, revision = snapshot()
    hashes = {name: hashlib.sha256((cv_output / name).read_bytes()).hexdigest()
              for name in ("selection.json", "oof-log-probs.npz")}
    cells = [{"cell_type": "markdown", "metadata": {}, "source":
              "# WasteLens descriptor ensemble: 300 vs 600 epochs\n"
              "Five seeds per descriptor, training on train + val. Calibration uses the previous OOF scores "
              "at each epoch budget. Both checkpoints are exported before evaluating the same 377 test images.\n"}]
    cells.append(code(f'''
        import base64, hashlib, io, json, os, subprocess, sys, time, zipfile
        from pathlib import Path
        import torch

        root = Path("/tmp/wastelens-ensemble-trial")
        root.mkdir()
        output = Path("/kaggle/working/ensemble-trial")
        output.mkdir()
        payload = base64.b64decode("{encoded}")
        assert hashlib.sha256(payload).hexdigest() == "{digest}"
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.extractall(root)
        (output / "source-snapshot.zip").write_bytes(payload)
        (output / "source-snapshot.json").write_text(json.dumps({{
            "sha256": "{digest}", "base_revision": "{revision}", "cv_input_sha256": {hashes!r}
        }}, indent=2))
        hashes = {hashes!r}
        for path in Path("/kaggle/input").rglob("selection.json"):
            if all((path.parent / name).is_file() and
                   hashlib.sha256((path.parent / name).read_bytes()).hexdigest() == expected
                   for name, expected in hashes.items()):
                cv_output = path.parent
                break
        else:
            raise FileNotFoundError("Previous Kaggle CV output with matching checksums is missing")
        assert torch.cuda.device_count() >= 2, "This trial expects two GPUs"
        print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
        subprocess.run([sys.executable, str(root / "scripts/prepare_trashnet.py")], cwd=root, check=True)
        env = dict(os.environ, PYTHONPATH=str(root / "src"), CUBLAS_WORKSPACE_CONFIG=":4096:8")
        cache = Path("/tmp/wastelens-ensemble-cache")
    '''))
    cells.append(code('''
        for epochs in (300, 600):
            log_path = output / f"train-{epochs}.log"
            with log_path.open("w", encoding="utf-8") as log:
                worker = subprocess.Popen(
                    [sys.executable, "-u", "-m", "wastelens.cv", "export",
                     "--cv-output", str(cv_output), "--cache", str(cache),
                     "--members", f"extra_lab:{epochs}", f"hsv_only:{epochs}",
                     "--device", "cuda", "--workers", "4",
                     "--output", str(output / f"ensemble-{epochs}.pt")],
                    cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
                previous = None
                while worker.poll() is None:
                    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    if lines and lines[-1] != previous:
                        print(f"{epochs} epochs: {lines[-1]}", flush=True)
                        previous = lines[-1]
                    time.sleep(10)
            print(log_path.read_text(encoding="utf-8", errors="replace")[-3500:], flush=True)
            if worker.returncode:
                raise RuntimeError(f"Export at {epochs} epochs exited with code {worker.returncode}")
    '''))
    cells.append(code('''
        import csv
        import numpy as np
        sys.path.insert(0, str(root / "src"))
        from wastelens.cv import feature_cache
        from wastelens.model import load_classifier
        from wastelens.stats import accuracy_and_macro_f1, clopper_pearson, confusion, mcnemar_exact

        torch.set_num_threads(2)
        result, predictions, scores = {}, {}, {}
        descriptors = {}
        for representation in ("hog_hsv_lab", "hsv_only"):
            features, _, labels, paths, classes = feature_cache(
                root / "data/trashnet", root / "data/splits/trashnet.csv", representation, cache, ("test",))
            if descriptors:
                assert np.array_equal(labels, reference_labels) and paths == reference_paths
            reference_labels, reference_paths = labels, paths
            descriptors[representation] = torch.from_numpy(features)
        assert len(labels) == 377
        for epochs in (300, 600):
            path = output / f"ensemble-{epochs}.pt"
            model, model_classes = load_classifier(path)
            assert model_classes == classes and len(model.members) == 10
            with torch.inference_mode():
                probabilities = model(descriptors).numpy()
            assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(1), 1)
            predicted = probabilities.argmax(1)
            confidence = probabilities.max(1)
            accepted = confidence.astype(np.float64) >= model.uncertain_threshold
            correct = predicted == labels
            accuracy, macro_f1 = accuracy_and_macro_f1(labels, predicted, len(classes))
            matrix = confusion(labels, predicted, len(classes))
            result[str(epochs)] = {
                "accuracy": float(accuracy), "macro_f1": float(macro_f1),
                "correct": int(correct.sum()), "test_images": len(labels),
                "accuracy_ci95": clopper_pearson(int(correct.sum()), len(labels)),
                "accepted": int(accepted.sum()), "coverage": float(accepted.mean()),
                "accepted_accuracy": float(correct[accepted].mean()) if accepted.any() else None,
                "threshold": model.uncertain_threshold, "temperature": model.temperature,
                "classes": classes, "confusion_matrix": matrix.tolist(),
                "calibration": torch.load(path, map_location="cpu", weights_only=True)["calibration"],
            }
            predictions[epochs] = predicted
            scores[epochs] = (confidence, accepted)
        result["paired_mcnemar"] = mcnemar_exact(predictions[300] == labels, predictions[600] == labels)
        (output / "test-result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\\n")
        with (output / "test-predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["path", "actual", "pred_300", "confidence_300", "accepted_300",
                             "pred_600", "confidence_600", "accepted_600"])
            for i, path in enumerate(paths):
                writer.writerow([path, classes[labels[i]], classes[predictions[300][i]],
                                 scores[300][0][i], scores[300][1][i], classes[predictions[600][i]],
                                 scores[600][0][i], scores[600][1][i]])
        print(json.dumps(result, indent=2), flush=True)
    '''))
    for index, cell in enumerate(cells):
        cell["id"] = f"ensemble-trial-{index}"
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
                "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
    staging.mkdir(parents=True, exist_ok=False)
    filename = "ensemble_trial.ipynb"
    (staging / filename).write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    metadata = {"id": f"luquang231/{slug}", "title": slug, "code_file": filename,
                "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
                "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [],
                "kernel_sources": ["luquang231/wastelens-mlp-cv-20261111"]}
    (staging / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Staged {slug} with source SHA-256 {digest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--cv-output", type=Path, required=True)
    parser.add_argument("--slug", required=True)
    args = parser.parse_args()
    build(args.staging, args.cv_output, args.slug)
