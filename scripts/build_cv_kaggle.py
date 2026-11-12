"""Build the self-contained Kaggle notebook for repeated CV, seed ensembles and the one-time test."""

import argparse
import json
from pathlib import Path

from build_mlp_kaggle import ROOT, code, snapshot

SLUG = "wastelens-mlp-cv-20261111"
NOTEBOOK = "mlp_cv_kaggle.ipynb"


def build(staging: Path, workers: int):
    encoded, sha256, revision = snapshot()
    cells = [{"cell_type": "markdown", "metadata": {}, "source":
              "# Descriptor MLP: repeated CV, seed ensembles and a one-time test\n\n"
              "Pools train + val (2,144 images) for 5-fold CV repeated 3 times, with 5 seeds per fold, "
              "for the round-three HOG + HSV, HOG + HSV + Lab and HSV-only MLPs. Out-of-fold scores at "
              "100–600 epochs select the configuration, epoch budget, ensemble size and post-hoc logit "
              "adjustment by the one-SE rule. The selection is written to disk before the test descriptors "
              "are computed; the test split is then evaluated once. No pretrained weights. "
              "The embedded source snapshot defines this run.\n"}]
    cells.append(code(f'''
        import base64, hashlib, io, json, os, subprocess, sys, time, zipfile
        from pathlib import Path
        import torch

        SOURCE_SHA256 = "{sha256}"
        SOURCE_BASE_REVISION = "{revision}"
        payload = base64.b64decode("{encoded}")
        assert hashlib.sha256(payload).hexdigest() == SOURCE_SHA256
        root = Path("/tmp/wastelens-cv-snapshot")
        root.mkdir(exist_ok=False)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.extractall(root)
        run_root = Path("/kaggle/working/wastelens-mlp-cv")
        run_root.mkdir(exist_ok=False)
        (run_root / "source-snapshot.zip").write_bytes(payload)
        (run_root / "source-snapshot.json").write_text(json.dumps({{
            "sha256": SOURCE_SHA256, "base_revision": SOURCE_BASE_REVISION}}, indent=2), encoding="utf-8")
        assert torch.cuda.is_available(), "Requires a Kaggle GPU (T4 x2)"
        devices = [(torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
                   for i in range(torch.cuda.device_count())]
        print(torch.__version__, devices, "CPUs:", os.cpu_count(), flush=True)
        torch.ones(2, device="cuda").sum().item()
        subprocess.run([sys.executable, str(root / "scripts/prepare_trashnet.py")], cwd=root, check=True)
        env = dict(os.environ, PYTHONPATH=str(root / "src"), CUBLAS_WORKSPACE_CONFIG=":4096:8")
    '''))
    cells.append(code(f'''
        # CV only: the test split is not read by this command.
        log_path = run_root / "cv.log"
        with log_path.open("w", encoding="utf-8") as log:
            worker = subprocess.Popen(
                [sys.executable, "-u", "-m", "wastelens.cv", "cv", "--device", "cuda", "--workers", "{workers}",
                 "--threads", "1", "--cache", str(run_root / "cache"), "--output", str(run_root / "cv")],
                cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            last = None
            while worker.poll() is None:
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                if lines and lines[-1] != last:
                    print(lines[-1], flush=True)
                    last = lines[-1]
                time.sleep(30)
        print(log_path.read_text(encoding="utf-8", errors="replace")[-4000:])
        if worker.returncode:
            raise RuntimeError(f"CV failed with exit code {{worker.returncode}}")
    '''))
    cells.append(code('''
        # The selection above is final; evaluate the test split exactly once.
        subprocess.run([sys.executable, "-u", "-m", "wastelens.cv", "final", "--device", "cuda",
                        "--cache", str(run_root / "cache"), "--cv-output", str(run_root / "cv"),
                        "--output", str(run_root / "final")], cwd=root, env=env, check=True)
        result = json.loads((run_root / "final" / "test-result.json").read_text(encoding="utf-8"))
        print(json.dumps({key: result[key] for key in ("candidate", "accuracy", "accuracy_ci95_clopper_pearson",
                          "macro_f1", "bootstrap_ci95", "single_seed", "mcnemar_single_vs_final")}, indent=2))
        for label, row in result["per_class"].items():
            print(f"{label:10s} {row['correct']}/{row['support']} recall CI {row['recall_ci95']}")
        # Descriptor caches are reproducible from the snapshot; keep the artifact small.
        subprocess.run(["rm", "-rf", str(run_root / "cache")], check=True)
    '''))
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    for index, cell in enumerate(cells):
        cell["id"] = f"mlp-cv-{index}"
    destination = ROOT / "notebooks" / NOTEBOOK
    destination.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8", newline="\n")
    staging.mkdir(parents=True, exist_ok=False)
    (staging / NOTEBOOK).write_bytes(destination.read_bytes())
    metadata = {
        "id": "luquang231/" + SLUG, "title": SLUG,
        "code_file": NOTEBOOK, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "kernel_sources": [], "competition_sources": [],
    }
    (staging / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"notebook": str(destination), "staging": str(staging), "snapshot_sha256": sha256}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4, help="parallel training processes on Kaggle")
    args = parser.parse_args()
    build(args.staging, args.workers)
