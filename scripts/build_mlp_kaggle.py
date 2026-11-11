"""Build a self-contained Kaggle notebook containing the exact local source snapshot."""

import argparse
import base64
import hashlib
import io
import json
import subprocess
import textwrap
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": textwrap.dedent(source).strip() + "\n"}


def build(staging: Path):
    files = sorted((ROOT / "src/wastelens").glob("*.py")) + [
        ROOT / "scripts/prepare_trashnet.py", ROOT / "data/splits/trashnet.csv",
        ROOT / "data/splits/trashnet-summary.json",
    ]
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            # Stable ZIP timestamps make unchanged snapshots byte-for-byte reproducible.
            entry = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), (2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, path.read_bytes())
    snapshot = stream.getvalue()
    sha256 = hashlib.sha256(snapshot).hexdigest()
    encoded = base64.b64encode(snapshot).decode("ascii")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    cells = [{"cell_type": "markdown", "metadata": {}, "source":
              "# Lab02 MLP: controlled from-scratch ablations\n\n"
              "Nine configurations, paired seeds 42/43, 600 epochs, RGB 64x64. "
              "No pretrained weights. Select the configuration by mean validation macro-F1 before test. "
              "The test split has been inspected in prior project experiments. "
              "The embedded source snapshot, rather than a mutable Git branch, defines this run.\n"}]
    cells.append(code(f'''
        import base64, hashlib, io, json, os, subprocess, sys, time, zipfile
        from pathlib import Path
        import torch

        SOURCE_SHA256 = "{sha256}"
        SOURCE_BASE_REVISION = "{revision}"
        payload = base64.b64decode("{encoded}")
        assert hashlib.sha256(payload).hexdigest() == SOURCE_SHA256
        root = Path("/tmp/wastelens-mlp-snapshot")
        root.mkdir(exist_ok=False)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.extractall(root)
        run_root = Path("/kaggle/working/wastelens-mlp-ablation")
        run_root.mkdir(exist_ok=False)
        (run_root / "source-snapshot.zip").write_bytes(payload)
        (run_root / "source-snapshot.json").write_text(json.dumps({{
            "sha256": SOURCE_SHA256, "base_revision": SOURCE_BASE_REVISION,
            "note": "Embedded local source including the new MLP experiments; base revision alone does not define this snapshot."
        }}, indent=2), encoding="utf-8")
        assert torch.cuda.device_count() >= 2, "Requires Kaggle T4 x2"
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        slots = visible.split(",") if visible else [str(i) for i in range(torch.cuda.device_count())]
        print(torch.__version__, [torch.cuda.get_device_name(i) for i in range(2)], flush=True)
        subprocess.run([sys.executable, str(root / "scripts/prepare_trashnet.py")], cwd=root, check=True)
    '''))
    cells.append(code('''
        from contextlib import ExitStack
        workers = []
        with ExitStack() as stack:
            for seed, slot in zip((42, 43), slots[:2], strict=True):
                log_path = run_root / f"seed-{seed}.log"
                log = stack.enter_context(log_path.open("w", encoding="utf-8"))
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=slot, PYTHONPATH=str(root / "src"),
                           OMP_NUM_THREADS="2", CUBLAS_WORKSPACE_CONFIG=":4096:8")
                args = [sys.executable, "-u", "-m", "wastelens.mlp_experiments", "train",
                        "--output", str(run_root / f"seed-{seed}"), "--seed", str(seed),
                        "--epochs", "600", "--batch-size", "64", "--device", "cuda"]
                worker = subprocess.Popen(args, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
                workers.append((seed, worker, log_path))
                print(f"Started seed {seed} on slot {slot}", flush=True)
            previous = {}
            while any(worker.poll() is None for _, worker, _ in workers):
                for seed, _, log_path in workers:
                    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    if lines and lines[-1] != previous.get(seed):
                        print(f"Seed {seed}: {lines[-1]}", flush=True)
                        previous[seed] = lines[-1]
                time.sleep(10)
            codes = [worker.wait() for _, worker, _ in workers]
        for (seed, _, log_path), exit_code in zip(workers, codes, strict=True):
            print(log_path.read_text(encoding="utf-8", errors="replace")[-1800:])
            if exit_code:
                raise RuntimeError(f"Seed {seed} failed with exit code {exit_code}")
        env = dict(os.environ, PYTHONPATH=str(root / "src"), CUBLAS_WORKSPACE_CONFIG=":4096:8")
        subprocess.run([sys.executable, "-u", "-m", "wastelens.mlp_experiments", "summarize",
                        "--output", str(run_root), "--seeds", "42", "43", "--device", "cuda"],
                       cwd=root, env=env, check=True)
    '''))
    cells.append(code('''
        import csv
        import matplotlib.pyplot as plt
        summary = json.loads((run_root / "run-summary.json").read_text(encoding="utf-8"))
        rows = summary["validation"]
        fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
        ax.bar([row["name"] for row in rows], [100 * row["val_accuracy_mean"] for row in rows],
               yerr=[100 * row["val_accuracy_std"] for row in rows], capsize=4, color="#267b69")
        ax.set(ylabel="Validation accuracy (%)", title="From-scratch MLP: mean and sample SD over two seeds")
        ax.tick_params(axis="x", rotation=30)
        fig.savefig(run_root / "validation-accuracy.png", dpi=170)
        plt.show()
        print("Winner selected on validation:", summary["winner"])
        for name, result in summary["test"].items():
            print(name, "test accuracy", result["accuracy_mean"], "macro-F1", result["macro_f1_mean"])
    '''))
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    for index, cell in enumerate(cells):
        cell["id"] = f"mlp-ablation-{index}"
    destination = ROOT / "notebooks/mlp_ablation_kaggle.ipynb"
    destination.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    staging.mkdir(parents=True, exist_ok=False)
    (staging / destination.name).write_bytes(destination.read_bytes())
    metadata = {
        "id": "luquang231/wastelens-mlp-ablation-20261111", "title": "wastelens-mlp-ablation-20261111",
        "code_file": destination.name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "kernel_sources": [], "competition_sources": [],
    }
    (staging / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"notebook": str(destination), "staging": str(staging), "snapshot_sha256": sha256}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, required=True)
    build(parser.parse_args().staging)
