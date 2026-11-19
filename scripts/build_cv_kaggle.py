"""Build self-contained Kaggle notebooks for the repeated-CV runs.

Variant "selection" reproduces the 11 Nov run: CV over three descriptor MLPs, then the one-time test.
Variant "single-vs-ensemble" runs CV only (never the test split) for two single networks with the
parameter budget of the 5-seed ensemble, on the same folds and seeds, for scripts/compare_cv_oof.py.
"""

import argparse
import json
from pathlib import Path

from build_mlp_kaggle import ROOT, code, snapshot

VARIANTS = {
    "selection": {"slug": "wastelens-mlp-cv-20261111", "notebook": "mlp_cv_kaggle.ipynb",
                  "run": "wastelens-mlp-cv", "configs": None, "final": True},
    "single-vs-ensemble": {"slug": "wastelens-mlp-single-vs-ensemble-20261113",
                           "notebook": "mlp_single_vs_ensemble_kaggle.ipynb", "run": "wastelens-single-vs-ensemble",
                           "configs": ["extra_lab_wide", "extra_lab_deep"], "final": False},
}
SINGLE_VS_ENSEMBLE_TEXT = (
    "# One large MLP versus the 5-seed ensemble\n\n"
    "Repeated stratified 5-fold CV (3 repeats, seeds 42-46, epochs 100-600) on train + val for two single "
    "HOG + HSV + Lab MLPs with the parameter budget of five 256-64-32 MLPs (3,646,590): wide 1200-256-64 "
    "(3,654,934) and deep 1056-512-256-128-64 (3,646,374). Folds and seeds match the 11 Nov CV run, so its "
    "out-of-fold scores can be compared pairwise. The test split is not read. No pretrained weights.\n")


def build(staging: Path, workers: int, variant: str = "selection"):
    settings = VARIANTS[variant]
    slug, notebook_name, run_name = settings["slug"], settings["notebook"], settings["run"]
    configs = ["--configs", *settings["configs"]] if settings["configs"] else []
    encoded, sha256, revision = snapshot()
    if variant != "selection":
        cells = [{"cell_type": "markdown", "metadata": {}, "source": SINGLE_VS_ENSEMBLE_TEXT}]
    else:
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
        run_root = Path("/kaggle/working/{run_name}")
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
                 "--threads", "1", "--cache", str(run_root / "cache"), "--output", str(run_root / "cv"),
                 *{configs!r}],
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
    cleanup = code('''
        # Descriptor caches are reproducible from the snapshot; keep the artifact small.
        subprocess.run(["rm", "-rf", str(run_root / "cache")], check=True)
    ''')
    final = code('''
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
    ''')
    cells.append(final if settings["final"] else cleanup)
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    for index, cell in enumerate(cells):
        cell["id"] = f"mlp-cv-{index}"
    destination = ROOT / "notebooks" / notebook_name
    destination.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8", newline="\n")
    staging.mkdir(parents=True, exist_ok=False)
    (staging / notebook_name).write_bytes(destination.read_bytes())
    metadata = {
        "id": "luquang231/" + slug, "title": slug,
        "code_file": notebook_name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "kernel_sources": [], "competition_sources": [],
    }
    (staging / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"notebook": str(destination), "staging": str(staging), "snapshot_sha256": sha256}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4, help="parallel training processes on Kaggle")
    parser.add_argument("--variant", choices=VARIANTS, default="selection")
    args = parser.parse_args()
    build(args.staging, args.workers, args.variant)
