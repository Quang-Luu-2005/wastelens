# WasteLens · Lab03

WasteLens is a six-class waste classification project for Digital Image & Video Processing Lab03. The classifier extends the Horse or Human MLP from Lab02 and trains on a fixed TrashNet split.

## Run locally

Use Python 3.12. These PowerShell commands install the CPU builds and prepare the dataset from a fresh clone.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe check_env.py
.\.venv\Scripts\python.exe scripts/prepare_trashnet.py
.\.venv\Scripts\python.exe -m wastelens.train --device cpu
```

The CPU check exercises OpenCV BGR-to-RGB conversion and a PyTorch forward pass with six finite outputs. The dataset script caches and verifies the source ZIP, extracts eligible images to `data/trashnet/`, and regenerates the versioned split files. For a verified local copy of the ZIP, run the same script with `--archive PATH`.

The training command writes to `outputs/lab02-mlp/`. For a quick end-to-end run, pass `--epochs 1 --output outputs/smoke`. Give each repeat run a new `--output` directory. With a CUDA build of PyTorch, use `--device cuda` to require GPU training.

## Train on Kaggle GPU

Import [`notebooks/train_kaggle.ipynb`](notebooks/train_kaggle.ipynb), enable **Internet** and select **GPU T4 ×2** or **L4** under Accelerator. Run all cells using Kaggle's PyTorch environment. The notebook uses one GPU with automatic mixed precision and stops with an explicit error if CUDA cannot run.

The notebook downloads the versioned dataset, previews the six classes and trains for 30 epochs by default. Download the run directory under `/kaggle/working/wastelens-results/` after execution.

| Output | Use |
| --- | --- |
| `best.pt`, `meta.json` | Checkpoint selected by validation macro-F1, class order and preprocessing for app inference |
| `metrics.json`, `history.csv` | Test accuracy, macro-F1, per-class precision/recall/F1, training configuration and validation curves |
| `confusion-matrix.csv`, `test-predictions.csv` | Confusion counts and per-image predictions on the held-out test split |
| Notebook PNGs | Learning curves, confusion matrix and examples of test errors |

[`wastelens.model`](src/wastelens/model.py) exposes `load_classifier()` and `predict_probabilities()` for use with a saved checkpoint.

## Lab02 to Lab03

Both Lab02 notebooks, `lab02_horse_or_human.ipynb` and `lab02_horse_or_human_local_gpu.ipynb`, flatten a `3 × 64 × 64` image and use hidden widths of `128` and `32`. This implementation keeps that topology, ReLU and the selected Xavier-normal initialization. The dense head grows from `32 → 1` (33 parameters) to `32 → 6` (198 parameters). The total grows from 1,577,153 to 1,577,318 parameters.

Lab02 computes BCE and a gradient on the binary logit using `sigmoid(z) − y`. Lab03 computes cross-entropy from six logits and integer class IDs. Its logit gradient is `softmax(z) − one_hot(y)` before batch averaging. The network uses `nn.Linear`, autograd and Adam to propagate gradients and update weights. Inference applies softmax to obtain the six probabilities in manifest order.

The repository separates the reusable model and data loading in `src/wastelens/`, the Kaggle notebook in `notebooks/`, dataset preparation in `scripts/`, and experiment outputs in `outputs/`.

## TrashNet split

The [upstream TrashNet release](https://github.com/garythung/trashnet/tree/6fa2b878c6c1b4304b91109070ce0edf9279bb31) by Gary Thung and Mindy Yang contains 2,527 images at 512 × 384 pixels. The photographs use a white background. The small `trash` class and the controlled background both matter when interpreting performance on webcam footage.

The preparation script checks the archive SHA-256 and compares hashes of decoded RGB pixels. It excludes all six files in the three identical-image pairs because each pair has conflicting labels. The remaining **2,521 images** are stratified per class with seed **42**, targeting **70% train, 15% validation, 15% test**. The manifest is the source of truth for subsequent experiments.

| Class | Source | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: |
| cardboard | 403 | 282 | 61 | 60 |
| glass | 501 | 348 | 75 | 75 |
| metal | 410 | 286 | 62 | 61 |
| paper | 594 | 416 | 89 | 89 |
| plastic | 482 | 336 | 72 | 72 |
| trash | 137 | 96 | 21 | 20 |
| **Total** | **2,527** | **1,764** | **380** | **377** |

- [`data/splits/trashnet.csv`](data/splits/trashnet.csv) contains image paths relative to `data/trashnet/`, labels, class IDs and split names.
- [`data/splits/trashnet-summary.json`](data/splits/trashnet-summary.json) records source provenance, counts and the excluded pairs.
- [`docs/lab03-rubric.md`](docs/lab03-rubric.md) maps the Lab03 requirements to the planned evidence.

The ZIP and extracted image files stay in the local data cache. See the [upstream license](https://github.com/garythung/trashnet/blob/6fa2b878c6c1b4304b91109070ce0edf9279bb31/LICENSE) and [original project report](https://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf) for attribution.
