# WasteLens · Lab03

WasteLens is a six-class waste classification project for Digital Image & Video Processing Lab03. The baseline classifier extends the Horse or Human MLP from Lab02; ImageNet-pretrained CNNs fine-tuned on the same fixed TrashNet split serve the application.

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

Two optional input experiments are available through the training command.

```powershell
# Rotate training images within 10 degrees and translate by at most 6.25%.
.\.venv\Scripts\python.exe -m wastelens.train --device cuda --epochs 600 --seed 43 --affine-augmentation --output outputs/affine-600
# Use each original TrashNet image at 512 x 384 pixels.
.\.venv\Scripts\python.exe -m wastelens.train --device cuda --epochs 60 --seed 43 --native-resolution --output outputs/native-60
```

Native-resolution training records the original `(width, height)` in the checkpoint and requires every input to have that size. It uses 75,501,926 model parameters, versus 1,577,318 with the 64 × 64 input. Future image uploads or webcam crops need a size-handling step to satisfy that MLP input contract. The affine option applies only to training, fills exposed borders with white and keeps validation and test preprocessing fixed.

The Kaggle experiments below use seed 43, batch size 64, the same versioned split and validation macro-F1 for checkpoint selection. Original-size trials tested learning rates 0.001, 0.0001 and 0.00002 for 60 epochs. The table reports the original-size trial with the highest validation macro-F1.

| Input and training | Epochs | Learning rate | Best validation macro-F1 | Test accuracy | Test macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 × 64 with horizontal flips | 60 | 0.001 | 0.4505 | 52.79% | 0.4963 |
| Original 512 × 384 with horizontal flips | 60 | 0.00002 | 0.4056 | 49.87% | 0.4150 |
| 64 × 64 with horizontal flips | 600 | 0.001 | 0.5801 | 58.62% | 0.5727 |
| 64 × 64 with flips, rotation and translation | 600 | 0.001 | 0.5740 | 59.95% | 0.5791 |

The affine checkpoint at epoch 460 scored 70.18% on unaugmented training images and 58.95% on validation, a gap of 11.23 percentage points. The baseline checkpoint scored 86.17% and 59.47%, a gap of 26.70 points. The smaller gap comes mainly from lower training accuracy. The baseline still leads on the validation selection metric, while the affine trial has a slightly higher test score. These single-seed experiments are exploratory comparisons.

The [input experiments](https://www.kaggle.com/code/luquang231/wastelens-lab03-input-ablation-20261110) and [original-size learning-rate experiments](https://www.kaggle.com/code/luquang231/wastelens-lab03-native-lr-20261110) contain their run histories and outputs. The Kaggle notebooks are private to the connected account.

## Pretrained backbones

`--architecture` selects `lab02-mlp` (default), `mobilenet_v3_large`, `efficientnet_b0` or `resnet50`. A backbone downloads its torchvision ImageNet weights, replaces the 1000-class layer with a six-class layer and fine-tunes the whole network.

```powershell
.\.venv\Scripts\python.exe -m wastelens.train --architecture mobilenet_v3_large --device cuda --epochs 20 --affine-augmentation --output outputs/mobilenet-20
```

| Setting | Lab02 MLP | Pretrained backbone |
| --- | --- | --- |
| Input | 64 × 64, RGB in [0, 1] | 224 × 224, RGB in [0, 1]; ImageNet mean/std applied inside the model |
| Optimizer | Adam, constant learning rate 0.001 | AdamW, weight decay 0.01, cosine decay over the run |
| Learning rate | 0.001 | 0.0001 for pretrained layers, 10× for the new output layer |
| Parameters | 1,577,318 | 4,209,718 · 4,015,234 · 23,520,326 |

`--learning-rate` and `--image-size` override the defaults. Images are resized to a square without cropping, so the app can classify crops of any aspect ratio with the same `preprocess()`. A saved checkpoint records its architecture, and `load_classifier()` rebuilds it without downloading weights.

## Train on Kaggle GPU

Import [`notebooks/train_kaggle.ipynb`](notebooks/train_kaggle.ipynb), enable **Internet** and select **GPU T4 ×2** or **L4** under Accelerator. Set `ARCHITECTURE` in the setup cell, then run all cells using Kaggle's PyTorch environment. The notebook uses one GPU with automatic mixed precision and stops with an explicit error if CUDA cannot run.

The notebook downloads the versioned dataset, previews the six classes and fine-tunes MobileNetV3-Large for 20 epochs with affine augmentation by default. Download the run directory under `/kaggle/working/wastelens-results/` after execution.

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
