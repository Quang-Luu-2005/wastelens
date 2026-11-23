# WasteLens · Lab03

WasteLens is a six-class waste classification project for Digital Image & Video Processing Lab03. It extends the Horse or Human MLP from Lab02 through from-scratch experiments. The app offers the adapted baseline, the improved MLP ensemble and an ImageNet-pretrained MobileNetV3 Large fine-tuned for comparison.

## Quick start (Windows)

You need Python 3.12 and an internet connection for the first run. Copy the three model files (`wastelens-baseline.pt`, `wastelens-ensemble.pt`, `wastelens-mobilenetv3.pt`) from the project's Google Drive folder into `models\`. Then run from this folder, in Command Prompt or PowerShell:

```powershell
.\run.bat
```

The first run creates `.venv`, installs CPU-only PyTorch and the pinned requirements (about 300 MB), downloads and verifies the TrashNet sample images and the webcam detector, and then opens the app at http://localhost:8501. Later runs start the app directly.

- `.\run.bat setup` only installs.
- `.\run.bat test` runs the unit tests.
- Double-clicking `run.bat` also works.

The script always runs from its own folder, so Streamlit picks up the light theme in `.streamlit\config.toml`. If the dataset or detector download fails, the image and video modes still work with your own files.

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

## Run the app

The Streamlit app offers three independently selectable checkpoints. Model weights are git-ignored. After downloading the original experiment outputs, install the verified checkpoints with the command below; see [model provenance and comparison](docs/app-models.md) for exact paths, hashes and training settings.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_app_models.py
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

| Model | Local checkpoint | Test accuracy (377 images) |
| --- | --- | ---: |
| Baseline: Lab02 MLP 128 → 32 → 6, RGB pixels, from scratch | `models/wastelens-baseline.pt` | 58.62% |
| Improved: 10 descriptor MLPs 256 → 64 → 32 → 6, from scratch | `models/wastelens-ensemble.pt` | 85.68% |
| MobileNetV3 Large, ImageNet pretrained then fine-tuned | `models/wastelens-mobilenetv3.pt` | 94.69% |

See [architecture and screenshots](docs/architecture-and-ui.md) for the pipeline diagram and one screenshot per feature.

The centered image workbench follows the supplied Stitch design. Upload an image or choose one of the six sample buttons, then select the model and processing level in the compact toolbar below the image. The improved MLP is the default. Click **Phân loại** to run inference; changing the image, model, settings or checkpoint invalidates the previous result and its downloads until you classify again.

Processing levels are **Gốc** (identity), **Nhẹ** (Gaussian σ=0.4), **Vừa** (gamma 1.15, CLAHE 2.0, Gaussian σ=0.8), and **Mạnh** (gamma 1.25, contrast 1.15, CLAHE 3.0, Gaussian σ=1.2). They are experimental filter presets, not quality guarantees. **Chỉnh sửa** opens the full Module 1 dialog; settings take effect only after **Áp dụng**. Preview modes show the original, processed image or both side by side.

Enable **So sánh cả 3 mô hình trên ảnh này** after inference to compare independent predictions on the same processed image. The app never chooses a winner by comparing confidence values. A missing optional checkpoint does not block the selected available model. The interface shows actual probabilities and timings; the reference design's sample scores, Platt calibration and CUDA/ONNX labels are not used.

These are exploratory results from existing experiments: baseline/MobileNet train on 1,764 images with a separate 380-image validation split, while the improved ensemble is refitted on all 2,144 development images after cross-validation. The table is not a controlled comparison with identical training budgets. All scores use the same fixed test set, including predictions below the ensemble's uncertainty threshold.

The image mode accepts a JPG or PNG upload, or a held-out TrashNet test image when `data/trashnet/` has been prepared.

Dataset samples display their ground-truth label from the versioned manifest. Each of the six class buttons selects a random test image of that class on every click; **Random rác** samples across all 377 test images. The current image is excluded whenever another candidate exists. **Chọn ảnh trong tập test** also lets you select a specific test image. For these samples, the prediction card is followed by a **Ground Truth: …** card containing the reference label and correct/incorrect/uncertain status. The three-model comparison uses four columns: **Tên mô hình**, **Dự đoán của mô hình**, **Nhãn thật** and **Đối chiếu**. Uploaded images have no label selector, ground-truth card or correctness indicators; their comparison table shows model names and predictions only. JSON exports keep unknown labels/correctness as null. For dataset samples, JSON and annotated PNGs include the reference label. These are per-image checks, not dataset accuracy.

| Part | What it does |
| --- | --- |
| Module 1 (presets and custom dialog) | Color: gamma, contrast and brightness, HSV hue/saturation/value, CLAHE, grayscale. Geometry: rotation, scale, flips, center crop. Smoothing: Gaussian, median, bilateral. Optional synthetic Gaussian and salt-and-pepper noise. Original/processed/side-by-side preview modes. |
| Module 2 | Explicit inference from the selected model: label, confidence and all six probabilities in a table with proportional bars. Only the improved ensemble uses calibrated confidence and its stored *uncertain* threshold; baseline/MobileNet show raw softmax confidence. Shows compact inference time, with full stage timings in the JSON export. When Module 1 is active, also shows the prediction on the original image. Ground-truth card and optional three-model comparison table. |
| Snapshot & export | Downloads the annotated result PNG, the processed PNG and a JSON record of the selected model, settings, predictions and timings, including all three results when comparison is enabled. |

### Video file mode

Choose **Video** in *Nguồn đầu vào*. Upload an MP4, MOV, AVI, MKV or WEBM file, or click **Tạo video mẫu từ tập test**, which builds a 12-second clip from one random test image per class with a slow zoom and known labels. Then select the model and the Module 1 processing (a preset, or the custom settings applied in the image tab) and click **Xử lý video**. [`wastelens.video`](src/wastelens/video.py) runs the pipeline for each frame:

1. Decode with OpenCV, which applies rotation metadata.
2. Downscale so the longest side is at most 960 px.
3. Apply Module 1.
4. Every N frames (default about five per second), classify the frame and update an exponential moving average (EMA) of the six probabilities.
5. Draw the smoothed label and confidence.
6. Encode H.264 with PyAV (`libx264`, `yuv420p`, `faststart`) so browsers can play the result.

The improved ensemble's *uncertain* threshold applies to the smoothed probabilities. **Thiết lập video** sets N, the processing limit (60 s by default), the output resolution and the EMA weight; a weight of 1.0 disables smoothing.

The result shows:

- the original and processed videos side by side;
- the most frequent confident label and the share of uncertain frames;
- frame counts, processing speed and mean milliseconds per stage (decode, Module 1, classification, encoding);
- a line chart of the six smoothed probabilities over time;
- a snapshot slider with the six-class table for that frame.

Downloads include the annotated MP4, a per-frame CSV of raw and smoothed probabilities, a JSON summary, and the snapshot PNG. For the sample clip, the app compares frames with the known segment labels: raw Top-1 is correct or wrong, and smoothed labels are correct, uncertain or wrong. These are images on TrashNet backgrounds stitched into a clip, not real-world video accuracy. Video mode runs locally only; the stateless deployment accepts at most 4 MB uploads and does not install PyAV. Uploaded and processed videos are stored under the system temporary folder in `wastelens-video/`.

The operations live in [`wastelens.ops`](src/wastelens/ops.py) as functions from a BGR image to a BGR image, so video, webcam and the filter experiment can reuse them. Uploads larger than 1024 px on the long side are downscaled first.

The calibrated confidence is reliable only for photos like TrashNet's. Shifting the hue of a held-out cardboard photo by 40 OpenCV units flips the prediction to glass at 97% confidence. Random-noise images can also receive confident labels.

### Filter x model experiment

[`scripts/filter_experiment.py`](scripts/filter_experiment.py) applies 76 OpenCV conditions to the 377 test photos at their original size, as Module 1 does: Module 1 filters, nine corruptions at five severities, and restoration filters after a corruption. It evaluates all three checkpoints and compares each condition with the clean predictions by exact McNemar with Holm correction.

| Model | Clean | Mean over 45 corruptions | Relative drop |
| --- | ---: | ---: | ---: |
| Baseline Lab 2 | 58.6% | 47.8% | 18.5% |
| Improved MLP ensemble | 85.7% | 62.1% | 27.5% |
| MobileNetV3 | 94.7% | 84.2% | 11.1% |

The descriptor ensemble is robust to blur and noise but loses 26 points under a 5% warm colour cast and 61 points when the image is darkened to V = 0.3. MobileNetV3 is robust to colour but falls to 43.8% under 15% salt-and-pepper noise. No Module 1 filter significantly improves accuracy on clean TrashNet photos. A median filter recovers salt-and-pepper damage almost completely, and gray-world recovers most of a colour cast. See the [analysis](docs/filter-experiment.md) and the [tables and figures](reports/filter-robustness-20261119/results.md).

### Binary versus multi-class

[`scripts/binary_vs_multiclass.py`](scripts/binary_vs_multiclass.py) keeps the descriptors, the 256-64-32 MLP, the optimizer and the CV folds fixed, and changes only the label map, the output units and the loss. With the same features, a binary fibre-vs-rest task reaches 91.0% out-of-fold accuracy against 76.9% for six classes. After correcting for chance, Cohen's κ is 0.811 against 0.716. A 1-logit sigmoid and a 2-logit softmax are statistically indistinguishable, and summing the six-class probabilities answers the binary questions as well as dedicated binary models. The test split is not used. See the [derivations and results](docs/binary-vs-multiclass.md).

To reproduce the deployed models' test scores and check that PIL/OpenCV inputs agree:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_app_models.py
```

This writes `outputs/app-model-comparison.json` with hashes, confusion matrices and per-image probabilities. It does not train or tune models on test images.

UI regression checks (the inference tests require the installed checkpoints):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Pretrained backbones

For the Vercel container deployment, build/run commands and the verified model/test-image assets, see [deployment/README.md](deployment/README.md). This uses the Vercel Container Runtime beta and CPU inference, with a 4 MB upload cap in the deployed container.

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

### From-scratch MLP ablations

The current investigation sets pretrained backbones aside and tests nine explicit
Lab02 MLP variants: depth, width, dropout, LayerNorm, AdamW with cosine decay,
label smoothing, class-weighted cross-entropy and focal loss. Each receives 600
epochs at RGB 64x64 with paired seeds 42/43. Configuration selection uses mean
validation macro-F1 before evaluating baseline and winner on test. See the
[experimental protocol](docs/mlp-ablation.md) and the self-contained
[Kaggle notebook](notebooks/mlp_ablation_kaggle.ipynb). Results are exploratory
because this project's test split has already been inspected in prior runs.

The [completed two-seed comparison](reports/mlp-ablation-20261111.md) selects the
256-64-32 MLP with LayerNorm and dropout, ordinary cross-entropy and Adam. Mean
test accuracy is **62.33%**, versus **55.04%** for the baseline rerun in the same
experiment (**+7.29 percentage points**). Both seeds improve. This is an
exploratory result, not an estimate of the best possible MLP performance.

The [second round](reports/mlp-round2-20261111.md) tests input normalization,
GELU, lighter/no dropout, matched dense blocks with/without skip connections,
and fixed OpenCV HOG/HSV descriptors. Validation selects **HOG + HSV → MLP**:
**77.32% mean test accuracy**, versus the reproduced first-round winner's
**62.33%** (**+14.99 percentage points**). The classifier is still a randomly
initialized 256-64-32 MLP with six outputs, now taking 2,268 descriptor values
instead of 12,288 RGB pixels; it has **600,294 trainable parameters**.
Descriptor extraction has no pretrained weights, and its train-fitted
normalization buffers are saved with the checkpoint. See the
[round-two protocol](docs/mlp-round2.md) and
[Kaggle notebook](notebooks/mlp_ablation_round2_kaggle.ipynb).

The [third round](reports/mlp-round3-20261111.md) selects HOG + HSV + Lab with
the same from-scratch MLP: **78.91% mean test accuracy**, up **1.59 percentage
points** from 77.32%. Both seeds improve in accuracy; test macro-F1 rises only
from **0.7741 to 0.7774**, with lower F1 for the small `trash` class. The model
has 2,772 inputs and 729,318 parameters. HSV-only is nearly tied on validation
macro-F1 with 148,710 parameters; the selection is exploratory, not a claim of
statistical superiority. See the [protocol](docs/mlp-round3.md) and
[notebook](notebooks/mlp_ablation_round3_kaggle.ipynb).

Single-split selection cannot resolve these differences: at about 79%, the 95%
interval on 377 test images is 74.3–82.8%. The [CV protocol](docs/mlp-cv.md)
re-ranks the round-three HOG + HSV, HOG + HSV + Lab and HSV-only MLPs with
repeated stratified 5-fold CV (3 repeats × 5 seeds) on train + val. It also
selects the epoch budget, seed ensembling and post-hoc logit adjustment by the
one-SE rule, then evaluates the test split once with exact and bootstrap
intervals and a McNemar test. Run it with `python -m wastelens.cv` or the
[Kaggle notebook](notebooks/mlp_cv_kaggle.ipynb).

The CV run selected 5 HOG + HSV + Lab MLPs at 300 epochs. That ensemble scored
**83.55%** test accuracy (315/377, 95% interval 79.4–87.2%) against 78.8%
out-of-fold. The application uses a two-view ensemble written by
`python -m wastelens.cv export`: those 5 MLPs plus 5 HSV-only MLPs, retrained on
all train + val images without reading the test split. Its out-of-fold accuracy
is 80.8%. A paired Kaggle run then evaluated this ensemble at 300 and 600 epochs
on the test split. The 300-epoch ensemble scored **85.68%** (323/377, 95%
interval 81.7–89.1%, macro-F1 0.861) and the 600-epoch ensemble 84.62%. With
only 14 discordant images (exact McNemar p = 0.42), the two are statistically
indistinguishable, so the cheaper 300-epoch checkpoint is deployed. This was
another look at an already-inspected test split. Calibrated predictions below the stored threshold are labelled
`uncertain`; the accepted out-of-fold predictions are 90% correct at 77%
coverage. `wastelens.classifier.WasteClassifier` loads the checkpoint. On CPU it
takes about 9 ms per image (median, features and model). See
[the export section](docs/mlp-cv.md#deployed-ensemble).

## Local webcam

The **Webcam** tab detects objects with SSDLite, draws bounding boxes, and classifies each crop using the selected Baseline Lab 2, improved MLP, or MobileNet model. Select a box to inspect its six-class probabilities. Run `python scripts/prepare_detector.py` once before opening the camera. See [local webcam setup and limitations](docs/webcam-local.md).

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
