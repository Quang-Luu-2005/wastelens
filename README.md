# WasteLens · Lab03

WasteLens is a six-class waste classification project for Digital Image & Video Processing Lab03. The repository contains the environment check, the TrashNet preparation pipeline, and a Kaggle-ready notebook for the next classifier task.

## Run locally

Use Python 3.12. These PowerShell commands install the CPU builds and prepare the dataset from a fresh clone.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe check_env.py
.\.venv\Scripts\python.exe scripts/prepare_trashnet.py
```

The CPU check exercises OpenCV BGR-to-RGB conversion and a PyTorch forward pass with six finite outputs. The dataset script caches and verifies the source ZIP, extracts eligible images to `data/trashnet/`, and regenerates the versioned split files. For a verified local copy of the ZIP, run the same script with `--archive PATH`.

On Kaggle, import [`notebooks/train_kaggle.ipynb`](notebooks/train_kaggle.ipynb) and enable Internet for the repository and dataset downloads. Run the notebook to inspect the split and preview one training image per class. Use `class_id` from the manifest for the classifier's label order.

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
