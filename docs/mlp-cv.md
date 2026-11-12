# Repeated CV and seed ensembles for the descriptor MLP

Rounds one to three selected configurations on a single 380-image validation split and then read the test split. At about 79% accuracy, the 95% Clopper-Pearson interval on 377 test images spans 74.3–82.8%. An exact McNemar test needs a true difference of roughly 4–6 points for 80% power, so most round-to-round differences of 0.5–2 points cannot be resolved. This protocol replaces validation-split selection with repeated cross-validation and evaluates the test split once.

## Protocol

1. **Lock the test split.** `python -m wastelens.cv cv` reads only the train and val images (2,144). The test descriptors are computed only by `final`, after `selection.json` is on disk.
2. **Repeated stratified 5-fold CV, 3 repeats.** Folds are stratified per class with split seeds 2026–2028. Each fold trains 5 seeds (42–46) of each configuration, using the round-three training recipe: batch size 64, Adam at 0.001, random horizontal flips from precomputed flipped descriptors. Standardization is fitted on the fold's training images only.
3. **Epoch budget as a hyperparameter.** No fold selects a checkpoint on its own held-out images. Out-of-fold (OOF) log-probabilities are recorded after 100, 200, 300, 400 and 600 epochs, and the budget is chosen like any other setting.
4. **Candidates.** Each candidate is a configuration (HOG + HSV `baseline`, HOG + HSV + Lab `extra_lab`, `hsv_only`), an epoch budget, and either a single seed or the 5-seed ensemble (mean of softmax probabilities). That gives 30 candidates. A single seed is scored seed by seed and averaged within each fold, so every candidate has 15 fold scores.
5. **Selection.** Highest mean fold macro-F1, then the one-standard-error rule: among candidates within one SE of the best, take the fewest parameters (an ensemble counts 5×), then the fewest epochs. The SE treats the 15 folds as independent and is therefore optimistic. It ranks candidates and is not a confidence interval.
6. **Post-hoc logit adjustment.** The scores become `log p − τ · log prior`, with τ ∈ {0, 0.25, …, 2}. In CV, τ is cross-fitted: each fold uses the τ that maximizes macro-F1 on the other folds of the same repeat. Adjustment is kept only if it raises the chosen candidate's mean fold macro-F1. The final τ is tuned once on the pooled OOF scores.
7. **One test evaluation.** `final` retrains the chosen candidate on all 2,144 images and refuses to run if its output directory exists. It reports:
   - accuracy with a Clopper-Pearson interval;
   - percentile bootstrap intervals (2,000 resamples) for accuracy, macro-F1 and their paired differences from the first seed alone;
   - an exact McNemar test against that single seed;
   - per-class `k/n` recall with exact intervals.

## Running

```powershell
# Local CPU smoke test (about 30 s); a full CPU run would take several hours.
.\.venv\Scripts\python.exe -m wastelens.cv cv --folds 3 --repeats 1 --seeds 42 43 --eval-epochs 5 10 --workers 4 --output outputs/cv-smoke
# Full protocol on GPUs (225 fits of 600 epochs), then the one-time test.
python -m wastelens.cv cv --device cuda --workers 4 --output outputs/mlp-cv
python -m wastelens.cv final --device cuda --cv-output outputs/mlp-cv --output outputs/mlp-cv-final
```

`scripts/build_cv_kaggle.py --staging DIR` writes [`notebooks/mlp_cv_kaggle.ipynb`](../notebooks/mlp_cv_kaggle.ipynb) with an embedded source snapshot, plus a `kernel-metadata.json` for a Kaggle T4 ×2 run. The notebook runs both stages.

| Output | Content |
| --- | --- |
| `cv/selection.json` | Chosen and best candidates, one-SE threshold, τ, fold seeds, OOF McNemar tests against the best single-seed HOG + HSV + Lab candidate |
| `cv/cv-candidates.csv` | Fold mean and SE of macro-F1, OOF accuracy, `trash` recall and cross-fitted adjusted macro-F1 for every candidate |
| `cv/oof-log-probs.npz` | OOF log-probabilities with shape (repeats, seeds, epoch budgets, images, classes), labels and fold assignments |
| `final/test-result.json`, `final/test-predictions.csv` | The single test evaluation and its per-image predictions |

## Deployed ensemble

`cv` and `final` keep no weights. `export` trains the model used by the application on all 2,144 train + val images and never reads the test split:

```powershell
.\.venv\Scripts\python.exe -m wastelens.cv export --cv-output outputs/kaggle-mlp-cv-20261111/cv --output models/wastelens-ensemble.pt --workers 5
```

- **Members.** `--members` defaults to `extra_lab:300 hsv_only:300` with the CV seeds 42–46, giving 10 MLPs. On the out-of-fold scores this two-view ensemble raises mean macro-F1 from 0.772 to 0.797 and accuracy from 78.8% to 80.8%. The combination was chosen after inspecting those scores, so treat the gain as about two points.
- **Calibration before training.** The temperature `T` is fitted on the OOF scores of exactly these members, minimizing negative log-likelihood. The "uncertain" threshold is the lowest calibrated confidence whose accepted OOF predictions are still 90% correct (`--target-accuracy`). A cross-fitted check, where each fold takes `T` and the threshold from the other folds, is stored beside them.
- **Checkpoint.** One file holds every member's `state_dict` with its standardization buffers, the descriptor name, class order, `T`, the threshold, calibration metrics and the training environment (`torch`, `cv2` versions, manifest hash). It loads with `torch.load(weights_only=True)`.
- **Reload check.** After saving, `export` reloads the file and recomputes probabilities for 16 training images from the original JPEGs. They must match the in-memory members, which also checks that inference descriptors equal the cached training descriptors.
- **Reproducibility.** These members are retrained on the export device. The test result of the 5-seed Kaggle ensemble (83.55%) does not transfer to them exactly, and the two-view ensemble has no test result yet.

The application loads it through `wastelens.classifier.WasteClassifier`. `classify()` accepts a PIL image or an OpenCV BGR frame and returns:

- the label, or `uncertain` below the threshold;
- the calibrated confidence;
- all six probabilities;
- feature, model and total time in milliseconds, measured after warm-up.

## Limitations

- The test split was inspected during rounds one to three, so the final number is not an untouched holdout. A webcam test set collected and locked separately is the clean estimate.
- CV estimates for the selected candidate are optimistically biased by the selection itself. Report the test result for the chosen model and treat CV numbers as ranking evidence.
- Seeds share one fold split per repeat. Repeats vary the split, and seeds vary initialization, dropout, batch order and flips.
