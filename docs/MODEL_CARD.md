# Model card

## Main model

The main-model workflow requires the downloaded MORE encoder at `models/pretrained/MORE.pth`. Five fold-specific checkpoints are stored under `models/main_model/fold_01` through `fold_05`. The reliability workflow uses the mean of their five property predictions.

## Default reliability model

`scripts/04_predict_single.py` uses `models/reliability_model/mlp_log_hpo100_reliability_model_bundle.pt`. It contains four property-specific Log-MLPs with 32 structural, applicability-domain, neighborhood, historical-error, fold-variation, and solvent occurrence/Tanimoto features. Each output estimates the absolute error of the corresponding property prediction.

The default model was selected through a 100-candidate hyperparameter search with leakage-safe five-fold canonical solute-solvent GroupKFold inside Calibration85. It uses sample-weighted L1 loss in `log(AE + epsilon)` space. Test15 is excluded from selection.

The historical run record `results/final/mlp_log_hpo100_20260824/validation.json` (omitted from this distribution) records trial 65, weighted pooled out-of-fold validation Spearman 0.6310563284911787, Test15 weighted Spearman 0.6589980233120313, and weighted AE-MAE 10.046811636040012. These metrics refer to that checkpoint’s model-selection run.

## Manual ensemble training

`03_train_log_mlp_cv5_ensemble.py` requires eight explicitly supplied training parameters, from a flat JSON file or command-line flags. It does not load the trial-65 selection file or run HPO. Training uses target scaling, weighted L1 loss, grouped folds, fold-local historical-error features, and raw-AE ensemble averaging.

New runs use separate manual-run output directories. They do not automatically replace the default full-Calibration checkpoint. The distributed `models/` directory contains only the seven default runtime weight files. Deployment caches are rebuilt under `data/cache/` when needed.

Interpret weighted aggregate MAE alongside property-level tables. The four properties have different physical scales; pooling regression values across incompatible physical units would change their meaning.

Step 3 requires generated calibration/test feature tables, test labels, and the matching historical Test15 baseline rows from the separate Colab artifacts. The default predictor requires calibration features before it can rebuild its cache. See `../README.md` for input paths and execution order.

## Intended use

The package supports reproduction of the manuscript analysis and reliability assessment for solute-solvent photophysical-property predictions. Experimental validation is still required.

## Limitations

Test15 comes from the merged evaluation pool, not a completely independent external cohort. Applicability may degrade for novel chemical scaffolds or solvents outside the represented domain. Load pickle and PyTorch checkpoint files only from trusted sources.
