# FLAIR reproducibility package

Version 5. Reliability workflows.

**Google Colab:** [Open in Colab](https://colab.research.google.com/drive/1mXn-YjMucoitroASAMLw1EJZrgW94WL2?usp=sharing)  
**Model Checkpoints:** [Download checkpoints](https://github.com/YCKevinZhao/OnlineModel-Predict)

This package contains source code, source datasets, fixed main-model data splits, and seven default runtime weight files for the FLAIR manuscript. It omits generated results, deployment caches, training logs, and historical experiment weights. Colab supplies the separate result artifacts needed for historical comparisons.

## Scientific scope

The main model predicts absorption wavelength (`abs`), emission wavelength (`emi`), photoluminescence quantum yield (`plqy`), and log10 molar extinction coefficient (`em`). The reliability model estimates the absolute error (AE) of each property prediction.

Test15 is a fixed held-out split of the merged evaluation pool, not a completely independent external test set. Data filenames use `holdout` to reflect this boundary.

The default checkpoint is `models/reliability_model/mlp_log_hpo100_reliability_model_bundle.pt`. It contains four property-specific Log-MLPs using 32 solvent-augmented features and sample-weighted L1 loss in `log(AE + epsilon)` space.

The default model was selected from a search that evaluated 100 shared hyperparameter candidates using leakage-safe five-fold canonical solute-solvent GroupKFold within Calibration85. Test15 was excluded from selection. Trial 65 had weighted validation Spearman 0.631056. The recorded Test15 weighted Spearman was 0.658998, with weighted AE-MAE 10.046812. These metrics refer to the default checkpoint’s model-selection run. Result tables are available separately through Colab. Step 3 trains an ensemble with manually supplied parameters.

## Directory layout

- `scripts/`: workflow wrappers and the release builder.
- `src/main_model/`: main-model training and inference.
- `src/reliability_model/`: feature preparation, manual MLP ensemble training, and inference.
- `src/data_splitting/`: split generation and validation.
- `data/source/`: source datasets and separate evaluation cohorts.
- `data/splits/deployment/`: main-model deployment and test tables, including fixed `cv_fold` assignments.
- `data/splits/development/`: alternative Random/Scaffold partitions for analysis.
- `data/templates/`: prediction input and manual MLP parameter examples.
- `environment/`: dependency installation commands.
- `models/`: seven default runtime weight files.
- `results/`: generated feature and evaluation outputs; absent from this distribution.
- `data/cache/`: runtime deployment caches; generated when needed and omitted from releases.

## Weight files

| Path under `models/` | Purpose |
|---|---|
| `pretrained/MORE.pth` | Downloaded pretrained encoder required by the current main-model workflow |
| `main_model/fold_01/best_model.pt` through `fold_05/best_model.pt` | Five main-model folds; reliability inference uses their mean predictions |
| `reliability_model/mlp_log_hpo100_reliability_model_bundle.pt` | Default single-model Log-MLP bundle |

The predictor builds its cache under `data/cache/` using calibration features and the deployment reference table.

`MORE.pth` is a required input. If a release distributes weights through a download link, place the file at `models/pretrained/MORE.pth` and check its SHA256 against the release information.

## Held-out solvent benchmark

`spec_solv.csv` contains 144 records across 12 solvent identities for the held-out-solvent benchmark. The manuscript reports 165 reserved records before PBS exclusion. This CSV excludes the 21 PBS (phosphate-buffered saline) records because PBS is a buffered mixture that cannot be represented as a single solvent SMILES for the current model input. The 144 records therefore describe the benchmark after PBS exclusion. Keep this benchmark separate from main-model training and Reliability Calibration85/Test15; its existing `split` labels do not define the held-out-solvent evaluation.

## Negative solvatochromism benchmark

`data/source/negative_solvatochromism_38_compounds.csv` contains 38 compounds and 76 compound-solvent records, with one dichloromethane (DCM) and one toluene (TOL) record per compound. `Compound_ID` numbers the compounds from 1 to 38 in the supplied order; each DCM/TOL pair shares one ID. The table contains solute and solvent SMILES, experimental absorption/emission wavelengths, and the source article DOI. All column names and text entries are in English. Keep this cohort separate from model training and Reliability calibration.

## Environment

`environment/conda_install_dependencies.txt` contains commands for a Python 3.10 Conda environment named `Pytorch`, scientific Python and RDKit dependencies, and the CUDA 12.8 PyTorch/PyG stack. Run them in a shell with Conda initialized. The final import command checks the listed packages.

These commands target NVIDIA CUDA systems. A separate CPU installation file is not included. The prediction wrapper can use another compatible environment name and checks imports before execution.

## Workflow

Run commands from the package root. The supplied main-model weights allow you to begin at Step 2; Step 1 retrains the five main models.

```bash
python scripts/01_train_main_model.py --check
python scripts/02_extract_reliability_features.py --check
python src/data_splitting/check_generated_splits.py
```

`--check` validates prerequisites without training or feature extraction. Remove it to execute a stage. Full main-model training is computationally intensive; a CUDA-capable GPU is recommended.

Step 2 generates Calibration85/Test15 split files, feature tables, and labels under `results/intermediate/`. With no saved Reliability split files, it constructs the grouped split using the supplied inputs, seed, and test fraction. Reproducing historical metrics requires the matching split, input contents, weights, and environment. The deployment `cv_fold` assignments are a separate fixed main-model partition.

After Step 2, the default prediction command is:

```bash
python scripts/04_predict_single.py --check
python scripts/04_predict_single.py --smiles "CCO" --solvent "O"
```

The wrapper requires `results/intermediate/calibration_features_ae_pre.csv` even when using the supplied default checkpoint. On first execution it builds the deployment cache. Prediction with the default checkpoint requires Steps 2 and 4.

## Manual reliability training

`scripts/03_train_reliability_model.py` forwards arguments to `src/reliability_model/03_train_log_mlp_cv5_ensemble.py`. This trains a five-fold Log-MLP ensemble, rebuilds target-derived features within each training fold, and averages member predictions in raw AE space. It writes ensemble weights to a separate run directory and leaves the default single-model checkpoint in place.

Step 3 requires these inputs, which are omitted from the distribution:

- `results/intermediate/calibration_features_ae_pre.csv`
- `results/intermediate/test_features_ae_pre.csv`
- `results/intermediate/test_labels_ae_pre.csv`
- `results/final/mlp_log_hpo100_20260824/test15_evaluated_rows.csv`

Step 2 generates the three intermediate tables. Download the matching default-model Test15 comparison table from Colab before running Step 3. Keep its Test15 rows aligned with the feature and label tables. Step 3 does not generate this historical baseline table.

Supply a flat JSON object with `hidden1`, `hidden2`, `activation`, `dropout`, `optimizer`, `learning_rate`, `weight_decay`, and `epochs`. The example JSON contains the trial-65 parameter values and is read only when passed explicitly.

```bash
python scripts/03_train_reliability_model.py --params-json data/templates/mlp_manual_parameters.json --check
python src/reliability_model/03_train_log_mlp_cv5_ensemble.py --params-json data/templates/mlp_manual_parameters.json --check
```

Alternatively, supply all eight flags:

```bash
python src/reliability_model/03_train_log_mlp_cv5_ensemble.py \
  --hidden1 64 --hidden2 32 --activation leaky_relu --dropout 0.15 \
  --optimizer adam --learning-rate 0.000868422081016009 \
  --weight-decay 5.8227148141978355e-06 --epochs 500 --check
```

Use either JSON or flags. Missing, unknown, mixed-source, or invalid parameters cause an error. Outputs go to `results/final/mlp_manual_cv5_ensemble/` and `models/reliability_model/mlp_manual_cv5_ensemble/`. `--run-name` selects a separate name; existing output directories cause an error. `--smoke-test` uses two epochs and separate `_smoke` directories.

Bootstrap and baseline comparisons remain part of Step 3. Each run records the supplied parameters and seed rule; a manual run is not identified as an HPO-selected trial.

## Reliability entry points

| Entry point in `src/reliability_model/` | Purpose |
|---|---|
| `01_generate_offline_features.py` | General offline feature preparation and checkpoint utilities |
| `02_generate_ae_pre_merged_features.py` | Calibration85/Test15 split and feature preparation |
| `03_train_log_mlp_cv5_ensemble.py` | Manual five-fold MLP ensemble training, evaluation, and baseline comparison |

Shared implementations are in `mlp_core.py`, `reliability_features.py`, and the data/model helpers. Feature metadata uses the compatibility key `final_rf_feature_columns`; feature-validation messages use the tag `[RF]`. Reliability training and prediction use MLP models.

Main-model factories support the encoders, fusion modes, and losses exposed by the training interfaces.

## Reproducibility and release status

- Fold *k* uses deployment rows with `cv_fold != k` for training and `cv_fold == k` for validation.
- Checkpoint metadata may contain paths from the model’s training environment. Runtime wrappers resolve input files relative to the package.
- `MANIFEST.sha256` records the supplied files, excluding the manifest itself and generated operating-system/Python metadata.
- `scripts/05_build_release_package.py` excludes deployment caches and experiment artifacts. Its current required-file check also expects `RELEASE_NOTES.md` and `environment/portable_cpu_commands.txt`, which are absent; the builder needs those files or an updated release configuration before use. Set `--name` to the desired release name; the default archive name refers to v3.
- Authors must complete the placeholder authorship in `CITATION.cff`, select code/data licenses in `LICENSE`, and confirm redistribution terms for downloaded weights and source data before public release.
