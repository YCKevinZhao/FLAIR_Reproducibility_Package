from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
RELIABILITY_DIR = PACKAGE_ROOT / "src/reliability_model"
CURRENT_CHECKPOINT = (
    PACKAGE_ROOT / "models/reliability_model/mlp_log_hpo100_reliability_model_bundle.pt"
)
CURRENT_EVALUATED = PACKAGE_ROOT / "results/final/mlp_log_hpo100_20260824/test15_evaluated_rows.csv"
CALIBRATION_FEATURES = PACKAGE_ROOT / "results/intermediate/calibration_features_ae_pre.csv"
TEST_FEATURES = PACKAGE_ROOT / "results/intermediate/test_features_ae_pre.csv"
TEST_LABELS = PACKAGE_ROOT / "results/intermediate/test_labels_ae_pre.csv"
DEPLOYMENT_REFERENCE = PACKAGE_ROOT / "data/splits/deployment/deployment.csv"

RUN_NAME = "mlp_manual_cv5_ensemble"
RESULT_DIR = PACKAGE_ROOT / "results/final" / RUN_NAME
MODEL_DIR = PACKAGE_ROOT / "models/reliability_model" / RUN_NAME
SMOKE_RESULT_DIR = PACKAGE_ROOT / "results/final" / f"{RUN_NAME}_smoke"
SMOKE_MODEL_DIR = PACKAGE_ROOT / "models/reliability_model" / f"{RUN_NAME}_smoke"

os.environ["MAIN_MODEL_ROOT"] = str(PACKAGE_ROOT / "src/main_model")
os.environ["MODEL_CODE_ROOT"] = str(PACKAGE_ROOT / "src/main_model")
sys.path.insert(0, str(RELIABILITY_DIR))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MLP06 = load_module("mlp_core_for_cv5_ensemble", RELIABILITY_DIR / "mlp_core.py")
STEP03 = MLP06.STEP03
STEP02 = MLP06.STEP02
PROPERTIES = list(MLP06.PROPERTIES)
FEATURE_COLUMNS = list(MLP06.FEATURE_COLUMNS)
EPSILON = dict(MLP06.EPSILON)
PROPERTY_INDEX = dict(MLP06.PROPERTY_INDEX)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a five-fold Log-MLP Reliability ensemble with manually supplied parameters."
    )
    parser.add_argument(
        "--params-json", type=Path, help="JSON object containing the eight training parameters."
    )
    parser.add_argument("--hidden1", type=int)
    parser.add_argument("--hidden2", type=int)
    parser.add_argument("--activation", choices=["relu", "leaky_relu", "silu", "gelu"])
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--optimizer", choices=["adam", "adamw"])
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--run-name",
        default=RUN_NAME,
        help="Output directory name under models/reliability_model and results/final.",
    )
    parser.add_argument(
        "--check", action="store_true", help="Validate parameters and inputs without writing files."
    )
    parser.add_argument(
        "--smoke-test", action="store_true", help="Use two epochs and separate smoke outputs."
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(tmp, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def load_manual_parameters(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    keys = (
        "hidden1",
        "hidden2",
        "activation",
        "dropout",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "epochs",
    )
    supplied = {key: getattr(args, key) for key in keys if getattr(args, key) is not None}
    if args.params_json is not None:
        if supplied:
            raise ValueError("Use either --params-json or individual parameter flags, not both.")
        supplied = json.loads(args.params_json.expanduser().read_text(encoding="utf-8"))
        if not isinstance(supplied, dict):
            raise ValueError(
                "Parameter JSON must be a flat object containing the eight training parameters."
            )
    missing = sorted(set(keys) - set(supplied))
    unknown = sorted(set(supplied) - set(keys))
    if missing or unknown:
        raise ValueError(f"Invalid manual parameter keys: missing={missing}, unknown={unknown}")
    for key in ("hidden1", "hidden2", "epochs"):
        if type(supplied[key]) is not int or supplied[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    for key in ("dropout", "learning_rate", "weight_decay"):
        value = supplied[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"{key} must be a finite number.")
    if not 0 <= supplied["dropout"] < 1:
        raise ValueError("dropout must be in [0, 1).")
    if supplied["learning_rate"] <= 0 or supplied["weight_decay"] < 0:
        raise ValueError("learning_rate must be positive and weight_decay nonnegative.")
    if not isinstance(supplied["activation"], str) or supplied["activation"] not in {
        "relu",
        "leaky_relu",
        "silu",
        "gelu",
    }:
        raise ValueError("Unsupported activation.")
    if not isinstance(supplied["optimizer"], str) or supplied["optimizer"] not in {"adam", "adamw"}:
        raise ValueError("Unsupported optimizer.")
    if not args.run_name or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in args.run_name
    ):
        raise ValueError("run-name must contain only letters, digits, underscores and hyphens.")
    if args.bootstrap_resamples < 0:
        raise ValueError("bootstrap-resamples must be nonnegative.")
    requested = {key: supplied[key] for key in keys}
    config = dict(requested)
    if args.smoke_test:
        config["epochs"] = 2
    provenance = {
        "parameter_source": "manual_json" if args.params_json is not None else "manual_cli",
        "requested_params": requested,
        "model_seed_rule": "42 + cv_fold*100 + property_index",
    }
    return provenance, config


def required_paths() -> list[Path]:
    return [
        CURRENT_CHECKPOINT,
        CURRENT_EVALUATED,
        CALIBRATION_FEATURES,
        TEST_FEATURES,
        TEST_LABELS,
        DEPLOYMENT_REFERENCE,
        RELIABILITY_DIR / "mlp_core.py",
        RELIABILITY_DIR / "rank_ensemble_features.py",
    ]


def inspect_existing_checkpoint() -> dict[str, Any]:
    try:
        bundle = torch.load(CURRENT_CHECKPOINT, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch 1.13, used for the original HPO run.
        bundle = torch.load(CURRENT_CHECKPOINT, map_location="cpu")
    models = bundle.get("models")
    if not isinstance(models, dict) or set(models) != set(PROPERTIES):
        raise ValueError("Existing MLP checkpoint does not contain exactly one model per property.")
    has_fold_members = any(isinstance(value, (list, tuple)) for value in models.values())
    return {
        "model_family": bundle.get("model_family"),
        "model_keys": sorted(models),
        "has_fold_members": bool(has_fold_members),
        "sha256": sha256(CURRENT_CHECKPOINT),
    }


def make_fold_frames(
    calibration: pd.DataFrame, test: pd.DataFrame
) -> tuple[list[pd.DataFrame], pd.DataFrame, list[dict[str, Any]]]:
    samples = calibration[["row_index", "smiles", "solvent"]].drop_duplicates().copy()
    if samples["row_index"].duplicated().any():
        raise ValueError("Calibration row_index maps to more than one solute-solvent pair.")
    samples["pair_group"] = STEP02.canonical_pair_keys(samples)
    splitter = GroupKFold(n_splits=5)
    frames: list[pd.DataFrame] = []
    assignment_frames: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []

    for cv_fold, (train_positions, validation_positions) in enumerate(
        splitter.split(samples, groups=samples["pair_group"]), start=1
    ):
        train_samples = samples.iloc[train_positions]
        validation_samples = samples.iloc[validation_positions]
        train_ids = set(train_samples["row_index"].astype(int))
        validation_ids = set(validation_samples["row_index"].astype(int))
        train_groups = set(train_samples["pair_group"].astype(str))
        validation_groups = set(validation_samples["pair_group"].astype(str))
        overlap = train_groups & validation_groups
        if overlap:
            raise ValueError(f"Fold {cv_fold} has {len(overlap)} canonical-pair group overlaps.")

        cal_work = calibration.copy()
        is_validation = cal_work["row_index"].astype(int).isin(validation_ids)
        if int((~is_validation).sum()) == 0 or int(is_validation.sum()) == 0:
            raise ValueError(f"Fold {cv_fold} has an empty training or validation frame.")
        cal_work["_role"] = np.where(is_validation, "validation", "calibration")

        test_work = test.copy()
        test_work["_role"] = "test15"
        combined = pd.concat([cal_work, test_work], ignore_index=True, sort=False)
        combined["split"] = np.where(combined["_role"] == "calibration", "calibration", "test")
        combined.loc[combined["_role"] != "calibration", "abs_error"] = np.nan

        # Target-derived history is rebuilt only from this member's four training folds.
        combined = STEP03.add_calibration_scaffold_error_features(combined)
        combined["split"] = combined["_role"]
        for prop in PROPERTIES:
            train_mask = (combined["split"] == "calibration") & (combined["property"] == prop)
            train_errors = combined.loc[train_mask, "abs_error"].to_numpy(dtype=float)
            combined.loc[train_mask, "sample_weight"] = STEP03.smooth_loss_weights(
                train_errors, prop
            )

        if not np.isfinite(combined[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
            raise ValueError(f"Fold {cv_fold} contains non-finite model features.")
        if combined.loc[combined["split"] == "test15", "abs_error"].notna().any():
            raise ValueError(f"Fold {cv_fold} Test15 frame contains target-derived abs_error.")
        if combined.loc[combined["split"] == "validation", "abs_error"].notna().any():
            raise ValueError(f"Fold {cv_fold} validation frame contains unmasked abs_error.")

        assignment = validation_samples.copy()
        assignment.insert(0, "cv_fold", cv_fold)
        assignment_frames.append(assignment)
        audits.append(
            {
                "cv_fold": cv_fold,
                "train_sample_count": len(train_ids),
                "validation_sample_count": len(validation_ids),
                "train_pair_group_count": len(train_groups),
                "validation_pair_group_count": len(validation_groups),
                "pair_group_overlap_count": len(overlap),
                "train_property_rows": int((combined["split"] == "calibration").sum()),
                "validation_property_rows": int((combined["split"] == "validation").sum()),
                "test15_property_rows": int((combined["split"] == "test15").sum()),
            }
        )
        frames.append(combined)

    assignments = pd.concat(assignment_frames, ignore_index=True)
    if assignments["row_index"].duplicated().any():
        raise ValueError("A Calibration sample was assigned to validation more than once.")
    if set(assignments["row_index"].astype(int)) != set(samples["row_index"].astype(int)):
        raise ValueError("Five-fold validation assignments do not cover Calibration exactly once.")
    return frames, assignments, audits


def metric_values(true_ae: np.ndarray, predicted_ae: np.ndarray) -> dict[str, float]:
    true_ae = np.asarray(true_ae, dtype=float)
    predicted_ae = np.asarray(predicted_ae, dtype=float)
    if len(true_ae) < 2 or not np.isfinite(true_ae).all() or not np.isfinite(predicted_ae).all():
        raise ValueError("Metric inputs must contain at least two finite rows.")
    residual_sum_squares = float(np.sum(np.square(true_ae - predicted_ae)))
    total_sum_squares = float(np.sum(np.square(true_ae - np.mean(true_ae))))
    r2 = 1.0 - residual_sum_squares / total_sum_squares
    return {
        "spearman": float(STEP03.spearman_manual(predicted_ae, true_ae)),
        "r2": float(r2),
        "mae": float(np.mean(np.abs(predicted_ae - true_ae))),
        "rmse": float(np.sqrt(np.mean(np.square(predicted_ae - true_ae)))),
    }


def metrics_by_property(frame: pd.DataFrame, predicted_column: str, variant: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for prop in PROPERTIES:
        subset = frame[frame["property"] == prop]
        values = metric_values(
            subset["true_ae"].to_numpy(dtype=float),
            subset[predicted_column].to_numpy(dtype=float),
        )
        rows.append({"variant": variant, "property": prop, "n": len(subset), **values})
    result = pd.DataFrame(rows)
    weights = result["n"].to_numpy(dtype=float)
    weighted = {
        "variant": variant,
        "property": "weighted_all",
        "n": int(result["n"].sum()),
        **{
            name: float(np.average(result[name].to_numpy(dtype=float), weights=weights))
            for name in ["spearman", "r2", "mae", "rmse"]
        },
    }
    return pd.concat([result, pd.DataFrame([weighted])], ignore_index=True)


def build_evaluation_frame(
    test: pd.DataFrame, predictions: pd.DataFrame, labels: pd.DataFrame, predicted_column: str
) -> pd.DataFrame:
    keys = ["property", "row_index", "fold"]
    evaluated = test.merge(predictions, on=keys, how="left", validate="one_to_one")
    evaluated = evaluated.merge(
        labels[["property", "row_index", "y_true"]],
        on=["property", "row_index"],
        how="left",
        validate="one_to_one",
    )
    if evaluated[[predicted_column, "y_true", "base_prediction"]].isna().any().any():
        raise ValueError("Missing Test15 value after label join.")
    evaluated["true_ae"] = np.abs(
        evaluated["base_prediction"].to_numpy(dtype=float)
        - evaluated["y_true"].to_numpy(dtype=float)
    )
    return evaluated


def paired_bootstrap(
    comparison_rows: pd.DataFrame, resamples: int, seed: int = 20260826
) -> pd.DataFrame:
    if resamples <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    property_frames = {
        prop: comparison_rows[comparison_rows["property"] == prop].reset_index(drop=True)
        for prop in PROPERTIES
    }
    delta_rows: list[dict[str, float | int]] = []
    for bootstrap_id in range(1, resamples + 1):
        single_rows = []
        ensemble_rows = []
        for prop, frame in property_frames.items():
            positions = rng.integers(0, len(frame), size=len(frame))
            sample = frame.iloc[positions]
            single = metric_values(sample["true_ae"], sample["single_pre_ae"])
            ensemble = metric_values(sample["true_ae"], sample["ensemble_pre_ae"])
            single_rows.append({"property": prop, "n": len(sample), **single})
            ensemble_rows.append({"property": prop, "n": len(sample), **ensemble})
        single_df = pd.DataFrame(single_rows)
        ensemble_df = pd.DataFrame(ensemble_rows)
        weights = single_df["n"].to_numpy(dtype=float)
        delta_rows.append(
            {
                "bootstrap_id": bootstrap_id,
                **{
                    f"delta_{name}": float(
                        np.average(ensemble_df[name] - single_df[name], weights=weights)
                    )
                    for name in ["spearman", "r2", "mae", "rmse"]
                },
            }
        )
    samples = pd.DataFrame(delta_rows)
    summary_rows = []
    for name in ["spearman", "r2", "mae", "rmse"]:
        values = samples[f"delta_{name}"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "metric": name,
                "delta_definition": "cv5_ensemble_minus_current_single",
                "point_estimate": float(weighted_metric_delta(comparison_rows, name)),
                "bootstrap_median": float(np.median(values)),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "probability_delta_gt_0": float(np.mean(values > 0)),
                "probability_delta_lt_0": float(np.mean(values < 0)),
                "n_resamples": resamples,
            }
        )
    return pd.DataFrame(summary_rows)


def weighted_metric_delta(frame: pd.DataFrame, metric: str) -> float:
    single_rows = []
    ensemble_rows = []
    for prop in PROPERTIES:
        subset = frame[frame["property"] == prop]
        single_rows.append(
            {"n": len(subset), **metric_values(subset["true_ae"], subset["single_pre_ae"])}
        )
        ensemble_rows.append(
            {"n": len(subset), **metric_values(subset["true_ae"], subset["ensemble_pre_ae"])}
        )
    single_df = pd.DataFrame(single_rows)
    ensemble_df = pd.DataFrame(ensemble_rows)
    weights = single_df["n"].to_numpy(dtype=float)
    return float(np.average(ensemble_df[metric] - single_df[metric], weights=weights))


def write_manifest(directory: Path) -> None:
    files = sorted(
        path for path in directory.rglob("*") if path.is_file() and path.name != "MANIFEST.sha256"
    )
    lines = [f"{sha256(path)}  {path.relative_to(directory)}" for path in files]
    manifest = directory / "MANIFEST.sha256"
    tmp = manifest.with_suffix(".sha256.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, manifest)


def main() -> None:
    args = parse_args()
    selection, config = load_manual_parameters(args)
    missing = [str(path) for path in required_paths() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
    existing = inspect_existing_checkpoint()
    if existing["has_fold_members"]:
        raise RuntimeError(
            "Existing checkpoint unexpectedly contains fold members; direct reuse is required."
        )

    calibration, test, solvent_summary = MLP06.prepare_base_frames()
    if args.check:
        print(
            json.dumps(
                {
                    "status": "passed",
                    "existing_checkpoint": existing,
                    "fallback_required": True,
                    "fallback_reason": "existing checkpoint has one full-Calibration model per property and no CV members",
                    "parameter_source": selection["parameter_source"],
                    "requested_params": selection["requested_params"],
                    "best_params": config,
                    "calibration_property_rows": len(calibration),
                    "test15_property_rows": len(test),
                    "n_features": len(FEATURE_COLUMNS),
                },
                indent=2,
            )
        )
        return

    output_name = args.run_name + "_smoke" if args.smoke_test else args.run_name
    result_dir = PACKAGE_ROOT / "results/final" / output_name
    model_dir = PACKAGE_ROOT / "models/reliability_model" / output_name
    if result_dir.exists() or model_dir.exists():
        raise FileExistsError("Output directories already exist. Choose a new --run-name.")
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    fold_frames, assignments, fold_audits = make_fold_frames(calibration, test)
    atomic_csv(assignments, result_dir / "fold_assignments.csv")
    atomic_csv(pd.DataFrame(fold_audits), result_dir / "fold_data_audit.csv")

    oof_frames: list[pd.DataFrame] = []
    test_member_frames: list[pd.DataFrame] = []
    process_rows: list[dict[str, Any]] = []
    fold_checkpoint_paths: list[Path] = []
    started_all = time.time()

    for cv_fold, frame in enumerate(fold_frames, start=1):
        fold_models: dict[str, Any] = {}
        for prop in PROPERTIES:
            train = frame[(frame["split"] == "calibration") & (frame["property"] == prop)]
            validation = frame[(frame["split"] == "validation") & (frame["property"] == prop)]
            test15 = frame[(frame["split"] == "test15") & (frame["property"] == prop)]
            seed = 42 + cv_fold * 100 + PROPERTY_INDEX[prop]
            started = time.time()
            y_train = np.log(train["abs_error"].to_numpy(dtype=float) + float(EPSILON[prop]))
            model = MLP06.FittedLogMLP(config).fit(
                train[FEATURE_COLUMNS].to_numpy(dtype=float),
                y_train,
                train["sample_weight"].to_numpy(dtype=float),
                seed,
            )
            runtime = time.time() - started
            fold_models[prop] = model.state_payload()

            validation_pred = MLP06.inverse_log(
                model.predict(validation[FEATURE_COLUMNS].to_numpy(dtype=float)), prop
            )
            validation_true = np.abs(
                validation["base_prediction"].to_numpy(dtype=float)
                - validation["y_true"].to_numpy(dtype=float)
            )
            validation_metrics = metric_values(validation_true, validation_pred)
            oof = validation[["property", "row_index", "fold"]].copy()
            oof.insert(0, "cv_fold", cv_fold)
            oof["predicted_ae"] = validation_pred
            oof["true_ae"] = validation_true
            oof_frames.append(oof)

            test_pred = MLP06.inverse_log(
                model.predict(test15[FEATURE_COLUMNS].to_numpy(dtype=float)), prop
            )
            test_member = test15[["property", "row_index", "fold"]].copy()
            test_member.insert(0, "cv_fold", cv_fold)
            test_member["predicted_ae"] = test_pred
            test_member_frames.append(test_member)

            with torch.no_grad():
                train_pred_log = model.predict(train[FEATURE_COLUMNS].to_numpy(dtype=float))
            final_train_weighted_l1_log = float(
                np.average(
                    np.abs(train_pred_log - y_train),
                    weights=train["sample_weight"].to_numpy(dtype=float),
                )
            )
            process_rows.append(
                {
                    "cv_fold": cv_fold,
                    "property": prop,
                    "seed": seed,
                    "epochs": int(config["epochs"]),
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "test15_rows": len(test15),
                    "final_train_weighted_l1_log": final_train_weighted_l1_log,
                    **{f"validation_{name}": value for name, value in validation_metrics.items()},
                    "runtime_seconds": runtime,
                }
            )
            print(
                f"Fold {cv_fold}/5 {prop}: val Spearman={validation_metrics['spearman']:.6f} "
                f"R2={validation_metrics['r2']:.6f}; runtime={runtime:.1f}s",
                flush=True,
            )

        checkpoint = {
            "format_version": 1,
            "model_family": "mlp_manual_cv_member",
            "purpose": "Experimental five-fold Test15 ensemble member; does not replace the deployed full-Calibration model.",
            "cv_fold": cv_fold,
            "parameter_source": selection["parameter_source"],
            "requested_params": selection["requested_params"],
            "selected_params": config,
            "model_seed_rule": selection["model_seed_rule"],
            "feature_columns": FEATURE_COLUMNS,
            "epsilon_by_property": EPSILON,
            "history_feature_policy": "fit only on this member's four Calibration training folds",
            "models": fold_models,
        }
        checkpoint_path = model_dir / f"fold_{cv_fold:02d}.pt"
        atomic_torch_save(checkpoint, checkpoint_path)
        fold_checkpoint_paths.append(checkpoint_path)

    oof_predictions = pd.concat(oof_frames, ignore_index=True)
    member_predictions = pd.concat(test_member_frames, ignore_index=True)
    atomic_csv(pd.DataFrame(process_rows), result_dir / "training_process.csv")
    atomic_csv(oof_predictions, result_dir / "oof_predictions.csv")
    atomic_csv(member_predictions, result_dir / "test15_member_predictions_label_free.csv")

    key_columns = ["property", "row_index", "fold"]
    member_counts = member_predictions.groupby(key_columns).size()
    if not (member_counts == 5).all() or len(member_counts) != len(test):
        raise ValueError("Each Test15 property-row must have exactly five fold-member predictions.")
    ensemble_predictions = member_predictions.groupby(key_columns, as_index=False)[
        "predicted_ae"
    ].agg(
        ensemble_pre_ae="mean",
        member_pre_ae_std=lambda values: float(np.std(values, ddof=0)),
        member_pre_ae_min="min",
        member_pre_ae_max="max",
    )
    ensemble_predictions["member_pre_ae_range"] = (
        ensemble_predictions["member_pre_ae_max"] - ensemble_predictions["member_pre_ae_min"]
    )
    atomic_csv(ensemble_predictions, result_dir / "test15_ensemble_predictions_label_free.csv")
    atomic_json(
        {
            "status": "predictions_frozen_before_test_labels_loaded",
            "frozen_at_unix_seconds": time.time(),
            "test15_used_for_hpo_or_model_selection": False,
            "member_prediction_rows": len(member_predictions),
            "ensemble_prediction_rows": len(ensemble_predictions),
            "member_checkpoint_sha256": {path.name: sha256(path) for path in fold_checkpoint_paths},
        },
        result_dir / "prediction_freeze.json",
    )

    # Test15 labels are first loaded after all member models and label-free predictions are frozen.
    labels = STEP03.load_test_labels(str(TEST_LABELS))
    ensemble_evaluated = build_evaluation_frame(
        test, ensemble_predictions, labels, "ensemble_pre_ae"
    )
    atomic_csv(ensemble_evaluated, result_dir / "test15_ensemble_evaluated_rows.csv")
    ensemble_metrics = metrics_by_property(
        ensemble_evaluated, "ensemble_pre_ae", "cv5_manual_ensemble"
    )

    member_evaluated = member_predictions.merge(
        ensemble_evaluated[key_columns + ["true_ae"]],
        on=key_columns,
        how="left",
        validate="many_to_one",
    )
    member_metric_frames = []
    for cv_fold in range(1, 6):
        subset = member_evaluated[member_evaluated["cv_fold"] == cv_fold]
        member_metric_frames.append(
            metrics_by_property(subset, "predicted_ae", f"cv_fold_{cv_fold}").assign(
                cv_fold=cv_fold
            )
        )
    member_metrics = pd.concat(member_metric_frames, ignore_index=True)
    atomic_csv(member_metrics, result_dir / "test15_member_metrics.csv")

    current = pd.read_csv(CURRENT_EVALUATED)
    if current.duplicated(["property", "row_index", "fold"]).any():
        raise ValueError("Current single-model evaluated rows contain duplicate keys.")
    current["true_ae"] = np.abs(
        current["base_prediction"].to_numpy(dtype=float) - current["y_true"].to_numpy(dtype=float)
    )
    current = current.rename(columns={"predicted_ae": "single_pre_ae"})
    single_metrics = metrics_by_property(
        current, "single_pre_ae", "current_full_calibration_single"
    )
    atomic_csv(single_metrics, result_dir / "current_single_metrics_recomputed.csv")
    atomic_csv(ensemble_metrics, result_dir / "test15_ensemble_metrics.csv")

    comparison_rows = current[key_columns + ["single_pre_ae", "true_ae"]].merge(
        ensemble_evaluated[key_columns + ["ensemble_pre_ae", "member_pre_ae_std", "true_ae"]],
        on=key_columns,
        how="inner",
        validate="one_to_one",
        suffixes=("_single", "_ensemble"),
    )
    if len(comparison_rows) != len(test):
        raise ValueError("Single and ensemble Test15 predictions do not have identical coverage.")
    if not np.allclose(
        comparison_rows["true_ae_single"], comparison_rows["true_ae_ensemble"], rtol=0, atol=1e-12
    ):
        raise ValueError("Single and ensemble evaluations disagree on Test15 true AE.")
    comparison_rows = comparison_rows.rename(columns={"true_ae_single": "true_ae"}).drop(
        columns=["true_ae_ensemble"]
    )
    atomic_csv(comparison_rows, result_dir / "test15_single_vs_ensemble_rows.csv")

    comparison = single_metrics.merge(
        ensemble_metrics,
        on=["property", "n"],
        how="inner",
        validate="one_to_one",
        suffixes=("_single", "_ensemble"),
    )
    for name in ["spearman", "r2", "mae", "rmse"]:
        comparison[f"delta_{name}_ensemble_minus_single"] = (
            comparison[f"{name}_ensemble"] - comparison[f"{name}_single"]
        )
    dispersion = (
        ensemble_evaluated.groupby("property")["member_pre_ae_std"]
        .agg(member_pre_ae_std_mean="mean", member_pre_ae_std_median="median")
        .reset_index()
    )
    comparison = comparison.merge(dispersion, on="property", how="left")
    atomic_csv(comparison, result_dir / "test15_comparison_by_property.csv")

    oof_metrics = metrics_by_property(oof_predictions, "predicted_ae", "manual_oof")
    atomic_csv(oof_metrics, result_dir / "oof_metrics.csv")
    bootstrap = paired_bootstrap(comparison_rows, args.bootstrap_resamples)
    if not bootstrap.empty:
        atomic_csv(bootstrap, result_dir / "paired_bootstrap_delta_summary.csv")

    weighted_comparison = comparison[comparison["property"] == "weighted_all"].iloc[0]
    primary_improved = bool(weighted_comparison["delta_spearman_ensemble_minus_single"] > 0)
    validation = {
        "status": "passed",
        "run_name": args.run_name,
        "smoke_test": bool(args.smoke_test),
        "decision": (
            "ensemble improves the primary weighted Test15 Spearman"
            if primary_improved
            else "ensemble does not improve the primary weighted Test15 Spearman"
        ),
        "primary_metric": "sample-count-weighted mean of per-property Spearman",
        "primary_metric_improved": primary_improved,
        "parameter_source": selection["parameter_source"],
        "requested_params": selection["requested_params"],
        "selected_params": config,
        "cv_method": "GroupKFold(n_splits=5, shuffle=False) by canonical solute-solvent pair",
        "history_feature_policy": "each member uses only its four Calibration training folds",
        "test15_used_for_hpo_or_model_selection": False,
        "test15_labels_loaded_after_model_and_prediction_freeze": True,
        "ensemble_rule": "arithmetic mean of five member Pre_AE predictions in raw AE space",
        "calibration_property_rows": len(calibration),
        "test15_property_rows": len(test),
        "oof_prediction_rows": len(oof_predictions),
        "test15_member_prediction_rows": len(member_predictions),
        "test15_ensemble_prediction_rows": len(ensemble_predictions),
        "n_features": len(FEATURE_COLUMNS),
        "fold_group_overlap_counts": [item["pair_group_overlap_count"] for item in fold_audits],
        "weighted_oof_spearman": float(
            oof_metrics.loc[oof_metrics["property"] == "weighted_all", "spearman"].iloc[0]
        ),
        "weighted_test15_single": {
            name: float(weighted_comparison[f"{name}_single"])
            for name in ["spearman", "r2", "mae", "rmse"]
        },
        "weighted_test15_ensemble": {
            name: float(weighted_comparison[f"{name}_ensemble"])
            for name in ["spearman", "r2", "mae", "rmse"]
        },
        "weighted_test15_delta_ensemble_minus_single": {
            name: float(weighted_comparison[f"delta_{name}_ensemble_minus_single"])
            for name in ["spearman", "r2", "mae", "rmse"]
        },
        "existing_checkpoint": existing,
        "existing_checkpoint_unchanged": sha256(CURRENT_CHECKPOINT) == existing["sha256"],
        "input_sha256": {path.name: sha256(path) for path in required_paths()},
        "fold_checkpoint_sha256": {path.name: sha256(path) for path in fold_checkpoint_paths},
        "solvent_reference_summary": solvent_summary,
        "runtime_seconds": float(time.time() - started_all),
        "interpretation_caveat": (
            "Test15 has been used previously for model-family comparisons; this is an exploratory "
            "internal-holdout comparison, not a new untouched external test."
        ),
    }
    atomic_json(validation, result_dir / "validation.json")
    write_manifest(result_dir)
    write_manifest(model_dir)
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
