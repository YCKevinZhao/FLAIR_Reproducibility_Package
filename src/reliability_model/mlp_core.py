from __future__ import annotations


import importlib.util


import os


import random


import sys


from contextlib import contextmanager


from pathlib import Path


from typing import Any


os.environ.setdefault("OMP_NUM_THREADS", "1")


os.environ.setdefault("MKL_NUM_THREADS", "1")


os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


import numpy as np


import pandas as pd


import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


RELIABILITY_DIR = PACKAGE_ROOT / "src/reliability_model"


sys.path.insert(0, str(RELIABILITY_DIR))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STEP03 = load_module("step03_log_mlp_hpo", RELIABILITY_DIR / "reliability_features.py")


STEP02 = importlib.import_module("02_generate_ae_pre_merged_features")


PROPERTIES = list(STEP03.PROPERTIES)


EPSILON = dict(STEP03.EPSILON_BY_PROPERTY)


FEATURE_COLUMNS = list(STEP03.SOLVENT_AUGMENTED_MODEL_FEATURE_COLUMNS)


PROPERTY_INDEX = {prop: index for index, prop in enumerate(PROPERTIES)}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


@contextmanager
def suppress_native_training_warnings():
    """Silence the x86 MKL-on-Apple-Silicon warning emitted for every matmul."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    with open(os.devnull, "w") as sink:
        try:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            yield
        finally:
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


def activation(name: str) -> torch.nn.Module:
    if name == "relu":
        return torch.nn.ReLU()
    if name == "leaky_relu":
        return torch.nn.LeakyReLU(negative_slope=0.05)
    if name == "silu":
        return torch.nn.SiLU()
    if name == "gelu":
        return torch.nn.GELU()
    raise KeyError(name)


def build_network(n_features: int, config: dict[str, Any]) -> torch.nn.Module:
    layers: list[torch.nn.Module] = [
        torch.nn.Linear(n_features, int(config["hidden1"])),
        activation(str(config["activation"])),
    ]
    if float(config["dropout"]) > 0:
        layers.append(torch.nn.Dropout(float(config["dropout"])))
    layers.extend(
        [
            torch.nn.Linear(int(config["hidden1"]), int(config["hidden2"])),
            activation(str(config["activation"])),
        ]
    )
    if float(config["dropout"]) > 0:
        layers.append(torch.nn.Dropout(float(config["dropout"])))
    layers.append(torch.nn.Linear(int(config["hidden2"]), 1))
    return torch.nn.Sequential(*layers)


class FittedLogMLP:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)
        self.x_mean: np.ndarray | None = None
        self.x_scale: np.ndarray | None = None
        self.y_center: float | None = None
        self.y_scale: float | None = None
        self.network: torch.nn.Module | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray, seed: int) -> "FittedLogMLP":
        seed_everything(seed)
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        weights = np.asarray(weights, dtype=np.float32)
        self.x_mean = x.mean(axis=0)
        self.x_scale = x.std(axis=0)
        self.x_scale[self.x_scale < 1e-8] = 1.0
        self.y_center = float(np.median(y))
        q25, q75 = np.percentile(y, [25, 75])
        self.y_scale = float(max(q75 - q25, np.std(y), 1e-6))
        xs = (x - self.x_mean) / self.x_scale
        ys = (y - self.y_center) / self.y_scale
        self.network = build_network(x.shape[1], self.config)
        if self.config["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(
                self.network.parameters(),
                lr=float(self.config["learning_rate"]),
                weight_decay=float(self.config["weight_decay"]),
            )
        else:
            optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=float(self.config["learning_rate"]),
                weight_decay=float(self.config["weight_decay"]),
            )
        xt = torch.from_numpy(xs)
        yt = torch.from_numpy(ys).reshape(-1, 1)
        wt = torch.from_numpy(weights).reshape(-1, 1)
        denominator = torch.clamp(wt.sum(), min=1e-12)
        self.network.train()
        with suppress_native_training_warnings():
            for _ in range(int(self.config["epochs"])):
                optimizer.zero_grad(set_to_none=True)
                prediction = self.network(xt)
                loss = (wt * torch.abs(prediction - yt)).sum() / denominator
                loss.backward()
                optimizer.step()
            self.network.eval()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.network is None or self.x_mean is None or self.x_scale is None:
            raise RuntimeError("Model is not fitted")
        xs = (np.asarray(x, dtype=np.float32) - self.x_mean) / self.x_scale
        with torch.no_grad():
            scaled = self.network(torch.from_numpy(xs)).numpy().reshape(-1)
        return scaled * float(self.y_scale) + float(self.y_center)

    def state_payload(self) -> dict[str, Any]:
        if self.network is None:
            raise RuntimeError("Model is not fitted")
        return {
            "config": self.config,
            "x_mean": self.x_mean,
            "x_scale": self.x_scale,
            "y_center": self.y_center,
            "y_scale": self.y_scale,
            "state_dict": {
                key: value.detach().cpu() for key, value in self.network.state_dict().items()
            },
        }


def inverse_log(values: np.ndarray, prop: str) -> np.ndarray:
    return np.maximum(np.exp(np.asarray(values, dtype=float)) - float(EPSILON[prop]), 0.0)


def prepare_base_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    calibration_path = PACKAGE_ROOT / "results/intermediate/calibration_features_ae_pre.csv"
    test_path = PACKAGE_ROOT / "results/intermediate/test_features_ae_pre.csv"
    reference_path = PACKAGE_ROOT / "data/splits/deployment/deployment.csv"
    calibration, _ = STEP03.load_calibration_features(
        str(calibration_path), feature_columns=FEATURE_COLUMNS
    )
    test = STEP03.load_test_features(str(test_path))
    calibration = STEP03.aggregate_base_predictions(calibration, include_targets=True)
    test = STEP03.aggregate_base_predictions(test, include_targets=False)
    calibration, test, solvent_summary = STEP03.add_solvent_reference_feature_columns(
        calibration, test, str(reference_path)
    )
    scaffold_cache: dict[str, str] = {}

    def scaffold(smiles: object) -> str:
        key = "" if pd.isna(smiles) else str(smiles).strip()
        if key not in scaffold_cache:
            scaffold_cache[key] = (
                STEP03.get_scaffold_smiles(STEP03.mol_from_smiles(key)) if key else ""
            )
        return scaffold_cache[key]

    calibration["_direct_scaffold"] = calibration["smiles"].map(scaffold)
    test["_direct_scaffold"] = test["smiles"].map(scaffold)
    return calibration, test, solvent_summary
