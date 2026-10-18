from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NAME = "FLAIR-reproducibility-package-v3-log-mlp-hpo100"
OFFICIAL_MODEL = "mlp_log_hpo100_reliability_model_bundle.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the v3 Log-MLP HPO100 release archive.")
    parser.add_argument("--destination_parent", default=str(ROOT.parent))
    parser.add_argument("--name", default=DEFAULT_NAME)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_ignore(path: str, names: list[str]) -> set[str]:
    directory = Path(path)
    relative = directory.relative_to(ROOT)
    ignored = {
        name
        for name in names
        if name in {".DS_Store", "__pycache__", ".git", "backups", "backups 2"}
        or name.endswith((".pyc", "_smoke"))
        or (directory / name).is_symlink()
    }
    if relative == Path("data"):
        ignored.add("cache")
    if relative.parts[:2] == ("models", "main_model"):
        ignored.update({"summary.csv", "test_predictions.csv", "training_log.csv", "training_log.npz"})
    if relative == Path("models/reliability_model"):
        ignored.update({
            "prediction_deploy_cache_ae.pkl",
            "prediction_deploy_cache_solvent32.pkl",
            "mlp_expanded_earlystop_hpo100_20260903",
            "mlp_fixed_capacity_64x32_vs_256x128_20260902",
            "mlp_log_hpo100_cv5_trial65_ensemble_20260826",
        })
    if relative == Path("."):
        ignored.update({"MANIFEST.sha256", "batch_predict_bins.py"})
    if relative in {Path("models/reliability_model"), Path("results/final"), Path("results/hpo")}:
        ignored.update(
            name
            for name in names
            if name.startswith(
                (
                    "reliability_rf_",
                    "property_specific_ensemble_",
                    "ml_model_families_",
                    "reliability_model_solvent_features",
                )
            )
        )
    if relative == Path("results/hpo"):
        ignored.add("candidates")
    if relative == Path("src/reliability_model"):
        ignored.update(
            {
                "03_train_rf_leaf1_sqrt_oob.py",
                "04_ablate_redundant_features.py",
                "05_hpo_rf_random_csv.py",
                "05_hpo_rf_staircase.py",
                "07_hpo_ml_model_families.py",
                "solvent_residual_features.py",
            }
        )
    if relative.parts[:2] == ("results", "hpo"):
        ignored.update({"cv_cache", "cv_feature_cache"})
    if relative == Path("results/intermediate"):
        ignored.update(
            name
            for name in names
            if name.startswith(
                ("offline_feature_rows_", "offline_predictions_", "offline_shared_solute_features_")
            )
            or name == "reference_cache_ae_train_and_val.pkl"
        )
    return ignored


def normalize_release_metadata(package_root: Path) -> None:
    """Remove workstation-specific absolute paths from distributed metadata."""
    source_prefix = str(ROOT)
    text_suffixes = {".json", ".csv", ".md", ".txt", ".log", ".cff"}
    for path in package_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8")
        if source_prefix in text:
            path.write_text(text.replace(source_prefix, "."), encoding="utf-8")


def write_manifest(package_root: Path) -> Path:
    manifest = package_root / "MANIFEST.sha256"
    lines = []
    for path in sorted(item for item in package_root.rglob("*") if item.is_file()):
        if path == manifest:
            continue
        relative = path.relative_to(package_root).as_posix()
        lines.append(f"{sha256(path)}  ./{relative}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def verify_required_files(package_root: Path) -> None:
    required = [
        package_root / "README.md",
        package_root / "RELEASE_NOTES.md",
        package_root / "environment/portable_cpu_commands.txt",
        package_root / "data/splits/deployment/deployment.csv",
        package_root / "data/splits/deployment/deployment_test.csv",
        package_root / "MANIFEST.sha256",
        package_root / "scripts/03_train_reliability_model.py",
        package_root / "scripts/04_predict_single.py",
        package_root / "src/reliability_model/03_train_log_mlp_cv5_ensemble.py",
        package_root / "src/reliability_model/mlp_core.py",
        package_root / "data/templates/mlp_manual_parameters.json",
        package_root / "src/reliability_model/log_mlp_reliability.py",
        package_root / "src/reliability_model/reliability_features.py",
        package_root / "models/reliability_model" / OFFICIAL_MODEL,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Release package is missing required files:\n" + "\n".join(missing))


def main() -> None:
    args = parse_args()
    destination_parent = Path(args.destination_parent).expanduser().resolve()
    package_root = destination_parent / args.name
    archive = destination_parent / f"{args.name}.zip"
    checksum = destination_parent / f"{args.name}.zip.sha256"
    existing = [path for path in [package_root, archive, checksum] if path.exists()]
    if existing:
        raise FileExistsError(
            "Release targets already exist:\n" + "\n".join(str(path) for path in existing)
        )

    shutil.copytree(ROOT, package_root, ignore=release_ignore, copy_function=shutil.copy2)
    normalize_release_metadata(package_root)
    write_manifest(package_root)
    verify_required_files(package_root)
    shutil.make_archive(
        str(archive.with_suffix("")), "zip", root_dir=destination_parent, base_dir=args.name
    )
    with zipfile.ZipFile(archive) as handle:
        bad_file = handle.testzip()
        if bad_file is not None:
            raise RuntimeError(f"ZIP CRC verification failed at {bad_file}")
    checksum.write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")
    file_count = sum(1 for path in package_root.rglob("*") if path.is_file())
    print(f"package_root={package_root}")
    print(f"archive={archive}")
    print(f"archive_sha256={sha256(archive)}")
    print(f"file_count={file_count}")
    print(f"archive_size_bytes={archive.stat().st_size}")


if __name__ == "__main__":
    main()
