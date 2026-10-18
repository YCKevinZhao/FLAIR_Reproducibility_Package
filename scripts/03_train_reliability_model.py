from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AE_DIR = ROOT / "src" / "reliability_model"
TRAIN_AE_SCRIPT = AE_DIR / "03_train_log_mlp_cv5_ensemble.py"


def main() -> None:
    subprocess.run([sys.executable, str(TRAIN_AE_SCRIPT), *sys.argv[1:]], check=True)


if __name__ == "__main__":
    main()
