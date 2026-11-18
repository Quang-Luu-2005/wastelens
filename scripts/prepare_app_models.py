"""Install the three existing trained checkpoints locally, verifying their provenance.

Run after downloading the experiment outputs listed in docs/app-models.md.
Large model files remain outside Git. No training or network download is performed.
"""

import hashlib
import shutil
from pathlib import Path

from wastelens.profiles import MODEL_PROFILES

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "baseline": ("outputs/kaggle-t4-600ep-20261110/seed-43/best.pt",
                 "4bcfae31391cdf900454785e10cd6345ea87716f22e1c0c82e7fa6add70aa8f1"),
    "improved": ("outputs/kaggle-ensemble-600-20261113/ensemble-300.pt",
                 "061513298ee3991b4c3de62f5ce7d2ff20c1907b8ce9e2803d0418ffe6880d45"),
    "mobilenet": ("outputs/kaggle-mobilenetv3-100ep-20261110/seed-43/best.pt",
                  "dd62a1e6500111cc02547bac4a1c96f386b9514fdb6f1b82f031ef7706203c36"),
}


def digest(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    # Check every source and existing destination before writing any files.
    for key, (relative, expected) in SOURCES.items():
        source, target = ROOT / relative, MODEL_PROFILES[key].checkpoint
        if not source.is_file() or digest(source) != expected:
            raise ValueError(f"Missing or unexpected checkpoint: {source}")
        if target.exists() and digest(target) != expected:
            raise ValueError(f"Refusing to replace a different checkpoint: {target}")
    for key, (relative, _) in SOURCES.items():
        target = MODEL_PROFILES[key].checkpoint
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(ROOT / relative, target)
        print(f"{key}: {target}")


if __name__ == "__main__":
    main()
