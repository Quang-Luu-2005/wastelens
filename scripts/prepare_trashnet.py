"""Download, audit, and split the pinned TrashNet dataset."""

import argparse
import csv
import hashlib
import io
import json
import shutil
import tempfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REVISION = "6fa2b878c6c1b4304b91109070ce0edf9279bb31"
SOURCE_URL = f"https://raw.githubusercontent.com/garythung/trashnet/{REVISION}/data/dataset-resized.zip"
SOURCE_SHA256 = "0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2"
SOURCE_COUNTS = {"cardboard": 403, "glass": 501, "metal": 410, "paper": 594, "plastic": 482, "trash": 137}
SPLIT_RATIOS = {"train": 70, "val": 15, "test": 15}
SEED = 42


def verify_archive(path: Path) -> None:
    with path.open("rb") as file:
        actual = hashlib.file_digest(file, "sha256").hexdigest()
    if actual != SOURCE_SHA256:
        raise ValueError(f"Unexpected TrashNet SHA-256 for {path}: {actual}")


def download_archive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response, temporary.open("wb") as file:
            shutil.copyfileobj(response, file)
        verify_archive(temporary)
        temporary.replace(path)


def inventory(archive: ZipFile) -> tuple[list[str], list[list[str]], Counter]:
    paths = []
    by_pixels = defaultdict(list)
    dimensions = Counter()
    for member in sorted(archive.namelist()):
        parts = PurePosixPath(member).parts
        if len(parts) != 3 or parts[0] != "dataset-resized" or not parts[2].endswith(".jpg"):
            continue
        relative = f"{parts[1]}/{parts[2]}"
        paths.append(relative)
        with Image.open(io.BytesIO(archive.read(member))) as image:
            rgb = image.convert("RGB")
            dimensions[f"{rgb.width}x{rgb.height}"] += 1
            pixel_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
            by_pixels[(rgb.size, pixel_hash)].append(relative)

    counts = Counter(path.split("/", 1)[0] for path in paths)
    if counts != SOURCE_COUNTS or len(paths) != len(set(paths)):
        raise ValueError(f"Unexpected TrashNet image inventory: {dict(counts)}")

    duplicates = [group for group in by_pixels.values() if len(group) > 1]
    excluded = {path for group in duplicates for path in group}
    return [path for path in paths if path not in excluded], duplicates, dimensions


def stratified_rows(paths: list[str]) -> list[dict]:
    by_class = defaultdict(list)
    for path in paths:
        by_class[path.split("/", 1)[0]].append(path)

    rows = []
    for class_id, label in enumerate(SOURCE_COUNTS):
        # Hash-based ordering keeps the seed stable across Python versions.
        ranked = sorted(by_class[label], key=lambda path: hashlib.sha256(f"{SEED}:{path}".encode()).digest())
        quotas = {split: len(ranked) * ratio for split, ratio in SPLIT_RATIOS.items()}
        counts = {split: quota // 100 for split, quota in quotas.items()}
        # Distribute leftover images by largest remainder in train/val/test order.
        remainder_order = sorted(quotas, key=lambda split: quotas[split] % 100, reverse=True)
        for split in remainder_order[:len(ranked) - sum(counts.values())]:
            counts[split] += 1

        start = 0
        for split, count in counts.items():
            for path in sorted(ranked[start:start + count]):
                rows.append({"path": path, "label": label, "class_id": class_id, "split": split})
            start += count
    return rows


def prepare(archive_path: Path) -> None:
    verify_archive(archive_path)
    image_root = ROOT / "data" / "trashnet"
    with ZipFile(archive_path) as archive:
        paths, duplicates, dimensions = inventory(archive)
        rows = stratified_rows(paths)
        for relative in paths:
            target = image_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(f"dataset-resized/{relative}"))

    splits = ROOT / "data" / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    with (splits / "trashnet.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "label", "class_id", "split"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    split_counts = Counter((row["label"], row["split"]) for row in rows)
    summary = {
        "source_url": SOURCE_URL,
        "archive_sha256": SOURCE_SHA256,
        "seed": SEED,
        "target_ratios_percent": SPLIT_RATIOS,
        "source_images": sum(SOURCE_COUNTS.values()),
        "image_dimensions": dict(dimensions),
        "excluded_duplicate_groups": duplicates,
        "retained_images": len(rows),
        "class_counts": {
            label: {"source": total, **{split: split_counts[label, split] for split in SPLIT_RATIOS}}
            for label, total in SOURCE_COUNTS.items()
        },
        "split_counts": dict(Counter(row["split"] for row in rows)),
    }
    (splits / "trashnet-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"TrashNet: {summary['source_images']} source images, {len(rows)} retained")
    print(f"Excluded identical images: {sum(map(len, duplicates))} across {len(duplicates)} groups")
    print(f"Split counts: {summary['split_counts']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="use an existing copy of the pinned source ZIP")
    args = parser.parse_args()
    archive = args.archive or ROOT / "data" / "raw" / "trashnet.zip"
    if args.archive is None and not archive.exists():
        download_archive(archive)
    prepare(archive)


if __name__ == "__main__":
    main()
