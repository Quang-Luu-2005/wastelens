"""Download the fixed deployment assets and verify hashes before installing them."""

import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads(Path(__file__).with_name("assets.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="wastelens-assets-") as temporary:
        for asset in manifest["assets"]:
            downloaded = Path(temporary) / asset["name"]
            request = urllib.request.Request(manifest["base_url"] + "/" + asset["name"],
                                             headers={"User-Agent": "WasteLens-deployment"})
            with urllib.request.urlopen(request, timeout=120) as response, downloaded.open("wb") as file:
                shutil.copyfileobj(response, file)
            with downloaded.open("rb") as file:
                actual = hashlib.file_digest(file, "sha256").hexdigest()
            if actual != asset["sha256"]:
                raise ValueError(f"Checkpoint/dataset integrity mismatch: {asset['name']}")
            if asset["name"].endswith(".zip"):
                with zipfile.ZipFile(downloaded) as archive:
                    for entry in archive.infolist():
                        target = (ROOT / entry.filename).resolve()
                        if not target.is_relative_to(ROOT / "data/trashnet"):
                            raise ValueError(f"Unexpected dataset archive entry: {entry.filename}")
                    archive.extractall(ROOT)
            else:
                target = ROOT / "models" / asset["name"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(downloaded, target)
            print(f"Verified {asset['name']}", flush=True)


if __name__ == "__main__":
    main()
