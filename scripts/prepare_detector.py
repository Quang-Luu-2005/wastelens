"""Download the official torchvision SSDLite COCO detector once for local webcam use."""
from pathlib import Path
import hashlib
import torch

target = Path(__file__).resolve().parents[1] / "models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth"
target.parent.mkdir(parents=True, exist_ok=True)
if not target.exists() or not hashlib.sha256(target.read_bytes()).hexdigest().startswith("a79551df"):
    temporary = target.with_suffix(".download")
    torch.hub.download_url_to_file("https://download.pytorch.org/models/" + target.name,
                                   str(temporary), hash_prefix="a79551df")
    temporary.replace(target)
print(f"Detector ready: {target}")
