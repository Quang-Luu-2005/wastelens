"""The three trained model options exposed by the application."""

from dataclasses import dataclass
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


@dataclass(frozen=True)
class ModelProfile:
    """Display name, checkpoint file and training description of one app model."""
    name: str
    filename: str
    description: str
    training: str

    @property
    def checkpoint(self) -> Path:
        """Absolute path of the checkpoint inside the repository's models/ folder."""
        return MODEL_DIR / self.filename


MODEL_PROFILES = {
    "baseline": ModelProfile(
        "1 · Baseline Lab 2 → 6 lớp", "wastelens-baseline.pt",
        "RGB 64×64 → MLP 128 → 32 → 6. Giữ hai tầng ẩn của Lab 2, mở rộng đầu ra và dùng cross-entropy.",
        "Huấn luyện từ đầu",
    ),
    "improved": ModelProfile(
        "2 · MLP cải tiến", "wastelens-ensemble.pt",
        "Trích đặc trưng HOG/HSV/Lab, MLP 256 → 64 → 32 → 6, LayerNorm và dropout. "
        "Kết hợp xác suất của 10 MLP; confidence đã hiệu chỉnh.",
        "Huấn luyện từ đầu · ensemble 10 MLP",
    ),
    "mobilenet": ModelProfile(
        "3 · MobileNetV3 Large", "wastelens-mobilenetv3.pt",
        "RGB 224×224 → MobileNetV3 Large → 6 lớp. Khởi tạo bằng trọng số ImageNet và fine-tuning trên TrashNet.",
        "ImageNet pretrained → fine-tuning",
    ),
}
DEFAULT_MODEL = "improved"
